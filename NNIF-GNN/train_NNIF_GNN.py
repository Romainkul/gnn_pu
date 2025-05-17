# -*- coding: utf-8 -*-
"""
Graph PU Learning Experiment Framework

This module implements the main experiment loop and key utilities for GNN-based
positive-unlabeled (PU) learning experiments.
"""

import os
import sys
import csv
import datetime
import random
import numpy as np
import warnings
import logging
import copy
from typing import Dict, Tuple, List, Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from torch_sparse import SparseTensor

# Project imports
from NN_loader import ShineLoader
from loss import LabelPropagationLoss, ContrastiveLoss
from NNIF import PNN, ReliableValues, WeightedIsoForest
from encoder import GraphEncoder
from data_generating import load_dataset, make_pu_dataset
from blp import *
from nnpu import train_nnpu
from NNIF import train_two

logger = logging.getLogger(__name__)

##############################################################################
# Utility: Print GPU Memory Usage
##############################################################################
def print_cuda_meminfo(step: str = "") -> None:
    """
    Print current GPU memory usage (allocated and reserved) in MB.
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[{step}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

##############################################################################
# Early Stopping for GNN
##############################################################################
class EarlyStopping_GNN:
    """
    Early stopping mechanism for GNN training.
    """
    def __init__(self, patience: int = 50, delta: float = 0.0, loss_diff_threshold: float = 1e-3) -> None:
        self.patience = patience
        self.delta = delta
        self.loss_diff_threshold = loss_diff_threshold
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.previous_loss = None

    def __call__(self, loss: float) -> bool:
        if self.previous_loss is None:
            self.previous_loss = loss

        loss_diff = abs(self.previous_loss - loss)

        if (loss_diff < self.loss_diff_threshold) or (loss > self.best_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        self.previous_loss = loss

        if loss < (self.best_loss - self.delta):
            self.best_loss = loss
            self.counter = 0

        return self.early_stop

##############################################################################
# Set Random Seed for Reproducibility
##############################################################################
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_numbers(n: int = 5, seed: int = 42, a: int = 0, b: int = 1000) -> List[float]:
    """
    Generates a list of n random integers between a and b (inclusive).
    """
    set_seed(seed)
    return [random.randint(a, b) for _ in range(n)]

##############################################################################
# GNN Training Loop (with NNIF/PU)
##############################################################################
def train_graph(
    model,
    data: Data,
    device: torch.device,
    K: int = 5,
    treatment: str = "removal",
    rate_pairs: int = 5,
    batch_size: int = 1028,
    ratio: float = 0.1,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-6,
    cluster: int = 1500,
    anomaly_detector: str = "nearest_neighbors",
    layers: int = 3,
    sampling: str = "cluster",
    abl_lpl: bool = False,
    abl_contrast: bool = False,
    abl_adasyn: bool = False,
    abl_inverse: bool = False,
    mechanism: str = "SCAR",
    dataset: str = "Cora",
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Trains a GNN model using Label Propagation + Contrastive Loss and NNIF-based
    reliable positive/negative discovery.
    """
    # Estimate prior
    estim_prior = data.train_mask.sum().item()/data.x.size(0) + ratio*(data.x.size(0)-data.train_mask.sum().item())/data.x.size(0)
    lp_criterion = LabelPropagationLoss(K=K, ratio=estim_prior, ablation=abl_lpl).to(device)
    contrast_criterion = ContrastiveLoss(abl_inverse=abl_inverse).to(device)
    early_stopper = EarlyStopping_GNN(patience=20)

    data.n_id = torch.arange(data.num_nodes)
    if sampling == "cluster":
        if batch_size == 256:
            batch_size = 5
        elif batch_size == 512:
            batch_size = 10
        elif batch_size == 768:
            batch_size = 15
        elif batch_size == 1024:
            batch_size = 20
        elif batch_size == 2048:
            batch_size = 40
        cluster_data = ClusterData(data, num_parts=cluster)
        train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
    elif sampling == "sage":
        train_loader = NeighborLoader(copy.copy(data), num_neighbors=[25, 10], batch_size=batch_size, shuffle=True)
    elif sampling == "shine":
        train_loader = ShineLoader(copy.copy(data), num_neighbors=[2, 32], shuffle=True, batch_size=batch_size, device=device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")

    model = model.to(device)
    data = data.to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(lp_criterion.parameters()) + list(contrast_criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scaler = GradScaler()
    losses_per_epoch = []
    reliable_pos_set = set()
    reliable_neg_set = set()

    # --- Training loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for sub_data in train_loader:
            sub_data = sub_data.to(device)
            global_nids = sub_data.n_id
            num_sub_nodes = global_nids.size(0)
            sub_adj = SparseTensor.from_edge_index(
                sub_data.edge_index,
                sparse_sizes=(num_sub_nodes, num_sub_nodes)
            ).coalesce().to(device)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available(), device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                sub_emb = model(sub_data.x, sub_adj)

                # Epoch 0: Find reliable samples via NNIF
                if epoch == 0:
                    norm_sub_emb = F.normalize(sub_emb, dim=1)
                    features_np = norm_sub_emb.detach().cpu().numpy()
                    y_labels = sub_data.train_mask.detach().cpu().numpy().astype(int)
                    nnif_detector = ReliableValues(
                        method=treatment,
                        treatment_ratio=ratio,
                        anomaly_detector=WeightedIsoForest(n_estimators=100, type_weight=anomaly_detector),
                        random_state=42,
                        high_score_anomaly=True,
                    )
                    neg_mask, pos_mask = nnif_detector.get_reliable(features_np, y_labels)
                    global_ids_np = global_nids.cpu().numpy()
                    for i in range(num_sub_nodes):
                        if pos_mask[i]:
                            reliable_pos_set.add(int(global_ids_np[i]))
                        if neg_mask[i]:
                            reliable_neg_set.add(int(global_ids_np[i]))
                    sub_pos_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set]
                    sub_neg_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set]
                else:
                    global_ids_np = global_nids.cpu().numpy()
                    sub_pos_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set]
                    sub_neg_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set]

                sub_pos = torch.tensor(sub_pos_idx, dtype=torch.long, device=device)
                sub_neg = torch.tensor(sub_neg_idx, dtype=torch.long, device=device)

                # Loss computation
                lp_loss, E = lp_criterion(sub_emb, sub_adj, sub_pos, sub_neg)
                if not abl_contrast:
                    contrast_loss = contrast_criterion(
                        sub_emb, E, num_pairs=sub_emb.size(0) * rate_pairs
                    )
                    loss = lp_loss + contrast_loss
                else:
                    loss = lp_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())

        losses_per_epoch.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} / {num_epochs}, Loss: {epoch_loss:.4f}")

    # --- Compute embeddings for the entire graph ---
    A_hat = SparseTensor.from_edge_index(data.edge_index).coalesce().to(device)
    model.eval()
    loader = NeighborLoader(
        copy.copy(data),
        input_nodes=data.test_mask,
        num_neighbors=[-1] * K,
        batch_size=2056,
        shuffle=False
    )
    emb_dim = model(data.x, data.edge_index).shape[1]
    embeddings = torch.zeros(data.num_nodes, emb_dim)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_emb = model(batch.x, batch.edge_index)
            embeddings[batch.n_id] = batch_emb.cpu()

    # --- PNN on final embeddings ---
    pnn_model = PNN(
        method=treatment,
        treatment_ratio=ratio,
        anomaly_detector=WeightedIsoForest(n_estimators=100, type_weight=anomaly_detector),
        random_state=42,
        high_score_anomaly=True,
        resampler=None if abl_adasyn else "adasyn",
    )
    norm_emb = F.normalize(embeddings, dim=1)

    if not torch.isnan(norm_emb).any():
        features_np = norm_emb.detach().cpu().numpy()
        train_y_labels = data.train_mask.detach().cpu().numpy().astype(int)
        pnn_model.fit(features_np, train_y_labels)
        predicted = pnn_model.predict(features_np)
        predicted_probs_np = pnn_model.predict_proba(features_np)[:, 1]
        predicted_t = torch.from_numpy(predicted).to(device)
        predicted_probs_t = torch.from_numpy(predicted_probs_np).to(device)
        combined_mask = (data.train_mask | data.test_mask | data.val_mask)
        reliable_neg_mask = (predicted_t[combined_mask] == 0)
        reliable_pos_mask = (predicted_t[combined_mask] == 1)
        combined_mask = reliable_pos_mask | reliable_neg_mask
        train_labels = torch.zeros_like(combined_mask, dtype=torch.float, device=device)
        train_labels[reliable_pos_mask] = 1.0
    else:
        print("[Warning] Found NaN in embeddings, reverting to data.train_mask.")
        train_labels = data.train_mask.float()
        predicted_probs_t = torch.zeros(data.num_nodes, device=device)

    return train_labels, predicted_probs_t, losses_per_epoch

##############################################################################
# BLP Training Wrapper
##############################################################################
def train_blp(
    dataset_name: str,
    data: Data,
    device: torch.device,
    hidden_size: int = 32,
    output_size: int = 16
) -> Tuple[float, float, float, float]:
    """
    Trains a BLP model and returns F1/AP for validation/test.
    """
    warnings.filterwarnings("ignore")
    data = data.to(device)
    dataset = [data]

    # prepare augment
    drop_edge_p_1, drop_feat_p_1, drop_edge_p_2, drop_feat_p_2 = agmt_dict[dataset_name]
    augment_1 = augment_graph(drop_edge_p_1, drop_feat_p_1)
    augment_2 = augment_graph(drop_edge_p_2, drop_feat_p_2)

    # build BLP networks
    input_size = data.x.size(1)
    encoder = GCNEncoder(input_size, hidden_size, output_size)
    predictor = MLP_Predictor(output_size, hidden_size, output_size)
    model = BLP(encoder, predictor).to(device)

    # optimizer
    optimizer = optim.AdamW(model.trainable_parameters(), lr=0.005, weight_decay=1e-6)
    positive_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_nodes = data.val_mask.nonzero(as_tuple=False).view(-1)

    def learn_repersentations():
        model.train()
        optimizer.zero_grad()
        g1, g2 = augment_1(data), augment_2(data)
        p1, aux_h2 = model(g1, g2)
        p2, aux_h1 = model(g2, g1)
        positive_loss = predict_positive_nodes(p1, p2, positive_nodes)
        unlabeld_loss = predict_unlabeled_nodes(p1, aux_h2.detach(), p2, aux_h1.detach())
        loss = unlabeld_loss if unlabeld_loss > positive_loss else positive_loss
        loss.backward()
        optimizer.step()
        model.update_aux_network(0.005)

    def select_reliable_negatives_train_classifier():
        model.eval()
        g1, g2 = augment_1(data), augment_2(data)
        p1, _ = model(g1, g2)
        p2, _ = model(g2, g1)

        if data.train_mask.sum().item() < 5 * positive_nodes.size(0):
            negative_nodes = find_reliable_negative_nodes(p1, p2, positive_nodes, val_nodes)
            perm_neg_idx = negative_nodes[torch.randperm(negative_nodes.size(0))]
            neg_train_idx = perm_neg_idx[val_nodes.size(0):]
            neg_val_idx = perm_neg_idx[:val_nodes.size(0)]
            data.train_mask[neg_train_idx] = True
            data.val_mask[neg_val_idx] = True
            data.y_psd_neg[negative_nodes] = 0

        tmp_encoder = copy.deepcopy(model.main_encoder).eval()
        representations = tmp_encoder(data).detach()
        labels = data.y.detach()
        f1_scores, ap_scores = train_binary_classifier_test(
            representations.cpu().numpy(), data.y_psd_neg.cpu().numpy(),
            labels.cpu().numpy(),
            data.train_mask.cpu().numpy(), data.val_mask.cpu().numpy(),
            data.test_mask.cpu().numpy())

        return f1_scores, ap_scores

    data.y_psd_neg = data.y.clone().to(device)
    for epoch in range(1, 50):
        learn_repersentations()
        if epoch==50:
            f1_score, ap_score = select_reliable_negatives_train_classifier()
            print("epoch: {}, val_f1_score: {:.4f}, test_f1_score: {:.4f}".format(epoch, f1_score[0], f1_score[1]))
            print("epoch: {}, val_ap_score: {:.4f}, test_ap_score: {:.4f}".format(epoch, ap_score[0], ap_score[1]))
            return f1_score[0], f1_score[1], ap_score[0], ap_score[1]

##############################################################################
# Main Experiment Loop
##############################################################################
def run_nnif_gnn_experiment(params: Dict[str, Any], seed: int = 42) -> Tuple[float, float]:
    """
    Main experiment runner for all GNN/PU/BLP/NNIF experiments.
    """
    # --- Unpack params ---
    methodology = params["methodology"]
    dataset_name = params["dataset_name"]
    train_pct = params["train_pct"]
    mechanism = params["mechanism"]
    K = params["K"]
    layers = params["layers"]
    hidden_channels = params["hidden_channels"]
    out_channels = params["out_channels"]
    norm = params["norm"]
    dropout = params["dropout"]
    ratio = params["ratio"]
    aggregation = params["aggregation"]
    treatment = params["treatment"]
    anomaly_detector = params["anomaly_detector"]
    model_type = params["model_type"]
    rate_pairs = params["rate_pairs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    clusters = params["clusters"]
    min = params["min"]
    n_seeds = params["seeds"]
    num_epochs = params["num_epochs"]
    sampling = params["sampling"]
    val = params["val"]
    # Handle ablation flags
    abl_lpl, abl_contrast, abl_adasyn, abl_inverse = False, False, False, False
    if "abl_lpl" in params.keys(): abl_lpl = params["abl_lpl"]
    if "abl_contrast" in params.keys(): abl_contrast = params["abl_contrast"]
    if "abl_adasyn" in params.keys(): abl_adasyn = params["abl_adasyn"]
    if "abl_inverse" in params.keys(): abl_inverse = params["abl_inverse"]

    f1_scores = []
    ap_scores = []

    # Prepare output CSV
    output_folder = f"{dataset_name}_experimentations"
    os.makedirs(output_folder, exist_ok=True)
    base_output_csv = params["output_csv"]
    timestamp = datetime.datetime.now().strftime("%d%m%H%M%S")
    if "." in base_output_csv:
        base, ext = base_output_csv.rsplit(".", 1)
        if methodology == "ours":
            output_csv = os.path.join(output_folder, f"{base}_{timestamp}.{ext}")
        else:
            output_csv = os.path.join(output_folder, f"{base}_{methodology}_{timestamp}.{ext}")
    else:
        if methodology == "ours":
            output_csv = os.path.join(output_folder, f"{base}_{timestamp}.csv")
        else:
            output_csv = os.path.join(output_folder, f"{base}_{methodology}_{timestamp}.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    seeds_list = generate_random_numbers(n=n_seeds)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "K", "layers", "hidden_channels", "out_channels", "norm", "lr", "treatment",
            "dropout", "ratio", "seed", "aggregation", "model_type", "batch_size",
            "rate_pairs", "clusters", "sampling", "num_epochs", "anomaly_detector",
            "accuracy", "f1", "recall", "precision", "losses", "test_accuracy",
            "test_f1", "test_recall", "test_precision"
        ])
        for exp_seed in seeds_list:
            # 1) Load dataset
            data = load_dataset(dataset_name)

            # 2) Create PU dataset
            data = make_pu_dataset(
                data,
                mechanism=mechanism,
                sample_seed=exp_seed,
                train_pct=train_pct,
                val=val
            )
            prior = data.prior
            if "mult" in params.keys():
                mult = params["mult"]
                ratio = (data.prior*mult - train_pct*data.y.sum().item()/data.x.size(0)) / (1 - train_pct*data.y.sum().item()/data.x.size(0))
                prior = mult*data.prior

            in_channels = data.num_node_features

            if torch.isnan(data.x).any():
                print("NaN values in node features! Skipping seed...")
                continue

            print(f"Running experiment with seed={exp_seed}:")
            print(f" - K={K}, layers={layers}, hidden={hidden_channels}, out={out_channels}")
            print(f" - norm={norm}, dropout={dropout}, batch_size={batch_size}, methodology={methodology}")
            print(f" - ratio={ratio}, aggregation={aggregation}, treatment={treatment}, anomaly_detector={anomaly_detector}, sampling={sampling}")
            print(f" - model_type={model_type}, rate_pairs={rate_pairs}, clusters={clusters}, lr={lr}")

            model = GraphEncoder(
                model_type=model_type,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=layers,
                dropout=dropout,
                norm=norm,
                aggregation=aggregation
            )
            in_numpy = False
            try:
                if methodology == "ours":
                    train_labels, train_proba, train_losses = train_graph(
                        model=model,
                        data=data,
                        device=device,
                        K=K,
                        ratio=ratio,
                        treatment=treatment,
                        anomaly_detector=anomaly_detector,
                        rate_pairs=rate_pairs,
                        batch_size=batch_size,
                        lr=lr,
                        cluster=clusters,
                        layers=layers,
                        num_epochs=num_epochs,
                        sampling=sampling,
                        abl_lpl=abl_lpl,
                        abl_contrast=abl_contrast,
                        abl_adasyn=abl_adasyn,
                        abl_inverse=abl_inverse,
                        dataset=dataset_name,
                        mechanism=mechanism
                    )
                elif methodology == "XGBoost":
                    model = XGBClassifier()
                    model.fit(data.x.cpu().numpy(), data.train_mask.cpu().numpy())
                    preds_np, proba_np = model.predict(data.x[data.val_mask].cpu().numpy()), model.predict_proba(data.x[data.val_mask].cpu().numpy())[:, 1]
                    preds_np_test, proba_np_test = model.predict(data.x[data.test_mask].cpu().numpy()), model.predict_proba(data.x[data.test_mask].cpu().numpy())[:, 1]
                    in_numpy = True
                    train_losses = []
                elif methodology == "NNIF":
                    model = PNN(
                        method=treatment,
                        treatment_ratio=ratio,
                        anomaly_detector=WeightedIsoForest(n_estimators=100, type_weight=anomaly_detector),
                        random_state=42,
                        high_score_anomaly=True
                    )
                    model.fit(data.x.cpu().numpy(), data.train_mask.cpu().numpy())
                    preds_np, proba_np = model.predict(data.x[data.val_mask].cpu().numpy()), model.predict_proba(data.x[data.val_mask].cpu().numpy())[:, 1]
                    preds_np_test, proba_np_test = model.predict(data.x[data.test_mask].cpu().numpy()), model.predict_proba(data.x[data.test_mask].cpu().numpy())[:, 1]
                    in_numpy = True
                    train_losses = []
                elif methodology in ["nnpu", "imbnnpu"]:
                    nnpu = True
                    imbnnpu = True if methodology == "imbnnpu" else False
                    train_labels, train_proba, train_losses = train_nnpu(
                        model, data, device, model_type, layers, batch_size, lr,
                        prior=prior, nnpu=nnpu, imbpu=imbnnpu, max_epochs=num_epochs)
                elif methodology in ["two_nnif", "spy", "naive"]:
                    methodo = "NNIF" if methodology == "two_nnif" else "SPY" if methodology == "spy" else "naive"
                    train_labels, train_proba, train_losses = train_two(
                        model, data, device, methodology=methodo, layers=layers, ratio=ratio,
                        model_type=model_type, num_epochs=num_epochs, batch_size=batch_size,
                        anomaly_detector=anomaly_detector, treatment=treatment)
                if methodology == "blp":
                    f1, f1_test, ap, ap_test = train_blp(
                        dataset_name, data, device, hidden_size=hidden_channels, output_size=out_channels)
                else:
                    mask = data.train_mask | data.val_mask | data.test_mask
                    val_mask = data.val_mask[mask]
                    test_mask = data.test_mask[mask]
                    labels_np = data.y[mask][val_mask].cpu().numpy()
                    if not in_numpy:
                        preds_np = train_labels[val_mask.cpu()].cpu().numpy()
                        proba_np = train_proba[val_mask.cpu()].cpu().numpy()
                    if val:
                        accuracy = accuracy_score(labels_np, preds_np)
                        f1 = f1_score(labels_np, preds_np)
                        recall = recall_score(labels_np, preds_np)
                        precision = precision_score(labels_np, preds_np)
                        ap = average_precision_score(labels_np, proba_np)
                    else:
                        accuracy, f1, recall, precision, ap = 0.0, 0.0, 0.0, 0.0, 0.0
                    labels_np_test = data.y[mask][test_mask].cpu().numpy()
                    if not in_numpy:
                        preds_np_test = train_labels[test_mask.cpu()].cpu().numpy()
                        proba_np_test = train_proba[test_mask.cpu()].cpu().numpy()
                    accuracy_test = accuracy_score(labels_np_test, preds_np_test)
                    f1_test = f1_score(labels_np_test, preds_np_test)
                    recall_test = recall_score(labels_np_test, preds_np_test)
                    precision_test = precision_score(labels_np_test, preds_np_test)
                    ap_test = average_precision_score(labels_np_test, proba_np_test)
                    print(f" - Test Metrics: Accuracy={accuracy_test:.4f}, F1={f1_test:.4f}, Recall={recall_test:.4f}, Precision={precision_test:.4f}")
                    print(f" - Validation Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

                # Collect results
                if val:
                    f1_scores.append(f1)
                    ap_scores.append(ap)
                else:
                    f1_scores.append(f1_test)
                    ap_scores.append(ap_test)

                writer.writerow([
                    K, layers, hidden_channels, out_channels, norm, lr, treatment, dropout,
                    ratio, exp_seed, aggregation, model_type, batch_size, rate_pairs, clusters,
                    sampling, num_epochs, anomaly_detector,
                    accuracy, f1, recall, precision, train_losses, accuracy_test,
                    f1_test, recall_test, precision_test
                ])
                if f1_scores[-1] < min:
                    print(f"F1 = {f1_scores[-1]:.2f} < {min}, skipping ...")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break

    # Summarize results
    if len(f1_scores) > 0:
        avg_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        avg_ap = float(np.mean(ap_scores))
        std_ap = float(np.std(ap_scores))
    else:
        avg_f1, std_f1, avg_ap, std_ap = 0.0, 0.0, 0.0, 0.0

    print(f"Done. Results written to {output_csv}.")
    print(f"Average F1 over valid seeds: {avg_f1:.4f} ± {std_f1:.4f}")
    if dataset_name == "elliptic-bitcoin":
        return avg_f1, std_f1, avg_ap, std_ap
    else:
        return avg_f1, std_f1

