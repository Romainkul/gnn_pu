import os
import sys
import csv
import datetime
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_sparse import SparseTensor
from typing import Dict, Tuple, List, Any
import logging
import copy

from loss import LabelPropagationLoss, ContrastiveLoss
from NNIF import PNN, ReliableValues, WeightedIsoForest
from encoder import GraphSAGEEncoder, GraphEncoder
from data_generating import load_dataset, make_pu_dataset

logger = logging.getLogger(__name__)

##############################################################################
# Utility to Print GPU Memory Usage
##############################################################################
def print_cuda_meminfo(step: str = "") -> None:
    """
    Print current GPU memory usage (allocated and reserved) in MB.

    Parameters
    ----------
    step : str, optional
        A label or step name to include in the printed output for clarity.
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[{step}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


##############################################################################
# Early Stopping
##############################################################################
class EarlyStopping_GNN:
    """
    Implements an early stopping mechanism for GNN training.

    The criterion checks:
      1) If the absolute difference between the current and previous loss 
         is below a specified threshold (loss_diff_threshold), or
      2) If the current loss is worse (higher) than the best loss so far,

    then it increments a patience counter. If the counter exceeds the 'patience'
    value, training is flagged to stop.

    Additionally, if a new best loss is found that improves by more than 'delta',
    the counter resets to 0.

    Parameters
    ----------
    patience : int, default=50
        Number of epochs to wait after the last improvement.
    delta : float, default=0.0
        Minimum absolute improvement in loss to reset the patience counter.
    loss_diff_threshold : float, default=5e-4
        Threshold for considering the current loss “close enough” 
        to the previous loss.

    Attributes
    ----------
    best_loss : float
        Tracks the best (lowest) loss encountered so far.
    counter : int
        Counts how many epochs have passed without sufficient improvement.
    early_stop : bool
        Flag that becomes True once patience is exceeded.
    previous_loss : float or None
        Stores the last epoch's loss to compare with the current loss.
    """

    def __init__(
        self,
        patience: int = 50,
        delta: float = 0.0,
        loss_diff_threshold: float = 5e-4
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.loss_diff_threshold = loss_diff_threshold
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.previous_loss = None

    def __call__(self, loss: float) -> bool:
        """
        Update state given the current loss, and decide whether to early-stop.

        Parameters
        ----------
        loss : float
            The loss value from the current epoch.

        Returns
        -------
        bool
            True if the criterion suggests stopping; False otherwise.
        """
        if self.previous_loss is None:
            self.previous_loss = loss

        loss_diff = abs(self.previous_loss - loss)

        # Check if the current loss is essentially unchanged or worse
        if (loss_diff < self.loss_diff_threshold) or (loss > self.best_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        self.previous_loss = loss

        # If there's a sufficiently large improvement over the best_loss, reset
        if loss < (self.best_loss - self.delta):
            self.best_loss = loss
            self.counter = 0

        return self.early_stop


##############################################################################
# Training Loop
##############################################################################
def train_graph(
    model,
    data: Data,
    device: torch.device,
    alpha: float = 0.5,
    K: int = 5,
    treatment: str = "removal",
    rate_pairs: int = 5,
    batch_size: int = 1028,
    ratio: float = 0.1,
    margin: float = 0.5,
    pos_weight: float = 1.0,
    lpl_weight: float = 0.5,
    num_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-6,
    reliable_mini_batch: bool = False)->List[float]:

    lp_criterion = LabelPropagationLoss(alpha=alpha, K=K, pos_weight=pos_weight).to(device)
    contrast_criterion = ContrastiveLoss(margin=margin).to(device)
    early_stopping = EarlyStopping_GNN(patience=20)
    model = model.to(device)
    data.n_id = torch.arange(data.num_nodes)
    data = data.to(device)
    from torch_geometric.loader import ClusterData, ClusterLoader
    cluster_data = ClusterData(data.cpu(), num_parts=1500)
    train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
    """    train_loader = NeighborLoader(
            data,
            num_neighbors=[-1]*K,
            batch_size=1028,
            shuffle=True)"""



    optimizer = optim.AdamW(list(model.parameters())
        + list(lp_criterion.parameters())
        + list(contrast_criterion.parameters()),
        lr=lr, weight_decay=weight_decay)
    
    scaler = GradScaler()

    losses_per_epoch=[]

    reliable_pos_set = set()
    reliable_neg_set = set()

    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch = 0.0
        if reliable_mini_batch:
            for subdata in train_loader:
                subdata = subdata.to(device)
                global_nids = subdata.n_id
                num_sub_nodes = global_nids.shape[0]
                sub_A = SparseTensor.from_edge_index(
                    subdata.edge_index,
                    sparse_sizes=(num_sub_nodes, num_sub_nodes)).coalesce().to(device)

                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    sub_emb = model(subdata.x, sub_A)

                    # Compute reliable values per mini-batch in epoch 0
                    if epoch == 0:
                        # Normalize embeddings and extract features for the mini-batch
                        norm_sub_emb = F.normalize(sub_emb, dim=1)
                        features_np = norm_sub_emb.detach().cpu().numpy()
                        # Assuming subdata contains a train_mask field for this mini-batch
                        y_labels = subdata.train_mask.detach().cpu().numpy().astype(int)

                        NNIF = ReliableValues(
                            method=treatment,
                            treatment_ratio=ratio,
                            anomaly_detector=WeightedIsoForest(n_estimators=200),
                            random_state=42,
                            high_score_anomaly=True
                        )
                        # Compute reliable values on the mini-batch embeddings
                        mini_reliable_neg_mask, mini_reliable_pos_mask = NNIF.get_reliable(features_np, y_labels)

                        # Convert boolean masks to mini-batch relative indices
                        sub_pos_idx = [i for i in range(num_sub_nodes) if mini_reliable_pos_mask[i]]
                        sub_neg_idx = [i for i in range(num_sub_nodes) if mini_reliable_neg_mask[i]]

                        # Save corresponding global node IDs to the global reliable sets
                        global_ids_np = global_nids.detach().cpu().numpy()
                        for i in range(num_sub_nodes):
                            if mini_reliable_pos_mask[i]:
                                reliable_pos_set.add(int(global_ids_np[i]))
                            if mini_reliable_neg_mask[i]:
                                reliable_neg_set.add(int(global_ids_np[i]))

                    else:
                        # For epochs > 0, you may reuse the reliable sets computed from the first epoch.
                        # Here, we assume that the reliable sets for the mini-batch are determined by comparing
                        # the global ids to a stored reliable set computed in epoch 0. If you wish to recompute
                        # them per mini-batch at every epoch, you could replicate the code above.
                        global_ids_np = global_nids.detach().cpu().numpy()
                        sub_pos_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set]
                        sub_neg_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set]

                    sub_pos = torch.tensor(sub_pos_idx, dtype=torch.long, device=device)
                    sub_neg = torch.tensor(sub_neg_idx, dtype=torch.long, device=device)

                    lp_loss, E = lp_criterion(sub_emb, sub_A, sub_pos, sub_neg)
                    contrast_loss = contrast_criterion(sub_emb, E, num_pairs=sub_emb.size(0) * 7)
                    loss = lpl_weight * lp_loss + (1.0 - lpl_weight) * contrast_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_epoch += loss.item()
        else:
            if epoch == 0:
                model.eval()
                data = data.to(device)
                num_nodes = data.x.size(0)
                full_A = SparseTensor.from_edge_index(
                    data.edge_index,
                    sparse_sizes=(num_nodes, num_nodes)
                ).coalesce().to(device)

                with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    full_emb = model(data.x, full_A)
                    norm_full_emb = F.normalize(full_emb, dim=1)
                    features_np = norm_full_emb.detach().cpu().numpy()
                    y_labels = data.train_mask.detach().cpu().numpy().astype(int)

                    NNIF = ReliableValues(
                        method=treatment,
                        treatment_ratio=ratio,
                        anomaly_detector=WeightedIsoForest(n_estimators=200),
                        random_state=42,
                        high_score_anomaly=True
                    )
                    global_reliable_neg_mask, global_reliable_pos_mask = NNIF.get_reliable(features_np, y_labels)

                    reliable_pos_set = set(np.where(global_reliable_pos_mask)[0].tolist())
                    reliable_neg_set = set(np.where(global_reliable_neg_mask)[0].tolist())

                model.train()

            for subdata in train_loader:
                subdata = subdata.to(device)
                global_nids = subdata.n_id
                num_sub_nodes = global_nids.shape[0]
                sub_A = SparseTensor.from_edge_index(
                    subdata.edge_index,
                    sparse_sizes=(num_sub_nodes, num_sub_nodes)
                ).coalesce().to(device)

                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    sub_emb = model(subdata.x, sub_A)
                    global_ids_np = global_nids.detach().cpu().numpy()
                    sub_pos_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set]
                    sub_neg_idx = [i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set]

                    sub_pos = torch.tensor(sub_pos_idx, dtype=torch.long, device=device)
                    sub_neg = torch.tensor(sub_neg_idx, dtype=torch.long, device=device)

                    lp_loss, E = lp_criterion(sub_emb, sub_A, sub_pos, sub_neg)
                    contrast_loss = contrast_criterion(sub_emb, E, num_pairs=sub_emb.size(0) * rate_pairs)
                    loss = lpl_weight * lp_loss + (1.0 - lpl_weight) * contrast_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_epoch += loss.item()

        losses_per_epoch.append(total_loss_epoch)


        # Early stopping
        if early_stopping(total_loss_epoch):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss_epoch:.4f}")

    return losses_per_epoch


##############################################################################
# Set Seed
##############################################################################
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to enhance reproducibility.

    This function also configures PyTorch's CuDNN backend to be deterministic, 
    which can reduce non-determinism on GPU.

    Parameters
    ----------
    seed : int, default=42
        The seed used for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_random_numbers(n:int=5, seed:int=42, a:int=0, b:int=1000) -> List[float]:
    """
    Generates a list of n random numbers using the provided seed.

    Parameters:
        n (int): Number of random numbers to generate.
        Seed (int): Seed for the random number generator.
        a (int): Lower bound (inclusive).
        b (int): Upper bound (inclusive).

    Returns:
        List[float]: A list containing n random numbers between 0 and 1.
    """
    # Set the random state for reproducibility
    set_seed(seed)
    
    # Generate and return a list of n random numbers
    return [random.randint(a,b) for _ in range(n)]

##############################################################################
# Main
##############################################################################
def main(
    data: Data,
    params: Dict[str, float],
    device: torch.device,
    seed: int = 42
) -> Tuple[torch.nn.Module, List[float], SparseTensor]:
    """
    Main routine to initialize and train a GraphSAGE model with label propagation 
    and contrastive losses.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG Data object, containing at least:
          - data.x (node features)
          - data.edge_index (graph structure)
          - data.num_node_features
          - data.y (node labels)
          - data.train_mask (node mask for training)
          - data.test_mask (node mask for testing)
    params : dict
        Dictionary of hyperparameters with keys such as:
          - "hidden_channels"
          - "out_channels"
          - "num_layers"
          - "dropout"
          - "norm"
          - "aggregation"
          - "alpha"
          - "K"
          - "method"
          - "margin"
          - "pos_weight"
          - "ratio"
          - "lpl_weight"
          - "treatment"
          - "model_type"
          - "batch size"
          - "rate_pairs"
          - "reliable_mini_batch"
        The exact usage depends on the GraphSAGE and training process.
    device : torch.device
        The device (CPU/GPU) on which computations will be performed.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    model : torch.nn.Module
        The trained GraphSAGEEncoder model.
    train_losses : list of float
        Recorded training losses for each epoch.
    final_A_hat : SparseTensor
        The final adjacency matrix (possibly modified) after label propagation.
    """
    # Fix random seed for reproducibility
    set_seed(seed)

    # Prepare model input size
    in_channels = data.num_node_features

    # Build the GraphSAGE model
    model = GraphEncoder(
            model_type=params["model_type"],
            in_channels=in_channels,
            hidden_channels=params["hidden_channels"],
            out_channels=params["out_channels"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            norm=params["norm"],
            aggregation=params["aggregation"])

    # Train the model
    train_losses = train_graph(
        model=model,
        data=data,
        device=device,
        alpha=params["alpha"],
        K=params["K"],
        margin=params["margin"],
        pos_weight=params["pos_weight"],
        ratio=params["ratio"],
        lpl_weight=params["lpl_weight"],
        treatment=params["treatment"],
        rate_pairs=params["rate_pairs"],
        batch_size=params["batch_size"],
        reliable_mini_batch=params["reliable_mini_batch"]
    )

    return model, train_losses

##############################################################################
# Experiment Loop
##############################################################################
def run_nnif_gnn_experiment(params: Dict[str, Any]) -> Tuple[float, float]:
    dataset_name = params["dataset_name"]
    train_pct = params["train_pct"]
    mechanism = params["mechanism"]

    alpha = params["alpha"]
    K = params["K"]
    layers = params["layers"]
    hidden_channels = params["hidden_channels"]
    out_channels = params["out_channels"]
    norm = params["norm"]
    dropout = params["dropout"]
    margin = params["margin"]
    lpl_weight = params["lpl_weight"]
    ratio = params["ratio"]
    pos_weight = params["pos_weight"]
    aggregation = params["aggregation"]
    treatment = params["treatment"]
    model_type = params["model_type"]
    rate_pairs = params["rate_pairs"]
    batch_size = params["batch_size"]
    reliable_mini_batch=params["reliable_mini_batch"]
    min=params["min"]

    n_seeds = params["seeds"]

    f1_scores = []

    # Prepare output folder and CSV
    output_folder = f"{dataset_name}_experimentations"
    os.makedirs(output_folder, exist_ok=True)

    base_output_csv = params["output_csv"]
    timestamp = datetime.datetime.now().strftime("%d%m%H%M%S")
    if "." in base_output_csv:
        base, ext = base_output_csv.rsplit(".", 1)
        output_csv = os.path.join(output_folder, f"{base}_{timestamp}.{ext}")
    else:
        output_csv = os.path.join(output_folder, f"{base_output_csv}_{timestamp}.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seeds=generate_random_numbers(n=n_seeds)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "alpha", "K", "layers", "hidden_channels", "out_channels", "norm",
            "dropout", "margin", "lpl_weight", "ratio", "seed", "aggregation",
            "model_type", "sampling_mode", "num_neighbors", "pos_weight","batch_size","rate_pairs","reliable_mini_batch"
            "accuracy", "f1", "recall", "precision","losses","average_precision","roc_auc","fpr","tpr","roc_thresholds","precisions","recalls","pr_thresholds"
        ])
        
        for seed in seeds:
            # 1) Load dataset
            data = load_dataset(dataset_name)

            # 2) Create PU dataset
            data = make_pu_dataset(
                data,
                mechanism=mechanism,
                sample_seed=seed,
                train_pct=train_pct
            )

            data = data.to(device)
            if torch.isnan(data.x).any():
                print("NaN values in node features! Skipping seed...")
                continue

            print(f"Running experiment with seed={seed}:")
            print(f" - alpha={alpha}, K={K}, layers={layers}, hidden={hidden_channels}, out={out_channels}")
            print(f" - norm={norm}, dropout={dropout}, margin={margin}, lpl_weight={lpl_weight}")
            print(f" - ratio={ratio}, pos_weight={pos_weight}, aggregation={aggregation}, treatment={treatment}")
            print(f" - model_type={model_type}, rate_pairs={rate_pairs}, batch_size={batch_size}, reliable_mini_batch={reliable_mini_batch}")

            # 3) Train
            train_params = {
                "aggregation": aggregation,
                "pos_weight": pos_weight,
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "num_layers": layers,
                "alpha": alpha,
                "K": K,
                "norm": norm,
                "dropout": dropout,
                "lpl_weight": lpl_weight,
                "margin": margin,
                "ratio": ratio,
                "treatment": treatment,
                "model_type": model_type,
                "rate_pairs": rate_pairs,
                "batch_size": batch_size,
                "reliable_mini_batch": reliable_mini_batch
            }
            model, train_losses = main(data, train_params, device)
            A_hat = SparseTensor.from_edge_index(data.edge_index).coalesce().to(device)
            # --- 4) Evaluate the trained GNN: get embeddings ---
            model.eval()
            loader = NeighborLoader(
                    copy.copy(data),
                    input_nodes=data.test_mask,
                    num_neighbors=[-1]*K,
                    batch_size=2056,
                    shuffle=False
                )

            emb_dim = model(data.x.to(device), data.edge_index.to(device)).shape[1]
            embeddings = torch.zeros(data.num_nodes, emb_dim)

            with torch.no_grad():
                for batch in loader:
                    # Move the batch to the proper device
                    batch = batch.to(device)
                    # Compute embeddings for the batch
                    batch_emb = model(batch.x, batch.edge_index)
                    # batch.n_id contains the global node indices for the batch.
                    embeddings[batch.n_id] = batch_emb.cpu()

            # --- 5) PU/NNIF approach ---
            pnn_model = PNN(
                method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,         # or pass `seed` if needed
                high_score_anomaly=True
            )
            norm_emb = F.normalize(embeddings[data.test_mask.cpu()], dim=1)
            if not torch.isnan(norm_emb).any():
                features_np = norm_emb.detach().cpu().numpy()
                y_labels = data.train_mask.detach().cpu().numpy().astype(int)

                # Fit PNN on the labeled/unlabeled data
                pnn_model.fit(features_np, y_labels)
                predicted = pnn_model.predict(features_np)
                predicted_probs = pnn_model.predict_proba(features_np)[:,1]
                predicted_t = torch.from_numpy(predicted).to(embeddings.device)

                # Determine reliable neg/pos
                reliable_negatives = (predicted_t == 0)
                reliable_positives = (predicted_t == 1)

                # Combine them for "training" data
                combined_mask = reliable_positives | reliable_negatives
                train_labels = torch.zeros_like(combined_mask, dtype=torch.float)
                train_labels[reliable_negatives] = 0.0
                train_labels[reliable_positives] = 1.0
                train_labels = train_labels[combined_mask]
            else:
                train_labels = data.train_mask
                print("NaN values in node embeddings! Using training labels...")

            # 6) Compute metrics against the ground truth
            labels_np = data.y[data.test_mask].cpu().numpy()           # ground truth
            preds_np = train_labels.cpu().numpy()      # predicted
            accuracy = accuracy_score(labels_np, preds_np)
            f1 = f1_score(labels_np, preds_np)
            recall = recall_score(labels_np, preds_np)
            precision = precision_score(labels_np, preds_np)

            average_precision = average_precision_score(labels_np, predicted_probs)
            roc_auc = roc_auc_score(labels_np, predicted_probs)
            fpr, tpr, roc_thresholds = roc_curve(labels_np, predicted_probs)
            precisions, recalls, pr_thresholds = precision_recall_curve(labels_np, predicted_probs)
            
            f1_scores.append(f1)  # Track F1 across seeds

            print(f" - Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

            # Otherwise record results
            f1_scores.append(f1)
            writer.writerow([
                alpha, K, layers, hidden_channels, out_channels, norm, dropout,
                margin, lpl_weight, ratio, seed, aggregation, model_type, pos_weight,batch_size,rate_pairs,
                accuracy, f1, recall, precision, train_losses, average_precision, roc_auc, fpr, tpr, roc_thresholds, precisions, recalls, pr_thresholds
            ])

            if f1 < min:
                print(f"F1 = {f1:.2f} < {min}, skipping ...")
                break

    # Summarize results
    if len(f1_scores) > 0:
        avg_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
    else:
        avg_f1, std_f1 = 0.0, 0.0

    print(f"Done. Results written to {output_csv}.")
    print(f"Average F1 over valid seeds: {avg_f1:.4f} ± {std_f1:.4f}")

    return avg_f1, std_f1
