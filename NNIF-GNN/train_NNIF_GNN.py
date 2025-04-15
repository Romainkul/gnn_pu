import os
import sys
import csv
import datetime
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,average_precision_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
#from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_sparse import SparseTensor
from typing import Dict, Tuple, List, Any
import logging
import copy
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.transforms import KNNGraph

from NN_loader import ShineLoader
from loss import LabelPropagationLoss, ContrastiveLoss
from NNIF import PNN, ReliableValues, WeightedIsoForest
from encoder import GraphEncoder
from data_generating import load_dataset, make_pu_dataset

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from nnpu import train_nnpu
from NNIF import train_two

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
        loss_diff_threshold: float = 1e-3
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
    sampling: str = "cluster"
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Trains a GNN model on the given graph data using Label Propagation + Contrastive Loss.
    Additionally, it performs an NNIF-based step at epoch 0 to identify reliable pos/neg
    samples in each mini-batch. After training, it computes embeddings for all nodes, then
    applies a PNN-based approach (PU/NNIF) to produce final 'train_labels'.

    Parameters
    ----------
    model : nn.Module
        The GNN model to train.
    data : torch_geometric.data.Data
        Graph data object containing:
          - data.x: node features
          - data.edge_index: graph edges
          - data.train_mask, data.test_mask: masks (optional)
    device : torch.device
        'cpu' or 'cuda' for computation.
    K : int
        Hyperparameter for LabelPropagationLoss (neighbors).
    treatment : str
        NNIF treatment strategy ('removal' or 'relabel').
    rate_pairs : int
        Multiplier for the number of negative pairs in ContrastiveLoss
        (e.g., sub_emb.size(0) * rate_pairs).
    batch_size : int
        Mini-batch size for ClusterLoader.
    ratio : float
        Ratio for ReliableValues to identify outliers vs. reliable samples.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the AdamW optimizer.
    weight_decay : float
        L2 regularization factor.
    cluster : int
        Number of parts/clusters for ClusterData (for large graphs).
    layers : int
        Number of GNN layers in the model.
    anomaly_detector : str
        Anomaly detector (e.g., 'nearest_neighbors' or 'unweighted').
    sampling : str
        Sampling method for NeighborLoader ('cluster', 'neighbor', or 'nearest_neighbor').

    Returns
    -------
    (train_labels, predicted_probs, losses_per_epoch)
        train_labels : torch.Tensor
            Binary labels determined by PNN on the final embeddings (1=pos, 0=neg).
        predicted_probs : torch.Tensor
            Probability estimates from PNN for each node being positive.
        losses_per_epoch : List[float]
            Total loss per epoch during training.
    """

    # Instantiate the needed criteria & early stopping
    lp_criterion = LabelPropagationLoss(K=K).to(device)
    contrast_criterion = ContrastiveLoss().to(device)
    early_stopper = EarlyStopping_GNN(patience=20)

    data.n_id = torch.arange(data.num_nodes)
    if sampling=="cluster":
        if batch_size==256:
            batch_size=5
        elif batch_size==512:
            batch_size=10
        elif batch_size==1024:
            batch_size=20
        #print(f"Batch size: {batch_size}")
        cluster_data = ClusterData(data, num_parts=cluster)
        #print(f"Number of clusters: {len(cluster_data)}")
        train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
    elif sampling=="sage":
        train_loader = NeighborLoader(copy.copy(data),num_neighbors=[25,10],batch_size=batch_size,shuffle=True)
    elif sampling=="shine":
        train_loader = ShineLoader(copy.copy(data),num_neighbors=[2,32], shuffle=True,batch_size=batch_size,device=device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")
    #print(f"Number of batches: {len(train_loader)}")
    # Move model & data to device
    model = model.to(device)
    data = data.to(device)

    # Set up optimizer
    optimizer = optim.AdamW(
        list(model.parameters())
        + list(lp_criterion.parameters())
        + list(contrast_criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scaler = GradScaler()

    # Track total loss each epoch
    losses_per_epoch = []

    # Global sets of reliable positives/negatives found at epoch 0
    reliable_pos_set = set()
    reliable_neg_set = set()

    # -----------------------
    # 1) Training loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for sub_data in train_loader:
            sub_data = sub_data.to(device)
            global_nids = sub_data.n_id
            num_sub_nodes = global_nids.size(0)

            # Build adjacency for subgraph
            sub_adj = SparseTensor.from_edge_index(
                sub_data.edge_index,
                sparse_sizes=(num_sub_nodes, num_sub_nodes)
            ).coalesce().to(device)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available(), device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                sub_emb = model(sub_data.x, sub_adj)

                # Epoch 0: identify reliable samples in each mini-batch
                if epoch == 0:
                    # Step A: NNIF detection
                    norm_sub_emb = F.normalize(sub_emb, dim=1)
                    features_np = norm_sub_emb.detach().cpu().numpy()

                    # Assume sub_data has train_mask for labeling
                    y_labels = sub_data.train_mask.detach().cpu().numpy().astype(int)
                    nnif_detector = ReliableValues(
                        method=treatment,
                        treatment_ratio=ratio,
                        anomaly_detector=WeightedIsoForest(n_estimators=100, type_weight=anomaly_detector),
                        random_state=42,
                        high_score_anomaly=True
                    )

                    neg_mask, pos_mask = nnif_detector.get_reliable(features_np, y_labels)
                    global_ids_np = global_nids.cpu().numpy()
                    # Collect in global sets
                    for i in range(num_sub_nodes):
                        if pos_mask[i]:
                            reliable_pos_set.add(int(global_ids_np[i]))
                        if neg_mask[i]:
                            reliable_neg_set.add(int(global_ids_np[i]))
                    sub_pos_idx = [
                            i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set
                        ]
                    sub_neg_idx = [
                            i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set
                        ]

                
                # Retrieve indices from global sets
                else:
                    global_ids_np = global_nids.cpu().numpy()
                    sub_pos_idx = [
                            i for i, gid in enumerate(global_ids_np) if gid in reliable_pos_set
                        ]
                    sub_neg_idx = [
                            i for i, gid in enumerate(global_ids_np) if gid in reliable_neg_set
                        ]
                sub_pos = torch.tensor(sub_pos_idx, dtype=torch.long, device=device)
                sub_neg = torch.tensor(sub_neg_idx, dtype=torch.long, device=device)

                # Label Propagation + Contrastive
                lp_loss, E = lp_criterion(sub_emb, sub_adj, sub_pos, sub_neg)
                contrast_loss = contrast_criterion(
                        sub_emb, E, num_pairs=sub_emb.size(0) * rate_pairs
                )
                loss = lp_loss + contrast_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

        losses_per_epoch.append(epoch_loss)

        # Early stopping check
        if early_stopper(epoch_loss):
            # logger.info(f"[Early Stopping] at epoch {epoch}")  # If you have a logger
            print(f"[Early Stopping] at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch} / {num_epochs}, Loss: {epoch_loss:.4f}")

    # -----------------------
    # 2) Compute final embeddings for entire graph
    # -----------------------
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

    # -----------------------
    # 3) PNN-based step (PU/NNIF) on final embeddings
    # -----------------------
    pnn_model = PNN(
        method=treatment,
        treatment_ratio=ratio,
        anomaly_detector=WeightedIsoForest(n_estimators=100, type_weight=anomaly_detector),
        random_state=42,
        high_score_anomaly=True
    )

    norm_emb = F.normalize(embeddings, dim=1)
    if not torch.isnan(norm_emb).any():
        features_np = norm_emb.detach().cpu().numpy()
        train_y_labels = data.train_mask.detach().cpu().numpy().astype(int)

        # Fit PNN on entire node set (or just training portion) as needed
        pnn_model.fit(features_np, train_y_labels)

        predicted = pnn_model.predict(features_np)  # 0 or 1
        predicted_probs_np = pnn_model.predict_proba(features_np)[:, 1]

        predicted_t = torch.from_numpy(predicted).to(device)
        predicted_probs_t = torch.from_numpy(predicted_probs_np).to(device)

        # Derive reliable pos/neg or combine them if needed
        reliable_neg_mask = (predicted_t == 0)
        reliable_pos_mask = (predicted_t == 1)

        # Combine them for some final training labels (binary)
        combined_mask = reliable_pos_mask | reliable_neg_mask
        train_labels = torch.zeros_like(combined_mask, dtype=torch.float, device=device)
        train_labels[reliable_pos_mask] = 1.0

    else:
        # Fallback: if embeddings have NaN
        print("[Warning] Found NaN in embeddings, reverting to data.train_mask.")
        train_labels = data.train_mask.float()
        predicted_probs_t = torch.zeros(data.num_nodes, device=device)

    return train_labels, predicted_probs_t, losses_per_epoch


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
# Experiment Loop
##############################################################################
def run_nnif_gnn_experiment(params: Dict[str, Any], seed:int=42) -> Tuple[float, float]:
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
    lr=params["lr"]
    clusters=params["clusters"]
    min=params["min"]
    n_seeds = params["seeds"]
    num_epochs = params["num_epochs"]
    sampling = params["sampling"]
    val=params["val"]

    f1_scores = []

    # Prepare output folder and CSV
    output_folder = f"{dataset_name}_experimentations"
    os.makedirs(output_folder, exist_ok=True)

    base_output_csv = params["output_csv"]
    timestamp = datetime.datetime.now().strftime("%d%m%H%M%S")
    if "." in base_output_csv:
        base, ext = base_output_csv.rsplit(".", 1)
        if methodology=="ours":
            output_csv = os.path.join(output_folder, f"{base}_{timestamp}.{ext}")
        else:
            output_csv = os.path.join(output_folder, f"{base}_{methodology}_{timestamp}.{ext}")
    else:
        if methodology=="ours":
            output_csv = os.path.join(output_folder, f"{base}_{timestamp}.csv")
        else:
            output_csv = os.path.join(output_folder, f"{base}_{methodology}_{timestamp}.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(seed)
    
    seeds_list=generate_random_numbers(n=n_seeds)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "K", "layers", "hidden_channels", "out_channels", "norm","lr","treatment",
            "dropout", "ratio", "seed", "aggregation", "model_type","batch_size","rate_pairs","clusters","sampling","num_epochs","anomaly_detector",
            "accuracy", "f1", "recall", "precision","losses"
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
            #print(data)
            # Prepare model input size
            in_channels = data.num_node_features

            #data = data.to(device)
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
                            aggregation=aggregation)
            in_numpy=False
            try:    
                if methodology=="ours":
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
                        sampling=sampling
                    )
                                    
                elif methodology == "XGBoost":
                    model = XGBClassifier()
                    model.fit(data.x.cpu().numpy(), data.train_mask.cpu().numpy())
                    preds_np, proba_np = model.predict(data.x[data.val_mask].cpu().numpy()), model.predict_proba(data.x[data.val_mask].cpu().numpy())[:, 1]
                    preds_np_test, proba_np_test = model.predict(data.x[data.test_mask].cpu().numpy()), model.predict_proba(data.x[data.test_mask].cpu().numpy())[:, 1]
                    in_numpy=True
                    train_losses=[]
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
                    in_numpy=True
                    train_losses=[]

                elif methodology in  ["nnpu","imbnnpu"]:
                    nnpu= True
                    imbnnpu = True if methodology=="imbnnpu" else False
                    train_labels, train_proba, train_losses = train_nnpu(model,data,device,model_type,layers,batch_size,lr,prior=data.prior,nnpu=nnpu,imbpu=imbnnpu, max_epochs=num_epochs)

                elif methodology in ["two_nnif","spy", "naive"]:
                    methodo = "NNIF" if methodology == "two_nnif" else "SPY" if methodology == "spy" else "naive"
                    train_labels, train_proba, train_losses = train_two(model,data, device,methodology=methodo,layers=layers,ratio=ratio,model_type=model_type,num_epochs=num_epochs, batch_size=batch_size)
                
                labels_np = data.y[data.val_mask].cpu().numpy()
                if not in_numpy:
                    preds_np = train_labels[data.val_mask.cpu()].cpu().numpy()
                    proba_np = train_proba[data.val_mask.cpu()].cpu().numpy()

                accuracy = accuracy_score(labels_np, preds_np)
                f1 = f1_score(labels_np, preds_np)
                recall = recall_score(labels_np, preds_np)
                precision = precision_score(labels_np, preds_np)
                labels_np_test = data.y[data.test_mask].cpu().numpy()    
                if not in_numpy:
                    preds_np_test = train_labels[data.test_mask.cpu()].cpu().numpy()
                    proba_np_test = train_proba[data.test_mask.cpu()].cpu().numpy()
                accuracy_test = accuracy_score(labels_np_test, preds_np_test)
                f1_test = f1_score(labels_np_test, preds_np_test)
                recall_test = recall_score(labels_np_test, preds_np_test)
                precision_test = precision_score(labels_np_test, preds_np_test)
                #ap_test=average_precision_score(labels_np_test, proba_np_test)
                
                print(f" - Test Metrics: Accuracy={accuracy_test:.4f}, F1={f1_test:.4f}, Recall={recall_test:.4f}, Precision={precision_test:.4f}")
                print(f" - Validation Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

                if val:
                    f1_scores.append(f1)  # Track F1 across seeds
                else:
                    f1_scores.append(f1_test)

                writer.writerow([
                        K, layers, hidden_channels, out_channels, norm, lr, treatment, dropout,
                        ratio, exp_seed, aggregation, model_type, batch_size, rate_pairs,clusters,sampling,num_epochs,anomaly_detector,
                        accuracy, f1, recall, precision, train_losses
                        ])

                if val and (f1 < min):
                    print(f"F1 = {f1:.2f} < {min}, skipping ...")
                    break
            except Exception as e:
                print(f"Error: {e}")
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
