import os
import sys
import csv
import datetime
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from typing import Dict, Tuple, List, Any
import logging

from loss import LabelPropagationLoss, ContrastiveLoss
from NNIF import PNN, ReliableValues, WeightedIsoForest
from encoder import GraphSAGEEncoder
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
    data,
    device: torch.device,
    alpha: float,
    K: int,
    treatment: str,
    margin: float,
    pos_weight: float,
    ratio: float = 0.1,
    lpl_weight: float = 0.5,
    num_epochs: int = 500,
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    weight_decay: float = 1e-6
):
    """
    Train a GraphSAGE-based model with a Label Propagation + Contrastive Loss workflow.

    The main steps are:
      1) Construct an adjacency (A_hat) from data.edge_index, including self-loops.
      2) Forward pass to get node embeddings.
      3) Anomaly detection => retrieve reliable positives/negatives.
      4) Label Propagation loss.
      5) Contrastive loss.
      6) Combine losses, backprop, and update parameters.
      7) (Optional) Early stopping.

    Parameters
    ----------
    model : nn.Module
        A GraphSAGEEncoder or similar model that produces embeddings.
    data : torch_geometric.data.Data
        PyG Data object with fields like data.x, data.edge_index, etc.
    device : torch.device
        The device (CPU/GPU) for training.
    alpha : float
        Parameter for label propagation (mixing factor).
    K : int
        Number of label propagation steps.
    treatment : str
        String indicating treatment for anomaly detection (passed to ReliableValues).
    margin : float
        Margin used in the contrastive loss.
    pos_weight : float
        Weight for positive loss in label propagation.
    ratio : float, default=0.1
        Fraction of negative samples to treat as anomalies (used in anomaly detector).
    lpl_weight : float, default=0.5
        Fraction of total loss allocated to the label propagation term 
        (the other 1 - lpl_weight goes to contrastive).
    num_epochs : int, default=500
        Maximum number of training epochs.
    lr : float, default=0.01
        Learning rate for the AdamW optimizer.
    max_grad_norm : float, default=1.0
        Gradient norm clipping threshold.
    weight_decay : float, default=1e-6
        Weight decay (L2 regularization) for AdamW.

    Returns
    -------
    train_losses : list of float
        The recorded training losses at each epoch.
    final_A_hat : SparseTensor
        Potentially updated adjacency after label propagation steps.
    """
    from torch.nn.utils import clip_grad_norm_  # local import for clarity

    optimizer = optim.Adam(model.parameters())#, lr=lr, weight_decay=weight_decay)
    """    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=1,
        epochs=num_epochs
    )"""
    early_stopping = EarlyStopping_GNN(patience=20)
    scaler = GradScaler()

    # Step 1: Build adjacency with self-loops
    #edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    #edge_index = coalesce(edge_index)
    A_hat = SparseTensor.from_edge_index(data.edge_index).coalesce().to(device)


    # Tracking variables
    train_losses = []
    best_loss = float('inf')

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):
        #print_cuda_meminfo(f"Epoch {epoch} start")

        model.train()
        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # 2) Forward pass with current adjacency
            embeddings = model(data.x.to(device), A_hat)
            #print_cuda_meminfo(f"Epoch {epoch} after forward")

            # Anomaly detection => reliable positives/negatives
            if epoch == 0:
                NNIF = ReliableValues(
                    method=treatment,
                    treatment_ratio=ratio,
                    anomaly_detector=WeightedIsoForest(n_estimators=200),
                    random_state=42,
                    high_score_anomaly=True,
                )
                norm_emb = F.normalize(embeddings, dim=1)
                features_np = norm_emb.detach().cpu().to(torch.float32).numpy()
                y_labels = data.train_mask.detach().cpu().numpy().astype(int)
                reliable_negatives, reliable_positives = NNIF.get_reliable(features_np, y_labels)

            # 3) Label Propagation
            lp_criterion = LabelPropagationLoss(
                A_hat=A_hat,
                alpha=alpha,
                K=K,
                pos_weight=pos_weight
                ).to(device)
            lpl_loss, updated_A_hat, E = lp_criterion(embeddings, reliable_positives, reliable_negatives)

            # 4) Contrastive Loss
            contrast_criterion = ContrastiveLoss(margin=margin,num_pairs=5*data.num_nodes).to(device)
            contrastive_loss = contrast_criterion(embeddings, E)

            # 5) Combine losses
            loss = lpl_weight * lpl_loss + (1.0 - lpl_weight) * contrastive_loss

        # Backprop
        scaler.scale(loss).backward(retain_graph=True)
        #print_cuda_meminfo(f"Epoch {epoch} after backward")

        # Gradient clipping
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        #scheduler.step()

        # Record loss
        loss_val = loss.item()
        train_losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val

        # 6) Update adjacency (if needed)
        A_hat = updated_A_hat
        # Example: If you need to re-inject self-loops:
        # A_hat = add_self_loops_to_sparse(A_hat).coalesce()

        # Logging
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss_val:.4f}, "
                f"LPL: {lpl_loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}"
            )
            logger.info(
                f"Epoch {epoch}, Loss: {loss_val:.4f}, "
                f"LPL: {lpl_loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}"
            )

        # Early stopping check
        if early_stopping(loss_val):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Optional: print(torch.cuda.memory_summary())

    return train_losses, A_hat

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
          - Optionally other fields like data.train_mask, etc.
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
    model = GraphSAGEEncoder(
        in_channels=in_channels,
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        norm=params["norm"],
        aggregation=params["aggregation"]
    )
    #print(params)
    # Train the model
    train_losses, final_A_hat = train_graph(
        model=model,
        data=data,
        device=device,
        alpha=params["alpha"],
        K=params["K"],
        margin=params["margin"],
        pos_weight=params["pos_weight"],
        ratio=params["ratio"],
        lpl_weight=params["lpl_weight"],
        treatment=params["treatment"]
    )

    return model, train_losses, final_A_hat

##############################################################################
# Experiment Loop
##############################################################################
def run_nnif_gnn_experiment(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Run a single experiment configuration for NNIF + GNN, looping only over 
    random seeds from 1..params['seeds'].

    All parameters are given via `params`, a dictionary that must contain:

        {
          "dataset_name": str,
          "train_pct": float,
          "mechanism": str,

          "alpha": float,
          "K": int,
          "layers": int,
          "hidden_channels": int,
          "out_channels": int,
          "norm": str or None,
          "dropout": float,
          "margin": float,
          "lpl_weight": float,
          "ratio": float,
          "pos_weight": float,
          "aggregation": str,
          "treatment": str,
          "seeds": int,            # number of repeated runs
          "output_csv": str        # path to CSV file
        }

    Returns
    -------
    (avg_f1, std_f1) : (float, float)
        The mean F1 score and standard deviation of F1 over all repeated seeds.
    """
    # Unpack parameters from dict
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

    n_seeds = params["seeds"]
    output_csv = params["output_csv"]

    output_folder = f"{dataset_name}_experimentations"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the output file name from parameters
    base_output_csv = params["output_csv"]
    
    # Append current day, month, hour, and second to the CSV filename
    timestamp = datetime.datetime.now().strftime("%d%m%H%M%S")
    if "." in base_output_csv:
        base, ext = base_output_csv.rsplit(".", 1)
        output_csv = os.path.join(output_folder, f"{base}_{timestamp}.{ext}")
    else:
        output_csv = os.path.join(output_folder, f"{base_output_csv}_{timestamp}.csv")

    # Decide on CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For storing F1 scores across all seeds
    f1_scores = []

    # Open the CSV file once, write all seeds' results inside
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "alpha", "K", "layers", "hidden_channels", "out_channels", "norm",
            "dropout", "margin", "lpl_weight", "ratio", "seed", "aggregation",
            "pos_weight", "accuracy", "f1", "recall", "precision"
        ])

        # Loop over seeds for repeated runs
        for seed in range(1, n_seeds + 1):
            # --- 1) Load dataset ---
            data = load_dataset(dataset_name)

            # --- 2) Create a PU dataset (some positives labeled) ---
            data = make_pu_dataset(
                data,
                mechanism=mechanism,
                sample_seed=seed,
                train_pct=train_pct
            )

            data = data.to(device)

            # Print parameters for reference
            print(f"Running experiment with seed={seed}:")
            print(f" - alpha={alpha}, K={K}, layers={layers}, hidden={hidden_channels}, out={out_channels}")
            print(f" - norm={norm}, dropout={dropout}, margin={margin}, lpl_weight={lpl_weight}")
            print(f" - ratio={ratio}, pos_weight={pos_weight}, aggregation={aggregation}, treatment={treatment}")

            # --- 3) Train GNN with these parameters ---
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
                "treatment":treatment
            }
            model, train_losses, A_hat = main(data, train_params, device)

            # --- 4) Evaluate the trained GNN: get embeddings ---
            model.eval()
            embeddings = model(data.x, A_hat)

            # --- 5) PU/NNIF approach ---
            pnn_model = PNN(
                method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,         # or pass `seed` if needed
                high_score_anomaly=True
            )
            norm_emb = F.normalize(embeddings, dim=1)
            features_np = norm_emb.detach().cpu().numpy()
            y_labels = data.train_mask.detach().cpu().numpy().astype(int)

            # Fit PNN on the labeled/unlabeled data
            pnn_model.fit(features_np, y_labels)
            predicted = pnn_model.predict(features_np)
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

            # 6) Compute metrics against the ground truth
            labels_np = data.y.cpu().numpy()           # ground truth
            preds_np = train_labels.cpu().numpy()      # predicted
            accuracy = accuracy_score(labels_np, preds_np)
            f1 = f1_score(labels_np, preds_np)
            recall = recall_score(labels_np, preds_np)
            precision = precision_score(labels_np, preds_np)

            f1_scores.append(f1)  # Track F1 across seeds

            print(f" - Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

            # 7) Write row to CSV
            writer.writerow([
                alpha, K, layers, hidden_channels, out_channels, norm, dropout,
                margin, lpl_weight, ratio, seed, aggregation, pos_weight,
                accuracy, f1, recall, precision
            ])

    # After all seeds, compute mean & std of F1
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0

    print(f"Done. Results written to {output_csv}.")
    print(f"Average F1 over {n_seeds} seeds: {avg_f1:.4f} ± {std_f1:.4f}")

    return avg_f1, std_f1
