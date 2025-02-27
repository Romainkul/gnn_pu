import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Custom modules
from nnPU import PULoss
from encoders import BaseEncoder  
from data_generating import load_dataset, make_pu_dataset

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to enhance reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate(
    dataset_name: str = "citeseer",
    mechanism: str = "SCAR",
    seed: int = 1,
    train_pct: float = 0.5,
    hidden_dim: int = 16,
    output_dim: int = 2,
    num_layers: int = 3,
    dropout: float = 0.5,
    num_epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    model_type: str = "GraphSAGE",
    imbalance: bool = False
):
    """
    Trains a PU-GNN model using PULoss and evaluates on a test set.
    This function ensures reproducibility and fixes the common issue where a two-dimensional
    model output causes a dimension mismatch with the PU loss, by selecting the positive class logit.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'citeseer').
    mechanism : str
        PU mechanism (e.g., 'SCAR').
    seed : int
        Random seed for reproducibility.
    train_pct : float
        Fraction of positive nodes to treat as labeled.
    hidden_dim : int
        Hidden layer dimensionality.
    output_dim : int
        Output dimensionality (2 for binary classification with two logits).
    num_layers : int
        Number of layers in the encoder.
    dropout : float
        Dropout probability.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for AdamW.
    model_type : str
        Type of encoder to use ('GCN', 'GAT', 'GraphSAGE', 'GIN', 'MLP').
    imbalance : bool
        Whether to use an imbalanced PU loss variant.
    
    Returns
    -------
    None
        Prints training loss and final test metrics.
    """
    # 1) Set reproducible seed
    set_seed(seed)

    # 2) Load and prepare dataset
    data = load_dataset(dataset_name)
    # make_pu_dataset should create train/val/test masks and update .y (with 1 for observed positive, -1 for unlabeled)
    data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # 2a) Enforce target encoding: ensure positives are 1 and unlabeled samples are -1
    # If your original data.y is not already in {1, -1} (e.g., it might be {1, 0}), convert it:
    data.y = torch.where(data.y > 0, torch.tensor(1, device=data.y.device), torch.tensor(-1, device=data.y.device))

    # 3) Initialize Model and Loss
    model = BaseEncoder(
        input_dim=data.num_features,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type
    ).to(device)

    # Use the dataset's known prior if available; else, default to 0.5
    pu_loss = PULoss(
        prior=data.prior if hasattr(data, 'prior') else 0.5,
        beta=0.0,
        nnpu=True,
        imbpu=imbalance
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4) Training Loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass: output shape is [num_nodes, output_dim]
        out = model(data.x, data.edge_index)
        # If output_dim == 2, select the logit corresponding to the positive class
        if output_dim == 2:
            out = out[:, 1]  # shape: [num_nodes]

        # Compute PU Loss on the training subset (using data.train_mask)
        train_mask = data.train_mask
        loss = pu_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1:03d}/{num_epochs}] Loss: {loss.item():.4f}")

    print("Training Complete!")

    # 5) Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if output_dim == 2:
            logits = logits[:, 1]  # get positive class logits
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().cpu().numpy()
        # Convert labels: positive is 1, unlabeled (-1) is converted to 0 for metric computation
        labels = (data.y == 1).long().cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    rec = recall_score(labels, preds)
    prec = precision_score(labels, preds)

    print("\n=== Evaluation on Test Set ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")

