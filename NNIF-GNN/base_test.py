import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# For classical ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# PyG imports
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Custom imports (assumed to exist)
from encoder import GraphEncoder
from nnpu import PULoss
from data_generating import load_dataset, make_pu_dataset
from NNIF import PNN, ReliableValues, WeightedIsoForest, SpyEM

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
    
###############################################################################
# train_model
###############################################################################
def train_model(
    model,
    data: Data,
    device: torch.device,
    model_type: str,
    num_epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    use_pu: bool = False,
    prior: float = 0.5
) -> list:
    """
    Trains either a classical ML model (RF, XGB, LogisticRegression) or a
    neural network (MLP/GNN) on the given PyG Data object.

    Parameters
    ----------
    model       : Object
        - If model_type is one of ["RandomForest", "LogisticRegression", "XGBoost"],
          'model' should be a scikit-learn-like model (fit/predict methods).
        - Otherwise, a PyTorch nn.Module (e.g., GraphEncoder).
    data        : torch_geometric.data.Data
        PyG Data with:
          - data.x            (features)       [num_nodes, num_features]
          - data.y            (labels)         [num_nodes]
          - data.edge_index   (graph structure) [2, num_edges] (for GNNs)
          - data.train_mask   (bool)           [num_nodes], optional
          - data.test_mask    (bool)           [num_nodes], optional
    device      : torch.device
        The device (CPU or GPU) to run on.
    model_type  : str
        E.g., "RandomForest", "LogisticRegression", "XGBoost",
               or "MLP", "GCNConv", "GATConv", "SAGEConv", "GINConv", etc.
    num_epochs  : int, default=50
        Number of training epochs (only for PyTorch models).
    lr          : float, default=0.01
        Learning rate (only for PyTorch models).
    weight_decay: float, default=5e-4
        Weight decay (L2 regularization) for PyTorch optimizers.
    use_pu      : bool, default=False
        If True, uses a PU learning loss (PULoss). If False, uses CrossEntropy.
        Only applies to neural models.
    prior       : float, default=0.5
        Class prior for PULoss if use_pu is True.

    Returns
    -------
    losses_per_epoch : list of float
        The training loss recorded at each epoch (for neural models). An empty
        list is returned for classical ML models.
    """
    # -----------------------------------------
    # 1) Classical ML Models
    # -----------------------------------------
    if model_type in ["RandomForest", "LogisticRegression", "XGBoost"]:
        # For scikit-learn style models, we train only on train_mask
        # (assuming it exists). If not, you might just use all nodes.
        train_mask = data.train_mask if hasattr(data, "train_mask") else None
        if train_mask is not None:
            train_idx = train_mask.nonzero(as_tuple=True)[0]
        else:
            # fallback: use all nodes
            train_idx = torch.arange(data.num_nodes)

        # Move features/labels to CPU numpy
        X_train = data.x[train_idx].cpu().numpy()
        y_train = data.y[train_idx].cpu().numpy()

        # Fit the model
        model.fit(X_train, y_train)

        # We don't have a training "loss" from classical ML, so return empty list
        return []

    # -----------------------------------------
    # 2) Neural Networks (MLP / GNN)
    # -----------------------------------------
    model = model.to(device)
    data = data.to(device)

    # Identify training nodes
    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    else:
        # fallback: use all
        train_idx = torch.arange(data.num_nodes, device=device)

    # Set up the loss function (CrossEntropy or PU)
    if use_pu:
        loss_fn = PULoss(prior=prior, gamma=1, beta=0, nnpu=True, imbpu=False).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping_GNN(patience=20)
    scaler = GradScaler()
    best_loss = float('inf')
    losses_per_epoch = []

    reliable_negatives, reliable_positives = None, None  # placeholders
    max_grad_norm: float = 1.0
    treatment: str = "removal"
    ratio: float = 0.1
    for epoch in range(num_epochs):
        # ---------------------------------------------------------
        # (A) Optionally extract reliable sets at epoch 0
        # ---------------------------------------------------------
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                # Get embeddings for all nodes
                embeddings = model(data.x, data.edge_index)  # [num_nodes, embed_dim] presumably
            # Normalize for the anomaly detector
            norm_emb = F.normalize(embeddings, dim=1)
            features_np = norm_emb.detach().cpu().numpy()
            y_labels = data.train_mask.detach().cpu().numpy().astype(int)

            # Use ReliableValues to find reliable neg/pos
            NNIF = ReliableValues(
                method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,
                high_score_anomaly=True
            )
            reliable_negatives, reliable_positives = NNIF.get_reliable(features_np, y_labels)

            # Switch back to training mode
            model.train()

        # ---------------------------------------------------------
        # (B) Standard forward + loss + backprop
        # ---------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            out = model(data.x, data.edge_index)  # shape: [num_nodes, out_dim]

            # Squeeze if needed (e.g., if out is [..., 1])
            if out.dim() > 1 and out.size(-1) == 1:
                out = out.squeeze(-1)

            # Training loss only on the train mask
            loss = loss_fn(out[train_idx], data.y[train_idx])

        scaler.scale(loss).backward()

        # Gradient clipping
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Record loss
        loss_val = loss.item()
        losses_per_epoch.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val

    return losses_per_epoch


###############################################################################
# evaluate_model
###############################################################################
def evaluate_model(
    model,
    data: Data,
    device: torch.device,
    model_type: str,
) -> dict:
    """
    Evaluates the model on the entire dataset (or a test split, if desired).

    For classical models, it uses model.predict / model.predict_proba on all nodes.
    For neural networks, it does a forward pass and obtains predictions.

    Returns a dictionary of evaluation metrics: precision, recall, f1, avg_precision.
    """
    data = data.to(device)
    if model_type in ["RandomForest", "LogisticRegression", "XGBoost"]:
        # Classical ML => predict on entire dataset or test_mask only
        X_all = data.x.cpu().numpy()
        y_all = data.y.cpu().numpy()

        # Predict
        y_pred = model.predict(X_all)
        y_proba = model.predict_proba(X_all)[:, 1]

        prec = precision_score(y_all, y_pred, pos_label=1)
        rec = recall_score(y_all, y_pred, pos_label=1)
        f1 = f1_score(y_all, y_pred, pos_label=1)
        ap = average_precision_score(y_all, y_proba)

        return {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "avg_precision": ap
        }

    # Neural model => forward pass
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # Predictions
        preds = out.argmax(dim=-1).cpu().numpy()
        # Probabilities (assuming out is [N, 2])
        probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

    y_all = data.y.cpu().numpy()
    prec = precision_score(y_all, preds, pos_label=1)
    rec = recall_score(y_all, preds, pos_label=1)
    f1 = f1_score(y_all, preds, pos_label=1)
    ap = average_precision_score(y_all, probs)

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "avg_precision": ap
    }


###############################################################################
# main
###############################################################################
def main():
    """
    Example main routine showing how to:
      1) Load/create the data
      2) Make PU dataset
      3) Pick model type
      4) Train
      5) Evaluate
    """
    dataset_name = 'citeseer'
    mechanism = 'SCAR'
    seed = 1
    train_pct = 0.5
    model_type = "GraphSAGE"  # or "RandomForest", "LogisticRegression", "XGBoost", etc.
    two_step = "NNIF"         # or "Spy"

    # --------------------------------------------------
    # 1) Load and prepare data
    # --------------------------------------------------
    data = load_dataset(dataset_name)
    data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 2) Construct the model
    # --------------------------------------------------
    if model_type == "RandomForest":
        model = RandomForestClassifier()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    elif model_type == "XGBoost":
        model = XGBClassifier()
        
    elif model_type == "NNIF":
        # If you have a 2-step approach or specialized class
        model = PNN(method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,
                high_score_anomaly=True)
    else:
        hidden_dim = 64
        output_dim = 2
        num_layers = 3
        dropout = 0.5
        model = GraphEncoder(
            num_nodes=data.num_nodes,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )

    # --------------------------------------------------
    # 3) Train
    # --------------------------------------------------
    use_pu = False  # Set True if you want PULoss
    losses = train_model(
        model=model,
        data=data,
        device=device,
        model_type=model_type,
        num_epochs=50,
        lr=0.01,
        weight_decay=5e-4,
        use_pu=use_pu
    )

    # --------------------------------------------------
    # 4) Possibly do a 2-step approach (NNIF, Spy, etc.)
    # --------------------------------------------------
    if two_step == "NNIF" and output_dim > 2:
            model.eval()
            embeddings = model(data.x, A_hat)

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
            proba = pnn_model.predict_proba(features_np)
            predicted_t = torch.from_numpy(predicted).to(embeddings.device)
    elif two_step == "IF" and output_dim > 2:
        model.eval()
            embeddings = model(data.x, A_hat)

            # --- 5) PU/NNIF approach ---
            pnn_model = PNN(
                method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200,type_weight='unweighted'),
                random_state=42,         # or pass `seed` if needed
                high_score_anomaly=True
            )
            norm_emb = F.normalize(embeddings, dim=1)
            features_np = norm_emb.detach().cpu().numpy()
            y_labels = data.train_mask.detach().cpu().numpy().astype(int)

            # Fit PNN on the labeled/unlabeled data
            pnn_model.fit(features_np, y_labels)
            predicted = pnn_model.predict(features_np)
            proba = pnn_model.predict_proba(features_np)
            predicted_t = torch.from_numpy(predicted).to(embeddings.device)
    elif two_step == "Spy" and output_dim > 2:
        model.eval()
            embeddings = model(data.x, A_hat)

            pnn_model = SpyEM(spy_ratio= 0.1, threshold= 0.15, resampler = True, random_state=42)
            norm_emb = F.normalize(embeddings, dim=1)
            features_np = norm_emb.detach().cpu().numpy()
            y_labels = data.train_mask.detach().cpu().numpy().astype(int)

            # Fit PNN on the labeled/unlabeled data
            pnn_model.fit(features_np, y_labels)
            predicted = pnn_model.predict(features_np)
            proba = pnn_model.predict_proba(features_np)
            predicted_t = torch.from_numpy(predicted).to(embeddings.device)

    # --------------------------------------------------
    # 5) Evaluate
    # --------------------------------------------------
    metrics = evaluate_model(model, data, device, model_type)
    print(f"\n{model_type} model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # If you want to track or return something:
    return model, metrics, losses
