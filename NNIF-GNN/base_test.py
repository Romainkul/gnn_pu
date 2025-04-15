import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Classical ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# PyG
from torch_geometric.data import Data

# Custom modules (assumed to exist in your environment)
from encoder import GraphEncoder
from nnPU import PULoss  # Not used if a two-step approach is selected
from data_generating import load_dataset, make_pu_dataset
from NNIF import PNN, ReliableValues, WeightedIsoForest, SpyEM
from TED import run_tedn_training

###############################################################################
# EarlyStopping_GNN
###############################################################################
class EarlyStopping_GNN:
    """
    Implements an early-stopping mechanism for GNN or neural training.
    Stop conditions:
      - The absolute difference in consecutive epoch losses < 'loss_diff_threshold', OR
      - The current loss is worse (greater) than the best so far.
    If either is triggered for 'patience' consecutive epochs, training is flagged to stop.
    """
    def __init__(self, patience=50, delta=0.0, loss_diff_threshold=5e-4):
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

        # Condition: if loss hasn't improved much or is worse than best so far
        if (loss_diff < self.loss_diff_threshold) or (loss > self.best_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        self.previous_loss = loss

        # If there's a sufficiently large improvement, reset
        if loss < (self.best_loss - self.delta):
            self.best_loss = loss
            self.counter = 0

        return self.early_stop


###############################################################################
# train_model
###############################################################################
from torch_geometric.loader import NeighborLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def train_model(
    model,
    data: Data,
    device: torch.device,
    model_type: str,
    num_epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    use_pu: bool = False,
    prior: float = 0.5,
    use_tedn: bool = False,
    tedn_kwargs: dict = None,
    two_step: str = None,
    ratio: float = 0.1
):
    if model_type == "RandomForest":
        model = RandomForestClassifier()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    elif model_type == "XGBoost":
        model = XGBClassifier()
    else:
        # e.g., GraphSAGE, GIN, GCN, etc.
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

    if model_type in ["RandomForest", "LogisticRegression", "XGBoost"]:
        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None:
            train_idx = train_mask.nonzero(as_tuple=True)[0]
        else:
            train_idx = torch.arange(data.num_nodes)

        X_train = data.x[train_idx].cpu().numpy()
        y_train = data.y[train_idx].cpu().numpy()

        model.fit(X_train, y_train)
        return [], None, None

    # Neural model setup
    model = model.to(device)
    data = data.to(device)

    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    else:
        train_idx = torch.arange(data.num_nodes, device=device)

    if two_step in ["NNIF", "Spy"]:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = PULoss(prior=prior, gamma=1, beta=0, nnpu=True, imbpu=False).to(device) if use_pu else nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    early_stopper = EarlyStopping_GNN(patience=20)
    losses_per_epoch = []
    max_grad_norm = 1.0

    if use_tedn:
        if tedn_kwargs is None:
            tedn_kwargs = {}
        run_tedn_training(
            net=model,
            device=device,
            optimizer=optimizer,
            criterion=loss_fn,
            **tedn_kwargs
        )
        return [], None, None

    if model_type.lower() == "sageconv":
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[10, 10],
            batch_size=1024,
            shuffle=True
        )
    else:
        train_loader = None

    if two_step in ["NNIF", "IF", "Spy"]:
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index)
        emb_np = F.normalize(emb, dim=1).cpu().numpy()
        y_np = sub_data.train_mask.detach().cpu().numpy().astype(int)

        if two_step == "NNIF":
            nnif_model = ReliableValues(
                method="removal",
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,
                high_score_anomaly=True
            )
            nnif_model.fit(emb_np, y_np)

        elif two_step == "IF":
            if_model = ReliableValues(
                method="removal",
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200, type_weight='unweighted'),
                random_state=42,
                high_score_anomaly=True
            )
            if_model.fit(emb_np, y_np)

        elif two_step == "Spy":
            spy_detector = SpyEM(spy_ratio=0.1, threshold=0.15, resampler=False, random_state=42)
            neg_mask, pos_mask = spy_detector.get_reliable(emb_np, y_np)

        model.train()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        if train_loader is not None:
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                with autocast(enabled=torch.cuda.is_available()):
                    out = model(batch.x, batch.edge_index)
                    if out.dim() > 1 and out.size(-1) == 1:
                        out = out.squeeze(-1)
                    loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            losses_per_epoch.append(total_loss / len(train_loader))

        else:
            with autocast(enabled=torch.cuda.is_available()):
                out = model(data.x, data.edge_index)
                if out.dim() > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                loss = loss_fn(out[train_idx], data.y[train_idx])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            curr_loss = loss.item()
            losses_per_epoch.append(curr_loss)

        if early_stopper(losses_per_epoch[-1]):
            print(f"[EarlyStopping] Stopped at epoch {epoch+1} / {num_epochs}")
            break

    # Evaluation: Predict probabilities and labels
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if logits.dim() > 1 and logits.size(-1) > 1:
            proba = F.softmax(logits, dim=-1)
            pred_y = proba.argmax(dim=-1)
        else:
            proba = torch.sigmoid(logits)
            pred_y = (proba > 0.5).long()

    return losses_per_epoch, proba.detach().cpu(), pred_y.detach().cpu()


###############################################################################
# evaluate_model
###############################################################################
def evaluate_model(model, data: Data, device: torch.device, model_type: str):
    """
    Evaluates the model on the entire dataset or a mask if desired,
    returning {precision, recall, f1_score, avg_precision}.
    """
    data = data.to(device)

    # Classical
    if model_type in ["RandomForest", "LogisticRegression", "XGBoost"]:
        X_all = data.x.cpu().numpy()
        y_all = data.y.cpu().numpy()
        y_pred = model.predict(X_all)
        y_proba = model.predict_proba(X_all)[:, 1]

        prec = precision_score(y_all, y_pred, pos_label=1)
        rec = recall_score(y_all, y_pred, pos_label=1)
        f1v = f1_score(y_all, y_pred, pos_label=1)
        ap = average_precision_score(y_all, y_proba)
        return {"precision": prec, "recall": rec, "f1_score": f1v, "avg_precision": ap}

    # Neural
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)  # [num_nodes, 2]
        preds = out.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

    y_all = data.y.cpu().numpy()
    prec = precision_score(y_all, preds, pos_label=1)
    rec = recall_score(y_all, preds, pos_label=1)
    f1v = f1_score(y_all, preds, pos_label=1)
    ap = average_precision_score(y_all, probs)
    return {"precision": prec, "recall": rec, "f1_score": f1v, "avg_precision": ap}


###############################################################################
# main
###############################################################################
def main():
    """
    Simple demonstration of:
      1) Load dataset
      2) Possibly create PU
      3) Initialize model (classical or GNN)
      4) Train (optionally with two-step or TED^n)
      5) Evaluate
    """
    dataset_name = 'citeseer'
    mechanism = 'SCAR'
    seed = 1
    train_pct = 0.5
    model_type = "GraphSAGE"
    use_pu = False
    use_tedn = False
    two_step = "NNIF"  # or "IF" or "Spy" or None

    # 1) Load & create data
    data = load_dataset(dataset_name)
    data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Construct a model
    if model_type == "RandomForest":
        model = RandomForestClassifier()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    elif model_type == "XGBoost":
        model = XGBClassifier()
    else:
        # e.g., GraphSAGE, GIN, GCN, etc.
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

    # 3) Train
    losses = train_model(
        model=model,
        data=data,
        device=device,
        model_type=model_type,
        num_epochs=50,
        lr=0.01,
        weight_decay=5e-4,
        use_pu=use_pu,          # Ignored if two_step is used
        prior=0.5,
        use_tedn=use_tedn,
        tedn_kwargs=None,       # if you'd like to pass args for run_tedn_training
        two_step=two_step,
        ratio=0.1
    )

    # 4) Evaluate
    metrics = evaluate_model(model, data, device, model_type)
    print(f"\n[{model_type}] Final Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics, losses