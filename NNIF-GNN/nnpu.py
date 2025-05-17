import torch
import torch.nn.functional as F
import copy
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_geometric.loader import NeighborLoader
from torch.nn import BCEWithLogitsLoss

class PULoss(nn.Module):
    """
    Implements the PU loss from:
      Ryuichi Kiryo, Gang Niu, Marthinus du Plessis, Masashi Sugiyama
      "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
       in NeurIPS 2017.

    * Single-logit binary classification:
        - x >  0  => more likely to be positive
        - x <= 0  => more likely to be negative

    * Non-negative correction (nnPU):
      If the estimated negative risk is < -beta, clamp objective to (positive_risk - beta).

    * Imbalanced weighting (if imbpu=True):
      Weighted combination of positive_risk and negative_risk 
      using alpha / prior, (1-alpha)/(1-prior).

    Args:
        prior (float):
            The class prior p(Pos) in unlabeled data. Must be in (0,1).
        gamma (float):
            Scales leftover negative risk if negative_risk < -beta.
            (Kiryo’s code calls this 'self.gamma', used in the partial backprop.)
        beta (float):
            Clipping threshold for negative risk under nnPU.
        nnpu (bool):
            If True, use non-negative correction; if False, unbiased PU.
        imbpu (bool):
            If True, apply imbalance weighting (ImbPU).
        alpha (float):
            Mixing parameter for ImbPU. alpha=0.5 => balanced weighting.
    """

    def __init__(self, prior, gamma=1.0, beta=0.0, nnpu=True, imbpu=False, alpha=0.5):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("KiryoPULoss: 'prior' must be in (0,1).")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnpu = nnpu
        self.imbpu = imbpu
        self.alpha = alpha

    def forward(self, x, t):
        """
        x: torch.Tensor of shape [N] or [N,1]
           Single-logit scores. If shape=[N,1], we squeeze it to [N].
        t: torch.Tensor of shape [N]
           Labels in {+1 => known positive, 0 => unlabeled}.

        Returns:
          A scalar torch.Tensor representing the PU loss.
        """
        if x.dim() == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)  # shape => [N]

        positive_mask = (t == 1).float()
        unlabeled_mask = (t == 0).float()

        n_pos = positive_mask.sum().clamp_min(1.0)
        n_unl = unlabeled_mask.sum().clamp_min(1.0)

        y_positive = torch.sigmoid(-x)  # shape [N]
        y_unlabeled = torch.sigmoid(x)  # shape [N]

        positive_risk = (
            self.prior *
            positive_mask / n_pos *
            y_positive
        ).sum()

        negative_risk = (
            (unlabeled_mask / n_unl) -
            (self.prior * positive_mask / n_pos)
        ) * y_unlabeled
        negative_risk = negative_risk.sum()

        objective = positive_risk + negative_risk

        if self.nnpu:
            if negative_risk.item() < -self.beta:
                objective = positive_risk - self.beta
                
        if self.imbpu:
            w_pos = self.alpha / self.prior
            w_neg = (1.0 - self.alpha) / (1.0 - self.prior)
            objective = w_pos * positive_risk + w_neg * negative_risk

        return objective

###############################################################################
# Training Loop
###############################################################################
def train_nnpu(
    model,
    data,
    device,
    model_type='SAGEConv',
    layers=2,
    batch_size=1024,
    lr=0.005,
    weight_decay=5e-4,
    max_epochs=100,
    # PU Loss params
    prior=0.5,
    gamma=1,
    beta=0,
    nnpu=True,
    imbpu=False,
    alpha=0.5
):
    """
    Example training loop for PU learning on graph data using PULoss.
    Assumes binary classification with labels in {+1, -1}.

    Args:
        model : nn.Module
            A PyTorch model (e.g., GNN or MLP).
        data : torch_geometric.data.Data
            Contains x, y, edge_index, etc. data.y in {+1, -1}.
        device : torch.device
            GPU or CPU.
        method : str
            GNN method name; if 'SAGEConv', neighbor sampling is used.
        layers : int
            Number of GNN layers (not directly used here, but you can pass as needed).
        batch_size : int
            For mini-batch neighbor sampling.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization.
        max_epochs : int
            Number of epochs to train.
        prior, gamma, beta, nnpu, imbpu, alpha : see PULoss docstring.

    Returns:
        losses_per_epoch : list
            Training loss values over epochs.
        proba : torch.Tensor
            Predicted probabilities for each node in data.x.
        pred_y : torch.Tensor
            Predicted labels (0 or 1) for each node in data.x.
    """
    # Copy data to avoid side-effects
    data = copy.copy(data)
    data = data.to(device)
    data.n_id = torch.arange(data.num_nodes, device=device)

    # Instantiate the PU loss
    if nnpu:
        pu_criterion = PULoss(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu, imbpu=imbpu, alpha=alpha).to(device)
    else:
        pu_criterion = BCEWithLogitsLoss().to(device)

    if model_type == "SAGEConv":
        train_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=batch_size,
            shuffle=True
        )
    else:
        train_loader = None

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    losses_per_epoch = []

    ##################################################################
    # Training loop
    ##################################################################
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        if train_loader is not None:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(batch.x, batch.edge_index)
                    labels = batch.train_mask.float()
                    if not nnpu:
                        loss = pu_criterion(logits, labels.unsqueeze(-1))
                    else:
                        loss = pu_criterion(logits.squeeze(-1), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            losses_per_epoch.append(avg_loss)
            print(f"Epoch [{epoch+1}/{max_epochs}] - Loss: {avg_loss:.4f}")

        else:
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(data.x, data.edge_index)
                labels = data.train_mask.float()
                if not nnpu:
                    loss = pu_criterion(logits, labels.unsqueeze(-1))
                else:
                    loss = pu_criterion(logits.squeeze(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_per_epoch.append(loss.item())
            print(f"Epoch [{epoch+1}/{max_epochs}] - Loss: {loss.item():.4f}")

    ##################################################################
    # Post-training inference
    ##################################################################
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        # Sigmoid for probabilities in [0,1]
        proba = torch.sigmoid(logits)

        # Convert from [0,1] to {0,1} threshold 0.5
        pred_y = (proba > 0.5).long()
    combined_mask = (data.train_mask | data.test_mask | data.val_mask)
    return pred_y[combined_mask].cpu(), proba[combined_mask].cpu(), losses_per_epoch
