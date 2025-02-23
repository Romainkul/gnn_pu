import torch
from torch_sparse import SparseTensor
from typing import List, Union, Tuple

import torch.nn as nn
import torch.nn.functional as F

class LabelPropagationLoss(nn.Module):
    """
    Implements K-step label propagation on a given adjacency matrix, followed 
    by a negative log-likelihood loss on positive and negative sets.

    Args:
        A_hat (SparseTensor):
            Sparse adjacency matrix with self-loops included.
        alpha (float):
            Weight balancing old distribution vs. neighbor distribution in each step.
        K (int):
            Number of propagation steps.
        pos_weight (float):
            Weight assigned to the positive loss term.
        init_temperature (float):
            Initial value for the temperature parameter (learnable).

    Shape:
        - The adjacency matrix A_hat should be of shape [N, N] (stored as SparseTensor).
        - The input embeddings should be of shape [N, d].

    Example:
        >>> criterion = LabelPropagationLoss(A_hat, alpha=0.5, K=10)
        >>> loss_val, updated_adj, E = criterion(node_embeddings, pos_indices, neg_indices)
    """

    def __init__(
        self,
        A_hat: SparseTensor,
        alpha: float = 0.5,
        K: int = 10,
        pos_weight: float = 1.0,
        init_temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.pos_weight = pos_weight

        # Temperature is a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

        # Buffers for adjacency sums and inverse degrees
        self.register_buffer('row_sums', A_hat.sum(dim=1))
        self.A_hat = A_hat
        self.register_buffer('d_inv', 1.0 / torch.clamp(self.row_sums, min=1e-12))

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_nodes: Union[List[int], torch.LongTensor],
        negative_nodes: Union[List[int], torch.LongTensor]
    ) -> Tuple[torch.Tensor, SparseTensor, torch.Tensor]:
        """
        Forward pass for label propagation and loss computation.

        Args:
            embeddings (torch.Tensor):
                Node embeddings of shape [N, d].
            positive_nodes (Union[List[int], torch.LongTensor]):
                Indices of nodes considered positive.
            negative_nodes (Union[List[int], torch.LongTensor]):
                Indices of nodes considered negative.

        Returns:
            (loss, updated_A_hat, E):
                - loss (torch.Tensor): The scalar loss value.
                - updated_A_hat (SparseTensor): Potentially updated adjacency (same here).
                - E (torch.Tensor): Soft assignment matrix of shape [N, 2], where 
                  E[i] = (prob_neg, prob_pos) for node i.
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Initialize the label distribution E:
        # - Column 0: probability of being negative
        # - Column 1: probability of being positive
        E = torch.zeros((N, 2), device=device)
        E[positive_nodes, 1] = 1.0
        E[negative_nodes, 0] = 1.0

        # Perform K-step label propagation
        for _ in range(self.K):
            # Multiply adjacency by E and row-normalize
            neighbor_E = self.A_hat.matmul(E)               # [N, 2]
            neighbor_E = self.d_inv.view(-1, 1) * neighbor_E

            # Update E: alpha * old + (1 - alpha) * neighbor
            E = self.alpha * E + (1.0 - self.alpha) * neighbor_E

            # Softmax with learnable temperature
            E = F.softmax(E / self.temperature, dim=1)

        # Compute loss for positive and negative sets
        # Clamping to avoid log(0)
        pos_probs = torch.clamp(E[positive_nodes, 1], min=1e-6)
        neg_probs = torch.clamp(E[negative_nodes, 0], min=1e-6)

        pos_loss = -torch.mean(torch.log(pos_probs))
        neg_loss = -torch.mean(torch.log(neg_probs))
        total_loss = self.pos_weight * pos_loss + neg_loss

        updated_A_hat = self.A_hat

        return total_loss, updated_A_hat, E

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

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.02,
        steps_per_epoch=1,
        epochs=num_epochs
    )
    early_stopping = EarlyStopping_GNN(patience=20)
    scaler = GradScaler()

    # Step 1: Build adjacency with self-loops
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    edge_index = coalesce(edge_index)
    A_hat = SparseTensor.from_edge_index(edge_index).coalesce().to(device)

    # Tracking variables
    train_losses = []
    best_loss = float('inf')

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):

        model.train()
        optimizer.zero_grad()

        with autocast():
            # 2) Forward pass with current adjacency
            embeddings = model(data.x.to(device), A_hat)

            # Anomaly detection => reliable positives/negatives
            NNIF = ReliableValues(
                method=treatment,
                treatment_ratio=ratio,
                anomaly_detector=WeightedIsoForest(n_estimators=200),
                random_state=42,
                high_score_anomaly=True,
            )
            norm_emb = F.normalize(embeddings, dim=1)
            features_np = norm_emb.detach().cpu().numpy()
            y_labels = data.train_mask.detach().cpu().numpy().astype(int)
            reliable_negatives, reliable_positives = NNIF.get_reliable(features_np, y_labels)

            # 3) Label Propagation
            lp_criterion = LabelPropagationLoss(
                A_hat=A_hat,
                alpha=alpha,
                K=K,
                init_temperature=1.0
            ).to(device)
            lpl_loss, updated_A_hat, E = lp_criterion(embeddings, reliable_positives, reliable_negatives)

            nnPU_loss = pu_loss(embeddings, data.y, prior=0.5, gamma=1, beta=0, nnpu=True)
            # 5) Combine losses
            loss = nnPU_loss

        # Backprop
        scaler.scale(loss).backward()

        # Gradient clipping
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

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
            )
            logger.info(
                f"Epoch {epoch}, Loss: {loss_val:.4f}, "
            )

        # Early stopping check
        if early_stopping(loss_val):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    return train_losses, A_hat