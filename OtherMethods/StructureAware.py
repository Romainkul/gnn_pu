import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.utils import to_undirected, k_hop_subgraph, shortest_path
from torch_sparse import SparseTensor
import numpy as np
from encoders import BaseEncoder
from data_generating import load_dataset, make_pu_dataset

class DistanceAwarePULoss(nn.Module):
    def __init__(self, prior_positive=0.5, delta=2):
        super(DistanceAwarePULoss, self).__init__()
        self.prior_positive = prior_positive
        self.delta = delta  # Distance threshold for splitting unlabeled nodes

    def forward(self, y_pred, y_true, pos_idx, unlabeled_idx, edge_index):
        """
        Computes Distance-Aware PU Loss.

        Args:
            y_pred (tensor): Model predictions.
            y_true (tensor): True labels (1 for positive, -1 for unlabeled).
            pos_idx (tensor): Indices of positive-labeled nodes.
            unlabeled_idx (tensor): Indices of unlabeled nodes.
            edge_index (tensor): Graph connectivity.

        Returns:
            torch.Tensor: Computed loss.
        """
        n_L = len(pos_idx)  # Number of labeled positives
        n_U = len(unlabeled_idx)  # Number of unlabeled nodes

        # Compute shortest path distances to positive nodes
        dist = shortest_path(edge_index, pos_idx, y_pred.shape[0])  # Shortest path to any positive node

        # Split unlabeled nodes into two subsets: near & far
        near_unlabeled = unlabeled_idx[dist[unlabeled_idx] <= self.delta]
        far_unlabeled = unlabeled_idx[dist[unlabeled_idx] > self.delta]

        n_nearU = len(near_unlabeled)
        n_farU = len(far_unlabeled)

        # Priors (π^ and πˇ with π^ > πˇ)
        pi_hat = self.prior_positive * 1.2
        pi_check = self.prior_positive * 0.8

        # Compute loss components
        loss_L = (1 / n_L) * torch.sum(torch.abs(y_pred[pos_idx] - 1))  # Loss for labeled positives
        loss_nearU = (1 / max(n_nearU, 1)) * torch.sum(torch.abs(y_pred[near_unlabeled] - pi_hat))
        loss_farU = (1 / max(n_farU, 1)) * torch.sum(torch.abs(y_pred[far_unlabeled] - pi_check))

        return 2 * (pi_hat + pi_check) * loss_L + loss_nearU + loss_farU


# ===========================
# 🔹 Structural Regularization
# ===========================

class StructuralRegularization(nn.Module):
    def __init__(self, k=5):
        super(StructuralRegularization, self).__init__()
        self.k = k  # Number of negative samples per node

    def forward(self, embeddings, edge_index):
        """
        Compute structural regularization loss.

        Args:
            embeddings (tensor): Node representations.
            edge_index (tensor): Graph connectivity.

        Returns:
            torch.Tensor: Structural loss.
        """
        loss = 0
        num_nodes = embeddings.size(0)

        # Compute similarity matrix (cosine similarity)
        similarity = torch.mm(embeddings, embeddings.T)
        similarity = torch.sigmoid(similarity)

        # Compute loss for each node
        for i in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == i]  # Get neighbors
            if len(neighbors) == 0:
                continue

            non_neighbors = torch.tensor(list(set(range(num_nodes)) - set(neighbors.tolist())), dtype=torch.long)
            neg_samples = non_neighbors[torch.randperm(len(non_neighbors))[:self.k]]  # Sample negative nodes

            # Structural loss computation
            loss += torch.sum((similarity[i, neighbors] - 1) ** 2)  # Encourage similarity between neighbors
            loss += torch.sum((similarity[i, neg_samples] - 0) ** 2)  # Encourage dissimilarity for non-neighbors

        return loss / num_nodes


# ===========================
# 🔹 Training Script
# ===========================

def train_pu_gnn(data, model, pu_loss, struct_reg, optimizer, device, num_epochs=50, alpha=0.1):
    """
    Train the PU-GNN model.

    Args:
        data (Data): Graph data object.
        model (PUGNN): The GNN model.
        pu_loss (DistanceAwarePULoss): PU loss function.
        struct_reg (StructuralRegularization): Regularization loss.
        optimizer (Optimizer): Optimizer.
        device (torch.device): CPU or GPU.
        num_epochs (int): Number of training epochs.
        alpha (float): Weight for structural loss.

    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))

        loss_pu = pu_loss(out.squeeze(), data.y.to(device), data.pos_idx.to(device), data.unlabeled_idx.to(device), data.edge_index.to(device))
        loss_reg = struct_reg(out, data.edge_index.to(device))

        loss = loss_pu + alpha * loss_reg  # Final loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, PU Loss: {loss_pu.item():.4f}, Reg Loss: {loss_reg.item():.4f}")

    print("Training Complete!")

# Example Usage:
# model = BaseEncoder(data.num_nodes, hidden_dim, output_dim, num_layers, dropout, model_type=model_type).to(device)
# pu_loss = DistanceAwarePULoss()
# struct_reg = StructuralRegularization()
# optimizer = optim.AdamW(model.parameters(), lr=0.01)
# train_pu_gnn(data, model, pu_loss, struct_reg, optimizer, device)
