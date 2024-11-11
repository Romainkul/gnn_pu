from torch_geometric.nn.models import DeepGraphInfomax
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#Can adapt with another gnn model
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class ContrastiveLossWithDiffusion(torch.nn.Module):
    def __init__(self, temperature=0.5, diffusion_steps=10):
        super(ContrastiveLossWithDiffusion, self).__init__()
        self.temperature = temperature
        self.diffusion_steps = diffusion_steps

    def forward(self, embeddings, edge_index):
        # Diffusion-based neighborhood similarity (adjacency matrix to propagation matrix)
        adj_matrix = torch.zeros((embeddings.size(0), embeddings.size(0)), device=embeddings.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        degree = torch.diag(torch.pow(adj_matrix.sum(1), -0.5))
        diffusion_matrix = degree @ adj_matrix @ degree

        # Perform diffusion propagation
        for _ in range(self.diffusion_steps):
            embeddings = diffusion_matrix @ embeddings
        
        # Contrastive Loss Computation
        pos_pairs = torch.einsum('nc,nc->n', embeddings[edge_index[0]], embeddings[edge_index[1]]) / self.temperature
        pos_loss = -F.logsigmoid(pos_pairs).mean()

        # Negative samples (random pairings)
        rand_indices = torch.randperm(embeddings.size(0))
        neg_pairs = torch.einsum('nc,nc->n', embeddings, embeddings[rand_indices]) / self.temperature
        neg_loss = -F.logsigmoid(-neg_pairs).mean()

        return pos_loss + neg_loss

class DeepGraphInfomaxLoss(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(DeepGraphInfomaxLoss, self).__init__()
        self.encoder = GNNEncoder(hidden_dim)
        self.model = DeepGraphInfomax(
            hidden_channels=hidden_dim,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=self.corruption_function
        )

    def corruption_function(self, x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index  # Shuffle node features as corruption

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.model(x, edge_index)
        dgi_loss = self.model.loss(pos_z, neg_z, summary)
        return dgi_loss


class DiffusionPseudoLabelingLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, num_iterations=10, threshold=0.5):
        super(DiffusionPseudoLabelingLoss, self).__init__()
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.threshold = threshold

    def forward(self, embeddings, edge_index, labels=None):
        # Initialize label matrix (binary labels: positive = 1, negative = 0)
        label_matrix = torch.zeros((embeddings.size(0),), device=embeddings.device)
        if labels is not None:
            label_matrix[labels] = 1

        # Diffusion for pseudo-labeling
        adj_matrix = torch.zeros((embeddings.size(0), embeddings.size(0)), device=embeddings.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        degree = torch.diag(torch.pow(adj_matrix.sum(1), -0.5))
        diffusion_matrix = degree @ adj_matrix @ degree

        for _ in range(self.num_iterations):
            label_matrix = self.alpha * diffusion_matrix @ label_matrix + (1 - self.alpha) * label_matrix

        # Convert to binary pseudo-labels with threshold
        pseudo_labels = (label_matrix >= self.threshold).float()
        
        # Apply Cross-Entropy Loss using pseudo-labels
        logits = embeddings  # assuming the output of encoder is logits for binary classification
        pseudo_loss = F.binary_cross_entropy_with_logits(logits, pseudo_labels)
        
        return pseudo_loss


class NeighborSimilarityLoss(torch.nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(NeighborSimilarityLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, embeddings, edge_index):
        # Calculate regularization loss between neighbors
        src, dst = edge_index
        similarity_loss = F.mse_loss(embeddings[src], embeddings[dst])
        return self.lambda_reg * similarity_loss
