from torch_geometric.nn.models import DeepGraphInfomax
import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

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

class LabelPropagationLoss(torch.nn.Module):
    def __init__(self, num_nodes, num_propagation_steps=10, alpha=0.5, learning_rate=0.01):
        """
        Initializes the Label Propagation Loss for reducing heterophily in graph PU learning.
        
        Parameters:
        - num_nodes: Total number of nodes in the graph.
        - num_propagation_steps: Number of label propagation steps to perform (iterations).
        - alpha: Propagation parameter controlling self-retention of labels (0 < alpha < 1).
        - learning_rate: Step size for updating adjacency weights to reduce heterophily.
        """
        super(LabelPropagationLoss, self).__init__()
        self.num_propagation_steps = num_propagation_steps
        self.alpha = alpha
        self.learning_rate = learning_rate
        
        # Initialize a learnable mask matrix M representing edge weights (homophily-focused)
        self.M = nn.Parameter(torch.ones((num_nodes, num_nodes)), requires_grad=True)

    def forward(self, adj_matrix, positive_nodes):
        """
        Applies label propagation on the adjacency matrix while reducing heterophilic influence.

        Parameters:
        - adj_matrix: The adjacency matrix of the graph.
        - positive_nodes: Binary tensor indicating positive nodes (1 for positive nodes, 0 otherwise).

        Returns:
        - adj_matrix: Updated adjacency matrix with reduced heterophily.
        """
        # Normalize adjacency matrix with degree matrix (for diffusion propagation)
        degree_matrix = torch.diag(torch.pow(adj_matrix.sum(1), -0.5))
        diffusion_matrix = degree_matrix @ adj_matrix @ degree_matrix

        # Initialize class-posterior probabilities, E(0)
        class_posterior = torch.zeros((adj_matrix.size(0), 2), device=adj_matrix.device)
        class_posterior[:, 0] = 1 - positive_nodes  # Initialize as negative for unlabeled nodes
        class_posterior[:, 1] = positive_nodes      # Initialize as positive for observed positives

        # Perform label propagation over multiple iterations
        for _ in range(self.num_propagation_steps):
            class_posterior = self.alpha * diffusion_matrix @ class_posterior + (1 - self.alpha) * class_posterior

        # Calculate propagation loss to minimize misclassification of positive nodes
        propagated_neg_probs = class_posterior[positive_nodes == 1, 0]  # Negative probabilities for positive nodes
        loss = -torch.log(1 - propagated_neg_probs + 1e-10).mean()  # Avoid log(0) with a small epsilon

        # Backpropagate to adjust mask matrix M (edge weights)
        loss.backward()
        with torch.no_grad():
            self.M.data -= self.learning_rate * self.M.grad  # Update M to refine edge weights
            self.M.grad.zero_()

        # Update adjacency matrix with optimized mask matrix M for reduced heterophily
        adj_matrix = self.M * adj_matrix  # Element-wise multiplication
        return adj_matrix