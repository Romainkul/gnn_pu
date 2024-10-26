import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load the Cora dataset and convert to binary
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())

# Get the first graph in the dataset
data = dataset[0]

# Ensure the graph is undirected
data.edge_index = to_undirected(data.edge_index)

# Convert the Cora dataset to binary classification:
# Let's make class '0' the positive class, and all other classes are unlabeled
data.y = (data.y == 0).long()

# Masking: only a small subset of the positive class is labeled, the rest are unlabeled
labeled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
unlabeled_mask = torch.ones(data.num_nodes, dtype=torch.bool)

# Let's say 20 nodes are labeled (we'll select 10 positive and 10 unlabeled at random)
positive_indices = (data.y == 1).nonzero(as_tuple=True)[0]
unlabeled_indices = (data.y == 0).nonzero(as_tuple=True)[0]

# Randomly select 10 labeled positives and 10 labeled unlabeled
num_labeled_positives = 10
num_labeled_unlabeled = 10

labeled_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:num_labeled_positives]]
labeled_unlabeled_indices = unlabeled_indices[torch.randperm(len(unlabeled_indices))[:num_labeled_unlabeled]]

# Update the labeled_mask and unlabeled_mask
labeled_mask[labeled_positive_indices] = True
labeled_mask[labeled_unlabeled_indices] = True
unlabeled_mask[labeled_positive_indices] = False
unlabeled_mask[labeled_unlabeled_indices] = False

# Add these masks to the data object
data.labeled_mask = labeled_mask
data.unlabeled_mask = unlabeled_mask

# Check the modified data
print(f'Number of labeled nodes: {labeled_mask.sum().item()}')
print(f'Number of unlabeled nodes: {unlabeled_mask.sum().item()}')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Distance-aware PU Loss function
def distance_aware_pu_loss(output, labels, dist_matrix, labeled_mask, unlabeled_mask, delta, pi_p_hat, pi_p_tilde):
    # Split unlabeled nodes into VˆU and V̆U based on distance threshold
    dist_to_positive = torch.min(dist_matrix[labeled_mask], dim=1).values
    close_unlabeled_mask = (dist_to_positive <= delta) & unlabeled_mask
    far_unlabeled_mask = ~close_unlabeled_mask & unlabeled_mask
    
    # Calculate number of nodes in each set
    nL = labeled_mask.sum().item()
    nU_hat = close_unlabeled_mask.sum().item()
    nU_tilde = far_unlabeled_mask.sum().item()

    # Calculate loss for labeled, close unlabeled, and far unlabeled nodes
    L_pos = (output[labeled_mask] - 1).pow(2).mean()
    L_U_hat = (output[close_unlabeled_mask] - pi_p_hat).pow(2).mean() if nU_hat > 0 else 0
    L_U_tilde = (output[far_unlabeled_mask] - pi_p_tilde).pow(2).mean() if nU_tilde > 0 else 0

    # Total loss combining the labeled and unlabeled losses
    return 2 * (pi_p_hat + pi_p_tilde) * (L_pos / nL + L_U_hat / nU_hat + L_U_tilde / nU_tilde)

# Structural regularization function
def structural_regularization(output, edge_index, K=5):
    num_nodes = output.size(0)
    # Similarity between node pairs
    similarity = torch.sigmoid(torch.matmul(output, output.T))  # S_ij = σ(z_i^T z_j)

    # First term: similarity for neighboring nodes (should be close to 1)
    src, dst = edge_index
    neighbor_loss = (similarity[src, dst] - 1).pow(2).sum()

    # Second term: similarity for K non-neighbors (should be close to 0)
    non_neighbors_loss = 0
    for vi in range(num_nodes):
        non_neighbors = torch.randint(0, num_nodes, (K,))  # Random K non-neighbor nodes
        non_neighbors_loss += (similarity[vi, non_neighbors] - 0).pow(2).sum()

    return neighbor_loss + non_neighbors_loss

# Combined loss function: L = L_G + α * R_S
def combined_loss(output, labels, dist_matrix, edge_index, labeled_mask, unlabeled_mask, delta, pi_p_hat, pi_p_tilde, alpha):
    # Compute distance-aware PU loss
    lg_loss = distance_aware_pu_loss(output, labels, dist_matrix, labeled_mask, unlabeled_mask, delta, pi_p_hat, pi_p_tilde)
    
    # Compute structural regularization loss
    rs_loss = structural_regularization(output, edge_index)
    
    # Combine the two losses
    return lg_loss + alpha * rs_loss

# Training function
def train(model, data, optimizer, delta=2, pi_p_hat=0.6, pi_p_tilde=0.3, alpha=0.5, num_epochs=100):
    dist_matrix = shortest_path(data.edge_index, num_nodes=data.num_nodes)  # Precompute distances between nodes
    #close=all_pairs_shortest_path(G, treshold=3)


    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        # Compute combined loss (L_G + α * R_S)
        loss = combined_loss(out, data.y, dist_matrix, data.edge_index, data.labeled_mask, data.unlabeled_mask, delta, pi_p_hat, pi_p_tilde, alpha)

        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Example usage
if __name__ == "__main__":
    # Assume 'data' is a preprocessed graph data object with necessary masks (labeled/unlabeled)
    model = GCN(in_channels=data.num_features, hidden_channels=64, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train(model, data, optimizer)
