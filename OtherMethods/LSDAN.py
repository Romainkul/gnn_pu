import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from nnPU import PULoss
from .NNIF-GNN.data_generating import load_dataset, make_pu_dataset
from torch_sparse import SparseTensor

class ShortDistanceAttention(MessagePassing):
    """
    Short-distance self-attention using a sparse adjacency matrix.
    """
    def __init__(self, in_dim, out_dim):
        super(ShortDistanceAttention, self).__init__(aggr='add')
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(2 * out_dim, 1)

    def forward(self, x, edge_index):
        x = self.W(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        attn_score = F.leaky_relu(self.att(torch.cat([x_i, x_j], dim=-1)), negative_slope=0.2)
        attn_score = torch.exp(attn_score)
        return attn_score * x_j

class LongDistanceAttention(nn.Module):
    """
    Long-distance attention using sparse adjacency matrices.
    """
    def __init__(self, in_dim, out_dim, num_hops):
        super(LongDistanceAttention, self).__init__()
        self.num_hops = num_hops
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(2 * out_dim, 1)

    def forward(self, x, adj_matrices):
        hop_embeddings = []
        for k in range(self.num_hops):
            # Efficient sparse matrix multiplication
            hk = F.leaky_relu(self.W(adj_matrices[k].matmul(x)))
            hop_embeddings.append(hk)

        # Compute attention weights over different hop embeddings
        hop_embeddings = torch.stack(hop_embeddings, dim=1)
        attn_weights = F.softmax(self.att(hop_embeddings), dim=1)
        output = torch.sum(attn_weights * hop_embeddings, dim=1)
        return output

class LSDANLayer(nn.Module):
    """
    A single LSDAN layer integrating short- and long-distance attention.
    """
    def __init__(self, in_dim, out_dim, num_hops):
        super(LSDANLayer, self).__init__()
        self.short_att = ShortDistanceAttention(in_dim, out_dim)
        self.long_att = LongDistanceAttention(in_dim, out_dim, num_hops)

    def forward(self, x, edge_index, adj_matrices):
        short_emb = self.short_att(x, edge_index)
        long_emb = self.long_att(x, adj_matrices)
        return short_emb + long_emb  # Combine both embeddings

class LSDAN(nn.Module):
    """
    Multi-layer LSDAN model for PU graph learning.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_hops):
        super(LSDAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSDANLayer(in_dim, hidden_dim, num_hops))
        for _ in range(num_layers - 1):
            self.layers.append(LSDANLayer(hidden_dim, hidden_dim, num_hops))
        self.classifier = nn.Linear(hidden_dim, out_dim)  # Final classification

    def forward(self, x, edge_index, adj_matrices):
        for layer in self.layers:
            x = layer(x, edge_index, adj_matrices)
        return self.classifier(x)

# Hyperparameters
hidden_dim = 64
output_dim = 2  # Binary classification
num_layers = 3
dropout = 0.5
num_hops = 3
num_epochs = 50
learning_rate = 0.01
weight_decay = 5e-4

# Dataset & Configuration
dataset_name = 'citeseer'
mechanism = 'SCAR'
seed = 1
train_pct = 0.5

# Load dataset and create PU labels
data = load_dataset(dataset_name)
data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)

# Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# Convert edge_index to a SparseTensor for efficiency
adj_matrix = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)

# Compute multi-hop adjacency matrices using sparse multiplications
adj_matrices = [adj_matrix]
for _ in range(1, num_hops):
    adj_matrix = adj_matrix @ adj_matrix  # Efficient k-hop adjacency computation
    adj_matrices.append(adj_matrix)

# Initialize Model, Loss, Optimizer
model = LSDAN(data.num_features, hidden_dim, output_dim, num_layers, num_hops).to(device)
pu_loss = PULoss(prior=0.5).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index, adj_matrices).squeeze()

    # Compute PU Loss
    loss = pu_loss(out, data.y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Logging
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training Complete!")
