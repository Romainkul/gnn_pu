import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class EdgeAggregator(nn.Module):
    def __init__(self, node_dim, edge_dim, graph_dim):
        super(EdgeAggregator, self).__init__()
        self.linear = nn.Linear(node_dim * 2 + edge_dim + graph_dim, edge_dim)
        
    def forward(self, h_i, h_j, h_e, h_G):
        # Concatenate node, edge, and graph features
        edge_features = torch.cat([h_i, h_j, h_e, h_G], dim=-1)
        # Linear transformation followed by ReLU
        return F.relu(self.linear(edge_features))

class NodeAggregator(nn.Module):
    def __init__(self, edge_dim, node_dim, graph_dim):
        super(NodeAggregator, self).__init__()
        self.linear = nn.Linear(edge_dim + node_dim + graph_dim, node_dim)
        
    def forward(self, h_i, m_N, h_G):
        # Concatenate node, aggregated edge, and graph features
        node_features = torch.cat([h_i, m_N, h_G], dim=-1)
        return F.relu(self.linear(node_features))

class GraphAggregator(nn.Module):
    def __init__(self, node_dim, edge_dim, graph_dim):
        super(GraphAggregator, self).__init__()
        self.linear = nn.Linear(node_dim + edge_dim + graph_dim, graph_dim)
        
    def forward(self, h_G, h_nodes, h_edges):
        # Concatenate graph, node, and edge features
        graph_features = torch.cat([h_G, h_nodes, h_edges], dim=-1)
        return F.relu(self.linear(graph_features))

class EthereumGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, graph_dim):
        super(EthereumGraphSAGE, self).__init__()
        
        # Initialize layers
        self.node_embed = nn.Linear(in_channels, hidden_channels)
        self.graph_embed = nn.Linear(graph_dim, graph_dim)
        
        self.edge_aggregator = EdgeAggregator(hidden_channels, edge_dim, graph_dim)
        self.node_aggregator = NodeAggregator(edge_dim, hidden_channels, graph_dim)
        self.graph_aggregator = GraphAggregator(hidden_channels, edge_dim, graph_dim)
        
        self.sage_conv = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, data):
        # Initial node, edge, and graph embeddings
        h_i = self.node_embed(data.x)  # Node features
        h_G = self.graph_embed(data.graph_attr)  # Graph feature
        h_e = data.edge_attr  # Edge features
        
        # Update edge features
        row, col = data.edge_index
        h_edge_agg = self.edge_aggregator(h_i[row], h_i[col], h_e, h_G)
        
        # Aggregate edge features for each node
        m_N = torch.zeros_like(h_i)
        for node in range(data.num_nodes):
            neighbor_edges = h_edge_agg[data.edge_index[0] == node]
            m_N[node] = neighbor_edges.mean(dim=0)
        
        # Update node features
        h_i = self.node_aggregator(h_i, m_N, h_G)
        
        # Update graph feature by aggregating all node and edge features
        h_G = self.graph_aggregator(h_G, h_i.mean(dim=0), h_edge_agg.mean(dim=0))
        
        # Apply GraphSAGE convolution and pooling for final output
        h_nodes = self.sage_conv(h_i, data.edge_index)
        h_nodes = global_mean_pool(h_nodes, data.batch)  # Pooling for graph-level features
        
        return F.log_softmax(h_nodes, dim=1)

# Sample data structure (replace with actual Ethereum data loading and preprocessing)
node_features = torch.rand((100, 16))  # 100 nodes with 16 features each
edge_features = torch.rand((200, 8))   # 200 edges with 8 features each
graph_feature = torch.rand((1, 8))     # Global graph feature with 8 dimensions
edge_index = torch.randint(0, 100, (2, 200))  # Random edges between nodes

# Pack into a data object (use actual Ethereum transaction data here)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, graph_attr=graph_feature)

# Model parameters
in_channels = node_features.size(1)
hidden_channels = 32
out_channels = 4  # Number of classes for node classification
edge_dim = edge_features.size(1)
graph_dim = graph_feature.size(1)

# Initialize and forward through the model
model = EthereumGraphSAGE(in_channels, hidden_channels, out_channels, edge_dim, graph_dim)
output = model(data)
#X=output.detach().numpy()
# output contains the node classifications or graph representations
print(output)
"""

# Mock data - replace with actual address features and labels
n_samples = 1000
n_features = 20
X = np.random.rand(n_samples, n_features)  # Sample features for addresses
y = np.array([1] * 100 + [-1] * 900)  # 100 labeled positive instances, rest are unlabeled (-1)

# Step 1: Spy Sampling
def spy_sampling(X, y, spy_ratio=0.15):
    positives = X[y == 1]
    unlabeled = X[y == -1]
    
    # Randomly select spies from the positive set
    n_spies = int(spy_ratio * len(positives))
    spy_indices = np.random.choice(len(positives), n_spies, replace=False)
    spies = positives[spy_indices]
    
    # Remaining positives (Pr) and combined unlabeled + spies (Ur)
    Pr = np.delete(positives, spy_indices, axis=0)
    Ur = np.vstack((unlabeled, spies))
    
    return Pr, Ur, spies

# Apply spy sampling
Pr, Ur, spies = spy_sampling(X, y, spy_ratio=0.15)

# Step 2: Train Classifier on Pr and Ur
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze()

# Convert data to torch tensors
X_Pr = torch.tensor(Pr, dtype=torch.float32)
X_Ur = torch.tensor(Ur, dtype=torch.float32)
y_Pr = torch.ones(len(X_Pr), dtype=torch.float32)  # Pr are positives
y_Ur = torch.zeros(len(X_Ur), dtype=torch.float32)  # Ur are assumed negative initially

# Train simple binary classifier
classifier = SimpleClassifier(n_features)
optimizer = optim.Adam(classifier.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs_Pr = classifier(X_Pr)
    outputs_Ur = classifier(X_Ur)
    
    # Combine Pr (positive) and Ur (assumed negative) for training
    outputs = torch.cat((outputs_Pr, outputs_Ur))
    targets = torch.cat((y_Pr, y_Ur))
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Step 3: Threshold-based Reliable Negative Selection
with torch.no_grad():
    probabilities = classifier(X_Ur).numpy()  # Get probabilities for Ur instances

# Sort Ur by probabilities, ascending (low probabilities indicate negatives)
sorted_indices = np.argsort(probabilities)
Ur_sorted = Ur[sorted_indices]
prob_sorted = probabilities[sorted_indices]

# Determine reliable negatives using threshold θ = 0.15
theta = 0.15
reliable_negatives
"""