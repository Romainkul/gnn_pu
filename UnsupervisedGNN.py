import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec, DeepGraphInfomax, SAGEConv
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from data_generating import load_dataset, make_pu_dataset
from torch_geometric.nn import Node2Vec
from sklearn.metrics import accuracy_score, f1_score

# Load PU dataset
data = load_dataset('citeseer')
data = make_pu_dataset(data, pos_index=[0,1,2], sample_seed=2,
                           train_pct=0.2, val_pct=0.2, test_pct=1.0)
print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_node2vec(data, embedding_dim=64):
    """Train Node2Vec in an unsupervised manner and return embeddings."""
    node2vec = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=20, context_size=10, walks_per_node=10, sparse=True).to(device)
    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 99:
            print(f"Node2Vec Epoch {epoch + 1}/100, Loss: {total_loss / len(loader):.4f}")
    
    node2vec.eval()
    with torch.no_grad():
        embeddings = node2vec(torch.arange(data.num_nodes, device=device)).cpu().numpy()
    return embeddings

import torch
from torch_geometric.nn import Node2Vec

def train_deepwalk(data, embedding_dim=64, walk_length=40, context_size=10, walks_per_node=10, epochs=100):
    """
    Simulate DeepWalk using Node2Vec by setting p=1 and q=1 to perform unbiased random walks.
    
    Parameters:
        data (Data): Graph data with edge_index and x attributes.
        embedding_dim (int): Size of each node embedding.
        walk_length (int): Length of each random walk.
        context_size (int): Size of the context window for Skip-Gram model.
        walks_per_node (int): Number of walks to start at each node.
        epochs (int): Number of training epochs.
    
    Returns:
        embeddings (np.ndarray): Node embeddings after DeepWalk training.
    """
    # Configure Node2Vec with p=1 and q=1 to simulate DeepWalk
    deepwalk = Node2Vec(
        data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
        context_size=context_size, walks_per_node=walks_per_node, p=1, q=1, sparse=True
    ).to(device)
    
    # Set up DataLoader
    loader = deepwalk.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(deepwalk.parameters(), lr=0.01)

    # Training loop
    deepwalk.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            # Calculate loss for the random walks
            loss = deepwalk.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"DeepWalk Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Obtain embeddings after training
    deepwalk.eval()
    with torch.no_grad():
        embeddings = deepwalk(torch.arange(data.num_nodes, device=device)).cpu().numpy()
    
    return embeddings

def train_graphsage(data, in_channels, hidden_channels=64):
    #add skip-connections
    #add residual connections
    #add batch normalization
    #add dropout
    """Train GraphSAGE in an unsupervised manner and return embeddings."""
    from torch_geometric.nn import GCNConv, SAGEConv

    class SAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                    dropout):
            super(SAGE, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

            self.dropout = dropout

        def reset_parameters(self):
            for conv in self.convs:
                conv.reset_parameters()

        def forward(self, x, adj_t):
            for conv in self.convs[:-1]:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)
            return torch.log_softmax(x, dim=-1)

    model = SAGE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = data.x.to(device)
    edge_index = data.train_pos_edge_index.to(device)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        embeddings = model(x, edge_index)
        loss = DeepGraphInfomax.loss(embeddings, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 99:
            print(f"GraphSAGE Epoch {epoch + 1}/100, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index).cpu().numpy()
    return embeddings

# Generate embeddings for Node2Vec, DeepWalk, and GraphSAGE
node2vec_embeddings = train_node2vec(data)
deepwalk_embeddings = train_deepwalk(data)
graphsage_embeddings = train_graphsage(data, in_channels=data.num_features)

# Combine embeddings with node features
combined_node2vec = np.hstack((node2vec_embeddings, data.x.cpu().numpy()))
combined_deepwalk = np.hstack((deepwalk_embeddings, data.x.cpu().numpy()))
combined_graphsage = np.hstack((graphsage_embeddings, data.x.cpu().numpy()))

# Function to select reliable positives and negatives
def select_reliable_samples(embeddings, labels, threshold=0.5):
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)
    positive_samples = []
    negative_samples = []
    
    for i in range(len(labels)):
        if labels[i] == 1:
            if similarities[i].mean() > threshold:
                positive_samples.append(embeddings[i])
        else:
            if similarities[i].mean() < threshold:
                negative_samples.append(embeddings[i])

    return np.array(positive_samples), np.array(negative_samples)

# Select reliable positive and negative samples
y = data.y.cpu().numpy()
pos_node2vec, neg_node2vec = select_reliable_samples(combined_node2vec, y)
pos_deepwalk, neg_deepwalk = select_reliable_samples(combined_deepwalk, y)
pos_graphsage, neg_graphsage = select_reliable_samples(combined_graphsage, y)

# Prepare data for logistic regression
def prepare_lr_data(pos_samples, neg_samples):
    X = np.vstack((pos_samples, neg_samples))
    y = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
    return X, y

X_node2vec, y_node2vec = prepare_lr_data(pos_node2vec, neg_node2vec)
X_deepwalk, y_deepwalk = prepare_lr_data(pos_deepwalk, neg_deepwalk)
X_graphsage, y_graphsage = prepare_lr_data(pos_graphsage, neg_graphsage)

# Train logistic regression and evaluate
def train_evaluate_lr(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_train)
    acc = accuracy_score(y_train, preds)
    f1 = f1_score(y_train, preds)
    return acc, f1

# Evaluating each model
acc_node2vec, f1_node2vec = train_evaluate_lr(X_node2vec, y_node2vec)
acc_deepwalk, f1_deepwalk = train_evaluate_lr(X_deepwalk, y_deepwalk)
acc_graphsage, f1_graphsage = train_evaluate_lr(X_graphsage, y_graphsage)

print(f"Node2Vec - Accuracy: {acc_node2vec:.4f}, F1 Score: {f1_node2vec:.4f}")
print(f"DeepWalk - Accuracy: {acc_deepwalk:.4f}, F1 Score: {f1_deepwalk:.4f}")
print(f"GraphSAGE - Accuracy: {acc_graphsage:.4f}, F1 Score: {f1_graphsage:.4f}")


"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
import numpy as np

class GraphDiffusionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, alpha=0.15, device='cpu'):
        super(GraphDiffusionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.edge_index = edge_index
        self.alpha = alpha
        self.device = device

        # Compute diffusion-based pseudo-labels
        self.pseudo_labels = self.compute_ppr(edge_index, alpha).to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x, self.edge_index))
        x = self.conv2(x, self.edge_index)
        return x

    def compute_ppr(self, edge_index, alpha=0.15):
        """#Compute Personalized PageRank (PPR) as diffusion-based pseudo-labels.
"""
        num_nodes = int(edge_index.max()) + 1
        adj_matrix = to_dense_adj(edge_index)[0].cpu().numpy()
        ppr_matrix = np.zeros((num_nodes, num_nodes))

        # Calculate PPR for each node using the closed-form solution
        for i in range(num_nodes):
            # Set up teleport vector (one-hot vector for each node)
            teleport = np.zeros(num_nodes)
            teleport[i] = 1
            ppr_matrix[i] = np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * adj_matrix) @ (alpha * teleport)

        # Convert PPR scores to torch tensor for training
        ppr_scores = torch.tensor(ppr_matrix, dtype=torch.float32)
        return ppr_scores

    def diffusion_cross_entropy_loss(self, embeddings):
        """#Cross-entropy loss based on PPR-based pseudo-labels.
"""
        # Calculate probabilities using sigmoid
        probs = torch.sigmoid(embeddings)
        
        # Cross-entropy loss with PPR as pseudo-labels
        cross_entropy_loss = F.binary_cross_entropy(probs, self.pseudo_labels, reduction='mean')
        return cross_entropy_loss

    def smoothness_loss(self, embeddings):
        """#Smoothness loss based on the difference between connected node embeddings.
"""
        row, col = self.edge_index
        smooth_loss = F.mse_loss(embeddings[row], embeddings[col], reduction='mean')
        return smooth_loss

    def combined_loss(self, embeddings, smoothness_weight=0.1):
        """#Combined loss function with diffusion cross-entropy and smoothness regularization.
"""
        ce_loss = self.diffusion_cross_entropy_loss(embeddings)
        smooth_loss = self.smoothness_loss(embeddings)
        return ce_loss + smoothness_weight * smooth_loss

# Training loop
def train(model, data, epochs=100, lr=0.01, smoothness_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data.x.to(model.device))
        
        # Compute combined loss
        loss = model.combined_loss(embeddings, smoothness_weight=smoothness_weight)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Assuming `data` is your graph data with attributes `x` (node features) and `edge_index` (edges)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphDiffusionGNN(in_channels=data.num_node_features, hidden_channels=64, 
                          out_channels=1, edge_index=data.edge_index, alpha=0.15, device=device).to(device)

train(model, data, epochs=100, lr=0.01, smoothness_weight=0.1)
"""

"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ContrastiveGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContrastiveGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def contrastive_loss(embeddings, edge_index, tau=0.5):
    """
"""Contrastive loss to enforce similarity for nodes that co-occur in random walks.
    
    Parameters:
        embeddings (Tensor): Node embeddings.
        edge_index (Tensor): Edge indices for the graph.
        tau (float): Temperature parameter for contrastive loss.
    
    Returns:
        loss (Tensor): Contrastive loss."""
"""
    # Positive pairs are nodes connected by an edge
    row, col = edge_index
    pos_sim = torch.exp(F.cosine_similarity(embeddings[row], embeddings[col]) / tau)
    
    # Negative pairs are randomly selected nodes
    num_nodes = embeddings.size(0)
    neg_indices = torch.randint(0, num_nodes, (row.size(0),), device=embeddings.device)
    neg_sim = torch.exp(F.cosine_similarity(embeddings[row], embeddings[neg_indices]) / tau)
    
    # Calculate the contrastive loss
    loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
    return loss

def smoothness_loss(embeddings, edge_index):
    """

"""Smoothness loss to enforce neighboring nodes have similar embeddings.
    
    Parameters:
        embeddings (Tensor): Node embeddings.
        edge_index (Tensor): Edge indices for the graph.
    
    Returns:
        loss (Tensor): Smoothness loss."""

"""
    row, col = edge_index
    loss = torch.norm(embeddings[row] - embeddings[col], p=2, dim=1).mean()
    return loss

def train_gnn(data, epochs=100, embedding_dim=64, lr=0.01, tau=0.5, alpha=0.5):
    """
"""
    Train GNN with combined Contrastive Loss and Smoothness Loss.
    
    Parameters:
        data (Data): Graph data.
        epochs (int): Number of training epochs.
        embedding_dim (int): Dimension of node embeddings.
        lr (float): Learning rate.
        tau (float): Temperature parameter for contrastive loss.
        alpha (float): Weight for smoothness loss in the final loss calculation."""
"""
    # Initialize the model and optimizer
    model = ContrastiveGNN(data.num_features, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data)
        
        # Calculate losses
        c_loss = contrastive_loss(embeddings, data.edge_index, tau)
        s_loss = smoothness_loss(embeddings, data.edge_index)
        total_loss = c_loss + alpha * s_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Contrastive Loss: {c_loss.item():.4f}, "
                  f"Smoothness Loss: {s_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    return model

# Example usage with your data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
embedding_dim = 64
epochs = 100
tau = 0.5
alpha = 0.5

# Train the model with combined loss
trained_model = train_gnn(data, epochs=epochs, embedding_dim=embedding_dim, lr=0.01, tau=tau, alpha=alpha)

# Obtain final embeddings
trained_model.eval()
with torch.no_grad():
    embeddings = trained_model(data).cpu().numpy()
"""