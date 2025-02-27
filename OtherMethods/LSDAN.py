import torch
import torch.nn as nn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from nnPU import PULoss
from data_generating import load_dataset, make_pu_dataset
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
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to enhance reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate_LSDAN(
    dataset_name: str = 'citeseer',
    mechanism: str = 'SCAR',
    seed: int = 1,
    train_pct: float = 0.5,
    hidden_dim: int = 64,
    output_dim: int = 2,  # Binary classification: two logits
    num_layers: int = 3,
    dropout: float = 0.5,
    num_hops: int = 3,
    num_epochs: int = 50,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4
):
    """
    Trains the LSDAN model using a non-negative unbiased PU estimator and evaluates performance.
    
    The evaluation selects the top fraction of nodes (equal to the class prior) as positives.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    mechanism : str
        PU mechanism to use (e.g., 'SCAR').
    seed : int
        Random seed for reproducibility.
    train_pct : float
        Fraction of positive nodes to treat as labeled.
    hidden_dim : int
        Hidden dimension of the model.
    output_dim : int
        Output dimension (2 for binary classification).
    num_layers : int
        Number of layers in the LSDAN encoder.
    dropout : float
        Dropout probability.
    num_hops : int
        Number of hops (multi-hop adjacency matrices) to compute.
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    
    Returns
    -------
    None
        Prints training loss and evaluation metrics.
    """
    # 1) Set the random seed for reproducibility
    set_seed(seed)
    
    # 2) Load dataset and create PU labels
    data = load_dataset(dataset_name)
    data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)
    
    # Enforce target encoding: positives -> 1, unlabeled -> -1
    data.y = torch.where(data.y > 0, torch.tensor(1, device=data.y.device), torch.tensor(-1, device=data.y.device))
    
    # 3) Move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # 4) Convert edge_index to a SparseTensor and compute multi-hop adjacency matrices
    adj_matrix = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                              sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
    adj_matrices = [adj_matrix]
    current_adj = adj_matrix
    for _ in range(1, num_hops):
        current_adj = current_adj @ adj_matrix  # Sparse multiplication for k-hop adjacency
        adj_matrices.append(current_adj)
    
    # 5) Initialize Model, Loss, and Optimizer
    model = LSDAN(data.num_features, hidden_dim, output_dim, num_layers, num_hops, dropout=dropout).to(device)
    # Use the dataset's prior if available; otherwise, default to 0.5.
    prior = data.prior if hasattr(data, 'prior') else 0.5
    pu_loss = PULoss(prior=prior).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 6) Training Loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass; model expects x, edge_index, and the list of adjacency matrices.
        out = model(data.x, data.edge_index, adj_matrices).squeeze()
        # If output_dim == 2, select the positive class logit.
        if output_dim == 2:
            out = out[:, 1]  # shape: [num_nodes]
        
        # Compute PU Loss on all training nodes (or you can use a mask if defined)
        loss = pu_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("Training Complete!")
    
    # 7) Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, adj_matrices).squeeze()
        if output_dim == 2:
            logits = logits[:, 1]
        
        # Convert logits to probabilities with sigmoid
        probs = torch.sigmoid(logits)
        
        # Instead of a fixed threshold, select the top fraction equal to the class prior as positives.
        prior = data.prior if hasattr(data, 'prior') else 0.5
        num_nodes = probs.shape[0]
        num_positive = int(round(num_nodes * prior))
        
        sorted_indices = torch.argsort(probs, descending=True)
        preds = torch.zeros_like(probs, dtype=torch.long)
        preds[sorted_indices[:num_positive]] = 1
        preds = preds.cpu().numpy()
        
        # For metric calculation, convert targets: 1 for positives, 0 for unlabeled.
        labels = (data.y == 1).long().cpu().numpy()
    
    # Compute evaluation metrics.
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    rec = recall_score(labels, preds)
    prec = precision_score(labels, preds)
    
    print("\n=== Evaluation on Test Set ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")