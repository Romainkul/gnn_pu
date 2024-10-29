import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from PU_Loss import PULoss 
from data_generating import load_dataset, make_pu_dataset

import torch

import torch

def edge_index_to_adjacency(num_nodes, edge_index):
    """
    Converts edge_index format to an adjacency matrix.

    Parameters:
    - num_nodes: Total number of nodes in the graph
    - edge_index: Tensor of shape (2, num_edges)

    Returns:
    - Adjacency matrix (num_nodes x num_nodes)
    """
    # Initialize an empty adjacency matrix
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Populate the adjacency matrix with edges
    A[edge_index[0], edge_index[1]] = 1

    # If the graph is undirected, mirror the edges
    A = A + A.T
    A[A > 1] = 1  # Ensure binary values in case of duplicate edges

    return A

# Short-Distance Attention Mechanism
class ShortDistanceAttention(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(ShortDistanceAttention, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)  # W^(1)
        self.r = nn.Parameter(torch.randn(out_features * 2, 1))  # Learnable attention vector r
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, X, A):
        """
        X: Node feature matrix (n_nodes x in_features)
        A: Adjacency matrix (n_nodes x n_nodes) for direct neighbors
        """
        Wh = self.W(X)  # Transform features: W * X
        n_nodes = X.size(0)
        Ak_list = [torch.matrix_power(A, k) for k in range(1, self.num_hops + 1)]

        # Expand dimensions for broadcasting
        Wh_i = Wh.unsqueeze(1).expand(n_nodes, n_nodes, -1)
        Wh_j = Wh.unsqueeze(0).expand(n_nodes, n_nodes, -1)

        # Concatenate features of nodes i and j, compute e_{ij}
        concatenated_features = torch.cat([Wh_i, Wh_j], dim=-1)  # Shape: (n_nodes, n_nodes, 2*out_features)
        e_ij = self.leakyrelu(torch.matmul(concatenated_features, self.r).squeeze(-1))  # Shape: (n_nodes, n_nodes)

        # Mask to keep only the neighbors in the adjacency matrix
        e_ij = e_ij.masked_fill(A == 0, float('-inf'))  # Set non-neighbors to -inf
        attention = torch.softmax(e_ij, dim=1)  # Apply softmax to compute normalized attention

        # Compute final output by aggregating attention-weighted features
        output = torch.matmul(attention, Wh)  # Shape: (n_nodes, out_features)
        hk=F.gelu(output)
        return hk


# Long-Short Distance Attention Mechanism with Residual Connections
class LongDistanceAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hops):
        super(LongDistanceAttention, self).__init__()
        self.num_hops = num_hops
        self.h, self.hk = ShortDistanceAttention(input_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # W^(2)
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Final layer for output transformation

    def forward(self, X, A):
        # Initial transformation
        h = self.h(X, A)  # Short-distance attention
        Wa = self.W(h)  # Linear transformation W^(2) * hk for attention

        # Save the initial hidden features for residual connections
        original_input = h.clone()

        # Generate k-hop adjacency matrices
        Ak_list = [torch.matrix_power(A, k) for k in range(1, self.num_hops + 1)]
        n_nodes = X.size(0)

        # Initialize output with the original features (for residual connections)
        output = original_input

        # Process each k-hop neighborhood
        for Ak in Ak_list:
            # Compute attention scores using matrix multiplication (vectorized)
            attention_scores = torch.matmul(h, Wa.T)  # Shape: (n_nodes, n_nodes)
            
            # Mask out non-neighbor values in attention scores
            attention_scores = attention_scores.masked_fill(Ak == 0, float('-inf'))
            
            # Softmax normalization along each row (neighbors only)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Aggregate neighbor features based on attention weights
            aggregated_features = torch.matmul(attention_weights, hk)  # Shape: (n_nodes, hidden_dim)
            
            # Add residual connection
            output += aggregated_features
        
        # Final transformation to output layer
        final_output = self.output_layer(output)
        return final_output

        """
        # Generate k-hop adjacency matrices
        Ak_list = [A]  # 1-hop neighbors
        for k in range(2, self.num_hops + 1):
            Ak_list.append(torch.matrix_power(A, k))

        n_nodes = X.size(0)
        
        # Loop through each k-hop adjacency matrix and compute attention
        for k, Ak in enumerate(Ak_list):
            attention = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=X.device)
            
            # Calculate attention scores for each k-hop neighbor
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if Ak[i, j] != 0:  # Only consider k-hop neighbors
                        attention[i, j] = torch.matmul(hk[i], Wa[j])
            
            # Apply softmax normalization across neighbors
            attention_exp = torch.exp(attention)
            for i in range(n_nodes):
                attention_exp[i] /= torch.sum(attention_exp[i][Ak[i] != 0])
            
            # Aggregate node features based on normalized attention
            output = torch.zeros((n_nodes, hk.size(1)), dtype=torch.float32, device=X.device)
            for i in range(n_nodes):
                neighbors = hk[Ak[i] != 0]
                weights = attention_exp[i][Ak[i] != 0].view(-1, 1)
                output[i] = torch.sum(weights * neighbors, dim=0)
            
            # Add residual connection
            output += original_input
            hk = output  # Update hk with residual connection
        
        # Final transformation to output layer
        final_output = self.output_layer(output)
        return final_output"""

# PU loss function for training
def pu_loss(x, t, prior, gamma=1, beta=0, nnpu=True):
    loss_fn = PULoss(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu)
    return loss_fn(x, t)

# Training function
def train(model, data, optimizer, prior, beta=0.0, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()
    
    # Check for NaN/Inf in input data
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        print("Warning: Data contains NaN or Inf values.")

    out = model(data.x, data.edge_index).squeeze()
    loss = pu_loss(out[data.train_mask], data.y[data.train_mask], prior)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()

# Evaluation function for validation
def evaluate(model, data, prior):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = torch.sigmoid(out)  # Get probabilities for binary classification
        val_loss = pu_loss(out[data.val_mask], data.y[data.val_mask], prior)
        correct = (pred[data.val_mask].round() == data.y[data.val_mask]).sum().item()
        accuracy = correct / data.val_mask.sum().item()
    return val_loss.item(), pred, accuracy

# Evaluation function for test
def evaluate_test(model, data, prior):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = torch.sigmoid(out)  # Get probabilities for binary classification
        test_loss = pu_loss(out[data.test_mask], data.y[data.test_mask], prior)
        correct = (pred[data.test_mask].round() == data.y[data.test_mask]).sum().item()
        accuracy = correct / data.test_mask.sum().item()
    return test_loss.item(), pred, accuracy

# Main training and evaluation routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and evaluate PU learning with GNNs')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer', help='Data set to use')
    parser.add_argument('--positive_index', '-c', type=list, default=[0], help='Label to treat as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=2, help='Seed for sampling labeled positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2, help='Training positive percentage')
    parser.add_argument('--val_pct', '-v', type=float, default=0.2, help='Validation positive percentage')
    parser.add_argument('--test_pct', '-t', type=float, default=1.0, help='Test set percentage for unknown nodes')
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load PU dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=args.positive_index, sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    data = data.to(device)
    print(data.edge_index)
    A = edge_index_to_adjacency(len(data.y), data.edge_index)
    A = A.to(device)  # Move adjacency matrix to the same device as model

    
    in_channels = data.num_node_features
    hidden_channels = 16
    out_channels = 1  # Binary classification

    # Initialize LongDistanceAttention model
    num_hops = 3  # Number of hops for multi-hop aggregation in LongDistanceAttention
    model = LongDistanceAttention(in_channels, hidden_channels, num_hops).to(device)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    # PU-Learning parameters
    prior = data.prior
    beta = 0.01

    # Training parameters
    epochs = 200
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    wait = 0

    # Training loop
    for epoch in range(epochs):
        loss = train(model, data, optimizer, prior, beta)
        val_loss, _, accuracy = evaluate(model, data, prior)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1:03d}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save the model if it's the best
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
        
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Testing phase
    test_loss, predictions, accuracy = evaluate_test(model, data, prior)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")