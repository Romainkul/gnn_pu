import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from typing import List, Tuple
from PU_Loss import PULoss 
from data_generating import load_dataset, make_pu_dataset
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Initialize logging
logging.basicConfig(level=logging.INFO)

def edge_index_to_adjacency(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Converts edge_index format to an adjacency matrix.

    Parameters:
    - num_nodes: Total number of nodes in the graph
    - edge_index: Tensor of shape (2, num_edges)

    Returns:
    - Adjacency matrix (num_nodes x num_nodes)
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1
    A = A + A.T
    A = (A > 0).float()  # Ensure binary values
    return A


class LSDAN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, num_hops: int, alpha: float = 0.2, residual: bool = True):
        super(LSDAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.W2 = nn.Linear(in_features, hidden_dim, bias=False)
        self.r = nn.Parameter(torch.randn(hidden_dim * 2, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.num_hops = num_hops
        self.residual = residual
        nn.init.kaiming_uniform_(self.W1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2.weight, a=math.sqrt(5))
        nn.init.normal_(self.r) 

    def forward(self, X: torch.Tensor, A: torch.Tensor, U_l: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        W1 = self.W1(X)
        W2 = self.W2(X)
        n_nodes = X.size(0)
        Ak_list = [torch.matrix_power(A, k) for k in range(1, self.num_hops + 1)]
        output = torch.zeros((n_nodes, self.hidden_dim))
        for k, Ak in enumerate(Ak_list, start=1):
            Wh_i = W1.unsqueeze(1).expand(n_nodes, n_nodes, -1)
            Wh_j = W1.unsqueeze(0).expand(n_nodes, n_nodes, -1)
            concatenated_features = torch.cat([Wh_i, Wh_j], dim=-1)
            e_ij = self.leakyrelu(torch.matmul(concatenated_features, self.r).squeeze(-1))
            e_ij = e_ij.masked_fill(Ak == 0, float('-1000'))
            attention = F.softmax(e_ij, dim=1)

            h_k = torch.matmul(attention, W1)
            attention_scores = torch.matmul(h_k, W2.T)
            attention_scores = attention_scores.masked_fill(Ak == 0, float('-1000'))
            attention_weights = F.softmax(attention_scores, dim=1)
            aggregated_features = torch.matmul(attention_weights, h_k)
            output += aggregated_features

        O_l = output
        U_l_plus_1 = U_l + O_l if self.residual else O_l

        return U_l_plus_1, O_l
class DeepLSDAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, num_layers: int, num_hops: int):
        super(DeepLSDAN, self).__init__()
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList()
        self.layers.append(LSDAN(in_features, hidden_features, out_features, num_hops, residual=False))
        self.W = nn.Linear(in_features, hidden_features, bias=False)
        self.output_layer = nn.Linear(hidden_features, out_features)

        for _ in range(1, num_layers - 1):
            self.layers.append(LSDAN(in_features, hidden_features, out_features, num_hops, residual=True))

        self.layers.append(LSDAN(in_features, hidden_features, out_features, num_hops, residual=False))

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n_nodes = X.size(0)
        U_l = torch.zeros((n_nodes, self.hidden_features))
        O_l = torch.zeros((n_nodes, self.hidden_features))

        for layer in self.layers:
            U_l, O_l = layer(X, A, U_l)

        output = self.output_layer(O_l)
        output = F.softmax(output, dim=1)  # Apply softmax if it's a classification problem
        return output


def pu_loss(x: torch.Tensor, t: torch.Tensor, prior: float, gamma: float = 1, beta: float = 0, nnpu: bool = True) -> torch.Tensor:
    loss_fn = PULoss(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu)
    return loss_fn(x, t)

def train(model: nn.Module, data,A, optimizer: torch.optim.Optimizer, prior: float, beta: float = 0.0, max_grad_norm: float = 1.0) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x,A).squeeze()
    loss = pu_loss(out[data.train_mask], data.y[data.train_mask], prior)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()

def evaluate(model: nn.Module, data, A, prior: float) -> Tuple[float, torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        out = model(data.x,A).squeeze()
        pred = torch.sigmoid(out)
        val_loss = pu_loss(out[data.val_mask], data.y[data.val_mask], prior)
        correct = (pred[data.val_mask].round() == data.y[data.val_mask]).sum().item()
        accuracy = correct / data.val_mask.sum().item()
    return val_loss.item(), pred, accuracy

def evaluate_test(model: nn.Module, data,A, prior: float) -> Tuple[float, torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        out = model(data.x,A).squeeze()
        pred = torch.sigmoid(out)
        test_loss = pu_loss(out[data.test_mask], data.y[data.test_mask], prior)
        correct = (pred[data.test_mask].round() == data.y[data.test_mask]).sum().item()
        accuracy = correct / data.test_mask.sum().item()
    return test_loss.item(), pred, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and evaluate PU learning with GNNs')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer', help='Data set to use')
    parser.add_argument('--positive_index', '-c', type=list, default=[0], help='Label to treat as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=2, help='Seed for sampling labeled positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2, help='Training positive percentage')
    parser.add_argument('--val_pct', '-v', type=float, default=0.2, help='Validation positive percentage')
    parser.add_argument('--test_pct', '-t', type=float, default=1.0, help='Test set percentage for unknown nodes')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=args.positive_index, sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    data = data.to(device)

    A = edge_index_to_adjacency(len(data.y), data.edge_index).to(device)
    in_channels = data.num_node_features
    hidden_channels = 16
    out_channels = 1

    model = DeepLSDAN(in_channels, out_channels,hidden_channels, num_layers=3, num_hops=3).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    prior = data.prior
    beta = 0.01

    # Training parameters
    epochs = 5
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    wait = 0

    # Training loop
    for epoch in range(epochs):
        loss = train(model, data,A, optimizer, prior, beta)
        val_loss, _, accuracy = evaluate(model, data,A, prior)
        
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
    test_loss, predictions, accuracy = evaluate_test(model, data,A, prior)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")
