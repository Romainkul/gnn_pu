import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import AdamW

from ..NNIF-GNN.data_generating import load_dataset, make_pu_dataset
from algorithm import *
from model_helper import *
from estimator import *

np.set_printoptions(suppress=True, precision=1)

# Argument Parsing
parser = argparse.ArgumentParser(description="PU Learning Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
parser.add_argument("--data-type", type=str, help="Dataset type: graph")
parser.add_argument("--train-method", type=str, help="Training algorithm to use: TEDn | CVIR | nnPU")
parser.add_argument("--net-type", type=str, help="GCN | GAT | GraphSAGE")
parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--alpha", type=float, default=0.5, help="PU mixture proportion")
parser.add_argument("--beta", type=float, default=0.5, help="Proportion of labeled data")
parser.add_argument("--log-dir", type=str, default="logging_accuracy", help="Logging directory")
parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer: SGD | Adam | AdamW")

args = parser.parse_args()

# Set Random Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
data_type = args.data_type
net_type = args.net_type
alpha = args.alpha
beta = args.beta
epochs = args.epochs
log_dir = os.path.join(args.log_dir, data_type)
optimizer_str = args.optimizer

# Load Full Graph Data in a Single Batch
graph_data = load_dataset(args.data_dir, data_type)
x, edge_index, y, pos_idx, unlabeled_idx = make_pu_dataset(graph_data, alpha, beta, device)

# Define GNN Models
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Model Selection
if net_type == "GCN":
    model = GCN(x.size(1)).to(device)
elif net_type == "GAT":
    model = GAT(x.size(1)).to(device)
elif net_type == "GraphSAGE":
    model = GraphSAGE(x.size(1)).to(device)
else:
    raise ValueError("Invalid model type. Choose from GCN, GAT, GraphSAGE.")

# Optimizer Selection
criterion = nn.CrossEntropyLoss()
if optimizer_str == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif optimizer_str == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif optimizer_str == "AdamW":
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    raise ValueError("Invalid optimizer. Choose from SGD, Adam, AdamW.")

# **TED (Two-step PU Learning)**
def train_TED(model, x, edge_index, pos_idx, unlabeled_idx, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x, edge_index)

        # Step 1: Rank Unlabeled Nodes & Remove Noisy Negatives
        keep_samples, neg_reject = rank_inputs(epoch, model, x, edge_index, device, alpha, len(unlabeled_idx))

        # Step 2: Train with Selected Reliable Negatives
        loss = criterion(outputs[pos_idx], torch.ones_like(pos_idx)) + \
               criterion(outputs[unlabeled_idx[keep_samples]], torch.zeros_like(unlabeled_idx[keep_samples]))

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()} | Negatives Rejected: {neg_reject:.4f}")

# **BBE (Bias Bound Estimation)**
def estimate_BBE(model, x, edge_index, pos_idx, unlabeled_idx):
    model.eval()
    with torch.no_grad():
        pos_probs = F.softmax(model(x, edge_index), dim=-1)[pos_idx, 1]
        unlabeled_probs = F.softmax(model(x, edge_index), dim=-1)[unlabeled_idx, 1]

        mpe_estimate, _, _ = BBE_estimator(pos_probs.cpu().numpy(), unlabeled_probs.cpu().numpy(), np.zeros(len(unlabeled_idx)))

    return mpe_estimate

# **Validation Function**
def validate(model, x, edge_index, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(x, edge_index)
        preds = (F.softmax(outputs, dim=-1)[:, 1] > threshold).long()
        correct = (preds == y).sum().item()
        accuracy = correct / y.size(0)
    return accuracy

# **Training Execution**
print("Starting Training...")
if train_method == "TEDn":
    train_TED(model, x, edge_index, pos_idx, unlabeled_idx, optimizer, criterion, epochs)
    alpha_estimate = estimate_BBE(model, x, edge_index, pos_idx, unlabeled_idx)
    print(f"Estimated Alpha (BBE): {alpha_estimate:.4f}")

# **Evaluating the Model**
accuracy = validate(model, x, edge_index, y)
print(f"Final Accuracy: {accuracy:.4f}")
