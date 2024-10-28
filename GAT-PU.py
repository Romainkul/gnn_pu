import torch
from torch_geometric.data import Data
import random
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch.optim import AdamW
import argparse
import warnings
warnings.filterwarnings("ignore")
from data_generating import load_dataset, make_pu_dataset
from PU_Loss import PULoss

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)

def pu_loss(x, t, prior, gamma=1, beta=0, nnpu=True):
    loss_fn = PULoss(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu)
    return loss_fn(x, t)

def train(model, data, optimizer, prior, beta=0.0, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()

    # Check for NaN/Inf in input data
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        print("Data contains NaN or Inf values.")

    out = model(data.x, data.edge_index).squeeze()
    loss = pu_loss(out[data.train_mask], data.y[data.train_mask], prior)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item()

def evaluate(model, data, prior):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = torch.sigmoid(out) 
        val_loss = pu_loss(out[data.val_mask], data.y[data.val_mask], prior,)
        correct = (pred[data.val_mask].round() == data.y[data.val_mask]).sum().item()
        accuracy = correct / data.val_mask.sum().item()
    return val_loss.item(), pred, accuracy

def evaluate_test(model, data, prior):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = torch.sigmoid(out) 
        test_loss = pu_loss(out[data.test_mask], data.y[data.test_mask], prior,)
        correct = (pred[data.test_mask].round() == data.y[data.test_mask]).sum().item()
        accuracy = correct / data.test_mask.sum().item()
    return test_loss.item(), pred, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load parameters for running and evaluating bootstrap PU learning')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer',
                        help='Data set to be used')
    parser.add_argument('--positive_index', '-c', type=list, default=[0],
                        help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=2,
                        help='Random seed for sampling labeled positive from all positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2,
                        help='Percentage of positive nodes to be used as training positive')
    parser.add_argument('--val_pct', '-v', type=float, default=0.2,
                        help='Percentage of positive nodes to be used as evaluating positive')
    parser.add_argument('--test_pct', '-t', type=float, default=1.0,
                        help='Percentage of unknown nodes to be used as test set')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load PU dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=args.positive_index, sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    data = data.to(device)
    in_channels = data.num_node_features
    out_channels = 1 

    model = GAT(in_channels, out_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    prior = data.prior
    beta = 0.01

    epochs = 200
    best_val_loss = float('inf')
    patience = 10  # number of epochs to wait before early stopping
    wait = 0

    for epoch in range(epochs):
        loss = train(model, data, optimizer, prior, beta)
        val_loss, _, _ = evaluate(model, data, prior)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save model if you want to keep the best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    test_loss, predictions, accuracy = evaluate_test(model, data, prior)
    print(f"Validation Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")
