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
        return F.log_softmax(x, dim=1)

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

    # Backpropagation
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item()

def evaluate(model, data, prior):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = torch.sigmoid(out)  # Convert logits to probabilities
        val_loss = pu_loss(out[data.val_mask], data.y[data.val_mask], prior,)
    return val_loss.item(), pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load parameters for running and evaluating bootstrap PU learning')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer',
                        help='Data set to be used')
    parser.add_argument('--positive_index', '-c', type=list, default=[0],
                        help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=1,
                        help='Random seed for sampling labeled positive from all positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2,
                        help='Percentage of positive nodes to be used as training positive')
    parser.add_argument('--val_pct', '-v', type=float, default=0.1,
                        help='Percentage of positive nodes to be used as evaluating positive')
    parser.add_argument('--test_pct', '-t', type=float, default=1.00,
                        help='Percentage of unknown nodes to be used as test set')
    parser.add_argument('--hidden_size', '-l', type=int, default=32,
                        help='Size of hidden layers')
    parser.add_argument('--output_size', '-o', type=int, default=16,
                        help='Dimension of output representations')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Dataset: ", args.dataset)
    # Load PU dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=args.positive_index, sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    data = data.to(device)

    # Initialize the model, optimizer, and other parameters
    in_channels = data.num_node_features
    out_channels = 1  # Binary classification

    model = GAT(in_channels, out_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Estimated prior probability P(Y=1)
    prior = data.prior
    beta = 0.0  # Hyperparameter for the non-negative risk estimator

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        loss = train(model, data, optimizer, prior, beta)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    val_loss, predictions = evaluate(model, data, prior)
    print(f"Validation Loss: {val_loss:.4f}")
