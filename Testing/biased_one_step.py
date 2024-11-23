import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, BatchNorm, GATConv, GINConv
from torch_geometric.utils import degree, to_undirected, to_dense_adj, contains_self_loops, add_self_loops, dense_to_sparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from data_generating import load_dataset, make_pu_dataset
from Loss_functions import LabelPropagationLoss
import logging
import argparse
import torch.optim as optim
#look into dropout_adj

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EarlyStopping_GNN:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class NNSampler:
    def __init__(self, num_neighbors, distance_function="cosine"):
        self.num_neighbors = num_neighbors
        self.distance_function = distance_function

    def sample_neighbors(self, node_features, edge_index, node_types=None):
        src, dst = edge_index  # source and destination nodes
        sampled_edges = []
        for node in torch.unique(src):
            neighbors = dst[src == node]
            if neighbors.numel() == 0:
                continue

            # Calculate distances
            node_feat = node_features[node].unsqueeze(0)
            neighbor_feats = node_features[neighbors]

            if self.distance_function == "euclidean":
                distances = torch.cdist(node_feat, neighbor_feats).squeeze(0)  # [num_neighbors]
                probabilities = F.softmax(-distances, dim=0)
            elif self.distance_function == "cosine":
                distances = F.cosine_similarity(node_feat, neighbor_feats)
                probabilities = F.softmax(distances, dim=0)

            # Sample neighbors based on probabilities
            sampled_neighbors = torch.multinomial(probabilities, min(self.num_neighbors, len(neighbors)), replacement=True)
            sampled_edges.extend([(node.item(), neighbors[idx].item()) for idx in sampled_neighbors])

        sampled_edge_index = torch.tensor(sampled_edges, dtype=torch.long).T  # shape [2, num_sampled_edges]
        return sampled_edge_index

class GNNClassifier(nn.Module):
    def __init__(self, encoder,in_channels,hidden_channels,num_layers=3,sampler=False,aggregation="max"):
        super(GNNClassifier, self).__init__()

        self.num_layers = num_layers
        self.sampler = sampler
        self.aggregation = aggregation
        self.out_channels = 2
    
        # Create a list of layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer (input to first hidden)
        self.convs.append(encoder(in_channels, hidden_channels, aggr=aggregation))
        self.bns.append(BatchNorm(hidden_channels))


        # Hidden layers
        for _ in range(num_layers - 2):  # num_layers - 2 because of input-to-hidden and last layer
            self.convs.append(encoder(hidden_channels, hidden_channels, aggr=aggregation))
            self.bns.append(BatchNorm(hidden_channels))

        # Final layer (hidden to output)
        self.convs.append(encoder(hidden_channels, self.out_channels, aggr=aggregation))

    def forward(self, x, edge_index):
        if self.sampler is not False:
            edge_index = self.sampler.sample_neighbors(x, edge_index)

        for i in range(self.num_layers - 1):
            x = F.gelu(self.bns[i](self.convs[i](x, edge_index)))

        x = self.convs[self.num_layers - 1](x, edge_index)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()  
        for bn in self.bns:
            bn.reset_parameters()  

def fit(encoder,data, hidden_channels,a,b,c,d,e,alpha,K,pos_treshold, neg_treshold, num_layers,sampler=False,num_epochs=200, lr=0.0001, max_grad_norm=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = data.num_node_features 
    model = GNNClassifier(encoder,in_channels, hidden_channels,num_layers,sampler=sampler).to(device)
    data = data.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping_GNN(patience=20)
    train_losses = []
    
    if contains_self_loops(data.edge_index):
        adjacency = to_dense_adj(data.edge_index).squeeze(0) 
    else:
        edge_index=add_self_loops(data.edge_index)[0]
        adjacency = to_dense_adj(data.edge_index).squeeze(0)
    
    criterion_8 = LabelPropagationLoss(adjacency, alpha=alpha, K=K)
    output_1=data.train_mask

    logger.info("Training GraphSAGE encoder...")
    a_hat_final=adjacency.clone()
    estimated_prior_1 = 0.5
    for epoch in range(num_epochs):
        # First epoch: only data.train_mask nodes are positives, no negatives yet
        if epoch == 0:
            positives = data.train_mask
            negatives = torch.zeros_like(data.train_mask, dtype=torch.bool)  # Empty tensor for negatives
        else:
            # Subsequent epochs: Add reliable positives and negatives based on thresholds
            positive_value = output_1[:, 1]
            positive_indices = (positive_value >= 2*estimated_prior_1) & data.un_train_mask
            reliable_negatives = (positive_value < estimated_prior_1/2) & data.un_train_mask
            
            # Update positives and negatives
            positives = data.train_mask | positive_indices
            negatives = torch.zeros_like(data.train_mask, dtype=torch.bool)| reliable_negatives
        
        # Compute Label Propagation Loss (LPL Loss) using updated positives and negatives
        lpl_loss, A_hat = criterion_8(output_1, positives, negatives,estimated_prior_1)
                
        #Output model
        with torch.no_grad():
            z = model(data.x, dense_to_sparse(A_hat)[0])
            pos_probs = z[data.train_mask, 0]
            un_probs = z[data.un_train_mask, 0]
            estimated_prior = un_probs.min() / pos_probs.min()
            estimated_prior_1 = estimated_prior
            output_1 = z.clone()

        # Select provisionally negative nodes
        _, sorted_indices = torch.sort(un_probs, descending=True)
        threshold_idx = int(estimated_prior * len(un_probs))
        neg_indices = sorted_indices[threshold_idx:].tolist()
        neg_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)  # Initialize a tensor with the same size as train_mask
        neg_mask[neg_indices] = True  # Set the nodes in neg_indices as True

        loss = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        CE_loss = loss(z[data.train_mask | neg_mask], data.y[data.train_mask | neg_mask])
        total_loss=CE_loss+a*lpl_loss

        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        train_losses.append(total_loss.item())
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1:03d}, CE Loss: {CE_loss:.4f}, LPL Loss: {lpl_loss:.4f}, Total Loss: {total_loss:.4f}")

        if early_stopping.step(total_loss.item()):
            logger.info(f"Early stopping at epoch {epoch+1}")
            a_hat_final=A_hat
            break
        
        if epoch == num_epochs - 1:
            a_hat_final=A_hat

    logger.info("Encoder training complete.")
    return model, train_losses, a_hat_final

def evaluate_model(model, data, A_hat, mask, mask_type="validation"):
    model.eval()
    with torch.no_grad():
        # Forward pass using the reduced heterophily adjacency matrix
        logits = model(data.x, dense_to_sparse(A_hat)[0])
        
        # Apply softmax to convert logits to probabilities (we assume binary classification)
        pred_probs = F.softmax(logits, dim=1)

        pred_labels = pred_probs.argmax(dim=1)  # Convert probabilities to class predictions
        
        # Convert tensors to numpy for metric calculations
        true_labels = data.y[mask].cpu().numpy()
        pred_probs_np = pred_probs[mask][:, 1].cpu().numpy()  # Probabilities for the positive class
        pred_labels_np = pred_labels[mask].cpu().numpy()

        # Metrics
        accuracy = (pred_labels[mask] == data.y[mask]).sum().item() / mask.sum().item()
        f1 = f1_score(true_labels, pred_labels_np)
        recall = recall_score(true_labels, pred_labels_np)
        auc = roc_auc_score(true_labels, pred_probs_np)

        # Logging
        logger.info(f"{mask_type.capitalize()} Accuracy: {accuracy:.4f}, "
                    f"{mask_type.capitalize()} F1: {f1:.4f}, "
                    f"{mask_type.capitalize()} Recall: {recall:.4f}, "
                    f"{mask_type.capitalize()} AUC: {auc:.4f}")

    return {"predictions": pred_labels,"accuracy": accuracy,"f1": f1,"recall": recall,"auc": auc}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load parameters for running and evaluating bootstrap PU learning')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer', help='Dataset to be used')
    parser.add_argument('--positive_index', '-c', type=list, default=[3], help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=2, help='Random seed for sampling labeled positive')
    parser.add_argument('--train_pct', '-p', type=float, default=0.1, help='Training positive percentage')
    parser.add_argument('--val_pct', '-v', type=float, default=0.1, help='Validation positive percentage')
    parser.add_argument('--test_pct', '-t', type=float, default=1, help='Test set percentage')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger.info("Loading dataset...")
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=args.positive_index, sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct, half=False)
    print(data)
    data = data.to(device)
    sampler=NNSampler(num_neighbors=30, distance_function="cosine")
    #sampler=False
    hidden_channels=128
    a=0.5
    b=1
    c=1
    d=1
    e=1
    pos_treshold=0.7
    neg_treshold=0.3
    alpha=0.2
    K=25
    num_layers=3
    encoders=[SAGEConv, GCNConv, GATConv]
    logger.info(f"Training with a: {a}, b: {b}, c: {c}, d:{d}, e:{e},alpha {alpha}, K{K},layers {num_layers},pos_treshold {pos_treshold}, neg treshold {neg_treshold} hidden_channels: {hidden_channels}, sampler {sampler}")                                                    
    for encoder in encoders:
        model, train_losses, a_hat = fit(encoder,data, hidden_channels,a,b,c,d,e,alpha,K,pos_treshold, neg_treshold, num_layers,sampler)
        logger.info("Evaluating model on validation set...")
        val_results = evaluate_model(model, data, a_hat, data.val_mask | data.un_val_mask, mask_type="validation")
        logger.info("Evaluating model on test set...")
        test_results = evaluate_model(model, data, a_hat, data.test_mask, mask_type="test")