import pandas as pd
from Encoder import GraphSAGEEncoder, Node2vecEncoder
from Loss_functions import *
from Reliable_negatives import *
import torch
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import to_undirected
from torch.optim import AdamW
import argparse
import warnings
from data_generating import load_dataset, make_pu_dataset
#from data_generating import *

#import data_generating library

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Assuming that `load_dataset`, `make_pu_dataset`, `PU_loss`, and `reliable_negative` functions exist.
contrastive_loss_fn = ContrastiveLossWithDiffusion(temperature=0.5, diffusion_steps=10)
neighbor_similarity_loss_fn=NeighborSimilarityLoss(lambda_reg=0.1)
#Adjusted_A_fn=LabelPropagationLoss()

def train_encoder(model, data, optimizer, beta=1, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()

    # Check for NaN/Inf in input data
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        print("Data contains NaN or Inf values.")

    out = model(data.x, data.edge_index).squeeze()
    c_loss = contrastive_loss_fn(out[data.train_mask], data.y[data.train_mask])
    n_loss = neighbor_similarity_loss_fn(out[data.train_mask], data.edge_index)
    loss=c_loss+beta*n_loss
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item()

# Reliable negative function: Modify according to your reliable negative definition
def reliable_negative(embeddings, labels, threshold=0.5):
    # Example: Returns negative samples with high confidence predictions
    reliable_negatives = (labels == 0) & (embeddings > threshold)
    return reliable_negatives

def evaluate_classifier(classifier, embeddings, labels, mask):
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(embeddings[mask])
        preds = torch.sigmoid(outputs).round()
        
        accuracy = accuracy_score(labels[mask].cpu(), preds.cpu())
        f1 = f1_score(labels[mask].cpu(), preds.cpu())
        recall = recall_score(labels[mask].cpu(), preds.cpu())
    
    return accuracy, f1, recall

def train_classifier(classifier, train_embeddings, train_labels, optimizer):
    classifier.train()
    optimizer.zero_grad()
    
    outputs = classifier(train_embeddings)
    loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), train_labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()

# Main workflow
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
    print(data)


    in_channels = data.num_node_features
    hidden_channels = 16
    out_channels = 1 
    #modify the adj matrix to make it more heterophilic

    # Encoder model (e.g., GCN)
    model = GraphSAGEEncoder(in_channels, hidden_channels, out_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Step 1: Train Encoder
    epochs = 200
    for epoch in range(epochs):
        loss = train_encoder(model, data, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1:03d}, Loss: {loss:.4f}")

    # Step 2: Get embeddings and use reliable negative function
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)  # Extract embeddings from the encoder

    reliable_negative=RocchioMethod()
    reliable_negatives = reliable_negative(embeddings[data.train_mask], (data.y[data.train_mask]==1).nonzero().squeeze(), threshold=0.5)
    # Extract positives (you can adjust this as needed based on your dataset)
    train_positives = (data.y[data.train_mask]==1)

    # Train a classifier using the reliable negatives and positives
    classifier = nn.Sequential(
        nn.Linear(hidden_channels, 1)
    ).to(device)

    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    # Train the classifier
    train_embeddings = embeddings[train_positives | reliable_negatives]
    train_labels = data.y[train_positives | reliable_negatives]
    classifier_loss = train_classifier(classifier, train_embeddings, train_labels, classifier_optimizer)

    # Evaluate classifier
    accuracy, f1, recall = evaluate_classifier(classifier, embeddings, data.y)
    print(f"Classifier Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}")

    # Test the classifier
    test_loss, predictions, accuracy = evaluate_classifier(model, data)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")
