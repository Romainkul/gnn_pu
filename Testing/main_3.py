import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, BatchNorm, GATConv
from torch_geometric.utils import degree, to_undirected, to_dense_adj, contains_self_loops, add_self_loops, dense_to_sparse
import os
import random
import logging
import argparse
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_fscore_support
from data_generating import load_dataset, make_pu_dataset
from Loss_functions import ContrastiveLossWithDiffusionReliableNegatives,ContrastiveSimilarityLoss ,NeighborhoodSimilarityLoss, LearnableDiffusionContrastiveLoss, NeighborhoodConsistencyLoss, AttentionBasedDGI, LabelPropagationLoss, AdjacencyBasedLoss, DistanceCentroid, ClusterCompactnessLoss, TripletMarginLoss, NeighborhoodSmoothnessLoss, ContrastiveLoss
from Reliable_negatives import RocchioMethod, SpyMethod, PositiveEnlargement, KMeansMethod, GenerativePUMethod, OneDNFMethod
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import csv
from Classifier import train_xgboost_classifier, evaluate_xgboost_classifier, train_random_forest_classifier, evaluate_random_forest_classifier, evaluate_logistic_regression_classifier, train_logistic_regression_classifier
from torch_geometric.loader import NeighborLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.parallel as parallel
from torch.optim.lr_scheduler import OneCycleLR

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def edge_dropping(adj_matrix, drop_prob=0.1):
    mask = (torch.rand_like(adj_matrix, dtype=torch.float) > drop_prob).float()
    return adj_matrix * mask

def feature_masking(features, mask_prob=0.1):
    mask = (torch.rand_like(features) > mask_prob).float()
    return features * mask

def node_perturbation(features, noise_std=0.01):
    noise = torch.randn_like(features) * noise_std
    return features + noise

def augment_graph(features, adj_matrix, drop_prob=0.1, mask_prob=0.1, noise_std=0.01):
    aug_adj_matrix = edge_dropping(adj_matrix, drop_prob)
    aug_features = feature_masking(features, mask_prob)
    aug_features = node_perturbation(aug_features, noise_std)
    return aug_features, dense_to_sparse(aug_adj_matrix)[0]

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

        # Get unique nodes and their neighbors
        unique_nodes, inverse_indices = torch.unique(src, return_inverse=True)
        num_nodes = unique_nodes.size(0)
        neighbors_list = [dst[inverse_indices == i] for i in range(num_nodes)]
        max_neighbors = max(len(neigh) for neigh in neighbors_list)

        # Precompute feature matrices for batch distance calculations
        node_feat_matrix = node_features[unique_nodes]  # shape [num_nodes, feature_dim]
        all_neighbors = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long, device=node_features.device)
        mask = torch.zeros_like(all_neighbors, dtype=torch.bool)

        for i, neighbors in enumerate(neighbors_list):
            all_neighbors[i, :len(neighbors)] = neighbors
            mask[i, :len(neighbors)] = True

        # Gather neighbor features
        neighbor_feat_matrix = node_features[all_neighbors]  # shape [num_nodes, max_neighbors, feature_dim]
        node_feat_matrix = node_feat_matrix.unsqueeze(1)  # shape [num_nodes, 1, feature_dim]

        # Compute distances
        if self.distance_function == "euclidean":
            distances = torch.norm(node_feat_matrix - neighbor_feat_matrix, dim=2)  # shape [num_nodes, max_neighbors]
            probabilities = torch.softmax(-distances, dim=1)
        elif self.distance_function == "cosine":
            distances = F.cosine_similarity(node_feat_matrix, neighbor_feat_matrix, dim=2)  # shape [num_nodes, max_neighbors]
            probabilities = torch.softmax(distances, dim=1)

        probabilities[~mask] = 0  # Mask invalid neighbors to prevent sampling

        # Sample neighbors for each node
        sampled_edges = []
        for i in range(num_nodes):
            valid_neighbors = mask[i].nonzero().squeeze(1)  # Indices of valid neighbors
            num_samples = min(self.num_neighbors, len(valid_neighbors))
            sampled_indices = torch.multinomial(probabilities[i, valid_neighbors], num_samples, replacement=True)
            sampled_edges.extend([(unique_nodes[i].item(), all_neighbors[i, idx].item()) for idx in sampled_indices])

        sampled_edge_index = torch.tensor(sampled_edges, dtype=torch.long, device=node_features.device).T  # shape [2, num_sampled_edges]
        return sampled_edge_index

class GraphSAGEEncoder(nn.Module):
    def __init__(self,encoder,in_channels,hidden_channels,out_channels,num_layers=3,dropout=0.5,sampler=False,aggregation="add"):

        super(GraphSAGEEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.sampler = sampler
        self.aggregation = aggregation
        self.encoder=encoder
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
    
        # Create a list of layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer (input to first hidden)
        self.convs.append(encoder(in_channels, hidden_channels, aggr=aggregation))
        self.bns.append(BatchNorm(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):  # num_layers - 2 because of input-to-hidden and last layer
            self.convs.append(encoder(hidden_channels, hidden_channels, aggr=aggregation))
            self.bns.append(BatchNorm(hidden_channels))
            self.dropout.append(nn.Dropout(dropout))

        # Final layer (hidden to output)
        self.convs.append(encoder(hidden_channels, self.out_channels, aggr=aggregation))

    def forward(self, x, edge_index):
        if self.sampler is not False:
            edge_index = self.sampler.sample_neighbors(x, edge_index)

        for i in range(self.num_layers - 1):
            x = F.gelu(self.bns[i](self.convs[i](x, edge_index)))
            x= self.dropouts[i](x)

        x = self.convs[self.num_layers - 1](x, edge_index)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()  
        for bn in self.bns:
            bn.reset_parameters()  
    
    def dynamic_weight_update(self, *losses):
        with torch.no_grad():
            loss_tensor = torch.tensor([loss.item() for loss in losses])
            inverse_magnitudes = 1.0 / (loss_tensor + 1e-8) 
            normalized_weights = inverse_magnitudes / inverse_magnitudes.sum()
        return normalized_weights.tolist()

    def fit(self, data, a, b, c, d, e, alpha, K, pos_treshold, neg_treshold, beta=0.7, num_epochs=2000, lr=0.001, max_grad_norm=1.0, weight_decay=1e-4):
        self.train()
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler =  OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=2000)
        #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True, min_lr=1e-7)
        early_stopping = EarlyStopping_GNN(patience=50)
        train_losses = []
        val_losses = []
        criterion_9 = AdjacencyBasedLoss()
        criterion_10 = ContrastiveSimilarityLoss()
        criterion_11 = DistanceCentroid()
        criterion_cluster = ClusterCompactnessLoss()
        criterion_triplet = TripletMarginLoss(margin=1.0, p=2)
        criterion_smoothness = NeighborhoodSmoothnessLoss(lambda_smooth=0.2)
        #contrastive = ContrastiveLoss()
            
        if contains_self_loops(data.edge_index):
            adjacency = to_dense_adj(data.edge_index).squeeze(0)
        else:
            edge_index = add_self_loops(data.edge_index)[0]
            adjacency = to_dense_adj(data.edge_index).squeeze(0)
        
        criterion_8 = LabelPropagationLoss(adjacency, alpha=alpha, K=K)
        output_1 = torch.zeros((data.train_mask.size(0), self.out_channels), dtype=torch.float)
        for epoch in range(num_epochs):
            self.train() 
            
            if epoch <=1:
                reliable_positives = data.train_mask
                reliable_negatives = torch.zeros_like(data.train_mask, dtype=torch.bool)  # Empty tensor for negatives
            else:
                reliable_negatives, reliable_positives = RocchioMethod(neg_treshold)(output_1, data.y[data.train_mask],pos_treshold)
                reliable_negatives = reliable_negatives 
                reliable_positives = reliable_positives | data.train_mask

            optimizer.zero_grad()
            
            if  epoch <=1:
                A_hat = adjacency
            
            embeddings = self(data.x, dense_to_sparse(A_hat)[0])
            #aug_features_1, aug_edge_index_1 = augment_graph(data.x, A_hat, drop_prob=0.1, mask_prob=0.1, noise_std=0.01)
            #embeddings_aug_1 = self(aug_features_1, aug_edge_index_1)
            output_1 = embeddings
            
            if  epoch <=1:
                lpl_loss = torch.tensor(200, dtype=float)                
            else:
                lpl_loss, A_hat = criterion_8(output_1, reliable_positives, reliable_negatives)

            distance_loss = criterion_11(embeddings, reliable_positives, reliable_negatives) if epoch >= 2 else torch.tensor(200.0)
            cluster_loss = criterion_cluster(embeddings, reliable_positives, reliable_negatives) if epoch >= 2 else torch.tensor(200.0)
            triplet_loss = criterion_triplet(embeddings, reliable_positives, reliable_negatives) if epoch >= 2 else torch.tensor(200.0)
            smoothness_loss = criterion_smoothness(embeddings, dense_to_sparse(A_hat)[0]) if epoch >= 2 else torch.tensor(200.0)
            #contrastive_loss = contrastive(embeddings_aug_1, embeddings)
            homo_loss, hetero_loss = criterion_9(data, embeddings, A_hat)
            
            w = self.dynamic_weight_update(lpl_loss, distance_loss, homo_loss, hetero_loss, cluster_loss, triplet_loss, smoothness_loss)#, contrastive_loss)

            loss = (a*w[0] * lpl_loss + b*w[1] * distance_loss + c*w[2] * homo_loss + d*w[3] * hetero_loss + w[4] * cluster_loss + w[5] * triplet_loss + w[6] * smoothness_loss)#+ w[7]*contrastive_loss)

            loss.backward() 
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch+1:03d}, Total Loss: {loss:.4f}, LPL Loss: {lpl_loss:.4f}, Distance Loss: {distance_loss:.4f}, Homophily Loss: {homo_loss:.4f}, Heterophily loss: {hetero_loss}, Cluster loss: {cluster_loss}, Triplet loss: {triplet_loss}, smoothness loss: {smoothness_loss}")

            scheduler.step(loss.item())

            if early_stopping.step(loss.item()):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return train_losses

    @torch.no_grad()
    def evaluate_val(embeddings, *loss_classes):
        return embeddings
    
def main(encoder,data, a,b,c,d,e,alpha,K,pos_treshold,neg_treshold, hidden_channels=128, out_channels=64,num_layers=3,sampler=False, aggregation="add"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = data.num_node_features
    model = GraphSAGEEncoder(encoder,in_channels, hidden_channels, out_channels,num_layers,sampler=sampler, aggregation=aggregation).to(device)
    data = data.to(device)

    logger.info("Training GraphSAGE encoder...")
    train_losses = model.fit(data,a=a,b=b,c=c,d=d,e=e,alpha=alpha,K=K,pos_treshold=pos_treshold,neg_treshold=neg_treshold)  
    logger.info("Encoder training complete.")
    return model, train_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load parameters for running and evaluating bootstrap PU learning')
    parser.add_argument('--dataset', '-d', type=str, default='citeseer', help='Dataset to be used')
    parser.add_argument('--positive_index', '-c', type=list, default=[3], help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=2, help='Random seed for sampling labeled positive')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2, help='Training positive percentage')
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

    with open('GraphSAGE_layers_tresh_layers_loss_23_11.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alpha','K','layers','pos_treshold','neg_treshold','hidden_channels', 'out_channels','aggregation','sampler', 'accuracy', 'f1', 'recall', 'auc'])
        encoder=SAGEConv
        a=0.9
        b=0.9
        c=0.4
        d=0.2
        e=0.9
        sampler=False
        for alpha in [0.2,0.4,0.6,0.8]:
            for K in [10,20,30,40]:
                for layers in [2,3,4]:
                    for hidden_channels in [128,256,512]:
                        for out_channels in [32,64,128]:
                            for pos_treshold in [0.6,0.7,0.8,0.9]:
                                for neg_treshold in [0.1,0.2,0.3,0.4]:
                                    for aggregation in ["add","mean", "max"]:
                                        for sampler in [False]:
                                            e=0
                                            logger.info(f"Training with a: {a}, b: {b}, c: {c}, e:{e},alpha {alpha}, K{K},layers {layers},pos_treshold {pos_treshold}, neg treshold {neg_treshold} hidden_channels: {hidden_channels}, out_channels: {out_channels}, sampler {sampler}")
                                                                
                                            # Train the model
                                            model, train_losses = main(encoder,data, a=a, b=b, c=c, d=d,e=e, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=layers,alpha=alpha,K=K, pos_treshold=pos_treshold,neg_treshold=neg_treshold,sampler=sampler, aggregation=aggregation)
                                                                
                                            # Get embeddings
                                            embeddings = model(data.x, data.edge_index)
                                            train_positives = data.train_mask

                                            # Iterate over each method
                                            #methods = [RocchioMethod(0.2),RocchioMethod(0.3),RocchioMethod(0.4),RocchioMethod(0.5),RocchioMethod(0.6),RocchioMethod(0.7),RocchioMethod(0.8),RocchioMethod(0.9),RocchioMethod(0.1)] 
                                            methods=[KMeansMethod()]
                                            for method in methods:
                                                logger.info(f"Running method: {method}")
                                                if method in (SpyMethod(),SpyMethod(spy_ratio=0.1),SpyMethod(spy_ratio=0.2)):
                                                    reliable_negatives = method(embeddings[data.train_mask | data.un_train_mask], data.y[data.train_mask | data.un_train_mask])
                                                    reliable_positives = data.train_mask
                                                else:
                                                    reliable_negatives, reliable_positives = method(embeddings, train_positives)
                                                    reliable_negatives = reliable_negatives & data.un_train_mask
                                                    reliable_positives = (reliable_positives & data.un_train_mask) | data.train_mask

                                                train_embeddings = embeddings[reliable_positives | reliable_negatives]
                                                train_labels = data.y[reliable_positives | reliable_negatives]

                                                val_embeddings = embeddings[data.val_mask | data.un_val_mask]
                                                val_labels = data.y[data.val_mask | data.un_val_mask]

                                                # Training and evaluating Logistic Regression (L2 Regularization)
                                                lr_model, tresh_lr = train_logistic_regression_classifier(train_embeddings, train_labels, val_embeddings, val_labels)
                                                accuracy, f1, recall, auc = evaluate_logistic_regression_classifier(lr_model, embeddings, data.y, data.test_mask, tresh_lr)
                                                logger.info(f"Test Metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
                                                if f1>0.65:
                                                    logger.info("Youhouuu getting closer")
                                                logger.info("Method Complete")                                                            
                                                writer.writerow([alpha, K, layers, pos_treshold, neg_treshold, hidden_channels, out_channels,aggregation,sampler, accuracy, f1, recall, auc])

                                                            
    """with open('GraphSAGE_3_layers_cd_loss.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['temperature', 'num_diffusion_steps', 'alpha', 'neg_samples', 'hidden_channels', 'out_channels', 'method', 'accuracy', 'f1', 'recall', 'auc'])

        for temperature in [0.05, 0.1, 0.2]:
            for num_diffusion_steps in [3, 5, 7]:
                for alpha in [0.2, 0.4, 0.6]:
                    for neg_samples in [10, 20, 30]:
                        for hidden_channels in [32, 64, 128]:
                            for out_channels in [32, 64, 128]:
                                logger.info(f"Training with temperature: {temperature}, num_diffusion_steps: {num_diffusion_steps}, alpha: {alpha}, neg_samples: {neg_samples}, hidden_channels: {hidden_channels}, out_channels: {out_channels}")
                                model, train_losses = main(data, temperature=temperature, num_diffusion_steps=num_diffusion_steps, alpha=alpha, neg_samples=neg_samples)

                                embeddings = model(data.x, data.edge_index)
                                train_positives = (data.y[data.train_mask | data.un_train_mask] == 1)
                                methods = [RocchioMethod, SpyMethod, PositiveEnlargement, KMeansMethod, GenerativePUMethod, OneDNFMethod]
                                for method in methods:
                                    logger.info(f"Running method: {method.__name__}")
                                    reliable_negative = method()
                                    reliable_negatives = reliable_negative(embeddings[data.train_mask | data.un_train_mask], train_positives)

                                    train_embeddings = embeddings[data.train_mask | data.un_train_mask]
                                    train_embeddings = train_embeddings[train_positives | reliable_negatives]
                                    train_labels = data.y[data.train_mask | data.un_train_mask]
                                    train_labels = train_labels[train_positives | reliable_negatives]

                                    val_embeddings = embeddings[data.val_mask | data.un_val_mask]
                                    val_labels = data.y[data.val_mask | data.un_val_mask]

                                    bst = train_xgboost_classifier(train_embeddings, train_labels, val_embeddings, val_labels, num_epochs=100, patience=10)

                                    accuracy, f1, recall, auc = evaluate_xgboost_classifier(bst, embeddings, data.y, data.test_mask)
                                    logger.info(f"Test Metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

                                    # Write the results to the CSV file
                                    writer.writerow([temperature, num_diffusion_steps, alpha, neg_samples, hidden_channels, out_channels, method.__name__, accuracy, f1, recall, auc])
                                    logger.info("Method Complete")"""
    