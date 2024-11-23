from torch_geometric.nn.models import DeepGraphInfomax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import degree,to_dense_adj

#Can adapt with another gnn model
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class ContrastiveLossWithDiffusion(nn.Module):
    def __init__(self, temperature=0.07, num_diffusion_steps=10, alpha=0.1, neg_samples=256):
        super(ContrastiveLossWithDiffusion, self).__init__()
        self.temperature = temperature
        self.num_diffusion_steps = num_diffusion_steps
        self.alpha = alpha
        self.neg_samples = neg_samples
        
    def compute_diffusion_matrix(self, edge_index, num_nodes,adj):
        # Convert to dense adjacency matrix
        #adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        # Compute degree matrix
        deg = adj.sum(dim=1)
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0
        # Compute normalized adjacency matrix
        norm_adj = torch.mm(torch.diag(deg_inv), 
                          torch.mm(adj, torch.diag(deg_inv)))
        # Initialize diffusion matrix
        diff_matrix = torch.eye(num_nodes).to(edge_index.device)
        
        # Compute diffusion matrix through power iteration
        for _ in range(self.num_diffusion_steps):
            diff_matrix = (1 - self.alpha) * torch.mm(norm_adj, diff_matrix) + \
                         self.alpha * torch.eye(num_nodes).to(edge_index.device)
        
        return diff_matrix
    
    def generate_positive_pairs(self, z, diff_matrix):
        # Generate diffused embeddings
        z_diffused = torch.mm(diff_matrix, z)
        
        return z, z_diffused
    
    def sample_negative_pairs(self, z, batch_size):
        idx = torch.randint(0, batch_size, (batch_size * self.neg_samples,))
        return z[idx]
    
    def forward(self, z, edge_index, adj,batch=None):
        batch_size = z.size(0)
        
        # Compute diffusion matrix
        diff_matrix = self.compute_diffusion_matrix(edge_index, batch_size,adj)
        
        # Generate positive pairs
        anchor, positive = self.generate_positive_pairs(z, diff_matrix)
        
        # Sample negative pairs
        negative = self.sample_negative_pairs(z, batch_size)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # Compute negative similarity
        neg_sim = torch.exp(torch.mm(anchor, negative.t()) / self.temperature)
        
        # Compute final loss
        loss = -torch.log(
            torch.exp(pos_sim) / 
            (torch.exp(pos_sim) + neg_sim.sum(dim=1))
        ).mean()
        
        return loss
    
    def get_positive_pairs(self, z, edge_index):
        diff_matrix = self.compute_diffusion_matrix(edge_index, z.size(0))
        return self.generate_positive_pairs(z, diff_matrix)
class ContrastiveLossWithDiffusionReliableNegatives(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.1, num_diffusion_steps=10, neg_samples=256):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.num_diffusion_steps = num_diffusion_steps
        self.neg_samples = neg_samples

    def compute_diffusion_matrix(self, adj):
        num_nodes = adj.size(0)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        diff_matrix = torch.eye(num_nodes, device=adj.device)
        for _ in range(self.num_diffusion_steps):
            diff_matrix = (1 - self.alpha) * torch.mm(norm_adj, diff_matrix) + self.alpha * torch.eye(num_nodes, device=adj.device)
        return diff_matrix

    def generate_positive_pairs(self, z, diff_matrix, positive_nodes):
        z_diffused = torch.mm(diff_matrix, z)
        return z[positive_nodes], z_diffused[positive_nodes]

    def sample_negative_pairs(self, z, reliable_negatives):
        neg_indices = reliable_negatives[torch.randint(0, len(reliable_negatives), (self.neg_samples,))]
        return z[neg_indices]

    def forward(self, z, adj, positive_nodes, reliable_negatives):
        diff_matrix = self.compute_diffusion_matrix(adj)
        anchor, positive = self.generate_positive_pairs(z, diff_matrix, positive_nodes)
        negative = self.sample_negative_pairs(z, reliable_negatives)
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        pos_sim = (anchor * positive).sum(dim=1) / self.temperature
        neg_sim = torch.exp(torch.mm(anchor, negative.t()) / self.temperature).sum(dim=1)
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim)).mean()
        return loss

class DeepGraphInfomaxLoss(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(DeepGraphInfomaxLoss, self).__init__()
        self.encoder = GNNEncoder(hidden_dim)
        self.model = DeepGraphInfomax(
            hidden_channels=hidden_dim,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=self.corruption_function
        )

    def corruption_function(self, x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index  # Shuffle node features as corruption

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.model(x, edge_index)
        dgi_loss = self.model.loss(pos_z, neg_z, summary)
        return dgi_loss
    
class NeighborhoodSimilarityLoss(torch.nn.Module):
    def __init__(self, lambda_reg=0.2):
        super(NeighborhoodSimilarityLoss, self).__init__()
        self.lambda_reg = lambda_reg
    def forward(self, embedding, edge_index):
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_embeddings = embedding[source_nodes]
        target_embeddings = embedding[target_nodes]

        # Cosine similarity between the embeddings of connected nodes
        cosine_sim = F.cosine_similarity(source_embeddings, target_embeddings, dim=1)

        # Loss: Encourage embeddings of connected nodes to be similar (maximize cosine similarity)
        loss = torch.mean(1 - cosine_sim)  # Penalize dissimilar pairs
        similarity_loss = F.mse_loss(embedding[source_nodes], embedding[target_nodes])
        loss = self.lambda_reg * similarity_loss + loss
        return loss

class ContrastiveLoss(nn.Module):  #Look into debiased contrastive loss
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        sim_matrix = torch.mm(embeddings1, embeddings2.T) / self.temperature

        positives = torch.diag(sim_matrix)
        negatives = sim_matrix.sum(dim=1) - positives
        loss = -torch.mean(torch.log((positives + 1e-8) / (negatives + 1e-8)))
                

        return loss

class LearnableDiffusionContrastiveLoss(nn.Module):
    def __init__(self, in_channels, hidden_channels, temperature=0.2, num_diffusion_steps=10, neg_samples=10):
        super(LearnableDiffusionContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_diffusion_steps = num_diffusion_steps
        self.neg_samples = neg_samples
        
        # Learnable diffusion matrix
        self.diffusion_matrix = nn.Parameter(torch.eye(in_channels, in_channels))

    def forward(self, z, edge_index, reliable_negatives, batch=None):
        batch_size = z.size(0)
        
        # Initialize diffusion matrix
        diff_matrix = self.diffusion_matrix

        # Perform diffusion steps
        for _ in range(self.num_diffusion_steps):
            # Compute degree matrix
            D = torch.diag(torch.sum(diff_matrix, dim=1))
            
            # Normalize (random walk normalization)
            diff_matrix = torch.mm(torch.inverse(D), diff_matrix)

        # Final normalization (optional if already done in the loop)
        D = torch.diag(torch.sum(diff_matrix, dim=1))
        epsilon = 1e-6  # To prevent division by zero
        D += epsilon * torch.eye(D.size(0))
        normalized_diff_matrix = torch.mm(torch.linalg.inv(D), diff_matrix)
        
        # Generate positive pairs
        anchor, positive = self.generate_positive_pairs(z, diff_matrix)
        
        # Sample negative pairs
        negative = self.sample_negative_pairs(z, reliable_negatives)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # Compute negative similarity
        neg_sim = torch.exp(torch.mm(anchor, negative.t()) / self.temperature)
        
        # Compute final loss
        loss = -torch.log(
            torch.exp(pos_sim) / 
            (torch.exp(pos_sim) + neg_sim.sum(dim=1))
        ).mean()
        
        return loss

    def generate_positive_pairs(self, z, diff_matrix):
        z_diffused = torch.mm(diff_matrix, z)
        return z, z_diffused

    def sample_negative_pairs(self, z, reliable_negatives):
        return z[torch.randint(0, len(reliable_negatives), (self.neg_samples,))]
    
class AttentionBasedDGI(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(AttentionBasedDGI, self).__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4)
        self.summary = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        pos_z, neg_z = self.encoder(x, edge_index)
        
        # Compute global summary using attention
        attn_output, _ = self.attention(pos_z, pos_z, pos_z)
        global_summary = self.summary(attn_output.mean(dim=1))
        
        dgi_loss = self.loss(pos_z, neg_z, global_summary)
        return dgi_loss
    
    def loss(self, pos_z, neg_z, summary):
        pos_loss = -F.logsigmoid((pos_z * summary).sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(-(neg_z * summary).sum(dim=1)).mean()
        return pos_loss + neg_loss

class NeighborhoodConsistencyLoss(nn.Module):
    def __init__(self, in_channels, hidden_channels, lambda_reg=0.1):
        super(NeighborhoodConsistencyLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, embeddings, edge_index):        
        # Calculate regularization loss between neighbors
        src, dst = edge_index
        neighbor_embeddings = embeddings[dst]
        attention_weights = F.softmax(torch.bmm(embeddings[src].unsqueeze(1), neighbor_embeddings.unsqueeze(2)), dim=1).squeeze()
        weighted_neighbor_embeddings = attention_weights.unsqueeze(1) * neighbor_embeddings
        weighted_sum = torch.sum(weighted_neighbor_embeddings, dim=0)  # Summing over neighbors
        similarity_loss = F.mse_loss(embeddings[src], weighted_sum)
        return self.lambda_reg * similarity_loss


class ContrastiveSimilarityLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveSimilarityLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, positive: torch.Tensor, reliable_negatives:torch.Tensor ) -> torch.Tensor:
        pos_embeddings = embeddings[positive]
        neg_embeddings = embeddings[reliable_negatives]
        # Compute similarity scores
        pos_sim = torch.exp(torch.mm(pos_embeddings, pos_embeddings.T) / self.temperature)  # (N, N)
        if reliable_negatives.size(0)==0:
            neg_sim=10
        else:
            neg_sim = torch.exp(torch.mm(pos_embeddings, neg_embeddings.T) / self.temperature)  # (N, M)

        # Compute the loss
        denominator = pos_sim + neg_sim.sum(dim=1, keepdim=True)  # Summing negative similarities along rows
        loss = -torch.log(pos_sim / denominator).mean()

        return loss
 

class LabelPropagationLoss(nn.Module):
    def __init__(self, adjacency_matrix: torch.Tensor, alpha: float = 0.5, K: int = 10):
        super(LabelPropagationLoss, self).__init__()
        
        self.A = adjacency_matrix + torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)
        self.alpha = alpha
        self.K = K
        
        # Initialize learnable edge mask
        self.M = nn.Parameter(torch.ones_like(self.A))
        
        # Compute and store degree matrix inverse
        self.D_inv = self._compute_degree_matrix_inverse()
    
    def _compute_degree_matrix_inverse(self) -> torch.Tensor:
        degrees = torch.sum(self.A, dim=1)
        D = torch.diag(degrees)
        epsilon = 1e-8  # Small value to avoid singularity
        D += torch.eye(D.shape[0], device=self.A.device) * epsilon  # Add epsilon to the diagonal
        return torch.inverse(D)

    def _initialize_posteriors(self, num_nodes: int, positive_nodes: torch.Tensor, reliable_negatives: torch.Tensor) -> torch.Tensor:
        # Initialize a posterior tensor where [0] = negative class, [1] = positive class
        E = torch.zeros((num_nodes, 2), device=positive_nodes.device)
        
        # Default: All nodes start as negative
        E[:, 0] = 0.5
        E[:, 1] = 0.5
        
        # Positive and reliable negative nodes
        E[positive_nodes, 0] = 0.0
        E[positive_nodes, 1] = 1.0
        E[reliable_negatives, 0] = 1.0
        E[reliable_negatives, 1] = 0.0
        
        return E
    
    def forward(self, node_embeddings: torch.Tensor, positive_nodes: torch.Tensor, reliable_negatives: torch.Tensor) -> torch.Tensor:
        num_nodes = node_embeddings.shape[0]
        
        # Initialize posterior probabilities
        E = self._initialize_posteriors(num_nodes, positive_nodes, reliable_negatives)
        
        # Label Propagation
        for _ in range(self.K):
            A_hat = self.M * self.A
            E = self.alpha * E.detach() + (1 - self.alpha) * torch.mm(self.D_inv, A_hat @ E)
        
        # Compute LPL Loss
        positive_prob = E[positive_nodes, 1]
        negative_prob = E[reliable_negatives, 0]
        
        positive_loss = -torch.mean(torch.log(positive_prob + 1e-10))
        negative_loss = -torch.mean(torch.log(negative_prob + 1e-10))
        
        if negative_loss.isnan():
            negative_loss = torch.tensor(1.75, device=negative_loss.device)

        # Total loss
        LPL_loss = positive_loss + negative_loss

        # Optional regularization on M
        reg_loss = torch.mean(self.M ** 2)
        total_loss = LPL_loss + 0.02 * reg_loss 
        
        return total_loss, A_hat
    
class AdjacencyBasedLoss(nn.Module):
    def __init__(self, sim_function='cosine'):
        super(AdjacencyBasedLoss, self).__init__()
        self.sim_function = sim_function

    def forward(self,data, Z, A_hat):
        # Normalize A_hat (optional for stability)
        A_normalized = A_hat.float()
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A_normalized.sum(dim=1) + 1e-10))
        A_normalized = D_inv_sqrt @ A_normalized @ D_inv_sqrt

        # Compute similarity matrix
        if self.sim_function == 'cosine':
            Z_norm = F.normalize(Z, p=2, dim=1)
            similarity = torch.mm(Z_norm, Z_norm.T)  # Cosine similarity
        elif self.sim_function == 'dot':
            similarity = torch.mm(Z, Z.T)  # Dot-product similarity
        else:
            raise ValueError("sim_function must be 'cosine' or 'dot'")

        # Homophilic loss
        homo_loss = -torch.sum(A_normalized * similarity)

        # Heterophilic loss
        hetero_loss = torch.sum((1 - A_normalized) * similarity)

        return homo_loss, hetero_loss
   
class DistanceCentroid(nn.Module):
    def __init__(self):
        super(DistanceCentroid, self).__init__()

    def forward(self, embeddings, positive_nodes, negative_nodes):
        # Get the centroid of the positive nodes
        pos_centroid = embeddings[positive_nodes].mean(dim=0)

        # Compute the loss for positives
        pos_loss = 2 - 2 * F.cosine_similarity(embeddings[positive_nodes], pos_centroid.unsqueeze(0), dim=-1).mean()

        # If negatives are provided, compute the negative loss
        if negative_nodes is not None and len(negative_nodes) > 0:
            neg_centroid = embeddings[negative_nodes].mean(dim=0)
            neg_loss = 2 - 2 * F.cosine_similarity(embeddings[negative_nodes], neg_centroid.unsqueeze(0), dim=-1).mean()
        else:
            neg_loss = 0  # No negative nodes, set loss to 0

        # Combine the losses (equal weighting for simplicity, can adjust if needed)
        total_loss = (pos_loss + neg_loss) / 2 if neg_loss != 0 else pos_loss

        return total_loss
class ClusterCompactnessLoss(nn.Module):
    def __init__(self):
        super(ClusterCompactnessLoss, self).__init__()

    def forward(self, embeddings, positive_nodes, negative_nodes):
        pos_centroid = embeddings[positive_nodes].mean(dim=0)
        neg_centroid = embeddings[negative_nodes].mean(dim=0)

        pos_variance = ((embeddings[positive_nodes] - pos_centroid)**2).mean()
        neg_variance = ((embeddings[negative_nodes] - neg_centroid)**2).mean()

        return pos_variance + neg_variance

class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p

    def forward(self, embeddings, positive_nodes, negative_nodes):
        positive_embeddings = embeddings[positive_nodes]
        negative_embeddings = embeddings[negative_nodes]

        pos_dists = torch.cdist(positive_embeddings, positive_embeddings, p=self.p)
        neg_dists = torch.cdist(positive_embeddings, negative_embeddings, p=self.p)

        triplet_loss = F.relu(self.margin + pos_dists.mean() - neg_dists.mean())
        return triplet_loss
    
class NeighborhoodSmoothnessLoss(nn.Module): 
    def __init__(self, lambda_smooth=0.1):
        super(NeighborhoodSmoothnessLoss, self).__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, embeddings, edge_index):
        row, col = edge_index
        neighbor_diffs = (embeddings[row] - embeddings[col]).pow(2).mean()
        return self.lambda_smooth * neighbor_diffs
    
class TwoHopNeighborLoss(nn.Module):
    def __init__(self, lambda_two_hop=1.0, sim_function='cosine'):
        super(TwoHopNeighborLoss, self).__init__()
        self.lambda_two_hop = lambda_two_hop
        self.sim_function = sim_function

    def get_two_hop_neighbors(A):
        A2 = torch.matmul(A, A)
        return A2

    def forward(self, embeddings, adjacency_matrix):
        # Compute 2-hop neighbors using the adjacency matrix squared (A^2)
        A2 = self.get_two_hop_neighbors(adjacency_matrix)

        # Normalize embeddings (optional)
        Z_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity for 2-hop neighbors
        if self.sim_function == 'cosine':
            similarity = torch.mm(Z_norm, Z_norm.T)
        else:
            raise ValueError("sim_function must be 'cosine'")

        # Get the pairs of 2-hop neighbors (A^2 > 0 indicates 2-hop neighbors)
        two_hop_mask = (A2 > 0).float()

        # Apply the mask to compute similarity for 2-hop neighbors only
        two_hop_similarity = similarity * two_hop_mask

        # For 2-hop neighbors, we want the similarity to be low (enforce distinct embeddings)
        two_hop_loss = torch.sum(two_hop_similarity)

        # Apply the regularization factor (lambda_two_hop)
        return self.lambda_two_hop * two_hop_loss
