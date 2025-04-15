import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Tuple, List, Union


##############################################################################
# Sparse Label Propagation
##############################################################################
class LabelPropagationLoss(nn.Module):
    """
    Performs K-step label propagation on a subgraph adjacency,
    with a learnable alpha in (0, 1).
    Returns both the scalar loss and the final label distribution E.
    """
    def __init__(self, K=5,alpha=0.5):
        super().__init__()
        # raw_alpha is a learnable parameter that we map to (0,1) via sigmoid
        self.raw_alpha = nn.Parameter(torch.tensor(alpha))
        self.K = K

    def forward(self, embeddings, sub_A: SparseTensor, sub_pos, sub_neg):
        """
        embeddings: [num_sub_nodes, embed_dim]
        sub_A:      subgraph adjacency (SparseTensor)
        sub_pos:    list or tensor of node indices (local to subgraph) that are positive
        sub_neg:    list or tensor of node indices (local to subgraph) that are negative

        Returns
        -------
        lp_loss: torch.Tensor (scalar)
        E: torch.Tensor of shape [num_sub_nodes, 2] -- final label distribution
        """
        device = embeddings.device
        alpha = torch.sigmoid(self.raw_alpha)

        num_nodes_sub = embeddings.size(0)
        E = torch.zeros((num_nodes_sub, 2), device=device)

        # One-hot for positives vs negatives
        E[sub_pos, 1] = 1.0
        E[sub_neg, 0] = 1.0

        # Row normalization
        row_sum = sub_A.sum(dim=1)  # shape [num_sub_nodes]
        d_inv = 1.0 / torch.clamp(row_sum, min=1e-12)
        # K-step label propagation
        for _ in range(self.K):
            neighbor_E = sub_A.matmul(E)
            neighbor_E = d_inv.view(-1, 1) * neighbor_E
            E = alpha * E + (1.0 - alpha) * neighbor_E

        # Compute negative log-likelihood loss
        eps = 1e-6
        pos_probs = torch.clamp(E[sub_pos, 1], min=eps)
        neg_probs = torch.clamp(E[sub_neg, 0], min=eps)

        pos_loss = -torch.log(pos_probs).mean() if len(sub_pos) > 0 else 0.0
        neg_loss = -torch.log(neg_probs).mean() if len(sub_neg) > 0 else 0.0

        lp_loss = pos_loss + neg_loss
        return lp_loss, E
    
##############################################################################
# Contrastive Loss
##############################################################################
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss with Posterior-Based Pair Sampling.

    This loss function samples a subset of node pairs (instead of using all O(N²)
    pairs) based on the nodes' posterior probabilities. Each node is sampled with a
    probability proportional to its posterior for a given class.

    Args:
        margin (float): Initial margin for negative pairs. This value is learnable.
        num_pairs (int): Number of node pairs to sample for computing the loss.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        # Use a raw margin parameter and map it to a positive value via softplus.
        self.raw_margin = nn.Parameter(torch.tensor(margin))

    def forward(self, embeddings: torch.Tensor, E: torch.Tensor, num_pairs: int) -> torch.Tensor:
        device = embeddings.device
        num_nodes = embeddings.size(0)
        #num_pairs = self.num_pairs

        # Normalize the embeddings.
        normalized_embeddings = F.normalize(embeddings, dim=1)

        # --- Sampling Pairs Based on Posterior ---
        global_class_probs = E.mean(dim=0)  # Shape: [2]
        class_distribution = torch.distributions.Categorical(global_class_probs)
        sampled_classes = class_distribution.sample((num_pairs,))  # Shape: [num_pairs]

        sampled_pairs_list = []
        eps = 1e-6  # Small constant to prevent zero weights

        for cls in [0, 1]:
            cls_pair_indices = (sampled_classes == cls).nonzero(as_tuple=True)[0]
            num_cls_pairs = cls_pair_indices.numel()
            if num_cls_pairs > 0:
                weights = E[:, cls] + eps  # Sampling weights based on posterior.
                pair_indices = torch.multinomial(weights, num_samples=2 * num_cls_pairs, replacement=True)
                pair_indices = pair_indices.view(-1, 2)
                sampled_pairs_list.append(pair_indices)

        if sampled_pairs_list:
            sampled_pairs = torch.cat(sampled_pairs_list, dim=0)
        else:
            sampled_pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        idx_i = sampled_pairs[:, 0]
        idx_j = sampled_pairs[:, 1]

        # Compute cosine similarities between pairs.
        cosine_similarities = (normalized_embeddings[idx_i] * normalized_embeddings[idx_j]).sum(dim=1)

        # Compute posterior similarity for each pair.
        posterior_similarity = E[idx_i, 0] * E[idx_j, 0] + E[idx_i, 1] * E[idx_j, 1]

        # Map raw_margin to a positive effective margin.
        effective_margin = F.softplus(self.raw_margin)

        positive_loss = (cosine_similarities - 1.0) ** 2
        negative_loss = F.relu(cosine_similarities - effective_margin) ** 2

        # Combine losses, weighting by the posterior similarity.
        pair_loss = positive_loss * posterior_similarity + negative_loss * (1.0 - posterior_similarity)
        return pair_loss.mean()