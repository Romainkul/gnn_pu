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
    Implements K-step label propagation on a given adjacency matrix, followed 
    by a negative log-likelihood loss on positive and negative sets.

    Args:
        A_hat (SparseTensor):
            Sparse adjacency matrix with self-loops included.
        alpha (float):
            Weight balancing old distribution vs. neighbor distribution in each step.
        K (int):
            Number of propagation steps.
        pos_weight (float):
            Weight assigned to the positive loss term.
        init_temperature (float):
            Initial value for the temperature parameter (learnable).

    Shape:
        - The adjacency matrix A_hat should be of shape [N, N] (stored as SparseTensor).
        - The input embeddings should be of shape [N, d].

    Example:
        >>> criterion = LabelPropagationLoss(A_hat, alpha=0.5, K=10)
        >>> loss_val, updated_adj, E = criterion(node_embeddings, pos_indices, neg_indices)
    """

    def __init__(
        self,
        A_hat: SparseTensor,
        alpha: float = 0.5,
        K: int = 10,
        pos_weight: float = 1.0,
        init_temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.pos_weight = pos_weight

        # Temperature is a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

        # Buffers for adjacency sums and inverse degrees
        self.register_buffer('row_sums', A_hat.sum(dim=1))
        self.A_hat = A_hat
        self.register_buffer('d_inv', 1.0 / torch.clamp(self.row_sums, min=1e-12))

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_nodes: Union[List[int], torch.LongTensor],
        negative_nodes: Union[List[int], torch.LongTensor]
    ) -> Tuple[torch.Tensor, SparseTensor, torch.Tensor]:
        """
        Forward pass for label propagation and loss computation.

        Args:
            embeddings (torch.Tensor):
                Node embeddings of shape [N, d].
            positive_nodes (Union[List[int], torch.LongTensor]):
                Indices of nodes considered positive.
            negative_nodes (Union[List[int], torch.LongTensor]):
                Indices of nodes considered negative.

        Returns:
            (loss, updated_A_hat, E):
                - loss (torch.Tensor): The scalar loss value.
                - updated_A_hat (SparseTensor): Potentially updated adjacency (same here).
                - E (torch.Tensor): Soft assignment matrix of shape [N, 2], where 
                  E[i] = (prob_neg, prob_pos) for node i.
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Initialize the label distribution E:
        # - Column 0: probability of being negative
        # - Column 1: probability of being positive
        E = torch.zeros((N, 2), device=device)
        E[positive_nodes, 1] = 1.0
        E[negative_nodes, 0] = 1.0

        # Perform K-step label propagation
        for _ in range(self.K):
            # Multiply adjacency by E and row-normalize
            neighbor_E = self.A_hat.matmul(E)               # [N, 2]
            neighbor_E = self.d_inv.view(-1, 1) * neighbor_E

            # Update E: alpha * old + (1 - alpha) * neighbor
            E = self.alpha * E + (1.0 - self.alpha) * neighbor_E

            # Softmax with learnable temperature
            E = F.softmax(E / self.temperature, dim=1)

        # Compute loss for positive and negative sets
        # Clamping to avoid log(0)
        pos_probs = torch.clamp(E[positive_nodes, 1], min=1e-6)
        neg_probs = torch.clamp(E[negative_nodes, 0], min=1e-6)

        pos_loss = -torch.mean(torch.log(pos_probs))
        neg_loss = -torch.mean(torch.log(neg_probs))
        total_loss = self.pos_weight * pos_loss + neg_loss

        updated_A_hat = self.A_hat

        return total_loss, updated_A_hat, E


##############################################################################
# Contrastive Loss
##############################################################################
class ContrastiveLoss(nn.Module):
    """
    Computes a contrastive objective for node embeddings based on the 
    soft assignment matrix E = [prob_neg, prob_pos] for each node.

    Args:
        margin (float):
            Margin for negative samples. If cos_sim > margin, it is penalized.
        chunk_size (int):
            Number of anchor nodes processed at once to handle large N.

    Shape:
        - Input embeddings: [N, d]
        - Soft assignment E: [N, 2]

    Example:
        >>> criterion = ContrastiveLoss(margin=0.5, chunk_size=512)
        >>> loss_val = criterion(node_embeddings, E)
    """

    def __init__(
        self,
        margin: float = 0.5,
        chunk_size: int = 512
    ) -> None:
        super().__init__()
        self.margin = margin
        self.chunk_size = chunk_size

    def forward(
        self,
        embeddings: torch.Tensor,
        E: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive loss.

        Args:
            embeddings (torch.Tensor):
                Node embeddings of shape [N, d].
            E (torch.Tensor):
                Soft assignment matrix of shape [N, 2], where E[i] = (prob_neg, prob_pos).

        Returns:
            torch.Tensor:
                A scalar tensor representing the mean contrastive loss across all 
                valid pairs (i != j).
        """
        device = embeddings.device
        N, d = embeddings.shape

        # Normalize embeddings for cosine similarity
        norm_emb = F.normalize(embeddings, dim=1)

        # Edge case: if only 1 node, no pairs exist
        total_pairs = N * N - N
        if total_pairs < 1:
            return torch.tensor(0.0, device=device)

        total_sum = torch.zeros(1, device=device, dtype=torch.float32)

        # Process anchor nodes in chunks to avoid OOM for large N
        for start_i in range(0, N, self.chunk_size):
            end_i = min(start_i + self.chunk_size, N)
            B = end_i - start_i  # number of anchors in this chunk

            # Compute cos-sim between anchors and all embeddings
            anchor_emb = norm_emb[start_i:end_i]  # shape: [B, d]
            cos_sim = anchor_emb @ norm_emb.t()   # shape: [B, N]

            # Probability of same class for each pair (i, j)
            anchor_E = E[start_i:end_i]           # shape: [B, 2]
            pos_anchor = anchor_E[:, 1].unsqueeze(1)  # [B, 1]
            neg_anchor = anchor_E[:, 0].unsqueeze(1)  # [B, 1]

            pos_all = E[:, 1].unsqueeze(0)            # [1, N]
            neg_all = E[:, 0].unsqueeze(0)            # [1, N]

            # P_same[b, j] = (p_neg_i * p_neg_j) + (p_pos_i * p_pos_j)
            P_same = neg_anchor * neg_all + pos_anchor * pos_all  # [B, N]

            # Positive part: want cos_sim ~ 1 => (cos_sim - 1)^2
            pos_diff = (cos_sim - 1.0).pow(2)

            # Negative part: penalize cos_sim > margin => ReLU(cos_sim - margin)^2
            neg_diff = F.relu(cos_sim - self.margin).pow(2)

            # Combine into a single loss
            loss_matrix = pos_diff * P_same + neg_diff * (1.0 - P_same)

            # Exclude diagonal (i == j) within this chunk
            for b in range(B):
                j_global = start_i + b
                if j_global < N:
                    loss_matrix[b, j_global] = 0.0

            # Accumulate sum over this chunk
            total_sum += loss_matrix.sum()

        # Mean over all valid pairs
        final_loss = total_sum / total_pairs
        return final_loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss with Posterior-Based Pair Sampling.

    This loss function samples a subset of node pairs (instead of using all O(N²)
    pairs) based on the nodes' posterior probabilities. Each node is sampled with a
    probability proportional to its posterior for a given class.

    Args:
        margin (float): Margin for negative pairs.
        num_pairs (int): Number of node pairs to sample for computing the loss.
    """
    def __init__(self, margin: float = 0.5, num_pairs: int = 10000):
        super().__init__()
        self.margin = margin
        self.num_pairs = num_pairs

    def forward(self, embeddings: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss on a sampled set of node pairs.

        Args:
            embeddings (torch.Tensor): Tensor of shape [N, d] containing node embeddings.
            E (torch.Tensor): Tensor of shape [N, 2] where each row represents the
                posterior probabilities (prob_negative, prob_positive) for a node.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        device = embeddings.device
        num_nodes = embeddings.size(0)
        num_pairs = self.num_pairs

        # Normalize the embeddings.
        normalized_embeddings = F.normalize(embeddings, dim=1)

        # --- Sampling Pairs Based on Posterior ---
        # Compute a global class distribution from the posterior probabilities.
        global_class_probs = E.mean(dim=0)  # Shape: [2]
        class_distribution = torch.distributions.Categorical(global_class_probs)
        # For each pair, sample a class (0 or 1).
        sampled_classes = class_distribution.sample((num_pairs,))  # Shape: [num_pairs]

        sampled_pairs_list = []
        eps = 1e-6  # Small constant to prevent zero weights

        # For each class, sample indices for pairs assigned to that class.
        for cls in [0, 1]:
            # Identify pairs assigned to the current class.
            cls_pair_indices = (sampled_classes == cls).nonzero(as_tuple=True)[0]
            num_cls_pairs = cls_pair_indices.numel()
            if num_cls_pairs > 0:
                # Use the posterior for the current class as sampling weights.
                weights = E[:, cls] + eps  # Shape: [N]
                # Sample two indices per pair (with replacement) using the weights.
                pair_indices = torch.multinomial(weights, num_samples=2 * num_cls_pairs, replacement=True)
                pair_indices = pair_indices.view(-1, 2)  # Shape: [num_cls_pairs, 2]
                sampled_pairs_list.append(pair_indices)

        if sampled_pairs_list:
            sampled_pairs = torch.cat(sampled_pairs_list, dim=0)  # Approximately [num_pairs, 2]
        else:
            # Fallback: return an empty tensor if no pairs were sampled (unlikely scenario).
            sampled_pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        # --- Compute Cosine Similarity and Contrastive Loss for Sampled Pairs ---
        # Extract indices for the two nodes in each pair.
        idx_i = sampled_pairs[:, 0]
        idx_j = sampled_pairs[:, 1]

        # Compute the cosine similarity for each sampled pair.
        cosine_similarities = (normalized_embeddings[idx_i] * normalized_embeddings[idx_j]).sum(dim=1)

        # Compute the posterior similarity for each pair:
        # For pair (i, j): P_same = E[i, 0] * E[j, 0] + E[i, 1] * E[j, 1]
        posterior_similarity = E[idx_i, 0] * E[idx_j, 0] + E[idx_i, 1] * E[idx_j, 1]

        # Compute the loss components.
        positive_loss = (cosine_similarities - 1.0) ** 2
        negative_loss = F.relu(cosine_similarities - self.margin) ** 2
        pair_loss = positive_loss * posterior_similarity + negative_loss * (1.0 - posterior_similarity)

        return pair_loss.mean()
