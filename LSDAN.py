# Check the long short
# Check the nnPU
# Create a datsaet generating function (binary labels) with occulting of a certain percentage of the positive class and all negative class
# Set up an evaluation function
import torch
import torch.nn as nn
import torch.nn.functional as F
from PU_Loss import PULoss

# Short-Distance Attention Mechanism
class ShortDistanceAttention(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        self.W = torch.nn.Linear(in_features, out_features, bias=False)  # W^(1)
        self.r = torch.nn.Parameter(torch.randn(out_features * 2, 1))  # Learnable attention vector r
        self.leakyrelu = torch.nn.LeakyReLU(alpha)

    def forward(self, X, A):
        """
        X: Input node feature matrix (n_nodes x in_features)
        A: Adjacency matrix (n_nodes x n_nodes)
        """
        # Step 1: Apply linear transformation
        Wh = self.W(X)  # Linear transformation W^(1) * X
        
        n_nodes = X.size(0)
        
        # Step 2: Compute attention scores
        # We need to compute the pairwise attention score α_{i,j} for every (i,j) pair where A[i,j] == 1
        attention = torch.zeros(n_nodes, n_nodes)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if A[i, j] != 0:  # Consider only neighbors (j in N(i))
                    # Step 3: Concatenate features of nodes i and j, then compute LeakyReLU(r^T [W^(1)x_i ⊕ W^(1)x_j])
                    concatenated_features = torch.cat([Wh[i], Wh[j]], dim=0)  # Concatenation W^(1)x_i ⊕ W^(1)x_j
                    e_ij = self.leakyrelu(torch.matmul(self.r.T, concatenated_features))  # r^T * concatenated_features
                    attention[i, j] = e_ij
        
        # Step 4: Apply softmax to get normalized attention scores (softmax over neighbors of each node)
        attention = torch.exp(attention)  # Exponential of the computed scores
        for i in range(n_nodes):
            attention[i] /= torch.sum(attention[i][A[i] != 0])  # Softmax normalization over neighbors
        
        # Generate k-hop adjacency matrices
        Ak_list = [A]  # 1-hop
        for k in range(2, self.num_hops+1):
            Ak = torch.matrix_power(A, k)
            Ak_list.append(Ak)

        return F.gelu(torch.matmul(attention, Wh, Ak_list[-1]))

# Long-Short Distance Attention Mechanism
class LongDistanceAttention(nn.Module):
    def __init__(self, in_features, out_features, num_hops):
        super(LongDistanceAttention, self).__init__()
        self.num_hops = num_hops
        self.h = ShortDistanceAttention(in_features, out_features)
        self.attention_weights = nn.Parameter(torch.ones(self.num_hops))  # Weights for each hop
        self.W = torch.nn.Linear(in_features, out_features, bias=False)  # W^(2)
        
    def forward(self, X, A):
        Wa = self.W(X)  # Linear transformation W^(2) * X
        hk=self.h(X,A)

        # Generate k-hop adjacency matrices
        Ak_list = [A]  # 1-hop
        for k in range(2, self.num_hops+1):
            Ak = torch.matrix_power(A, k)
            Ak_list.append(Ak)

        n_nodes = X.size(0)
        attention = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

        # Step 3: Calculate attention scores for each pair of nodes based on k-hop neighbors
        for k, Ak in enumerate(Ak_list):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if Ak[i, j] != 0:  # Only consider k-hop neighbors
                        # Concatenate features of nodes i and j, then compute attention score
                        cij = torch.matmul(hk[i], Wa[j])  # Dot-product attention
                        attention[i, j] = cij

            # Step 4: Apply softmax normalization for each node's neighbors
            attention_exp = torch.exp(attention)  # Exponential of attention scores
            for i in range(n_nodes):
                attention_exp[i] /= torch.sum(attention_exp[i][Ak[i] != 0])  # Softmax over neighbors
            
            # Step 5: Aggregate node features based on attention scores
            output = torch.zeros((n_nodes, hk.size(1)), dtype=torch.float32)
            for i in range(n_nodes):
                neighbors = hk[Ak[i] != 0]  # Select k-hop neighbors for node i
                weights = attention_exp[i][Ak[i] != 0].view(-1, 1)  # Corresponding attention weights
                output[i] = torch.sum(weights * neighbors, dim=0)  # Weighted sum for each node
            
        # Final transformation
        final_output = self.output_layer(output)  # Map to final output dimensions
        return final_output

# Positive-Unlabeled Learning with Risk Estimators
class PUPositiveUnlabeledLearning(nn.Module):
    def __init__(self, in_features, out_features, num_hops):
        super(PUPositiveUnlabeledLearning, self).__init__()
        self.long_short_attention = LongDistanceAttention(in_features, out_features, num_hops)
        self.fc = nn.Linear(out_features, 1)  # Binary classification

    def forward(self, X, A):
        # Learn node embeddings via long-short distance attention
        H = self.long_short_attention(X, A)
        out = torch.sigmoid(self.fc(H))  # Binary classification output
        return out

    def pu_loss(x, t, prior, gamma=1, beta=0, nnpu=True):
        loss_fn = PULoss(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu)
        return loss_fn(x, t)

# Model initialization and forward pass
model = PUPositiveUnlabeledLearning(m, d, num_hops)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
