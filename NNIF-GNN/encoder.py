import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm
from typing import Union, List


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for node embeddings.

    This module applies multiple layers of SAGE convolutions, with optional 
    normalization (LayerNorm or GraphNorm), ReLU activation, and dropout.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input features.
    hidden_channels : int
        Dimensionality of hidden representations in intermediate layers.
    out_channels : int
        Dimensionality of output embeddings.
    num_layers : int, default=2
        Number of SAGE layers.
    dropout : float, default=0.1
        Dropout probability applied after each ReLU activation.
    norm : {'layernorm', 'graphnorm'}, default='layernorm'
        Type of normalization to apply. If 'layernorm', uses `nn.LayerNorm`; 
        if 'graphnorm', uses `GraphNorm`.
    aggregation : str, default='max'
        Aggregation method for SAGEConv. Options include {'mean', 'sum', 'max'}.

    Example
    -------
    >>> encoder = GraphSAGEEncoder(
    ...     in_channels=16,
    ...     hidden_channels=32,
    ...     out_channels=64,
    ...     num_layers=3,
    ...     dropout=0.2,
    ...     norm='graphnorm',
    ...     aggregation='mean'
    ... )
    >>> out = encoder(x, edge_index)

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = 'layernorm',
        aggregation: str = 'sum'
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # SAGE Convolutions
        self.convs = nn.ModuleList()

        # Normalization layers (optional)
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
            if norm == 'layernorm':
                self.norms.append(nn.LayerNorm(hidden_channels))
            elif norm == 'graphnorm':
                self.norms.append(GraphNorm(hidden_channels))

        # Final layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregation))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [N, in_channels], where N is the 
            number of nodes.
        edge_index : Union[torch.Tensor, List[torch.Tensor]]
            Graph connectivity in COO format, typically [2, E] for PyG 
            or a SparseTensor.

        Returns
        -------
        torch.Tensor
            Output node embeddings of shape [N, out_channels].
        """
        # Iterate through all but the last convolution layer
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final convolution layer (no ReLU/Dropout)
        x = self.convs[-1](x, edge_index)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    TransformerConv,
    GraphNorm
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_sparse import SparseTensor

class Sampler(nn.Module):
    """
    A unified sampler for 'SparseTensor' that can perform:
      - 'nn': Distance-based sampling (neighbors closer in feature space 
              have higher selection probability).
      - 'weighted': Sampling edges according to a learnable or given mask M 
                    (values in [0,1]).
      - 'random': Random neighbor sampling.
      - 'feature': Sampling neighbors by feature norm (like FastGCN).

    Args:
        mode (str): One of {'nn', 'weighted', 'random', 'feature'}.
        num_samples (int): Max neighbors to keep per node.
        distance_op (callable, optional): Operation for computing distance
            if mode='nn'. Default is lambda a, b: a - b (Euclidean).
        EPS (float): Small constant to avoid zero probabilities in sampling.
    """
    def __init__(
        self,
        mode: str,
        num_samples: int = 5,
        distance_op=None,
        EPS: float = 1e-6
    ):
        super().__init__()
        valid_modes = ['nn', 'weighted', 'random', 'feature']
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode: {mode}, must be one of {valid_modes}.")
        self.mode = mode
        self.num_samples = num_samples
        self.distance_op = distance_op if distance_op is not None else (lambda a, b: a - b)
        self.EPS = EPS

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        edge_index: SparseTensor,
        M: Optional[torch.Tensor] = None
    ) -> SparseTensor:
        """
        Subsamples edges based on the chosen mode. Input and output are 
        always SparseTensors.

        Args:
            x (FloatTensor): Node features of shape [N, d].
            edge_index (SparseTensor): Adjacency with .nnz() edges. 
            M (FloatTensor, optional): A parameter of shape [E], in [0,1], 
                                       used if mode='weighted'.

        Returns:
            A new SparseTensor with at most 'num_samples' neighbors per node.
        """
        device = x.device
        N = x.size(0)

        # Extract COO from the SparseTensor
        row, col, _ = edge_index.coo()   # e_val is not used for sampling
        E = row.size(0)

        # Build adjacency list
        neighbors = [[] for _ in range(N)]
        weights = [[] for _ in range(N)]  # used differently by each mode

        row_list = row.tolist()
        col_list = col.tolist()

        # 1) Collect neighbors/weights
        for e_id in range(E):
            i = row_list[e_id]
            j = col_list[e_id]
            neighbors[i].append(j)

            if self.mode == 'weighted':
                if M is None:
                    raise ValueError("Must provide M for 'weighted' sampling.")
                w = float(M[e_id].item())
                weights[i].append(w)
            elif self.mode in ['nn', 'feature']:
                # store neighbor ID; we'll compute distances or feature norms later
                weights[i].append(j)
            else:
                # 'random' => uniform
                weights[i].append(1.0)

        new_edges_src = []
        new_edges_dst = []

        # 2) For each node, sample up to num_samples neighbors
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) <= self.num_samples:
                # Keep all
                for nbr in nbrs:
                    new_edges_src.append(i)
                    new_edges_dst.append(nbr)
            else:
                if self.mode == 'random':
                    chosen_idx = torch.randperm(len(nbrs))[:self.num_samples]
                    for idx in chosen_idx:
                        new_edges_src.append(i)
                        new_edges_dst.append(nbrs[idx])

                elif self.mode == 'weighted':
                    w_i = torch.tensor(weights[i], device=device) + self.EPS
                    chosen_idx = torch.multinomial(w_i, self.num_samples, replacement=False)
                    for idx in chosen_idx:
                        new_edges_src.append(i)
                        new_edges_dst.append(nbrs[idx])

                elif self.mode == 'nn':
                    # Distance-based sampling
                    i_feat = x[i].unsqueeze(0)  # (1, d)
                    nbr_idx_list = weights[i]   # the neighbor IDs
                    nbr_feats = x[nbr_idx_list] # shape: (len(nbrs), d)
                    diff = self.distance_op(i_feat, nbr_feats)
                    dist_sq = torch.sum(diff * diff, dim=-1)
                    # closer => higher prob => exp(-dist^2)
                    nn_weights = torch.exp(-dist_sq) + self.EPS
                    chosen_idx = torch.multinomial(nn_weights, self.num_samples, replacement=False)
                    for c in chosen_idx:
                        new_edges_src.append(i)
                        new_edges_dst.append(nbr_idx_list[c])

                elif self.mode == 'feature':
                    # e.g. L2 norm of neighbor's features
                    nbr_idx_list = weights[i]
                    nbr_feats = x[nbr_idx_list]
                    scores = torch.sum(nbr_feats * nbr_feats, dim=-1) + self.EPS
                    chosen_idx = torch.multinomial(scores, self.num_samples, replacement=False)
                    for c in chosen_idx:
                        new_edges_src.append(i)
                        new_edges_dst.append(nbr_idx_list[c])

        new_src = torch.tensor(new_edges_src, device=device)
        new_dst = torch.tensor(new_edges_dst, device=device)

        # 3) Create a new SparseTensor with the retained edges
        new_edge_index = SparseTensor(
            row=new_src,
            col=new_dst,
            sparse_sizes=edge_index.sparse_sizes()  # same shape as original
        )
        return new_edge_index


##############################################################################
# GraphEncoder that uses a SparseTensor and optional Sampler
##############################################################################

class GraphEncoder(nn.Module):
    """
    A unified graph encoder that applies graph convolutions 
    (GCNConv, GATConv, SAGEConv, etc.) layer by layer. Each layer
    can optionally subsample edges with the provided Sampler.

    We assume 'edge_index' is always a SparseTensor, 
    and if the sampler mode is 'weighted', M is an nn.Parameter 
    of shape [edge_index.nnz()].

    Args:
        model_type (str): 'GCNConv', 'GATConv', 'SAGEConv', 'GINConv', 'TransformerConv'.
        in_channels (int): Dimensionality of input features.
        hidden_channels (int): Dimensionality of hidden representations.
        out_channels (int): Dimensionality of output embeddings.
        num_layers (int): Number of convolution layers.
        dropout (float): Dropout probability.
        norm (str): 'layernorm' or 'graphnorm'.
        aggregation (str): Only for SAGEConv (e.g., 'sum', 'mean', 'max').
        model_kwargs (dict): Additional kwargs for the chosen conv.
        sampler (Sampler or None): If provided, edges are subsampled each layer.
    """
    def __init__(
        self,
        model_type: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = 'layernorm',
        aggregation: str = 'sum',
        model_kwargs: Optional[dict] = None,
        sampler: Optional[Sampler] = None
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm = norm
        self.sampler = sampler

        if model_kwargs is None:
            model_kwargs = {}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = in_channels
                out_dim = hidden_channels if num_layers > 1 else out_channels
            elif i == num_layers - 1:
                in_dim = hidden_channels
                out_dim = out_channels
            else:
                in_dim = hidden_channels
                out_dim = hidden_channels

            conv = self.build_conv_layer(
                model_type=model_type,
                in_channels=in_dim,
                out_channels=out_dim,
                aggregation=aggregation,
                **model_kwargs
            )
            self.convs.append(conv)

            # Add normalization for intermediate layers
            #if i < num_layers - 1:
            #    if norm == 'layernorm':
            #        self.norms.append(nn.LayerNorm(out_dim))
            #    elif norm == 'graphnorm':
            #        self.norms.append(GraphNorm(out_dim))
            #   else:
            #        raise ValueError(f"Unsupported norm: {norm}")

    def build_conv_layer(
        self,
        model_type: str,
        in_channels: int,
        out_channels: int,
        aggregation: str,
        **model_kwargs
    ):
        if model_type == 'GCNConv':
            return GCNConv(in_channels, out_channels, **model_kwargs)
        elif model_type == 'GATConv':
            return GATConv(in_channels, out_channels, **model_kwargs)
        elif model_type == 'SAGEConv':
            return SAGEConv(in_channels, out_channels, aggr=aggregation, **model_kwargs)
        elif model_type == 'GINConv':
            mlp_hidden = model_kwargs.get('mlp_hidden_channels', out_channels)
            mlp = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, out_channels)
            )
            return GINConv(mlp, **model_kwargs)
        elif model_type == 'TransformerConv':
            return TransformerConv(in_channels, out_channels, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: SparseTensor,
        M: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional sampling each layer.

        Args:
            x (FloatTensor): [N, in_channels].
            edge_index (SparseTensor): adjacency.
            M (FloatTensor): shape [E], if sampler.mode == 'weighted'.

        Returns:
            (FloatTensor) [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            # Optional sampling
            if self.sampler is not None:
                edge_index = self.sampler(x, edge_index, M=M)

            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                #x = self.norms[i](x)
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x