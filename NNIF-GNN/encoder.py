import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm
from typing import Union, List

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
        model_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm = norm
        self.out_channels = out_channels

        if model_kwargs is None:
            model_kwargs = {}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.model_type = model_type

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
            if i < num_layers - 1 and self.norm is not None:
                if norm == 'layernorm':
                    self.norms.append(nn.LayerNorm(out_dim))
                elif norm == 'graphnorm':
                    self.norms.append(GraphNorm(out_dim))
                elif norm == 'batchnorm':
                    self.norms.append(nn.BatchNorm1d(out_dim))
                elif norm == 'instancenorm':
                    self.norms.append(nn.InstanceNorm1d(out_dim))
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

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
        elif model_type == 'MLP':
            return nn.Linear(in_channels, out_channels, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: SparseTensor
    ) -> torch.Tensor:
        """
        Forward pass with optional sampling each layer.

        Args:
            x (FloatTensor): [N, in_channels].
            edge_index (SparseTensor): adjacency.

        Returns:
            (FloatTensor) [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            if self.model_type == 'MLP':
                x = conv(x)
            else:
                x = conv(x, edge_index)
            if i < self.num_layers - 1:
                if self.norm is not None:
                    x = self.norms[i](x)
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
