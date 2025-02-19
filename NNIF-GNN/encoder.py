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
