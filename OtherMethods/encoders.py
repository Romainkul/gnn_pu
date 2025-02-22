import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

class BaseEncoder(nn.Module):
    """A modular class for GNN encoders with support for GCN, GAT, GraphSAGE, GIN, and MLP."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, model_type="GCN", heads=8):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
            num_layers (int): Number of layers.
            dropout (float): Dropout probability.
            model_type (str): Type of model: "GCN", "GAT", "GraphSAGE", "GIN", "MLP".
            heads (int): Number of attention heads (for GAT).
        """
        super(BaseEncoder, self).__init__()

        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout

        # Create layers based on the model type
        self.convs = nn.ModuleList()

        if model_type == "GCN":
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))

        elif model_type == "GAT":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))

        elif model_type == "GraphSAGE":
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, output_dim))

        elif model_type == "GIN":
            mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            for _ in range(num_layers - 2):
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(mlp))
            mlp = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
            self.convs.append(GINConv(mlp))

        elif model_type == "MLP":
            # MLP: Fully connected layers without message passing
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        
        else:
            raise ValueError("Invalid model type. Choose from 'GCN', 'GAT', 'GraphSAGE', 'GIN', 'MLP'.")

    def forward(self, x, edge_index=None):
        """Forward pass through the selected encoder."""
        if self.model_type == "MLP":
            return self.mlp(x)  # MLP doesn't use edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Apply non-linearity and dropout to all but the last layer
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
