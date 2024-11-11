import torch
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx
from node2vec import Node2Vec

class Node2vecEncoder:
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, window=10, min_count=1, batch_words=4):
        """
        Initialize the Node2Vec encoder with the specified hyperparameters.

        Parameters:
        - dimensions (int): Size of the embedding vector.
        - walk_length (int): Length of each random walk.
        - num_walks (int): Number of random walks per node.
        - window (int): Maximum distance between the current and predicted node in the context window.
        - min_count (int): Minimum count of occurrences of words (nodes) to be considered.
        - batch_words (int): Number of words to process in each batch.
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words

    def fit(self, data):
        """
        Perform Node2Vec encoding on the input graph data.

        Parameters:
        - data: Graph data containing node features and edge indices (`x`, `edge_index`).

        Returns:
        - embeddings (dict): Dictionary with node IDs as keys and embedding vectors as values.
        """
        nx_graph = to_networkx(data, to_undirected=True)  # Convert PyG data to NetworkX graph
        node2vec = Node2Vec(
            nx_graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words,
            workers=4
        )
        model = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=self.batch_words)
        
        # Convert node embeddings to a dictionary
        embeddings = {str(node): model.wv[str(node)] for node in nx_graph.nodes()}
        return embeddings
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, out_channels=64):
        """
        Initialize the GraphSAGE model.

        Parameters:
        - in_channels (int): Number of input features per node.
        - hidden_channels (int): Number of hidden channels in GraphSAGE layers.
        - out_channels (int): Number of output features (embedding size).
        """
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GraphSAGE layers.

        Parameters:
        - x (tensor): Node features.
        - edge_index (tensor): Edge indices representing the graph.

        Returns:
        - embeddings (tensor): Node embeddings after passing through the model.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def fit(self, data):
        """
        Perform GraphSAGE encoding on the input graph data.

        Parameters:
        - data: Graph data containing node features (`x`) and edge indices (`edge_index`).

        Returns:
        - embeddings (tensor): Encoded node embeddings.
        """
        self.train()
        x, edge_index = data.x, data.edge_index
        embeddings = self.forward(x, edge_index)
        return embeddings
