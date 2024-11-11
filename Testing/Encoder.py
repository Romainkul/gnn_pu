import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx
from node2vec import Node2Vec  # Use Node2Vec from an external library

class Encoder:
    def __init__(self, method='GraphSAGE', **kwargs):
        """
        Initialize the Encoder class with the specified encoding method.
        
        Parameters:
        - method (str): The encoding method to use. Options are 'GraphSAGE' or 'Node2vec'.
        - kwargs: Hyperparameters specific to the chosen encoding method.
        """
        self.method = method
        self.kwargs = kwargs
        if method == 'GraphSAGE':
            # Setup for GraphSAGE
            self.in_channels = kwargs.get('in_channels', 64)
            self.hidden_channels = kwargs.get('hidden_channels', 128)
            self.out_channels = kwargs.get('out_channels', 64)
            self.model = self._init_graphsage()
        elif method == 'Node2vec':
            # Setup for Node2vec
            self.dimensions = kwargs.get('dimensions', 64)
            self.walk_length = kwargs.get('walk_length', 30)
            self.num_walks = kwargs.get('num_walks', 200)
            self.window = kwargs.get('window', 10)
            self.min_count = kwargs.get('min_count', 1)
            self.batch_words = kwargs.get('batch_words', 4)
        else:
            raise ValueError("Unsupported method. Choose 'GraphSAGE' or 'Node2vec'.")

    def _init_graphsage(self):
        """
        Initialize GraphSAGE model with specified layers.
        """
        class GraphSAGEModel(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GraphSAGEModel, self).__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        return GraphSAGEModel(self.in_channels, self.hidden_channels, self.out_channels)

    def fit(self, data):
        """
        Fit the encoder model to the graph data.
        
        Parameters:
        - data: The input graph data, which should include attributes:
            - x (node features)
            - edge_index (graph edges)
        
        Returns:
        - Encoded node embeddings.
        """
        if self.method == 'GraphSAGE':
            return self._fit_graphsage(data)
        elif self.method == 'Node2vec':
            return self._fit_node2vec(data)

    def _fit_graphsage(self, data):
        """
        Perform GraphSAGE encoding on the input data.
        """
        self.model.train()
        x, edge_index = data.x, data.edge_index
        embeddings = self.model(x, edge_index)
        return embeddings

    def _fit_node2vec(self, data):
        """
        Perform Node2vec encoding on the input data.
        
        Parameters:
        - data: Graph data in torch_geometric format.
        
        Returns:
        - embeddings (dict): Node embeddings indexed by node IDs.
        """
        nx_graph = to_networkx(data, to_undirected=True)
        node2vec = Node2Vec(
            nx_graph, 
            dimensions=self.dimensions, 
            walk_length=self.walk_length, 
            num_walks=self.num_walks, 
            workers=4
        )
        model = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=self.batch_words)
        embeddings = {str(node): model.wv[str(node)] for node in nx_graph.nodes()}
        return embeddings

    def encode(self, data):
        """
        Public method to perform encoding based on the chosen method.
        """
        return self.fit(data)
