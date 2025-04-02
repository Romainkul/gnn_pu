import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import filter_data
from torch_geometric.utils import to_torch_csc_tensor
from torch_geometric.typing import OptTensor
from typing import Optional, Tuple, Union
from torch import Tensor

def custom_weighted_sample_fn(colptr, row_data, index, num_neighbors,
                              node_features, replace=False, directed=True):
    """
    Optimized weighted sampling using node similarity.
    NumPy conversions and debug print statements have been removed.
    """
    if not isinstance(index, torch.LongTensor):
        index = torch.LongTensor(index)

    samples = []
    to_local_node = {}
    for i in range(len(index)):
        v = index[i]
        samples.append(v)
        to_local_node[v.item()] = i

    rows = []
    cols = []
    edges = []

    begin = 0
    end = len(samples)
    for ell in range(len(num_neighbors)):
        num_samples = num_neighbors[ell]
        for i in range(begin, end):
            w = samples[i]
            # Get neighbor range from the CSC structure.
            col_start = colptr[w].item() if isinstance(colptr[w], torch.Tensor) else colptr[w]
            col_end = colptr[w + 1].item() if isinstance(colptr[w + 1], torch.Tensor) else colptr[w + 1]
            col_count = col_end - col_start

            if col_count == 0:
                continue

            # Create tensor of neighbor offsets on the correct device.
            neighbor_offsets = torch.arange(col_start, col_end, device=node_features.device)
            
            # Get neighbor indices and features.
            neighbor_indices = row_data.index_select(0, neighbor_offsets).clone()
            neighbor_feats = torch.index_select(node_features, 0, neighbor_indices).contiguous()
            
            # Compute cosine similarity.
            w_feat = node_features[w].unsqueeze(0).expand_as(neighbor_feats)
            sims = F.cosine_similarity(w_feat, neighbor_feats, dim=1)
            sim_weights = (sims + 1.0) / 2.0  # Map from [-1,1] to [0,1]
            
            # Normalize weights entirely on PyTorch.
            sim_sum = sim_weights.sum()
            if sim_sum.item() == 0:
                sim_weights = torch.ones_like(sim_weights) / sim_weights.numel()
            else:
                sim_weights = sim_weights / sim_sum

            if (num_samples < 0) or (not replace and (num_samples >= col_count)):
                # Process all neighbors.
                for offset in neighbor_offsets.tolist():
                    neighbor = row_data[offset]
                    if neighbor.item() not in to_local_node:
                        to_local_node[neighbor.item()] = len(samples)
                        samples.append(neighbor)
                    if directed:
                        cols.append(i)
                        rows.append(to_local_node[neighbor.item()])
                        edges.append(offset)
            elif replace:
                # Replace functionality not implemented.
                raise NotImplementedError("Replace functionality is not implemented for similarity-based sampling.")
            else:
                # Use torch.multinomial for sampling without replacement.
                sampled_indices = torch.multinomial(sim_weights, num_samples=num_samples, replacement=False)
                sampled_offsets = neighbor_offsets[sampled_indices]
                for offset in sampled_offsets.tolist():
                    neighbor = row_data[offset]
                    if neighbor.item() not in to_local_node:
                        to_local_node[neighbor.item()] = len(samples)
                        samples.append(neighbor)
                    if directed:
                        cols.append(i)
                        rows.append(to_local_node[neighbor.item()])
                        edges.append(offset)
        begin = end
        end = len(samples)

    if not directed:
        for i in range(len(samples)):
            w = samples[i]
            col_start = colptr[w].item() if isinstance(colptr[w], torch.Tensor) else colptr[w]
            col_end = colptr[w + 1].item() if isinstance(colptr[w + 1], torch.Tensor) else colptr[w + 1]
            for offset in range(col_start, col_end):
                neighbor = row_data[offset]
                if neighbor.item() in to_local_node:
                    rows.append(to_local_node[neighbor.item()])
                    cols.append(i)
                    edges.append(offset)

    try:
        samples = torch.stack(samples)
        rows = torch.tensor(rows)
        cols = torch.tensor(cols)
        edges = torch.tensor(edges)
    except Exception as e:
        raise RuntimeError("Error stacking outputs") from e

    return samples, rows, cols, edges


def to_weighted_csc(
    data: Union[Data],
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, OptTensor]:
    if hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()
        return colptr.to(device), row.to(device), None
    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        size = data.size()
        perm = (col * size[0]).add_(row).argsort()
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])
        if data.edge_weight is not None:
            edge_weight = data.edge_weight
            return colptr.to(device), row[perm].to(device), perm.to(device), edge_weight[perm].to(device)
        raise AttributeError("Data object does not contain attribute 'edge_weight'")
    raise AttributeError("Data object does not contain attributes 'adj_t' or 'edge_index'")


class NeighborSampler:
    def __init__(
        self,
        data: Data,
        num_neighbors,
        replace: bool = False,
        directed: bool = True,
        labeled: bool = False,
        weight_func: Optional[str] = 'similarity',
        device: Optional[torch.device] = None,
    ):
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.labeled = labeled
        self.data = data
        self.weight_func = weight_func
        # Use torch_geometric utility to get a CSC representation.
        csc = to_torch_csc_tensor(data.edge_index)
        self.colptr, self.row, self.perm = csc.ccol_indices(), csc.row_indices(), None
        self.colptr = self.colptr.to(device) if device is not None else self.colptr
        self.row = self.row.to(device) if device is not None else self.row
        if data.x is None:
            raise ValueError("Node features (data.x) are required for similarity-based weighting.")
        self.node_features = data.x.contiguous()

    def __call__(self, index: torch.Tensor):
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)
        if self.weight_func == 'similarity':
            sample_fn = custom_weighted_sample_fn
            node, row, col, edge = sample_fn(
                self.colptr,
                self.row,
                index,
                self.num_neighbors,
                self.node_features,
                self.replace,
                self.directed,
            )
        else:
            raise ValueError('Invalid weight_func specified. Use "similarity".')

        return node, row, col, edge, index.numel()


class ShineLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: Data,
        num_neighbors,
        input_nodes=None,
        replace: bool = False,
        directed: bool = True,
        transform: Optional[callable] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.data = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSampler(
                data, num_neighbors,
                replace, directed, False,
                weight_func='similarity', device=device
            )
        else:
            self.neighbor_sampler = neighbor_sampler

        if input_nodes is None:
            input_node_indices = range(data.num_nodes)
        elif isinstance(input_nodes, torch.Tensor) and input_nodes.dtype == torch.bool:
            input_node_indices = input_nodes.nonzero(as_tuple=False).view(-1)
        elif isinstance(input_nodes, torch.Tensor):
            input_node_indices = input_nodes.tolist()
        else:
            input_node_indices = input_nodes

        super().__init__(input_node_indices, collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out):
        node, row, col, edge, batch_size = out
        data = filter_data(self.data, node, row, col, edge, self.neighbor_sampler.perm)
        data.batch_size = batch_size
        return data if self.transform is None else self.transform(data)

    def _get_iterator(self):
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
