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

"""class SimilarityCache:
    def __init__(self, colptr, row_data, node_features, device=None):
        self.colptr = colptr
        self.row_data = row_data
        self.device = device or node_features.device
        self.node_features = node_features
        self.cache = {}

    def get_sim_weights(self, node_id):
        if node_id in self.cache:
            return self.cache[node_id]

        col_start, col_end = self.colptr[node_id].item(), self.colptr[node_id+1].item()
        neighbors = self.row_data[col_start:col_end]

        if neighbors.numel() == 0:
            return neighbors, torch.tensor([], device=self.device)

        neighbor_feats = self.node_features[neighbors]
        node_feat = self.node_features[node_id].unsqueeze(0).expand_as(neighbor_feats)

        sims = F.cosine_similarity(node_feat, neighbor_feats, dim=1)
        sim_weights = (sims + 1.0) / 2.0
        sim_sum = sim_weights.sum()
        sim_weights = sim_weights / sim_sum if sim_sum > 0 else torch.ones_like(sim_weights)/len(sim_weights)

        # Cache computed similarities
        self.cache[node_id] = (neighbors, sim_weights)
        return neighbors, sim_weights

    def clear_cache(self):
        self.cache.clear()

def cached_weighted_sample_fn(colptr, row_data, index, num_neighbors,
                              node_features, sim_cache: SimilarityCache,
                              replace=False, directed=True):

    device = node_features.device
    index = index.to(device, dtype=torch.long)

    samples = index.clone().tolist()
    to_local_node = {nid.item(): i for i, nid in enumerate(index)}

    rows, cols, edges = [], [], []

    begin, end = 0, len(samples)

    for ell, num_samples in enumerate(num_neighbors):
        current_nodes = samples[begin:end]
        batch_rows, batch_cols = [], []

        for local_idx, w in enumerate(current_nodes):

            neighbors, sim_weights = sim_cache.get_sim_weights(w)

            if neighbors.numel() == 0:
                continue

            if num_samples < 0 or num_samples >= neighbors.size(0):
                selected_neighbors = neighbors
            else:
                sampled_indices = torch.multinomial(sim_weights, num_samples, replacement=False)
                selected_neighbors = neighbors[sampled_indices]

            for neighbor in selected_neighbors:
                neighbor_item = neighbor.item()
                if neighbor_item not in to_local_node:
                    to_local_node[neighbor_item] = len(samples)
                    samples.append(neighbor_item)

                batch_cols.append(begin + local_idx)
                batch_rows.append(to_local_node[neighbor_item])

        rows.extend(batch_rows)
        cols.extend(batch_cols)

        begin, end = end, len(samples)

    samples = torch.tensor(samples, dtype=torch.long, device=device)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    edges = torch.arange(len(rows), device=device)

    return samples, rows, cols, edges"""

def custom_weighted_sample_fn(colptr, row_data, index, num_neighbors,
                              node_features, replace=False, directed=True):

    device = node_features.device
    index = index.to(device, dtype=torch.long)

    samples = index.clone().tolist()
    to_local_node = {nid.item(): i for i, nid in enumerate(index)}

    rows, cols, edges = [], [], []

    begin, end = 0, len(samples)

    for ell, num_samples in enumerate(num_neighbors):
        current_nodes = samples[begin:end]
        batch_rows, batch_cols = [], []

        for local_idx, w in enumerate(current_nodes):
            w_tensor = torch.tensor(w, device=device)
            col_start = colptr[w_tensor].item()
            col_end = colptr[w_tensor + 1].item()
            
            neighbors = row_data[col_start:col_end].to(device)

            if neighbors.numel() == 0:
                continue

            # Features are already on the same device
            neighbor_feats = node_features[neighbors]
            w_feat = node_features[w].unsqueeze(0).expand_as(neighbor_feats)

            sims = F.cosine_similarity(w_feat, neighbor_feats, dim=1)
            sim_weights = (sims + 1.0) / 2.0
            sim_sum = sim_weights.sum()
            sim_weights = sim_weights / sim_sum if sim_sum > 0 else torch.ones_like(sim_weights) / len(sim_weights)

            if num_samples < 0 or num_samples >= neighbors.size(0):
                selected_neighbors = neighbors
            else:
                sampled_indices = torch.multinomial(sim_weights, num_samples, replacement=False)
                selected_neighbors = neighbors[sampled_indices]

            for neighbor in selected_neighbors:
                neighbor_item = neighbor.item()
                if neighbor_item not in to_local_node:
                    to_local_node[neighbor_item] = len(samples)
                    samples.append(neighbor_item)

                batch_cols.append(begin + local_idx)
                batch_rows.append(to_local_node[neighbor_item])

        rows.extend(batch_rows)
        cols.extend(batch_cols)

        begin, end = end, len(samples)

    samples = torch.tensor(samples, dtype=torch.long, device=device)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    edges = torch.arange(len(rows), device=device)

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
        #self.sim_cache = SimilarityCache(self.colptr, self.row, self.node_features, device=device)

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
