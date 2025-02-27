import argparse
import copy
import json
import random
import warnings

import numpy as np
import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch.nn.functional import normalize
from torch.optim import AdamW
from torch_geometric import datasets
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

warnings.filterwarnings("ignore")


##############################################################################
# Code adapted from BPL to extract data, but PU generating code is new
##############################################################################

def get_dataset(
    root: str,
    name: str,
    transform=NormalizeFeatures()
):
    """
    Retrieve a PyG dataset by name (e.g., 'coauthor-cs', 'coauthor-physics', 
    'amazon-computers', 'amazon-photos') and apply a transform if desired.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be stored.
    name : str
        Name of the dataset. Must be one of {'coauthor-cs', 'coauthor-physics', 
        'amazon-computers', 'amazon-photos'}.
    transform : torch_geometric.transforms.Transform, optional
        Transformation to apply to the data (default: NormalizeFeatures()).

    Returns
    -------
    dataset : torch_geometric.data.InMemoryDataset
        The requested dataset object.
    """
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'Physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
    }

    if name not in pyg_dataset_dict:
        valid_keys = list(pyg_dataset_dict.keys())
        raise ValueError(f"Dataset name must be one of {valid_keys}, got '{name}'.")

    dataset_class, subset_name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=subset_name, transform=transform)
    return dataset


def get_wiki_cs(
    root: str,
    transform=NormalizeFeatures()
):
    """
    Fetch the WikiCS dataset, standardize the node features, and ensure 
    the graph is undirected.

    Parameters
    ----------
    root : str
        Path to the directory where the WikiCS data is stored (or will be downloaded).
    transform : torch_geometric.transforms.Transform, optional
        Transformation to apply to the data (default: NormalizeFeatures()).

    Returns
    -------
    dataset_list : list of Data
        A single-element list containing the WikiCS Data object.
    train_mask : np.ndarray
        Training mask from the dataset.
    val_mask : np.ndarray
        Validation mask.
    test_mask : np.ndarray
        Test mask.
    """
    dataset = datasets.WikiCS(root, is_undirected=True, transform=transform)
    data = dataset[0]

    # Standardize features
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std

    # Ensure undirected
    data.edge_index = to_undirected(data.edge_index)

    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)


def get_lasftm_asia(dataset_name: str) -> Data:
    """
    Load the LastFM Asia dataset from local text/json files. Builds a PyG 
    Data object including features, edges, and labels.

    Parameters
    ----------
    dataset_name : str
        Unused within this function (kept for interface consistency).

    Returns
    -------
    data : Data
        PyG Data object with:
          - data.x (node features)
          - data.edge_index (graph edges)
          - data.y (labels)
    """
    graph_edges = "./data/lasftm-asia/lastfm_asia_edges.txt"
    graph_node_feature = "./data/lasftm-asia/lastfm_asia_features.json"
    graph_node_label = "./data/lasftm-asia/lastfm_asia_target.txt"

    start, to = [], []
    with open(graph_edges, 'r') as f:
        for line in f:
            strlist = line.split()
            start.append(int(strlist[0]))
            to.append(int(strlist[1]))
    edge_index = torch.tensor([start, to], dtype=torch.int64)

    label_list = []
    with open(graph_node_label, 'r') as f:
        for line in f:
            _, label_str = line.split()
            label_list.append(int(label_str))
    y = torch.tensor(label_list)

    x_values = []
    with open(graph_node_feature, 'r') as fp:
        json_data = json.load(fp)

    # max_index can be static or inferred from data
    max_index = 7841
    for raw_feat in json_data.values():
        mask = torch.tensor(raw_feat, dtype=torch.long)
        x_value = torch.zeros(max_index + 1, dtype=torch.float32)
        if len(raw_feat) > 0:
            x_value[mask] = 1.0
        x_values.append(x_value.tolist())

    x = torch.tensor(x_values, dtype=torch.float32)
    x = normalize(x, p=2.0, dim=0)  # L2-normalize features
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def get_planetoid(dataset_name: str) -> Data:
    """
    Load a Planetoid dataset (Cora, Citeseer, Pubmed) using PyG's Planetoid class.

    Parameters
    ----------
    dataset_name : str
        One of {'cora', 'citeseer', 'pubmed'}.

    Returns
    -------
    data : Data
        PyG Data object with the typical Planetoid fields.
    """
    dataset = Planetoid("./data", dataset_name, transform=T.TargetIndegree())
    data = dataset[0]
    return data


def get_wiki(dataset_name: str) -> Data:
    """
    Retrieve the WikiCS dataset (wrapper around get_wiki_cs), ignoring the 
    train/val/test masks returned by that function.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Unused here (kept for interface consistency).

    Returns
    -------
    data : Data
        The PyG Data object for WikiCS.
    """
    dataset_list, _, _, _ = get_wiki_cs('./data/wiki-cs')
    data = dataset_list[0]
    return data


def get_common_dataset(dataset_name: str) -> Data:
    """
    Wrapper to load common PyG datasets (Coauthor CS/Physics, Amazon Computers/Photos).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, one of {'coauthor-cs', 'coauthor-physics', 
        'amazon-computers', 'amazon-photos'}.

    Returns
    -------
    data : Data
        The first Data object from the loaded dataset.
    """
    dataset = get_dataset('./data', dataset_name)
    data = dataset[0]
    return data


def load_dataset(dataset_name: str) -> Data:
    """
    Load a graph dataset by name, set data.num_classes, and return the Data object.

    Supported datasets:
      - amazon-computers, amazon-photos
      - coauthor-cs, coauthor-physics
      - wiki-cs
      - cora, citeseer, pubmed
      - lasftm-asia

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.

    Returns
    -------
    data : Data
        PyG Data object with .num_classes set.
    """
    loader_dict = {
        "amazon-computers": get_common_dataset,
        "amazon-photos": get_common_dataset,
        "coauthor-cs": get_common_dataset,
        "coauthor-physics": get_common_dataset,
        "wiki-cs": get_wiki,
        "cora": get_planetoid,
        "citeseer": get_planetoid,
        "pubmed": get_planetoid,
        "lasftm-asia": get_lasftm_asia
    }

    key = dataset_name.lower()
    if key not in loader_dict:
        valid_keys = list(loader_dict.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. "
                         f"Must be one of {valid_keys}.")

    data = loader_dict[key](key)
    data.num_classes = torch.max(data.y, dim=0)[0].item() + 1
    return data

def make_pu_dataset(
    data: Data,
    mechanism: str = "SCAR",  # or "SAR"
    fixed_seed: bool = True,
    sample_seed: int = 5,
    train_pct: float = 0.5,
    show_count: bool = False
) -> Data:
    """
    Convert a multi-class dataset into a PU setting, but ONLY mark labeled positives 
    in data.train_mask. All other nodes remain unlabeled (or effectively not used).

    Parameters
    ----------
    data : Data
        PyG Data object with multi-class labels (data.y) and possibly data.x for SAR.
    mechanism : {'SCAR', 'SAR'}, default='SCAR'
        How to select which positives become labeled:
          - 'SCAR': random
          - 'SAR': distance-based (requires data.x, picks positives 
            that are far from negative class).
    fixed_seed : bool, default=True
        If True, fix random seeds for reproducibility.
    sample_seed : int, default=5
        The seed used if fixed_seed=True.
    train_pct : float, default=0.07
        Fraction of positives to label as “known positive.”
    show_count : bool, default=False
        If True, prints the number of nodes per class before re-labeling.

    Returns
    -------
    data : Data
        Modified data object with:
          - data.y in {0,1} (largest class => 1)
          - data.train_mask (only labeled positives = 1)
          - data.prior => fraction of positives
        The rest of the nodes have no specific mask set; you can treat them as unlabeled.
    """
    # 1) Identify largest class => positive
    data.num_classes = torch.max(data.y).item() + 1
    class_sizes = [(data.y == c).sum().item() for c in range(data.num_classes)]
    max_class = int(torch.tensor(class_sizes).argmax().item())

    if show_count:
        for c, size in enumerate(class_sizes):
            print(f"Class {c}: {size} nodes")
        print(f"Largest class => {max_class} with {class_sizes[max_class]} nodes")

    # Binarize
    data.y = (data.y == max_class).long()
    n_nodes = data.y.size(0)

    # Initialize train_mask as all False
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)

    # If you want to store the prior
    data.prior = data.y.sum().item() / float(n_nodes)

    # Identify positives
    pos_idx = (data.y == 1).nonzero(as_tuple=False).view(-1)
    pos_num = pos_idx.size(0)
    if pos_num == 0:
        print("No positives found => skipping.")
        return data

    train_pos_num = round(pos_num * train_pct)
    if train_pos_num > pos_num:
        train_pos_num = pos_num

    if fixed_seed:
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)

    # SCAR => random shuffle of positives
    if mechanism.upper() == "SCAR":
        pos_list = pos_idx.tolist()
        random.shuffle(pos_list)
        chosen_pos = torch.tensor(pos_list[:train_pos_num], dtype=torch.long)
    elif mechanism.upper() == "SAR":
        # For SAR, we need data.x to compute distances
        if data.x is None:
            raise ValueError("data.x is required for SAR distance-based approach.")
        neg_idx = (data.y == 0).nonzero(as_tuple=False).view(-1)
        if len(neg_idx) == 0:
            # No negatives => can't do distance-based selection
            chosen_pos = pos_idx[:train_pos_num]
        else:
            x_pos = data.x[pos_idx]
            x_neg = data.x[neg_idx]
            dist_matrix = torch.cdist(x_pos, x_neg, p=2)
            dist_mean = dist_matrix.mean(dim=1) + 1e-8
            probs = dist_mean / dist_mean.sum()
            chosen_ids = torch.multinomial(probs, num_samples=train_pos_num, replacement=False)
            chosen_pos = pos_idx[chosen_ids]
    else:
        raise ValueError(f"Invalid mechanism '{mechanism}' (use 'SCAR' or 'SAR').")

    data.train_mask[chosen_pos] = True

    return data