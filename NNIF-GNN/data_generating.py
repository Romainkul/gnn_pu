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

def get_elliptic_bitcoin(dataset_name: str,
                         path: str = r"C:\Users\romai\Desktop\elliptic_bitcoin_dataset") -> Data:
    """
    Loads the Elliptic Bitcoin dataset and reindexes node indices to start from 0.

    Parameters:
        dataset_name: str
            The name of the dataset (not used but kept for consistency).
        path: str
            Path to the dataset folder.

    Returns:
        data: torch_geometric.data.Data
            A PyG Data object with reindexed edges and node features.
    """
    import pandas as pd
    # -----------------------------
    # (1) Load Node Features
    # -----------------------------
    # Load the features file; first column is original node index,
    # second column is timestamp, remaining columns are features.
    features = np.loadtxt(path + r"\elliptic_txs_features.csv",
                          delimiter=",", skiprows=0, usecols=range(0, 167),
                          dtype=np.float32)
    # Extract original node indices from the features file.
    original_node_ids = features[:, 0].astype(np.int64)
    # Create a complete list of nodes based on the features file.
    unique_nodes = np.unique(original_node_ids)
    # Mapping: original node id -> new sequential id (0 to num_nodes - 1)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}

    # Prepare node feature tensor (exclude the first two columns: index and timestamp)
    x = torch.tensor(features[:, 2:], dtype=torch.float32)
    # Timestamp is the second column.
    time = torch.tensor(features[:, 1], dtype=torch.long)

    # -----------------------------
    # (2) Load Edge List (Transactions Graph)
    # -----------------------------
    # The edge list is an [N, 2] array.
    edges = np.loadtxt(path + r"\elliptic_txs_edgelist.csv",
                       delimiter=",", skiprows=1, usecols=(0, 1), dtype=np.int64)
    # Map edges: ensure both source and target are in our mapping.
    edges_mapped = np.array([[node_mapping[src], node_mapping[dst]]
                             for src, dst in edges
                             if src in node_mapping and dst in node_mapping])
    edges_mapped = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()

    # -----------------------------
    # (3) Load Labels
    # -----------------------------
    # The classes file: we assume it provides labels in the same order as the features file.
    y_df = pd.read_csv(path + r"\elliptic_txs_classes.csv",
                       delimiter=",", usecols=(0,1), dtype=str)
    label_mapping = {"1": 0, "2": 1}
    y_mapped = torch.tensor(y_df.iloc[:, 1].map(lambda x: int(x) - 1 if x in label_mapping else 2).values, dtype=torch.long)
    num_classes = y_mapped.max().item() + 1
    # -----------------------------
    # (4) Reindex Additional Attributes
    # -----------------------------
    # Create a tensor of new node indices for each row in the features file.
    new_indices = torch.tensor([node_mapping[old_id] for old_id in original_node_ids],
                               dtype=torch.long)

    # -----------------------------
    # (5) Create PyG Data Object
    # -----------------------------
    data = Data(
        x=x,
        edge_index=edges_mapped,
        y=y_mapped,
        num_nodes=y_mapped.size()[0],  # Total number of nodes from the features file.
        num_classes=num_classes,
        is_elliptic=True,
        time=time,
        index=new_indices  # New sequential node IDs for the features.
    )

    return data

def get_ibm_aml(path:str=r"C:\Users\Romain\OneDrive - KU Leuven\trans_3000p2_list.txt")->Data:
    timestamp=np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0))
    bank_out,account_out, bank_in, account_in = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 4))
    edge_in = np.concatenate((bank_in, account_in))
    edge_out = np.concatenate((bank_out, account_out))
    edges = np.stack((edge_out, edge_in), axis=0)

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(5, 8), dtype=np.float32)
    np.concatenate((x, timestamp), axis=1)
    x = torch.tensor(x, dtype=torch.float32)

    y = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(9), dtype=np.int64)
    y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edges, y=y)
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
        "lasftm-asia": get_lasftm_asia,
        "elliptic-bitcoin": get_elliptic_bitcoin,
        "ibm-aml": get_ibm_aml
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
    mechanism: str = "SCAR",  # or "SAR" or "SAR2"
    fixed_seed: bool = True,
    sample_seed: int = 5,
    train_pct: float = 0.5,
    show_count: bool = False
) -> Data:
    """
    Convert a multi-class dataset into a PU setting. We produce:
      - data.train_mask: 'train_pct' of the positive nodes = 1 (labeled),
        all other nodes = 0
      - data.test_mask: all positives (y=1) AND negatives (y=0) = 1,
        unlabeled (y=2) = 0

    Mechanisms:
      - SCAR: random subset of positives
      - SAR: distance-based selection of positives (requires data.x)
      - SAR2: time-based selection of positives (requires data.time)
               picks earliest 'train_pct' fraction in ascending order
               if data.is_elliptic or your dataset has time.

    If 'data.is_elliptic' is True, we treat the second dimension of y
    as [0=lowest-class => positive, 1=second => negative, 2=unknown]
    and skip the usual largest-class logic. For other datasets, we do
    'largest class => 1'.

    Parameters
    ----------
    data : Data
        PyG Data object with data.y in {0,1,2} for Elliptic,
        or multi-class for standard GNN tasks. If not Elliptic, we binarize
        by picking the largest class => positive (1).
    mechanism : str
        One of {'SCAR', 'SAR', 'SAR2'}.
    fixed_seed : bool
        If True, sets random seeds for reproducibility.
    sample_seed : int
        Random seed used if fixed_seed=True.
    train_pct : float
        Fraction of positive nodes to mark as labeled (train).
    show_count : bool
        If True, print class distribution info.

    Returns
    -------
    data : Data
        Updated with:
          - data.y in {0,1,2}, if Elliptic (0=neg,1=pos,2=unknown),
            or {0,1} for non-Elliptic (0=neg,1=pos).
          - data.train_mask
          - data.test_mask
          - data.prior => fraction of y=1 in the entire dataset
    """
    # ----- Special handling for Elliptic dataset -----
    if getattr(data, "is_elliptic", False):
        """
        Elliptic has y[:,1] => [0,1,2].
          - 0 => "lowest-numbered class" => positive
          - 1 => "second class" => negative
          - 2 => unknown
        We'll transform the label in y_col to:
          - 0 => 1 (pos)
          - 1 => 0 (neg)
          - 2 => 2 (unknown)
        Then we skip the usual "largest class => 1" logic.
        """
        y_col = data.y  # shape [N], e.g. 0,1,2
        y_new = torch.clone(y_col)

        pos_mask = (y_col == 0)
        neg_mask = (y_col == 1)
        unk_mask = (y_col == 2)

        y_new[pos_mask] = 1  # 0 => positive => 1
        y_new[neg_mask] = 0  # 1 => negative => 0
        y_new[unk_mask] = 2  # 2 => unknown => 2
        data.y = y_new
        data.num_classes = 3  # effectively {0,1,2}, but only {0,1} are "known" classes
    # ----- End special Elliptic handling -----

    # For other datasets, if not Elliptic => do "largest class => 1" logic
    data.num_classes = int(torch.max(data.y).item() + 1)
    if not getattr(data, "is_elliptic", False):
        class_sizes = [(data.y == c).sum().item() for c in range(data.num_classes)]
        max_class = int(torch.tensor(class_sizes).argmax().item())
        if show_count:
            for c, size in enumerate(class_sizes):
                print(f"Class {c}: {size} nodes")
            print(f"Largest class => {max_class} with {class_sizes[max_class]} nodes")

        # Binarize => largest class => 1
        data.y = (data.y == max_class).long()
        data.num_classes = 2  # now in {0,1}

    # Print final distribution if requested
    if show_count and getattr(data, "is_elliptic", False):
        c0 = (data.y == 0).sum().item()
        c1 = (data.y == 1).sum().item()
        c2 = (data.y == 2).sum().item()
        print(f"After mapping: class 0 => {c0} nodes, class 1 => {c1}, class 2 => {c2} unknown")

    # Initialize masks
    n_nodes = data.y.size(0)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(n_nodes, dtype=torch.bool)

    # test_mask => all nodes with y in {0,1}
    # unlabeled => y=2 => test_mask=0
    known_mask = (data.y == 0) | (data.y == 1)
    data.test_mask[known_mask] = True
    # Store the prior => fraction of positives among all nodes
    data.prior = (data.y == 1).sum().item() / float(n_nodes)

    # Identify positives
    pos_idx = (data.y == 1).nonzero(as_tuple=False).view(-1)
    pos_num = pos_idx.size(0)
    if pos_num == 0:
        print("No positives found => skipping.")
        return data

    pos_train_val = round(pos_num * train_pct)
    train_pos_num = round(pos_train_val * 0.7)

    """    if fixed_seed:
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)"""

    # --------------------------------------
    # SCAR => random subset of positives
    # SAR  => distance-based
    # SAR2 => time-based (ascending order)
    # --------------------------------------
    if mechanism.upper() == "SCAR":
        pos_list = pos_idx.tolist()
        random.shuffle(pos_list)
        chosen_pos = torch.tensor(pos_list[:pos_train_val], dtype=torch.long)
        data.train_mask[chosen_pos[:train_pos_num]] = True
        data.val_mask[chosen_pos] = True

    elif mechanism.upper() == "SAR":
        if data.x is None:
            raise ValueError("data.x is required for SAR distance-based approach.")
        neg_idx = (data.y == 0).nonzero(as_tuple=False).view(-1)
        if len(neg_idx) == 0:
            chosen_pos = pos_idx[:pos_train_val]
        else:
            x_pos = data.x[pos_idx]
            x_neg = data.x[neg_idx]
            dist_matrix = torch.cdist(x_pos, x_neg, p=2)  # [pos_num, neg_num]
            dist_mean = dist_matrix.mean(dim=1) + 1e-8
            probs = dist_mean / dist_mean.sum()
            chosen_ids = torch.multinomial(probs, num_samples=pos_train_val, replacement=False)
            chosen_pos = pos_idx[chosen_ids]
            data.train_mask[chosen_pos[:pos_train_val]] = True
            data.val_mask[chosen_pos] = True

    elif mechanism.upper() == "SAR2" and hasattr(data, 'time'):
        if not hasattr(data, 'time'):
            raise ValueError("data.time is required for 'SAR2' approach.")
        pos_times = data.time[pos_idx]
        _, sorted_ids = torch.sort(pos_times, descending=False)
        chosen_pos = pos_idx[sorted_ids[:pos_train_val]]
        data.train_mask[chosen_pos[:train_pos_num]] = True
        data.val_mask[chosen_pos] = True

    else:
        raise ValueError(f"Invalid mechanism '{mechanism}'. Use 'SCAR', 'SAR' or 'SAR2'.")    

    return data