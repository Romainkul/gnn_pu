import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import AdamW
from encoder import GraphEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_generating import load_dataset, make_pu_dataset

def p_probs(net, X_pos, device):
    """
    Probability that each sample in X_pos is class 0 (positive).
    """
    net.eval()
    with torch.no_grad():
        outputs = net(X_pos.to(device))
        # Probability of 'positive' (index 0)
        probs = torch.softmax(outputs, dim=-1)[:, 0]
    return probs.cpu().numpy()

def u_probs(net, X_unlabeled, Y_unlabeled, device):
    """
    Probability that each sample in X_unlabeled is class 0 (positive),
    plus the known/true targets in Y_unlabeled for alpha-estimation.
    """
    net.eval()
    with torch.no_grad():
        outputs = net(X_unlabeled.to(device))
        probs = torch.softmax(outputs, dim=-1)
    return probs.cpu().numpy(), Y_unlabeled.numpy()

def DKW_bound(x,y,t,m,n,delta=0.1, gamma= 0.01):

    temp = np.sqrt(np.log(4/delta)/2/n) + np.sqrt(np.log(4/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound


def BBE_estimator(pdata_probs, udata_probs, udata_targets):
    """
    Breadth-Based Estimator for alpha, adapted from the approach that
    ranks by predicted probabilities. 
    
    Args:
        pdata_probs (ndarray): Prob(pos) for positive-labeled data.
        udata_probs (ndarray): Nx2 array, each row is [prob(pos), prob(neg)] for unlabeled data.
        udata_targets (ndarray): True 0/1 labels for that unlabeled subset (for alpha estimation).
    
    Returns:
        float: estimated alpha
    """
    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    u_indices = np.argsort(udata_probs[:,0])
    sorted_u_probs = udata_probs[:,0][u_indices]
    sorted_u_targets = udata_targets[u_indices]

    sorted_u_probs = sorted_u_probs[::-1]
    sorted_p_probs = sorted_p_probs[::-1]
    sorted_u_targets = sorted_u_targets[::-1]
    num = len(sorted_u_probs)

    estimate_arr = []

    upper_cfb = []
    lower_cfb = []            

    i = 0
    j = 0
    num_u_samples = 0

    while (i < num):
        start_interval =  sorted_u_probs[i]   
        k = i 
        if (i<num-1 and start_interval> sorted_u_probs[i+1]): 
            pass
        else: 
            i += 1
            continue
        if (sorted_u_targets[i]==1):
            num_u_samples += 1

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
            estimate, lower , upper = DKW_bound(i, j, t, len(sorted_u_probs), len(sorted_p_probs))
            estimate_arr.append(estimate)
            upper_cfb.append( upper)
            lower_cfb.append( lower)
        i+=1

    if (len(upper_cfb) != 0): 
        idx = np.argmin(upper_cfb)
        mpe_estimate = estimate_arr[idx]

        return mpe_estimate, lower_cfb, upper_cfb
    else: 
        return 0.0, 0.0, 0.0

def estimate_alpha_tedn(net, X_pos_val, Y_pos_val, X_unl_val, Y_unl_val, device):
    """
    Given a small validation set with known labels:
      - P-labeled data (X_pos_val)
      - U-labeled data (X_unl_val) but with ground-truth Y_unl_val
    Returns an alpha estimate (float).
    """
    net.eval()
    # Probability of positive for the small pos val set
    pdata_probs = p_probs(net, X_pos_val, device)
    # Probability + true labels for the unlabeled val set
    udata_probs, udata_targets = u_probs(net, X_unl_val, Y_unl_val, device)
    # Use BBE to estimate alpha
    alpha_est = BBE_estimator(pdata_probs, udata_probs, udata_targets)
    return alpha_est

def rank_inputs(net, X_unlabeled, device, alpha):
    """
    Ranks unlabeled samples by predicted prob(pos) & discards the top alpha*N as negative.
    Returns a 1D array `keep_samples` of 0/1.
    """
    net.eval()
    with torch.no_grad():
        outputs = net(X_unlabeled.to(device))
        probs = torch.softmax(outputs, dim=-1)[:, 0].cpu().numpy()

    N = len(probs)
    sorted_idx = np.argsort(probs)  # ascending
    cutoff = int(alpha * N)
    
    keep_samples = np.ones(N, dtype=int)
    # Discard top alpha*N
    keep_samples[sorted_idx[N - cutoff :]] = 0
    return keep_samples

def train_pu_discard(net, X_pos, Y_pos, X_unlabeled, Y_unlabeled,
                     keep_samples, device, optimizer, criterion):
    """
    Single-epoch training step for TED(n): discard high-prob-likely-neg, 
    then train on positives + the kept unlabeled portion.
    """
    net.train()
    optimizer.zero_grad()

    # Select unlabeled samples to keep
    idx_keep = np.where(keep_samples == 1)[0]
    X_keep = X_unlabeled[idx_keep]
    Y_keep = Y_unlabeled[idx_keep]

    # Combine all data for a single pass
    X_all = torch.cat([X_pos, X_keep], dim=0).to(device)
    Y_all = torch.cat([Y_pos, Y_keep], dim=0).to(device)

    outputs = net(X_all)
    # Split so we can log or handle them differently if needed
    p_out = outputs[: len(X_pos)]
    u_out = outputs[len(X_pos) :]

    p_loss = criterion(p_out, Y_all[: len(X_pos)])
    u_loss = criterion(u_out, Y_all[len(X_pos) :])
    loss = 0.5 * (p_loss + u_loss)

    loss.backward()
    optimizer.step()

    # Compute training accuracy
    _, predicted = outputs.max(dim=1)
    correct = (predicted == Y_all).sum().item()
    total = Y_all.size(0)
    return 100.0 * correct / total

def run_tedn_training(
    net,
    X_pos, Y_pos,
    X_unl, Y_unl,  # unlabeled data + pseudo-labels
    X_pos_val, Y_pos_val,  # labeled positives for alpha estimation
    X_unl_val, Y_unl_val,  # labeled 'unlabeled' for alpha estimation
    device,
    alpha_init=0.3,        # initial alpha
    epochs=10,
    optimizer=None,
    criterion=None
    ):
    """
    Runs a multi-epoch TED(n) training loop. 
    Each epoch:
      1. Optionally estimate alpha using a labeled validation set.
      2. Discard top alpha*N unlabeled samples.
      3. Train on positives + kept unlabeled.
      4. Validate on your chosen set.

    Args:
        net (nn.Module): model
        X_pos, Y_pos (Tensors): all-labeled positives and their labels (usually 0)
        X_unl, Y_unl (Tensors): unlabeled data & pseudo-labels (0 or 1)
        X_pos_val, Y_pos_val (Tensors): small labeled positive data for alpha estimation
        X_unl_val, Y_unl_val (Tensors): small labeled 'unlabeled' data for alpha estimation
        device (str): 'cpu' or 'cuda'
        alpha_init (float): starting alpha
        epochs (int): number of epochs
        optimizer (torch.optim.Optimizer): an optimizer instance
        criterion (nn.Module): e.g. nn.CrossEntropyLoss()
        estimate_alpha_every (int): re-estimate alpha every N epochs

    Returns:
        None
    """
    alpha_used = alpha_init

    for epoch in range(epochs):

        alpha_est = estimate_alpha_tedn(net, X_pos_val, Y_pos_val, X_unl_val, Y_unl_val, device)
        alpha_used = alpha_est

        # 2) Rank & discard unlabeled
        keep_samples = rank_inputs(net, X_unl, device, alpha_used)

        # 3) Train
        train_acc = train_pu_discard(net, X_pos, Y_pos, X_unl, Y_unl, keep_samples,
                                     device, optimizer, criterion)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Alpha={alpha_used:.3f} | "
              f"Train Acc={train_acc:.2f}% | ")


if __name__ == '__main__':
    seed = 42

    # Set Random Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_type = "GCN"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_type = model_type
    alpha = 0.2
    beta = 0.5
    epochs = 10
    optimizer_str = "AdamW"
    lr=0.01
    wd=0.0

    # Dataset & Configuration
    hidden_dim = 64
    output_dim = 2  # Binary classification
    num_layers = 3
    dropout = 0.5
    model_type = "GCN"  # Default model type
    dataset_name = 'citeseer'
    mechanism = 'SCAR'
    seed = 1
    train_pct = 0.5

    # Load dataset and create PU labels
    data = load_dataset(dataset_name)
    data = make_pu_dataset(data, mechanism=mechanism, sample_seed=seed, train_pct=train_pct)

    model = GraphEncoder(data.num_nodes, hidden_dim, output_dim, num_layers, dropout, model_type=model_type).to(device)

    # Optimizer Selection
    criterion = nn.CrossEntropyLoss()

    if optimizer_str == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_str == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_str == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer. Choose from SGD, Adam, AdamW.")

    net = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    # Suppose your data is loaded as entire Tensors
    # X_pos, Y_pos : all-labeled positives (Y=0)
    # X_unl, Y_unl : unlabeled + pseudo-labels (0 or 1)
    # X_pos_val, Y_pos_val: small labeled positives for alpha estimation
    # X_unl_val, Y_unl_val: small labeled 'unlabeled' set for alpha estimation

    # (Below: just random Tensors as placeholders)
    X_pos = torch.randn(200, 100)
    Y_pos = torch.zeros(200, dtype=torch.long)  # e.g., class=0
    X_unl = torch.randn(1000, 100)
    # Pseudo-labels might all be 1 or some mixture
    Y_unl = torch.ones(1000, dtype=torch.long)

    # For alpha estimation, we need some ground truth in a 'validation' portion:
    X_pos_val = torch.randn(50, 100)
    Y_pos_val = torch.zeros(50, dtype=torch.long)
    X_unl_val = torch.randn(50, 100)
    Y_unl_val = torch.randint(0, 2, size=(50,))  # true 0 or 1

    # Define optimizer and loss
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Run training
    run_tedn_training(
        net=net,
        X_pos=X_pos, Y_pos=Y_pos,
        X_unl=X_unl, Y_unl=Y_unl,
        X_pos_val=X_pos_val, Y_pos_val=Y_pos_val,
        X_unl_val=X_unl_val, Y_unl_val=Y_unl_val,
        device=device,
        alpha_init=0.3,
        epochs=5,
        optimizer=optimizer,
        criterion=criterion,
        estimate_alpha_every=1
    )
