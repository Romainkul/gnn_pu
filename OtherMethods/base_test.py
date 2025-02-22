import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from nnPU import PULoss
from encoders import BaseEncoder  

hidden_dim = 64
output_dim = 2
num_layers = 3
dropout = 0.5
num_epochs = 50

dataset_name='citeseer'
mechanism='SCAR'
seed=1
train_pct=0.5
data = load_dataset(dataset_name)
data = make_pu_dataset(data,mechanism=mechanism,sample_seed=seed,train_pct=train_pct)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "GraphSAGE"  # Choose from "GCN", "GAT", "GraphSAGE", "GIN", "MLP"

model = BaseEncoder(data.num_nodes, hidden_dim, output_dim, num_layers, dropout, model_type=model_type).to(device)
pu_loss = PULoss(prior=0.5, gamma=1, beta=0, nnpu=True, imbpu=False, ted=False).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    data = data.to(device)
    out = model(data.x, data.edge_index).squeeze()  # [num_nodes]

    # Compute PU Loss
    loss = pu_loss(out, data.y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Logging
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training Complete!")
