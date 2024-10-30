import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms import NormalizeFeatures

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

target_classes = [0, 1, 2]  # Combine these into one binary label
binary_labels = torch.where(torch.isin(data.y, torch.tensor(target_classes)), torch.tensor(0), torch.tensor(1))
data.y = binary_labels

# Define the VGAE model for node classification
class VGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(VGAE, self).__init__()
        # Encoder layers
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc_mu = GCNConv(hidden_channels, out_channels)
        self.gc_logvar = GCNConv(hidden_channels, out_channels)
        # Classification layer
        self.classifier = torch.nn.Linear(out_channels, num_classes)

    def encode(self, x, edge_index):
        # First GCN layer with ReLU activation
        x = F.relu(self.gc1(x, edge_index))
        # Compute mean and log variance
        mu = self.gc_mu(x, edge_index)
        logvar = self.gc_logvar(x, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: sample z = mu + std * epsilon
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # Encode inputs to latent variables
        mu, logvar = self.encode(x, edge_index)
        # Sample latent variables
        z = self.reparameterize(mu, logvar)
        # Predict node classes
        out = self.classifier(z)
        return out, mu, logvar
    
# Loss function for node classification
def loss_function(out, labels, mask, mu, logvar):
    # Classification loss (cross-entropy for node labels)
    class_loss = F.cross_entropy(out[mask], labels[mask])
    
    # KL divergence loss for variational regularization
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return class_loss + kl_loss

# Example training loop for node classification
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    out, mu, logvar = model(data.x, data.edge_index)
    # Compute loss
    loss = loss_function(out, data.y, data.train_mask, mu, logvar)
    loss.backward()
    optimizer.step()
    return loss.item()

# Example evaluation function
@torch.no_grad()
def test(model, data):
    model.eval()
    out, _, _ = model(data.x, data.edge_index)
    # Get predictions for train, validation, and test sets
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    return train_acc, val_acc, test_acc

# Model parameters
in_channels = dataset.num_features
hidden_channels = 32
out_channels = 16
num_classes = dataset.num_classes
print(num_classes)

# Initialize model, optimizer
vgae = VGAE(in_channels, hidden_channels, out_channels, num_classes)
optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01)

# Training loop
epochs = 200
for epoch in range(epochs):
    loss = train(vgae, data, optimizer)
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test(vgae, data)
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')