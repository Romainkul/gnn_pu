import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, to_dense_adj, add_self_loops, dense_to_sparse, remove_self_loops
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.parallel as parallel
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, aggregation="mean",encoder=SAGEConv,):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(encoder(in_channels, hidden_channels, aggr=aggregation))
        self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(encoder(hidden_channels, hidden_channels, aggr=aggregation))
            self.bns.append(BatchNorm(hidden_channels))
        self.convs.append(encoder(hidden_channels, out_channels, aggr=aggregation))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = checkpoint(lambda x: F.relu(self.bns[i](self.convs[i](x, edge_index))), x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class Trainer:
    def __init__(self, model, data, optimizer, scheduler, device, amp_enabled=True):
        self.model = model.to(device)
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp_enabled = amp_enabled
        self.scaler = GradScaler(enabled=amp_enabled)

    def train_epoch(self, loss_fn, train_loader, max_grad_norm=1.0):
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc="Training Batches"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=self.amp_enabled):
                embeddings = self.model(batch.x, batch.edge_index)
                loss = loss_fn(embeddings, batch.y, batch.train_mask)

            self.scaler.scale(loss).backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()

        self.scheduler.step()
        return epoch_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, loss_fn, val_loader):
        self.model.eval()
        epoch_loss = 0

        for batch in tqdm(val_loader, desc="Validation Batches"):
            batch = batch.to(self.device)
            embeddings = self.model(batch.x, batch.edge_index)
            loss = loss_fn(embeddings, batch.y, batch.val_mask)
            epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

def train_and_evaluate(data, in_channels, hidden_channels, out_channels, num_layers, learning_rate, weight_decay, dropout, epochs, num_neighbors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
    model = parallel.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=1, epochs=epochs)

    # NeighborLoader for mini-batch training
    train_loader = NeighborLoader(data,num_neighbors=[num_neighbors] * num_layers,input_nodes=data.train_mask,batch_size=128,shuffle=True)
    val_loader = NeighborLoader(data,num_neighbors=[num_neighbors] * num_layers,input_nodes=data.val_mask,batch_size=128)

    trainer = Trainer(model, data, optimizer, scheduler, device)

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(adjacency_loss, train_loader)
        val_loss = trainer.evaluate(adjacency_loss, val_loader)
        test_accuracy = trainer.test(test_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model
