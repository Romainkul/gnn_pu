import copy
import torch
from torch.nn.functional import cosine_similarity

class BLP(torch.nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.main_encoder = encoder
        self.mlp_predictor = predictor
        self.aux_encoder = copy.deepcopy(encoder)
        self.aux_encoder.reset_parameters()
        for param in self.aux_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return list(self.main_encoder.parameters()) + list(self.mlp_predictor.parameters())

    @torch.no_grad()
    def update_aux_network(self, tau):
        for param_q, param_k in zip(self.main_encoder.parameters(), self.aux_encoder.parameters()):
            param_k.data.mul_(tau).add_(param_q.data, alpha=1. - tau)

    def forward(self, main_x, aux_x):
        main_h = self.main_encoder(main_x)
        main_p = self.mlp_predictor(main_h)

        with torch.no_grad():
            aux_h = self.aux_encoder(aux_x).detach()
        return main_p, aux_h

def predict_unlabeled_nodes(p1, aux_h2, p2, aux_h1):
    loss1 = 2 - 2*cosine_similarity(p1, aux_h2, dim=-1).mean()
    loss2 = 2 - 2*cosine_similarity(p2, aux_h1, dim=-1).mean()
    return (loss1 + loss2)/2

def predict_positive_nodes(p1, p2, positive_nodes):
    loss1 = _predict_positive(p1, positive_nodes)
    loss2 = _predict_positive(p2, positive_nodes)
    loss = loss1 + loss2
    return loss/2

def _predict_positive(p, seed_nodes):
    pos_centroid = p[seed_nodes].mean(dim=0)
    pos_loss = 2 - 2*cosine_similarity(p[seed_nodes], pos_centroid, dim=-1).mean()
    return pos_loss

def find_reliable_negative_nodes(p1, p2, labeled_positive_nodes, val_positive_nodes):
    pos_centroid1 = p1[labeled_positive_nodes].mean(dim=0)
    pos_centroid2 = p2[labeled_positive_nodes].mean(dim=0)

    # distance measure
    similarities1 = cosine_similarity(p1, pos_centroid1, dim=-1)
    similarities2 = cosine_similarity(p2, pos_centroid2, dim=-1)
    similarities = (similarities1*similarities2)

    similarities[labeled_positive_nodes] = float('inf')
    similarities[val_positive_nodes] = float('inf')
    tk = torch.topk(similarities, k=labeled_positive_nodes.size(0)+val_positive_nodes.size(0), largest=False, dim=0)
    negative_nodes = tk.indices.reshape(-1)

    return negative_nodes
