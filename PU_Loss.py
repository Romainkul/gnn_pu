import random
import torch
import torch.nn.functional as F


class PULoss(torch.nn.Module):
    def __init__(self, prior, gamma=1, beta=0, nnpu=True):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnpu = nnpu

    def forward(self, x, t):
        # Assuming x is the output of the model and t is the target labels
        positive = (t == 1).float()  # 1 for positive class
        unlabeled = (t == -1).float()  # -1 for unlabeled class (if applicable)

        n_positive = positive.sum().clamp(min=1)  # Ensure non-zero
        n_unlabeled = unlabeled.sum().clamp(min=1)  # Ensure non-zero

        # Compute positive and negative risks
        y_positive = torch.sigmoid(-x)  # Loss for positive
        y_unlabeled = torch.sigmoid(x)  # Loss for unlabeled

        # Calculate risks
        positive_risk = (self.prior * positive / n_positive * y_positive).sum()
        negative_risk = ((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled).sum()

        objective = positive_risk + negative_risk
        
        if self.nnpu:
            if negative_risk < -self.beta:
                objective = positive_risk - self.beta
        
        return objective