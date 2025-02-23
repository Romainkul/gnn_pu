import torch
import torch.nn.functional as F


class PULoss(torch.nn.Module):
    """
    Implements PU learning loss with nnPU, Imbalanced nnPU, and TED.
    
    Args:
        prior (float): Estimated class prior (fraction of positives in unlabeled data).
        gamma (float): Weighting factor for ImbPU (default=1 for balanced).
        beta (float): Threshold for nnPU risk correction.
        nnpu (bool): Whether to apply the non-negative correction (nnPU).
        imbpu (bool): Whether to apply imbalance correction (ImbPU).
        ted (bool): Whether to use TED, which removes high-loss samples.
        alpha (float): Mixing parameter for ImbPU (0.5 means equal treatment of both classes).
    """

    def __init__(self, prior, gamma=1, beta=0, nnpu=True, imbpu=False, ted=False, alpha=0.5):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnpu = nnpu
        self.imbpu = imbpu
        self.ted = ted
        self.alpha = alpha  # Used for ImbPU weighting

    def forward(self, x, t):
        """
        Computes PU learning loss.

        Args:
            x (torch.Tensor): Model predictions (logits).
            t (torch.Tensor): Target labels (1 for positive, -1 for unlabeled).
        
        Returns:
            torch.Tensor: Computed loss.
        """
        positive = (t == 1).float()
        unlabeled = (t == -1).float()

        n_positive = positive.sum().clamp(min=1)
        n_unlabeled = unlabeled.sum().clamp(min=1)

        # Compute risks
        y_positive = torch.sigmoid(-x)  # Loss for positive
        y_unlabeled = torch.sigmoid(x)  # Loss for unlabeled

        positive_risk = (self.prior * positive / n_positive * y_positive).sum()
        negative_risk = ((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled).sum()

        objective = positive_risk + negative_risk

        if self.nnpu:
            if negative_risk < -self.beta:
                objective = positive_risk - self.beta

        if self.imbpu:
            weight_pos = self.alpha / self.prior
            weight_neg = (1 - self.alpha) / (1 - self.prior) 
            objective = weight_pos * positive_risk + weight_neg * negative_risk

        return objective
