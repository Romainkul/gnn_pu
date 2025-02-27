import torch
import torch.nn as nn
import torch.nn.functional as F

class PULoss(nn.Module):
    """
    Implements the non-negative unbiased PU risk estimator with an option for an imbalanced version.

    The standard (nnPU) loss is given by:
        risk = π * E[L(g(x)) for positive] + max{0, E[L(-g(x)) for unlabeled] - π * E[L(-g(x)) for positive]}
    and the imbalanced version reweights the two risk components:
        risk = ω_p * (π * E[L(g(x)) for positive])
             + ω_n * (E[L(-g(x)) for unlabeled] - π * E[L(-g(x)) for positive])
    where ω_p = α / π and ω_n = (1 - α) / (1 - π).

    Args:
        prior (float): Estimated class prior (0 < prior < 1).
        beta (float): Threshold for non-negative risk correction (default=0).
        nnpu (bool): If True, applies non-negative risk correction.
        imbpu (bool): If True, uses the imbalanced version of the risk estimator.
        alpha (float): Mixing parameter for imbalanced PU (used only if imbpu=True; default=0.5).
    """
    def __init__(self, prior, beta=0.0, nnpu=True, imbpu=False, alpha=0.5):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.beta = beta
        self.nnpu = nnpu
        self.imbpu = imbpu
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Computes the PU loss.

        Args:
            logits (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Target labels, where positives are 1 and unlabeled samples are -1.
        
        Returns:
            torch.Tensor: A scalar loss.
        """
        # Create masks for positive and unlabeled examples
        pos_mask = (targets == 1)
        unl_mask = (targets == -1)

        # Compute binary cross-entropy losses.
        # For positives: we compute loss when treating them as positive and as negative.
        if pos_mask.sum() > 0:
            # Loss on positive examples (as positive)
            loss_pos = F.binary_cross_entropy_with_logits(
                logits[pos_mask],
                torch.ones_like(logits[pos_mask]),
                reduction='mean'
            )
            # Loss on positive examples if treated as negative
            loss_neg_pos = F.binary_cross_entropy_with_logits(
                logits[pos_mask],
                torch.zeros_like(logits[pos_mask]),
                reduction='mean'
            )
        else:
            # If no positive samples, use 0 (or raise an error as appropriate)
            loss_pos = torch.tensor(0.0, device=logits.device)
            loss_neg_pos = torch.tensor(0.0, device=logits.device)

        # For unlabeled samples, treat them as negatives.
        if unl_mask.sum() > 0:
            loss_unl = F.binary_cross_entropy_with_logits(
                logits[unl_mask],
                torch.zeros_like(logits[unl_mask]),
                reduction='mean'
            )
        else:
            loss_unl = torch.tensor(0.0, device=logits.device)

        # Compute the two risk components:
        # Positive risk: π * loss on positive examples (as positive)
        risk_positive = self.prior * loss_pos
        # Negative risk: loss on unlabeled samples minus π * loss on positives if treated as negative.
        risk_negative = loss_unl - self.prior * loss_neg_pos

        # Optionally apply imbalanced reweighting:
        if self.imbpu:
            weight_pos = self.alpha / self.prior
            weight_neg = (1 - self.alpha) / (1 - self.prior)
            # Compute the reweighted risk.
            risk = weight_pos * risk_positive + weight_neg * risk_negative
            # If nnPU correction is enabled, clip the reweighted negative risk.
            if self.nnpu and risk_negative < -self.beta:
                risk = weight_pos * risk_positive - self.beta
        else:
            # Standard nnPU risk: clip the negative part if needed.
            if self.nnpu and risk_negative < -self.beta:
                risk = risk_positive - self.beta
            else:
                risk = risk_positive + risk_negative

        return risk
