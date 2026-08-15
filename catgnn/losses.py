from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        targets = targets.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE with a scalar pos_weight, eg (num_neg / num_pos) for the batch
    or the full training set.
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        targets = targets.float()
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class CompositeLoss(nn.Module):
    """Task loss + causal-invariant consistency term + optional explicit L2."""

    def __init__(
        self,
        base_loss: str = "focal",
        gamma: float = 1.0,
        eta: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float| None = None,
    ):
        super().__init__()
        if base_loss == "focal":
            self.task_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif base_loss == "weighted_bce":
            self.task_loss = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"unknown base_loss '{base_loss}', expected 'focal' or 'weighted_bce'")
        self.gamma = gamma
        self.eta = eta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logits_intervened: torch.Tensor | None = None,
        model_parameters: Iterable[torch.nn.Parameter] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_task = self.task_loss(logits, targets)
        total = l_task
        components = {"task_loss": l_task.detach()}

        if logits_intervened is not None:
            l_inv = self.task_loss(logits_intervened, targets)
            total = total + self.gamma * l_inv
            components["invariant_loss"] = l_inv.detach()

        if self.eta > 0.0 and model_parameters is not None:
            l2 = sum(p.pow(2).sum() for p in model_parameters if p.requires_grad)
            total = total + self.eta * l2
            components["l2_penalty"] = l2.detach()

        components["total_loss"] = total.detach()
        return total, components
