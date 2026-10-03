from collections.abc import Iterable

import torch
from torch import nn

from counterfraud.losses.focal import FocalLoss
from counterfraud.losses.weighted_bce import WeightedBCEWithLogitsLoss


class CompositeLoss(nn.Module):
    """L_total = L_task(logits, y) + gamma * L_task(logits_int, y) + eta * ||w||_2^2."""

    def __init__(
        self,
        base_loss: str = "focal",
        gamma: float = 1.0,
        eta: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
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
