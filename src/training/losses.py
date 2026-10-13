from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """[MOD-2] Multi-class Focal Loss: FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] = (0.25, 0.40, 0.35),
        num_classes: int = 3,
    ):
        super().__init__()
        assert len(alpha) == num_classes, (
            f"len(alpha)={len(alpha)} must equal num_classes={num_classes}"
        )
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        probs = log_probs.exp()
        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)
        alpha_t = self.alpha.to(logits.device)[targets]                # (B,)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()
