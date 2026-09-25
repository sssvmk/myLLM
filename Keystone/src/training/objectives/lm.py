"""Next-token cross-entropy over non-padding targets (S1-1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    m = target_mask.reshape(-1).float()
    return (ce * m).sum() / m.sum().clamp_min(1.0)
