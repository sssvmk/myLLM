"""Assistant-only cross-entropy (S2-1). The loop sums `sft_loss_sum` over micro-batches and
divides by the global target count of the optimizer step (all micro-batches, all ranks)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss_sum(logits, targets, loss_mask):
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    m = loss_mask.reshape(-1).float()
    return (ce * m).sum(), m.sum()
