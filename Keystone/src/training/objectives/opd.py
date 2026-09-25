"""On-policy distillation, per-token reverse KL in REINFORCE form (DI-5):
L = mean over response tokens of sg(logp_s - logp_t) * logp_s."""
from __future__ import annotations

import torch


def opd_loss(logp_s: torch.Tensor, logp_t: torch.Tensor, mask: torch.Tensor):
    m = mask.float()
    adv = (logp_s - logp_t).detach()
    denom = m.sum().clamp_min(1.0)
    loss = (adv * logp_s * m).sum() / denom
    return loss, {"reverse_kl_per_token": ((adv * m).sum() / denom).item()}
