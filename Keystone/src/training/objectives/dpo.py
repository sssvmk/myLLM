"""DPO loss (PO-3): L = -log sigmoid(beta * (Delta_c - Delta_r)), Delta = logp_theta - ref_logp."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float):
    rc = beta * (pol_c - ref_c)
    rr = beta * (pol_r - ref_r)
    loss = -F.logsigmoid(rc - rr)
    return loss.mean(), {"chosen_reward": rc.detach().mean().item(), "rejected_reward": rr.detach().mean().item(),
                         "reward_margin": (rc - rr).detach().mean().item(),
                         "accuracy": (rc > rr).float().mean().item()}
