"""SimPO loss (PO-4): L = -log sigmoid(beta/|y_c| logp_c - beta/|y_r| logp_r - gamma)."""
from __future__ import annotations

import torch.nn.functional as F


def simpo_loss(pol_c, pol_r, len_c, len_r, beta: float, gamma: float):
    rc = beta * pol_c / len_c.float()
    rr = beta * pol_r / len_r.float()
    loss = -F.logsigmoid(rc - rr - gamma)
    return loss.mean(), {"chosen_reward": rc.detach().mean().item(), "rejected_reward": rr.detach().mean().item(),
                         "reward_margin": (rc - rr).detach().mean().item(),
                         "accuracy": (rc > rr).float().mean().item()}
