"""Group advantages (RL-4) and the GSPO sequence-level clipped objective (RL-7), with the
optional KL-to-parent term (RL-8)."""
from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, std_normalization: bool, eps: float) -> torch.Tensor:
    """rewards (N,) ordered group by group. Std is the unbiased (Bessel-corrected) group std,
    matching TRL's GRPOTrainer (IG-3 note)."""
    r = rewards.float().view(-1, group_size)
    centered = r - r.mean(dim=1, keepdim=True)
    if std_normalization:
        centered = centered / (r.std(dim=1, keepdim=True, unbiased=True) + eps)
    return centered.view(-1)


def gspo_per_sequence(logp_new: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor,
                      eps_low: float, eps_high: float, logp_ref: torch.Tensor = None, kl_coef: float = 0.0):
    """Per-response loss vector (N,) and a bool vector marking where the clipped term is selected."""
    m = mask.float()
    n_tok = m.sum(-1).clamp_min(1.0)
    log_ratio = ((logp_new - logp_old) * m).sum(-1) / n_tok
    s = torch.exp(log_ratio)
    unclipped = s * adv
    clipped = torch.clamp(s, 1 - eps_low, 1 + eps_high) * adv
    per_seq = -torch.minimum(unclipped, clipped)
    if kl_coef > 0:
        if logp_ref is None:
            raise ValueError("kl_coef > 0 requires logp_ref")
        per_seq = per_seq + kl_coef * ((logp_new - logp_ref) * m).sum(-1) / n_tok
    return per_seq, clipped < unclipped


def gspo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor,
              eps_low: float, eps_high: float, logp_ref: torch.Tensor = None, kl_coef: float = 0.0):
    """logp_* (N, T) fp32 per-token log-probs; mask (N, T) bool; adv (N,)."""
    m = mask.float()
    n_tok = m.sum(-1).clamp_min(1.0)
    log_ratio = ((logp_new - logp_old) * m).sum(-1) / n_tok
    s = torch.exp(log_ratio)
    unclipped = s * adv
    clipped = torch.clamp(s, 1 - eps_low, 1 + eps_high) * adv
    per_seq = -torch.minimum(unclipped, clipped)
    if kl_coef > 0:
        if logp_ref is None:
            raise ValueError("kl_coef > 0 requires logp_ref")
        per_seq = per_seq + kl_coef * ((logp_new - logp_ref) * m).sum(-1) / n_tok
    clip_fraction = (clipped < unclipped).float().mean()
    return per_seq.mean(), {"clip_fraction": clip_fraction.item(), "ratio_mean": s.detach().mean().item()}


def gspo_per_sequence(logp_new, logp_old, mask, adv, eps_low, eps_high, logp_ref=None, kl_coef=0.0):
    """Per-response loss vector (N,) and a bool vector of responses whose clipped term was selected.
    `gspo_loss` is the mean of this vector; Stage 4 weights it by a validity mask instead (RW-5
    exclude mode) so excluded responses keep the micro-batch structure identical across ranks."""
    m = mask.float()
    n_tok = m.sum(-1).clamp_min(1.0)
    s = torch.exp(((logp_new - logp_old) * m).sum(-1) / n_tok)
    unclipped = s * adv
    clipped = torch.clamp(s, 1 - eps_low, 1 + eps_high) * adv
    per_seq = -torch.minimum(unclipped, clipped)
    if kl_coef > 0:
        per_seq = per_seq + kl_coef * ((logp_new - logp_ref) * m).sum(-1) / n_tok
    return per_seq, (clipped < unclipped)
