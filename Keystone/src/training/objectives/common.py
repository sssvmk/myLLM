"""Shared log-probability helpers. All log-probs are fp32 (TR-4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logps(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(B, T, V) logits, (B, T) targets -> (B, T) fp32 log p(target)."""
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)


def sequence_logps(logits, targets, mask) -> torch.Tensor:
    """Summed response log-prob per sequence (PO-2). mask (B, T) bool."""
    return (token_logps(logits, targets) * mask.float()).sum(-1)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def shift(tokens: torch.Tensor, mask: torch.Tensor):
    """inputs, targets, target mask for next-token prediction (DL-2)."""
    return tokens[:, :-1], tokens[:, 1:], mask[:, 1:]
