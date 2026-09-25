"""Sampling pipeline on fp32 logits (GN-3, GN-4)."""
from __future__ import annotations

import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Step 1: pad and undefined ids -> -inf. logits (B, V) fp32; allowed (V,) bool."""
    return logits.masked_fill(~allowed[None, :], float("-inf"))


def apply_repetition_penalty(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """Step 2 (CTRL form): for ids already in the sequence, positive logits are divided by the
    penalty and negative ones multiplied. `seen` (B, V) bool."""
    if penalty == 1.0:
        return logits
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) > p           # keep the smallest prefix whose mass reaches p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)


def sample_next(logits: torch.Tensor, allowed: torch.Tensor, seen: torch.Tensor, temperature: float,
                top_k: int, top_p: float, repetition_penalty: float, generator: torch.Generator):
    """Returns (next_ids (B,), logprobs_of_next (B,)). logprobs follow GN-4: unfiltered model
    distribution at temperature 1 with only step-1 masking."""
    logits = mask_logits(logits.float(), allowed)
    base_logprobs = torch.log_softmax(logits, dim=-1)
    x = apply_repetition_penalty(logits, seen, repetition_penalty)
    if temperature == 0:
        nxt = torch.argmax(x, dim=-1)
    else:
        x = x / temperature
        x = top_k_filter(x, top_k)
        x = top_p_filter(x, top_p)
        probs = torch.softmax(x, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return nxt, base_logprobs.gather(-1, nxt[:, None]).squeeze(-1)
