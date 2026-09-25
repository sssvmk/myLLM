"""Routing-bias balancing for DeepSeekMoE under data parallelism (PT-9).

Why this exists (docs/prd_review.md #19, reproduced with torch 2.14, 2-process gloo):
  * `DeepSeekMoE.update_bias()` reads `_last_usage`, the token counts of the *local* last
    micro-batch. With data parallelism each rank sees different tokens, so the biases drift apart
    between ranks (routing_bias has requires_grad=False, so neither DDP nor FSDP synchronises it).
  * Under FSDP `use_orig_params=True` the bias is flattened into a sharded parameter: one rank holds
    all of it and the others see a zero-size view, so `update_bias()` raises there.

Fix: keep `routing_bias` out of FSDP (`ignored_states`, see parallel.wrap_for_training), sum usage
over all micro-batches of the optimizer step and over all ranks, and apply the identical update on
every rank. foundation_llm is not modified; only its public method `update_bias` is called.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..foundation import bridge


def moe_modules(model: nn.Module) -> list:
    """Every DeepSeekMoE in `model`, through DDP / FSDP / torch.compile wrappers."""
    moe_cls = bridge.fl_moe.DeepSeekMoE
    return [m for m in model.modules() if isinstance(m, moe_cls)]


def routing_bias_params(model: nn.Module) -> List[nn.Parameter]:
    return [m.routing_bias for m in moe_modules(model)]


class UsageAccumulator:
    """Call `add()` after every micro-batch forward and `apply()` once per optimizer step."""

    def __init__(self, model: nn.Module):
        self.mods = moe_modules(model)
        self._acc: Optional[torch.Tensor] = None

    def add(self) -> None:
        if not self.mods:
            return
        usage = torch.stack([m._last_usage.float() for m in self.mods])      # (layers, n_routed)
        self._acc = usage if self._acc is None else self._acc + usage

    def apply(self) -> None:
        """Sum usage over ranks, hand each layer its global usage, then update the biases."""
        if not self.mods or self._acc is None:
            self._acc = None
            return
        acc = self._acc
        if bridge.fl_distributed.is_distributed():
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        for m, u in zip(self.mods, acc):
            m._last_usage = u
            m.update_bias()
        self._acc = None


# --------------------------------------------------------------------------- differentiable balance loss (TR-2, PT-9)
def _capture_scores(moe: nn.Module):
    def hook(_mod, _inp, out):
        moe._pf_scores = out                      # router logits, still attached to the autograd graph
    return hook


def _balance_hook(moe: nn.Module, _inp, out):
    """Replaces the aux loss `DeepSeekMoE.forward` returns (built from token counts: no gradient,
    docs/prd_review.md #3) with an expert-level balance loss in the DeepSeek-V2 form

        L = aux_loss_weight * n_routed * sum_i f_i * P_i

    f_i: fraction of the (token, slot) assignments that went to expert i (from the same biased
    top-k selection the forward used; a constant), P_i: mean router probability of expert i over
    the tokens (differentiable). L is 1 * weight for perfectly uniform routing, larger when
    imbalanced, and its gradient pushes router probability away from over-used experts. With
    weight 0 the module's own output is left untouched."""
    scores = moe.__dict__.pop("_pf_scores", None)
    w = float(getattr(moe, "aux_loss_weight", 0.0))
    if scores is None or w == 0.0:
        return None
    out_x = out[0]
    scores = scores.float()
    probs = torch.softmax(scores, dim=-1)                                  # (N, n_routed)
    idx = (scores.detach() + moe.routing_bias.detach().float()).topk(moe.top_k, dim=-1).indices
    counts = torch.zeros(moe.n_routed, device=scores.device).scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=scores.device))
    f = counts / max(idx.numel(), 1)
    return out_x, w * moe.n_routed * (f * probs.mean(0)).sum()


def install_balance_loss(model: nn.Module) -> int:
    """Attach the hooks above to every DeepSeekMoE (idempotent). Returns how many were installed.
    foundation_llm is not modified: the hooks read the router's output and replace the module's
    returned (output, aux_loss) tuple; `Block.forward` and `GPTModel.forward` already sum it."""
    n = 0
    for m in moe_modules(model):
        if getattr(m, "_pf_balance_installed", False):
            continue
        m.router.register_forward_hook(_capture_scores(m))
        m.register_forward_hook(_balance_hook)
        m._pf_balance_installed = True
        n += 1
    return n
