"""DeepSeekMoE-style feed-forward layer: fine-grained routed experts + always-on shared
experts, SwiGLU expert MLPs, and auxiliary-loss-free load balancing via a per-expert bias
term (DeepSeek-V3 style) instead of a traditional load-balancing loss.

This is a simplified reproduction of the *mechanism*, not the exact DeepSeek dimensions or
hyperparameters -- expert counts/widths are config knobs (see config.py), not fixed to any
particular published DeepSeek checkpoint.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()
    self.gate = nn.Linear(d_model, d_ff, bias=False)
    self.up = nn.Linear(d_model, d_ff, bias=False)
    self.down = nn.Linear(d_ff, d_model, bias=False)

  def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))


class DeepSeekMoE(nn.Module):
  """
  Routes each token to `top_k` of `n_routed_experts`, always also runs every token through
  `n_shared_experts` dense experts, and sums the two.

  Load balancing uses DeepSeek-V3's auxiliary-loss-free scheme: a learnable, non-gradient
  per-expert bias is added to routing scores before top-k *selection* (but NOT before the
  softmax weights used to *combine* expert outputs), and is nudged up/down after each step
  based on how over/under-used each expert was -- see `update_bias`, which the training loop
  calls once per optimizer step. Keeping the balancing signal out of the backward pass means
  it doesn't trade off against language-modeling loss quality the way a classic auxiliary
  load-balancing loss can.

  A small variance-based auxiliary term is still added to `forward`'s returned loss as a
  supplement (not a replacement) for the bias mechanism, mainly useful early in training
  before the bias has had time to adapt.
  """
  def __init__(self, d_model: int, d_ff: int, n_routed_experts: int = 8,
               n_shared_experts: int = 1, top_k: int = 2, expert_d_ff: int = 0,
               bias_update_rate: float = 1e-3, aux_loss_weight: float = 0.01):
    super().__init__()
    self.n_routed = n_routed_experts
    self.top_k = top_k
    self.bias_update_rate = bias_update_rate
    self.aux_loss_weight = aux_loss_weight
    # routed experts are narrower than a dense MLP, since several fire per token
    # (that's the "fine-grained experts" part of DeepSeekMoE)
    edf = expert_d_ff or max(d_ff // 2, 64)

    self.router = nn.Linear(d_model, n_routed_experts, bias=False)
    self.routing_bias = nn.Parameter(torch.zeros(n_routed_experts), requires_grad=False)
    self.routed_experts = nn.ModuleList([SwiGLUExpert(d_model, edf) for _ in range(n_routed_experts)])
    self.shared_experts = nn.ModuleList([SwiGLUExpert(d_model, edf) for _ in range(n_shared_experts)])

    self._last_usage = None  # populated each forward, consumed by update_bias

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, C = x.shape
    flat = x.reshape(-1, C)  # (N, C), N = B*T

    scores = self.router(flat)                                 # (N, n_routed)
    probs = F.softmax(scores, dim=-1)
    biased = scores + self.routing_bias                          # bias only affects selection
    topk_val, topk_idx = biased.topk(self.top_k, dim=-1)          # (N, top_k)
    topk_probs = probs.gather(-1, topk_idx)                        # combine weights use unbiased softmax
    topk_probs = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-9)

    out = torch.zeros_like(flat)
    usage = torch.zeros(self.n_routed, device=x.device)
    for e in range(self.n_routed):
      sel = (topk_idx == e)                                       # (N, top_k) bool
      if not sel.any():
        continue
      token_mask = sel.any(dim=-1)                                 # (N,) tokens routed to expert e
      weight = (topk_probs * sel).sum(-1)                          # (N,) combine weight (0 if unrouted)
      usage[e] = token_mask.sum()
      out[token_mask] += weight[token_mask].unsqueeze(-1) * self.routed_experts[e](flat[token_mask])

    for se in self.shared_experts:
      out = out + se(flat)

    out = out.reshape(B, T, C)
    self._last_usage = usage.detach()

    load = usage / usage.sum().clamp_min(1)
    aux_loss = load.float().var() * self.aux_loss_weight
    return out, aux_loss

  @torch.no_grad()
  def update_bias(self):
    """Call once per optimizer step (train.py does this automatically): nudges each expert's
    routing bias up if it was under-used relative to the mean usage this step, down if
    over-used. Non-differentiable by design."""
    if self._last_usage is None:
      return
    usage = self._last_usage
    mean_usage = usage.mean()
    update = torch.sign(mean_usage - usage) * self.bias_update_rate
    self.routing_bias += update
