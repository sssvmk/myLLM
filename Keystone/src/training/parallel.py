"""Model wrapping and optimizer construction for ZeRO stages (TR-3, TR-5, TR-5a).

Deviation from TR-5a / TR-3 (see docs/implementation_notes.md):
  * ZeRO 2/3 cannot use foundation_llm's `wrap_model`: it builds FSDP with
    use_orig_params=False, which (a) refuses to flatten DeepSeekMoE blocks because
    `routing_bias` has requires_grad=False ("Must flatten tensors with uniform requires_grad"),
    and (b) exposes only 1-D FlatParameters, so `build_optimizer_for_zero`'s `p.dim() < 2`
    split puts every parameter in the no-decay group. Both reproduced with torch 2.14.
  * So stages 2/3 are wrapped here with use_orig_params=True, and the decay/no-decay split is
    decided from parameter names and shapes captured BEFORE wrapping, for every ZeRO stage
    (ZeroRedundancyOptimizer also receives the two groups instead of one flat list).
"""
from __future__ import annotations

import functools
from typing import Dict, Set

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..foundation import bridge
from ..modeling.loading import strip_prefixes
from .moe_balance import routing_bias_params


def decay_param_names(model: nn.Module) -> Set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2}


def wrap_for_training(model: nn.Module, zero_stage: int, device: torch.device, find_unused_parameters: bool) -> nn.Module:
    fd = bridge.fl_distributed
    if not fd.is_distributed():
        return model.to(device)
    if zero_stage in (0, 1):
        dev_ids = [device.index] if device.type == "cuda" else None
        return DDP(model.to(device), device_ids=dev_ids, find_unused_parameters=find_unused_parameters)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    strategy = ShardingStrategy.SHARD_GRAD_OP if zero_stage == 2 else ShardingStrategy.FULL_SHARD
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={bridge.fl_model.Block})
    # routing_bias stays out of FSDP: sharded, it is zero-size on all but one rank and update_bias()
    # would raise. Ignored states must already be on `device`, hence the explicit .to().
    model = model.to(device)
    ignored = routing_bias_params(model)
    return FSDP(model, sharding_strategy=strategy, auto_wrap_policy=policy, device_id=device,
                use_orig_params=True, ignored_states=ignored or None)


def build_optimizer(wrapped: nn.Module, zero_stage: int, decay_names: Set[str], optim_cfg):
    decay, no_decay = [], []
    seen: Set[int] = set()
    for n, p in wrapped.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        clean = next(iter(strip_prefixes({n: None})))
        (decay if clean in decay_names else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": optim_cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=optim_cfg.lr, betas=tuple(optim_cfg.betas), eps=optim_cfg.eps)
    if bridge.fl_distributed.is_distributed() and zero_stage == 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW,
                                      weight_decay=optim_cfg.weight_decay, **kw)
        opt.add_param_group(groups[1])
        return opt
    fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    return torch.optim.AdamW(groups, fused=fused, **kw)
