"""Distributed training: ZeRO / FSDP integration (per CS336 lecture_08's ZeRO stages 1-3).

Mapping used here:
  zero_stage 0 -> plain DDP. No state sharding -- every rank holds full params, grads,
                  optimizer state. Baseline "naive data parallel" from the lecture.
  zero_stage 1 -> DDP + optimizer-state sharding only, via
                  torch.distributed.optim.ZeroRedundancyOptimizer. Matches the lecture's P_os:
                  everyone has full params + grads, but each rank only owns/updates a slice
                  of the optimizer state.
  zero_stage 2 -> FSDP with ShardingStrategy.SHARD_GRAD_OP. Shards optimizer state AND
                  gradients; full parameters are still materialized (all-gathered) during the
                  forward/backward compute, matching the lecture's P_os+g.
  zero_stage 3 -> FSDP with ShardingStrategy.FULL_SHARD ("aka FSDP" in the lecture, P_os+g+p):
                  shards parameters too, all-gathering/freeing them layer by layer.

Single-process runs (no torchrun / no WORLD_SIZE env var) skip all of this: wrap_model and
build_optimizer_for_zero both fall through to plain, unwrapped model/optimizer, since there's
nothing to shard across one process. main.py calls setup_distributed() unconditionally; it is
a no-op in that case.

Honest caveat: this has only been exercised in this sandbox via a local 2-process, CPU/gloo
run (see tests/test_distributed_smoke.py) -- not against real multi-GPU/NCCL hardware, and
not at the scale (many nodes, large models) the lecture actually discusses.
"""
from __future__ import annotations
import os, functools
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Block


def is_distributed() -> bool:
  return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
  return (not is_distributed()) or dist.get_rank() == 0


def setup_distributed() -> Tuple[int, int, int]:
  """Reads standard torchrun env vars (RANK, WORLD_SIZE, LOCAL_RANK). Returns
  (rank, world_size, local_rank). If WORLD_SIZE isn't set (ordinary single-process run),
  returns (0, 1, 0) without touching torch.distributed at all -- safe to call unconditionally."""
  if "WORLD_SIZE" not in os.environ:
    return 0, 1, 0
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  backend = "nccl" if torch.cuda.is_available() else "gloo"
  if not dist.is_initialized():
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
  return rank, world_size, local_rank


def wrap_model(model: nn.Module, zero_stage: int, device: torch.device) -> nn.Module:
  """Wraps `model` per `zero_stage`. Falls through to a plain .to(device) with no wrapper at
  all outside a distributed context (single-process run)."""
  if not is_distributed():
    return model.to(device)

  if zero_stage in (0, 1):
    # zero_stage 1's optimizer-state sharding is handled entirely on the optimizer side
    # (build_optimizer_for_zero below) -- the model itself is wrapped in plain DDP for both
    # stage 0 and stage 1, exactly as the lecture describes (DDP for stage 1, just with a
    # smarter optimizer).
    return DDP(model.to(device))

  # stage 2 / 3 -> FSDP. Auto-wrap at transformer Block boundaries so each block is its own
  # FSDP unit -- this is what makes the incremental "all-gather this block, compute, free"
  # pattern from the lecture possible. Wrapping the whole model as a single FSDP unit would
  # all-gather everything at once and lose the memory benefit.
  sharding_strategy = ShardingStrategy.SHARD_GRAD_OP if zero_stage == 2 else ShardingStrategy.FULL_SHARD
  auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
  return FSDP(
    model,
    sharding_strategy=sharding_strategy,
    auto_wrap_policy=auto_wrap_policy,
    device_id=device,   # explicit device_id (even torch.device("cpu")) skips FSDP's
                         # accelerator-autodetection, which otherwise refuses to init on a
                         # machine with no GPU/accelerator at all -- needed for CPU/gloo runs
  )


def build_optimizer_for_zero(model: nn.Module, zero_stage: int, lr: float, weight_decay: float,
                              betas=(0.9, 0.95), eps: float = 1e-8):
  """zero_stage 1 needs a different optimizer constructor: ZeroRedundancyOptimizer wraps a
  regular optimizer class and shards its state across ranks. Stages 0/2/3 just build plain
  AdamW -- for 2/3, FSDP already did the sharding at the model-wrapping level (wrap_model
  above), so the optimizer itself doesn't need to know about it.

  Weight-decay param-group split (norms/1-D params excluded, matching train.build_optimizer)
  is preserved for stages 0/2/3. It is NOT preserved for stage 1: ZeroRedundancyOptimizer
  takes one flat parameter list plus a single set of kwargs applied to all of them, so under
  zero_stage=1 every parameter (including norms) gets the same weight_decay. This is a real,
  documented tradeoff of PyTorch's ZeRO-1 wrapper, not an oversight -- flagged again in the
  README.
  """
  decay, no_decay = [], []
  for _, p in model.named_parameters():
    if not p.requires_grad:
      continue
    (no_decay if p.dim() < 2 else decay).append(p)

  if is_distributed() and zero_stage == 1:
    return ZeroRedundancyOptimizer(
      decay + no_decay, optimizer_class=torch.optim.AdamW,
      lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
    )

  param_groups = [{"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0}]
  fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
  return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=fused)


def full_model_state_dict(model: nn.Module) -> Optional[dict]:
  """Returns a full (unsharded) model state_dict. Under FSDP this is a COLLECTIVE call --
  every rank must call it (it triggers an all-gather), even though with rank0_only=True only
  rank 0's return value is actually populated; other ranks get an empty dict. Callers must
  call this unconditionally on every rank and only act on the result where is_main_process().

  This is the lecture's 'baby version' of FSDP checkpointing -- simplest to get right, but
  gathering the full state dict onto one rank doesn't scale to the largest models. A
  production setup would use StateDictType.SHARDED_STATE_DICT and write one shard file per
  rank instead of gathering here.
  """
  if isinstance(model, FSDP):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
      return model.state_dict()
  if isinstance(model, DDP):
    return model.module.state_dict()
  return model.state_dict()


def load_full_model_state_dict(model: nn.Module, full_msd: dict) -> None:
  """Loading counterpart to full_model_state_dict. Under FSDP, every rank must call
  model.load_state_dict with the SAME full state dict inside the FULL_STATE_DICT context --
  unlike saving, loading has no rank0_only broadcast built in, so the caller must broadcast
  full_msd (e.g. via torch.distributed.broadcast_object_list) from rank 0 before calling this."""
  if isinstance(model, FSDP):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
      model.load_state_dict(full_msd)
  elif isinstance(model, DDP):
    model.module.load_state_dict(full_msd)
  else:
    model.load_state_dict(full_msd)


def full_optimizer_state_dict(model: nn.Module, optimizer) -> Optional[dict]:
  """Optimizer-side counterpart to full_model_state_dict -- also collective under FSDP/ZeRO.
  Call unconditionally on every rank; only rank 0's return value is populated."""
  if isinstance(optimizer, ZeroRedundancyOptimizer):
    optimizer.consolidate_state_dict(to=0)   # collective
    return optimizer.state_dict() if is_main_process() else None
  if isinstance(model, FSDP):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
      return FSDP.optim_state_dict(model, optimizer)
  return optimizer.state_dict()


def load_full_optimizer_state_dict(model: nn.Module, optimizer, full_osd: Optional[dict]):
  """Loading counterpart: reshards a full optimizer state dict (loaded on rank 0, as
  full_osd) back onto each rank's local optimizer shard. Also collective under FSDP -- call
  on every rank; ranks other than 0 pass full_osd=None and FSDP broadcasts internally."""
  if isinstance(model, FSDP):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
      sharded_osd = FSDP.optim_state_dict_to_load(model, optimizer, full_osd)
    optimizer.load_state_dict(sharded_osd)
  else:
    optimizer.load_state_dict(full_osd)
