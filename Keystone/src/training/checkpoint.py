"""Resumable checkpoints (CK-1, CK-3..CK-6).

`save_training_checkpoint` is COLLECTIVE (every rank calls it): the full-state-dict helpers of
foundation_llm gather onto rank 0 under FSDP/ZeRO-1, and each rank's RNG state and data position
are gathered with all_gather_object; only rank 0 writes.

File layout under `<output_uri>/checkpoints/`: step_000500.pt, latest.pt (copy of the newest),
best.pt (weights only; kept by objectives that select on validation).
"""
from __future__ import annotations

import os
import random
import tempfile
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from ..foundation import bridge
from ..io.storage import Storage
from ..modeling.loading import strip_prefixes

FORMAT = 1


def _is_fsdp(model) -> bool:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return isinstance(model, FSDP)


NAMED_FORMAT = "named_full_v1"


def _clean(name: str) -> str:
    return next(iter(strip_prefixes({name: None})))


def _optimizer_params(model, optimizer):
    """[(clean name, parameter)] for the parameters the optimizer owns, in optimizer order."""
    by_id = {id(p): _clean(n) for n, p in model.named_parameters()}
    return [(by_id[id(p)], p) for g in optimizer.param_groups for p in g["params"]]


def gather_named_optimizer_state(model, optimizer, full_model_sd) -> Optional[Dict[str, Any]]:
    """Full optimizer state keyed by parameter name, assembled by hand from each rank's local shards.
    COLLECTIVE (gather to rank 0); only rank 0 gets the result.

    Why not `FSDP.optim_state_dict` / `get_optimizer_state_dict`: with FSDP(use_orig_params=True) both
    return a full state that is wrong (measured on torch 2.14, 2 processes: for a 2-layer Linear, the
    gathered exp_avg differs from the ground truth computed with a plain AdamW step by as much as
    the values themselves; with use_orig_params=False they are exact). Training under
    use_orig_params=True is itself correct (parameters and local moment shards equal the plain
    reference exactly), only the gather is broken (docs/prd_review.md #20).

    Under FSDP a parameter's local optimizer state is one contiguous slice of its flattened tensor,
    and slices follow rank order, so concatenating the ranks' pieces in rank order restores the
    full tensor. The total size is checked against the model's full parameter shape, so a layout
    that does not fit this assumption fails loudly instead of saving a wrong state."""
    fd = bridge.fl_distributed
    local: Dict[str, Dict[str, Any]] = {}
    for name, p in _optimizer_params(model, optimizer):
        if p in optimizer.state and p.numel() > 0:
            local[name] = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                           for k, v in optimizer.state[p].items()}
    if fd.is_distributed():
        gathered: Optional[List[Any]] = [None] * dist.get_world_size() if fd.is_main_process() else None
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if not fd.is_main_process():
        return None
    state: Dict[str, Dict[str, Any]] = {}
    names = []
    for g in gathered:
        for n in g:
            if n not in names:
                names.append(n)
    for n in names:
        pieces = [g[n] for g in gathered if n in g]
        shape = tuple(full_model_sd[n].shape)
        entry: Dict[str, Any] = {}
        for key in pieces[0]:
            v0 = pieces[0][key]
            if isinstance(v0, torch.Tensor) and v0.dim() > 0:
                flat = torch.cat([pc[key].flatten() for pc in pieces])
                if flat.numel() != int(torch.tensor(shape).prod()):
                    raise RuntimeError(f"optimizer state for {n!r}: shards add up to {flat.numel()} elements, parameter has "
                                       f"{tuple(shape)}; FSDP layout differs from the assumed rank-ordered contiguous shards")
                entry[key] = flat.reshape(shape)
            else:
                entry[key] = v0
        state[n] = entry
    return {"format": NAMED_FORMAT, "state": state}


def load_named_optimizer_state(model, optimizer, named: Dict[str, Any], sharded: bool) -> None:
    """Inverse of gather_named_optimizer_state, for ANY world size. Each rank takes the slice of
    every full tensor that its own (possibly re-sharded) local parameter view covers: slices are
    rank-ordered, so a rank's offset is the sum of the local sizes of the ranks before it."""
    fd = bridge.fl_distributed
    params = _optimizer_params(model, optimizer)
    offsets = [0] * len(params)
    if sharded and fd.is_distributed():
        mine = [p.numel() for _, p in params]
        allk: List[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(allk, mine)
        offsets = [sum(allk[r][i] for r in range(dist.get_rank())) for i in range(len(params))]
    # index-keyed groups built by hand: ZeroRedundancyOptimizer.state_dict() needs a consolidate first
    groups, k = [], 0
    for g in optimizer.param_groups:
        groups.append(dict({kk: vv for kk, vv in g.items() if kk != "params"}, params=list(range(k, k + len(g["params"])))))
        k += len(g["params"])
    osd: Dict[str, Any] = {"param_groups": groups}
    new_state: Dict[int, Dict[str, Any]] = {}
    for i, (name, p) in enumerate(params):
        ent = named["state"].get(name)
        if ent is None or p.numel() == 0:
            continue
        piece: Dict[str, Any] = {}
        for key, v in ent.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                piece[key] = v.flatten()[offsets[i]:offsets[i] + p.numel()].reshape(p.shape).clone()
            else:
                piece[key] = v.clone() if isinstance(v, torch.Tensor) else v
        new_state[i] = piece
    osd["state"] = new_state
    optimizer.load_state_dict(osd)


def _clone_state(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: _clone_state(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_state(v) for v in x]
    return x


def optimizer_payload(model, optimizer, full_model_sd):
    """The optimizer state keyed by parameter name (`named_full_v1`) for every ZeRO stage, so a
    checkpoint can be resumed under a different stage or world size. COLLECTIVE.
    FSDP: assembled from the ranks' shards. ZeRO-1: consolidated onto rank 0 first. DDP/single
    process: the (replicated) state of rank 0."""
    fd = bridge.fl_distributed
    if _is_fsdp(model):
        return gather_named_optimizer_state(model, optimizer, full_model_sd)
    from torch.distributed.optim import ZeroRedundancyOptimizer
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=0)
    if not fd.is_main_process():
        return None
    names = [n for n, _ in _optimizer_params(model, optimizer)]
    osd = optimizer.state_dict()
    return {"format": NAMED_FORMAT, "state": {names[i]: _clone_state(v) for i, v in osd["state"].items()}}


def ckpt_uri(out_uri: str, name: str) -> str:
    return Storage.join(out_uri, "checkpoints", name)


def rng_state() -> Dict[str, Any]:
    st = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def python_rng_state(r: random.Random) -> list:
    v, internal, g = r.getstate()
    return [v, list(internal), g]


def set_python_rng_state(r: random.Random, s: list) -> None:
    r.setstate((s[0], tuple(s[1]), s[2]))


def _gather_ranks(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    fd = bridge.fl_distributed
    if not fd.is_distributed():
        return [obj]
    out: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out  # type: ignore[return-value]


def _save_local(payload: Dict[str, Any], storage: Storage, uri: str) -> None:
    with tempfile.TemporaryDirectory(dir=_tmp_root(storage)) as d:
        p = os.path.join(d, "ckpt.pt")
        torch.save(payload, p)
        storage.put_file(p, uri)


def _tmp_root(storage: Storage) -> str:
    os.makedirs(storage.local_work_dir, exist_ok=True)
    return storage.local_work_dir


def save_training_checkpoint(storage: Storage, out_uri: str, model, optimizer, scheduler, step: int,
                             rank_state: Dict[str, Any], extra: Dict[str, Any], keep_last: int) -> None:
    fd = bridge.fl_distributed
    msd = fd.full_model_state_dict(model)                    # collective under FSDP
    osd = optimizer_payload(model, optimizer, msd)           # collective under FSDP and ZeRO-1
    ranks = _gather_ranks(rank_state)                        # collective
    if not fd.is_main_process():
        return
    msd = strip_prefixes(msd)
    payload = {"format": FORMAT, "model": msd, "optimizer": osd, "scheduler": scheduler.state_dict(),
               "step": step, "world_size": len(ranks), "ranks": ranks, "extra": extra}
    step_uri = ckpt_uri(out_uri, f"step_{step:06d}.pt")
    _save_local(payload, storage, step_uri)
    _copy(storage, step_uri, ckpt_uri(out_uri, "latest.pt"))
    _prune(storage, out_uri, keep_last)


def _copy(storage: Storage, src: str, dst: str) -> None:
    if Storage.is_local(src):
        storage.put_file(Storage.local_path(src), dst)
    else:
        storage.write_bytes(dst, storage.read_bytes(src))


def _prune(storage: Storage, out_uri: str, keep_last: int) -> None:
    """CK-6: keep the newest `keep_last` periodic checkpoints."""
    files = sorted(f for f in storage.glob(Storage.join(out_uri, "checkpoints"), "step_*.pt"))
    for f in files[:-keep_last] if keep_last > 0 else []:
        storage.delete(f)


def save_weights(storage: Storage, uri: str, model, step: int, metric: Optional[float]) -> None:
    """Weights-only checkpoint (best.pt). Collective under FSDP."""
    msd = bridge.fl_distributed.full_model_state_dict(model)
    if bridge.fl_distributed.is_main_process():
        _save_local({"format": FORMAT, "model": strip_prefixes(msd), "step": step, "metric": metric}, storage, uri)


def load_payload(storage: Storage, uri: str) -> Dict[str, Any]:
    return torch.load(storage.cached_local_path(uri), map_location="cpu", weights_only=True)


def restore(storage: Storage, uri: str, model, optimizer, scheduler, is_fsdp: bool) -> Dict[str, Any]:
    """Restores model, optimizer and scheduler. Returns the payload (step, this rank's state, extra).
    Under FSDP the payload is loaded on rank 0 and broadcast (loading is collective there)."""
    fd = bridge.fl_distributed
    if is_fsdp and fd.is_distributed():
        if fd.is_main_process():
            payload = load_payload(storage, uri)
            box = [payload]
        else:
            box = [None]
        dist.broadcast_object_list(box, src=0)
        payload = box[0]
    else:
        payload = load_payload(storage, uri)
    fd.load_full_model_state_dict(model, payload["model"])
    opt = payload["optimizer"]
    if isinstance(opt, dict) and opt.get("format") == NAMED_FORMAT:
        load_named_optimizer_state(model, optimizer, opt, sharded=is_fsdp)
    elif is_fsdp:
        raise RuntimeError("this checkpoint's optimizer state is in the foundation_llm full-state format and cannot be "
                           "loaded into an FSDP run (docs/prd_review.md #20)")
    else:
        fd.load_full_optimizer_state_dict(model, optimizer, opt)
    scheduler.load_state_dict(payload["scheduler"])
    return payload


def rank_entry(payload: Dict[str, Any], rank: int, world: Optional[int] = None) -> Dict[str, Any]:
    """This rank's saved RNG state and objective state. With a different world size than at save
    time there is no per-rank match: rank 0's entry is used and the objective's data position
    (`draws`, per-rank rows drawn) is rescaled so that the same total number of rows counts as
    consumed. Which rows those were cannot be reproduced under another sharding, so the data order
    after such a resume is a fresh, seeded one (a warning is printed by the trainer)."""
    ranks = payload["ranks"]
    old = payload["world_size"]
    if world is None or world == old:
        return ranks[rank] if rank < len(ranks) else ranks[0]
    entry = dict(ranks[0])
    obj = dict(entry.get("objective", {}))
    if "draws" in obj:
        obj["draws"] = int(obj["draws"] * old // world)
    entry["objective"] = obj
    return entry
