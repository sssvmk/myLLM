"""The one training loop every stage uses (TR-1).

A stage supplies an `Objective`; this module owns everything else: model wrapping (parallel.py),
optimizer and LR schedule, autocast / gradient scaling (TR-3, DV-5), gradient accumulation with
`no_sync`, clipping, MoE bias update (PT-9), metrics (MT-L1, MT-L2, PT-10), validation and
best-checkpoint tracking, periodic checkpoints and resume (CK-*), and the final artifacts (CK-1,
CK-2).

Objective contract
------------------
`train_step(trainer, step)` performs every forward/backward for one loop step and calls
`trainer.apply_update()` once (supervised objectives) or several times (`updates_per_step` in
Stage 4). It returns the numeric fields of the step event. `validate` returns numeric fields for
the `val` event; its `best_key` value drives best-checkpoint selection when `best_mode` is set.
Objectives must make identical decisions on every rank (all-reduce anything they branch on).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from ..config.loader import sha256_json
from ..foundation import bridge
from ..io.storage import Storage
from ..metrics.logger import PFMetricsLogger
from ..modeling.loading import configure_moe, moe_modules, set_dropout, unwrap
from . import checkpoint as ck
from . import lineage as lin
from .moe_balance import UsageAccumulator, install_balance_loss
from .parallel import build_optimizer, decay_param_names, wrap_for_training
from .schedules import Scheduler

log = logging.getLogger(__name__)


class Objective:
    event: str = "lm_step"
    total_steps: int = 0                 # loop steps
    total_updates: int = 0               # optimizer steps (drives the LR schedule, LR-2)
    best_mode: Optional[str] = None      # "min" | "max" | None (last checkpoint is the output)
    best_key: str = "val_loss"

    def setup(self, trainer: "Trainer") -> None: ...

    def train_step(self, trainer: "Trainer", step: int) -> Dict[str, float]:
        raise NotImplementedError

    def validate(self, trainer: "Trainer", step: int) -> Dict[str, float]:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, d: Dict[str, Any]) -> None: ...

    def stop_reason(self) -> Optional[str]:
        return None

    def improved(self) -> bool:
        """Hook: an objective can veto a best-checkpoint update (Stage 4 entropy window)."""
        return True

    def final_from(self, trainer: "Trainer") -> Optional[str]:
        """'best' | 'last' | None (default: 'best' when best_mode is set, else last)."""
        return None

    def final_override(self, trainer: "Trainer") -> Optional[Dict[str, Any]]:
        """Rank-0 hook: {"state": weights, "step": n, "selected": label} replaces the default choice
        of final weights (Stage 4 entropy-floor stop)."""
        return None

    def data_manifest(self) -> List[Dict[str, Any]]:
        return []

    def services(self) -> Dict[str, Any]:
        return {}

    def lineage_extra(self) -> Dict[str, Any]:
        return {}


@dataclass
class TrainResult:
    final_step: int
    stop_reason: Optional[str]
    best_step: Optional[int]
    best_metric: Optional[float]
    output_sha256: Optional[str] = None
    final_from: str = "last"


class Trainer:
    def __init__(self, sctx, model: torch.nn.Module, objective: Objective):
        bridge.require()
        self.sctx = sctx
        self.cfg = sctx.cfg
        self.block = sctx.block
        self.storage: Storage = sctx.storage
        self.device = sctx.device
        self.rank, self.world = sctx.rank, sctx.world
        self.objective = objective
        self.autocast_dtype = sctx.autocast_dtype
        self.raw = model
        self.is_moe = bool(moe_modules(model))

        configure_moe(model, self.block.moe.bias_update_rate, self.block.moe.aux_loss_weight)     # PT-9
        install_balance_loss(model)                                                                  # TR-2: aux loss with a gradient
        set_dropout(model, self.block.dropout)                                                     # TR-7
        decay_names = decay_param_names(model)
        self.wrapped = wrap_for_training(model, self.block.distributed.zero_stage, self.device,
                                         self.block.distributed.find_unused_parameters)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        self.is_fsdp = isinstance(self.wrapped, FSDP)
        self.optimizer = build_optimizer(self.wrapped, self.block.distributed.zero_stage, decay_names, self.block.optim)
        self.scheduler = Scheduler(self.optimizer, objective.total_updates, self.block.schedule,
                                   self.block.optim.lr, self.block.optim.lr_min)
        self.scaler = self._make_scaler()
        self.usage = UsageAccumulator(self.wrapped) if (self.is_moe and self.block.moe.update_routing_bias) else None
        self.last_lr = self.scheduler.current_lr
        self.step = 0
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.metrics = PFMetricsLogger(sctx.local_metrics_dir, self.cfg.logging.tensorboard) if self.rank == 0 else None
        self.started = lin.now_iso()

    # ------------------------------------------------------------------ helpers used by objectives
    def _make_scaler(self):
        if self.autocast_dtype != torch.float16:
            return None
        if self.is_fsdp:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            return ShardedGradScaler()
        return torch.amp.GradScaler(self.device.type)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype or torch.float32,
                              enabled=self.autocast_dtype is not None)

    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, track_usage: bool = True):
        """GPTModel.forward(idx, attn_mask) unchanged (TR-2). Returns (logits, aux_loss or None).
        `track_usage=False` for validation / no-grad passes so they do not feed the bias update."""
        with self.autocast():
            out = self.wrapped(idx.to(self.device), attn_mask=None if attn_mask is None else attn_mask.to(self.device))
        if self.usage is not None and track_usage:
            self.usage.add()
        return out

    def accumulate(self, last: bool):
        """DDP/FSDP `no_sync` on all but the last micro-batch of an optimizer step."""
        if not last and self.world > 1 and hasattr(self.wrapped, "no_sync"):
            return self.wrapped.no_sync()
        return contextlib.nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        (self.scaler.scale(loss) if self.scaler is not None else loss).backward()

    def apply_update(self) -> float:
        """Clip, optimizer step, LR step, MoE bias update. Returns the pre-clip gradient norm."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        gc = self.block.optim.grad_clip
        max_norm = gc if gc > 0 else float("inf")
        if self.is_fsdp:
            gn = self.wrapped.clip_grad_norm_(max_norm)
        else:
            gn = torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], max_norm)
        self.last_lr = self.scheduler.current_lr
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        if self.usage is not None:                                # PT-9: only when update_routing_bias is true
            self.usage.apply()
        return float(gn)

    def all_reduce(self, t: torch.Tensor, op=None) -> torch.Tensor:
        t = t.to(self.device)
        if self.world > 1:
            dist.all_reduce(t, op=op if op is not None else dist.ReduceOp.SUM)
        return t

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def log(self, event: str, **fields: Any) -> None:
        if self.metrics is not None:
            self.metrics.log(event, **fields)

    def moe_usage(self) -> Optional[List[List[float]]]:
        mods = moe_modules(self.wrapped)
        if not mods or any(m._last_usage is None for m in mods):
            return None
        return [m._last_usage.tolist() for m in mods]

    # ------------------------------------------------------------------ persistence
    def _rank_state(self) -> Dict[str, Any]:
        return {"rng": ck.rng_state(), "objective": self.objective.state_dict()}

    def _extra(self) -> Dict[str, Any]:
        return {"best_metric": self.best_metric, "best_step": self.best_step, "stage": self.sctx.stage_id,
                "config_hash": self.sctx.config_hash}

    def checkpoint(self, step: int) -> None:
        ck.save_training_checkpoint(self.storage, self.sctx.out_uri, self.wrapped, self.optimizer, self.scheduler,
                                    step, self._rank_state(), self._extra(), self.block.keep_last_checkpoints)
        if self.rank == 0:                                   # lets a resume check the config hash without loading the checkpoint
            self.storage.write_text(ck.ckpt_uri(self.sctx.out_uri, "latest.json"),
                                    json.dumps({"step": step, "config_hash": self.sctx.config_hash}))
        self.sync_metrics()

    def sync_metrics(self) -> None:
        if self.rank == 0 and self.metrics is not None:
            self.storage.put_dir(self.sctx.local_metrics_dir, Storage.join(self.sctx.out_uri, "metrics"))

    def maybe_resume(self) -> int:
        latest = ck.ckpt_uri(self.sctx.out_uri, "latest.pt")
        side = ck.ckpt_uri(self.sctx.out_uri, "latest.json")
        stale = self.storage.exists(side) and json.loads(self.storage.read_text(side)).get("config_hash") != self.sctx.config_hash
        if self.sctx.resume and stale and self.rank == 0:
            print(f"[{self.sctx.stage_id}] checkpoint in {self.sctx.out_uri} was written under a different configuration; "
                  f"starting fresh")
        if not (self.sctx.resume and self.storage.exists(latest)) or stale:
            print(f"[{self.sctx.stage_id}] starting fresh at step 0 on {self.device}") if self.rank == 0 else None
            return 0
        payload = ck.restore(self.storage, latest, self.wrapped, self.optimizer, self.scheduler, self.is_fsdp)
        mine = ck.rank_entry(payload, self.rank, self.world)
        if payload["world_size"] != self.world and self.rank == 0:
            print(f"[{self.sctx.stage_id}] resuming with world size {self.world}; the checkpoint was written with "
                  f"{payload['world_size']}. Model, optimizer and LR schedule are restored exactly; the data position is "
                  f"rescaled and the data order from here on is a fresh seeded one.")
        ck.set_rng_state(mine["rng"])
        self.objective.load_state_dict(mine["objective"])
        self.best_metric = payload["extra"].get("best_metric")
        self.best_step = payload["extra"].get("best_step")
        step = int(payload["step"])
        print(f"[{self.sctx.stage_id}] resumed from {latest} at step {step}") if self.rank == 0 else None
        return step

    # ------------------------------------------------------------------ the loop
    def run(self) -> TrainResult:
        obj, blk = self.objective, self.block
        obj.setup(self)
        self.step = self.maybe_resume()
        total = obj.total_steps
        reason: Optional[str] = None
        try:
            while self.step < total:
                t0 = time.time()
                m = obj.train_step(self, self.step)
                self.step += 1
                m = dict(m)
                m.setdefault("lr", self.last_lr)
                tokens = m.pop("_tokens", None)
                if tokens is not None:
                    m["tokens_per_sec"] = tokens / max(time.time() - t0, 1e-9)
                if self.step % self.cfg.logging.log_every_steps == 0 or self.step == total:
                    self.log(obj.event, step=self.step, **m)
                if self.step % self.cfg.logging.moe_usage_every_steps == 0:
                    u = self.moe_usage()
                    if u is not None:
                        self.log("moe_usage", step=self.step, usage_per_layer=u)
                if self.step % blk.eval_every_steps == 0 or self.step == total:
                    self._validate()
                if self.step % blk.checkpoint_every_steps == 0 or self.step == total:
                    self.checkpoint(self.step)
                reason = obj.stop_reason()
                if reason:
                    print(f"[{self.sctx.stage_id}] stopping early at step {self.step}: {reason}") if self.rank == 0 else None
                    if self.step % blk.checkpoint_every_steps != 0 and self.step != total:
                        self.checkpoint(self.step)
                    break
        finally:
            if self.metrics is not None:
                self.sync_metrics()
        return self._finalize(reason)

    def _validate(self) -> None:
        obj = self.objective
        v = obj.validate(self, self.step)
        self.log("val", step=self.step, **v)
        if obj.best_mode is None:
            return
        cur = float(v[obj.best_key])
        better = (self.best_metric is None or (cur < self.best_metric if obj.best_mode == "min" else cur > self.best_metric))
        if better and obj.improved():
            self.best_metric, self.best_step = cur, self.step
            ck.save_weights(self.storage, ck.ckpt_uri(self.sctx.out_uri, "best.pt"), self.wrapped, self.step, cur)

    def _finalize(self, reason: Optional[str]) -> TrainResult:
        obj = self.objective
        want = obj.final_from(self) or ("best" if obj.best_mode is not None else "last")
        best_uri = ck.ckpt_uri(self.sctx.out_uri, "best.pt")
        use_best = want == "best" and self.best_step is not None and self.storage.exists(best_uri)
        state = bridge.fl_distributed.full_model_state_dict(self.wrapped)      # collective: every rank calls it
        sha = None
        step_for_lineage = self.step
        if bridge.fl_distributed.is_main_process():
            override = obj.final_override(self)
            selected = "best" if use_best else "last"
            if override is not None:
                state, step_for_lineage, selected = override["state"], override["step"], override["selected"]
            elif use_best:
                state = ck.load_payload(self.storage, best_uri)["model"]
                step_for_lineage = self.best_step
            state = ck.strip_prefixes(state)
            from ..data.prepare import read_stage_manifest
            manifest = read_stage_manifest(self.storage, self.cfg, self.sctx.stage_id) or self.sctx.prepared_manifest
            data = lin.data_manifest(manifest, obj.data_manifest())
            tok = self.cfg.tokenizer
            lineage = lin.build_lineage(
                stage=self.sctx.stage_id, parent_uri=self.sctx.parent.uri, parent_sha256=self.sctx.parent.sha256,
                parent_stage=self.sctx.parent.stage, architecture=self.sctx.architecture,
                tokenizer={"ranks_sha256": tok.ranks_sha256, "eot_token_id": tok.eot_token_id,
                           "pad_token_id": tok.pad_token_id,
                           "chat_special_tokens": {k: v.model_dump() for k, v in tok.chat_special_tokens.items()}},
                chat_template=self.cfg.chat_template.model_dump(), data=data, config_hash=self.sctx.config_hash,
                device=str(self.device), started=self.started, final_step=step_for_lineage,
                services=obj.services(),
                extra=dict(obj.lineage_extra(), trained_steps=self.step, stop_reason=reason,
                           selected=selected, best_metric=self.best_metric))
            sha = lin.write_final(self.storage, self.sctx.out_uri, state, lineage, self.sctx.lc.redacted)
            lin.write_complete(self.storage, self.sctx.out_uri, self.sctx.config_hash, sha, step_for_lineage)
            self.sync_metrics()
        self.barrier()
        if self.metrics is not None:
            self.metrics.close()
        return TrainResult(self.step, reason, self.best_step, self.best_metric, sha, selected if bridge.fl_distributed.is_main_process() else ("best" if use_best else "last"))
