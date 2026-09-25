"""Pieces shared by the stage drivers: the per-stage context, model loading for a stage, and the
packed-block objective used by Stages 1, 1b, 2 and 5a (LM and SFT differ only in the loss mask and
the event they log)."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..config.loader import LoadedConfig
from ..config.validate import device_errors, resolve_device
from ..data import loaders, mixture, prepare
from ..foundation import bridge
from ..io.storage import Storage
from ..modeling.loading import is_base_checkpoint, load_model, load_state, validate_base_args
from ..tokenization.chat_template import ChatTemplate
from ..tokenization.tokenizer import ChatTokenizer
from ..training import checkpoint as ck
from ..training.loop import Objective, Trainer, TrainResult
from ..training.objectives.sft import sft_loss_sum
from . import resolve

AUTOCAST = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


@dataclass
class StageContext:
    lc: LoadedConfig
    storage: Storage
    stage_id: str
    tok: ChatTokenizer
    template: ChatTemplate
    device: torch.device
    rank: int
    world: int
    local_rank: int
    parent: resolve.ParentRef
    architecture: Dict[str, Any]
    config_hash: str
    resume: bool
    prepared_manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def cfg(self):
        return self.lc.cfg

    @property
    def block(self):
        return self.cfg.stage_block(self.stage_id)

    @property
    def out_uri(self) -> str:
        return self.block.output_uri

    @property
    def autocast_dtype(self):
        return AUTOCAST[self.cfg.run.precision]

    @property
    def local_metrics_dir(self) -> str:
        return os.path.join(self.cfg.run.local_work_dir, self.cfg.run.name, self.stage_id, "metrics")

    @property
    def ctx_len(self) -> int:
        return int(self.architecture["ctx"])


def setup_process() -> tuple:
    """(rank, world_size, local_rank); a no-op returning (0, 1, 0) outside torchrun."""
    bridge.require()
    return bridge.fl_distributed.setup_distributed()


def make_stage_context(lc: LoadedConfig, stage_id: str, resume: bool, storage: Optional[Storage] = None,
                       dist_info: Optional[tuple] = None) -> StageContext:
    cfg = lc.cfg
    storage = storage or Storage.from_config(cfg)
    errs = device_errors(cfg)
    if errs:
        raise SystemExit("\n".join(errs))
    rank, world, local = dist_info if dist_info is not None else setup_process()
    dev = resolve_device(cfg.run.device)
    device = torch.device(f"cuda:{local}" if dev == "cuda" else "cpu")
    parent = resolve.resolve_parent(storage, cfg, stage_id)
    arch = resolve.stage_architecture(cfg, stage_id, parent)
    tok = ChatTokenizer.from_config(cfg, storage)
    manifest = prepare.read_stage_manifest(storage, cfg, stage_id)
    return StageContext(lc, storage, stage_id, tok, ChatTemplate.from_config(cfg, tok), device, rank, world, local,
                        parent, arch, resolve.stage_config_hash(cfg, stage_id, arch, parent.sha256), resume, manifest)


def load_stage_model(sctx: StageContext, architecture: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    """MD-1..MD-3. Base checkpoints are validated against the architecture they were trained with
    (the parent's), then the model is built with the stage's architecture (Stage 1b changes ctx and
    rope_theta only, so the weights load unchanged)."""
    ckpt = load_state(sctx.storage.cached_local_path(sctx.parent.uri))
    if is_base_checkpoint(ckpt):
        validate_base_args(ckpt["args"], sctx.parent.architecture)
    torch.manual_seed(sctx.cfg.run.seed)                       # TR-6
    return load_model(ckpt, architecture or sctx.architecture, sctx.device, validate_base=False)


def make_prep_context(sctx: StageContext, force: bool = False, log=print) -> prepare.PrepContext:
    """PL-9: preparation runs single-process on the resolved device."""
    from ..data.registry import DatasetRegistry
    return prepare.PrepContext(sctx.cfg, sctx.storage, sctx.tok, sctx.template,
                               DatasetRegistry.from_config(sctx.cfg, sctx.storage), force=force, log=log)


def check_block_len(sctx: StageContext, dataset: str, splits=("train", "test")) -> None:
    """Prepared LM/SFT blocks must match the stage's ctx (a Stage 1b ctx change needs re-preparation)."""
    for split in splits:
        m = prepare.read_marker(sctx.storage, prepare.prepared_uri(sctx.cfg, sctx.stage_id, dataset, split))
        if m is not None and m["entry"].get("block_len") not in (None, sctx.ctx_len + 1):
            raise ValueError(f"{dataset}/{split} was prepared with block length {m['entry']['block_len']} but stage "
                             f"{sctx.stage_id} has ctx {sctx.ctx_len}; re-run `pf prepare-data --stage {sctx.stage_id}`")


def run_training(sctx: StageContext, objective: Objective, model: Optional[torch.nn.Module] = None) -> TrainResult:
    model = model if model is not None else load_stage_model(sctx)
    return Trainer(sctx, model, objective).run()


# --------------------------------------------------------------------------- packed-block objective
class BlockObjective(Objective):
    """LM (Stage 1/1b, S1-1) and SFT (Stage 2/5a, S2-1). Both normalise the summed cross-entropy
    over masked targets by the global target count of the optimizer step (micro-batches and ranks)
    and multiply by the world size to undo DDP/FSDP gradient averaging (docs/prd_review.md #16)."""

    def __init__(self, sctx: StageContext, kind: str):
        assert kind in ("lm", "sft")
        self.s, self.kind = sctx, kind
        self.event = "lm_step" if kind == "lm" else "sft_step"
        cfg, blk = sctx.cfg, sctx.block
        self.micro, self.accum = blk.batch.micro_batch_size, blk.batch.grad_accum
        self.pad = cfg.tokenizer.pad_token_id
        self.sources = list(blk.data.sources)
        self.names = [s.dataset for s in self.sources]
        self.weights = mixture.normalise([s.weight for s in self.sources])
        self.rows = [prepare.prepared_rows(sctx.storage, cfg, sctx.stage_id, n, "train") for n in self.names]
        for n in self.names:
            check_block_len(sctx, n, ("train",))
        world = sctx.world
        if kind == "lm":
            self.counts = None
            self.quotas = None
            self.total_steps = mixture.steps_for_budget(blk.data.token_budget, self.micro, self.accum, world, sctx.ctx_len)
        else:
            self.counts = mixture.epoch_counts(self.rows, self.weights)                       # DP-9
            self.quotas = [max(1, round(c / world)) for c in self.counts]
            self.total_steps = mixture.steps_for_epochs(blk.data.epochs, self.counts, self.micro, self.accum, world)
        self.total_updates = self.total_steps
        self.best_mode = "min" if kind == "sft" else None                                  # S2-3; Stage 1 keeps the last step
        self.mix: Optional[mixture.MixtureStream] = None
        self.val_names = [n for n in self.names if prepare.read_marker(
            sctx.storage, prepare.prepared_uri(cfg, sctx.stage_id, n, "test")) is not None]
        if not self.val_names and kind == "sft":
            raise ValueError(f"stage {sctx.stage_id}: none of {self.names} has a prepared test split for validation (S2-3)")
        self.val_src = list(self.val_names)
        self.retention = None
        if kind == "lm":
            self.retention = blk.data.retention_eval_dataset
            if self.retention not in self.val_names:
                self.val_names.append(self.retention)


    # ------------------------------------------------------------------ data
    def _uri(self, name: str, split: str) -> str:
        return prepare.prepared_uri(self.s.cfg, self.s.stage_id, name, split)

    def setup(self, trainer: Trainer) -> None:
        cfg, s = self.s.cfg, self.s
        cols = ["tokens", "doc_start"] + (["loss_mask"] if self.kind == "sft" else [])
        streams = [loaders.RowStream(s.storage, self._uri(n, "train"), cols, s.rank, s.world, cfg.run.seed + i, cycle=True)
                   for i, n in enumerate(self.names)]
        self.mix = mixture.MixtureStream(streams, self.weights, seed=cfg.run.seed + s.rank, quotas=self.quotas)   # TR-6

    def state_dict(self) -> Dict[str, Any]:
        return {"draws": self.mix.draws}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mix.fast_forward(d["draws"])

    def _batch(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return loaders.collate_blocks(rows, self.pad, lm=self.kind == "lm")

    def _mask(self, b: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.s.block.use_doc_mask:
            return None
        return bridge.fl_model.build_doc_attention_mask(b["doc_start"][:, :-1])                 # DL-2

    # ------------------------------------------------------------------ training
    def train_step(self, trainer: Trainer, step: int) -> Dict[str, float]:
        batches = [self._batch([self.mix.next()[1] for _ in range(self.micro)]) for _ in range(self.accum)]
        n_local = float(sum(int(b["loss_mask"][:, 1:].sum()) for b in batches))
        n_global = max(trainer.all_reduce(torch.tensor(n_local, dtype=torch.float64)).item(), 1.0)
        ce_total = torch.zeros((), dtype=torch.float64)
        for i, b in enumerate(batches):
            x, y = b["tokens"][:, :-1], b["tokens"][:, 1:].to(trainer.device)
            m = b["loss_mask"][:, 1:].to(trainer.device)
            with trainer.accumulate(i == self.accum - 1):
                logits, aux = trainer.forward(x, self._mask(b))
                ce_sum, _ = sft_loss_sum(logits, y, m)
                loss = ce_sum / n_global * trainer.world
                if aux is not None:
                    loss = loss + aux / self.accum                                               # TR-2
                trainer.backward(loss)
            ce_total += ce_sum.detach().double().cpu()
        gn = trainer.apply_update()
        ce_global = trainer.all_reduce(ce_total.to(trainer.device)).item()
        out = {"loss": ce_global / n_global, "grad_norm": gn,
               "_tokens": self.micro * self.accum * trainer.world * self.s.ctx_len}
        if self.kind == "sft":
            out["assistant_tokens"] = int(n_global)
        return out

    # ------------------------------------------------------------------ validation (S1-4, S2-3)
    @torch.no_grad()
    def _eval_dataset(self, trainer: Trainer, name: str):
        s = self.s
        cols = ["tokens", "doc_start"] + (["loss_mask"] if self.kind == "sft" else [])
        stream = loaders.RowStream(s.storage, self._uri(name, "test"), cols, s.rank, s.world, s.cfg.run.seed,
                                   cycle=False, shuffle=False)
        rows = list(stream)
        batches = [self._batch(rows[i:i + self.micro]) for i in range(0, len(rows), self.micro)]
        n_max = int(trainer.all_reduce(torch.tensor(len(batches)), dist_max()).item())
        dummy = self._batch([{"tokens": [self.pad] * (s.ctx_len + 1), "doc_start": [False] * (s.ctx_len + 1),
                              "loss_mask": [False] * (s.ctx_len + 1)}])
        total, count = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
        for k in range(n_max):
            b = batches[k] if k < len(batches) else dummy           # equal forward counts per rank (FSDP all-gathers)
            logits, _ = trainer.forward(b["tokens"][:, :-1], self._mask(b), track_usage=False)
            ce, n = sft_loss_sum(logits, b["tokens"][:, 1:].to(trainer.device), b["loss_mask"][:, 1:].to(trainer.device))
            total += ce.double().cpu()
            count += n.double().cpu()
        pair = trainer.all_reduce(torch.stack([total, count]).to(trainer.device))
        return float(pair[0]), float(pair[1])

    def validate(self, trainer: Trainer, step: int) -> Dict[str, float]:
        trainer.wrapped.eval()
        try:
            per = {n: self._eval_dataset(trainer, n) for n in self.val_names}
        finally:
            trainer.wrapped.train()
        src_loss = sum(per[n][0] for n in self.val_names if n in self.names)
        src_cnt = max(sum(per[n][1] for n in self.val_names if n in self.names), 1.0)
        out = {"val_loss": src_loss / src_cnt}
        for n, (t, c) in per.items():
            out[f"val_loss__{n}"] = t / max(c, 1.0)
        if self.retention is not None:
            out["retention_loss"] = per[self.retention][0] / max(per[self.retention][1], 1.0)      # S1-4
        return out

    def data_manifest(self) -> List[Dict[str, Any]]:
        counts = self.counts or [None] * len(self.names)
        return [{"dataset": n, "weight": w, "rows_available": r, "rows_per_epoch": c}
                for n, w, r, c in zip(self.names, self.weights, self.rows, counts)]


def dist_max():
    import torch.distributed as dist
    return dist.ReduceOp.MAX
