"""Stage orchestration: per-stage prepare / train / eval / gate, and `pf run-pipeline` (§6, §15).

Artifacts under each stage's `output_uri` (CK-1 plus the status files in resolve.py):
    final/eval_report.json   carries `eval_config_hash`; evaluation reruns when it changes (PL-5)
    final/gate_result.json   recomputed whenever `pf gate` runs (cheap)
    _COMPLETE / _SKIPPED     carry the stage's configuration hash
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config.loader import LoadedConfig
from ..config.schema import STAGE_IDS
from ..config.validate import static_errors
from ..io.storage import Storage
from ..training.lineage import read_complete, read_lineage, sha256_uri
from . import distill, midtrain, preference, resolve, rlvr, sft
from .common import make_prep_context, make_stage_context

PREPARE = {"midtrain": midtrain.prepare_stage, "midtrain_long": midtrain.prepare_stage, "sft": sft.prepare_stage,
           "preference": preference.prepare_stage, "rlvr": rlvr.prepare_stage,
           "distill_offpolicy": distill.prepare_offpolicy, "distill_onpolicy": distill.prepare_onpolicy}
TRAIN = {"midtrain": midtrain.train, "midtrain_long": midtrain.train, "sft": sft.train, "preference": preference.train,
         "rlvr": rlvr.train, "distill_offpolicy": distill.train_offpolicy, "distill_onpolicy": distill.train_onpolicy}

EXIT_ENTRY_GATE_STOP = 4


class PipelineError(RuntimeError):
    pass


def base_dir_uri(cfg) -> str:
    return Storage.join(cfg.run.output_root_uri, cfg.run.name, "base")


# --------------------------------------------------------------------------- prepare / train
def prepare_stage_data(lc: LoadedConfig, storage: Storage, stage_id: str, force: bool = False) -> None:
    """PL-9: single process on the resolved device."""
    sctx = make_stage_context(lc, stage_id, False, storage, (0, 1, 0))
    PREPARE[stage_id](sctx, make_prep_context(sctx, force=force))


def invalidate_if_stale(storage: Storage, cfg, stage_id: str, config_hash: str) -> bool:
    """Removes outputs of a completed run made under a different configuration (PL-5), so they can
    never be picked up as a parent or final model. Returns True if something was removed."""
    out = resolve.out_uri(cfg, stage_id)
    c = read_complete(storage, out)
    skipped = resolve.is_skipped(storage, cfg, stage_id)
    old = (c or skipped or {}).get("config_hash")
    if (c is None and skipped is None) or old == config_hash:
        return False
    for name in ("_COMPLETE", "_SKIPPED", "final"):
        storage.delete(Storage.join(out, name))
    return True


def train_stage(lc: LoadedConfig, storage: Storage, stage_id: str, resume: bool, dist_info=None):
    """Runs one stage's training in this process (every rank under torchrun calls this)."""
    sctx = make_stage_context(lc, stage_id, resume, storage, dist_info)
    if sctx.rank == 0:
        invalidate_if_stale(storage, lc.cfg, stage_id, sctx.config_hash)
    res = TRAIN[stage_id](sctx)
    if res.stop_reason == "entry_gate_skip" and sctx.rank == 0:            # RL-1: record the hash the decision was made under
        u = Storage.join(sctx.out_uri, "_SKIPPED")
        rec = json.loads(storage.read_text(u))
        rec["config_hash"] = sctx.config_hash
        storage.write_text(u, json.dumps(rec, sort_keys=True))
    return res


def launch_training(lc: LoadedConfig, stage_id: str, resume: bool = True) -> int:
    """PL-7: `torchrun ... -m src.cli train` when launcher.nproc_per_node > 1."""
    cfg = lc.cfg
    n = cfg.launcher.nproc_per_node
    tail = ["-m", "src.cli", "train", "--stage", stage_id, "-c", lc.path]
    for ov in lc.overrides:
        tail += ["--set", ov]
    if resume:
        tail.append("--resume")
    exe = shutil.which("torchrun")
    head = [exe] if exe else [sys.executable, "-m", "torch.distributed.run"]
    cmd = head + [f"--nproc_per_node={n}"] + list(cfg.launcher.extra_torchrun_args) + tail
    env = dict(os.environ)
    import src
    root = os.path.dirname(os.path.dirname(os.path.abspath(src.__file__)))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call(cmd, env=env)


# --------------------------------------------------------------------------- eval / gate
def _device_dtype(cfg):
    import torch
    from ..config.validate import resolve_device
    from .common import AUTOCAST
    return torch.device(resolve_device(cfg.run.device)), AUTOCAST[cfg.run.precision]


def read_report(storage: Storage, out_dir_uri: str) -> Optional[Dict[str, Any]]:
    u = Storage.join(out_dir_uri, "eval_report.json")
    return json.loads(storage.read_text(u)) if storage.exists(u) else None


def eval_baseline(lc: LoadedConfig, storage: Storage, judge=None) -> Dict[str, Any]:
    from ..eval import suite
    cfg = lc.cfg
    uri = cfg.base_model.checkpoint_uri
    sha = sha256_uri(storage, uri)
    cur = read_report(storage, base_dir_uri(cfg))
    if cur is not None and cur["eval_config_hash"] == suite.eval_config_hash(cfg, sha):
        return cur
    dev, dt = _device_dtype(cfg)
    return suite.evaluate_checkpoint(cfg, storage, uri, base_dir_uri(cfg), dev, dt, sha, judge=judge)


def eval_stage(lc: LoadedConfig, storage: Storage, stage_id: str, judge=None) -> Dict[str, Any]:
    from ..eval import suite
    from ..metrics.logger import PFMetricsLogger
    cfg = lc.cfg
    out = resolve.out_uri(cfg, stage_id)
    comp = read_complete(storage, out)
    if comp is None:
        raise PipelineError(f"stage {stage_id} has no completed output to evaluate")
    sha = comp["output_sha256"]
    final = Storage.join(out, "final")
    cur = read_report(storage, final)
    if cur is not None and cur["eval_config_hash"] == suite.eval_config_hash(cfg, sha):
        return cur
    dev, dt = _device_dtype(cfg)
    mdir = os.path.join(cfg.run.local_work_dir, cfg.run.name, stage_id, "metrics")
    lg = PFMetricsLogger(mdir, cfg.logging.tensorboard)
    try:
        rep = suite.evaluate_checkpoint(cfg, storage, resolve.final_model_uri(cfg, stage_id), final, dev, dt, sha,
                                        metrics_logger=lg, judge=judge)
    finally:
        lg.close()
    storage.put_dir(mdir, Storage.join(out, "metrics"))
    return rep


def gate_stage(lc: LoadedConfig, storage: Storage, stage_id: str) -> Dict[str, Any]:
    """GT-1..GT-3: compare the stage's report with its parent's (the stage `init_from` resolves to)."""
    from ..eval.gates import evaluate_gate
    cfg = lc.cfg
    out = resolve.out_uri(cfg, stage_id)
    cur = read_report(storage, Storage.join(out, "final"))
    if cur is None:
        raise PipelineError(f"stage {stage_id} has no eval_report.json; run `pf eval` first")
    parent = resolve.resolve_parent(storage, cfg, stage_id)
    if parent.kind == "base":
        par = read_report(storage, base_dir_uri(cfg))
    elif parent.kind == "stage":
        par = read_report(storage, Storage.join(resolve.out_uri(cfg, parent.stage), "final"))
    else:
        par = read_report(storage, parent.uri.rsplit("/", 1)[0])
    if par is None:
        raise PipelineError(f"parent of {stage_id} has no eval_report.json to compare with")
    res = evaluate_gate(stage_id, cur, par, cfg.gates)
    res["parent"] = {"kind": parent.kind, "stage": parent.stage, "uri": parent.uri}
    storage.write_text(Storage.join(out, "final", "gate_result.json"), json.dumps(res, indent=2, sort_keys=True, default=str))
    return res


# --------------------------------------------------------------------------- export (PL-6, EX-*)
def export_checkpoint(lc: LoadedConfig, storage: Storage, checkpoint_uri: str) -> str:
    import torch
    from ..export.export import export_model
    from ..modeling.loading import load_model, load_state
    cfg = lc.cfg
    ckpt = load_state(storage.cached_local_path(checkpoint_uri))
    lineage = ckpt.get("lineage")
    arch = (lineage or {}).get("architecture") or cfg.base_model.architecture.model_dump()
    model = load_model(ckpt, arch, torch.device("cpu"))
    t = cfg.tokenizer
    tok_rec = {"ranks_sha256": t.ranks_sha256, "eot_token_id": t.eot_token_id, "pad_token_id": t.pad_token_id,
               "chat_special_tokens": {k: v.model_dump() for k, v in t.chat_special_tokens.items()}}
    report = read_report(storage, checkpoint_uri.rsplit("/", 1)[0])
    export_model(model, dict(arch), storage, cfg.export.output_uri, cfg.export.dtype, t.ranks_file_uri, tok_rec,
                 cfg.chat_template.model_dump(), lineage=lineage, eval_report=report)
    return cfg.export.output_uri


# --------------------------------------------------------------------------- run-pipeline
@dataclass
class StageOutcome:
    stage: str
    status: str                       # trained | reused | skipped | disabled | failed_training | stopped
    gate_passed: Optional[bool] = None
    note: str = ""


@dataclass
class PipelineResult:
    ok: bool
    outcomes: List[StageOutcome] = field(default_factory=list)
    final_stage: Optional[str] = None
    export_uri: Optional[str] = None
    message: str = ""


def run_pipeline(lc: LoadedConfig, storage: Optional[Storage] = None, judge=None, log=print) -> PipelineResult:
    cfg = lc.cfg
    storage = storage or Storage.from_config(cfg)
    errs = static_errors(cfg, for_pipeline=True)                       # includes PL-8 (nnodes > 1)
    if errs:
        raise PipelineError("configuration errors:\n  - " + "\n  - ".join(errs))
    res = PipelineResult(ok=True)
    log(f"[pipeline] baseline evaluation of {cfg.base_model.checkpoint_uri}")
    eval_baseline(lc, storage, judge)
    for sid in STAGE_IDS:
        if not cfg.stage_enabled(sid):
            res.outcomes.append(StageOutcome(sid, "disabled"))
            continue
        parent = resolve.resolve_parent(storage, cfg, sid)
        arch = resolve.stage_architecture(cfg, sid, parent)
        h = resolve.stage_config_hash(cfg, sid, arch, parent.sha256)
        skipped = resolve.is_skipped(storage, cfg, sid)
        if resolve.stage_up_to_date(storage, cfg, sid, h) or (skipped is not None and skipped.get("config_hash") == h):
            status = "skipped" if skipped is not None and skipped.get("config_hash") == h else "reused"
            log(f"[pipeline] {sid}: {status} (unchanged inputs, PL-5)")
        else:
            invalidate_if_stale(storage, cfg, sid, h)
            log(f"[pipeline] {sid}: preparing data")
            prepare_stage_data(lc, storage, sid)
            log(f"[pipeline] {sid}: training")
            if cfg.launcher.nproc_per_node > 1:
                rc = launch_training(lc, sid, resume=True)
            else:
                try:
                    train_stage(lc, storage, sid, True, (0, 1, 0))
                    rc = 0
                except rlvr.EntryGateFailure as e:
                    log(f"[pipeline] {sid}: {e}")
                    rc = EXIT_ENTRY_GATE_STOP
            if rc == EXIT_ENTRY_GATE_STOP:
                res.outcomes.append(StageOutcome(sid, "stopped", note="RL-1 entry gate (on_failure: stop)"))
                res.ok, res.message = False, f"{sid}: entry gate failed with on_failure=stop"
                return res
            if rc != 0:
                res.outcomes.append(StageOutcome(sid, "failed_training", note=f"exit code {rc}"))
                res.ok, res.message = False, f"{sid}: training exited with code {rc}; stage left incomplete (PL-7)"
                return res
            status = "skipped" if resolve.is_skipped(storage, cfg, sid) is not None else "trained"
        if status == "skipped":
            res.outcomes.append(StageOutcome(sid, "skipped", note="entry gate"))
            continue
        eval_stage(lc, storage, sid, judge)
        g = gate_stage(lc, storage, sid)
        res.outcomes.append(StageOutcome(sid, status, gate_passed=g["passed"]))
        log(f"[pipeline] {sid}: gate {'passed' if g['passed'] else 'FAILED'}")
        if not g["passed"] and cfg.gates.on_failure == "stop":
            res.ok, res.message = False, f"{sid}: gate failed (gates.on_failure: stop)"
            return res
    fs = resolve.final_stage(storage, cfg)
    res.final_stage = fs
    if fs is None:
        res.ok, res.message = False, "no stage completed and passed its gate; nothing to export (PL-6)"
        return res
    res.export_uri = export_checkpoint(lc, storage, resolve.final_model_uri(cfg, fs))
    log(f"[pipeline] final model = {fs}; exported to {res.export_uri}")
    return res
