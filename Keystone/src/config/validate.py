"""Configuration validation beyond the schema.

`static_errors(cfg)` needs nothing but the parsed config (cheap; run at every startup).
`environment_errors(cfg, ...)` touches storage, the tokenizer file, the base checkpoint, and
endpoints (CF-7; run by `pf validate-config`). Both return lists of "<key>: <problem>" strings.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .schema import PFConfig, STAGE_IDS

PRODUCED = {"on_policy": "preference", "teacher_traces": "distill_offpolicy"}
STAGE_KINDS = {
    "midtrain": {"text", "packed_tokens"},
    "midtrain_long": {"text", "packed_tokens"},
    "sft": {"conversations"},
    "preference": {"preference"},
    "rlvr": {"rl_math", "rl_code"},
    "distill_offpolicy": {"conversations"},
    "distill_onpolicy": {"rl_math", "rl_code", "prompts"},
}
BENCH_KINDS = {"mmlu": "eval_mmlu", "gsm8k": "eval_gsm8k", "math500": "eval_math", "humaneval": "eval_humaneval",
               "ifeval": "eval_ifeval", "alpaca_eval_lc": "eval_alpaca", "safety": "eval_safety", "length": "eval_alpaca"}
GEN_STAGES = ("rlvr", "distill_onpolicy")


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def device_errors(cfg: PFConfig, cuda_available: Optional[bool] = None) -> List[str]:
    """DV-2, DV-5."""
    errs = []
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    dev = cfg.run.device if cfg.run.device != "auto" else ("cuda" if cuda_available else "cpu")
    if cfg.run.device == "cuda" and not cuda_available:
        errs.append("run.device: 'cuda' requested but no usable GPU is available (DV-2)")
    if cfg.run.precision == "fp16" and dev == "cpu":
        errs.append("run.precision: fp16 is not supported on cpu (DV-5)")
    if cfg.run.precision == "bf16" and dev == "cuda" and cuda_available:
        import torch
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            errs.append("run.precision: bf16 not supported by this GPU (DV-5)")
    return errs


def static_errors(cfg: PFConfig, for_pipeline: bool = True, cuda_available: Optional[bool] = None) -> List[str]:
    errs: List[str] = []
    errs += device_errors(cfg, cuda_available)
    world = cfg.launcher.nnodes * cfg.launcher.nproc_per_node
    ds = cfg.datasets
    arch = cfg.base_model.architecture

    if for_pipeline and cfg.launcher.nnodes > 1:
        errs.append("launcher.nnodes: pf run-pipeline supports one node; use per-stage commands for multi-node (PL-8)")

    for sid in STAGE_IDS:
        block = cfg.stage_block(sid)
        key = f"stages.{sid}" if not sid.startswith("distill_") else f"stages.distill.{sid.split('_', 1)[1]}"
        z = block.distributed.zero_stage
        # TR-5
        if sid in GEN_STAGES and z > 1:
            errs.append(f"{key}.distributed.zero_stage: stage {sid} supports ZeRO 0/1 only (TR-5)")
        # TR-5a
        if arch.arch == "deepseek" and z in (0, 1) and not block.distributed.find_unused_parameters:
            errs.append(f"{key}.distributed.find_unused_parameters: must be true for arch deepseek at ZeRO {z} (TR-5a)")
        # RL-7a / RL-3a
        if sid in GEN_STAGES:
            if block.batch.grad_accum != 1:
                errs.append(f"{key}.batch.grad_accum: must be 1 for {sid} (RL-7a)")
            if block.prompts_per_step % world:
                errs.append(f"{key}.prompts_per_step: {block.prompts_per_step} not divisible by world size {world} (RL-3a)")
        # init_from (PL-2)
        init = block.init_from
        if init not in ("base", "previous") and init not in STAGE_IDS and "://" not in init and not init.startswith("/"):
            errs.append(f"{key}.init_from: {init!r} is not base, previous, a stage id, or a URI (PL-2)")
        if init in STAGE_IDS and STAGE_IDS.index(init) >= STAGE_IDS.index(sid):
            errs.append(f"{key}.init_from: {init!r} does not run before {sid} (PL-1)")
        # data sources
        sources = getattr(block.data, "sources", [])
        for i, src in enumerate(sources):
            name = src.dataset
            if name in PRODUCED:
                if PRODUCED[name] != sid:
                    errs.append(f"{key}.data.sources[{i}].dataset: {name!r} is produced by stage {PRODUCED[name]} only (DS-3b)")
                continue
            if name not in ds:
                errs.append(f"{key}.data.sources[{i}].dataset: unknown dataset {name!r}")
            elif ds[name].kind not in STAGE_KINDS[sid]:
                errs.append(f"{key}.data.sources[{i}].dataset: kind {ds[name].kind!r} not usable in {sid} (expects {sorted(STAGE_KINDS[sid])})")
        if sid in ("midtrain", "midtrain_long"):
            r = block.data.retention_eval_dataset
            if r not in ds or ds[r].kind not in STAGE_KINDS[sid]:
                errs.append(f"{key}.data.retention_eval_dataset: {r!r} must be a text/packed_tokens dataset")
        if block.checkpoint_every_steps < 1:
            errs.append(f"{key}.checkpoint_every_steps must be positive")

    # Stage-specific
    p = cfg.stages.preference
    if p.on_policy.enabled:
        n = p.on_policy.prompts_dataset
        if n not in ds or ds[n].kind != "prompts":
            errs.append(f"stages.preference.on_policy.prompts_dataset: {n!r} must be a prompts dataset")
    if not p.on_policy.enabled and any(src.dataset == "on_policy" for src in p.data.sources):
        errs.append("stages.preference.data.sources: lists on_policy but stages.preference.on_policy.enabled is false")
    if cfg.stages.rlvr.moe.update_routing_bias:
        errs.append("stages.rlvr.moe.update_routing_bias: must be false (RL-9)")
    for i, n in enumerate(cfg.stages.distill.offpolicy.prompts_datasets):
        if n not in ds or ds[n].kind not in {"rl_math", "rl_code", "prompts"}:
            errs.append(f"stages.distill.offpolicy.prompts_datasets[{i}]: {n!r} must be rl_math/rl_code/prompts")
    if not any(s.dataset == "teacher_traces" for s in cfg.stages.distill.offpolicy.data.sources) and cfg.stage_enabled("distill_offpolicy"):
        errs.append("stages.distill.offpolicy.data.sources: must include teacher_traces (DI-3)")

    # JG-5
    if cfg.judge.model == cfg.teachers.offpolicy.model:
        errs.append("judge.model: must differ from teachers.offpolicy.model (JG-5)")

    # Decontamination & benchmarks
    for i, n in enumerate(cfg.prepared_data.decontamination.against):
        if n not in ds or not ds[n].kind.startswith("eval_"):
            errs.append(f"prepared_data.decontamination.against[{i}]: {n!r} must be an eval_* dataset")
        elif not set(cfg.prepared_data.decontamination.splits) & set(ds[n].files):
            errs.append(f"prepared_data.decontamination.against[{i}]: dataset {n!r} has none of the splits "
                        f"{cfg.prepared_data.decontamination.splits} (its splits: {sorted(ds[n].files)})")
    if not cfg.prepared_data.decontamination.splits:
        errs.append("prepared_data.decontamination.splits: must not be empty")
    b = cfg.eval.benchmarks
    for bname, kind in BENCH_KINDS.items():
        bench = getattr(b, bname)
        if bench is None:
            continue
        if bench.dataset not in ds or ds[bench.dataset].kind != kind:
            errs.append(f"eval.benchmarks.{bname}.dataset: {bench.dataset!r} must be a {kind} dataset")
    if b.calibration is not None and b.mmlu is None:
        errs.append("eval.benchmarks.calibration: requires eval.benchmarks.mmlu (B-8)")
    if b.gsm8k is not None and b.gsm8k.dataset in ds and "fewshot" not in ds[b.gsm8k.dataset].files and b.gsm8k.n_shot > 0:
        errs.append(f"datasets.{b.gsm8k.dataset}.files: needs a 'fewshot' split for n_shot > 0 (B-2)")

    # SB-2: needed when anything executes code
    code_used = b.humaneval is not None or any(
        ds.get(s.dataset) is not None and ds[s.dataset].kind == "rl_code"
        for sid in ("rlvr", "distill_onpolicy") if cfg.stage_enabled(sid)
        for s in cfg.stage_block(sid).data.sources) or (
        cfg.stage_enabled("distill_offpolicy") and any(ds.get(n) is not None and ds[n].kind == "rl_code"
                                                        for n in cfg.stages.distill.offpolicy.prompts_datasets))
    if code_used and not cfg.code_execution.isolated_host_confirmed:
        errs.append("code_execution.isolated_host_confirmed: must be true when code is executed (SB-2)")

    # Tokenizer static part of TK-3
    t = cfg.tokenizer
    ids = [v.id for v in t.chat_special_tokens.values()]
    if len(set(ids)) != len(ids):
        errs.append("tokenizer.chat_special_tokens: duplicate ids")
    for name, v in t.chat_special_tokens.items():
        if not 0 <= v.id < arch.vocab_rows:
            errs.append(f"tokenizer.chat_special_tokens.{name}.id: {v.id} not below vocab_rows {arch.vocab_rows}")
        if v.id == t.pad_token_id:
            errs.append(f"tokenizer.chat_special_tokens.{name}.id: equals pad_token_id")
    if not 0 <= t.pad_token_id < arch.vocab_rows:
        errs.append("tokenizer.pad_token_id: must be below vocab_rows")

    # Teacher (TC-2)
    if cfg.stage_enabled("distill_onpolicy"):
        ta = cfg.teachers.onpolicy.architecture
        if ta.vocab_rows != arch.vocab_rows:
            errs.append("teachers.onpolicy.architecture.vocab_rows: must equal base_model.architecture.vocab_rows (TC-2)")

    # Stage 1b
    if cfg.stage_enabled("midtrain_long"):
        if cfg.stages.midtrain_long.architecture_override.ctx < arch.ctx:
            errs.append("stages.midtrain_long.architecture_override.ctx: should not be smaller than base ctx")
    return errs


def environment_errors(cfg: PFConfig, storage, check_endpoints: bool = True) -> List[str]:
    """CF-7 checks that touch the outside world."""
    errs: List[str] = []
    from ..foundation import bridge
    try:
        bridge.load(cfg.foundation_llm.code_path)
    except Exception as e:  # noqa: BLE001
        return [f"foundation_llm.code_path: {e}"]

    # tokenizer + TK-3
    try:
        from ..tokenization.tokenizer import ChatTokenizer
        ChatTokenizer.from_config(cfg, storage)
    except Exception as e:  # noqa: BLE001
        errs.append(f"tokenizer: {e}")

    # base checkpoint (MD-3)
    try:
        import torch
        from ..modeling.loading import load_model, load_state
        ckpt = load_state(storage.cached_local_path(cfg.base_model.checkpoint_uri))
        load_model(ckpt, cfg.base_model.architecture, torch.device("cpu"))
    except Exception as e:  # noqa: BLE001
        errs.append(f"base_model.checkpoint_uri: {e}")

    # datasets: globs and field_map in first record
    for name, d in cfg.datasets.items():
        for split, pattern in d.files.items():
            try:
                matches = storage.glob(d.uri, pattern)
            except Exception as e:  # noqa: BLE001
                errs.append(f"datasets.{name}.uri: unreachable ({e})")
                break
            if not matches:
                errs.append(f"datasets.{name}.files.{split}: no files match {pattern!r} under {d.uri}")
                continue
            try:
                rec = first_record(storage, matches[0], d.format)
                for fkey, path in d.field_map.items():
                    if path is None or fkey in ("role_key", "content_key"):
                        continue
                    if get_path(rec, path) is _MISSING:
                        errs.append(f"datasets.{name}.field_map.{fkey}: field {path!r} not in first record of {matches[0]}")
            except Exception as e:  # noqa: BLE001
                errs.append(f"datasets.{name}.files.{split}: cannot read first record ({e})")

    # prompt files
    for key, uri in _prompt_uris(cfg).items():
        try:
            if not storage.exists(uri):
                errs.append(f"{key}: {uri} not found")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{key}: {e}")

    if check_endpoints:
        b = cfg.eval.benchmarks
        need_judge = (b.alpaca_eval_lc is not None or b.safety is not None or
                      (cfg.stage_enabled("preference") and cfg.stages.preference.on_policy.enabled))
        if need_judge:
            e = ping_chat_endpoint(cfg.judge.base_url, cfg.judge.model, cfg.judge.api_key_env, cfg.judge.timeout_s)
            if e:
                errs.append(f"judge.base_url: {e}")
        if cfg.stage_enabled("distill_offpolicy"):
            t = cfg.teachers.offpolicy
            e = ping_chat_endpoint(t.base_url, t.model, t.api_key_env, t.timeout_s)
            if e:
                errs.append(f"teachers.offpolicy.base_url: {e}")
    return errs


def _prompt_uris(cfg: PFConfig) -> Dict[str, str]:
    out = {"judge.prompts.pairwise_uri": cfg.judge.prompts.pairwise_uri,
           "judge.prompts.refusal_uri": cfg.judge.prompts.refusal_uri,
           "judge.prompts.scoring_uri": cfg.judge.prompts.scoring_uri,
           "stages.rlvr.system_prompts.rl_math": cfg.stages.rlvr.system_prompts.rl_math,
           "stages.rlvr.system_prompts.rl_code": cfg.stages.rlvr.system_prompts.rl_code}
    b = cfg.eval.benchmarks
    if b.gsm8k:
        out["eval.benchmarks.gsm8k.base_prompt_uri"] = b.gsm8k.base_prompt_uri
        out["eval.benchmarks.gsm8k.chat_prompt_uri"] = b.gsm8k.chat_prompt_uri
    if b.math500:
        out["eval.benchmarks.math500.chat_prompt_uri"] = b.math500.chat_prompt_uri
    if b.humaneval:
        out["eval.benchmarks.humaneval.chat_prompt_uri"] = b.humaneval.chat_prompt_uri
    return out


_MISSING = object()


def get_path(rec, dotted: str):
    cur = rec
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def first_record(storage, uri: str, fmt: str):
    if fmt in ("parquet", "packed_tokens"):
        import pyarrow.parquet as pq
        with storage.open(uri, "rb") as f:
            pf = pq.ParquetFile(f)
            batch = next(pf.iter_batches(batch_size=1))
            return batch.to_pylist()[0]
    with storage.open(uri, "rb") as f:
        if fmt == "jsonl":
            for line in f:
                if line.strip():
                    return json.loads(line)
            raise ValueError("empty jsonl")
        data = json.loads(f.read())
        return data[0] if isinstance(data, list) else data


def ping_chat_endpoint(base_url: str, model: str, api_key_env: str, timeout_s: float) -> Optional[str]:
    """One-token Chat Completions request (CF-7). Returns an error string or None."""
    import urllib.request
    key = os.environ.get(api_key_env)
    if key is None:
        return f"environment variable {api_key_env} is not set"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            json.loads(r.read())
        return None
    except Exception as e:  # noqa: BLE001
        return f"endpoint did not answer a one-token request: {e}"
