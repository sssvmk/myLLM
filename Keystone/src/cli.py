"""`pf` entry point (§15). DV-3: the device is resolved and CUDA_VISIBLE_DEVICES set BEFORE torch
is imported in this process, so nothing at module level here imports torch."""
from __future__ import annotations

import argparse
import json
import os
import sys



def _peek_device(config_path: str, overrides) -> str:
    """Reads run.device without importing torch (OmegaConf only)."""
    from omegaconf import OmegaConf
    c = OmegaConf.load(config_path)
    for ov in overrides:
        if ov.startswith("run.device="):
            return ov.split("=", 1)[1]
    return str(OmegaConf.select(c, "run.device"))


def _setup_device_env(config_path: str, overrides) -> None:
    dev = _peek_device(config_path, overrides)
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _load(args):
    from .config.loader import ConfigError, load_config
    from .foundation import bridge
    try:
        lc = load_config(args.config, args.set)
    except ConfigError as e:
        print(f"configuration error:\n{e}", file=sys.stderr)
        sys.exit(2)
    bridge.load(lc.cfg.foundation_llm.code_path)
    return lc


def cmd_validate(args) -> int:
    from .config.validate import environment_errors, static_errors
    from .io.storage import Storage
    lc = _load(args)
    errs = static_errors(lc.cfg, for_pipeline=True)
    errs += environment_errors(lc.cfg, Storage.from_config(lc.cfg), check_endpoints=not args.skip_endpoints)
    if errs:
        print(f"{len(errs)} problem(s):", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("configuration OK")
    return 0


def _device_and_dtype(cfg):
    import torch
    from .config.validate import device_errors, resolve_device
    errs = device_errors(cfg)
    if errs:
        raise SystemExit("\n".join(errs))
    dev = torch.device(resolve_device(cfg.run.device))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg.run.precision]
    return dev, dtype


def cmd_generate(args) -> int:
    from .generation.api import GenerationConfig, generate
    from .io.storage import Storage
    from .modeling.inference import InferenceModel
    from .modeling.loading import load_model, load_state
    from .tokenization.chat_template import ChatTemplate
    from .tokenization.tokenizer import ChatTokenizer
    lc = _load(args)
    cfg = lc.cfg
    st = Storage.from_config(cfg)
    dev, dtype = _device_and_dtype(cfg)
    tok = ChatTokenizer.from_config(cfg, st)
    tmpl = ChatTemplate.from_config(cfg, tok)
    ckpt = load_state(st.cached_local_path(args.checkpoint))
    architecture = ckpt.get("lineage", {}).get("architecture") or cfg.base_model.architecture
    model = InferenceModel(load_model(ckpt, architecture, dev))
    if args.prompt is not None:
        items = [{"id": "0", "prompt": args.prompt}]
    else:
        items = [json.loads(l) for l in st.read_text(args.prompts_file).splitlines() if l.strip()]
    def enc(p):
        return tmpl.render_prompt([{"role": "user", "content": p}]) if args.chat else tok.encode_ordinary(p)
    stops = [cfg.tokenizer.eot_token_id] + ([tok.im_end_id] if args.chat else [])
    gcfg = GenerationConfig.from_block(cfg.generation, num_samples=1, stop_token_ids=stops)
    results = generate(model, tok, [enc(i["prompt"]) for i in items], gcfg, autocast_dtype=dtype)
    lines = [json.dumps({"id": i["id"], **r.__dict__}) for i, r in zip(items, results)]
    if args.out:
        st.write_text(args.out, "\n".join(lines) + "\n")
    else:
        print("\n".join(lines))
    return 0


def cmd_export(args) -> int:
    """`pf export`: exports --checkpoint, or by default the final model of the pipeline (PL-6)."""
    from .io.storage import Storage
    from .stages import pipeline, resolve
    lc = _load(args)
    st = Storage.from_config(lc.cfg)
    uri = args.checkpoint
    if uri is None:
        fs = resolve.final_stage(st, lc.cfg)
        if fs is None:
            print("no stage completed and passed its gate; nothing to export (PL-6)", file=sys.stderr)
            return 1
        uri = resolve.final_model_uri(lc.cfg, fs)
        print(f"final model is stage {fs}")
    print(f"exported to {pipeline.export_checkpoint(lc, st, uri)}")
    return 0


def _stage_ids(cfg, arg: str):
    from .config.schema import STAGE_IDS
    if arg == "all":
        return [s for s in STAGE_IDS if cfg.stage_enabled(s)]
    if arg not in STAGE_IDS:
        raise SystemExit(f"unknown stage {arg!r}; expected one of {STAGE_IDS} or 'all'")
    return [arg]


def cmd_prepare(args) -> int:
    from .io.storage import Storage
    from .stages import pipeline
    lc = _load(args)
    st = Storage.from_config(lc.cfg)
    for sid in _stage_ids(lc.cfg, args.stage):
        print(f"preparing {sid}")
        pipeline.prepare_stage_data(lc, st, sid, force=args.force)
    return 0


def cmd_train(args) -> int:
    from .io.storage import Storage
    from .stages import pipeline, rlvr
    lc = _load(args)
    if args.stage == "all":
        raise SystemExit("pf train takes one stage; use pf run-pipeline for the whole pipeline")
    (sid,) = _stage_ids(lc.cfg, args.stage)
    try:
        res = pipeline.train_stage(lc, Storage.from_config(lc.cfg), sid, args.resume)
    except rlvr.EntryGateFailure as e:
        print(f"entry gate failed: {e}", file=sys.stderr)
        return pipeline.EXIT_ENTRY_GATE_STOP
    print(f"{sid}: finished at step {res.final_step}" + (f" ({res.stop_reason})" if res.stop_reason else ""))
    return 0


def cmd_eval(args) -> int:
    from .eval import suite
    from .io.storage import Storage
    lc = _load(args)
    cfg = lc.cfg
    st = Storage.from_config(cfg)
    dev, dtype = _device_and_dtype(cfg)
    rep = suite.evaluate_checkpoint(cfg, st, args.checkpoint, args.out, dev, dtype)
    print(json.dumps({"mode": rep["mode"], "stage": rep["stage"], "metrics": rep["metrics"], "skipped": rep["skipped"]}, indent=2))
    return 0


def cmd_gate(args) -> int:
    from .io.storage import Storage
    from .stages import pipeline
    lc = _load(args)
    (sid,) = _stage_ids(lc.cfg, args.stage)
    res = pipeline.gate_stage(lc, Storage.from_config(lc.cfg), sid)
    for c in res["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + json.dumps({k: v for k, v in c.items() if k != "passed"}))
    print("gate " + ("passed" if res["passed"] else "FAILED"))
    return 0 if res["passed"] else 1


def cmd_run_pipeline(args) -> int:
    from .stages import pipeline
    lc = _load(args)
    try:
        res = pipeline.run_pipeline(lc)
    except pipeline.PipelineError as e:
        print(str(e), file=sys.stderr)
        return 2
    for o in res.outcomes:
        print(f"  {o.stage:20s} {o.status:16s} gate={o.gate_passed} {o.note}")
    if not res.ok:
        print(res.message, file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pf")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("-c", "--config", required=True)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        return sp
    v = common(sub.add_parser("validate-config"))
    v.add_argument("--skip-endpoints", action="store_true", help="skip judge/teacher reachability checks")
    g = common(sub.add_parser("generate"))
    g.add_argument("--checkpoint", required=True)
    grp = g.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompt")
    grp.add_argument("--prompts-file")
    g.add_argument("--out")
    g.add_argument("--chat", action="store_true")
    e = common(sub.add_parser("export"))
    e.add_argument("--checkpoint", help="stage final/model.pt to export; default: the pipeline's final model (PL-6)")
    pd = common(sub.add_parser("prepare-data"))
    pd.add_argument("--stage", required=True, help="stage id or 'all'")
    pd.add_argument("--force", action="store_true", help="ignore cached prepared data")
    tr = common(sub.add_parser("train"))
    tr.add_argument("--stage", required=True)
    tr.add_argument("--resume", action="store_true")
    ev = common(sub.add_parser("eval"))
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--out", required=True, help="directory URI that receives eval_report.json")
    gt = common(sub.add_parser("gate"))
    gt.add_argument("--stage", required=True)
    common(sub.add_parser("run-pipeline"))
    args = p.parse_args(argv)
    _setup_device_env(args.config, args.set)
    return {"validate-config": cmd_validate, "generate": cmd_generate, "export": cmd_export, "prepare-data": cmd_prepare,
            "train": cmd_train, "eval": cmd_eval, "gate": cmd_gate, "run-pipeline": cmd_run_pipeline}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
