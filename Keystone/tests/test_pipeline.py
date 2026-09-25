"""Chunk 8: PL-2 resolution (AT-25), the whole pipeline on the tiny config (AT-20), torchrun launch and
resume-by-hash (AT-29, PL-5, PL-7, PL-8), CLI commands, device checks (AT-26)."""
import json
import os
import subprocess
import sys
import uuid

import pytest
import torch

from src import cli
from src.config.loader import load_config
from src.export.export import load_exported
from src.io.storage import Storage
from src.stages import pipeline, resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def new_lc(extra=(), stub=None, name=None):
    ov = [f"run.name={name or 'pl' + uuid.uuid4().hex[:8]}", *extra] + (stub.overrides() if stub else [])
    lc = load_config(TINY, ov)
    return lc, Storage.from_config(lc.cfg)


# --------------------------------------------------------------------------- AT-25
def fake_output(st, cfg, sid, gate=None, skipped=False, complete=True):
    out = resolve.out_uri(cfg, sid)
    if complete:
        st.write_text(Storage.join(out, "_COMPLETE"), json.dumps({"config_hash": "h", "output_sha256": f"sha_{sid}", "final_step": 1}))
        arch = cfg.base_model.architecture.model_dump()
        st.write_text(Storage.join(out, "final/lineage.json"), json.dumps({"stage": sid, "output_sha256": f"sha_{sid}", "architecture": arch}))
    if gate is not None:
        st.write_text(Storage.join(out, "final/gate_result.json"), json.dumps({"passed": gate}))
    if skipped:
        st.write_text(Storage.join(out, "_SKIPPED"), json.dumps({"reason": "entry_gate"}))


def test_init_from_previous_passes_over_failed_and_skipped_stages(tiny_root):
    lc, st = new_lc(["stages.midtrain_long.enabled=true"])
    cfg = lc.cfg
    fake_output(st, cfg, "midtrain", gate=True)
    fake_output(st, cfg, "midtrain_long", gate=False)                     # failed its gate under continue
    fake_output(st, cfg, "sft", gate=True)
    fake_output(st, cfg, "preference", gate=False)                        # failed
    fake_output(st, cfg, "rlvr", gate=None, skipped=True)                 # skipped by its entry gate
    assert resolve.previous_stage(st, cfg, "rlvr") == "sft"               # 1b and preference passed over
    p = resolve.resolve_parent(st, cfg, "distill_offpolicy")
    assert p.kind == "stage" and p.stage == "sft" and p.sha256 == "sha_sft"
    assert resolve.previous_stage(st, cfg, "midtrain_long") == "midtrain"
    assert resolve.previous_stage(st, cfg, "midtrain") is None
    assert resolve.final_stage(st, cfg) == "sft"                          # PL-6: failed/skipped are not eligible
    lc2, st2 = new_lc(["stages.midtrain_long.enabled=true"])
    fake_output(st2, lc2.cfg, "midtrain", gate=False)
    assert resolve.resolve_parent(st2, lc2.cfg, "sft").kind == "base"     # nothing qualifies -> base


# --------------------------------------------------------------------------- AT-26
def test_device_checks_and_gloo_subprocess(tiny_root):
    from src.config.validate import device_errors
    lc, _ = new_lc(["run.device=cuda"])
    assert any("DV-2" in e for e in device_errors(lc.cfg, cuda_available=False))
    lc, _ = new_lc(["run.device=cpu", "run.precision=fp16"])
    assert any("DV-5" in e for e in device_errors(lc.cfg, cuda_available=False))
    lc, _ = new_lc(["run.device=auto"])
    assert device_errors(lc.cfg, cuda_available=False) == []
    code = ("import os,sys;from src import cli;cli._setup_device_env(sys.argv[1], []);"
            "import torch;print(torch.cuda.is_available());"
            "from src.foundation import bridge;bridge.load(os.environ['PF_FOUNDATION_LLM']);"
            "os.environ.update(RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='29777');"
            "bridge.fl_distributed.setup_distributed();import torch.distributed as d;print(d.get_backend())")
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    out = subprocess.check_output([sys.executable, "-c", code, TINY], env=env, text=True).split()
    assert out[-2:] == ["False", "gloo"]


# --------------------------------------------------------------------------- AT-20
def test_full_pipeline_gates_resume_export_and_cli(tiny_root, stub, monkeypatch, tmp_path, capsys):
    lc, st = new_lc(stub=stub)
    cfg = lc.cfg
    res = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert res.ok, res.message
    by = {o.stage: o for o in res.outcomes}
    assert by["midtrain_long"].status == "disabled"
    for sid in ("midtrain", "sft", "preference", "rlvr", "distill_offpolicy", "distill_onpolicy"):
        assert by[sid].status == "trained" and by[sid].gate_passed is True, sid
        out = resolve.out_uri(cfg, sid)
        assert st.exists(Storage.join(out, "_COMPLETE")) and st.exists(Storage.join(out, "final/gate_result.json"))
        assert st.exists(Storage.join(out, "final/eval_report.json")), sid
    # lineage chain: each stage's parent is the previous stage's output
    chain = ["midtrain", "sft", "preference", "rlvr", "distill_offpolicy", "distill_onpolicy"]
    for prev, cur in zip(chain, chain[1:]):
        lin = json.loads(st.read_text(Storage.join(resolve.out_uri(cfg, cur), "final/lineage.json")))
        pl = json.loads(st.read_text(Storage.join(resolve.out_uri(cfg, prev), "final/lineage.json")))
        assert lin["parent"]["stage"] == prev and lin["parent"]["sha256"] == pl["output_sha256"]
    assert res.final_stage == "distill_onpolicy"
    # eval modes: base before Stage 2, chat after (EV-2)
    assert json.loads(st.read_text(Storage.join(resolve.out_uri(cfg, "midtrain"), "final/eval_report.json")))["mode"] == "base"
    sft_rep = json.loads(st.read_text(Storage.join(resolve.out_uri(cfg, "sft"), "final/eval_report.json")))
    assert sft_rep["mode"] == "chat" and "template_adherence" in sft_rep["metrics"]
    assert st.exists(Storage.join(pipeline.base_dir_uri(cfg), "eval_report.json"))
    # EX-2: the exported model reloads with logits identical to the final checkpoint (fp32 export)
    from src.modeling.loading import load_model, load_state
    final = load_state(st.cached_local_path(resolve.final_model_uri(cfg, "distill_onpolicy")))
    src = load_model(final, final["lineage"]["architecture"], torch.device("cpu")).eval()
    exp, arch, tokrec = load_exported(cfg.export.output_uri, st, torch.device("cpu"))
    idx = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        assert torch.equal(src(idx)[0], exp(idx)[0])
    assert os.path.exists(st.local_path(Storage.join(cfg.export.output_uri, "eval_report.json")))

    # PL-5: a second run touches nothing
    import src.eval.suite as suite
    for d in (pipeline.TRAIN, pipeline.PREPARE):
        for k in list(d):
            monkeypatch.setitem(d, k, lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerun")))
    monkeypatch.setattr(suite, "evaluate_checkpoint", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-eval")))
    res2 = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert res2.ok and all(o.status in ("reused", "disabled") for o in res2.outcomes)

    # CLI: eval / gate / export on the finished run (same config file + overrides)
    monkeypatch.undo()
    base = ["-c", TINY, "--set", f"run.name={cfg.run.name}"] + sum((["--set", o] for o in stub.overrides()), [])
    out_dir = f"file://{tmp_path}/cli_eval"
    ckpt = resolve.final_model_uri(cfg, "sft")
    assert cli.main(["eval", *base, "--checkpoint", ckpt, "--out", out_dir]) == 0
    assert json.loads(st.read_text(Storage.join(out_dir, "eval_report.json")))["stage"] == "sft"
    assert cli.main(["gate", *base, "--stage", "sft"]) == 0
    assert cli.main(["export", *base]) == 0 and "final model is stage distill_onpolicy" in capsys.readouterr().out
    assert cli.main(["prepare-data", *base, "--stage", "sft"]) == 0


def test_gate_failure_stops_or_continues(tiny_root, stub):
    """GT-3: on_failure stop ends the pipeline; continue records it and later stages skip the stage."""
    only_mid = ["stages.sft.enabled=false", "stages.preference.enabled=false", "stages.rlvr.enabled=false",
                "stages.distill.enabled=false", "stages.midtrain.data.token_budget=768",
                "gates.require_improvement.midtrain=[mmlu_acc]", "gates.min_improvement=5.0"]
    lc, st = new_lc(only_mid + ["gates.on_failure=stop"], stub)
    res = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert not res.ok and "gate failed" in res.message and res.outcomes[-1].gate_passed is False
    assert resolve.final_stage(st, lc.cfg) is None
    lc2, st2 = new_lc(only_mid + ["gates.on_failure=continue"], stub)
    res2 = pipeline.run_pipeline(lc2, st2, log=lambda *_: None)
    assert not res2.ok and "nothing to export" in res2.message                         # failed stage is not a final model (GT-3)


def test_entry_gate_skip_and_stop_in_pipeline(tiny_root, stub):
    base = ["stages.sft.enabled=false", "stages.preference.enabled=false", "stages.distill.enabled=false",
            "stages.midtrain.data.token_budget=768", "stages.rlvr.entry_gate.min_pass_at_k=1.0"]
    lc, st = new_lc(base + ["stages.rlvr.entry_gate.on_failure=skip"], stub)
    res = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert res.ok and {o.stage: o.status for o in res.outcomes}["rlvr"] == "skipped" and res.final_stage == "midtrain"
    sk = resolve.is_skipped(st, lc.cfg, "rlvr")
    assert sk["reason"] == "entry_gate" and sk["config_hash"]
    lc2, st2 = new_lc(base + ["stages.rlvr.entry_gate.on_failure=stop"], stub)
    res2 = pipeline.run_pipeline(lc2, st2, log=lambda *_: None)
    assert not res2.ok and res2.outcomes[-1].status == "stopped"


# --------------------------------------------------------------------------- AT-29
def test_torchrun_launch_reuse_and_eval_rerun(tiny_root, stub):
    only_mid = ["stages.sft.enabled=false", "stages.preference.enabled=false", "stages.rlvr.enabled=false",
                "stages.distill.enabled=false", "stages.midtrain.data.token_budget=768", "launcher.nproc_per_node=2"]
    name = "tr" + uuid.uuid4().hex[:6]
    lc, st = new_lc(only_mid, stub, name)
    res = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert res.ok, res.message
    out = resolve.out_uri(lc.cfg, "midtrain")
    lin = json.loads(st.read_text(Storage.join(out, "final/lineage.json")))
    assert lin["stage"] == "midtrain" and st.exists(Storage.join(out, "checkpoints/latest.json"))
    ck = torch.load(st.cached_local_path(Storage.join(out, "checkpoints/latest.pt")), weights_only=True)
    assert ck["world_size"] == 2                                                        # trained by two torchrun ranks (PL-7)
    complete = st.read_text(Storage.join(out, "_COMPLETE"))
    rep1 = json.loads(st.read_text(Storage.join(out, "final/eval_report.json")))
    # change only an evaluation setting: evaluation reruns, training does not
    lc2, st2 = new_lc(only_mid + ["eval.benchmarks.mmlu.max_examples=1"], stub, name)
    res2 = pipeline.run_pipeline(lc2, st2, log=lambda *_: None)
    assert res2.ok and res2.outcomes[0].status == "reused"
    assert st.read_text(Storage.join(out, "_COMPLETE")) == complete
    rep2 = json.loads(st.read_text(Storage.join(out, "final/eval_report.json")))
    assert rep2["eval_config_hash"] != rep1["eval_config_hash"] and rep2["details"]["mmlu_n"] == 1
    # changing a training setting invalidates the completed stage and retrains it
    lc3, st3 = new_lc(only_mid + ["stages.midtrain.optim.lr=2e-3"], stub, name)
    res3 = pipeline.run_pipeline(lc3, st3, log=lambda *_: None)
    assert res3.ok and res3.outcomes[0].status == "trained" and st.read_text(Storage.join(out, "_COMPLETE")) != complete


def test_nnodes_above_one_is_a_validation_error(tiny_root, stub):
    lc, st = new_lc(["launcher.nnodes=2"], stub)
    with pytest.raises(pipeline.PipelineError, match="PL-8"):
        pipeline.run_pipeline(lc, st)
    rc = cli.main(["run-pipeline", "-c", TINY, "--set", "launcher.nnodes=2", "--set", f"run.name=nn{uuid.uuid4().hex[:5]}"])
    assert rc == 2


def test_train_nonzero_exit_leaves_stage_incomplete(tiny_root, stub, monkeypatch):
    monkeypatch.setattr(pipeline, "launch_training", lambda *a, **k: 1)
    lc, st = new_lc(["launcher.nproc_per_node=2", "stages.midtrain.data.token_budget=768"], stub)
    res = pipeline.run_pipeline(lc, st, log=lambda *_: None)
    assert not res.ok and res.outcomes[-1].status == "failed_training" and "PL-7" in res.message
    assert not st.exists(Storage.join(resolve.out_uri(lc.cfg, "midtrain"), "_COMPLETE"))
