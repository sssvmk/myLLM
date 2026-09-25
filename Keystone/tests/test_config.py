"""AT-1 (config), AT-26 validation part, static rules."""
import os

import pytest

from src.config.loader import ConfigError, load_config, to_yaml
from src.config.validate import environment_errors, static_errors
from src.io.storage import Storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def test_tiny_loads_and_passes_static(tiny_root):
    lc = load_config(TINY)
    assert static_errors(lc.cfg, cuda_available=False) == []


def test_environment_validation_passes(tiny_root):
    lc = load_config(TINY)
    st = Storage.from_config(lc.cfg)
    assert environment_errors(lc.cfg, st, check_endpoints=False) == []


def test_missing_key_named(tiny_root, tmp_path):
    txt = open(TINY).read().replace("  seed: 1234\n", "", 1)
    p = tmp_path / "c.yaml"
    p.write_text(txt)
    with pytest.raises(ConfigError, match=r"run\.seed: missing required key"):
        load_config(str(p))


def test_unknown_key_named(tiny_root):
    with pytest.raises(ConfigError, match=r"run\.bogus: unknown key"):
        load_config(TINY, ["run.bogus=1"])


def test_bad_type_named(tiny_root):
    with pytest.raises(ConfigError, match=r"stages\.sft\.optim\.lr"):
        load_config(TINY, ["stages.sft.optim.lr=abc"])


def test_set_override_applies(tiny_root):
    lc = load_config(TINY, ["stages.sft.optim.lr=2e-5", "run.name=exp2"])
    assert lc.cfg.stages.sft.optim.lr == 2e-5 and lc.cfg.run.name == "exp2"
    assert lc.cfg.stages.sft.output_uri.endswith("/exp2/sft")     # interpolation after override


def test_secrets_redacted(tiny_root, monkeypatch):
    monkeypatch.setenv("PF_SECRET", "hunter2")
    lc = load_config(TINY, ["storage.protocols.abfss.account_key=${oc.env:PF_SECRET}",
                            "storage.protocols.abfss.account_name=acct"])
    assert "hunter2" not in to_yaml(lc.redacted)
    assert lc.resolved["storage"]["protocols"]["abfss"]["account_key"] == "hunter2"
    assert lc.redacted["tokenizer"]["eot_token_id"] == 100257     # ids that contain 'token' survive


def test_judge_equal_teacher_fails(tiny_root):
    lc = load_config(TINY, ["teachers.offpolicy.model=stub-judge"])
    assert any("JG-5" in e for e in static_errors(lc.cfg, cuda_available=False))


def test_device_rules(tiny_root):
    lc = load_config(TINY, ["run.precision=fp16"])
    assert any("DV-5" in e for e in static_errors(lc.cfg, cuda_available=False))
    lc = load_config(TINY, ["run.device=cuda"])
    assert any("DV-2" in e for e in static_errors(lc.cfg, cuda_available=False))
    lc = load_config(TINY, ["run.device=auto"])
    assert not any("DV-" in e for e in static_errors(lc.cfg, cuda_available=False))


def test_stage_rules(tiny_root):
    errs = static_errors(load_config(TINY, ["stages.rlvr.distributed.zero_stage=2",
                                            "stages.rlvr.batch.grad_accum=2",
                                            "stages.rlvr.moe.update_routing_bias=true",
                                            "stages.sft.distributed.find_unused_parameters=false",
                                            "launcher.nnodes=2"]).cfg, cuda_available=False)
    for tag in ("TR-5)", "RL-7a", "RL-9", "TR-5a", "PL-8"):
        assert any(tag in e for e in errs), tag
    errs = static_errors(load_config(TINY, ["launcher.nproc_per_node=3"]).cfg, cuda_available=False)
    assert any("RL-3a" in e for e in errs)


def test_produced_dataset_only_in_producer(tiny_root):
    errs = static_errors(load_config(TINY, ["stages.sft.data.sources.0.dataset=teacher_traces"]).cfg,
                         cuda_available=False)
    assert any("DS-3b" in e for e in errs)


def test_wsd_fractions_must_sum(tiny_root):
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(TINY, ["stages.midtrain.schedule.decay_fraction=0.5"])


def test_example_config_parses(monkeypatch):
    for k in ("PF_AZURE_ACCOUNT", "PF_AZURE_KEY", "HF_TOKEN", "PF_JUDGE_BASE_URL", "PF_TEACHER_BASE_URL"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("PF_JUDGE_MODEL", "judge")
    monkeypatch.setenv("PF_TEACHER_MODEL", "teacher")
    lc = load_config(os.path.join(ROOT, "configs", "example_deepseek_1p3b.yaml"))
    assert static_errors(lc.cfg, cuda_available=True, for_pipeline=True) == []


def test_cli_validate_and_cpu_env(tiny_root):
    """AT-26 (partial): a CLI subprocess with run.device=cpu sees no CUDA and validates the tiny config."""
    import subprocess
    import sys
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = ROOT
    r = subprocess.run([sys.executable, "-m", "src.cli", "validate-config", "-c", TINY, "--skip-endpoints"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    code = ("import os,sys; sys.argv=['pf']; from src import cli; "
            f"cli._setup_device_env({TINY!r}, []); import torch; print(torch.cuda.is_available())")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stderr
