"""Chunk 7: evaluation suite end to end on the tiny model (EV-1..EV-6, B-1..B-10, AT-22)."""
import json
import os
import uuid

import pytest
import torch

from src.config.loader import load_config
from src.eval import suite
from src.eval.benchmarks.ifeval import score_ifeval
from src.io.storage import Storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def write_eval_data(root):
    d = os.path.join(root, "data", "eval")
    def jl(name, rows):
        with open(os.path.join(d, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    jl("math500.jsonl", [{"problem": "1+1?", "answer": "2"}, {"problem": "2+2?", "answer": "4"}])
    jl("humaneval.jsonl", [{"prompt": "def f():\n    ", "test": "def check(c):\n    assert c() == 1", "entry_point": "f"}])
    jl("ifeval.jsonl", [{"key": 1, "prompt": "Write about cats in all lowercase.",
                         "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}])
    jl("alpaca.jsonl", [{"instruction": f"tell me {i}", "reference_output": "sure " * (i + 1)} for i in range(4)])
    jl("safety.jsonl", [{"prompt": f"p{i}", "type": "safe" if i % 2 else "unsafe"} for i in range(6)])


EXTRA_DS = """
  eval_math500:
    uri: {root}/data/eval
    files: {{test: "math500.jsonl"}}
    format: jsonl
    kind: eval_math
    adapter: eval_math
    field_map: {{problem: problem, answer: answer}}
    license: synthetic
    third_party_generated: false
    max_samples: null
  eval_humaneval:
    uri: {root}/data/eval
    files: {{test: "humaneval.jsonl"}}
    format: jsonl
    kind: eval_humaneval
    adapter: eval_humaneval
    field_map: {{prompt: prompt, test: test, entry_point: entry_point}}
    license: synthetic
    third_party_generated: false
    max_samples: null
  eval_ifeval:
    uri: {root}/data/eval
    files: {{test: "ifeval.jsonl"}}
    format: jsonl
    kind: eval_ifeval
    adapter: eval_ifeval
    field_map: {{key: key, prompt: prompt, instruction_id_list: instruction_id_list, kwargs: kwargs}}
    license: synthetic
    third_party_generated: false
    max_samples: null
  eval_alpaca:
    uri: {root}/data/eval
    files: {{test: "alpaca.jsonl"}}
    format: jsonl
    kind: eval_alpaca
    adapter: eval_alpaca
    field_map: {{instruction: instruction, reference_output: reference_output}}
    license: synthetic
    third_party_generated: false
    max_samples: null
  eval_safety:
    uri: {root}/data/eval
    files: {{test: "safety.jsonl"}}
    format: jsonl
    kind: eval_safety
    adapter: eval_safety
    field_map: {{prompt: prompt, label: type}}
    safe_label: safe
    license: synthetic
    third_party_generated: false
    max_samples: null
"""


@pytest.fixture()
def full_cfg(tiny_root, tmp_path):
    """tiny_test.yaml plus every benchmark dataset enabled."""
    write_eval_data(str(tiny_root))
    text = open(TINY).read()
    for k, v in BENCH.items():
        text = text.replace(f"    {k}: null", f"    {k}: {v}", 1)
    text = text.replace("\nprepared_data:", EXTRA_DS.format(root="${oc.env:PF_TINY_ROOT}") + "\nprepared_data:", 1)
    p = tmp_path / "full.yaml"
    p.write_text(text)
    return str(p)


BENCH = {
    "math500": "{dataset: eval_math500, max_examples: 2, chat_prompt_uri: '${oc.env:PF_PROMPTS_DIR}/math_chat.txt'}",
    "humaneval": "{dataset: eval_humaneval, max_examples: 1, n_samples: 2, k_values: [1, 2], temperature: 0.8, top_p: 0.95, "
                 "base_stop_strings: ['\\ndef '], chat_prompt_uri: '${oc.env:PF_PROMPTS_DIR}/humaneval_chat.txt'}",
    "ifeval": "{dataset: eval_ifeval, max_examples: 1}",
    "alpaca_eval_lc": "{dataset: eval_alpaca, max_examples: 4, max_iter: 25}",
    "safety": "{dataset: eval_safety, max_examples: 6}",
    "length": "{dataset: eval_alpaca, n_prompts: 3}",
}


def bench_overrides():
    return []


def make_ctx(cfg_path, stub, mode_stage, tiny_root, extra=()):
    lc = load_config(cfg_path, stub.overrides() + bench_overrides() + list(extra) + [f"run.name=e{uuid.uuid4().hex[:6]}"])
    st = Storage.from_config(lc.cfg)
    ckpt = os.path.join(tiny_root, "base", "latest_check.pt")
    ec, _ = suite.build_context(lc.cfg, st, ckpt, torch.device("cpu"), None)
    ec.stage, ec.mode = mode_stage, suite.eval_mode(mode_stage)
    return lc.cfg, st, ec


def test_mode_from_lineage_stage():
    assert [suite.eval_mode(s) for s in (None, "midtrain", "midtrain_long", "sft", "preference", "rlvr", "distill_onpolicy")] == \
        ["base", "base", "base", "chat", "chat", "chat", "chat"]


def test_base_mode_runs_base_benchmarks_and_skips_chat_only(full_cfg, tiny_root, stub):
    cfg, st, ec = make_ctx(full_cfg, stub, None, tiny_root)
    r = suite.run_suite(ec, "ckpt", "abc")
    assert r["mode"] == "base" and r["stage"] is None
    assert {"mmlu_acc", "mmlu_ece", "gsm8k_acc", "math500_acc", "humaneval_pass@1", "humaneval_pass@2"} <= set(r["metrics"])
    assert set(r["skipped"]) == {"ifeval", "alpaca_eval_lc", "safety", "length", "adherence"}
    assert all(0.0 <= v <= 1.0 for v in r["metrics"].values())
    assert r["metrics"]["humaneval_pass@1"] <= r["metrics"]["humaneval_pass@2"]
    assert not stub.requests                                            # base mode never calls the judge


def test_chat_mode_runs_everything(full_cfg, tiny_root, stub):
    cfg, st, ec = make_ctx(full_cfg, stub, "sft", tiny_root)
    r = suite.run_suite(ec, "ckpt", "abc")
    assert r["mode"] == "chat" and not r["skipped"]
    m = r["metrics"]
    assert {"alpaca_win_rate", "alpaca_lc_win_rate", "safety_unsafe_refusal_rate", "safety_safe_compliance_rate",
            "response_len_mean", "response_len_median", "template_adherence", "ifeval_prompt_strict",
            "ifeval_inst_loose"} <= set(m)
    assert m["response_len_mean"] <= cfg.generation.max_new_tokens and m["response_len_median"] > 0
    # judged in both orders for every alpaca instruction, once per safety prompt
    pair = [x for x in stub.requests if "[Response A]" in x[1]]
    assert len(pair) == 8 and len([x for x in stub.requests if "[Response]" in x[1] and "compliance" in x[1]]) == 6


def test_report_hash_changes_only_with_eval_settings(full_cfg, tiny_root, stub):
    cfg, st, ec = make_ctx(full_cfg, stub, None, tiny_root)
    h1 = suite.eval_config_hash(cfg, "s")
    cfg2, _, _ = make_ctx(full_cfg, stub, None, tiny_root, ["eval.benchmarks.mmlu.max_examples=1"])
    assert suite.eval_config_hash(cfg2, "s") != h1
    assert suite.eval_config_hash(cfg2, "t") != suite.eval_config_hash(cfg2, "s")
    cfg3, _, _ = make_ctx(full_cfg, stub, None, tiny_root, ["stages.midtrain.optim.lr=5e-3"])
    assert suite.eval_config_hash(cfg3, "s") == h1


def test_choice_logliks_and_mmlu_argmax(full_cfg, tiny_root, stub):
    cfg, st, ec = make_ctx(full_cfg, stub, None, tiny_root)
    ll = suite.choice_logliks(ec, "2+2?", ["3", "4", "5", "6"])
    assert len(ll) == 4 and all(x < 0 for x in ll)
    # equals the manual sum of token log-probs of the single sequence
    ids = torch.tensor([ec.tok.encode_ordinary("2+2?\nAnswer: 4")])
    with torch.no_grad():
        logits, _ = ec.model(ids[:, :-1])
    lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None])[..., 0].sum()
    assert ll[1] == pytest.approx(float(lp), rel=1e-4)


def test_evaluate_checkpoint_writes_report_and_event(full_cfg, tiny_root, stub, tmp_path):
    from src.metrics.logger import PFMetricsLogger
    cfg, st, ec = make_ctx(full_cfg, stub, None, tiny_root)
    out = f"file://{tmp_path}/evalout"
    lg = PFMetricsLogger(str(tmp_path / "m"), tensorboard=False)
    rep = suite.evaluate_checkpoint(cfg, st, os.path.join(tiny_root, "base", "latest_check.pt"), out, torch.device("cpu"), None,
                                    metrics_logger=lg)
    lg.close()
    saved = json.loads(st.read_text(Storage.join(out, "eval_report.json")))
    assert saved["metrics"] == rep["metrics"] and saved["stage"] is None
    ev = [json.loads(l) for l in open(tmp_path / "m" / "metrics.jsonl")]
    assert ev[0]["event"] == "eval_suite" and ev[0]["mmlu_acc"] == rep["metrics"]["mmlu_acc"]


def test_ifeval_scoring_with_lm_eval_checkers():
    pytest.importorskip("lm_eval")
    docs = [{"key": 1, "prompt": "p", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]},
            {"key": 2, "prompt": "p", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}]
    m = score_ifeval(docs, ["all lowercase here", "Has Capitals"])
    assert m == {"ifeval_prompt_strict": 0.5, "ifeval_prompt_loose": 0.5, "ifeval_inst_strict": 0.5, "ifeval_inst_loose": 0.5}
