"""AT-22 metrics & gates, EX-2 export, AT-28 TensorBoard, ZeRO-2/3 wrapping fix."""
import math
import os
import socket
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import FOUNDATION, arch, make_model
from src.eval.benchmarks.alpaca_lc import lc_win_rate
from src.eval.benchmarks.calibration import choice_confidence, ece
from src.eval.benchmarks.gsm8k import gsm8k_correct
from src.eval.benchmarks.humaneval import pass_at_k
from src.eval.gates import evaluate_gate
from src.io.storage import Storage


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0 and pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(2 / 8)
    assert pass_at_k(8, 2, 8) == 1.0
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - math.comb(7, 5) / math.comb(10, 5))


def test_ece():
    assert ece([0.9, 0.9, 0.3, 0.3], [True, False, False, False], 10) == pytest.approx(0.5 * 0.4 + 0.5 * 0.3)
    conf, pred = choice_confidence([math.log(0.7), math.log(0.2), math.log(0.1)])
    assert conf == pytest.approx(0.7) and pred == 0


def test_lc_win_rate():
    r = lc_win_rate([1, 0, 1, 0], [10, 10, 10, 10], [10, 10, 10, 10], 50)
    assert r["alpaca_lc_win_rate"] == pytest.approx(0.5, abs=1e-6)
    r = lc_win_rate([1, 1, 1, 0, 0.5, 0], [30, 25, 28, 10, 20, 12], [10] * 6, 50)
    raw = r["alpaca_win_rate"]
    assert raw == pytest.approx(3.5 / 6) and r["alpaca_lc_win_rate"] < raw     # longer wins are discounted


def test_gsm8k_parse():
    assert gsm8k_correct("so #### 1,234", "#### 1234", r"####\s*(-?[\d,\.]+)")
    assert gsm8k_correct("the answer is 18.", "#### 18", r"####\s*(-?[\d,\.]+)")


def _gates(**kw):
    g = dict(on_failure="stop", min_improvement=0.0, lower_is_better=["mmlu_ece"],
             tolerances={"mmlu_acc": 0.02, "mmlu_ece": 0.05},
             require_improvement={"rlvr": ["math500_acc"]},
             thresholds=NS(length_ratio_max=1.5, template_adherence_min=0.9, safety_unsafe_refusal_min=0.8,
                           safety_safe_compliance_min=0.8))
    g.update(kw)
    return NS(**g)


def test_gate_directions():
    par = {"mode": "chat", "metrics": {"mmlu_acc": 0.5, "mmlu_ece": 0.10, "math500_acc": 0.2, "response_len_mean": 100}}
    ok = {"mode": "chat", "metrics": {"mmlu_acc": 0.49, "mmlu_ece": 0.14, "math500_acc": 0.25,
                                      "response_len_mean": 140, "template_adherence": 0.95}}
    assert evaluate_gate("rlvr", ok, par, _gates())["passed"]
    worse_ece = dict(ok, metrics=dict(ok["metrics"], mmlu_ece=0.16))
    res = evaluate_gate("rlvr", worse_ece, par, _gates())
    assert not res["passed"] and [c["metric"] for c in res["checks"] if not c["passed"]] == ["mmlu_ece"]
    better_ece = dict(ok, metrics=dict(ok["metrics"], mmlu_ece=0.01))
    assert evaluate_gate("rlvr", better_ece, par, _gates())["passed"]
    long = dict(ok, metrics=dict(ok["metrics"], response_len_mean=160))
    assert not evaluate_gate("rlvr", long, par, _gates())["passed"]
    base_parent = {"mode": "base", "metrics": {"mmlu_acc": 0.5}}
    assert evaluate_gate("sft", {"mode": "chat", "metrics": {"mmlu_acc": 0.5, "template_adherence": 0.5}},
                         base_parent, _gates())["passed"] is False           # threshold applies in chat mode


def test_export_roundtrip(tmp_path, tok):
    from src.export.export import export_model, load_exported
    st = Storage({}, 1, str(tmp_path / "work"))
    ranks = tmp_path / "r.tiktoken"
    ranks.write_bytes(b"x")
    m = make_model("deepseek")
    export_model(m, arch("deepseek"), st, f"file://{tmp_path}/exp", "fp32", str(ranks),
                 {"ranks_sha256": "abc", "eot_token_id": 100257, "pad_token_id": 100277}, {"turn_start": "x"})
    m2, a, t = load_exported(f"file://{tmp_path}/exp", st, torch.device("cpu"))
    idx = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        assert torch.equal(m(idx)[0], m2(idx)[0])
    assert a["tied_embeddings"] is True and t["ranks_file"] == "cl100k_base.tiktoken"
    assert (tmp_path / "exp" / "cl100k_base.tiktoken").read_bytes() == b"x"


def test_tensorboard_generic_events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from src.metrics.logger import PFMetricsLogger
    lg = PFMetricsLogger(str(tmp_path), tensorboard=True)
    lg.log("sft_step", step=1, loss=2.0, lr=1e-3)
    lg.log("rl_step", step=1, reward_mean=0.3)
    lg.log("moe_usage", step=1, usage_per_layer=[[3, 1], [2, 2]])
    lg.close()
    ea = EventAccumulator(str(tmp_path / "tensorboard"))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    assert {"sft_step/loss", "rl_step/reward_mean", "moe/layer0/expert_0"} <= tags
    assert (tmp_path / "metrics.jsonl").read_text().count("\n") == 3


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _fsdp_worker(rank, port, zero_stage, q):
    import sys
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    from src.training.parallel import build_optimizer, decay_param_names, wrap_for_training
    bridge.fl_distributed.setup_distributed()
    m = make_model("deepseek")
    names = decay_param_names(m)
    w = wrap_for_training(m, zero_stage, torch.device("cpu"), find_unused_parameters=True)
    opt = build_optimizer(w, zero_stage, names, NS(weight_decay=0.1, lr=1e-3, betas=[0.9, 0.95], eps=1e-8))
    idx = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(rank))
    logits, aux = w(idx[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    n_decay = sum(p.numel() for p in opt.param_groups[0]["params"])
    n_nodecay = sum(p.numel() for p in opt.param_groups[1]["params"])
    q.put((rank, n_decay, n_nodecay, float(loss)))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("zero_stage", [0, 2, 3])
def test_deepseek_zero_wrapping(zero_stage):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(_fsdp_worker, args=(_free_port(), zero_stage, q), nprocs=2, join=True, start_method="spawn")
    res = sorted(q.get() for _ in range(2))
    total_decay = sum(r[1] for r in res)
    assert total_decay > 0, "weight decay group must not be empty"
    assert all(math.isfinite(r[3]) for r in res)
