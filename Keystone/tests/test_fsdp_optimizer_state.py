"""Optimizer state under FSDP (docs/prd_review.md #20): the hand-assembled full state equals the ground
truth of a plain AdamW step, and it moves between ZeRO stages and world sizes."""
import os
import socket
import sys
import uuid

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUNDATION = os.environ.get("PF_FOUNDATION_LLM", os.path.join(os.path.dirname(ROOT), "foundation_llm"))
BATCHES = [torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(r)) for r in range(2)]
LR, WD, BETAS = 1e-3, 0.1, [0.9, 0.95]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _loss(model, idx):
    logits, _ = model(idx[:, :-1])
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))


def _truth():
    """Moments after ONE AdamW step from the mean of the two ranks' gradients (plain model, no FSDP)."""
    from conftest import make_model
    m = make_model("dense")
    grads = []
    for b in BATCHES:
        m.zero_grad()
        _loss(m, b).backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    avg = {n: (grads[0][n] + grads[1][n]) / 2 for n in grads[0]}
    return {n: {"exp_avg": (1 - BETAS[0]) * g, "exp_avg_sq": (1 - BETAS[1]) * g * g} for n, g in avg.items()}


def _setup(rank, port, world):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    bridge.fl_distributed.setup_distributed()


def _build(zero):
    from types import SimpleNamespace as NS
    from conftest import make_model
    from src.training.parallel import build_optimizer, decay_param_names, wrap_for_training
    m = make_model("dense")
    names = decay_param_names(m)
    w = wrap_for_training(m, zero, torch.device("cpu"), True)
    return w, build_optimizer(w, zero, names, NS(weight_decay=WD, lr=LR, betas=BETAS, eps=1e-8))


def _fsdp_worker(rank, port, zero, out_file, in_file, q):
    _setup(rank, port, 2)
    from src.foundation import bridge
    from src.training import checkpoint as ck
    w, opt = _build(zero)
    if in_file:                                        # state written by a single-process run: load it sharded
        named = torch.load(in_file, weights_only=True)
        ck.load_named_optimizer_state(w, opt, named, sharded=True)
    else:
        _loss(w, BATCHES[rank]).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    msd = bridge.fl_distributed.full_model_state_dict(w)
    named = ck.gather_named_optimizer_state(w, opt, ck.strip_prefixes(msd) if rank == 0 else {})
    if rank == 0:
        torch.save(named, out_file)
        q.put("saved")
    torch.distributed.destroy_process_group()


def _run_fsdp(zero, out_file, in_file=None):
    import torch.multiprocessing as mp
    q = mp.get_context("spawn").Queue()
    mp.start_processes(_fsdp_worker, args=(_free_port(), zero, out_file, in_file, q), nprocs=2, join=True,
                       start_method="spawn")
    assert q.get() == "saved"


def _close(a, b, tol=1e-6):
    return float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))


@pytest.mark.parametrize("zero", [2, 3])
def test_gathered_state_equals_plain_adamw_truth(tmp_path, zero):
    out = str(tmp_path / "named.pt")
    _run_fsdp(zero, out)
    named = torch.load(out, weights_only=True)
    assert named["format"] == "named_full_v1"
    truth = _truth()
    got = {k.replace("lm_head.weight", "tok_emb.weight"): v for k, v in named["state"].items()}
    assert set(truth) <= set(got) | {"lm_head.weight"}
    for n, t in truth.items():
        if n not in got:
            continue                                     # tied lm_head: state lives under tok_emb.weight
        for key in ("exp_avg", "exp_avg_sq"):
            assert got[n][key].shape == t[key].shape and _close(got[n][key], t[key]), (n, key)
        assert float(got[n]["step"]) == 1.0


def test_state_moves_from_world_2_to_a_single_process_and_back(tmp_path):
    """FSDP (2 ranks) -> plain single process -> FSDP (2 ranks) reproduces the same full state."""
    from conftest import make_model
    from src.foundation import bridge
    from src.training import checkpoint as ck
    from src.training.parallel import build_optimizer, decay_param_names
    from types import SimpleNamespace as NS
    f2 = str(tmp_path / "from_fsdp.pt")
    _run_fsdp(3, f2)
    named = torch.load(f2, weights_only=True)
    # single process: load, then read the moments back and compare with the file and the truth
    m = make_model("dense")
    opt = build_optimizer(m, 0, decay_param_names(m), NS(weight_decay=WD, lr=LR, betas=BETAS, eps=1e-8))
    ck.load_named_optimizer_state(m, opt, named, sharded=False)
    truth = _truth()
    byname = dict(m.named_parameters())
    for n, p in byname.items():
        if p not in opt.state:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.state[p][key], named["state"][n][key]) or n == "lm_head.weight"
            assert _close(opt.state[p][key], truth[n][key])
    # and back out of the single process, then into a 2-rank FSDP run, and gather again
    payload = ck.optimizer_payload(m, opt, {})
    f1 = str(tmp_path / "from_single.pt")
    torch.save(payload, f1)
    f3 = str(tmp_path / "roundtrip.pt")
    _run_fsdp(2, f3, in_file=f1)
    back = torch.load(f3, weights_only=True)
    assert set(back["state"]) == set(payload["state"])
    for n in payload["state"]:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(back["state"][n][key], payload["state"][n][key]), (n, key)


def test_rank_entry_rescales_data_position():
    from src.training.checkpoint import rank_entry
    payload = {"world_size": 4, "ranks": [{"objective": {"draws": 40}, "rng": 0}] * 4}
    assert rank_entry(payload, 1, 4)["objective"]["draws"] == 40                 # same topology: as saved
    assert rank_entry(payload, 0, 2)["objective"]["draws"] == 80                 # same rows consumed over half the ranks
    assert rank_entry(payload, 0, 8)["objective"]["draws"] == 20
