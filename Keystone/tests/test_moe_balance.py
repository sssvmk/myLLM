"""PT-9 under data parallelism: routing bias stays identical across ranks (ZeRO 0/2/3) and is
present in the collective full state dict. Regression test for docs/prd_review.md #19."""
import os
import socket

import pytest
import torch

from conftest import FOUNDATION, make_model


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, port, zero_stage, q):
    import sys
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    import torch.distributed as dist
    from src.training.moe_balance import UsageAccumulator, moe_modules
    from src.training.parallel import wrap_for_training
    bridge.fl_distributed.setup_distributed()

    m = make_model("deepseek")
    w = wrap_for_training(m, zero_stage, torch.device("cpu"), find_unused_parameters=True)
    acc = UsageAccumulator(w)
    g = torch.Generator().manual_seed(100 + rank)                      # different data on each rank
    for _ in range(3):                                                 # 3 micro-batches per optimizer step
        idx = torch.randint(0, 256, (4, 16), generator=g)
        logits, _ = w(idx[:, :-1])
        logits.float().mean().backward()
        acc.add()
    local = torch.stack([mm._last_usage for mm in moe_modules(w)]).clone()
    acc.apply()                                                        # raises on FSDP rank>0 without the fix

    bias = torch.stack([mm.routing_bias.detach().clone() for mm in moe_modules(w)])
    all_bias, all_local = [None, None], [None, None]
    dist.all_gather_object(all_bias, bias.tolist())      # plain lists: tensors on an mp queue are shared via
    dist.all_gather_object(all_local, local.tolist())    # a socket that no longer exists after the worker exits
    sd = bridge.fl_distributed.full_model_state_dict(w)               # collective under FSDP: every rank calls it
    saved = None
    if rank == 0:
        keys = sorted(k for k in sd if k.endswith("routing_bias"))
        saved = torch.stack([sd[k] for k in keys]).tolist()
    q.put((rank, all_bias, all_local, saved))
    dist.destroy_process_group()


@pytest.mark.parametrize("zero_stage", [0, 2, 3])
def test_routing_bias_identical_across_ranks(zero_stage):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(_worker, args=(_free_port(), zero_stage, q), nprocs=2, join=True, start_method="spawn")
    res = {r[0]: r for r in (q.get() for _ in range(2))}
    _, all_bias, all_local, saved = res[0]
    all_bias, all_local, saved = [torch.tensor(x) for x in all_bias], [torch.tensor(x) for x in all_local], torch.tensor(saved)
    assert not torch.equal(all_local[0], all_local[1]), "test is only meaningful if local usage differs across ranks"
    assert torch.equal(all_bias[0], all_bias[1]), "routing_bias diverged between ranks"
    assert all_bias[0].abs().sum() > 0, "bias was never updated"
    assert torch.equal(saved, all_bias[0]), "full state dict must carry the updated bias"


def test_accumulator_matches_single_process_update():
    """Single process: accumulating 3 micro-batches equals updating from their summed usage."""
    from src.training.moe_balance import UsageAccumulator, moe_modules
    m = make_model("deepseek")
    ref = make_model("deepseek")
    g = torch.Generator().manual_seed(0)
    acc = UsageAccumulator(m)
    total = None
    for _ in range(3):
        idx = torch.randint(0, 256, (4, 16), generator=g)
        with torch.no_grad():
            m(idx[:, :-1])
            ref(idx[:, :-1])
        acc.add()
        u = torch.stack([mm._last_usage for mm in moe_modules(ref)])
        total = u if total is None else total + u
    acc.apply()
    for mm, u in zip(moe_modules(ref), total):
        mm._last_usage = u
        mm.update_bias()
    a = torch.stack([mm.routing_bias for mm in moe_modules(m)])
    b = torch.stack([mm.routing_bias for mm in moe_modules(ref)])
    assert torch.equal(a, b) and a.abs().sum() > 0


def test_dense_model_is_a_noop():
    from src.training.moe_balance import UsageAccumulator
    acc = UsageAccumulator(make_model("dense"))
    acc.add()
    acc.apply()
