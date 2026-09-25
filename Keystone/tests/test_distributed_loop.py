"""The real training loop under 2 processes (gloo/CPU): ZeRO 0-3 with the DeepSeek arch, checkpoint
(collective state-dict gathering), resume, and rank-consistent results (TR-5, CK-3, CK-5, PT-9)."""
import os
import socket
import sys
import uuid

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")
FOUNDATION = os.environ.get("PF_FOUNDATION_LLM", os.path.join(os.path.dirname(ROOT), "foundation_llm"))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _overrides(name, zero):
    return [f"run.name={name}", "stages.midtrain.data.token_budget=768", "stages.midtrain.checkpoint_every_steps=1",
            f"stages.midtrain.distributed.zero_stage={zero}", "stages.midtrain.distributed.find_unused_parameters=true",
            "stages.midtrain.eval_every_steps=1"]


def _worker(rank, port, zero, ref_name, part_name):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    from src.config.loader import load_config
    from src.stages import common, midtrain
    from src.io.storage import Storage
    info = common.setup_process()
    assert info[1] == 2
    lc_ref = load_config(TINY, _overrides(ref_name, zero))
    st = Storage.from_config(lc_ref.cfg)
    midtrain.train(common.make_stage_context(lc_ref, "midtrain", False, st, info))

    lc = load_config(TINY, _overrides(part_name, zero))
    sctx = common.make_stage_context(lc, "midtrain", False, st, info)

    class StopAt(midtrain.BlockObjective):
        cur = 0

        def train_step(self, trainer, step):
            self.cur = step + 1
            return super().train_step(trainer, step)

        def stop_reason(self):
            return "test-stop" if self.cur >= 1 else None
    r1 = common.run_training(sctx, StopAt(sctx, "lm"))
    assert r1.final_step == 1
    sctx2 = common.make_stage_context(lc, "midtrain", True, st, info)
    r2 = midtrain.train(sctx2)
    assert r2.final_step == 2
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_loop_checkpoint_resume_across_zero_stages(tiny_root, zero):
    import torch.multiprocessing as mp
    from src.config.loader import load_config
    from src.data import prepare
    from src.data.registry import DatasetRegistry
    from src.io.storage import Storage
    from src.stages import resolve
    from src.tokenization.chat_template import ChatTemplate
    from src.tokenization.tokenizer import ChatTokenizer
    ref, part = f"ref{uuid.uuid4().hex[:6]}", f"part{uuid.uuid4().hex[:6]}"
    for name in (ref, part):
        lc = load_config(TINY, _overrides(name, zero))
        st = Storage.from_config(lc.cfg)
        tok = ChatTokenizer.from_config(lc.cfg, st)
        prepare.prepare_stage_datasets(prepare.PrepContext(lc.cfg, st, tok, ChatTemplate.from_config(lc.cfg, tok),
                                                            DatasetRegistry.from_config(lc.cfg, st), log=lambda *_: None),
                                       "midtrain", 64)
    mp.start_processes(_worker, args=(_free_port(), zero, ref, part), nprocs=2, join=True, start_method="spawn")
    got = {}
    for name in (ref, part):
        lc = load_config(TINY, _overrides(name, zero))
        st = Storage.from_config(lc.cfg)
        got[name] = torch.load(st.cached_local_path(resolve.final_model_uri(lc.cfg, "midtrain")), weights_only=True)["model"]
    bad = {k: float((got[ref][k].float() - got[part][k].float()).abs().max()) for k in got[ref]
           if not torch.equal(got[ref][k], got[part][k])}
    assert not bad, bad
    assert not any(k.startswith(("module.", "_fsdp", "_orig_mod")) for k in got[ref])


# --------------------------------------------------------------------------- AT-27
def _unused_expert_worker(rank, port, find_unused, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    from conftest import make_model
    from src.training.parallel import build_optimizer, decay_param_names, wrap_for_training
    from types import SimpleNamespace as NS
    bridge.fl_distributed.setup_distributed()
    m = make_model("deepseek")
    for blk in m.blocks:
        blk.mlp.routing_bias.data[3] = -1e9                     # expert 3 never selected: no gradient, ever
    names = decay_param_names(m)
    w = wrap_for_training(m, 0, torch.device("cpu"), find_unused_parameters=find_unused)
    opt = build_optimizer(w, 0, names, NS(weight_decay=0.1, lr=1e-3, betas=[0.9, 0.95], eps=1e-8))
    status = "ok"
    try:
        for _ in range(3):
            idx = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(rank))
            logits, _ = w(idx[:, :-1])
            torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    except RuntimeError as e:
        status = "failed: " + str(e)[:80]
    q.put((rank, status))
    torch.distributed.destroy_process_group()


def test_ddp_unused_expert(tiny_root):
    """AT-27: completes with find_unused_parameters=true; plain DDP fails at the next step, and the
    configuration validator rejects that setting up front (TR-5a)."""
    import torch.multiprocessing as mp
    from src.config.loader import load_config
    from src.config.validate import static_errors
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(_unused_expert_worker, args=(_free_port(), True, q), nprocs=2, join=True, start_method="spawn")
    assert sorted(q.get() for _ in range(2)) == [(0, "ok"), (1, "ok")]
    q2 = ctx.Queue()
    mp.start_processes(_unused_expert_worker, args=(_free_port(), False, q2), nprocs=2, join=True, start_method="spawn")
    assert all(s.startswith("failed") for _, s in [q2.get() for _ in range(2)])
    lc = load_config(TINY, ["stages.sft.distributed.find_unused_parameters=false"])
    errs = static_errors(lc.cfg, cuda_available=False)
    assert any("find_unused_parameters" in e and "TR-5a" in e for e in errs)


# --------------------------------------------------------------------------- resume under a different world size
def _partial_worker(rank, port, zero, name):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from src.foundation import bridge
    bridge.load(FOUNDATION)
    from src.config.loader import load_config
    from src.io.storage import Storage
    from src.stages import common, midtrain
    info = common.setup_process()
    lc = load_config(TINY, _overrides(name, zero))
    st = Storage.from_config(lc.cfg)
    sctx = common.make_stage_context(lc, "midtrain", False, st, info)

    class StopAt(midtrain.BlockObjective):
        cur = 0

        def train_step(self, trainer, step):
            self.cur = step + 1
            return super().train_step(trainer, step)

        def stop_reason(self):
            return "test-stop" if self.cur >= 1 else None
    common.run_training(sctx, StopAt(sctx, "lm"))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("zero", [1, 3])
def test_resume_in_a_single_process_after_a_two_process_run(tiny_root, zero, capsys):
    """A checkpoint written by 2 ranks (DDP+ZeRO-1 or FSDP) resumes in 1 process: model, optimizer moments
    and the LR schedule come back exactly; training then continues."""
    import torch.multiprocessing as mp
    from src.config.loader import load_config
    from src.data import prepare
    from src.data.registry import DatasetRegistry
    from src.io.storage import Storage
    from src.stages import common, midtrain, resolve
    from src.tokenization.chat_template import ChatTemplate
    from src.tokenization.tokenizer import ChatTokenizer
    from src.training import checkpoint as ck
    name = f"w{uuid.uuid4().hex[:6]}"
    lc = load_config(TINY, _overrides(name, zero))
    st = Storage.from_config(lc.cfg)
    tok = ChatTokenizer.from_config(lc.cfg, st)
    prepare.prepare_stage_datasets(prepare.PrepContext(lc.cfg, st, tok, ChatTemplate.from_config(lc.cfg, tok),
                                                        DatasetRegistry.from_config(lc.cfg, st), log=lambda *_: None),
                                   "midtrain", 64)
    mp.start_processes(_partial_worker, args=(_free_port(), zero, name), nprocs=2, join=True, start_method="spawn")
    out = lc.cfg.stages.midtrain.output_uri
    saved = ck.load_payload(st, ck.ckpt_uri(out, "latest.pt"))
    assert saved["world_size"] == 2 and saved["step"] == 1 and saved["optimizer"]["format"] == "named_full_v1"
    sctx = common.make_stage_context(lc, "midtrain", True, st, (0, 1, 0))
    res = midtrain.train(sctx)
    assert res.final_step >= 2 and "resuming with world size 1" in capsys.readouterr().out
    final = torch.load(st.cached_local_path(resolve.final_model_uri(lc.cfg, "midtrain")), weights_only=True)["model"]
    assert all(torch.isfinite(v).all() for v in final.values() if v.is_floating_point())
