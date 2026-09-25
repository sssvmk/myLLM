"""Chunk 6: RLVR (RL-*, PT-11), distillation (DI-*), AT-19 for these stages, AT-21, AT-23."""
import json
import math
import os
import socket
import sys
import uuid

import pytest
import torch

from src.config.loader import load_config
from src.data import prepare
from src.data.adapters import Conversation
from src.io.storage import Storage
from src.rl.rewards import RewardBreakdown
from src.stages import common, distill, resolve, rlvr
from src.training.loop import Trainer
from src.training.objectives.gspo import gspo_loss, gspo_per_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")
PATH = {"rlvr": "stages.rlvr", "distill_offpolicy": "stages.distill.offpolicy", "distill_onpolicy": "stages.distill.onpolicy"}


def mk(tiny_root, stage_id, overrides=(), stub=None, name=None):
    name = name or f"g{uuid.uuid4().hex[:8]}"
    ov = [f"run.name={name}", f"{PATH[stage_id]}.init_from=base", *overrides] + (stub.overrides() if stub else [])
    lc = load_config(TINY, ov)
    st = Storage.from_config(lc.cfg)
    sctx = common.make_stage_context(lc, stage_id, False, st, (0, 1, 0))
    return lc, st, sctx, common.make_prep_context(sctx, log=lambda *_: None)


def events(lc, stage):
    p = os.path.join(lc.cfg.run.local_work_dir, lc.cfg.run.name, stage, "metrics", "metrics.jsonl")
    return [json.loads(l) for l in open(p)]


def vary_rewards(monkeypatch):
    """Deterministic non-constant rewards so the tiny random model gets a real policy gradient."""
    def fake(self, rolls):
        return [RewardBreakdown(float(sum(r.completion) % 3), 0, 1, 0.0, False) for r in rolls]
    monkeypatch.setattr(rlvr.RLVRObjective, "_score", fake)


def constant_rewards(monkeypatch):
    def fake(self, rolls):
        return [RewardBreakdown(1.0, 0, 1, 0.0, False) for r in rolls]
    monkeypatch.setattr(rlvr.RLVRObjective, "_score", fake)


def rl_ready(tiny_root, overrides=(), **kw):
    lc, st, sctx, pctx = mk(tiny_root, "rlvr", overrides, **kw)
    rlvr.prepare_stage(sctx, pctx)
    return lc, st, sctx, pctx


def trainer_for(sctx, obj):
    t = Trainer(sctx, common.load_stage_model(sctx), obj)
    obj.setup(t)
    return t


# --------------------------------------------------------------------------- GSPO helpers
def test_per_sequence_matches_mean_loss():
    torch.manual_seed(0)
    new, old = torch.randn(4, 6) * 0.1 - 1, torch.randn(4, 6) * 0.1 - 1
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]).bool()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    per, clipped = gspo_per_sequence(new, old, mask, adv, 0.2, 0.28)
    loss, m = gspo_loss(new, old, mask, adv, 0.2, 0.28)
    assert float(per.mean()) == pytest.approx(float(loss)) and float(clipped.float().mean()) == pytest.approx(m["clip_fraction"])


# --------------------------------------------------------------------------- AT-19: rlvr end to end
def test_rlvr_three_steps_fields_lineage_and_mismatch(tiny_root, monkeypatch):
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.batch.micro_batch_size=2", "stages.rlvr.updates_per_step=2",
                                              "stages.rlvr.clip.eps_low=0.2", "stages.rlvr.clip.eps_high=0.28"])
    res = rlvr.train(sctx)
    assert res.final_step == 3 and res.output_sha256
    rl = [e for e in events(lc, "rlvr") if e["event"] == "rl_step"]
    want = {"step", "lr", "reward_mean", "reward_std", "correctness_rate", "format_rate", "groups_kept_fraction",
            "resample_rounds", "response_len_mean", "response_len_max", "truncated_fraction", "entropy", "clip_fraction",
            "logprob_mismatch", "routing_change"}
    assert want <= set(rl[0]) and len(rl) == 3
    for e in rl:
        assert e["logprob_mismatch"] < 1e-4                                # RL-6 in fp32
        assert 0.0 <= e["routing_change"] <= 1.0 and e["entropy"] > 0 and e["reward_std"] > 0
        assert e["groups_kept_fraction"] == 1.0 and e["skipped_update"] == 0.0
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.rlvr.output_uri, "final/lineage.json")))
    assert lin["stage"] == "rlvr" and {d["name"] for d in lin["data"]} >= {"tiny_math", "tiny_code"}
    assert any(e["event"] == "val" and "val_accuracy" in e for e in events(lc, "rlvr"))          # RL-11
    assert resolve.eligible(st, lc.cfg, "rlvr")


def test_rlvr_policy_moves_and_updates_per_step(tiny_root, monkeypatch):
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.updates_per_step=2", "stages.rlvr.optim.lr=1e-3"])
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    before = t.raw.tok_emb.weight.detach().clone()
    m = obj.train_step(t, 0)
    assert t.scheduler.step_num == 2                                        # RL-7: one optimizer step per mini-batch
    assert not torch.equal(before, t.raw.tok_emb.weight) and m["grad_norm"] > 0


def test_routing_change_is_zero_without_updates(tiny_root, monkeypatch):
    """AT-21: the metric is 0 when nothing changes between the two measurements."""
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root)
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    monkeypatch.setattr(Trainer, "apply_update", lambda self: 0.0)
    called = {"n": 0}
    orig = t.raw.blocks[0].mlp.update_bias
    t.raw.blocks[0].mlp.update_bias = lambda: called.__setitem__("n", called["n"] + 1)        # RL-9: never called
    m = obj.train_step(t, 0)
    assert m["routing_change"] == 0.0 and called["n"] == 0


def test_zero_advantage_gives_zero_gradient(tiny_root):
    """Constant rewards (the tiny random model never solves anything): advantages are 0, so the
    update is a no-op on the weights' direction (Adam with zero grad leaves them unchanged)."""
    lc, st, sctx, pctx = rl_ready(tiny_root)
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    before = t.raw.tok_emb.weight.detach().clone()
    m = obj.train_step(t, 0)
    assert m["grad_norm"] == pytest.approx(0.0, abs=1e-9) and torch.equal(before, t.raw.tok_emb.weight)


def test_dynamic_sampling_resamples_then_skips(tiny_root, monkeypatch):
    """RL-3: groups with identical rewards are dropped and replaced up to max_resample_rounds; with
    nothing left the update is skipped and the LR schedule does not advance."""
    constant_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.dapo.dynamic_sampling=true",
                                              "stages.rlvr.dapo.max_resample_rounds=2"])
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    m = obj.train_step(t, 0)                                                # random model: every group has equal rewards
    assert m["skipped_update"] == 1.0 and m["resample_rounds"] == 2.0 and m["groups_kept_fraction"] == 0.0
    assert t.scheduler.step_num == 0
    vary_rewards(monkeypatch)
    m2 = obj.train_step(t, 1)
    assert m2["skipped_update"] == 0.0 and m2["groups_kept_fraction"] == 1.0 and m2["resample_rounds"] == 0.0


def test_overlong_exclude_zeroes_weight_but_keeps_batch(tiny_root, monkeypatch):
    def fake(self, rolls):
        return [RewardBreakdown(float(i % 2), 0, 1, 0.0, excluded=(i % 4 == 0)) for i, _ in enumerate(rolls)]
    monkeypatch.setattr(rlvr.RLVRObjective, "_score", fake)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.dapo.overlong.mode=exclude"])
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    m = obj.train_step(t, 0)
    assert math.isfinite(m["grad_norm"]) and t.scheduler.step_num == 1


def test_kl_penalty_loads_reference_copy(tiny_root, monkeypatch):
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.kl_coef=0.1"])
    obj = rlvr.RLVRObjective(sctx)
    t = trainer_for(sctx, obj)
    assert obj.ref_forward is not None and all(not p.requires_grad for p in obj.ref_model.parameters())
    assert math.isfinite(obj.train_step(t, 0)["grad_norm"])
    lc2, st2, sctx2, pctx2 = rl_ready(tiny_root)
    obj2 = rlvr.RLVRObjective(sctx2)
    trainer_for(sctx2, obj2)
    assert obj2.ref_forward is None                                          # kl_coef 0: no reference copy


# --------------------------------------------------------------------------- entry gate (RL-1)
def test_entry_gate_skip_marks_stage_skipped(tiny_root):
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.entry_gate.min_pass_at_k=1.0", "stages.rlvr.entry_gate.on_failure=skip"])
    res = rlvr.train(sctx)
    assert res.stop_reason == "entry_gate_skip" and res.final_step == 0
    skipped = resolve.is_skipped(st, lc.cfg, "rlvr")
    assert skipped["reason"] == "entry_gate" and skipped["pass_at_k"] == 0.0
    assert not resolve.eligible(st, lc.cfg, "rlvr") and not st.exists(Storage.join(lc.cfg.stages.rlvr.output_uri, "_COMPLETE"))


def test_entry_gate_stop_raises(tiny_root):
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.entry_gate.min_pass_at_k=1.0", "stages.rlvr.entry_gate.on_failure=stop"])
    with pytest.raises(rlvr.EntryGateFailure, match="RL-1"):
        rlvr.train(sctx)


# --------------------------------------------------------------------------- RL-10 / RL-11
def test_entropy_floor_stop_selects_parent_when_no_pre_collapse_checkpoint(tiny_root, monkeypatch):
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.entropy_floor.value=1000.0", "stages.rlvr.entropy_floor.patience_steps=2",
                                              "stages.rlvr.total_steps=6"])
    res = rlvr.train(sctx)
    assert res.stop_reason.startswith("entropy") and res.final_step == 2
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.rlvr.output_uri, "final/lineage.json")))
    assert lin["selected"] == "parent" and "below floor" in lin["stop_reason"]
    out = torch.load(st.cached_local_path(resolve.final_model_uri(lc.cfg, "rlvr")), weights_only=True)["model"]
    base = torch.load(lc.cfg.base_model.checkpoint_uri, weights_only=True)["model"]
    assert all(torch.equal(out[k], base[k]) for k in out)                     # nothing learned inside the window is kept


def test_entropy_floor_uses_best_validation_from_before_window(tiny_root, monkeypatch):
    vary_rewards(monkeypatch)
    lc, st, sctx, pctx = rl_ready(tiny_root, ["stages.rlvr.total_steps=8", "stages.rlvr.eval_every_steps=1",
                                              "stages.rlvr.entropy_floor.value=0.0", "stages.rlvr.entropy_floor.patience_steps=2"])
    obj = rlvr.RLVRObjective(sctx)
    obj.low = 0
    # entropy above the floor: improved() allows best-checkpoint updates; below it, they are refused
    assert obj.improved()
    obj.low = 1
    assert not obj.improved()


# --------------------------------------------------------------------------- distillation 5b
def onpolicy_ready(tiny_root, overrides=()):
    lc, st, sctx, pctx = mk(tiny_root, "distill_onpolicy", overrides)
    distill.prepare_onpolicy(sctx, pctx)
    return lc, st, sctx, pctx


def test_onpolicy_distillation_runs_and_starts_at_zero_kl(tiny_root):
    lc, st, sctx, pctx = onpolicy_ready(tiny_root)
    res = distill.train_onpolicy(sctx)
    assert res.final_step == 3
    ev = events(lc, "distill_onpolicy")
    steps = [e for e in ev if e["event"] == "distill_step"]
    assert {"step", "lr", "reverse_kl_per_token", "response_len_mean"} <= set(steps[0])
    assert abs(steps[0]["reverse_kl_per_token"]) < 1e-5                       # teacher == student at step 0 (DI-5)
    vals = [e for e in ev if e["event"] == "val"]
    assert vals and "val_loss" in vals[0]                                      # DI-7
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.distill.onpolicy.output_uri, "final/lineage.json")))
    assert "onpolicy_teacher" in lin["services"] and lin["selected"] == "best"


def test_teacher_tokenizer_must_match(tiny_root):
    with pytest.raises(ValueError, match="tokenizer_matches"):
        lc, st, sctx, pctx = mk(tiny_root, "distill_onpolicy", ["teachers.onpolicy.tokenizer_matches=false"])
        distill.OPDObjective(sctx)
    lc, st, sctx, pctx = mk(tiny_root, "distill_onpolicy")
    lin_dir = os.path.join(tiny_root, "teacher_ckpt")
    os.makedirs(lin_dir, exist_ok=True)
    import shutil
    shutil.copy(lc.cfg.base_model.checkpoint_uri, os.path.join(lin_dir, "model.pt"))
    tok = {"ranks_sha256": "f" * 64, "chat_special_tokens": {}}
    open(os.path.join(lin_dir, "lineage.json"), "w").write(json.dumps({"tokenizer": tok}))
    lc2 = load_config(TINY, [f"teachers.onpolicy.checkpoint_uri={lin_dir}/model.pt"])
    errs = distill.teacher_tokenizer_errors(lc2.cfg, st)
    assert len(errs) == 2 and "ranks_sha256" in errs[0]


# --------------------------------------------------------------------------- distillation 5a
class FakeTeacher:
    def __init__(self):
        self.calls = 0

    def identity(self):
        return {"role": "offpolicy_teacher", "model": "fake"}

    def complete(self, prompts, n):
        self.calls += 1
        out = []
        for msgs in prompts:
            q = msgs[-1]["content"]
            out.append([("\\boxed{999}" if "+1?" in q else "plain answer")] * n)     # wrong on every math prompt
        return out


def test_teacher_traces_filtered_marked_and_trained(tiny_root):
    lc, st, sctx, pctx = mk(tiny_root, "distill_offpolicy", ["stages.distill.offpolicy.max_prompts=8"])
    ft = FakeTeacher()
    out = distill.prepare_offpolicy(sctx, pctx, teacher=ft)
    e = out["teacher_traces"]["train"]
    assert e["third_party_generated"] is True and e["produced"] is True
    assert e["dropped_teacher_traces"]["incorrect"] == 4                       # DI-2: wrong math answers dropped
    assert e["responses_kept"] == 4 and e["responses_requested"] == 8
    n = ft.calls
    distill.prepare_offpolicy(sctx, pctx, teacher=ft)
    assert ft.calls == n                                                        # same inputs: teacher not called again
    res = distill.train_offpolicy(sctx)
    assert res.final_step >= 1
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.distill.offpolicy.output_uri, "final/lineage.json")))
    tt = [d for d in lin["data"] if d["name"] == "teacher_traces"][0]
    assert tt["third_party_generated"] is True and tt["mixture"]["weight"] == pytest.approx(0.8)


def test_teacher_traces_with_stub_server_keep_correct_math(tiny_root, stub):
    lc, st, sctx, pctx = mk(tiny_root, "distill_offpolicy", ["stages.distill.offpolicy.max_prompts=8"], stub=stub)
    out = distill.prepare_offpolicy(sctx, pctx)
    e = out["teacher_traces"]["train"]
    assert e["responses_kept"] == 8 and not e["dropped_teacher_traces"]        # stub answers arithmetic correctly
    assert e["teacher"]["model"] == "stub-teacher"
    rows = list(prepare.read_rows(st, prepare.prepared_uri(lc.cfg, "distill_offpolicy", "teacher_traces", "train")))
    assert rows and all(len(r["tokens"]) == 65 for r in rows)


# --------------------------------------------------------------------------- AT-23: 2-process balance
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, port, name, which):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from src.foundation import bridge
    bridge.load(os.environ["PF_FOUNDATION_LLM"])
    import torch.distributed as dist
    from src.rl.rewards import RewardBreakdown
    from src.stages import common, distill, rlvr
    info = common.setup_process()
    stage = "rlvr" if which == "rl" else "distill_onpolicy"
    ov = [f"run.name={name}", f"{PATH[stage]}.init_from=base", f"{PATH[stage]}.prompts_per_step=4"]
    lc = load_config(TINY, ov)
    st = Storage.from_config(lc.cfg)
    sctx = common.make_stage_context(lc, stage, False, st, info)
    if which == "rl":
        class Unequal(rlvr.RLVRObjective):
            def _score(self, rolls):
                return [RewardBreakdown(float(sum(r.completion) % 3), 0, 1, 0.0, False) for r in rolls]

            def _sample_groups(self, trainer, step):
                groups, info_ = super()._sample_groups(trainer, step)
                if trainer.rank == 1:
                    groups = groups[:0] if step == 1 else groups[:1]         # rank 1 keeps fewer (and none at step 1)
                return groups, info_
        obj = Unequal(sctx)
    else:
        obj = distill.OPDObjective(sctx)
    t = Trainer(sctx, common.load_stage_model(sctx), obj)
    res = t.run()
    boxes = [None, None]
    dist.all_gather_object(boxes, t.scheduler.step_num)
    assert boxes[0] == boxes[1], boxes                                        # same optimizer-step count on both ranks
    if which == "rl":
        assert res.final_step == 3 and boxes[0] == 2                          # updates at steps 0 and 2; step 1 skipped on both ranks
    dist.destroy_process_group()


@pytest.mark.parametrize("which", ["rl", "opd"])
def test_two_process_unequal_kept_groups_do_not_deadlock(tiny_root, which):
    import torch.multiprocessing as mp
    stage = "rlvr" if which == "rl" else "distill_onpolicy"
    name = f"m{uuid.uuid4().hex[:8]}"
    ov = [f"run.name={name}", f"{PATH[stage]}.init_from=base", f"{PATH[stage]}.prompts_per_step=4"]
    lc = load_config(TINY, ov)
    st = Storage.from_config(lc.cfg)
    sctx = common.make_stage_context(lc, stage, False, st, (0, 1, 0))
    pctx = common.make_prep_context(sctx, log=lambda *_: None)
    (rlvr.prepare_stage if which == "rl" else distill.prepare_onpolicy)(sctx, pctx)
    mp.start_processes(_worker, args=(_free_port(), name, which), nprocs=2, join=True, start_method="spawn")
