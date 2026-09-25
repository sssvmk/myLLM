"""Chunk 4: one loop for LM and SFT stages, checkpoints, resume, lineage, MoE bias switch (AT-19 for
midtrain/sft, AT-21, CK-*, MT-L1/L2, PT-9/10)."""
import json
import os
import uuid

import pytest
import torch

from src.config.loader import load_config
from src.data import prepare
from src.data.registry import DatasetRegistry
from src.io.storage import Storage
from src.stages import common, midtrain, resolve, sft
from src.tokenization.chat_template import ChatTemplate
from src.tokenization.tokenizer import ChatTokenizer
from src.training import checkpoint as ck

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def make_run(tiny_root, stage_id, overrides=(), resume=False, name=None):
    name = name or f"r{uuid.uuid4().hex[:8]}"
    lc = load_config(TINY, [f"run.name={name}", *overrides])
    st = Storage.from_config(lc.cfg)
    return lc, st


def prep(lc, st, stage_id):
    tok = ChatTokenizer.from_config(lc.cfg, st)
    ctx = prepare.PrepContext(lc.cfg, st, tok, ChatTemplate.from_config(lc.cfg, tok),
                              DatasetRegistry.from_config(lc.cfg, st), log=lambda *_: None)
    prepare.prepare_stage_datasets(ctx, stage_id, lc.cfg.base_model.architecture.ctx)


def final_state(st, lc, stage_id):
    return torch.load(st.cached_local_path(resolve.final_model_uri(lc.cfg, stage_id)), weights_only=True)


def events(lc, stage_id):
    p = os.path.join(lc.cfg.run.local_work_dir, lc.cfg.run.name, stage_id, "metrics", "metrics.jsonl")
    return [json.loads(l) for l in open(p)]


class StopAt(midtrain.BlockObjective):
    """Test double: ends the loop early as if the process died after `at` steps."""
    def __init__(self, sctx, kind, at):
        super().__init__(sctx, kind)
        self.at, self.cur = at, 0

    def train_step(self, trainer, step):
        self.cur = step + 1
        return super().train_step(trainer, step)

    def stop_reason(self):
        return "test-stop" if self.cur >= self.at else None


# --------------------------------------------------------------------------- AT-19: midtrain
MID = ["stages.midtrain.data.token_budget=768"]           # 768 / (2*2*64) = 3 optimizer steps


def test_midtrain_three_steps_checkpoint_lineage_and_events(tiny_root):
    lc, st = make_run(tiny_root, "midtrain", MID)
    prep(lc, st, "midtrain")
    sctx = common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0))
    res = midtrain.train(sctx)
    assert res.final_step == 3 and res.stop_reason is None and res.output_sha256
    out = lc.cfg.stages.midtrain.output_uri
    for rel in ("checkpoints/latest.pt", "checkpoints/step_000003.pt", "final/model.pt", "final/lineage.json",
                "final/resolved_config.yaml", "_COMPLETE", "metrics/metrics.jsonl"):
        assert st.exists(Storage.join(out, rel)), rel
    complete = json.loads(st.read_text(Storage.join(out, "_COMPLETE")))
    assert complete["config_hash"] == sctx.config_hash and complete["output_sha256"] == res.output_sha256
    lin = json.loads(st.read_text(Storage.join(out, "final/lineage.json")))
    assert lin["stage"] == "midtrain" and lin["parent"]["stage"] is None and lin["architecture"]["arch"] == "deepseek"
    assert {d["name"] for d in lin["data"]} == {"tiny_text", "tiny_packed"}
    assert lin["tokenizer"]["eot_token_id"] == 100257 and lin["final_step"] == 3 and lin["device"] == "cpu"
    ev = events(lc, "midtrain")
    kinds = {e["event"] for e in ev}
    assert {"lm_step", "val", "moe_usage"} <= kinds
    step_ev = [e for e in ev if e["event"] == "lm_step"]
    assert {"step", "loss", "lr", "tokens_per_sec", "grad_norm"} <= set(step_ev[0]) and len(step_ev) == 3
    val = [e for e in ev if e["event"] == "val"][-1]
    assert "val_loss" in val and "retention_loss" in val                  # S1-4, reported separately
    # redacted config saved alongside (CF-5)
    assert "ranks_sha256" in st.read_text(Storage.join(out, "final/resolved_config.yaml"))
    # CK-4: no wrapper prefixes; weights load into a fresh model
    sd = final_state(st, lc, "midtrain")
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in sd["model"]) and sd["lineage"]["stage"] == "midtrain"


def test_wsd_lr_recorded_and_loss_decreases_with_repeated_data(tiny_root):
    lc, st = make_run(tiny_root, "midtrain", ["stages.midtrain.data.token_budget=2560"])
    prep(lc, st, "midtrain")
    common_ctx = common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0))
    midtrain.train(common_ctx)
    lrs = [e["lr"] for e in events(lc, "midtrain") if e["event"] == "lm_step"]
    assert lrs[0] == 0.0 and max(lrs) == pytest.approx(1e-3) and lrs[-1] < max(lrs)      # warmup from 0, then decay
    losses = [e["loss"] for e in events(lc, "midtrain") if e["event"] == "lm_step"]
    assert losses[-1] < losses[0]


def test_resume_reaches_identical_state(tiny_root):
    lc_a, st = make_run(tiny_root, "midtrain", MID)
    prep(lc_a, st, "midtrain")
    common_a = common.make_stage_context(lc_a, "midtrain", False, st, (0, 1, 0))
    midtrain.train(common_a)
    ref = final_state(st, lc_a, "midtrain")["model"]

    lc_b, _ = make_run(tiny_root, "midtrain", MID)
    prep(lc_b, st, "midtrain")
    sctx = common.make_stage_context(lc_b, "midtrain", False, st, (0, 1, 0))
    part = common.run_training(sctx, StopAt(sctx, "lm", 2))
    assert part.final_step == 2 and part.stop_reason == "test-stop"
    latest = ck.load_payload(st, ck.ckpt_uri(lc_b.cfg.stages.midtrain.output_uri, "latest.pt"))
    assert latest["step"] == 2 and latest["scheduler"]["step_num"] == 2 and "rng" in latest["ranks"][0]

    sctx2 = common.make_stage_context(lc_b, "midtrain", True, st, (0, 1, 0))
    res = midtrain.train(sctx2)
    assert res.final_step == 3
    got = final_state(st, lc_b, "midtrain")["model"]
    bad = {k: float((got[k].float() - ref[k].float()).abs().max()) for k in ref if not torch.equal(got[k], ref[k])}
    assert set(got) == set(ref) and not bad, bad        # CK-5: identical state


def test_keep_last_checkpoints(tiny_root):
    lc, st = make_run(tiny_root, "midtrain", ["stages.midtrain.data.token_budget=1536",
                                              "stages.midtrain.checkpoint_every_steps=1",
                                              "stages.midtrain.keep_last_checkpoints=2"])
    prep(lc, st, "midtrain")
    midtrain.train(common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0)))
    files = st.glob(Storage.join(lc.cfg.stages.midtrain.output_uri, "checkpoints"), "step_*.pt")
    assert [os.path.basename(f) for f in files] == ["step_000005.pt", "step_000006.pt"]      # CK-6


# --------------------------------------------------------------------------- SFT
def test_sft_runs_selects_best_validation_and_masks_loss(tiny_root):
    lc, st = make_run(tiny_root, "sft", ["stages.sft.init_from=base", "stages.sft.data.epochs=3"])
    prep(lc, st, "sft")
    sctx = common.make_stage_context(lc, "sft", False, st, (0, 1, 0))
    res = sft.train(sctx)
    assert res.final_step >= 3 and res.best_step is not None and res.final_from == "best"
    ev = events(lc, "sft")
    step = [e for e in ev if e["event"] == "sft_step"]
    assert {"loss", "lr", "assistant_tokens", "tokens_per_sec", "grad_norm"} <= set(step[0])
    assert all(e["assistant_tokens"] > 0 for e in step)
    vals = [e for e in ev if e["event"] == "val"]
    best = min(vals, key=lambda e: e["val_loss"])
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.sft.output_uri, "final/lineage.json")))
    assert lin["final_step"] == best["step"] and lin["selected"] == "best"                # S2-3


def test_sft_loss_matches_hand_computation(tiny_root):
    """S2-1: the loop's reported loss for the first step equals masked CE computed by hand from the
    initial weights on the same batch (world size 1, accum 1)."""
    lc, st = make_run(tiny_root, "sft", ["stages.sft.init_from=base", "stages.sft.batch.grad_accum=1",
                                         "stages.sft.optim.lr=1e-9", "stages.sft.data.epochs=1"])
    prep(lc, st, "sft")
    sctx = common.make_stage_context(lc, "sft", False, st, (0, 1, 0))
    obj = sft.build_objective(sctx)
    model = common.load_stage_model(sctx)
    model.eval()
    from src.data import loaders
    probe = sft.BlockObjective(sctx, "sft")
    class _T:  # minimal stand-in: only setup() reads nothing from the trainer
        pass
    probe.setup(_T())
    rows = [probe.mix.next()[1] for _ in range(obj.micro)]
    b = loaders.collate_blocks(rows, sctx.cfg.tokenizer.pad_token_id, lm=False)
    with torch.no_grad():
        logits, _ = model(b["tokens"][:, :-1], attn_mask=None if not sctx.block.use_doc_mask else
                          __import__("src.foundation.bridge", fromlist=["x"]).fl_model.build_doc_attention_mask(b["doc_start"][:, :-1]))
    ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), b["tokens"][:, 1:].reshape(-1), reduction="none")
    m = b["loss_mask"][:, 1:].reshape(-1).float()
    expected = float((ce * m).sum() / m.sum())
    res = common.run_training(sctx, obj)
    first = [e for e in events(lc, "sft") if e["event"] == "sft_step"][0]
    assert first["loss"] == pytest.approx(expected, rel=1e-4)


# --------------------------------------------------------------------------- AT-21 (bias switch), PT-9
def _count_update_bias(monkeypatch):
    from src.foundation import bridge
    calls = {"n": 0}
    orig = bridge.fl_moe.DeepSeekMoE.update_bias

    def counting(self):
        calls["n"] += 1
        return orig(self)
    monkeypatch.setattr(bridge.fl_moe.DeepSeekMoE, "update_bias", counting)
    return calls


@pytest.mark.parametrize("flag,expect_calls", [(False, 0), (True, 3 * 2)])          # 3 optimizer steps x 2 MoE layers
def test_update_bias_only_when_configured(tiny_root, monkeypatch, flag, expect_calls):
    calls = _count_update_bias(monkeypatch)
    lc, st = make_run(tiny_root, "midtrain", MID + [f"stages.midtrain.moe.update_routing_bias={str(flag).lower()}"])
    prep(lc, st, "midtrain")
    midtrain.train(common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0)))
    assert calls["n"] == expect_calls


def test_moe_settings_applied_to_modules(tiny_root):
    lc, st = make_run(tiny_root, "midtrain", MID + ["stages.midtrain.moe.bias_update_rate=0.5",
                                                    "stages.midtrain.moe.aux_loss_weight=0.25", "stages.midtrain.dropout=0.3"])
    prep(lc, st, "midtrain")
    sctx = common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0))
    from src.training.loop import Trainer
    t = Trainer(sctx, common.load_stage_model(sctx), midtrain.build_objective(sctx))
    for m in common.load_stage_model.__globals__["torch"].nn.Module.modules(t.raw):
        if type(m).__name__ == "DeepSeekMoE":
            assert m.bias_update_rate == 0.5 and m.aux_loss_weight == 0.25       # PT-9
        if isinstance(m, torch.nn.Dropout):
            assert m.p == 0.3                                                     # TR-7


# --------------------------------------------------------------------------- resolution (PL-2, PL-3, PL-5)
def test_parent_resolution_and_hash(tiny_root):
    lc, st = make_run(tiny_root, "sft")
    cfg = lc.cfg
    p = resolve.resolve_parent(st, cfg, "midtrain")
    assert p.kind == "base" and p.sha256 and p.architecture["arch"] == "deepseek"
    assert resolve.resolve_parent(st, cfg, "sft").kind == "base"                 # previous with nothing complete -> base
    with pytest.raises(FileNotFoundError, match="PL-2"):
        lc2, _ = make_run(tiny_root, "sft", ["stages.sft.init_from=midtrain"])
        resolve.resolve_parent(st, lc2.cfg, "sft")
    h1 = resolve.stage_config_hash(cfg, "midtrain", p.architecture, p.sha256)
    lc3, _ = make_run(tiny_root, "midtrain", ["stages.midtrain.optim.lr=2e-3"], name=cfg.run.name)
    assert resolve.stage_config_hash(lc3.cfg, "midtrain", p.architecture, p.sha256) != h1
    lc4, _ = make_run(tiny_root, "midtrain", ["eval.benchmarks.mmlu.max_examples=1"], name=cfg.run.name)
    assert resolve.stage_config_hash(lc4.cfg, "midtrain", p.architecture, p.sha256) == h1     # eval settings do not count


def test_stage_1b_overrides_ctx_and_rope(tiny_root):
    lc, st = make_run(tiny_root, "midtrain_long", ["stages.midtrain_long.enabled=true"])
    p = resolve.resolve_parent(st, lc.cfg, "midtrain_long")
    a = resolve.stage_architecture(lc.cfg, "midtrain_long", p)
    assert a["ctx"] == 128 and a["rope_theta"] == 50000.0 and a["d_model"] == 32
    assert resolve.planned_architecture(lc.cfg, "sft")["ctx"] == 128 and resolve.planned_architecture(lc.cfg, "midtrain")["ctx"] == 64
