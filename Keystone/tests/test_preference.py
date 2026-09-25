"""Chunk 5: preference stage (DP-8 reference log-probs, PO-5 on-policy pairs, DPO/SimPO in the loop)."""
import json
import math
import os
import uuid

import pytest
import torch

from src.config.loader import load_config
from src.data import prepare
from src.io.storage import Storage
from src.stages import common, preference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def setup(tiny_root, overrides=(), stub=None):
    name = f"p{uuid.uuid4().hex[:8]}"
    ov = [f"run.name={name}", "stages.preference.init_from=base", *overrides] + (stub.overrides() if stub else [])
    lc = load_config(TINY, ov)
    st = Storage.from_config(lc.cfg)
    sctx = common.make_stage_context(lc, "preference", False, st, (0, 1, 0))
    return lc, st, sctx, common.make_prep_context(sctx, log=lambda *_: None)


def events(lc):
    p = os.path.join(lc.cfg.run.local_work_dir, lc.cfg.run.name, "preference", "metrics", "metrics.jsonl")
    return [json.loads(l) for l in open(p)]


def test_dpo_reference_logps_and_loss_is_log2_at_step_zero(tiny_root):
    lc, st, sctx, pctx = setup(tiny_root, ["stages.preference.optim.lr=1e-9"])
    preference.prepare_stage(sctx, pctx)
    rows = list(prepare.read_rows(st, prepare.prepared_uri(lc.cfg, "preference", "tiny_pref", "train")))
    assert {"ref_logp_chosen", "ref_logp_rejected"} <= set(rows[0]) and rows[0]["ref_logp_chosen"] < 0      # DP-3 schema
    marker = json.loads(st.read_text(Storage.join(prepare.prepared_uri(lc.cfg, "preference", "tiny_pref", "test"), "_ref.json")))
    assert marker["ref_sha256"] == sctx.parent.sha256                                                        # DP-8: sha recorded
    res = preference.train(sctx)
    ev = events(lc)
    first = [e for e in ev if e["event"] == "pref_step"][0]
    assert first["loss"] == pytest.approx(math.log(2), abs=1e-3)                                            # AT-15
    assert {"step", "loss", "lr", "reward_margin", "chosen_reward", "rejected_reward", "accuracy", "chosen_len",
            "rejected_len"} <= set(first)
    val = [e for e in ev if e["event"] == "val"][-1]
    assert {"val_loss", "val_accuracy"} <= set(val)                                                         # PO-6
    assert res.final_from == "last"


def test_dpo_learns_the_preference(tiny_root):
    lc, st, sctx, pctx = setup(tiny_root, ["stages.preference.data.epochs=6", "stages.preference.optim.lr=3e-4",
                                           "stages.preference.beta=0.5"])
    preference.prepare_stage(sctx, pctx)
    preference.train(sctx)
    steps = [e for e in events(lc) if e["event"] == "pref_step"]
    assert steps[-1]["loss"] < steps[0]["loss"] and steps[-1]["reward_margin"] > 0


def test_reference_pass_reruns_only_when_parent_changes(tiny_root):
    lc, st, sctx, pctx = setup(tiny_root)
    preference.prepare_stage(sctx, pctx)
    uri = prepare.prepared_uri(lc.cfg, "preference", "tiny_pref", "train")
    shard = st.glob(uri, prepare.SHARD_GLOB)[0]
    t0 = os.path.getmtime(st.local_path(shard))
    preference.prepare_stage(sctx, pctx)
    assert os.path.getmtime(st.local_path(shard)) == t0                     # unchanged parent: no rewrite
    m = Storage.join(uri, "_ref.json")
    st.write_text(m, json.dumps({"ref_sha256": "0" * 64, "autocast": str(sctx.autocast_dtype)}))
    preference.prepare_stage(sctx, pctx)
    assert os.path.getmtime(st.local_path(shard)) > t0                      # sha changed: recomputed
    assert json.loads(st.read_text(m))["ref_sha256"] == sctx.parent.sha256


def test_simpo_needs_no_reference_columns(tiny_root):
    lc, st, sctx, pctx = setup(tiny_root, ["stages.preference.objective=simpo", "stages.preference.beta=2.0"])
    preference.prepare_stage(sctx, pctx)
    rows = list(prepare.read_rows(st, prepare.prepared_uri(lc.cfg, "preference", "tiny_pref", "train")))
    assert "ref_logp_chosen" not in rows[0]
    preference.train(sctx)
    first = [e for e in events(lc) if e["event"] == "pref_step"][0]
    assert math.isfinite(first["loss"]) and "reward_margin" in first


def test_on_policy_pairs_judged_and_mixed(tiny_root, stub):
    ov = ["stages.preference.on_policy.enabled=true", "stages.preference.on_policy.max_prompts=16",
          "stages.preference.on_policy.samples_per_prompt=3"]
    lc, st, sctx, pctx = setup(tiny_root, ov, stub)
    out = preference.prepare_stage(sctx, pctx)
    e = out["on_policy"]["train"]
    assert e["rows"] > 0 and e["judge"]["model"] == "stub-judge" and e["pairs"] >= e["rows"]
    assert e["third_party_generated"] is False
    assert all(m == "stub-judge" for m, _ in stub.requests)                                    # only the judge was called
    n_req = len(stub.requests)
    rows = list(prepare.read_rows(st, prepare.prepared_uri(lc.cfg, "preference", "on_policy", "train")))
    assert "ref_logp_chosen" in rows[0] and rows[0]["chosen_tokens"][-1] == 100265            # DP-7 im_end, DP-8 for on_policy
    preference.prepare_stage(sctx, pctx)                                                       # same inputs: nothing regenerated
    assert len(stub.requests) == n_req
    obj = preference.build_objective(sctx)
    assert obj.names == ["tiny_pref", "on_policy"] and obj.weights[1] == pytest.approx(0.3 / 1.3)     # PO-5 weight
    assert obj.val_names == ["tiny_pref"]
    preference.train(sctx)
    lin = json.loads(st.read_text(Storage.join(lc.cfg.stages.preference.output_uri, "final/lineage.json")))
    assert lin["services"]["judge"]["model"] == "stub-judge"
    assert {d["name"] for d in lin["data"]} == {"tiny_pref", "on_policy"}


def test_on_policy_skips_equal_scores(tiny_root, stub):
    class ConstJudge:
        def map(self, fn, items):
            return [fn(i) for i in items]

        def score(self, instr, resp):
            return 5

        def identity(self):
            return {"model": "const"}
    lc, st, sctx, pctx = setup(tiny_root, ["stages.preference.on_policy.enabled=true"], stub)
    model = common.load_stage_model(sctx)
    pairs, meta = preference.generate_on_policy_pairs(sctx, pctx, model, ConstJudge())
    assert pairs == [] and meta["skipped_equal_scores"] == meta["prompts"]                    # PO-5
