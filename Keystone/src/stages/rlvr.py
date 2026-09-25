"""Stage 4: RLVR with GSPO and DAPO techniques (RL-1..RL-11, RW-*, PT-11).

Per rollout step (one loop step): sample groups (dynamic sampling), balance group counts across
ranks (RL-3a), advantages (RL-4), recompute logp_old with the training forward (RL-6), then
`updates_per_step` mini-batches with one optimizer step each (RL-7, RL-7a), optional KL to the
parent (RL-8), entropy monitoring (RL-10), routing-change probe (PT-11).

Conventions the PRD leaves open are in docs/implementation_notes.md: the policy runs in eval mode
(dropout off) so the importance ratio is not disturbed; RW-5 `exclude` zeroes a response's weight
(so every rank keeps the same micro-batch count); skipped steps (no group kept anywhere) do not
advance the LR schedule; on an entropy-floor stop the output is the best validation checkpoint from
before the collapse window.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..data import prepare
from ..foundation import bridge
from ..io.storage import Storage
from ..modeling.inference import InferenceModel
from ..rl import rollouts as ro
from ..rl.rewards import RewardBreakdown, correctness, score_response
from ..rl.sandbox import Sandbox
from ..training import checkpoint as ck
from ..training.loop import Objective, Trainer, TrainResult
from ..training.objectives.common import token_logps
from ..training.objectives.gspo import group_advantages, gspo_per_sequence
from .common import StageContext, load_stage_model, run_training

STAGE = "rlvr"


class EntryGateFailure(RuntimeError):
    """RL-1 with entry_gate.on_failure: stop."""


@dataclass
class Group:
    rollouts: List[ro.Rollout]
    breakdowns: List[RewardBreakdown]


def build_sandbox(cfg, block) -> Optional[Sandbox]:
    kinds = {cfg.datasets[s.dataset].kind for s in block.data.sources}
    return Sandbox.from_config(cfg.code_execution) if "rl_code" in kinds else None


def stop_ids(sctx: StageContext) -> List[int]:
    return [sctx.cfg.tokenizer.eot_token_id, sctx.tok.im_end_id]           # GN-1, TK-7


def _reduce(t: torch.Tensor, op=None) -> torch.Tensor:
    if bridge.fl_distributed.is_distributed():
        dist.all_reduce(t, op=op if op is not None else dist.ReduceOp.SUM)
    return t


# --------------------------------------------------------------------------- entry gate (RL-1)
def entry_gate(sctx: StageContext, model, sandbox) -> Dict[str, Any]:
    cfg, blk, s = sctx.cfg, sctx.block, sctx
    pool = ro.PromptPool(s.storage, cfg, STAGE, [(x.dataset, x.weight) for x in blk.data.sources], "train")
    items = pool.draw(random.Random(cfg.run.seed), blk.entry_gate.sample_prompts)[s.rank::s.world]
    hits = 0
    if items:
        groups = ro.generate_rollouts(InferenceModel(model), s.tok, items, blk.entry_gate.k, blk.generation, stop_ids(s),
                                      s.autocast_dtype, return_logprobs=False)
        rx = blk.rewards.math.final_answer_regex
        for g in groups:
            if any(correctness(r.item, r.text, s.template, rx, sandbox) for r in g):
                hits += 1
    t = _reduce(torch.tensor([float(hits), float(len(items))], dtype=torch.float64, device=s.device))
    n = max(float(t[1]), 1.0)
    return {"pass_at_k": float(t[0]) / n, "k": blk.entry_gate.k, "prompts": int(t[1]),
            "min_pass_at_k": blk.entry_gate.min_pass_at_k}


# --------------------------------------------------------------------------- objective
class RLVRObjective(Objective):
    event = "rl_step"
    best_mode = "max"
    best_key = "val_accuracy"

    def __init__(self, sctx: StageContext, sandbox: Optional[Sandbox] = None):
        self.s, self.sandbox = sctx, sandbox
        blk = sctx.block
        self.total_steps = blk.total_steps
        self.total_updates = blk.total_steps * blk.updates_per_step
        self.low = 0
        self.window_start: Optional[int] = None
        self._stop: Optional[str] = None
        self.rng = random.Random(sctx.cfg.run.seed + sctx.rank)
        self.probe_rng = random.Random(sctx.cfg.run.seed + sctx.rank + 1_000_003)
        self.ref_forward = None
        self.sources = [(x.dataset, x.weight) for x in blk.data.sources]

    # ------------------------------------------------------------------ setup / state
    def setup(self, trainer: Trainer) -> None:
        s, blk, cfg = self.s, self.s.block, self.s.cfg
        trainer.wrapped.eval()                                   # see module docstring: no dropout in RL
        self.infer = InferenceModel(trainer.raw)
        self.pool = ro.PromptPool(s.storage, cfg, STAGE, self.sources, "train")
        try:
            self.val_pool = ro.PromptPool(s.storage, cfg, STAGE, self.sources, "test")
        except FileNotFoundError:
            self.val_pool = None
        self.probe = ro.RoutingProbe(trainer.raw, blk.routing_metric_sample_tokens, self.probe_rng)
        if blk.kl_coef > 0:                                      # RL-8: frozen copy of the parent
            ref = load_stage_model(s)
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            self.ref_model = ref

            def ref_forward(idx):
                with trainer.autocast():
                    return self.ref_model(idx.to(s.device))[0]
            self.ref_forward = ref_forward

    def state_dict(self) -> Dict[str, Any]:
        return {"rng": ck.python_rng_state(self.rng), "probe_rng": ck.python_rng_state(self.probe_rng),
                "low": self.low, "window_start": self.window_start}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        ck.set_python_rng_state(self.rng, d["rng"])
        ck.set_python_rng_state(self.probe_rng, d["probe_rng"])
        self.low, self.window_start = d["low"], d["window_start"]

    # ------------------------------------------------------------------ rewards / sampling (RL-2, RL-3, RW-*)
    def _score(self, rollouts: List[ro.Rollout]) -> List[RewardBreakdown]:
        blk, s = self.s.block, self.s
        workers = s.cfg.code_execution.max_parallel if self.sandbox is not None else 1
        return ro.parallel_map(
            lambda r: score_response(r.item, r.text, r.finish, len(r.completion), s.template, blk.rewards,
                                     blk.dapo.overlong, blk.generation.max_new_tokens, self.sandbox), rollouts, workers)

    def _generate(self, items, step: int, rnd: int):
        s, blk = self.s, self.s.block
        gcfg_seed = blk.generation.seed + 7919 * step + 104729 * s.rank + rnd
        gen = blk.generation.model_copy(update={"seed": gcfg_seed})
        return ro.generate_rollouts(self.infer, s.tok, items, blk.group_size, gen, stop_ids(s), s.autocast_dtype)

    def _sample_groups(self, trainer: Trainer, step: int) -> Tuple[List[Group], Dict[str, Any]]:
        blk, s = self.s.block, self.s
        need = blk.prompts_per_step // s.world
        kept: List[Group] = []
        every: List[Tuple[ro.Rollout, RewardBreakdown]] = []
        generated, rounds, n_draw = 0, 0, need
        while True:
            items = self.pool.draw(self.rng, n_draw)
            groups = self._generate(items, step, rounds)
            flat = [r for g in groups for r in g]
            bds = self._score(flat)
            G = blk.group_size
            for gi, g in enumerate(groups):
                gb = bds[gi * G:(gi + 1) * G]
                generated += 1
                every.extend(zip(g, gb))
                rewards = [b.reward for b in gb]
                if not blk.dapo.dynamic_sampling or max(rewards) != min(rewards):        # RL-3
                    kept.append(Group(g, gb))
            if not blk.dapo.dynamic_sampling or len(kept) >= need or rounds >= blk.dapo.max_resample_rounds:
                break
            rounds += 1
            n_draw = need - len(kept)
        return kept[:need], {"generated_groups": generated, "kept_groups": min(len(kept), need), "rounds": rounds,
                             "every": every}

    # ------------------------------------------------------------------ one rollout step
    def train_step(self, trainer: Trainer, step: int) -> Dict[str, float]:
        s, blk, dev = self.s, self.s.block, trainer.device
        pad = s.cfg.tokenizer.pad_token_id
        groups, info = self._sample_groups(trainer, step)
        every = info["every"]
        # ---- step statistics over everything generated this step (reduced over ranks)
        rew = torch.tensor([b.reward for _, b in every], dtype=torch.float64)
        lens = torch.tensor([len(r.completion) for r, _ in every], dtype=torch.float64)
        stat = torch.tensor([len(every), float(rew.sum()), float((rew ** 2).sum()),
                             float(sum(b.correct for _, b in every)), float(sum(b.format_ok for _, b in every)),
                             float(lens.sum()), float(sum(r.finish == "length" for r, _ in every)),
                             info["generated_groups"], info["kept_groups"]], dtype=torch.float64, device=dev)
        stat = _reduce(stat)
        mx = _reduce(torch.tensor([float(lens.max()) if len(every) else 0.0], dtype=torch.float64, device=dev),
                     dist.ReduceOp.MAX if bridge.fl_distributed.is_distributed() else None)
        n = max(float(stat[0]), 1.0)
        out: Dict[str, float] = {
            "reward_mean": float(stat[1]) / n, "reward_std": math.sqrt(max(float(stat[2]) / n - (float(stat[1]) / n) ** 2, 0.0)),
            "correctness_rate": float(stat[3]) / n, "format_rate": float(stat[4]) / n,
            "groups_kept_fraction": float(stat[8]) / max(float(stat[7]), 1.0), "resample_rounds": float(info["rounds"]),
            "response_len_mean": float(stat[5]) / n, "response_len_max": float(mx[0]),
            "truncated_fraction": float(stat[6]) / n, "entropy": float("nan"), "clip_fraction": 0.0,
            "logprob_mismatch": 0.0, "routing_change": 0.0, "skipped_update": 0.0,
        }
        # ---- RL-3a: every rank trains on the same number of groups
        k = int(_reduce(torch.tensor(len(groups), device=dev),
                        dist.ReduceOp.MIN if bridge.fl_distributed.is_distributed() else None).item())
        if k == 0:
            out["skipped_update"] = 1.0
            print(f"[rlvr] step {step + 1}: no group kept on any rank; update skipped") if s.rank == 0 else None
            return out
        groups = groups[:k]
        rolls = [r for g in groups for r in g.rollouts]
        bds = [b for g in groups for b in g.breakdowns]
        prompts, resp = [r.prompt for r in rolls], [r.completion for r in rolls]
        # ---- RL-4
        adv = group_advantages(torch.tensor([b.reward for b in bds]), blk.group_size, blk.advantage_std_normalization, blk.adv_eps)
        valid = torch.tensor([0.0 if b.excluded else 1.0 for b in bds])
        # ---- RL-6: logp_old with the training forward; entropy; mismatch against generation log-probs
        fwd = lambda idx: trainer.forward(idx, None, track_usage=False)[0]          # noqa: E731
        old, ent, old_masked = ro.response_token_logps(fwd, prompts, resp, blk.batch.micro_batch_size, pad, dev,
                                                       entropy=True, allowed=s.tok.allowed_mask(dev))
        ref = ro.response_token_logps(self.ref_forward, prompts, resp, blk.batch.micro_batch_size, pad, dev) \
            if self.ref_forward is not None else None
        mm = torch.tensor([sum(float((o - torch.tensor(r.logprobs)).abs().sum()) for o, r in zip(old_masked, rolls)),
                           float(sum(len(r.completion) for r in rolls)),
                           float(sum(float(e.sum()) for e in ent))], dtype=torch.float64, device=dev)
        mm = _reduce(mm)
        out["logprob_mismatch"] = float(mm[0]) / max(float(mm[1]), 1.0)
        out["entropy"] = float(mm[2]) / max(float(mm[1]), 1.0)
        # ---- PT-11 probe before updates
        self.probe.choose(prompts, resp)
        before = self.probe.measure(fwd, prompts, resp, blk.batch.micro_batch_size, pad, dev)
        # ---- RL-7: mini-batches
        N = len(rolls)
        bounds = [round(i * N / blk.updates_per_step) for i in range(blk.updates_per_step + 1)]
        clip_num = clip_den = 0.0
        gnorms: List[float] = []
        micro = blk.batch.micro_batch_size
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo:
                continue
            n_valid = max(float(valid[lo:hi].sum()), 1.0)
            for m0 in range(lo, hi, micro):
                m1 = min(m0 + micro, hi)
                ps, rs = prompts[m0:m1], resp[m0:m1]
                idx, mask = ro.loaders.response_batch(ps, rs, pad)
                with trainer.accumulate(m1 == hi):
                    logits, aux = trainer.forward(idx, None, track_usage=False)
                    new = token_logps(logits[:, :-1], idx[:, 1:].to(dev))
                    T = new.shape[1]
                    old_t = ro.aligned(old[m0:m1], ps, T).to(dev)
                    ref_t = ro.aligned(ref[m0:m1], ps, T).to(dev) if ref is not None else None
                    per, clipped = gspo_per_sequence(new, old_t, mask.to(dev), adv[m0:m1].to(dev), blk.clip.eps_low,
                                                     blk.clip.eps_high, ref_t, blk.kl_coef)
                    v = valid[m0:m1].to(dev)
                    loss = (per * v).sum() / n_valid
                    if aux is not None:
                        loss = loss + aux                                            # TR-2 (weight 0 unless configured)
                    trainer.backward(loss)
                clip_num += float((clipped.float() * v).sum())
                clip_den += float(v.sum())
            gnorms.append(trainer.apply_update())
        out["grad_norm"] = sum(gnorms) / max(len(gnorms), 1)                      # mean over the step's mini-batch updates
        cd = _reduce(torch.tensor([clip_num, clip_den], dtype=torch.float64, device=dev))
        out["clip_fraction"] = float(cd[0]) / max(float(cd[1]), 1.0)
        # ---- PT-11 probe after updates
        if self.probe.active:
            after = self.probe.measure(fwd, prompts, resp, blk.batch.micro_batch_size, pad, dev)
            d, t = ro.RoutingProbe.changed(before, after)
            dt = _reduce(torch.tensor([d, t], dtype=torch.float64, device=dev))
            out["routing_change"] = float(dt[0]) / max(float(dt[1]), 1.0)
        # ---- RL-10 bookkeeping (entropy is identical on every rank: it was all-reduced)
        floor = blk.entropy_floor
        if out["entropy"] < floor.value:
            self.low += 1
            if self.low == 1:
                self.window_start = step
            if self.low >= floor.patience_steps:
                self._stop = (f"entropy {out['entropy']:.4f} below floor {floor.value} for {self.low} consecutive steps "
                              f"(window began after step {self.window_start})")
        else:
            self.low, self.window_start = 0, None
        return out

    # ------------------------------------------------------------------ RL-10 / RL-11
    def stop_reason(self) -> Optional[str]:
        return self._stop

    def improved(self) -> bool:
        return self.low == 0                           # no best-checkpoint updates inside a collapse window

    def final_override(self, trainer: Trainer) -> Optional[Dict[str, Any]]:
        if self._stop is None or trainer.best_step is not None:
            return None
        s = self.s
        files = sorted(s.storage.glob(Storage.join(s.out_uri, "checkpoints"), "step_*.pt"))
        ok = [f for f in files if int(f.rsplit("step_", 1)[1].split(".")[0]) <= (self.window_start or 0)]
        if ok:
            p = ck.load_payload(s.storage, ok[-1])
            return {"state": p["model"], "step": int(p["step"]), "selected": "pre_collapse_checkpoint"}
        p = ck.load_payload(s.storage, s.parent.uri)
        return {"state": ck.strip_prefixes(p["model"]), "step": 0, "selected": "parent"}

    @torch.no_grad()
    def validate(self, trainer: Trainer, step: int) -> Dict[str, float]:
        s, blk = self.s, self.s.block
        acc = [0.0, 0.0]
        if self.val_pool is not None:
            items = self.val_pool.fixed(blk.validation_prompts, s.rank, s.world)
            if items:
                greedy = blk.generation.model_copy(update={"temperature": 0.0})
                groups = ro.generate_rollouts(self.infer, s.tok, items, 1, greedy, stop_ids(s), s.autocast_dtype,
                                              return_logprobs=False)
                bds = self._score([g[0] for g in groups])
                acc = [float(sum(b.correct for b in bds)), float(len(bds))]
        t = _reduce(torch.tensor(acc, dtype=torch.float64, device=trainer.device))
        a = float(t[0]) / max(float(t[1]), 1.0)
        return {"val_accuracy": a, "val_loss": 1.0 - a, "val_prompts": float(t[1])}

    def data_manifest(self) -> List[Dict[str, Any]]:
        return [{"dataset": n, "weight": w} for n, w in self.sources]


def train(sctx: StageContext) -> TrainResult:
    cfg, blk = sctx.cfg, sctx.block
    model = load_stage_model(sctx)
    sandbox = build_sandbox(cfg, blk)
    gate = entry_gate(sctx, model, sandbox)
    if sctx.rank == 0:
        print(f"[rlvr] entry gate: pass@{gate['k']} = {gate['pass_at_k']:.3f} on {gate['prompts']} prompts "
              f"(minimum {gate['min_pass_at_k']})")
    if gate["pass_at_k"] < blk.entry_gate.min_pass_at_k:
        if blk.entry_gate.on_failure == "stop":
            raise EntryGateFailure(f"RL-1: pass@{gate['k']} {gate['pass_at_k']:.3f} < {blk.entry_gate.min_pass_at_k}")
        if sctx.rank == 0:
            sctx.storage.write_text(Storage.join(sctx.out_uri, "_SKIPPED"),
                                    json.dumps({"reason": "entry_gate", **gate}, sort_keys=True))
        if bridge.fl_distributed.is_distributed():
            dist.barrier()
        return TrainResult(0, "entry_gate_skip", None, None)
    return run_training(sctx, RLVRObjective(sctx, sandbox), model)


def prepare_stage(sctx: StageContext, pctx: prepare.PrepContext):
    return prepare.prepare_stage_datasets(pctx, STAGE, sctx.ctx_len)
