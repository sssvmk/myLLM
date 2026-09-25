"""Stage 5: distillation.

5a (`distill_offpolicy`): request teacher responses (DI-1), keep the usable ones (DI-2), register
them as the produced dataset `teacher_traces` (third_party_generated), then train with the Stage 2
objective and loaders (DI-3, see sft.py).

5b (`distill_onpolicy`): the student samples a response per prompt; per-token reverse KL to a
same-tokenizer teacher checkpoint in REINFORCE form (DI-4, DI-5, TC-2).
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..config.loader import sha256_json
from ..data import prepare
from ..data.adapters import Conversation, PromptItem, RLCodeItem, RLMathItem
from ..foundation import bridge
from ..io.storage import Storage
from ..modeling.inference import InferenceModel
from ..modeling.loading import is_base_checkpoint, load_model, load_state, validate_base_args
from ..rl import rollouts as ro
from ..rl.rewards import correctness
from ..training import checkpoint as ck
from ..training.loop import Objective, Trainer, TrainResult
from ..training.objectives.common import token_logps
from ..training.objectives.opd import opd_loss
from . import sft
from .common import StageContext, load_stage_model, run_training
from .rlvr import _reduce, build_sandbox, stop_ids

OFF, ON = "distill_offpolicy", "distill_onpolicy"


# --------------------------------------------------------------------------- 5a: teacher traces
def _prompt_items(pctx: prepare.PrepContext, names: List[str], max_prompts: int):
    """Round-robin over the prompt datasets until `max_prompts` (decontaminated, DP-4)."""
    counter: Dict[str, int] = {}
    streams = [iter(pctx.decontaminate(pctx.registry.items(n, "train"), counter)) for n in names]
    out, live = [], list(range(len(streams)))
    while live and len(out) < max_prompts:
        for i in list(live):
            try:
                out.append((names[i], next(streams[i])))
            except StopIteration:
                live.remove(i)
            if len(out) >= max_prompts:
                break
    return out, counter


def _split_holdout(convs: List[Conversation], seed: int, fraction: float) -> Tuple[List[Conversation], List[Conversation]]:
    """Deterministic train/test split by hashing each conversation with the run seed (DS-4 style);
    at least one conversation goes to the test split when there are two or more."""
    tr, te = [], []
    for c in convs:
        h = hashlib.sha256(f"{seed}|{json.dumps(c.messages, sort_keys=True)}".encode()).digest()
        (te if int.from_bytes(h[:8], "big") / 2 ** 64 < fraction else tr).append(c)
    if not te and len(tr) >= 2:
        te.append(tr.pop())
    return tr, te


def prepare_teacher_traces(sctx: StageContext, pctx: prepare.PrepContext, teacher, sandbox) -> Dict[str, Any]:
    cfg, blk = sctx.cfg, sctx.block
    names = list(blk.prompts_datasets)
    train_uri = prepare.prepared_uri(cfg, OFF, "teacher_traces", "train")
    h = sha256_json({"teacher": teacher.identity(), "datasets": {n: [pctx.registry.file_manifest(n, "train"), cfg.datasets[n].model_dump()] for n in names},
                     "max_prompts": blk.max_prompts, "samples": blk.samples_per_prompt, "max_resp": blk.max_response_tokens,
                     "tok": cfg.tokenizer.model_dump(), "tmpl": cfg.chat_template.model_dump(), "seed": cfg.run.seed,
                     "holdout": cfg.prepared_data.holdout_fraction, "ctx": sctx.ctx_len,
                     "system": [pctx.storage.read_text(getattr(cfg.stages.rlvr.system_prompts, k)) for k in ("rl_math", "rl_code")],
                     "rx": cfg.stages.rlvr.rewards.math.final_answer_regex})
    cur = prepare.read_marker(sctx.storage, train_uri)
    if cur is not None and cur.get("input_hash") == h:
        found = {s: prepare.read_marker(sctx.storage, prepare.prepared_uri(cfg, OFF, "teacher_traces", s))
                 for s in ("train", "test")}
        return {s: m["entry"] for s, m in found.items() if m is not None}
    picked, counter = _prompt_items(pctx, names, blk.max_prompts)
    prompts, refs = [], []
    for name, it in picked:
        kind = cfg.datasets[name].kind
        prefix = [{"role": "system", "content": prepare.read_system_prompt(pctx, kind)}] if kind in ("rl_math", "rl_code") else []
        prompts.append(prefix + it.messages)
        if isinstance(it, RLMathItem):
            refs.append({"kind": "rl_math", "ground_truth": it.ground_truth})
        elif isinstance(it, RLCodeItem):
            refs.append({"kind": "rl_code", "tests": it.tests, "entry_point": it.entry_point})
        else:
            refs.append(None)
    answers = teacher.complete(prompts, blk.samples_per_prompt)
    rx = cfg.stages.rlvr.rewards.math.final_answer_regex
    convs: List[Conversation] = []
    dropped: Dict[str, int] = {}

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1
    for msgs, ref, outs in zip(prompts, refs, answers):
        for text in outs:
            if text is None or not text.strip():
                drop("empty_or_failed")
            elif len(sctx.template.render_response(text)) > blk.max_response_tokens:
                drop("longer_than_max_response_tokens")
            elif ref is not None and not correctness(ref, text, sctx.template, rx, sandbox):
                drop("incorrect")                                   # DI-2
            else:
                convs.append(Conversation(msgs + [{"role": "assistant", "content": text}]))
    if not convs:
        raise ValueError("teacher produced no usable traces (all empty, too long or incorrect)")
    tr, te = _split_holdout(convs, cfg.run.seed, cfg.prepared_data.holdout_fraction)
    meta = {"teacher": teacher.identity(), "prompts": len(prompts), "responses_requested": len(prompts) * blk.samples_per_prompt,
            "responses_kept": len(convs), "dropped_teacher_traces": dropped, "decontamination_prompts": counter}
    out = {}
    for split, cs in (("train", tr), ("test", te)):
        if cs:
            e = prepare.prepare_produced_sft(pctx, OFF, "teacher_traces", split, cs, sctx.ctx_len, True, meta, input_hash=h)
            if e is not None:
                out[split] = e
    prepare.merge_stage_manifest(pctx, OFF, {"teacher_traces": out})
    return out


def prepare_offpolicy(sctx: StageContext, pctx: prepare.PrepContext, teacher=None):
    out = prepare.prepare_stage_datasets(pctx, OFF, sctx.ctx_len)
    if teacher is None:
        from ..services.teacher import Teacher
        teacher = Teacher.from_config(sctx.cfg)
    out["teacher_traces"] = prepare_teacher_traces(sctx, pctx, teacher, build_sandbox_for_traces(sctx))
    return out


def build_sandbox_for_traces(sctx: StageContext):
    from ..rl.sandbox import Sandbox
    cfg = sctx.cfg
    if any(cfg.datasets[n].kind == "rl_code" for n in sctx.block.prompts_datasets):
        return Sandbox.from_config(cfg.code_execution)
    return None


def train_offpolicy(sctx: StageContext) -> TrainResult:
    return sft.train(sctx)


# --------------------------------------------------------------------------- 5b: on-policy
def teacher_tokenizer_errors(cfg, storage) -> List[str]:
    """TC-2: the teacher must share ranks sha256 and chat special tokens. A lineage record next to
    the checkpoint is compared; a base-format teacher needs `tokenizer_matches: true`."""
    t = cfg.teachers.onpolicy
    sibling = t.checkpoint_uri.rsplit("/", 1)[0] + "/lineage.json"
    if storage.exists(sibling):
        rec = json.loads(storage.read_text(sibling)).get("tokenizer", {})
        mine = {"ranks_sha256": cfg.tokenizer.ranks_sha256,
                "chat_special_tokens": {k: v.model_dump() for k, v in cfg.tokenizer.chat_special_tokens.items()}}
        errs = []
        if rec.get("ranks_sha256") != mine["ranks_sha256"]:
            errs.append("teacher lineage ranks_sha256 differs from tokenizer.ranks_sha256")
        if rec.get("chat_special_tokens") != mine["chat_special_tokens"]:
            errs.append("teacher lineage chat_special_tokens differ from tokenizer.chat_special_tokens")
        return errs
    return [] if t.tokenizer_matches else ["teachers.onpolicy.tokenizer_matches must be true for a checkpoint without lineage (TC-2)"]


class OPDObjective(Objective):
    event = "distill_step"
    best_mode = "min"
    best_key = "val_loss"

    def __init__(self, sctx: StageContext):
        self.s = sctx
        blk = sctx.block
        self.total_steps = blk.total_steps
        self.total_updates = blk.total_steps
        self.sources = [(x.dataset, x.weight) for x in blk.data.sources]
        self.rng = random.Random(sctx.cfg.run.seed + sctx.rank)
        errs = teacher_tokenizer_errors(sctx.cfg, sctx.storage)
        if errs:
            raise ValueError("; ".join(errs))

    def setup(self, trainer: Trainer) -> None:
        s, cfg = self.s, self.s.cfg
        trainer.wrapped.eval()
        self.infer = InferenceModel(trainer.raw)
        self.pool = ro.PromptPool(s.storage, cfg, ON, self.sources, "train")
        try:
            self.val_pool = ro.PromptPool(s.storage, cfg, ON, self.sources, "test")
        except FileNotFoundError:
            self.val_pool = None
        t = cfg.teachers.onpolicy
        ckpt = load_state(s.storage.cached_local_path(t.checkpoint_uri))
        if is_base_checkpoint(ckpt):
            validate_base_args(ckpt["args"], t.architecture)
        self.teacher = load_model(ckpt, t.architecture, s.device, validate_base=False)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if t.architecture.vocab_rows != s.architecture["vocab_rows"]:
            raise ValueError("teacher vocab_rows differ from the student's (TC-2)")

        def teacher_forward(idx):
            with trainer.autocast():
                return self.teacher(idx.to(s.device))[0]
        self.teacher_forward = teacher_forward

    def state_dict(self):
        return {"rng": ck.python_rng_state(self.rng)}

    def load_state_dict(self, d):
        ck.set_python_rng_state(self.rng, d["rng"])

    def _rollouts(self, items, seed: int):
        s, blk = self.s, self.s.block
        gen = blk.generation.model_copy(update={"seed": seed})
        return [g[0] for g in ro.generate_rollouts(self.infer, s.tok, items, 1, gen, stop_ids(s), s.autocast_dtype,
                                                   return_logprobs=False)]

    def train_step(self, trainer: Trainer, step: int) -> Dict[str, float]:
        s, blk, dev = self.s, self.s.block, trainer.device
        pad = s.cfg.tokenizer.pad_token_id
        need = blk.prompts_per_step // s.world                                           # RL-3a: equal prompts per rank
        items = self.pool.draw(self.rng, need)
        rolls = self._rollouts(items, blk.generation.seed + 7919 * step + 104729 * s.rank)
        prompts, resp = [r.prompt for r in rolls], [r.completion for r in rolls]
        fwd = lambda idx: trainer.forward(idx, None, track_usage=False)[0]               # noqa: E731
        micro = blk.batch.micro_batch_size
        lp_t = ro.response_token_logps(self.teacher_forward, prompts, resp, micro, pad, dev)
        total_tokens = float(sum(len(r) for r in resp))
        kl_sum = 0.0
        N = len(rolls)
        for m0 in range(0, N, micro):                                                    # DI-4a: one mini-batch, RL-7a
            m1 = min(m0 + micro, N)
            ps, rs = prompts[m0:m1], resp[m0:m1]
            idx, mask = ro.loaders.response_batch(ps, rs, pad)
            with trainer.accumulate(m1 == N):
                logits, aux = trainer.forward(idx, None)
                new = token_logps(logits[:, :-1], idx[:, 1:].to(dev))
                t = ro.aligned(lp_t[m0:m1], ps, new.shape[1]).to(dev)
                m = mask.to(dev)
                loss, mets = opd_loss(new, t, m)
                n_tok = float(m.sum())
                loss = loss * (n_tok / max(total_tokens, 1.0))                            # mean over the whole mini-batch
                if aux is not None:
                    loss = loss + aux
                trainer.backward(loss)
            kl_sum += mets["reverse_kl_per_token"] * n_tok
        gn = trainer.apply_update()
        red = _reduce(torch.tensor([kl_sum, total_tokens, float(N)], dtype=torch.float64, device=dev))
        return {"reverse_kl_per_token": float(red[0]) / max(float(red[1]), 1.0),
                "response_len_mean": float(red[1]) / max(float(red[2]), 1.0), "grad_norm": gn}

    @torch.no_grad()
    def validate(self, trainer: Trainer, step: int) -> Dict[str, float]:
        """DI-7: mean reverse KL (student minus teacher log-prob on student samples) on held-out prompts."""
        s, blk, dev = self.s, self.s.block, trainer.device
        pad = s.cfg.tokenizer.pad_token_id
        acc = [0.0, 0.0]
        if self.val_pool is not None:
            items = self.val_pool.fixed(blk.validation_prompts, s.rank, s.world)
            if items:
                rolls = self._rollouts(items, blk.generation.seed)
                prompts, resp = [r.prompt for r in rolls], [r.completion for r in rolls]
                fwd = lambda idx: trainer.forward(idx, None, track_usage=False)[0]       # noqa: E731
                ls = ro.response_token_logps(fwd, prompts, resp, blk.batch.micro_batch_size, pad, dev)
                lt = ro.response_token_logps(self.teacher_forward, prompts, resp, blk.batch.micro_batch_size, pad, dev)
                acc = [float(sum(float((a - b).sum()) for a, b in zip(ls, lt))), float(sum(len(r) for r in resp))]
        t = _reduce(torch.tensor(acc, dtype=torch.float64, device=dev))
        return {"val_loss": float(t[0]) / max(float(t[1]), 1.0), "val_tokens": float(t[1])}

    def data_manifest(self):
        return [{"dataset": n, "weight": w} for n, w in self.sources]

    def services(self):
        t = self.s.cfg.teachers.onpolicy
        return {"onpolicy_teacher": {"checkpoint_uri": t.checkpoint_uri, "architecture": t.architecture.model_dump()}}


def train_onpolicy(sctx: StageContext) -> TrainResult:
    return run_training(sctx, OPDObjective(sctx))


def prepare_onpolicy(sctx: StageContext, pctx: prepare.PrepContext):
    return prepare.prepare_stage_datasets(pctx, ON, sctx.ctx_len)
