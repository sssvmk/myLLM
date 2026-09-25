"""Evaluation suite: one runner per benchmark of §11.2, mode selection, the report (EV-1..EV-6).

Modes (EV-2): checkpoints from before Stage 2 (the base checkpoint and Stages 1/1b) are evaluated in
`base` mode (plain text, few-shot); Stage 2 and later in `chat` mode (chat template, zero-shot).
Benchmarks the PRD marks chat-only are skipped in base mode and listed under `skipped`. B-9 (length)
is treated as chat-only as well: in base mode greedy decoding runs to `max_new_tokens`, which would
make the length-ratio gate meaningless (docs/prd_review.md #11).

Everything runs single-process on the unwrapped model with the `generation` settings unless a
benchmark overrides them (EV-4).
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..config.loader import sha256_json
from ..data.registry import DatasetRegistry
from ..foundation import bridge
from ..generation.api import GenerationConfig, generate
from ..io.storage import Storage
from ..modeling.inference import InferenceModel
from ..modeling.loading import load_model, load_state
from ..rl.rewards.code import build_program, last_code_block
from ..rl.rewards.math import extract_answer, math_equal
from ..tokenization.chat_template import ChatTemplate
from ..tokenization.tokenizer import ChatTokenizer
from ..training.objectives.common import token_logps
from .benchmarks.alpaca_lc import lc_win_rate
from .benchmarks.calibration import choice_confidence, ece
from .benchmarks.gsm8k import gsm8k_correct
from .benchmarks.humaneval import mean_pass_at_k

BASE_STAGES = {None, "midtrain", "midtrain_long"}
CHAT_ONLY = ("ifeval", "alpaca_eval_lc", "safety", "length", "adherence")


def eval_mode(stage: Optional[str]) -> str:
    return "base" if stage in BASE_STAGES else "chat"


@dataclass
class EvalContext:
    cfg: Any
    storage: Storage
    tok: ChatTokenizer
    template: ChatTemplate
    model: torch.nn.Module
    device: torch.device
    autocast_dtype: Optional[torch.dtype]
    mode: str
    stage: Optional[str]
    registry: DatasetRegistry
    judge: Any = None
    sandbox: Any = None
    _infer: Optional[InferenceModel] = field(default=None, repr=False)

    @property
    def infer(self) -> InferenceModel:
        if self._infer is None:
            self._infer = InferenceModel(self.model)
        return self._infer

    def get_judge(self):
        if self.judge is None:
            from ..services.judge import Judge
            self.judge = Judge.from_config(self.cfg, self.storage)
        return self.judge

    def get_sandbox(self):
        if self.sandbox is None:
            from ..rl.sandbox import Sandbox
            self.sandbox = Sandbox.from_config(self.cfg.code_execution)
        return self.sandbox

    # ------------------------------------------------------------------ helpers
    def prompt_tokens(self, text: str) -> List[int]:
        if self.mode == "chat":
            return self.template.render_prompt([{"role": "user", "content": text}])
        return self.tok.encode_ordinary(text)

    def stop_ids(self) -> List[int]:
        ids = [self.cfg.tokenizer.eot_token_id]
        return ids + [self.tok.im_end_id] if self.mode == "chat" else ids

    def generate(self, prompts: List[List[int]], num_samples: int = 1, greedy: bool = True, stop_strings=(), **over):
        if greedy:
            over.setdefault("temperature", 0.0)
        gcfg = GenerationConfig.from_block(self.cfg.generation, num_samples=num_samples, stop_token_ids=self.stop_ids(),
                                           stop_strings=stop_strings, **over)
        return generate(self.infer, self.tok, prompts, gcfg, autocast_dtype=self.autocast_dtype)

    def items(self, dataset: str, split: str, limit: Optional[int]):
        out = []
        for it in self.registry.items(dataset, split):
            out.append(it)
            if limit is not None and len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- B-1 / B-8
@torch.no_grad()
def choice_logliks(ec: EvalContext, question: str, choices: Sequence[str]) -> List[float]:
    """Sum of log p(token | prefix) over `question\\nAnswer: {choice}` (all tokens after the first).
    The shared question prefix contributes the same constant to every choice, so the argmax and the
    softmax over choices are those of the choice continuation."""
    ctx = ec.model.ctx
    seqs = [ec.tok.encode_ordinary(f"{question}\nAnswer: {c}")[-ctx:] for c in choices]
    L = max(len(s) for s in seqs)
    pad = ec.cfg.tokenizer.pad_token_id
    idx = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), L - 1, dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s) - 1] = True
    with torch.autocast(device_type=ec.device.type, dtype=ec.autocast_dtype or torch.float32, enabled=ec.autocast_dtype is not None):
        logits, _ = ec.model(idx.to(ec.device))
    lp = token_logps(logits[:, :-1], idx[:, 1:].to(ec.device))
    return (lp * mask.to(ec.device).float()).sum(-1).cpu().tolist()


def run_mmlu(ec: EvalContext) -> Tuple[Dict[str, float], Dict[str, Any]]:
    b = ec.cfg.eval.benchmarks
    items = ec.items(b.mmlu.dataset, "test", b.mmlu.max_examples)
    correct, conf, hit, logliks = 0, [], [], []
    for it in items:
        d = it.data
        ll = choice_logliks(ec, d["question"], d["choices"])
        c, pred = choice_confidence(ll)
        ok = pred == d["answer"]
        correct += ok
        conf.append(c)
        hit.append(ok)
    m = {"mmlu_acc": correct / max(len(items), 1)}
    if b.calibration is not None:
        m["mmlu_ece"] = ece(conf, hit, b.calibration.bins)              # B-8
    return m, {"mmlu_n": len(items)}


# --------------------------------------------------------------------------- B-2
def run_gsm8k(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.gsm8k
    items = ec.items(b.dataset, "test", b.max_examples)
    if ec.mode == "base":
        tmpl = ec.storage.read_text(b.base_prompt_uri)
        shots = ec.items(b.dataset, "fewshot", b.n_shot) if b.n_shot > 0 else []
        ex = "".join(f"Question: {s.data['question']}\nAnswer: {s.data['answer']}\n\n" for s in shots)
        texts = [tmpl.format(examples=ex, question=it.data["question"]) for it in items]
    else:
        tmpl = ec.storage.read_text(b.chat_prompt_uri)
        texts = [tmpl.format(question=it.data["question"]) for it in items]
    res = ec.generate([ec.prompt_tokens(t) for t in texts])
    ok = sum(gsm8k_correct(r.text, it.data["answer"], b.answer_regex) for r, it in zip(res, items))
    return {"gsm8k_acc": ok / max(len(items), 1)}, {"gsm8k_n": len(items)}


# --------------------------------------------------------------------------- B-3
def run_math500(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.math500
    items = ec.items(b.dataset, "test", b.max_examples)
    tmpl = ec.storage.read_text(b.chat_prompt_uri)
    res = ec.generate([ec.prompt_tokens(tmpl.format(problem=it.data["problem"])) for it in items])
    rx = ec.cfg.stages.rlvr.rewards.math.final_answer_regex
    ok = 0
    for r, it in zip(res, items):
        stripped = ec.template.strip_reasoning(r.text)
        if stripped is not None and math_equal(it.data["answer"], extract_answer(stripped, rx)):
            ok += 1
    return {"math500_acc": ok / max(len(items), 1)}, {"math500_n": len(items)}


# --------------------------------------------------------------------------- B-4
def run_humaneval(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.humaneval
    items = ec.items(b.dataset, "test", b.max_examples)
    if ec.mode == "chat":
        tmpl = ec.storage.read_text(b.chat_prompt_uri)
        prompts = [ec.prompt_tokens(tmpl.format(prompt=it.data["prompt"])) for it in items]
        stops: Tuple[str, ...] = ()
    else:
        prompts = [ec.prompt_tokens(it.data["prompt"]) for it in items]
        stops = tuple(b.base_stop_strings)
    res = ec.generate(prompts, num_samples=b.n_samples, greedy=False, stop_strings=stops,
                      temperature=b.temperature, top_p=b.top_p)
    programs, owners = [], []
    for i, it in enumerate(items):
        for r in res[i * b.n_samples:(i + 1) * b.n_samples]:
            if ec.mode == "chat":
                code = last_code_block(ec.template.strip_reasoning(r.text) or "")
                prog = None if code is None else build_program(code, it.data["test"], it.data["entry_point"])
            else:
                prog = build_program(it.data["prompt"] + r.text, it.data["test"], it.data["entry_point"])
            programs.append(prog)
            owners.append(i)
    sb = ec.get_sandbox()
    runnable = [p for p in programs if p is not None]
    results = iter(sb.run_many(runnable))
    passed = [False if p is None else next(results).passed for p in programs]
    c = [sum(passed[j] for j in range(len(programs)) if owners[j] == i) for i in range(len(items))]
    n = [b.n_samples] * len(items)
    return {f"humaneval_pass@{k}": mean_pass_at_k(n, c, k) for k in b.k_values}, {"humaneval_n": len(items)}


# --------------------------------------------------------------------------- B-5
def run_ifeval(ec: EvalContext):
    from .benchmarks.ifeval import score_ifeval
    b = ec.cfg.eval.benchmarks.ifeval
    items = ec.items(b.dataset, "test", b.max_examples)
    res = ec.generate([ec.prompt_tokens(it.data["prompt"]) for it in items])
    return score_ifeval([it.data for it in items], [r.text for r in res]), {"ifeval_n": len(items)}


# --------------------------------------------------------------------------- B-6
def run_alpaca(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.alpaca_eval_lc
    items = ec.items(b.dataset, "test", b.max_examples)
    judge = ec.get_judge()
    res = ec.generate([ec.prompt_tokens(it.data["instruction"]) for it in items])

    def outcome(pair):
        it, r = pair
        ins, ref = it.data["instruction"], it.data["reference_output"]
        first = judge.pairwise(ins, r.text, ref)          # model as A
        second = judge.pairwise(ins, ref, r.text)         # model as B
        score = lambda v, model_label: 1.0 if v == model_label else (0.5 if v == "TIE" else 0.0)      # noqa: E731
        return (score(first, "A") + score(second, "B")) / 2
    outs = judge.map(outcome, list(zip(items, res)))
    len_m = [len(r.completion_tokens) for r in res]
    len_r = [len(ec.tok.encode_ordinary(it.data["reference_output"])) for it in items]
    m = lc_win_rate(outs, len_m, len_r, b.max_iter)
    return {"alpaca_win_rate": m["alpaca_win_rate"], "alpaca_lc_win_rate": m["alpaca_lc_win_rate"]}, {"alpaca_n": len(items)}


# --------------------------------------------------------------------------- B-7
def run_safety(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.safety
    items = ec.items(b.dataset, "test", b.max_examples)
    judge = ec.get_judge()
    res = ec.generate([ec.prompt_tokens(it.data["prompt"]) for it in items])
    classes = judge.map(lambda p: judge.refusal(p[0].data["prompt"], p[1].text), list(zip(items, res)))
    unsafe = [c for it, c in zip(items, classes) if not it.data["is_safe"]]
    safe = [c for it, c in zip(items, classes) if it.data["is_safe"]]
    m: Dict[str, float] = {}
    if unsafe:
        m["safety_unsafe_refusal_rate"] = sum(c in ("refusal", "partial_refusal") for c in unsafe) / len(unsafe)
    if safe:
        m["safety_safe_compliance_rate"] = sum(c == "compliance" for c in safe) / len(safe)
    return m, {"safety_n": len(items)}


# --------------------------------------------------------------------------- B-9 / B-10
def run_length(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.length
    items = ec.items(b.dataset, "test", b.n_prompts)
    res = ec.generate([ec.prompt_tokens(it.data["instruction"]) for it in items])
    lens = [len(r.completion_tokens) for r in res]
    return {"response_len_mean": float(statistics.fmean(lens)), "response_len_median": float(statistics.median(lens))}, {"length_n": len(lens)}


def run_adherence(ec: EvalContext):
    b = ec.cfg.eval.benchmarks.adherence
    sft = ec.cfg.stages.sft
    prompts = []
    for src in sft.data.sources:                                  # held-out SFT prompts (docs/prd_review.md #12)
        for it in ec.registry.items(src.dataset, "test"):
            msgs = []
            for m in it.messages:
                if m["role"] == "assistant":
                    break
                msgs.append(m)
            if msgs:
                prompts.append(ec.template.render_prompt(msgs))
            if len(prompts) >= b.n_prompts:
                break
        if len(prompts) >= b.n_prompts:
            break
    res = ec.generate(prompts)
    ok = sum(r.finish_reason == "eos" and r.completion_tokens and r.completion_tokens[-1] == ec.tok.im_end_id for r in res)
    return {"template_adherence": ok / max(len(res), 1)}, {"adherence_n": len(res)}


RUNNERS = {"mmlu": run_mmlu, "gsm8k": run_gsm8k, "math500": run_math500, "humaneval": run_humaneval,
           "ifeval": run_ifeval, "alpaca_eval_lc": run_alpaca, "safety": run_safety, "length": run_length,
           "adherence": run_adherence}


def run_suite(ec: EvalContext, checkpoint_uri: str, checkpoint_sha256: Optional[str] = None) -> Dict[str, Any]:
    ec.model.eval()
    b = ec.cfg.eval.benchmarks
    metrics: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for name, fn in RUNNERS.items():
        if getattr(b, name) is None:
            continue
        if ec.mode == "base" and name in CHAT_ONLY:
            skipped[name] = "chat-mode benchmark; checkpoint is evaluated in base mode (EV-2)"
            continue
        m, d = fn(ec)
        metrics.update(m)
        details.update(d)
    if b.calibration is not None and b.mmlu is None:
        skipped["calibration"] = "requires mmlu"
    return {"checkpoint_uri": checkpoint_uri, "checkpoint_sha256": checkpoint_sha256, "stage": ec.stage, "mode": ec.mode,
            "metrics": metrics, "details": details, "skipped": skipped, "eval_config_hash": eval_config_hash(ec.cfg, checkpoint_sha256)}


def eval_config_hash(cfg, checkpoint_sha256: Optional[str]) -> str:
    """Hash of everything an evaluation result depends on. PL-5: evaluation and gating rerun when
    this changes, training does not."""
    b = cfg.eval.benchmarks
    ds = {n: cfg.datasets[n].model_dump() for n in {getattr(b, k).dataset for k in
          ("mmlu", "gsm8k", "math500", "humaneval", "ifeval", "alpaca_eval_lc", "safety", "length") if getattr(b, k) is not None}}
    return sha256_json({"eval": cfg.eval.model_dump(), "datasets": ds, "generation": cfg.generation.model_dump(),
                        "tokenizer": cfg.tokenizer.model_dump(), "chat_template": cfg.chat_template.model_dump(),
                        "judge": {"model": cfg.judge.model, "temperature": cfg.judge.temperature, "parse": cfg.judge.parse.model_dump()},
                        "code_execution": cfg.code_execution.model_dump(),
                        "math_regex": cfg.stages.rlvr.rewards.math.final_answer_regex,
                        "adherence_sources": [s.dataset for s in cfg.stages.sft.data.sources],
                        "checkpoint_sha256": checkpoint_sha256})


# --------------------------------------------------------------------------- entry points
def build_context(cfg, storage: Storage, checkpoint_uri: str, device: torch.device, autocast_dtype, judge=None,
                  sandbox=None) -> Tuple[EvalContext, Optional[str]]:
    ckpt = load_state(storage.cached_local_path(checkpoint_uri))
    lineage = ckpt.get("lineage")
    arch = lineage["architecture"] if lineage else cfg.base_model.architecture
    model = load_model(ckpt, arch, device).eval()
    tok = ChatTokenizer.from_config(cfg, storage)
    stage = lineage["stage"] if lineage else None
    ec = EvalContext(cfg, storage, tok, ChatTemplate.from_config(cfg, tok), model, device, autocast_dtype, eval_mode(stage),
                     stage, DatasetRegistry.from_config(cfg, storage), judge=judge, sandbox=sandbox)
    return ec, stage


def evaluate_checkpoint(cfg, storage: Storage, checkpoint_uri: str, out_uri: Optional[str], device: torch.device,
                        autocast_dtype, checkpoint_sha256: Optional[str] = None, metrics_logger=None, judge=None,
                        sandbox=None) -> Dict[str, Any]:
    from ..training.lineage import sha256_uri
    sha = checkpoint_sha256 or sha256_uri(storage, checkpoint_uri)
    ec, _ = build_context(cfg, storage, checkpoint_uri, device, autocast_dtype, judge, sandbox)
    report = run_suite(ec, checkpoint_uri, sha)
    if out_uri is not None:
        storage.write_text(Storage.join(out_uri, "eval_report.json"), json.dumps(report, indent=2, sort_keys=True))
    if metrics_logger is not None:
        metrics_logger.log("eval_suite", step=0, checkpoint=checkpoint_uri, **report["metrics"])
    return report
