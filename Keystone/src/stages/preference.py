"""Stage 3: preference optimization (PO-1..PO-6, DP-8).

Preparation, in order: offline pairs (prepare.prepare_pref); on-policy judged pairs (PO-5) as the
produced dataset `on_policy`; reference log-probabilities of chosen and rejected responses under
the parent (DP-8, DPO only), written as extra Parquet columns.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch

from ..config.loader import sha256_json
from ..data import loaders, mixture, prepare
from ..data.adapters import PrefPair
from ..foundation import bridge
from ..generation.api import GenerationConfig, generate
from ..io.storage import Storage
from ..modeling.inference import InferenceModel
from ..training import checkpoint as ck
from ..training.loop import Objective, Trainer, TrainResult
from ..training.objectives.common import token_logps
from ..training.objectives.dpo import dpo_loss
from ..training.objectives.simpo import simpo_loss
from .common import StageContext, load_stage_model, run_training

STAGES = ("preference",)
STAGE = "preference"


# --------------------------------------------------------------------------- sources (PO-5)
def pref_sources(cfg) -> List[Tuple[str, float]]:
    """Mixture sources of Stage 3. With on-policy pairs enabled, `on_policy` joins the mixture with
    `on_policy.weight` (PO-5), whether or not the source list names it."""
    p = cfg.stages.preference
    out = [(s.dataset, s.weight) for s in p.data.sources if s.dataset != "on_policy"]
    if p.on_policy.enabled:
        out.append(("on_policy", p.on_policy.weight))
    return out


# --------------------------------------------------------------------------- PO-5
def generate_on_policy_pairs(sctx: StageContext, pctx: prepare.PrepContext, model, judge) -> Tuple[List[PrefPair], Dict[str, Any]]:
    cfg = sctx.cfg
    op = cfg.stages.preference.on_policy
    items = []
    for it in pctx.registry.items(op.prompts_dataset, "train"):
        items.append(it)
        if len(items) >= op.max_prompts:
            break
    prompts = [sctx.template.render_prompt(it.messages) for it in items]
    gcfg = GenerationConfig.from_block(op.generation, num_samples=op.samples_per_prompt,
                                       stop_token_ids=[cfg.tokenizer.eot_token_id, sctx.tok.im_end_id])
    results = generate(InferenceModel(model), sctx.tok, prompts, gcfg, autocast_dtype=sctx.autocast_dtype)
    k = op.samples_per_prompt
    jobs = [(items[i].messages[-1]["content"], results[i * k + j].text) for i in range(len(items)) for j in range(k)]
    scores = judge.map(lambda a: judge.score(*a), jobs)
    pairs: List[PrefPair] = []
    skipped_tie, skipped_unscored = 0, 0
    for i, it in enumerate(items):
        sc = [(scores[i * k + j], j) for j in range(k) if scores[i * k + j] is not None]
        if len(sc) < 2:
            skipped_unscored += 1
            continue
        hi, lo = max(sc, key=lambda x: (x[0], -x[1])), min(sc, key=lambda x: (x[0], x[1]))
        if hi[0] == lo[0]:                                  # PO-5: top and bottom equal -> no pair
            skipped_tie += 1
            continue
        pairs.append(PrefPair(it.messages, results[i * k + hi[1]].text, results[i * k + lo[1]].text, it.texts))
    return pairs, {"prompts": len(items), "samples_per_prompt": k, "pairs": len(pairs),
                   "skipped_equal_scores": skipped_tie, "skipped_unscored": skipped_unscored}


# --------------------------------------------------------------------------- DP-8
@torch.no_grad()
def response_logps(model, idx: torch.Tensor, resp_mask: torch.Tensor, device, autocast_dtype) -> torch.Tensor:
    """Summed fp32 log-prob of response tokens per sequence (PO-2), under the same autocast the
    training forward uses (docs/prd_review.md #15)."""
    with torch.autocast(device_type=device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None):
        logits, _ = model(idx.to(device))
    return (token_logps(logits[:, :-1], idx[:, 1:].to(device)) * resp_mask.to(device).float()).sum(-1).cpu()


def add_reference_logps(sctx: StageContext, pctx: prepare.PrepContext, model, dataset: str, split: str) -> None:
    cfg, st = sctx.cfg, sctx.storage
    uri = prepare.prepared_uri(cfg, STAGE, dataset, split)
    ref_marker = Storage.join(uri, prepare.REF_MARKER)
    want = {"ref_sha256": sctx.parent.sha256, "autocast": str(sctx.autocast_dtype)}
    if st.exists(ref_marker) and json.loads(st.read_text(ref_marker)) == want:
        return
    rows = list(prepare.read_rows(st, uri))
    model.eval()
    micro = cfg.stages.preference.batch.micro_batch_size
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        b = loaders.collate_pref(chunk, cfg.tokenizer.pad_token_id)
        lp = response_logps(model, b["idx"], b["resp_mask"], sctx.device, sctx.autocast_dtype)
        n = len(chunk)
        for j, r in enumerate(chunk):
            r["ref_logp_chosen"] = float(lp[j])
            r["ref_logp_rejected"] = float(lp[n + j])
    entry = (prepare.read_marker(st, uri) or {})
    prepare.write_shards(st, uri, "pref", rows, cfg.prepared_data.shard_rows, extra=prepare.REF_COLUMNS)
    if entry:
        st.write_text(Storage.join(uri, prepare.MARKER), json.dumps(entry, sort_keys=True, default=str))
    st.write_text(ref_marker, json.dumps(want))


def prepare_stage(sctx: StageContext, pctx: prepare.PrepContext, judge=None) -> Dict[str, Dict[str, Any]]:
    cfg, st = sctx.cfg, sctx.storage
    pref = cfg.stages.preference
    out = prepare.prepare_stage_datasets(pctx, STAGE, sctx.ctx_len)
    need_model = pref.on_policy.enabled or pref.objective == "dpo"
    model = None
    if need_model:
        model = load_stage_model(sctx)
        model.eval()
    if pref.on_policy.enabled:
        if judge is None:
            from ..services.judge import Judge
            judge = Judge.from_config(cfg, st)
        h = sha256_json({"parent": sctx.parent.sha256, "on_policy": pref.on_policy.model_dump(), "judge": judge.identity(),
                         "prompts_files": pctx.registry.file_manifest(pref.on_policy.prompts_dataset, "train"),
                         "max_total_tokens": pref.max_total_tokens, "tok": cfg.tokenizer.model_dump(),
                         "template": cfg.chat_template.model_dump(), "seed": cfg.run.seed})
        uri = prepare.prepared_uri(cfg, STAGE, "on_policy", "train")
        cur = prepare.read_marker(st, uri)
        if cur is not None and cur.get("input_hash") == h:
            entry = cur["entry"]
        else:
            pairs, meta = generate_on_policy_pairs(sctx, pctx, model, judge)
            if not pairs:
                raise ValueError("on-policy generation produced no pairs (all prompts had equal or missing scores)")
            entry = prepare.prepare_produced_pref(pctx, STAGE, "on_policy", "train", pairs, pref.max_total_tokens,
                                                  dict(meta, judge=judge.identity()), input_hash=h)
        out["on_policy"] = {"train": entry}
        prepare.merge_stage_manifest(pctx, STAGE, {"on_policy": {"train": entry}})
    if pref.objective == "dpo":
        for name, splits in out.items():
            for split in splits:
                add_reference_logps(sctx, pctx, model, name, split)
    return out


# --------------------------------------------------------------------------- objective
class PreferenceObjective(Objective):
    event = "pref_step"

    def __init__(self, sctx: StageContext):
        self.s = sctx
        cfg, blk = sctx.cfg, sctx.block
        self.kind = blk.objective
        self.micro, self.accum = blk.batch.micro_batch_size, blk.batch.grad_accum
        self.pad = cfg.tokenizer.pad_token_id
        srcs = pref_sources(cfg)
        self.names = [n for n, _ in srcs]
        self.weights = mixture.normalise([w for _, w in srcs])
        self.rows = [prepare.prepared_rows(sctx.storage, cfg, STAGE, n, "train") for n in self.names]
        self.counts = mixture.epoch_counts(self.rows, self.weights)                                   # DP-9
        self.quotas = [max(1, round(c / sctx.world)) for c in self.counts]
        self.total_steps = mixture.steps_for_epochs(blk.data.epochs, self.counts, self.micro, self.accum, sctx.world)
        self.total_updates = self.total_steps
        self.val_names = [n for n in self.names if prepare.read_marker(
            sctx.storage, prepare.prepared_uri(cfg, STAGE, n, "test")) is not None]
        self.cols = ["prompt_tokens", "chosen_tokens", "rejected_tokens"] + (
            ["ref_logp_chosen", "ref_logp_rejected"] if self.kind == "dpo" else [])
        self.mix: Optional[mixture.MixtureStream] = None

    def _uri(self, name, split):
        return prepare.prepared_uri(self.s.cfg, STAGE, name, split)

    def setup(self, trainer: Trainer) -> None:
        s = self.s
        streams = [loaders.RowStream(s.storage, self._uri(n, "train"), self.cols, s.rank, s.world,
                                     s.cfg.run.seed + i, cycle=True) for i, n in enumerate(self.names)]
        self.mix = mixture.MixtureStream(streams, self.weights, seed=s.cfg.run.seed + s.rank, quotas=self.quotas)

    def state_dict(self):
        return {"draws": self.mix.draws}

    def load_state_dict(self, d):
        self.mix.fast_forward(d["draws"])

    # ------------------------------------------------------------------ loss on one batch
    def _loss(self, trainer: Trainer, b: Dict[str, torch.Tensor], track_usage=True):
        idx, mask = b["idx"], b["resp_mask"].to(trainer.device)
        logits, aux = trainer.forward(idx, None, track_usage=track_usage)                     # DL-3: no attention mask
        lp = (token_logps(logits[:, :-1], idx[:, 1:].to(trainer.device)) * mask.float()).sum(-1)     # PO-2, fp32
        n = int(b["n"])
        pol_c, pol_r = lp[:n], lp[n:]
        if self.kind == "dpo":
            loss, m = dpo_loss(pol_c, pol_r, b["ref_chosen"][:n].float().to(trainer.device),
                               b["ref_rejected"][:n].float().to(trainer.device), self.s.block.beta)
        else:
            loss, m = simpo_loss(pol_c, pol_r, b["len_chosen"].to(trainer.device), b["len_rejected"].to(trainer.device),
                                 self.s.block.beta, self.s.block.simpo_gamma)
        m["chosen_len"] = float(b["len_chosen"].float().mean())
        m["rejected_len"] = float(b["len_rejected"].float().mean())
        return loss, aux, m

    def train_step(self, trainer: Trainer, step: int) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        loss_total = 0.0
        for i in range(self.accum):
            b = loaders.collate_pref([self.mix.next()[1] for _ in range(self.micro)], self.pad)
            with trainer.accumulate(i == self.accum - 1):
                loss, aux, m = self._loss(trainer, b)
                total = loss / self.accum
                if aux is not None:
                    total = total + aux / self.accum                                          # TR-2
                trainer.backward(total)
            loss_total += float(loss.detach())
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v / self.accum
        gn = trainer.apply_update()
        keys = sorted(agg)
        red = trainer.all_reduce(torch.tensor([loss_total / self.accum] + [agg[k] for k in keys], dtype=torch.float64))
        red = (red / trainer.world).tolist()
        out = {"loss": red[0], **{k: v for k, v in zip(keys, red[1:])}, "grad_norm": gn}
        return out

    # ------------------------------------------------------------------ PO-6
    @torch.no_grad()
    def validate(self, trainer: Trainer, step: int) -> Dict[str, float]:
        s = self.s
        trainer.wrapped.eval()
        try:
            batches = []
            for n in self.val_names:
                stream = loaders.RowStream(s.storage, self._uri(n, "test"), self.cols, s.rank, s.world, s.cfg.run.seed,
                                           cycle=False, shuffle=False)
                rows = list(stream)
                batches += [rows[i:i + self.micro] for i in range(0, len(rows), self.micro)]
            from ..stages.common import dist_max
            n_max = int(trainer.all_reduce(torch.tensor(len(batches)), dist_max()).item())
            loss_sum = torch.zeros((), dtype=torch.float64)
            acc_sum = torch.zeros((), dtype=torch.float64)
            cnt = torch.zeros((), dtype=torch.float64)
            for k in range(n_max):
                rows = batches[k] if k < len(batches) else batches[0] if batches else None
                real = k < len(batches)
                if rows is None:
                    rows = [{"prompt_tokens": [self.pad], "chosen_tokens": [self.pad], "rejected_tokens": [self.pad - 1],
                             "ref_logp_chosen": 0.0, "ref_logp_rejected": 0.0}]
                b = loaders.collate_pref(rows, self.pad)
                loss, _, m = self._loss(trainer, b, track_usage=False)
                if real:
                    n = len(rows)
                    loss_sum += float(loss) * n
                    acc_sum += m["accuracy"] * n
                    cnt += n
            red = trainer.all_reduce(torch.stack([loss_sum, acc_sum, cnt]).to(trainer.device))
        finally:
            trainer.wrapped.train()
        c = max(float(red[2]), 1.0)
        return {"val_loss": float(red[0]) / c, "val_accuracy": float(red[1]) / c}

    def data_manifest(self) -> List[Dict[str, Any]]:
        return [{"dataset": n, "weight": w, "rows_available": r, "rows_per_epoch": c}
                for n, w, r, c in zip(self.names, self.weights, self.rows, self.counts)]

    def services(self) -> Dict[str, Any]:
        if not self.s.cfg.stages.preference.on_policy.enabled:
            return {}
        j = self.s.cfg.judge
        return {"judge": {"model": j.model, "base_url": j.base_url}}


def build_objective(sctx: StageContext) -> PreferenceObjective:
    return PreferenceObjective(sctx)


def train(sctx: StageContext) -> TrainResult:
    return run_training(sctx, build_objective(sctx))
