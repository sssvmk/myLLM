"""Data preparation (DP-1..DP-7): read datasets from their URIs, adapt, decontaminate, tokenize,
pack, and write Parquet shards under
`prepared_data.root_uri/<run.name>/<stage>/<dataset>/<split>/part-*.parquet` (DP-1).

Training reads only what this module wrote. Runs in plain Python with `prepared_data.num_workers`
processes for tokenization (DP-2, no Spark). Two things the PRD leaves open (docs/implementation_notes.md):

* Documents of one dataset split are held in memory as token lists before packing, because DP-5
  shuffles document order globally. Very large corpora are prepared one `max_samples`-bounded
  dataset entry at a time.
* Each (stage, dataset, split) directory carries `_prepared.json` with a hash of everything that
  influenced it; an unchanged input hash makes preparation a no-op (PL-5).
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from ..config.loader import sha256_json
from ..io.storage import Storage
from .adapters import Conversation, PackedBlock, PrefPair, PromptItem, RLCodeItem, RLMathItem, TextDoc
from .decontam import build_eval_ngram_set, filter_contaminated
from .packing import pack_lm, pack_sft, repack_packed
from .registry import AdaptStats, DatasetRegistry

LM_KINDS = {"text", "packed_tokens"}
SHARD_GLOB = "part-*.parquet"
MARKER = "_prepared.json"
REF_MARKER = "_ref.json"

SCHEMAS: Dict[str, List[Tuple[str, pa.DataType]]] = {
    "lm": [("tokens", pa.list_(pa.int64())), ("doc_start", pa.list_(pa.bool_()))],
    "sft": [("tokens", pa.list_(pa.int64())), ("doc_start", pa.list_(pa.bool_())), ("loss_mask", pa.list_(pa.bool_()))],
    "pref": [("prompt_tokens", pa.list_(pa.int64())), ("chosen_tokens", pa.list_(pa.int64())),
             ("rejected_tokens", pa.list_(pa.int64())), ("source", pa.string())],
    "rl_math": [("prompt_tokens", pa.list_(pa.int64())), ("ground_truth", pa.string()), ("source", pa.string())],
    "rl_code": [("prompt_tokens", pa.list_(pa.int64())), ("tests", pa.string()), ("entry_point", pa.string()),
                ("source", pa.string())],
    "prompts": [("prompt_tokens", pa.list_(pa.int64())), ("source", pa.string())],
}
REF_COLUMNS = [("ref_logp_chosen", pa.float64()), ("ref_logp_rejected", pa.float64())]     # DP-8

# --------------------------------------------------------------------------- paths
def prepared_uri(cfg, stage_id: str, dataset: str, split: str) -> str:
    return Storage.join(cfg.prepared_data.root_uri, cfg.run.name, stage_id, dataset, split)


def stage_manifest_uri(cfg, stage_id: str) -> str:
    return Storage.join(cfg.prepared_data.root_uri, cfg.run.name, stage_id, "manifest.json")


def read_produced_marker(storage: Storage, uri: str) -> Optional[Dict[str, Any]]:
    return read_marker(storage, uri)


def read_marker(storage: Storage, uri: str) -> Optional[Dict[str, Any]]:
    m = Storage.join(uri, MARKER)
    return json.loads(storage.read_text(m)) if storage.exists(m) else None


def prepared_rows(storage: Storage, cfg, stage_id: str, dataset: str, split: str) -> int:
    m = read_marker(storage, prepared_uri(cfg, stage_id, dataset, split))
    if m is None:
        raise FileNotFoundError(f"{dataset}/{split} has not been prepared for stage {stage_id}; run `pf prepare-data`")
    return int(m["rows"])


# --------------------------------------------------------------------------- shard writing
def _table(schema_name: str, rows: List[Dict[str, Any]], extra: Sequence[Tuple[str, pa.DataType]] = ()) -> pa.Table:
    spec = list(SCHEMAS[schema_name]) + list(extra)
    return pa.table({name: pa.array([r[name] for r in rows], type=typ) for name, typ in spec},
                    schema=pa.schema(spec))


def _clear(storage: Storage, uri: str) -> None:
    for f in storage.glob(uri, SHARD_GLOB):
        storage.delete(f)
    ref = Storage.join(uri, REF_MARKER)
    if storage.exists(ref):
        storage.delete(ref)              # reference log-probs (DP-8) belonged to the shards just removed


def write_shards(storage: Storage, uri: str, schema_name: str, rows: Iterable[Dict[str, Any]], shard_rows: int,
                 extra: Sequence[Tuple[str, pa.DataType]] = ()) -> Tuple[int, List[str]]:
    """Writes `rows` in shards of `shard_rows`; returns (row count, shard URIs). Existing shards
    in `uri` are removed first so a rerun never mixes generations."""
    _clear(storage, uri)
    buf: List[Dict[str, Any]] = []
    files: List[str] = []
    n = 0

    def flush():
        nonlocal buf
        if not buf:
            return
        sink = io.BytesIO()
        pq.write_table(_table(schema_name, buf, extra), sink)
        f = Storage.join(uri, f"part-{len(files):05d}.parquet")
        storage.write_bytes(f, sink.getvalue())
        files.append(f)
        buf = []

    for r in rows:
        buf.append(r)
        n += 1
        if len(buf) == shard_rows:
            flush()
    flush()
    return n, files


def read_rows(storage: Storage, uri: str) -> Iterator[Dict[str, Any]]:
    for f in storage.glob(uri, SHARD_GLOB):
        yield from pq.read_table(storage.cached_local_path(f)).to_pylist()


# --------------------------------------------------------------------------- tokenization workers
_STATE: Dict[str, Any] = {}


def _w_text(docs: List[str]):
    return [_STATE["tok"].encode_ordinary(d) for d in docs]


def _w_sft(convs: List[List[Dict[str, str]]]):
    out = []
    for m in convs:
        r = _STATE["tmpl"].render(m)
        out.append((r.tokens, r.loss_mask))
    return out


def _w_pref(items):
    t = _STATE["tmpl"]
    return [(t.render_prompt(p), t.render_response(c), t.render_response(r)) for p, c, r in items]


def _w_prompt(msgs: List[List[Dict[str, str]]]):
    return [_STATE["tmpl"].render_prompt(m) for m in msgs]


def _map_chunks(fn: Callable, items: Sequence, workers: int, chunk: int = 256) -> List:
    """Applies `fn` (a list -> list worker) over chunks, in order. With workers > 1 and a fork
    start method the module state (tokenizer) is inherited by the workers."""
    chunks = [list(items[i:i + chunk]) for i in range(0, len(items), chunk)]
    if workers <= 1 or len(chunks) <= 1 or "fork" not in mp.get_all_start_methods():
        return [x for c in chunks for x in fn(c)]
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return [x for part in ex.map(fn, chunks) for x in part]


# --------------------------------------------------------------------------- context
@dataclass
class PrepContext:
    cfg: Any
    storage: Storage
    tok: Any
    template: Any
    registry: DatasetRegistry
    force: bool = False
    _eval_ngrams: Optional[Any] = field(default=None, repr=False)
    _eval_used: Dict[str, int] = field(default_factory=dict)
    log: Callable[[str], None] = print

    def __post_init__(self):
        _STATE["tok"], _STATE["tmpl"] = self.tok, self.template

    @property
    def workers(self) -> int:
        return self.cfg.prepared_data.num_workers

    def eval_ngrams(self):
        if self._eval_ngrams is None:
            d = self.cfg.prepared_data.decontamination
            self._eval_ngrams, self._eval_used = build_eval_ngram_set(self.registry, d.against, d.splits, d.ngram)
            self.log(f"decontamination: {len(self._eval_ngrams)} n-grams from {self._eval_used}")
        return self._eval_ngrams

    def decontaminate(self, items: Iterable, counter: dict) -> Iterator:
        d = self.cfg.prepared_data.decontamination
        return filter_contaminated(items, self.eval_ngrams(), d.ngram, d.overlap_threshold, counter)

    def input_hash(self, stage_id: str, dataset: str, split: str, extra: Dict[str, Any]) -> str:
        cfg = self.cfg
        d = cfg.prepared_data.decontamination
        against = {n: {"cfg": cfg.datasets[n].model_dump(), "files": self._safe_manifest(n, s)}
                   for n in d.against for s in d.splits if n in cfg.datasets and self.registry.has_split(n, s)}
        return sha256_json({
            "dataset": cfg.datasets[dataset].model_dump(), "files": self._safe_manifest(dataset, split),
            "split": split, "seed": cfg.run.seed, "holdout": cfg.prepared_data.holdout_fraction,
            "shard_rows": cfg.prepared_data.shard_rows, "max_drop": cfg.prepared_data.max_drop_fraction,
            "tokenizer": cfg.tokenizer.model_dump(), "chat_template": cfg.chat_template.model_dump(),
            "decontam": {"ngram": d.ngram, "threshold": d.overlap_threshold, "against": against},
            "stage": stage_id, "extra": extra})

    def _safe_manifest(self, name: str, split: str):
        return [{k: v for k, v in e.items()} for e in self.registry.file_manifest(name, split)]


def _record(ctx: PrepContext, stage_id: str, name: str, split: str, stats: AdaptStats, uri: str,
            rows: int, files: List[str], input_hash: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    entry = ctx.registry.manifest_entry(name, split, stats, extra=dict(extra, rows=rows, prepared_uri=uri,
                                                                       shards=len(files)))
    marker = {"input_hash": input_hash, "rows": rows, "entry": entry}
    ctx.storage.write_text(Storage.join(uri, MARKER), json.dumps(marker, sort_keys=True, default=str))
    return entry


def _cached(ctx: PrepContext, uri: str, input_hash: str) -> Optional[Dict[str, Any]]:
    if ctx.force:
        return None
    m = read_marker(ctx.storage, uri)
    return m["entry"] if m and m.get("input_hash") == input_hash and ctx.storage.glob(uri, SHARD_GLOB) else None


# --------------------------------------------------------------------------- per-kind preparation
def prepare_lm(ctx: PrepContext, stage_id: str, name: str, split: str, seq_ctx: int) -> Dict[str, Any]:
    cfg = ctx.cfg
    uri = prepared_uri(cfg, stage_id, name, split)
    h = ctx.input_hash(stage_id, name, split, {"kind": "lm", "ctx": seq_ctx})
    if (hit := _cached(ctx, uri, h)) is not None:
        return hit
    ds = cfg.datasets[name]
    stats = AdaptStats(name, split)
    counter: Dict[str, int] = {}
    pad, eot = cfg.tokenizer.pad_token_id, cfg.tokenizer.eot_token_id
    items = ctx.decontaminate(ctx.registry.items(name, split, stats), counter)
    if ds.kind == "text":
        docs = [it.text for it in items if isinstance(it, TextDoc)]
        toks = _map_chunks(_w_text, docs, ctx.workers)
        blocks = pack_lm(toks, seq_ctx, eot, pad, cfg.run.seed)
    else:
        packed: List[PackedBlock] = list(items)
        stats.check(cfg.prepared_data.max_drop_fraction)
        blocks = repack_packed(((b.tokens, b.doc_start) for b in packed), seq_ctx, pad)
    rows = ({"tokens": t, "doc_start": s} for t, s in blocks)
    n, files = write_shards(ctx.storage, uri, "lm", rows, cfg.prepared_data.shard_rows)
    if n == 0:
        raise ValueError(f"dataset {name!r} split {split!r} produced no blocks for stage {stage_id}")
    return _record(ctx, stage_id, name, split, stats, uri, n, files, h,
                   {"decontamination": counter, "block_len": seq_ctx + 1, "tokens": n * seq_ctx})


def _sft_rows(ctx: PrepContext, items: List[Conversation], seq_ctx: int, stats: AdaptStats):
    """DP-6 on adapted conversations: render, drop those without assistant tokens or longer than
    ctx + 1, pack greedily. Returns (blocks, conversations kept)."""
    cfg = ctx.cfg
    rendered = _map_chunks(_w_sft, [c.messages for c in items], ctx.workers)
    convs = []
    for toks, mask in rendered:
        if not any(mask):
            stats.drop("no_assistant_tokens")
            continue
        convs.append((toks, mask))
    too_long: List[int] = []
    blocks = list(pack_sft(convs, seq_ctx, cfg.tokenizer.pad_token_id, cfg.run.seed, dropped=too_long))
    for _ in too_long:
        stats.drop("longer_than_ctx")
    kept = [c for i, c in enumerate(convs) if i not in set(too_long)]
    return blocks, kept


def prepare_sft(ctx: PrepContext, stage_id: str, name: str, split: str, seq_ctx: int) -> Dict[str, Any]:
    cfg = ctx.cfg
    uri = prepared_uri(cfg, stage_id, name, split)
    h = ctx.input_hash(stage_id, name, split, {"kind": "sft", "ctx": seq_ctx})
    if (hit := _cached(ctx, uri, h)) is not None:
        return hit
    stats = AdaptStats(name, split)
    counter: Dict[str, int] = {}
    items = list(ctx.decontaminate(ctx.registry.items(name, split, stats), counter))
    blocks, convs = _sft_rows(ctx, items, seq_ctx, stats)
    stats.check(cfg.prepared_data.max_drop_fraction)
    rows = ({"tokens": t, "doc_start": s, "loss_mask": m} for t, s, m in blocks)
    n, files = write_shards(ctx.storage, uri, "sft", rows, cfg.prepared_data.shard_rows)
    if n == 0:
        raise ValueError(f"dataset {name!r} split {split!r} produced no SFT blocks for stage {stage_id}")
    return _record(ctx, stage_id, name, split, stats, uri, n, files, h,
                   {"decontamination": counter, "block_len": seq_ctx + 1, "conversations": len(convs),
                    "assistant_tokens": sum(sum(m) for _, m in convs)})


def _produced_entry(ctx: PrepContext, name: str, split: str, uri: str, rows: int, files: List[str],
                    third_party_generated: bool, extra: Dict[str, Any], input_hash: Optional[str] = None) -> Dict[str, Any]:
    entry = {"name": name, "split": split, "uri": uri, "files": [], "license": "generated by this pipeline",
             "third_party_generated": third_party_generated, "rows": rows, "prepared_uri": uri,
             "shards": len(files), "produced": True}
    entry.update(extra)
    ctx.storage.write_text(Storage.join(uri, MARKER), json.dumps({"input_hash": input_hash, "rows": rows, "entry": entry},
                                                                 sort_keys=True, default=str))
    return entry


def prepare_produced_sft(ctx: PrepContext, stage_id: str, name: str, split: str, convs: List[Conversation],
                         seq_ctx: int, third_party_generated: bool, meta: Dict[str, Any],
                         input_hash: Optional[str] = None) -> Dict[str, Any]:
    """`teacher_traces` (DI-2): a pipeline-made conversations dataset, packed like any SFT data."""
    uri = prepared_uri(ctx.cfg, stage_id, name, split)
    stats = AdaptStats(name, split)
    stats.read = len(convs)
    stats.kept = len(convs)
    blocks, kept = _sft_rows(ctx, list(convs), seq_ctx, stats)
    rows = ({"tokens": t, "doc_start": s, "loss_mask": m} for t, s, m in blocks)
    n, files = write_shards(ctx.storage, uri, "sft", rows, ctx.cfg.prepared_data.shard_rows)
    if n == 0:
        if split == "train":
            raise ValueError(f"{name}: every conversation is longer than ctx {seq_ctx} + 1 tokens ({dict(stats.dropped)})")
        return None                                   # an empty held-out split is simply absent (no marker)
    return _produced_entry(ctx, name, split, uri, n, files, third_party_generated,
                           dict(meta, conversations=len(kept), dropped=dict(stats.dropped)), input_hash)


def prepare_produced_pref(ctx: PrepContext, stage_id: str, name: str, split: str, pairs: List[PrefPair],
                          max_total_tokens: int, meta: Dict[str, Any], input_hash: Optional[str] = None) -> Dict[str, Any]:
    """`on_policy` (PO-5): judged pairs, filtered per DP-7."""
    uri = prepared_uri(ctx.cfg, stage_id, name, split)
    toks = _map_chunks(_w_pref, [(p.prompt_messages, p.chosen, p.rejected) for p in pairs], ctx.workers)
    rows, dropped = [], {}
    for (p, c, r) in toks:
        if len(p) + max(len(c), len(r)) > max_total_tokens:
            dropped["longer_than_max_total_tokens"] = dropped.get("longer_than_max_total_tokens", 0) + 1
        elif c == r:
            dropped["chosen_equals_rejected"] = dropped.get("chosen_equals_rejected", 0) + 1
        else:
            rows.append({"prompt_tokens": p, "chosen_tokens": c, "rejected_tokens": r, "source": name})
    n, files = write_shards(ctx.storage, uri, "pref", rows, ctx.cfg.prepared_data.shard_rows)
    return _produced_entry(ctx, name, split, uri, n, files, False, dict(meta, dropped=dropped), input_hash)


def prepare_pref(ctx: PrepContext, stage_id: str, name: str, split: str, max_total_tokens: int) -> Dict[str, Any]:
    cfg = ctx.cfg
    uri = prepared_uri(cfg, stage_id, name, split)
    h = ctx.input_hash(stage_id, name, split, {"kind": "pref", "max_total_tokens": max_total_tokens})
    if (hit := _cached(ctx, uri, h)) is not None:
        return hit
    stats = AdaptStats(name, split)
    counter: Dict[str, int] = {}
    items: List[PrefPair] = list(ctx.decontaminate(ctx.registry.items(name, split, stats), counter))
    toks = _map_chunks(_w_pref, [(p.prompt_messages, p.chosen, p.rejected) for p in items], ctx.workers)
    rows = []
    for (p, c, r) in toks:
        if len(p) + max(len(c), len(r)) > max_total_tokens:            # DP-7
            stats.drop("longer_than_max_total_tokens")
        elif c == r:
            stats.drop("chosen_equals_rejected")
        else:
            rows.append({"prompt_tokens": p, "chosen_tokens": c, "rejected_tokens": r, "source": name})
    stats.check(cfg.prepared_data.max_drop_fraction)
    n, files = write_shards(ctx.storage, uri, "pref", rows, cfg.prepared_data.shard_rows)
    if n == 0:
        raise ValueError(f"dataset {name!r} split {split!r} produced no preference pairs")
    return _record(ctx, stage_id, name, split, stats, uri, n, files, h, {"decontamination": counter})


def read_system_prompt(ctx: PrepContext, kind: str) -> str:
    sp = ctx.cfg.stages.rlvr.system_prompts
    return ctx.storage.read_text(getattr(sp, kind)).strip()


def prepare_prompts_like(ctx: PrepContext, stage_id: str, name: str, split: str) -> Dict[str, Any]:
    """`prompts`, `rl_math`, `rl_code` datasets (RL-1a: rl_* prompts carry the stage's system message)."""
    cfg = ctx.cfg
    ds = cfg.datasets[name]
    uri = prepared_uri(cfg, stage_id, name, split)
    sysmsg = read_system_prompt(ctx, ds.kind) if ds.kind in ("rl_math", "rl_code") else None
    h = ctx.input_hash(stage_id, name, split, {"kind": ds.kind, "system": sysmsg})
    if (hit := _cached(ctx, uri, h)) is not None:
        return hit
    stats = AdaptStats(name, split)
    counter: Dict[str, int] = {}
    items = list(ctx.decontaminate(ctx.registry.items(name, split, stats), counter))
    prefix = [{"role": "system", "content": sysmsg}] if sysmsg is not None else []
    prompts = _map_chunks(_w_prompt, [prefix + it.messages for it in items], ctx.workers)
    rows = []
    for it, p in zip(items, prompts):
        if ds.kind == "rl_math":
            rows.append({"prompt_tokens": p, "ground_truth": it.ground_truth, "source": name})
        elif ds.kind == "rl_code":
            rows.append({"prompt_tokens": p, "tests": it.tests, "entry_point": it.entry_point, "source": name})
        else:
            rows.append({"prompt_tokens": p, "source": name})
    schema = {"rl_math": "rl_math", "rl_code": "rl_code"}.get(ds.kind, "prompts")
    n, files = write_shards(ctx.storage, uri, schema, rows, cfg.prepared_data.shard_rows)
    if n == 0:
        raise ValueError(f"dataset {name!r} split {split!r} produced no prompts")
    return _record(ctx, stage_id, name, split, stats, uri, n, files, h, {"decontamination": counter})


# --------------------------------------------------------------------------- stage preparation
def stage_dataset_plan(cfg, stage_id: str) -> List[Tuple[str, List[str]]]:
    """(dataset, splits) pairs a stage reads. Produced datasets (on_policy, teacher_traces) are
    prepared by the stage that makes them."""
    block = cfg.stage_block(stage_id)
    plan: Dict[str, List[str]] = {}
    for s in block.data.sources:
        if s.dataset in ("on_policy", "teacher_traces"):
            continue
        plan[s.dataset] = ["train", "test"]
    if stage_id in ("midtrain", "midtrain_long"):
        plan.setdefault(block.data.retention_eval_dataset, [])
        if "test" not in plan[block.data.retention_eval_dataset]:
            plan[block.data.retention_eval_dataset].append("test")
    return list(plan.items())


def prepare_stage_datasets(ctx: PrepContext, stage_id: str, seq_ctx: int) -> Dict[str, Dict[str, Any]]:
    """Prepares every declared dataset of `stage_id` (DP-1). Returns {dataset: {split: manifest entry}}
    and writes it to the stage's manifest file."""
    cfg = ctx.cfg
    out: Dict[str, Dict[str, Any]] = {}
    for name, splits in stage_dataset_plan(cfg, stage_id):
        ds = cfg.datasets[name]
        out[name] = {}
        for split in splits:
            if not ctx.registry.has_split(name, split):
                raise ValueError(f"dataset {name!r} has no {split!r} split and none can be derived (DS-4)")
            if stage_id in ("midtrain", "midtrain_long"):
                e = prepare_lm(ctx, stage_id, name, split, seq_ctx)
            elif stage_id in ("sft", "distill_offpolicy"):
                e = prepare_sft(ctx, stage_id, name, split, seq_ctx)
            elif stage_id == "preference":
                e = prepare_pref(ctx, stage_id, name, split, cfg.stages.preference.max_total_tokens)
            else:
                e = prepare_prompts_like(ctx, stage_id, name, split)
            ctx.log(f"[{stage_id}] {name}/{split}: {e['rows']} rows, dropped {e.get('dropped', {})}")
            out[name][split] = e
    merge_stage_manifest(ctx, stage_id, out)
    return out


def merge_stage_manifest(ctx: PrepContext, stage_id: str, entries: Dict[str, Dict[str, Any]]) -> None:
    uri = stage_manifest_uri(ctx.cfg, stage_id)
    cur = json.loads(ctx.storage.read_text(uri)) if ctx.storage.exists(uri) else {}
    for name, splits in entries.items():
        cur.setdefault(name, {}).update(splits)
    ctx.storage.write_text(uri, json.dumps(cur, indent=2, sort_keys=True, default=str))


def read_stage_manifest(storage: Storage, cfg, stage_id: str) -> Dict[str, Dict[str, Any]]:
    uri = stage_manifest_uri(cfg, stage_id)
    return json.loads(storage.read_text(uri)) if storage.exists(uri) else {}
