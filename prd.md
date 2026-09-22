# PRD — Foundation LLM Pretraining Pipeline (`foundation_llm`)

## 1. Purpose

Split-out, bug-fixed, multi-source rewrite of the original `MyLLM.py` Databricks notebook: a
from-scratch GPT-style foundation model pretraining pipeline (data ETL, packing, multi-source
mixture, training loop, ZeRO/FSDP distributed training, metrics/charting). This document
covers what the pipeline needs to actually run: where its data lives, what software/infra it
depends on, and what's confirmed-working vs. still open.

## 2. Goals

- Pretrain a GPT-style causal LM (dense or DeepSeek-style MLA+MoE) from a multi-source token
  corpus, with corpus-hardening (dedup/decontamination/quality filtering) ahead of training.
- Scale from a single process up to multi-GPU/multi-node via ZeRO stages 0-3 (plain DDP
  through full FSDP), selected by one CLI flag.
- Produce persisted, chartable training metrics (loss, LR, throughput, perplexity, MoE expert
  balance) rather than stdout-only logging.

## 3. Non-Goals

- Not a fine-tuning/instruction-tuning/RLHF pipeline — causal LM pretraining only.
- Not a benchmark-evaluation harness (MMLU/GSM8K/HumanEval-style accuracy) — `evaluate()`
  reports held-out LM loss/perplexity only (see `README.md`'s metrics section).
- Not a trained quality classifier — `data_quality.quality_filter_classifier` is a deliberate
  stub (needs a labeled reference-quality set this project doesn't have).

## 4. Data sources & availability

### 4.1 Storage layout (confirmed from the original notebook)

The original `MyLLM.py` reads/writes against Databricks-mounted Azure Data Lake Storage and
DBFS, e.g.:

```
abfss://root@coentus6abfsprod001.dfs.core.windows.net/data/raw/.../transformed/openwebtext/openwebtext.parquet
/dbfs/FileStore/shared_uploads/<user>/data/temp/train
/dbfs/FileStore/shared_uploads/<user>/models/toy_gpt35/
```

This confirms the pipeline runs **inside Databricks**, against **Azure Data Lake Storage
(`abfss://`)** for corpus data and **DBFS** for scratch/checkpoint output. `main.py`'s
`--out`/`--train`/`--test` defaults still point at these original paths as placeholders —
update them for wherever this actually runs.

### 4.2 Expected layout for the multi-source pipeline

`config.DATA_SOURCES` assumes every source lives under one shared "transformed" root, one
subdirectory per source, each already tokenized and packed into fixed-length blocks:

```
{data_root}/
  openwebtext/{train,test}/shard_id=*/part-*.parquet
  code/{train,test}/shard_id=*/part-*.parquet
  wikipedia/{train,test}/shard_id=*/part-*.parquet
  books/{train,test}/shard_id=*/part-*.parquet
  math/{train,test}/shard_id=*/part-*.parquet
```

Each `part-*.parquet` has a `tokens` column (`array<int64>`, one packed block per row) and
optionally a `doc_start` column (`array<bool>`, same length, from `packing.py`'s EOS-insertion
step) — see `data.ParquetTokenDataset`.

### 4.3 Source-by-source status

| Source | Subdir | Status | Notes |
|---|---|---|---|
| Web (OpenWebText) | `openwebtext` | **Confirmed** | Original raw path in `MyLLM.py`; already flowing through the pipeline before this rewrite |
| Code | `code` | **Assumed** available under the same transformed root, per your instruction to build the mixture against it | Raw pre-packing source path not yet supplied — `packing.RAW_SOURCE_PATHS` has a placeholder |
| Wikipedia | `wikipedia` | **Assumed** | Same — placeholder path |
| Books | `books` | **Assumed** | Same — placeholder path |
| Math | `math` | **Assumed** | Same — placeholder path |

**Action needed**: confirm the actual raw (pre-packing) source paths for code/wikipedia/books/
math and fill them into `packing.py`'s `RAW_SOURCE_PATHS` dict, then run
`packing.pack_source_to_shards` once per source to materialize the layout in §4.2. Until that
happens, `--data_root` will fail with `FileNotFoundError` for any source not yet packed —
`--sources web` (or whichever subset actually exists) works as a subset filter in the
meantime.

### 4.4 Eval/decontamination sets

`data_quality.decontaminate_against_eval_sets` takes `eval_texts` (the raw text of benchmark
questions/contexts — MMLU, GSM8K, HumanEval, etc.) as a Python list loaded into the driver.
**No source or location for these is specified yet** — this needs to be sourced (e.g. the
public HuggingFace dataset versions of each benchmark) before decontamination can actually run.

## 5. Dependencies

### 5.1 Python packages — confirmed by actually installing and running them in this session

| Package | Version tested here | Used by | Verified how |
|---|---|---|---|
| `torch` | 2.14.0+cu130 | `model.py`, `moe.py`, `train.py`, `distributed.py`, `main.py` | Full forward/backward/optimizer-step runs, single- and multi-process |
| `pyarrow` | 25.0.1 | `data.py` (`ParquetTokenDataset`) | Real synthetic Parquet shards read end-to-end |
| `matplotlib` | 3.10.8 | `plot_metrics.py` | Real PNG charts generated from a real training run's `metrics.jsonl` |
| `tiktoken` | 0.14.0 | `main.py` (`cl100k_base` tokenizer, vocab size, EOT token id) | Installed here — **but see §5.3, the actual encoding load failed in this sandbox** |

### 5.2 Python packages — required but NOT available/testable in this sandbox

| Package | Used by | Why untested here |
|---|---|---|
| `pyspark` | `packing.py`, `data_quality.py` | No Spark cluster in this sandbox; these files are syntax-checked only, not executed |
| `pyspark.ml` (`Tokenizer`, `NGram`, `HashingTF`, `MinHashLSH`) | `data_quality.near_dedup_minhash` | Same |

These are provided by the Databricks runtime in the target environment — not something to
`pip install` separately there, but flagged here because they were never actually run against
real data in this project's development.

### 5.3 Confirmed real dependency risk: `tiktoken`'s network call

`tiktoken.get_encoding("cl100k_base")` is not purely local — on first use it fetches the BPE
merge file from **`https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`**
(then caches it locally). Installing and running this exact call in this sandbox produced:

```
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url:
https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
```

This sandbox's outbound network is allowlisted to a small set of domains and blocked this one
— which is exactly the failure mode a locked-down corporate network (proxy/firewall-restricted
egress, as Databricks jobs often run behind) would produce. **Before `main.py` can run for
real, confirm one of:**
- the target environment's egress allowlist includes `openaipublic.blob.core.windows.net`, or
- the `.tiktoken` file is pre-fetched once (from a machine with open internet) and placed in
  tiktoken's local cache directory / `TIKTOKEN_CACHE_DIR`, or mirrored to an internally
  reachable blob store and loaded via `tiktoken.Encoding(...)` with a custom loader instead of
  `get_encoding`.

This is not a hypothetical — it reproduced on the first real attempt to exercise this
dependency in this session.

### 5.4 Infrastructure dependencies

| Requirement | Needed for | Notes |
|---|---|---|
| Databricks workspace + Spark cluster | ETL: `packing.py`, `data_quality.py` | Original notebook's `spark.conf.set(...)` and `# Databricks notebook source` header confirm this is the intended runtime |
| Azure Data Lake Storage (`abfss://`) access | Reading/writing corpus data | Path pattern confirmed in §4.1 |
| GPU(s) | Any real training run (`main.py`) | `PRESETS` up to `6.7B`; CPU fallback exists in code but is not a realistic training path at these sizes |
| Multi-GPU / multi-node + `torchrun` | `--zero_stage 1/2/3` (ZeRO/FSDP) | Single-process runs skip distributed wrapping entirely; see `distributed.py` |
| Outbound network access to `openaipublic.blob.core.windows.net` (or a pre-cached/mirrored tokenizer file) | Tokenizer load | See §5.3 — confirmed blocking failure mode |

## 6. Functional scope (what's implemented)

- **Data ETL**: exact + near-duplicate dedup, benchmark decontamination, Gopher/C4-style
  quality-heuristic filtering (`data_quality.py`); EOS-separated packing with document-boundary
  tracking and corrected global ordering (`packing.py`).
- **Data loading**: worker-safe, padding-safe streaming Parquet dataset; weighted multi-source
  mixture with upsampling for small high-quality sources (`data.py`).
- **Model**: dense RoPE + standard attention baseline, or DeepSeek-style Multi-head Latent
  Attention + DeepSeekMoE (aux-loss-free load balancing), selectable per run (`model.py`,
  `moe.py`).
- **Training**: gradient accumulation, mixed precision, cosine LR schedule, checkpointing with
  correct step/best-loss propagation (`train.py`, `scheduler.py`).
- **Distributed**: ZeRO stages 0-3 via DDP / `ZeroRedundancyOptimizer` / FSDP, with
  collective-safe checkpointing (`distributed.py`).
- **Observability**: structured JSONL metrics logging and chart rendering (`metrics.py`,
  `plot_metrics.py`).

## 7. Testing status (see `README.md` for full detail)

- **Executed in this sandbox**: model forward/backward (both archs), MoE routing/bias update,
  data padding/mixture-sampling logic, a full single-process `train_loop` → checkpoint → resume
  cycle, real chart generation from a real training run, and a real 2-process ZeRO/FSDP
  distributed run (all 4 stages, including checkpoint round-trip).
- **Not executed anywhere yet**: `packing.py`/`data_quality.py` against a real Spark cluster,
  `main.py` end-to-end against real corpus data (blocked today by §5.3's tokenizer network
  issue), any GPU/NCCL run, any run beyond 2 processes.

## 8. Open risks / action items

1. Confirm raw source paths for code/wikipedia/books/math and run `packing.py` against them (§4.3).
2. Resolve `tiktoken`'s network dependency for the target environment (§5.3) — this blocks
   `main.py` from running at all until addressed one way or another.
3. Source eval-set texts for `decontaminate_against_eval_sets` (§4.4).
4. Run `packing.py`/`data_quality.py` against a real Spark cluster with a small sample before
   trusting them at full corpus scale (never executed against real Spark).
5. Tune `DATA_SOURCES` mixture weights via small-scale ablations — current values are
   starting points, not measured.
6. Train (or otherwise obtain) a real quality classifier before `quality_filter_classifier`
   can be used — it currently raises `NotImplementedError` by design.
7. Verify ZeRO stages 2/3 (FSDP) on real GPU/NCCL hardware — only CPU/gloo verified here.
