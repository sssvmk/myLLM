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
- Not a full benchmark-evaluation harness — `benchmark_eval.py` implements MMLU (log-likelihood
  scoring needs only a forward pass) but not GSM8K/HumanEval, which need an autoregressive
  generation loop this codebase doesn't have (see §6). `train.evaluate()` separately reports
  held-out LM loss/perplexity, the standard pretraining-time signal, regardless.

## 4. Data sources & availability

**Everything in this section lives in `config.py`, not in this document alone** — this is the
prose version of what `config.py`'s `RAW_SOURCE_PATHS`, `TOKENIZER_SOURCES`, and
`EVAL_BENCHMARK_SOURCES` dicts already state as data. If the two ever disagree, `config.py` is
the source of truth; update this section to match it, not the other way around.

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

### 4.3 Where each raw (pre-packing) source is available for download

`config.RAW_SOURCE_PATHS` is the single config entry this maps to — each source has a real,
verified public download location plus its license, and an empty `raw_path` field that YOU
fill in once you've downloaded (or internally mirrored) it. `packing.pack_all_configured_sources`
reads this dict directly, so filling in `raw_path` is the only step needed to make packing
runnable against a given source — nothing else in the code changes.

| Source | `RAW_SOURCE_PATHS` key | Where to get it | License note |
|---|---|---|---|
| Web (OpenWebText) | `web` | `huggingface.co/datasets/Skylion007/openwebtext` (homepage: `skylion007.github.io/OpenWebTextCorpus`) | Unspecified by curator — an open replication of OpenAI's WebText corpus; review before commercial use |
| Code | `code` | `huggingface.co/datasets/bigcode/the-stack-v2` (homepage: `bigcode-project.org`) | Per-file license from source repos, opt-out-respecting — see the dataset card |
| Wikipedia | `wikipedia` | `huggingface.co/datasets/wikimedia/wikipedia` (raw dumps: `dumps.wikimedia.org`) | CC BY-SA 4.0 / GFDL |
| Books | `books` | `huggingface.co/datasets/deepmind/pg19` (source: `gutenberg.org`) | Public domain (pre-1919 Project Gutenberg texts) |
| Math | `math` | `huggingface.co/datasets/open-web-math/open-web-math` | ODC-By 1.0 |

**Status**: Web is the only source already confirmed flowing through the original pipeline
(it's the literal path in `MyLLM.py`). The other four now have real, verified public sources
documented in `config.py` — what's still open is the mechanical step of downloading each one
and writing its local/internal path into `RAW_SOURCE_PATHS[name]["raw_path"]`.
`--sources web` (or whichever subset has `raw_path` filled in) works as a subset filter for
`main.py` in the meantime; `pack_all_configured_sources` skips any source with an empty
`raw_path` and prints which ones it skipped, rather than failing the whole run.

### 4.4 Eval/decontamination benchmark sources

`config.EVAL_BENCHMARK_SOURCES` documents where each standard pretraining-era benchmark is
publicly available, used by both `data_quality.decontaminate_against_eval_sets` (raw
question/context text, to exclude near-matches from training data) and `benchmark_eval.py`
(same text, plus answers, to actually score the model where implemented):

| Benchmark | `EVAL_BENCHMARK_SOURCES` key | HF dataset | Scoring implemented? |
|---|---|---|---|
| MMLU | `mmlu` | `cais/mmlu` (config `"all"`) | **Yes** — `benchmark_eval.evaluate_mmlu`, log-likelihood over answer choices, no generation needed |
| GSM8K | `gsm8k` | `openai/gsm8k` (config `"main"`) | No — needs free-form generation + numeric-answer extraction (this codebase has no sampling loop) |
| HumanEval | `humaneval` | `openai/openai_humaneval` | No — needs generation + sandboxed code execution for pass@k |

Each entry's `local_path` (empty by default) mirrors the tokenizer's local-first pattern: fill
it in with a downloaded copy for environments without direct HuggingFace access at load/eval
time, following the conversion shown in `benchmark_eval.load_local_mmlu`'s docstring.

### 4.5 Quality classifier source

`config.QUALITY_CLASSIFIER` points `data_quality.quality_filter_classifier` at
**`huggingface.co/HuggingFaceFW/fineweb-edu-classifier`** — the actual, real, publicly
released model HuggingFace used to build the FineWeb-Edu dataset (Apache 2.0, 109M params,
built on Snowflake-arctic-embed-m, scores 0-5 for educational value, trained on 450k
Llama3-70B-Instruct-annotated web samples; `score_threshold=3.0` per the model card's own
keep/remove cutoff). This is a real pretrained model, not something that needs training from
scratch or a reference-quality dataset assembled in-house — `local_model_path` follows the
same local-first pattern as the tokenizer for environments without direct HF access.

## 5. Dependencies

### 5.1 Python packages — confirmed by actually installing and running them in this session

| Package | Version tested here | Used by | Verified how |
|---|---|---|---|
| `torch` | 2.14.0+cu130 | `model.py`, `moe.py`, `train.py`, `distributed.py`, `main.py` | Full forward/backward/optimizer-step runs, single- and multi-process |
| `pyarrow` | 25.0.1 | `data.py` (`ParquetTokenDataset`) | Real synthetic Parquet shards read end-to-end |
| `matplotlib` | 3.10.8 | `plot_metrics.py` | Real PNG charts generated from a real training run's `metrics.jsonl` |
| `tiktoken` | 0.14.0 | `tokenizer.py` (`cl100k_base` vocab, EOT token id) | Installed here — **but see §5.3, the network-dependent `get_encoding` path failed in this sandbox; the offline `tokenizer.py` path is tested and doesn't hit this** |
| `transformers` | (installed, version not pinned here) | `data_quality.quality_filter_classifier` (FineWeb-Edu classifier) | Scoring *mechanism* tested with a local, zero-download BERT config (see `tests/test_quality_classifier_smoke.py`); the real `HuggingFaceFW/fineweb-edu-classifier` weights themselves were not loadable in this sandbox — no `huggingface.co` access |
| `tensorboard` | (installed, version not pinned here) | `metrics.py`'s `--tensorboard` path | Real event files written and read back with `EventAccumulator` in this session — see §7 |

### 5.2 Python packages — required but NOT available/testable in this sandbox

| Package | Used by | Why untested here |
|---|---|---|
| `pyspark` | `packing.py`, `data_quality.py` | No Spark cluster in this sandbox; these files are syntax-checked only, not executed |
| `pyspark.ml` (`Tokenizer`, `NGram`, `HashingTF`, `MinHashLSH`) | `data_quality.near_dedup_minhash` | Same |

These are provided by the Databricks runtime in the target environment — not something to
`pip install` separately there, but flagged here because they were never actually run against
real data in this project's development.

### 5.3 Confirmed real dependency risk — and its fix — for `tiktoken`

`tiktoken.get_encoding("cl100k_base")` is not purely local — on first use it fetches the BPE
merge file from **`https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`**
(then caches it locally). Installing and running this exact call in this sandbox produced:

```
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url:
https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
```

This sandbox's outbound network is allowlisted to a small set of domains and blocked this one
— exactly the failure mode a locked-down corporate network (proxy/firewall-restricted egress,
as Databricks jobs often run behind) would produce.

**Fixed**: `tokenizer.py`'s `load_cl100k_encoding(local_path=...)` builds the encoding by hand
from a local ranks file — `tiktoken.load.load_tiktoken_bpe` treats any path without `"://"` as
a local file and never touches the network for it (confirmed from `tiktoken`'s own source and
tested end-to-end in this sandbox with a synthetic local ranks file: loaded, built into a real
`tiktoken.Encoding`, and successfully encoded/decoded text with zero network calls). `main.py`
now calls this via `--tokenizer_path`; omit it to fall back to the old network-dependent
`tiktoken.get_encoding` behavior.

**What's still needed**: the actual `cl100k_base.tiktoken` ranks file itself (1.68 MB), which
this loader validates against its known SHA-256 (`223921b7...`) before using — get it once
from any machine with normal internet access (the official URL above, or a verified mirror —
several exist on `huggingface.co`, e.g. in the `microsoft/Phi-3-small-8k-instruct` repo) and
place it wherever this pipeline runs. **Note**: mirrors are themselves hosted on generic
file-sharing platforms that may be blocked by the same kind of network policy that blocked the
official URL — the robust path is a one-time manual transfer (download once from any
unrestricted machine, copy the file in), not assuming any particular mirror domain is reachable.

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
  quality-heuristic filtering, and a real FineWeb-Edu-classifier-based quality filter
  (`data_quality.py`); EOS-separated packing with document-boundary tracking, corrected global
  ordering, and config-driven multi-source packing (`packing.py`).
- **Data loading**: worker-safe, padding-safe streaming Parquet dataset; weighted multi-source
  mixture with upsampling for small high-quality sources and CLI-overridable mixture weights
  (`data.py`, `config.apply_weight_overrides`).
- **Model**: dense RoPE + standard attention baseline, or DeepSeek-style Multi-head Latent
  Attention + DeepSeekMoE (aux-loss-free load balancing), selectable per run (`model.py`,
  `moe.py`).
- **Training**: gradient accumulation, mixed precision, cosine LR schedule, checkpointing with
  correct step/best-loss propagation (`train.py`, `scheduler.py`).
- **Distributed**: ZeRO stages 0-3 via DDP / `ZeroRedundancyOptimizer` / FSDP, with
  collective-safe checkpointing (`distributed.py`).
- **Observability**: structured JSONL metrics logging, optional TensorBoard, and chart
  rendering (`metrics.py`, `plot_metrics.py`).
- **Benchmark evaluation**: MMLU (log-likelihood scoring, implemented and tested); GSM8K/
  HumanEval documented as needing generation infra this codebase doesn't have yet
  (`benchmark_eval.py`).
- **Offline tokenizer loading**: bypasses `tiktoken`'s hardcoded network fetch entirely given
  a local ranks file (`tokenizer.py`).

## 7. Testing status (see `README.md` for full detail)

- **Executed in this sandbox**: model forward/backward (both archs), MoE routing/bias update,
  data padding/mixture-sampling logic, `config.apply_weight_overrides` (override, partial
  override, and error cases), the offline tokenizer loader (synthetic local ranks file, zero
  network calls), `benchmark_eval.evaluate_mmlu` against a real `GPTModel`, the
  `quality_filter_classifier` scoring mechanism (zero-download local BERT), a full
  single-process `train_loop` → checkpoint → resume cycle (including `--tensorboard`), real
  chart generation from a real training run, and a real 2-process ZeRO/FSDP distributed run
  (all 4 stages, including checkpoint round-trip).
- **Not executed anywhere yet**: `packing.py`/`data_quality.py` against a real Spark cluster,
  the real `fineweb-edu-classifier` model weights (no `huggingface.co` access in this
  sandbox), `main.py` end-to-end against real corpus data (blocked today by §5.3's tokenizer
  file not yet being placed, and by §4.3's raw source paths not yet being filled in), any
  GPU/NCCL run, any run beyond 2 processes.

## 8. Open risks / action items

1. Fill in `config.RAW_SOURCE_PATHS[name]["raw_path"]` for code/wikipedia/books/math once
   downloaded from the sources documented in §4.3, then run
   `packing.pack_all_configured_sources` against them.
2. Get the `cl100k_base.tiktoken` ranks file (§5.3, §4.5's tokenizer entry) onto whatever
   machine runs this pipeline and set `config.TOKENIZER_SOURCES["cl100k_base"]["local_path"]`
   (or pass `--tokenizer_path`) — the offline-loading code is done and tested; only the
   one-time file transfer remains.
3. Run `packing.py`/`data_quality.py` against a real Spark cluster with a small sample before
   trusting them at full corpus scale (never executed against real Spark).
4. Verify the real `HuggingFaceFW/fineweb-edu-classifier` weights load and score sensibly on
   real text — only the surrounding scoring mechanism is verified here, not the actual model.
5. If GSM8K/HumanEval scoring is wanted, the prerequisite is an autoregressive generation loop
   (KV-cached sampling) — `benchmark_eval.py` documents this gap; MMLU doesn't need it and is
   already implemented.
6. Verify ZeRO stages 2/3 (FSDP) on real GPU/NCCL hardware — only CPU/gloo verified here.
7. `DATA_SOURCES` mixture weights are still starting-point guesses — `--mixture_weights` makes
   running ablations a CLI flag rather than a code edit, but the ablations themselves (i.e.
   actually finding better weights) haven't been run.
