# foundation_llm

A from-scratch GPT-style **foundation model pretraining pipeline**: corpus preparation (dedup,
decontamination, quality filtering, packing), a multi-source training-data mixture, two
selectable model architectures, single-to-multi-GPU distributed training (ZeRO/FSDP), and
metrics/checkpointing. It's a bug-fixed, extended rewrite of an original single-file Databricks
notebook (`MyLLM.py`), split into one module per concern.

This document covers what the project does, how it's put together, how to size a model and
read/change its configuration, where to get training data, and how to actually run it.
`prd.md` covers dependencies, infrastructure requirements, and the current open items in more
formal detail — this file is the "how do I use this" reference.

---

## 1. What this does

Given a tokenized, multi-source text corpus, this trains a decoder-only transformer language
model from randomly-initialized weights using the standard causal next-token-prediction
objective — the same mechanics behind GPT-2/3, LLaMA, DeepSeek, and similar models. It is not
fine-tuning: there's no chat template, no instruction data, no starting checkpoint. It is not
a benchmark harness, though it can score MMLU. The pieces:

1. **Corpus hardening** (`data_quality.py`) — exact + near-duplicate removal, benchmark
   decontamination, heuristic and model-based quality filtering — run once per raw source,
   before packing.
2. **Packing** (`packing.py`) — tokenizes and concatenates documents into fixed-length blocks
   with document-boundary tracking, ready for streaming.
3. **Training-data mixture** (`data.py`, `config.DATA_SOURCES`) — samples blocks from several
   sources at configured ratios, upsampling small high-quality sources.
4. **Model** (`model.py`, `moe.py`) — a dense RoPE transformer, or a DeepSeek-style
   Multi-head Latent Attention + Mixture-of-Experts variant.
5. **Training loop** (`train.py`, `scheduler.py`, `distributed.py`) — gradient accumulation,
   mixed precision, cosine LR schedule, checkpointing, and ZeRO/FSDP distributed scaling.
6. **Observability** (`metrics.py`, `plot_metrics.py`, `benchmark_eval.py`) — structured
   metrics (JSONL + optional TensorBoard), chart rendering, and MMLU-style evaluation.

---

## 2. Architecture

Both variants share the same skeleton: a token embedding, `n_layer` pre-norm transformer
blocks, a final `RMSNorm`, and a weight-tied `lm_head`. They differ in what each block's
attention and feed-forward sub-layers are.

### 2.1 Dense (`--arch dense`, the default)

- **Attention**: standard multi-head causal self-attention with **RoPE** (rotary position
  embeddings) applied to queries and keys — no learned absolute position embedding.
- **Feed-forward**: a single dense two-layer GELU MLP (`d_model -> d_ff -> d_model`).
- This is the GPT-2/LLaMA-style baseline: every parameter is used on every token.

### 2.2 DeepSeek-style (`--arch deepseek`)

- **Attention**: **Multi-head Latent Attention (MLA)** — keys and values are compressed
  through a shared low-rank latent (`d_latent`, default `d_model // 8`) instead of being
  computed at full width per head, plus a small decoupled RoPE-only path
  (`d_rope`, default 32) concatenated on. Queries go through the same kind of low-rank
  down/up projection. This is a *structurally faithful* reproduction (correct shapes and
  information flow) of DeepSeek-V2/V3's attention — not an exact hyperparameter match, and it
  does not implement incremental KV-cache serving (this codebase only ever does full-sequence
  training forward passes).
- **Feed-forward**: **DeepSeekMoE** — `n_shared_experts` dense SwiGLU experts that run on
  every token, plus `n_routed_experts` narrower SwiGLU experts of which only the top
  `moe_top_k` fire per token. Load balancing is **auxiliary-loss-free**: a per-expert bias
  (not part of the gradient) is nudged up or down after every optimizer step based on that
  expert's usage, following DeepSeek-V3's scheme, rather than a traditional load-balancing
  loss term that trades off against language-modeling quality.

**Why this matters for sizing**: in the dense architecture every parameter contributes to
every token's compute. In the MoE architecture, most parameters (the routed experts not
selected for a given token) sit idle for that token — see §4.2's "total vs. active
parameters" distinction, which is the standard way MoE model sizes are reported (e.g.
"671B total, 37B active" for DeepSeek-V3).

### 2.3 Document-boundary attention masking

Both architectures accept an optional `attn_mask` built by `model.build_doc_attention_mask`
from a `doc_start` flag array (produced by `packing.py`'s EOS-insertion step). This prevents a
packed training block — which is often several concatenated documents — from letting attention
cross from one document into an unrelated one. Toggle with `--use_doc_mask` (on by default).

---

## 3. How training works, stage by stage

```
raw corpus (per source)
   │  data_quality.py: exact_dedup → near_dedup_minhash → quality_filter_heuristics
   │                    → quality_filter_classifier → decontaminate_against_eval_sets
   ▼
packing.py: tokenize, insert EOS between documents, pack into fixed-length blocks,
            track doc_start, write sharded Parquet under {data_root}/{source}/{split}/
   ▼
data.py: ParquetTokenDataset (per source, worker-safe, padding-safe)
         → MixtureIterableDataset (weighted sampling across sources, per config.DATA_SOURCES)
   ▼
main.py: build model (dense or deepseek), wrap for ZeRO/FSDP if distributed,
         build optimizer, cosine LR scheduler
   ▼
train.py: train_loop — per micro-batch: forward, loss (next-token cross-entropy,
          padding ignored), backward; every `accum` micro-batches: grad clip,
          optimizer step, LR step; every `ckpt_every` steps: checkpoint;
          every `eval_every` steps (and always before exit): held-out loss/perplexity
   ▼
metrics.jsonl (+ optional TensorBoard) → plot_metrics.py charts
checkpoints under {out}/ (+ {out}/latest/latest_check.pt for --resume)
```

**Distributed scaling** (`distributed.py`) is selected with `--zero_stage`, mapped directly to
DeepSpeed/PyTorch's ZeRO stages:

| `--zero_stage` | Mechanism | What's sharded across GPUs |
|---|---|---|
| 0 | Plain `DistributedDataParallel` | Nothing — every rank holds a full copy of params/grads/optimizer state |
| 1 | DDP + `ZeroRedundancyOptimizer` | Optimizer state only |
| 2 | FSDP, `SHARD_GRAD_OP` | Optimizer state + gradients |
| 3 | FSDP, `FULL_SHARD` | Optimizer state + gradients + parameters |

This only activates under `torchrun` (multiple processes); a single-process run gets the model
and optimizer unwrapped, since there's nothing to shard across one process.

---

## 4. Estimating parameter count

### 4.1 Dense architecture

Every dense block has: QKV projection + output projection in attention (`4·d_model²`), a
two-layer MLP (`2·d_model·d_ff`), and two small RMSNorm weight vectors (`2·d_model`, usually
negligible). Add the (weight-tied) token embedding and a final norm:

```
total_params = vocab_size · d_model
             + n_layer · (4·d_model² + 2·d_model·d_ff + 2·d_model)
             + d_model
```

This formula was verified against the actual model code in this session (exact match, not
approximate) for multiple configurations.

### 4.2 DeepSeek architecture — total vs. active parameters

MLA and DeepSeekMoE both add more terms; the important conceptual split is that MoE gives you
two different "size" numbers:

- **Total parameters**: every routed expert counts, whether or not it fires for a given token.
- **Active parameters**: only `moe_top_k` of the `n_routed_experts` routed experts (plus all
  `n_shared_experts` shared experts, plus the attention/embedding parameters) actually run on
  any given token. This is the number that determines per-token compute (FLOPs), while total
  parameters determines memory footprint.

Both were also verified exact-match against the model code in this session.

### 4.3 Verified numbers for the built-in presets

Computed with `vocab_size = 100,278` (cl100k_base + the reserved pad id) and the default
DeepSeek knobs (`n_routed_experts=8, n_shared_experts=1, moe_top_k=2, d_rope=32`):

| Preset | `d_model` | `n_layer` | `n_heads` | `d_ff` | Dense params | DeepSeek total | DeepSeek active |
|---|---|---|---|---|---|---|---|
| `125M` | 768 | 12 | 12 | 3072 | 161,967,360 | 474,797,664 | 219,993,696 |
| `355M` | 1024 | 12 | 16 | 4096 | 253,705,216 | 809,814,112 | 356,829,280 |
| `1.3B` | 2048 | 24 | 16 | 8192 | 1,413,429,248 | 5,833,582,784 | 2,209,704,128 |
| `6.7B` | 4096 | 32 | 32 | 16384 | 6,853,455,872 | 30,426,525,952 | 11,099,173,120 |

**Preset names are conventional labels (in the GPT-2/GPT-3 naming tradition), not exact
counts** — the dense `"125M"` preset is actually ~162M params, `"6.7B"` is actually ~6.85B.
Note how much bigger the DeepSeek *total* column is than dense at the same `(d_model, n_layer,
d_ff)` — routed experts add parameters fast; the *active* column is the fairer
apples-to-apples comparison against the dense column for "how much compute does this cost."

### 4.4 Training memory, not just parameter memory

Parameter count alone understates training memory. Per the lecture material this project's
`--zero_stage` options are modeled on: full (unsharded) mixed-precision training needs
roughly **16 bytes per parameter** — 2 (bf16 weights) + 2 (bf16 grads) + 4 (fp32 master
weights) + 4 + 4 (fp32 Adam first/second moments) — before accounting for activations. ZeRO
stage 1/2/3 divides progressively more of that 16 bytes/param across the data-parallel group
(see §3's table); activation memory is separate again and scales with batch size, sequence
length, and (for the dense architecture) quadratically with sequence length unless recomputed.

---

## 5. Configuration reference (`config.py`)

`config.py` is the single place every "what size is this model" and "where does this data come
from" question resolves to.

| Section | What it holds |
|---|---|
| `PRESETS` | `d_model`, `n_layer`, `n_heads`, `d_ff` per named size (`125M`/`355M`/`1.3B`/`6.7B`) — see §6 for how changing these changes model size |
| `TOKENIZER_SOURCES` | `cl100k_base`'s official URL, mirrors, sha256, and `local_path` for fully offline loading (`tokenizer.py`) |
| `RAW_SOURCE_PATHS` | Per-source download location + license + `raw_path` (where you've put it after downloading) — see §7 |
| `DATA_SOURCES` | Per-source `subdir`, mixture `weight`, and `cycle` (upsample) flag — see §6.3 |
| `apply_weight_overrides(data_sources, overrides_str)` | Parses `--mixture_weights` into a modified `DATA_SOURCES`-shaped list, for ablations without editing this file |
| `EVAL_BENCHMARK_SOURCES` | MMLU/GSM8K/HumanEval HF dataset paths, for decontamination and `benchmark_eval.py` |
| `QUALITY_CLASSIFIER` | The FineWeb-Edu classifier model id, threshold, and local-path override |
| `parse_args()` | The full CLI — every flag below is defined here |

The CLI groups roughly into: **paths** (`--out`, `--data_root`, `--sources`,
`--mixture_weights`, `--tokenizer_path`), **architecture** (`--model`, `--arch`, `--ctx`,
`--rope_theta`, the `--d_latent`/`--d_rope`/`--n_routed_experts`/`--n_shared_experts`/
`--moe_top_k` DeepSeek-only knobs, `--use_doc_mask`), **optimization** (`--batch-tokens`,
`--accum`, `--lr`, `--min_lr`, `--warmup`, `--max_steps`, `--weight_decay`, `--grad_clip`,
`--dropout`, `--seed`, `--use_bfloat`), **runtime** (`--compile`, `--zero_stage`,
`--num_workers`, `--ckpt_every`, `--eval_every`, `--log_interval`, `--resume`), and
**observability** (`--tensorboard`, `--eval_benchmarks`).

---

## 6. How configuration changes affect model size

### 6.1 The four architecture knobs, and which matters most

From §4.1's formula, `total_params ≈ vocab_size·d_model + n_layer·(4·d_model² + 2·d_model·d_ff)`:

| Change | Effect on param count | Why |
|---|---|---|
| `n_layer` ↑ | **Linear** — doubling `n_layer` roughly doubles total params | Every layer is an independent, full-size block |
| `d_model` ↑ | **Roughly quadratic** (the `4·d_model²` attention term dominates at typical sizes) | Attention projections are `d_model × d_model`-shaped; a 2x increase in `d_model` is a ~4x increase in attention params alone |
| `d_ff` ↑ | **Linear**, and only affects the MLP term (`2·d_model·d_ff`) | The MLP hidden width scales the two linear layers proportionally |
| `n_heads` | **No direct effect on param count** (attention param count only depends on `d_model`) — but must evenly divide `d_model`, and changes `head_dim = d_model / n_heads` | More heads = more, narrower attention heads at the same total width |

This is why `d_model` is the single highest-leverage knob for model size, and why the built-in
presets scale it fastest (768 → 1024 → 2048 → 4096) while scaling `n_layer` more gradually
(12 → 12 → 24 → 32).

### 6.2 DeepSeek-only knobs

| Change | Effect |
|---|---|
| `n_routed_experts` ↑ | **Linear increase in total params** (more expert copies), but **no change in active params** if `moe_top_k` stays fixed — this is the "make it bigger without making it slower per token" lever MoE exists for |
| `moe_top_k` ↑ | **Linear increase in active params** (more experts fire per token) — trades speed for per-token capacity, total params unchanged |
| `n_shared_experts` ↑ | Increases both total and active params equally (shared experts always run) |
| `d_latent` ↑ (0 = auto `d_model // 8`) | Increases MLA's parameter count and expressiveness; DeepSeek-V2/V3 use a similar `d_model // 8`-ish ratio, so the default is a reasonable starting point, not an arbitrary placeholder |
| `d_rope` ↑ | Small, roughly linear increase (it only sizes the decoupled RoPE path, not the main content path) |

### 6.3 Data-side "size" knobs (don't change the model, change what it sees)

`DATA_SOURCES`' `weight` values and `--ctx` don't change parameter count at all, but they're
worth knowing here because they're the other axis of "making the run bigger": `--ctx` (context
length) increases activation memory roughly linearly (or quadratically for dense attention
without recomputation — see §4.4), and increasing `--batch-tokens` / `--accum` changes the
effective batch size, not the model.

---

## 7. Datasets — where to download

All five configured sources have real, verified public download locations, held in
`config.RAW_SOURCE_PATHS` (this table is the prose version of that dict — if they disagree,
the dict is correct):

| Source | Where to get it | License note |
|---|---|---|
| Web (OpenWebText) | `huggingface.co/datasets/Skylion007/openwebtext` | Unspecified by curator — review before commercial use |
| Code | `huggingface.co/datasets/bigcode/the-stack-v2` | Per-file license from source repos, opt-out-respecting |
| Wikipedia | `huggingface.co/datasets/wikimedia/wikipedia` | CC BY-SA 4.0 / GFDL |
| Books | `huggingface.co/datasets/deepmind/pg19` | Public domain (pre-1919 Project Gutenberg texts) |
| Math | `huggingface.co/datasets/open-web-math/open-web-math` | ODC-By 1.0 |

Download each one, fill in its `raw_path` under `config.RAW_SOURCE_PATHS[name]`, then run
`packing.pack_all_configured_sources(spark, ctx, out_root)` — it reads the config directly and
skips (with a clear message) any source whose `raw_path` is still empty, so filling paths in
one at a time is fine.

The tokenizer file and evaluation-benchmark datasets have their own sources documented in
`config.TOKENIZER_SOURCES` and `config.EVAL_BENCHMARK_SOURCES` respectively — see §5's table
and `prd.md` §4.4-4.5 for the full writeup (official URL + mirrors + hash for the tokenizer;
HF dataset ids for MMLU/GSM8K/HumanEval).

---

## 8. Code files

| File | Contents |
|---|---|
| `config.py` | `PRESETS`, `DATA_SOURCES` + `apply_weight_overrides`, `TOKENIZER_SOURCES`, `RAW_SOURCE_PATHS`, `EVAL_BENCHMARK_SOURCES`, `QUALITY_CLASSIFIER`, CLI (`parse_args`) |
| `tokenizer.py` | Fully offline `cl100k_base` loader (`load_cl100k_encoding`), bypassing tiktoken's hardcoded network fetch |
| `model.py` | `RMSNorm`, RoPE helpers, dense `CausalSelfAttention`, `MultiHeadLatentAttention` (MLA), `Block`, `GPTModel`, `build_doc_attention_mask` |
| `moe.py` | `DeepSeekMoE` (routed + shared experts, aux-loss-free bias balancing) |
| `data.py` | `ParquetTokenDataset` (padding + worker-safety + doc_start), `MixtureIterableDataset`, `build_mixture_dataset` |
| `data_quality.py` | Spark dedup / decontamination / quality-filter utilities, including the real FineWeb-Edu-classifier-based filter |
| `packing.py` | Spark packing: EOS insertion, `doc_start` tracking, fixed global ordering, `pack_all_configured_sources` |
| `scheduler.py` | `CosineLRScheduler` |
| `metrics.py` | `MetricsLogger` — JSONL always, TensorBoard optionally (`--tensorboard`) |
| `plot_metrics.py` | Reads `metrics.jsonl`, renders loss/LR/throughput/perplexity/MoE-usage PNG charts |
| `distributed.py` | ZeRO / FSDP integration (stages 0-3) — `setup_distributed`, `wrap_model`, `build_optimizer_for_zero`, collective-safe checkpoint helpers |
| `benchmark_eval.py` | `evaluate_mmlu` (implemented); GSM8K/HumanEval documented as needing generation infra not yet present |
| `train.py` | Training loop, eval, checkpoint save/load, optimizer construction |
| `main.py` | Driver (`drive(args)`), CLI entry point |
| `tests/test_model_smoke.py` | Synthetic-data smoke test for both architectures |
| `tests/test_distributed_smoke.py` | Real 2-process ZeRO/FSDP smoke test (all 4 stages) |
| `tests/test_quality_classifier_smoke.py` | Network-free scoring-mechanism test |

---

## 9. How to deploy / run this

### 9.1 Setup (one-time)

```bash
pip install torch pyarrow tiktoken transformers matplotlib tensorboard
# pyspark is provided by the Databricks runtime this is designed for -- only needed for
# packing.py / data_quality.py, not for training itself.
```

1. Get the tokenizer file and set its path (§5, §7): download
   `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`, verify its
   sha256 matches `config.TOKENIZER_SOURCES["cl100k_base"]["sha256"]`, then either set
   `local_path` in that config entry or pass `--tokenizer_path /path/to/cl100k_base.tiktoken`
   on every run.
2. Get at least one data source (§7), pack it (see §3's pipeline), and confirm
   `{data_root}/{source}/{train,test}/shard_id=*/part-*.parquet` exists.

### 9.2 Single-process run

```bash
python main.py \
  --data_root /path/to/transformed \
  --sources web \
  --tokenizer_path /path/to/cl100k_base.tiktoken \
  --out /path/to/checkpoints \
  --model 125M --arch dense \
  --ctx 1024 --batch-tokens 65536 --accum 4 \
  --max_steps 1000 --eval_every 200 --ckpt_every 200 \
  --tensorboard
```

Swap `--sources web` for `--sources web,code,wikipedia,books,math` (or omit `--sources`
entirely) once more sources are packed, and add `--mixture_weights web=0.5,code=0.3,...` to
override the default mixture ratios for an ablation. Swap `--arch dense` for
`--arch deepseek` to use MLA + MoE instead.

### 9.3 Distributed run (ZeRO/FSDP)

Same command, launched with `torchrun` instead of `python`, plus `--zero_stage`:

```bash
torchrun --nproc_per_node=8 main.py \
  --data_root /path/to/transformed --tokenizer_path /path/to/cl100k_base.tiktoken \
  --out /path/to/checkpoints --model 1.3B --arch dense \
  --zero_stage 3 --batch-tokens 262144 --accum 8 --max_steps 100000
```

`--zero_stage 0` (plain DDP) needs the most memory per GPU but the least communication;
`--zero_stage 3` (full FSDP) needs the least memory per GPU but the most communication — see
§3's table and §4.4's memory formula for the tradeoff.

### 9.4 Resuming

```bash
python main.py ... --resume latest   # resumes from {out}/latest/latest_check.pt
python main.py ... --resume /path/to/specific_checkpoint.pt
```

### 9.5 Monitoring

```bash
python plot_metrics.py /path/to/checkpoints     # writes loss/lr/throughput/perplexity PNGs to {out}/plots/
tensorboard --logdir /path/to/checkpoints/tensorboard   # if run with --tensorboard
```

---


