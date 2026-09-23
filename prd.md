# PRD 


## 1. Purpose


The pipeline covers corpus hardening, packing, multi-source data mixture, model training (dense or DeepSeek-style MLA + MoE), distributed scaling, evaluation and observability. Its output is a base-model checkpoint that later post-training phases (SFT, preference optimization, RL) build on.

This document states what the pipeline must do and how acceptance is judged. Design detail lives in the README; run history lives in the run log.


## 2. Goals and success criteria

The pipeline succeeds when it produces a base model that meets the targets below, on the agreed budget, reproducibly. Values marked TBD are decided in section 14.

| ID | Goal | Success criterion |
| --- | --- | --- |
| G-1 | Train a DeepSeek-style (MLA + MoE) base model | Target preset trained to the agreed token budget (D-1, D-2) |
| G-2 | Reach a defined model quality | Held-out loss and benchmark scores meet thresholds set in D-3 |
| G-3 | Train on a curated, governed corpus | Every source used has governance approval (DR-1) and passed the hardening steps (FR-DATA-1 to FR-DATA-5) |
| G-4 | Scale from one GPU to multi-node | Same config runs single-process and at the target GPU count with no code change (FR-DIST-1) |
| G-5 | Stay within compute budget | Total GPU-hours within the budget set in D-4 |
| G-6 | Make runs observable and repeatable | Every run meets NFR-4 (reproducibility) and NFR-5 (observability) |

## 3. Scope and non-goals

This PRD covers pretraining only, up to and including a base checkpoint. Post-training gets its own PRD; its needs from pretraining are captured in section 11.

**In scope**

- Corpus hardening: deduplication, decontamination, heuristic and model-based quality filtering
- Tokenization and packing into fixed-length blocks with document boundaries
- Weighted multi-source data mixture
- Dense and DeepSeek-style (MLA + MoE) architectures, selectable per run
- Single-process through multi-node training (ZeRO stages 0 to 3)
- Checkpointing, resume, metrics, charts
- Evaluation: held-out loss and perplexity, and log-likelihood benchmarks (MMLU and similar)
- A mid-training anneal phase (LR decay with a changed data mixture), if D-5 confirms it

**Non-goals**

- SFT, preference optimization, RLHF or RL (separate PRD)
- Generation-based benchmarks (GSM8K, HumanEval) as a pretraining acceptance gate
- Serving, deployment or inference optimization of the model
- Training a custom tokenizer (cl100k\_base is fixed, see C-4)

## 4. Assumptions and constraints

| ID | Type | Statement |
| --- | --- | --- |
| C-1 | Constraint | ETL runs on Databricks Spark; training runs on Databricks GPU clusters or equivalent Azure GPU compute |
| C-2 | Constraint | Corpus data is stored in Azure Data Lake Storage (`abfss://`); checkpoints and scratch in DBFS or ADLS |
| C-3 | Constraint | Runtime nodes have no general outbound internet access (corporate egress controls) |
| C-4 | Constraint | Tokenizer is cl100k\_base (fixed vocabulary), plus reserved ids defined by this PRD |
| C-5 | Constraint | All configuration is editable without code changes, through one settings file and CLI overrides |
| A-1 | Assumption | GPU type and count available per run are as set in D-4 |
| A-2 | Assumption | Raw corpora can be transferred into ADLS once, from a machine with internet access |
| A-3 | Assumption | PySpark and Spark ML are provided by the Databricks runtime |

## 5. Functional requirements

Each requirement has an acceptance criterion that a test or a run must demonstrate.

**Data preparation**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-DATA-1 | Remove exact-duplicate documents per source | Zero identical content hashes remain in output |
| FR-DATA-2 | Remove near-duplicate documents (MinHash-LSH, configurable Jaccard threshold) | Seeded near-duplicates in a test sample are removed at the configured threshold |
| FR-DATA-3 | Filter documents by heuristic quality rules (length, symbol ratio, repeated lines), thresholds configurable | Documents violating each rule are dropped in a labelled test sample |
| FR-DATA-4 | Filter documents by a model-based quality score (FineWeb-Edu classifier), threshold configurable | Documents below threshold are dropped; classifier loads from internal storage |
| FR-DATA-5 | Remove training documents with n-gram overlap above threshold against every evaluation benchmark used | Planted benchmark passages in a test sample are removed |
| FR-DATA-6 | Pack documents into fixed-length blocks of ctx + 1 tokens, with an EOS token after every document | Every document boundary in output carries EOS; block length is exact except the final block |
| FR-DATA-7 | Emit a per-token document-start flag aligned with tokens | Flag count equals token count per block; flags match EOS positions |
| FR-DATA-8 | Preserve a deterministic, globally ordered token stream per source and split | Two packing runs on the same input produce identical output |
| FR-DATA-9 | Split each source into train and test with a fixed seed and no document in both | Zero document overlap between splits |

**Data loading and mixture**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-MIX-1 | Sample training blocks across sources at configured weights | Observed source shares over 10,000 blocks are within ±1 percentage point of configured weights |
| FR-MIX-2 | Repeat small sources marked for upsampling so they do not run out before the driving source | Upsampled sources remain present until the driving source is exhausted |
| FR-MIX-3 | Override mixture weights per run from the CLI | A run with overrides logs and uses the overridden weights |
| FR-MIX-4 | Restrict a run to a subset of sources | Only named sources appear in sampled data |
| FR-MIX-5 | With multiple data-loader workers, each block is read by exactly one worker | No duplicate blocks per epoch across workers |

**Model**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-MOD-1 | Provide a dense architecture: RoPE, causal multi-head attention, MLP, RMSNorm, tied embeddings | Forward and backward pass run at every preset |
| FR-MOD-2 | Provide a DeepSeek-style architecture: Multi-head Latent Attention and DeepSeekMoE with shared and routed experts | Forward and backward pass run at every preset |
| FR-MOD-3 | Balance MoE expert load without an auxiliary-loss dependency (per-expert routing bias) | No expert receives under 25% or over 400% of mean load after warmup (threshold to confirm in D-3) |
| FR-MOD-4 | Prevent attention across document boundaries within a packed block, switchable per run | A token cannot attend to any token from a different document in a mask test |
| FR-MOD-5 | Size models from named presets and per-knob overrides (layers, width, heads, FFN width, expert counts, top-k, latent dims) | Reported parameter counts (total and active) match the analytic formula exactly |
| FR-MOD-6 | Exclude padding tokens from the loss | Loss is unchanged when padding length changes |

**Training**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-TRN-1 | Support gradient accumulation, gradient clipping, bf16 mixed precision | Accumulated steps match the equivalent large-batch step within numerical tolerance |
| FR-TRN-2 | Apply linear warmup and a configurable decay schedule (cosine; warmup-stable-decay if D-5 adopts it) | Logged LR matches the schedule at every step |
| FR-TRN-3 | Exclude norms and other 1-D parameters from weight decay | Optimizer groups show zero decay for 1-D parameters (ZeRO-1 exception documented in section 13) |
| FR-TRN-4 | Save checkpoints on a step interval and on best validation loss | Checkpoints appear at the configured interval; best checkpoint tracks lowest val loss |
| FR-TRN-5 | Resume from the latest or a named checkpoint with identical state | Resumed run continues step count, LR, optimizer state and best loss; loss curve continues without a jump |
| FR-TRN-6 | Evaluate held-out loss and perplexity on a step interval and at the final step | Eval events are logged at every interval and at the final step |

**Distributed**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-DIST-1 | Select ZeRO stage 0 to 3 (DDP, DDP + optimizer sharding, FSDP grad sharding, FSDP full sharding) by one flag | Each stage trains and checkpoints at the target GPU count |
| FR-DIST-2 | Run single-process with no distributed wrapping when not launched under torchrun | Same command trains on one GPU without distributed setup |
| FR-DIST-3 | Save and load checkpoints correctly under every stage, in a format independent of stage | A checkpoint saved at one stage loads at another stage |

**Evaluation**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| FR-EVAL-1 | Score multiple-choice benchmarks by log-likelihood (MMLU and the small-model set chosen in D-3) | Scores reproduce on a fixed checkpoint to within ±0.5 points |
| FR-EVAL-2 | Load benchmark data from internal storage | Evaluation runs with no outbound network access |
| FR-EVAL-3 | Run benchmarks periodically during training, subset selectable by flag | Benchmark events are logged at the configured interval |

## 6. Non-functional requirements

| ID | Area | Requirement | Acceptance criterion |
| --- | --- | --- | --- |
| NFR-1 | Offline operation | Every runtime artifact (tokenizer ranks file, quality classifier weights, benchmark data) loads from internal storage, with SHA-256 verification where a reference hash exists | A full ETL and training run completes with outbound network blocked |
| NFR-2 | Throughput | Training reaches a minimum model FLOPs utilization per GPU | MFU at or above the target set in D-4, measured over 1,000 steps |
| NFR-3 | Fault tolerance | Lost work on failure is bounded by the checkpoint interval | Killing a run and resuming loses no more than one checkpoint interval of steps |
| NFR-4 | Reproducibility | A run is reproducible from its checkpoint alone: seed, full config, data mixture and code version are stored in it | Re-launching from the stored config on the same hardware reproduces the first 100 steps' loss within tolerance |
| NFR-5 | Observability | Train loss, LR, tokens/sec, val loss, perplexity and per-layer expert usage are logged in a machine-readable format, readable mid-run; TensorBoard optional | All listed metrics present in the metrics log for every run |
| NFR-6 | Configurability | Paths, presets, data sources, weights and artifact locations change through the settings file or CLI, never code | A new source or preset is added with no code change |
| NFR-7 | Security | Data access uses managed identities or scoped credentials; no secrets or personal paths in committed config | Config review finds no credentials or user-specific paths |
| NFR-8 | Portability of checkpoints | Checkpoints load outside the training cluster for evaluation and post-training | A checkpoint loads on a single machine with no distributed setup |

## 7. Data requirements and governance

No corpus enters training without governance approval of its license and content. Two current sources need review before any use beyond research.

**Sources**

| Source | Dataset | License | Default weight | Governance note |
| --- | --- | --- | --- | --- |
| Web | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) | Unspecified by curator | 0.65 | Review required before any commercial or internal-production use |
| Code | [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Per-file, from source repositories | 0.15 | Review required; license filtering per file may be needed |
| Wikipedia | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 4.0 / GFDL | 0.10 | Attribution and share-alike obligations to confirm |
| Books | [deepmind/pg19](https://huggingface.co/datasets/deepmind/pg19) | Public domain | 0.07 | Low risk |
| Math | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | ODC-By 1.0 | 0.03 | Attribution obligations to confirm |

Weights are starting defaults, to be set by ablation (D-6).

**Requirements**

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| DR-1 | Each source has a recorded governance approval covering license, permitted use and retention | Approval record exists before the source is packed |
| DR-2 | Each source has a target token volume after hardening | Volumes set in D-2 and recorded per source |
| DR-3 | Personal data in web and code sources is detected and removed or masked | PII scan on a sample meets the threshold agreed with governance |
| DR-4 | Lineage is kept from raw source to packed shard: source version, download date, filter settings, counts removed per step | Lineage record exists for every packed shard set |
| DR-5 | Evaluation benchmarks used for decontamination are versioned and stored internally | Benchmark versions are recorded with each run |

## 8. Interfaces

**Packed data layout.** Each source is stored under one data root:

```
{data_root}/{source}/{train,test}/shard_id=*/part-*.parquet
```

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `tokens` | array\<int64> | Yes | One packed block of ctx + 1 token ids |
| `doc_start` | array\<bool> | Yes for new data | Same length as `tokens`; true at the first token of each document |

The loader accepts data without `doc_start` for backward compatibility and treats the block as one document.

**Configuration.** One settings file (`settings.yaml`) is the single source of truth for presets, tokenizer location, raw source locations, data mixture, benchmark sources, quality classifier and output location. CLI flags override it per run. The settings file location is overridable by environment variable.

**Raw source input to packing.** One row per document, with a `token_content` array\<int64> column and a stable ordering column.

**Checkpoint.** A single file per checkpoint containing full (unsharded) model state, full optimizer state, step, best validation loss and the complete run configuration. A `latest` pointer is always updated. The format is independent of ZeRO stage (FR-DIST-3).

**Metrics.** Append-only JSON Lines, one event per line, each with an event type, wall time and step. Event types: `train_step`, `eval`, `moe_usage`, and benchmark events.

## 9. Dependencies

Package versions must be pinned in `pyproject.toml`; the minimum versions below are set in D-7.

**Software**

| Dependency | Needed for | Source in target environment |
| --- | --- | --- |
| PyTorch (with FSDP, ZeroRedundancyOptimizer) | Model, training, distributed | Internal package mirror |
| PyArrow | Reading packed shards | Internal package mirror |
| tiktoken | cl100k\_base encoding | Internal package mirror |
| transformers | Quality classifier | Internal package mirror |
| PyYAML | Settings file | Internal package mirror |
| matplotlib, TensorBoard | Charts and optional dashboards | Internal package mirror |
| PySpark, Spark ML | ETL, dedup, packing | Databricks runtime |

**Artifacts (all stored internally, per NFR-1)**

| Artifact | Origin | Integrity check |
| --- | --- | --- |
| cl100k\_base ranks file | [openaipublic.blob.core.windows.net](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) | SHA-256 `223921b7…65b2a7` |
| FineWeb-Edu quality classifier | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) (Apache 2.0) | Model revision pinned |
| MMLU | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Dataset revision pinned |
| GSM8K, HumanEval | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), [openai/openai\_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | Revision pinned; decontamination only in this phase |

**Infrastructure**

| Requirement | Needed for |
| --- | --- |
| Databricks workspace with Spark clusters | ETL and packing |
| ADLS containers for raw, packed and checkpoint data | Storage (C-2) |
| GPU clusters with NCCL interconnect, torchrun | Training at target scale (D-4) |

## 10. Requirements from post-training

These are cheap to meet before the main pretraining run and expensive to retrofit after it.

| ID | Requirement | Why | Acceptance criterion |
| --- | --- | --- | --- |
| PT-1 | Reserve token ids for chat roles and turn markers in the embedding table (count set in D-8) | Avoids resizing the tied embedding and output layer after pretraining | Reserved ids exist, are never produced by the tokenizer on corpus text, and are excluded from pretraining loss |
| PT-2 | Support a learning-rate schedule that allows branching a decay phase from a stable-LR checkpoint (warmup-stable-decay) | Enables mid-training anneals and data-mix experiments without restarting | A decay phase with a different mixture can start from any stable-phase checkpoint |
| PT-3 | Expose a model interface for incremental decoding, including a key/value cache (the compressed latent for MLA) | SFT evaluation, preference optimization and RL all need efficient sampling | Cached and uncached decoding produce identical logits |
| PT-4 | Support per-token loss masks in addition to padding masks | SFT trains only on assistant tokens | Masked tokens contribute zero loss and zero gradient |
| PT-5 | Allow freezing or controlling MoE routing-bias updates per run | Fine-tuning on narrow data can skew expert routing | Bias updates can be disabled by flag |
| PT-6 | Checkpoints load without optimizer state for downstream use | Post-training starts from weights only | Weights-only load succeeds (NFR-8) |


```

| Milestone | Gate |
| --- | --- |
| M1 Governance and artifacts | DR-1 approvals recorded; all NFR-1 artifacts in internal storage; D-1 to D-8 closed |
| M2 ETL on sample | FR-DATA-1 to FR-DATA-9 pass on a 1% sample per source |
| M3 Full ETL and packing | Packed volumes meet DR-2; DR-4 lineage complete |
| M4 Single-GPU run | Smallest preset trains; FR-TRN-1 to FR-TRN-6 and FR-MOD-1 to FR-MOD-6 pass |
| M5 Multi-node scale test | FR-DIST-1 to FR-DIST-3 pass at target GPU count; NFR-2 MFU met; NFR-3 verified |
| M6 Full pretraining run | Token budget reached within compute budget (G-1, G-5) |
| M7 Eval sign-off | G-2 thresholds met; product owner signs off |

## 11. Risks and mitigations

| ID | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R-1 | Governance rejects web or code corpus licenses | Main data sources unavailable | Start review at M1; identify replacement corpora with clear licenses (for example FineWeb-Edu, license-filtered code) |
| R-2 | MoE expert load collapses or routing becomes unstable | Wasted capacity, loss spikes | Monitor per-layer expert usage (NFR-5); FR-MOD-3 thresholds trigger investigation |
| R-3 | FSDP full sharding underperforms or fails at multi-node scale | Missed throughput or schedule | M5 scale test before the full run; fall back to a lower ZeRO stage if memory allows |
| R-4 | Full-state checkpointing to one rank does not fit at the largest presets | Checkpoint failures at scale | Adopt sharded checkpoints for presets above an agreed size |
| R-5 | ZeRO stage 1 applies weight decay to all parameters | Slightly different optimization than other stages | Accept and document, or avoid stage 1 for final runs |
| R-6 | Spark near-dedup cost is too high at full corpus scale | ETL delays | Exact dedup first to cut volume; tune LSH tables on the M2 sample |
| R-7 | Mixture weights are poorly chosen | Weaker model for the same compute | Run small-scale weight ablations (D-6) before M6 |
| R-8 | Benchmark contamination missed by n-gram matching | Inflated eval scores | Decontaminate against every benchmark used in G-2; report per-benchmark overlap stats |

