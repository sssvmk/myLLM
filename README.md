# foundation_llm

Split-out, bug-fixed, multi-source version of `MyLLM.py`. Every file below has been syntax-
checked; `model.py`, `moe.py`, `data.py`, `scheduler.py`, and `train.py` have additionally
been run end-to-end against synthetic data in this environment (see "What was actually
tested" below). `packing.py` and `data_quality.py` (the Spark-dependent files) could not be
executed here — no Spark cluster is available in this sandbox — so **dry-run them against a
small real sample before trusting them at full corpus scale.**

## Files

| File | Contents |
|---|---|
| `config.py` | `PRESETS`, `DATA_SOURCES` mixture table, CLI (`parse_args`) |
| `model.py` | `RMSNorm`, RoPE helpers, dense `CausalSelfAttention`, `MultiHeadLatentAttention` (MLA), `Block`, `GPTModel`, `build_doc_attention_mask` |
| `moe.py` | `DeepSeekMoE` (routed + shared experts, aux-loss-free bias balancing) |
| `data.py` | `ParquetTokenDataset` (fixed padding + worker-safety + doc_start), `MixtureIterableDataset`, `build_mixture_dataset` |
| `data_quality.py` | Spark dedup / decontamination / quality-filter utilities (pre-tokenization stage) |
| `packing.py` | Spark packing: EOS insertion, `doc_start` tracking, fixed global ordering |
| `scheduler.py` | `CosineLRScheduler` |
| `metrics.py` | `MetricsLogger` — writes structured train/eval/MoE-usage events to `{out}/metrics.jsonl` |
| `plot_metrics.py` | Reads `metrics.jsonl`, renders loss/LR/throughput/perplexity/MoE-usage PNG charts to `{out}/plots/` |
| `distributed.py` | ZeRO / FSDP integration (stages 0-3, per lecture_08) — `setup_distributed`, `wrap_model`, `build_optimizer_for_zero`, collective-safe checkpoint helpers |
| `train.py` | Training loop, eval, checkpoint save/load, optimizer construction |
| `main.py` | Driver (`drive(args)`), CLI entry point |
| `tests/test_model_smoke.py` | Synthetic-data smoke test for both architectures |



- `step`/`best_loss` now propagate across epochs in `train_loop` (previously silently reset each epoch)
- dropout is actually applied (`Block.forward`) — previously constructed but never called
- tied embedding/`lm_head` init order fixed — previously Xavier-clobbered the intended embedding init
- `evaluate()` runs under `@torch.no_grad()`
- checkpoint/resume: `optimizer.load_state_dict` (was a nonexistent `load_state_device`), optimizer
  state tensors moved to device manually (optimizers have no `.to()`), the undefined `args.latest`
  reference removed, `best_loss` restored from the checkpoint instead of being reset to `inf`
- `save_check_point` creates its `latest/` subdirectory before copying into it, and no longer
  swallows the copy failure silently
- fused-AdamW capability check guarded for CPU-only runs
- padding uses a reserved id (`vocab_size`, one past the real tokenizer vocab) instead of `0`
  (a real `cl100k_base` token), and the loss uses `ignore_index` on that id
- RoPE is actually implemented and applied (dense arch); the previous unused `rope` flag / dead
  learned position embedding are gone
- AdamW uses separate param groups so 1-D params (norms, the tied embedding) skip weight decay
- `ParquetTokenDataset` is worker-safe: each `DataLoader` worker reads its own shard of files
- an eval now always runs before training exits, even if `max_steps` is hit between exact
  `eval_every` boundaries (found by the integration test below — `best_loss` could otherwise
  stay `inf` even on a "successful" run)



- **Multi-source data mixture** (`config.DATA_SOURCES`, `data.MixtureIterableDataset`,
  `main.py`'s `--data_root`/`--sources`): weighted sampling across sources, with
  `cycle=True` sources (wiki/books/math) upsampled by repeating instead of running out early.
  `--train`/`--test` still work standalone as a legacy single-source fallback.
- **EOS document separators + document-boundary attention masking**: `packing.py` inserts an
  EOS token between concatenated documents and emits a `doc_start` column; `data.py` reads it;
  `model.build_doc_attention_mask` + `train._forward_loss` build and apply the mask so
  attention can't cross into an unrelated document within a packed block. Toggle with
  `--use_doc_mask` (on by default).
- **Fixed Spark ordering bug**: `packing.py` replaces `monotonically_increasing_id()` (only
  monotonic *within* a partition) with `row_number().over(Window.orderBy(...))` (a true global
  order), and reconstructs each packed block via `sort_array` over `(pos, tokens, doc_start)`
  structs, since `collect_list` after `groupBy` doesn't itself guarantee order.
- **Dedup / decontamination / quality filtering** (`data_quality.py`): `exact_dedup`,
  `near_dedup_minhash`, `decontaminate_against_eval_sets`, `quality_filter_heuristics`.
  `quality_filter_classifier` is a deliberate stub — see the note in that file for why.
- **DeepSeek-style architecture** (`--arch deepseek`): `MultiHeadLatentAttention` (low-rank
  KV + decoupled RoPE query/key) and `DeepSeekMoE` (routed + shared experts, aux-loss-free
  bias-based load balancing), selectable per run alongside the `dense` baseline.

## Evaluation metrics + charting (added after a direct question about this)

The version handed over earlier only printed loss/LR/tok-s to stdout — no persisted metrics,
no charts. Fixed:

- `metrics.py`'s `MetricsLogger` writes one JSON line per event to `{args.out}/metrics.jsonl`:
  `train_step` (step, loss, lr, tok_per_sec), `eval` (step, val_loss, perplexity), and
  `moe_usage` (per-layer routed-expert token counts, logged at each checkpoint for `--arch
  deepseek` runs). `train_loop` owns one by default; pass your own if you want to route it
  elsewhere (e.g. also mirror events to a tracking service).
- `plot_metrics.py path/to/args.out` reads that file (safe to run mid-training -- it just
  reads whatever's logged so far) and writes `loss.png`, `lr_schedule.png`, `throughput.png`,
  `perplexity.png`, and (for `deepseek` runs) `moe_usage.png` to `{out}/plots/`, plus prints a
  one-line summary (latest/best val loss + perplexity).

Actually run and verified in this session (not just written): a 30-step synthetic run with
`--arch deepseek` produced a real `metrics.jsonl` with all three event types, and
`plot_metrics.py` turned it into five real PNGs, including an expert-usage chart that visibly
confirms the aux-loss-free load-balancing bias is doing its job (roughly even bars per layer).

Not included: no TensorBoard/W&B integration, no downstream-benchmark evaluation (MMLU-style
held-out accuracy) — `evaluate()` only reports LM loss/perplexity on the held-out split, which
is the standard pretraining-time metric, but it isn't the same as benchmark evaluation.

## ZeRO / FSDP 

The earlier "parameterize the ZeroD-FDSP training framework" todo item turned out to be a
garbled reference to **ZeRO / FSDP** from the uploaded CS336 lecture_08 slides (ZeRO stages
1-3, where stage 3 = FSDP). Implemented in `distributed.py`, mapped directly onto the
lecture's terminology:

| `--zero_stage` | Lecture term | Mechanism | What's sharded |
|---|---|---|---|
| 0 | baseline DDP | `torch.nn.parallel.DistributedDataParallel` | nothing |
| 1 | P_os | DDP + `torch.distributed.optim.ZeroRedundancyOptimizer` | optimizer state only |
| 2 | P_os+g | FSDP, `ShardingStrategy.SHARD_GRAD_OP` | optimizer state + gradients |
| 3 | P_os+g+p (FSDP) | FSDP, `ShardingStrategy.FULL_SHARD` | optimizer state + gradients + params |

`--zero_stage` takes effect only under `torchrun` (`WORLD_SIZE > 1` in the environment);
single-process runs skip it entirely and get the model/optimizer unwrapped, since there's
nothing to shard across one process. FSDP auto-wraps at `model.Block` boundaries via
`transformer_auto_wrap_policy`, matching the lecture's per-layer all-gather/free pattern
rather than treating the whole model as one FSDP unit.

**Checkpointing under ZeRO/FSDP is genuinely different from the single-process case**: saving
and loading full state dicts are *collective* operations (every rank must call them, or the
other ranks deadlock waiting on an all-gather that never happens), even though only rank 0
ends up with the actual data to write to disk. `distributed.py` provides
`full_model_state_dict`/`full_optimizer_state_dict` (save side) and
`load_full_model_state_dict`/`load_full_optimizer_state_dict` (load side, with an explicit
`broadcast_object_list` from rank 0 since FSDP doesn't broadcast on load the way it does on
save). `train.py`'s `save_check_point`/`load_checkpoint` were rewired to use these instead of
calling `.state_dict()`/`.load_state_dict()` directly — this is the "baby version" FSDP
checkpointing approach from the lecture (gather everything onto one rank), not the
production-scale sharded-checkpoint approach, which would write one shard file per rank
instead.

**Known limitation, documented rather than silently ignored**: `ZeroRedundancyOptimizer`
(used for `zero_stage=1`) takes one flat parameter list with a single set of kwargs, not
per-group kwargs — so the norm/1-D-parameter weight-decay exclusion that `train.py`'s regular
optimizer path applies (see "What was fixed" above) is **not** preserved under
`zero_stage=1`; every parameter gets the same `weight_decay` there. Stages 0, 2, and 3 keep
the exclusion.

### Actually tested — real 2-process distributed runs, not mocked

This is the one piece of the whole package that got the deepest testing, because
correctness here is easy to get subtly wrong (collective-vs-local calls, when a return value
is populated vs. `None`, rank-gating). In this sandbox (`tests/test_distributed_smoke.py`),
using `torch.multiprocessing.spawn` with the `gloo` CPU backend:

- All four `zero_stage` values (0/1/2/3) were run as a genuine 2-process job: real
  `DistributedDataParallel`, real `ZeroRedundancyOptimizer`, real
  `FullyShardedDataParallel` (both `SHARD_GRAD_OP` and `FULL_SHARD`) — forward, backward,
  optimizer step, then a full checkpoint save + round-trip load into a *fresh* model and
  optimizer, with every parameter tensor verified to match after reload.
- `train.py`'s actual `save_check_point`/`load_checkpoint` functions (not reimplemented
  logic in the test) were separately verified end-to-end under both `zero_stage=0` and
  `zero_stage=3` in a real 2-process run.
- One real bug surfaced and got fixed by this testing: this torch version's FSDP refuses to
  initialize at all without a detected accelerator (GPU) — even under CPU/`gloo` — unless
  `device_id` is passed explicitly. `wrap_model` now always passes it. Without that fix,
  `--zero_stage 2` or `3` would crash immediately on any machine without a GPU visible to
  the process, which would have been a nasty surprise to hit only once you actually had a
  multi-GPU allocation.
- A second real bug caught during this round: `MetricsLogger` would otherwise have every
  rank opening and appending to the same `metrics.jsonl` concurrently. `train_loop` now
  only gives rank 0 a real `MetricsLogger`; other ranks get a no-op logger.

**Not tested here** (needs real hardware this sandbox doesn't have): actual multi-GPU/NCCL
runs, more than 2 ranks, `CPUOffload`, mixed precision under FSDP, and anything at the scale
the lecture actually discusses (hundreds+ of GPUs). The deprecation warnings from
`FSDP.state_dict_type` in this torch version are real (PyTorch is migrating to
`torch.distributed.checkpoint.state_dict.get_state_dict`/`set_state_dict`) — noted but not
migrated, since the deprecated API is still fully functional and switching would need its
own round of real distributed testing to confirm rather than swapping blind.

## Still open / needs your input

- **`DATA_SOURCES` weights** are a starting point, not tuned values — run small-scale
  ablations before trusting them (see the earlier discussion on mixture-ratio tuning).
- **Real source paths**: `packing.py`'s `RAW_SOURCE_PATHS` example block needs your actual
  per-source raw paths filled in before it can run.
- **`quality_filter_classifier`** needs an actual trained classifier + reference-quality set —
  it currently raises `NotImplementedError` rather than pretending to be functional.
- **Spark-dependent code is untested here** (`packing.py`, `data_quality.py`) — no cluster in
  this sandbox. Dry-run against a small sample first.
- **MLA/MoE are simplified reproductions**, not exact DeepSeek dimensions/hyperparameters —
  correct shapes and information flow, but expert counts/widths/`d_latent` are config knobs
  to tune, and there's no KV-cache wiring (this codebase only does full-sequence training,
  not incremental generation).




