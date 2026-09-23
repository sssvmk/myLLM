# Codebase Architectural Mental Map
The repository represents a foundational LLM pretraining pipeline with support for advanced architectures (DeepSeek-style MoE and Multi-head Latent Attention) and distributed FSDP/ZeRO training.

- **Orchestration & Training (`main.py`, `train.py`, `scheduler.py`)**: `main.py` is the driver that initializes distributed environments and configurations, subsequently calling `train.py`'s `train_loop` to run gradient accumulations, MoE balancing, and evaluations.
- **Model Architecture (`model.py`, `moe.py`)**: Defines a highly modular transformer. The `GPTModel` composes multiple `Block`s. Each block injects either dense self-attention or MLA (`MultiHeadLatentAttention`), coupled with either a standard `MLP` or `DeepSeekMoE` layer.
- **Data & Tokenization (`data.py`, `tokenizer.py`, `config.py`)**: `data.py` uses PyTorch IterableDatasets to stream chunked Parquet files with built-in token-level padding and doc-start masks. `tokenizer.py` provides offline cl100k_base encoding.
- **Offline ETL (`packing.py`, `data_quality.py`)**: Provides Spark jobs for min-hash deduplication, evaluation-set decontamination, quality classification, and document packing. These scripts are run before main training execution.
- **Distributed Utils (`distributed.py`)**: Houses boilerplate for ZeRO stage 1-3 wrapping, FSDP state consolidation, and process group instantiation.
- **Observability (`metrics.py`, `plot_metrics.py`, `benchmark_eval.py`)**: Dumps telemetry to JSONL and TensorBoard. Offline scripts render matplotlib visualizations.

---

# Function-Level Dependency Tree

## `benchmark_eval.py`
* **`_sequence_loglik(model, tokens, device)`**
  * **Upstream (Callers):** `evaluate_mmlu`
  * **Downstream (Calls):** `model.forward`, `torch.no_grad`, `F.log_softmax`
* **`evaluate_mmlu(model, tokenizer, examples, device, max_examples)`**
  * **Upstream:** None (Designed to be called periodically via script invocation, currently unused in `train.py`)
  * **Downstream:** `_sequence_loglik`, `tokenizer.encode`, `torch.tensor`
* **`load_local_mmlu(path)`**
  * **Upstream:** None
  * **Downstream:** `json.loads`

## `config.py`
* **`apply_weight_overrides(data_sources, overrides_str)`**
  * **Upstream:** `main.drive`
  * **Downstream:** None
* **`pad_token_id_for(vocab_size)`**
  * **Upstream:** `main.drive`
  * **Downstream:** None
* **`parse_args()`**
  * **Upstream:** `main.py` (global scope `__main__`)
  * **Downstream:** `argparse.ArgumentParser`

## `data.py`
* **`ParquetTokenDataset.__init__`**
  * **Upstream:** `build_mixture_dataset`, `main.drive`
  * **Downstream:** `glob.glob`, `pyarrow.parquet.ParquetFile`
* **`ParquetTokenDataset._worker_files`**
  * **Upstream:** `ParquetTokenDataset.__iter__`
  * **Downstream:** `torch.utils.data.get_worker_info`, `random.shuffle`
* **`ParquetTokenDataset.__iter__`**
  * **Upstream:** PyTorch DataLoader iterations
  * **Downstream:** `ParquetTokenDataset._worker_files`, `pyarrow.parquet.read_table`, `F.pad`, `torch.tensor`
* **`_infinite_iter(dataset)`**
  * **Upstream:** `MixtureIterableDataset.__iter__`
  * **Downstream:** Iterates over dataset generator
* **`MixtureIterableDataset.__init__`**
  * **Upstream:** `build_mixture_dataset`
  * **Downstream:** None
* **`MixtureIterableDataset.__iter__`**
  * **Upstream:** PyTorch DataLoader iterations
  * **Downstream:** `torch.utils.data.get_worker_info`, `random.Random`, `_infinite_iter`
* **`build_mixture_dataset(...)`**
  * **Upstream:** `main.drive`
  * **Downstream:** `ParquetTokenDataset.__init__`, `MixtureIterableDataset.__init__`, `os.path.join`

## `data_quality.py`
*(Note: These functions are PySpark ETL utilities and have no upstream callers within the core PyTorch training loop.)*
* **`exact_dedup`** -> Downstream: `hashlib.sha256`
* **`near_dedup_minhash`** -> Downstream: `pyspark.ml.feature` operations (Tokenizer, NGram, MinHashLSH)
* **`decontaminate_against_eval_sets`** -> Downstream: Nested function `ngrams` and `overlap_ratio`
* **`quality_filter_heuristics`** -> Downstream: Nested functions `word_count`, `symbol_ratio`, `repeated_line_ratio`
* **`quality_filter_classifier`** -> Downstream: Nested functions `_get_model`, `score_batch`, `transformers.AutoTokenizer`, `transformers.AutoModelForSequenceClassification`

## `distributed.py`
* **`is_distributed()`**
  * **Upstream:** `is_main_process`, `wrap_model`, `build_optimizer_for_zero`, `train.load_checkpoint`
  * **Downstream:** `torch.distributed.is_available`, `torch.distributed.is_initialized`
* **`is_main_process()`**
  * **Upstream:** `main.drive`, `train.save_check_point`, `train.load_checkpoint`, `train.train_loop`, `full_optimizer_state_dict`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `is_distributed`, `torch.distributed.get_rank`
* **`setup_distributed()`**
  * **Upstream:** `main.drive`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `torch.distributed.init_process_group`
* **`wrap_model(...)`**
  * **Upstream:** `main.drive`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `is_distributed`, `torch.nn.parallel.DistributedDataParallel`, `torch.distributed.fsdp.FullyShardedDataParallel`
* **`build_optimizer_for_zero(...)`**
  * **Upstream:** `main.drive`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `is_distributed`, `torch.distributed.optim.ZeroRedundancyOptimizer`, `torch.optim.AdamW`
* **`full_model_state_dict(model)`**
  * **Upstream:** `train.save_check_point`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `FSDP.state_dict_type`, `model.state_dict`
* **`load_full_model_state_dict(model, full_msd)`**
  * **Upstream:** `train.load_checkpoint`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `FSDP.state_dict_type`, `model.load_state_dict`
* **`full_optimizer_state_dict(model, optimizer)`**
  * **Upstream:** `train.save_check_point`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `is_main_process`, `optimizer.consolidate_state_dict`, `FSDP.optim_state_dict`
* **`load_full_optimizer_state_dict(model, optimizer, full_osd)`**
  * **Upstream:** `train.load_checkpoint`, `tests/test_distributed_smoke._worker`
  * **Downstream:** `FSDP.optim_state_dict_to_load`, `optimizer.load_state_dict`

## `main.py`
* **`set_seed(seed)`**
  * **Upstream:** `drive`
  * **Downstream:** `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
* **`create_model(vocab_size, args)`**
  * **Upstream:** `drive`
  * **Downstream:** `model.GPTModel`
* **`drive(args)`**
  * **Upstream:** Python global scope (`if __name__ == '__main__'`)
  * **Downstream:** `distributed.setup_distributed`, `set_seed`, `tokenizer.load_cl100k_encoding`, `config.pad_token_id_for`, `config.apply_weight_overrides`, `data.build_mixture_dataset`, `data.ParquetTokenDataset`, `create_model`, `distributed.wrap_model`, `distributed.build_optimizer_for_zero`, `scheduler.CosineLRScheduler`, `train.load_checkpoint`, `train.train_loop`, `torch.compile`

## `metrics.py`
* **`MetricsLogger.__init__`**
  * **Upstream:** `train.train_loop`
  * **Downstream:** `os.makedirs`, `torch.utils.tensorboard.SummaryWriter`
* **`MetricsLogger.log`**
  * **Upstream:** `train.train_one_epoch`, `train.evaluate`
  * **Downstream:** `json.dumps`, `MetricsLogger._log_tensorboard`
* **`MetricsLogger._log_tensorboard`**
  * **Upstream:** `MetricsLogger.log`
  * **Downstream:** `SummaryWriter.add_scalar`, `SummaryWriter.add_scalars`
* **`MetricsLogger.close`**
  * **Upstream:** `train.train_loop`
  * **Downstream:** `SummaryWriter.close`
* **`NullMetricsLogger.log`** / **`NullMetricsLogger.close`**
  * **Upstream:** `train.train_loop`, `train.train_one_epoch`, `train.evaluate`
  * **Downstream:** None

## `model.py`
* **`RMSNorm.__init__`** / **`RMSNorm.forward`**
  * **Upstream:** `MultiHeadLatentAttention.__init__/forward`, `Block.__init__/forward`, `GPTModel.__init__/forward`
  * **Downstream:** None
* **`build_rope_cache(...)`**
  * **Upstream:** `CausalSelfAttention.forward`, `MultiHeadLatentAttention.forward`
  * **Downstream:** `torch.arange`, `torch.outer`, `torch.cat`
* **`rotate_half(x)`**
  * **Upstream:** `apply_rope`
  * **Downstream:** `torch.cat`
* **`apply_rope(x, cos, sin)`**
  * **Upstream:** `CausalSelfAttention.forward`, `MultiHeadLatentAttention.forward`
  * **Downstream:** `rotate_half`
* **`build_doc_attention_mask(doc_start)`**
  * **Upstream:** `train._forward_loss`, `tests/test_model_smoke._run_arch`
  * **Downstream:** `torch.cumsum`, `torch.tril`
* **`MLP.__init__`** / **`MLP.forward`**
  * **Upstream:** `Block.forward`, `GPTModel.__init__`
  * **Downstream:** `nn.Linear`, `nn.GELU`
* **`CausalSelfAttention.__init__`** / **`CausalSelfAttention.forward`**
  * **Upstream:** `Block.forward`, `GPTModel.__init__`
  * **Downstream:** `build_rope_cache`, `apply_rope`, `F.scaled_dot_product_attention`
* **`MultiHeadLatentAttention.__init__`** / **`MultiHeadLatentAttention.forward`**
  * **Upstream:** `Block.forward`, `GPTModel.__init__`
  * **Downstream:** `RMSNorm`, `build_rope_cache`, `apply_rope`, `F.scaled_dot_product_attention`
* **`Block.__init__`** / **`Block.forward`**
  * **Upstream:** `GPTModel.__init__`, `GPTModel.forward`
  * **Downstream:** `RMSNorm.forward`, `self.attn.forward`, `self.mlp.forward`
* **`GPTModel.__init__`**
  * **Upstream:** `main.create_model`, `tests/test_distributed_smoke._worker`, `tests/test_model_smoke._run_arch`
  * **Downstream:** `MultiHeadLatentAttention`, `moe.DeepSeekMoE`, `CausalSelfAttention`, `MLP`, `Block`, `RMSNorm`, `GPTModel._init`
* **`GPTModel._init`**
  * **Upstream:** `GPTModel.__init__`
  * **Downstream:** `nn.init.xavier_uniform_`, `nn.init.normal_`
* **`GPTModel.forward`**
  * **Upstream:** `train._forward_loss`, `tests/test_distributed_smoke._worker`, `benchmark_eval._sequence_loglik`, `tests/test_model_smoke._run_arch`
  * **Downstream:** `Block.forward`, `RMSNorm.forward`

## `moe.py`
* **`SwiGLUExpert.__init__`** / **`SwiGLUExpert.forward`**
  * **Upstream:** `DeepSeekMoE.__init__`, `DeepSeekMoE.forward`
  * **Downstream:** `F.silu`
* **`DeepSeekMoE.__init__`**
  * **Upstream:** `model.GPTModel.__init__`
  * **Downstream:** `SwiGLUExpert.__init__`
* **`DeepSeekMoE.forward`**
  * **Upstream:** `model.Block.forward`
  * **Downstream:** `F.softmax`, `SwiGLUExpert.forward`, `torch.topk`
* **`DeepSeekMoE.update_bias`**
  * **Upstream:** `train._update_moe_bias`
  * **Downstream:** None

## `packing.py`
*(Note: These functions are PySpark ETL utilities and have no upstream callers within the core PyTorch training loop.)*
* **`_doc_start_flags(arr)`** -> Upstream: `_doc_start_udf`
* **`pack_source_to_shards`** -> Upstream: `pack_all_configured_sources`. Downstream: Nested function `_pack`
* **`pack_all_configured_sources`** -> Upstream: None. Downstream: `pack_source_to_shards`

## `plot_metrics.py`
*(Note: Exclusively consumed offline/interactively)*
* **`load_events`**, **`plot_loss_and_lr`**, **`plot_throughput`**, **`plot_perplexity`**, **`plot_moe_usage`**, **`summarize`**
  * **Upstream:** `main` (in `plot_metrics.py`)
  * **Downstream:** `json.loads`, `matplotlib.pyplot` methods
* **`main(out_dir)`**
  * **Upstream:** Python global scope
  * **Downstream:** All local plot and summarize functions above.

## `scheduler.py`
* **`CosineLRScheduler.__init__`**
  * **Upstream:** `main.drive`
  * **Downstream:** None
* **`CosineLRScheduler.step`**
  * **Upstream:** `train.train_one_epoch`
  * **Downstream:** None

## `tokenizer.py`
* **`load_cl100k_encoding(local_path)`**
  * **Upstream:** `main.drive`
  * **Downstream:** `hashlib.sha256`, `tiktoken.load.load_tiktoken_bpe`, `tiktoken.Encoding`, `tiktoken.get_encoding`

## `train.py`
* **`build_optimizer(...)`**
  * **Upstream:** `tests/test_model_smoke._run_arch`
  * **Downstream:** `torch.optim.AdamW`
* **`_unpack_batch(batch, device)`**
  * **Upstream:** `train_one_epoch`, `evaluate`
  * **Downstream:** None
* **`_forward_loss(model, block, doc_start, pad_token_id, args)`**
  * **Upstream:** `train_one_epoch`, `evaluate`
  * **Downstream:** `model.build_doc_attention_mask`, `model.forward`, `F.cross_entropy`, `torch.amp.autocast`
* **`_update_moe_bias(model)`**
  * **Upstream:** `train_one_epoch`, `tests/test_model_smoke._run_arch`
  * **Downstream:** `moe.DeepSeekMoE.update_bias`
* **`_moe_usage_snapshot(model)`**
  * **Upstream:** `train_one_epoch`
  * **Downstream:** None
* **`train_one_epoch(...)`**
  * **Upstream:** `train_loop`
  * **Downstream:** `_unpack_batch`, `_forward_loss`, `scaler.scale`, `torch.nn.utils.clip_grad_norm_`, `_update_moe_bias`, `scheduler.step`, `metrics.log`, `save_check_point`, `_moe_usage_snapshot`
* **`evaluate(...)`**
  * **Upstream:** `train_loop`
  * **Downstream:** `_unpack_batch`, `_forward_loss`, `metrics.log`, `torch.no_grad`
* **`save_check_point(model, optimizer, step, best_loss, args)`**
  * **Upstream:** `train_one_epoch`, `train_loop`
  * **Downstream:** `distributed.full_model_state_dict`, `distributed.full_optimizer_state_dict`, `distributed.is_main_process`, `torch.save`, `shutil.copyfile`, `os.makedirs`
* **`load_checkpoint(model, optimizer, path, device)`**
  * **Upstream:** `main.drive`
  * **Downstream:** `distributed.is_main_process`, `torch.load`, `distributed.is_distributed`, `torch.distributed.broadcast_object_list`, `distributed.load_full_model_state_dict`, `distributed.load_full_optimizer_state_dict`
* **`train_loop(...)`**
  * **Upstream:** `main.drive`
  * **Downstream:** `distributed.is_main_process`, `metrics.MetricsLogger`, `train_one_epoch`, `evaluate`, `save_check_point`, `metrics.close`

## `tests/test_distributed_smoke.py`
* **`_worker(...)`**
  * **Upstream:** `run_stage`
  * **Downstream:** `distributed.setup_distributed`, `model.GPTModel`, `distributed.wrap_model`, `distributed.build_optimizer_for_zero`, `distributed.full_model_state_dict`, `distributed.full_optimizer_state_dict`, `distributed.is_main_process`, `distributed.load_full_model_state_dict`, `distributed.load_full_optimizer_state_dict`
* **`run_stage(...)`**
  * **Upstream:** Global scope
  * **Downstream:** `torch.multiprocessing.spawn`, `_worker`

## `tests/test_model_smoke.py`
* **`_run_arch(arch)`**
  * **Upstream:** `test_dense`, `test_deepseek`
  * **Downstream:** `model.GPTModel`, `train.build_optimizer`, `model.build_doc_attention_mask`, `train._update_moe_bias`
* **`test_dense()` / `test_deepseek()`**
  * **Upstream:** Global scope
  * **Downstream:** `_run_arch`

## `tests/test_quality_classifier_smoke.py`
* **`test_scoring_mechanism()`**
  * **Upstream:** Global scope
  * **Downstream:** Nested `score_batch`, `transformers.BertTokenizer`, `transformers.BertForSequenceClassification`