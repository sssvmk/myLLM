# Codebase Dependency Tree

Below is the comprehensive function-level dependency tree mapped across all Python files in the repository. It details both the **upstream dependencies** (which functions/methods call this one) and **downstream dependencies** (which functions/methods this one calls) for every function and class method.

## `benchmark_eval.py`
### `_sequence_loglik(model, tokens, device)`
- **Upstream Dependencies:** `evaluate_mmlu` (within same file)
- **Downstream Dependencies:** `model()` (forward pass), `torch.no_grad()`, `F.log_softmax()`, `torch.Tensor.gather()`, `torch.Tensor.squeeze()`, `torch.Tensor.sum()`, `torch.Tensor.item()`, `torch.Tensor.unsqueeze()`, `torch.Tensor.to()`

### `evaluate_mmlu(model, tokenizer, examples, device, max_examples)`
- **Upstream Dependencies:** None in this repository (intended for external/future use)
- **Downstream Dependencies:** `model.eval()`, `tokenizer.encode()`, `_sequence_loglik()`, `torch.tensor()`, `model.train()`

### `load_local_mmlu(path)`
- **Upstream Dependencies:** None in this repository (intended for external/future use)
- **Downstream Dependencies:** `open()`, `str.strip()`, `json.loads()`

---

## `config.py`
### `apply_weight_overrides(data_sources, overrides_str)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** `str.split()`, `str.strip()`, `str.partition()`, `ValueError`

### `pad_token_id_for(vocab_size)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** None

### `parse_args()`
- **Upstream Dependencies:** `__main__` block (in `main.py`)
- **Downstream Dependencies:** `argparse.ArgumentParser()`, `ArgumentParser.add_argument()`, `ArgumentParser.parse_known_args()`

---

## `data_quality.py`
*(Note: These functions are for the Spark ETL stage and are currently not called by the main runtime codebase.)*

### `exact_dedup(df, text_col)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `hashlib.sha256()`, `pyspark.sql.functions.udf()`, `DataFrame.withColumn()`, `DataFrame.dropDuplicates()`, `DataFrame.drop()`, `pyspark.sql.functions.col()`

### `near_dedup_minhash(df, text_col, ngram_n, jaccard_threshold, num_hash_tables)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `DataFrame.withColumn()`, `pyspark.sql.functions.monotonically_increasing_id()`, `Tokenizer().transform()`, `NGram().transform()`, `HashingTF().transform()`, `DataFrame.filter()`, `pyspark.sql.functions.size()`, `MinHashLSH().fit()`, `MinHashLSHModel.transform()`, `MinHashLSHModel.approxSimilarityJoin()`, `pyspark.sql.functions.col()`, `DataFrame.select()`, `DataFrame.distinct()`, `DataFrame.join()`, `DataFrame.drop()`

### `decontaminate_against_eval_sets(train_df, eval_texts, text_col, n, overlap_threshold)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `ngrams()` (inner), `set.union()`, `train_df.sql_ctx.sparkSession.sparkContext.broadcast()`, `overlap_ratio()` (inner), `pyspark.sql.functions.udf()`, `DataFrame.withColumn()`, `DataFrame.filter()`, `DataFrame.drop()`

### `quality_filter_heuristics(df, text_col, min_words, max_symbol_ratio, max_repeated_line_ratio)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `word_count()` (inner), `symbol_ratio()` (inner), `repeated_line_ratio()` (inner), `pyspark.sql.functions.udf()`, `DataFrame.withColumn()`, `pyspark.sql.functions.col()`, `DataFrame.filter()`, `DataFrame.drop()`

### `quality_filter_classifier(df, text_col)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `_get_model()` (inner), `score_batch()` (inner), `pyspark.sql.functions.pandas_udf()`, `DataFrame.withColumn()`, `DataFrame.filter()`, `DataFrame.drop()`

---

## `data.py`
### `ParquetTokenDataset.__init__(self, path, ctx, pad_token_id, shuffle)`
- **Upstream Dependencies:** `build_mixture_dataset()` (within same file), `drive()` (in `main.py`)
- **Downstream Dependencies:** `glob.glob()`, `pyarrow.parquet.ParquetFile()`

### `ParquetTokenDataset._worker_files(self)`
- **Upstream Dependencies:** `ParquetTokenDataset.__iter__()` (within same file)
- **Downstream Dependencies:** `random.shuffle()`, `torch.utils.data.get_worker_info()`

### `ParquetTokenDataset.__iter__(self)`
- **Upstream Dependencies:** Core iteration in `torch.utils.data.DataLoader` (downstream)
- **Downstream Dependencies:** `self._worker_files()`, `pyarrow.parquet.read_table()`, `Table.to_batches()`, `F.pad()`, `torch.tensor()`

### `_infinite_iter(dataset)`
- **Upstream Dependencies:** `MixtureIterableDataset.__iter__()` (within same file)
- **Downstream Dependencies:** `iter()`

### `MixtureIterableDataset.__init__(self, datasets, weights, cycle_flags, names, seed)`
- **Upstream Dependencies:** `build_mixture_dataset()` (within same file)
- **Downstream Dependencies:** None

### `MixtureIterableDataset.__iter__(self)`
- **Upstream Dependencies:** Core iteration in `torch.utils.data.DataLoader` (downstream)
- **Downstream Dependencies:** `torch.utils.data.get_worker_info()`, `random.Random()`, `_infinite_iter()`, `iter()`, `random.Random.choices()`, `next()`

### `build_mixture_dataset(data_root, split, ctx, pad_token_id, sources, shuffle, source_configs)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** `os.path.join()`, `ParquetTokenDataset()`, `MixtureIterableDataset()`

---

## `distributed.py`
### `is_distributed()`
- **Upstream Dependencies:** `is_main_process()`, `setup_distributed()`, `wrap_model()`, `build_optimizer_for_zero()` (all in `distributed.py`), `load_checkpoint()` (in `train.py`)
- **Downstream Dependencies:** `torch.distributed.is_available()`, `torch.distributed.is_initialized()`

### `is_main_process()`
- **Upstream Dependencies:** `full_optimizer_state_dict()` (in `distributed.py`), `drive()` (in `main.py`), `save_check_point()` (in `train.py`), `load_checkpoint()` (in `train.py`), `train_loop()` (in `train.py`), `MetricsLogger.__init__()` (indirectly through `train_loop`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `is_distributed()`, `torch.distributed.get_rank()`

### `setup_distributed()`
- **Upstream Dependencies:** `drive()` (in `main.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `torch.distributed.is_initialized()`, `torch.distributed.init_process_group()`, `torch.cuda.is_available()`, `torch.cuda.set_device()`

### `wrap_model(model, zero_stage, device)`
- **Upstream Dependencies:** `drive()` (in `main.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `is_distributed()`, `torch.nn.parallel.DistributedDataParallel()`, `functools.partial()`, `torch.distributed.fsdp.FullyShardedDataParallel()`

### `build_optimizer_for_zero(model, zero_stage, lr, weight_decay, betas, eps)`
- **Upstream Dependencies:** `drive()` (in `main.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `model.named_parameters()`, `is_distributed()`, `torch.distributed.optim.ZeroRedundancyOptimizer()`, `torch.cuda.is_available()`, `torch.cuda.get_device_capability()`, `torch.optim.AdamW()`

### `full_model_state_dict(model)`
- **Upstream Dependencies:** `save_check_point()` (in `train.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `FullyShardedDataParallel.state_dict_type()`, `model.state_dict()`, `model.module.state_dict()`

### `load_full_model_state_dict(model, full_msd)`
- **Upstream Dependencies:** `load_checkpoint()` (in `train.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `FullyShardedDataParallel.state_dict_type()`, `model.load_state_dict()`, `model.module.load_state_dict()`

### `full_optimizer_state_dict(model, optimizer)`
- **Upstream Dependencies:** `save_check_point()` (in `train.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `ZeroRedundancyOptimizer.consolidate_state_dict()`, `optimizer.state_dict()`, `is_main_process()`, `FullyShardedDataParallel.state_dict_type()`, `FullyShardedDataParallel.optim_state_dict()`

### `load_full_optimizer_state_dict(model, optimizer, full_osd)`
- **Upstream Dependencies:** `load_checkpoint()` (in `train.py`), `_worker()` (in `test_distributed_smoke.py`)
- **Downstream Dependencies:** `FullyShardedDataParallel.state_dict_type()`, `FullyShardedDataParallel.optim_state_dict_to_load()`, `optimizer.load_state_dict()`

---

## `main.py`
### `set_seed(seed)`
- **Upstream Dependencies:** `drive()` (within same file)
- **Downstream Dependencies:** `random.seed()`, `torch.manual_seed()`, `torch.cuda.is_available()`, `torch.cuda.manual_seed_all()`

### `create_model(vocab_size, args)`
- **Upstream Dependencies:** `drive()` (within same file)
- **Downstream Dependencies:** `GPTModel()` (from `model.py`)

### `drive(args)`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `setup_distributed()`, `set_seed()`, `torch.device()`, `torch.cuda.is_available()`, `load_cl100k_encoding()`, `pad_token_id_for()`, `apply_weight_overrides()`, `build_mixture_dataset()`, `ParquetTokenDataset()`, `is_main_process()`, `torch.utils.data.DataLoader()`, `create_model()`, `wrap_model()`, `torch.compile()`, `build_optimizer_for_zero()`, `CosineLRScheduler()`, `torch.cuda.amp.GradScaler()`, `os.path.join()`, `os.path.exists()`, `load_checkpoint()`, `os.makedirs()`, `train_loop()`

---

## `metrics.py`
### `MetricsLogger.__init__(self, out_dir, filename, tensorboard)`
- **Upstream Dependencies:** `train_loop()` (in `train.py`)
- **Downstream Dependencies:** `os.makedirs()`, `os.path.join()`, `open()`, `torch.utils.tensorboard.SummaryWriter()`

### `MetricsLogger.log(self, event, **fields)`
- **Upstream Dependencies:** `train_one_epoch()`, `evaluate()` (in `train.py`)
- **Downstream Dependencies:** `time.time()`, `json.dumps()`, `self._fh.write()`, `self._fh.flush()`, `self._log_tensorboard()`

### `MetricsLogger._log_tensorboard(self, event, fields)`
- **Upstream Dependencies:** `MetricsLogger.log()` (within same file)
- **Downstream Dependencies:** `self._tb.add_scalar()`, `self._tb.add_scalars()`, `self._tb.flush()`

### `MetricsLogger.close(self)`
- **Upstream Dependencies:** `train_loop()` (in `train.py`)
- **Downstream Dependencies:** `self._fh.close()`, `self._tb.close()`

### `NullMetricsLogger.log(self, event, **fields)`
- **Upstream Dependencies:** `train_one_epoch()`, `evaluate()` (in `train.py`, indirectly)
- **Downstream Dependencies:** None

### `NullMetricsLogger.close(self)`
- **Upstream Dependencies:** `train_loop()` (in `train.py`, indirectly)
- **Downstream Dependencies:** None

---

## `model.py`
### `RMSNorm.__init__(self, d, eps)`
- **Upstream Dependencies:** `MultiHeadLatentAttention.__init__()`, `Block.__init__()`, `GPTModel.__init__()`
- **Downstream Dependencies:** `torch.nn.Parameter()`, `torch.ones()`

### `RMSNorm.forward(self, x)`
- **Upstream Dependencies:** `MultiHeadLatentAttention.forward()`, `Block.forward()`, `GPTModel.forward()`
- **Downstream Dependencies:** `torch.Tensor.pow()`, `torch.Tensor.mean()`, `torch.sqrt()`

### `build_rope_cache(seq_len, dim, theta, device, dtype)`
- **Upstream Dependencies:** `CausalSelfAttention.forward()`, `MultiHeadLatentAttention.forward()` (within same file)
- **Downstream Dependencies:** `torch.arange()`, `torch.outer()`, `torch.cat()`, `torch.Tensor.cos()`, `torch.Tensor.sin()`

### `rotate_half(x)`
- **Upstream Dependencies:** `apply_rope()` (within same file)
- **Downstream Dependencies:** `torch.Tensor.chunk()`, `torch.cat()`

### `apply_rope(x, cos, sin)`
- **Upstream Dependencies:** `CausalSelfAttention.forward()`, `MultiHeadLatentAttention.forward()` (within same file)
- **Downstream Dependencies:** `rotate_half()`

### `build_doc_attention_mask(doc_start)`
- **Upstream Dependencies:** `_forward_loss()` (in `train.py`), `_run_arch()` (in `test_model_smoke.py`)
- **Downstream Dependencies:** `torch.cumsum()`, `torch.tril()`, `torch.ones()`, `torch.Tensor.unsqueeze()`

### `MLP.__init__(self, d_model, d_ff)`
- **Upstream Dependencies:** `GPTModel.__init__()` (within same file)
- **Downstream Dependencies:** `torch.nn.Linear()`, `torch.nn.GELU()`

### `MLP.forward(self, x)`
- **Upstream Dependencies:** `Block.forward()` (within same file)
- **Downstream Dependencies:** `self.fc1()`, `self.act()`, `self.fc2()`

### `CausalSelfAttention.__init__(self, d_model, n_heads, rope_theta)`
- **Upstream Dependencies:** `GPTModel.__init__()` (within same file)
- **Downstream Dependencies:** `torch.nn.Linear()`

### `CausalSelfAttention.forward(self, x, attn_mask)`
- **Upstream Dependencies:** `Block.forward()` (within same file)
- **Downstream Dependencies:** `self.qkv()`, `torch.Tensor.chunk()`, `torch.Tensor.view()`, `torch.Tensor.transpose()`, `build_rope_cache()`, `apply_rope()`, `torch.nn.functional.scaled_dot_product_attention()`, `torch.Tensor.contiguous()`, `self.proj()`

### `MultiHeadLatentAttention.__init__(self, d_model, n_heads, d_latent, d_rope, rope_theta)`
- **Upstream Dependencies:** `GPTModel.__init__()` (within same file)
- **Downstream Dependencies:** `torch.nn.Linear()`, `RMSNorm()`

### `MultiHeadLatentAttention.forward(self, x, attn_mask)`
- **Upstream Dependencies:** `Block.forward()` (within same file)
- **Downstream Dependencies:** `self.kv_down()`, `self.kv_norm()`, `self.kv_up()`, `torch.Tensor.view()`, `torch.Tensor.transpose()`, `torch.Tensor.split()`, `self.k_rope()`, `self.q_down()`, `self.q_norm()`, `self.q_up()`, `self.q_rope()`, `build_rope_cache()`, `apply_rope()`, `torch.cat()`, `torch.nn.functional.scaled_dot_product_attention()`, `torch.Tensor.contiguous()`, `self.proj()`

### `Block.__init__(self, d_model, attn, mlp, dropout)`
- **Upstream Dependencies:** `GPTModel.__init__()` (within same file)
- **Downstream Dependencies:** `RMSNorm()`, `torch.nn.Dropout()`

### `Block.forward(self, x, attn_mask)`
- **Upstream Dependencies:** `GPTModel.forward()` (within same file)
- **Downstream Dependencies:** `self.norm1()`, `self.attn()`, `self.dropout()`, `self.norm2()`, `self.mlp()`

### `GPTModel.__init__(self, vocab_size, d_model, ctx, n_layers, d_ff, n_heads, dropout, arch, d_latent, d_rope, n_routed_experts, n_shared_experts, moe_top_k, rope_theta)`
- **Upstream Dependencies:** `create_model()` (in `main.py`), `_worker()` (in `test_distributed_smoke.py`), `_run_arch()` (in `test_model_smoke.py`)
- **Downstream Dependencies:** `torch.nn.Embedding()`, `torch.nn.ModuleList()`, `MultiHeadLatentAttention()`, `DeepSeekMoE()`, `CausalSelfAttention()`, `MLP()`, `Block()`, `RMSNorm()`, `torch.nn.Linear()`, `self.apply()`, `self._init()`

### `GPTModel._init(self, m)`
- **Upstream Dependencies:** `GPTModel.__init__()` (within same file)
- **Downstream Dependencies:** `torch.nn.init.xavier_uniform_()`, `torch.nn.init.zeros_()`, `torch.nn.init.normal_()`

### `GPTModel.forward(self, idx, attn_mask)`
- **Upstream Dependencies:** `_forward_loss()` (in `train.py`), `_worker()` (in `test_distributed_smoke.py`), `_run_arch()` (in `test_model_smoke.py`)
- **Downstream Dependencies:** `idx.to()`, `self.tok_emb()`, `blk()`, `self.norm()`, `self.lm_head()`, `torch.stack()`, `torch.Tensor.sum()`

---

## `moe.py`
### `SwiGLUExpert.__init__(self, d_model, d_ff)`
- **Upstream Dependencies:** `DeepSeekMoE.__init__()` (within same file)
- **Downstream Dependencies:** `torch.nn.Linear()`

### `SwiGLUExpert.forward(self, x)`
- **Upstream Dependencies:** `DeepSeekMoE.forward()` (within same file)
- **Downstream Dependencies:** `self.gate()`, `torch.nn.functional.silu()`, `self.up()`, `self.down()`

### `DeepSeekMoE.__init__(self, d_model, d_ff, n_routed_experts, n_shared_experts, top_k, expert_d_ff, bias_update_rate, aux_loss_weight)`
- **Upstream Dependencies:** `GPTModel.__init__()` (in `model.py`)
- **Downstream Dependencies:** `torch.nn.Linear()`, `torch.nn.Parameter()`, `torch.zeros()`, `torch.nn.ModuleList()`, `SwiGLUExpert()`

### `DeepSeekMoE.forward(self, x)`
- **Upstream Dependencies:** `Block.forward()` (in `model.py`)
- **Downstream Dependencies:** `torch.Tensor.reshape()`, `self.router()`, `torch.nn.functional.softmax()`, `torch.Tensor.topk()`, `torch.Tensor.gather()`, `torch.Tensor.sum()`, `torch.zeros_like()`, `torch.zeros()`, `torch.Tensor.any()`, `torch.Tensor.unsqueeze()`, `SwiGLUExpert.forward()`, `torch.Tensor.detach()`, `torch.Tensor.clamp_min()`, `torch.Tensor.float()`, `torch.Tensor.var()`

### `DeepSeekMoE.update_bias(self)`
- **Upstream Dependencies:** `_update_moe_bias()` (in `train.py`)
- **Downstream Dependencies:** `torch.Tensor.mean()`, `torch.sign()`

---

## `packing.py`
*(Note: These functions execute independent ETL tasks and aren't directly referenced elsewhere in the repo's runtime.)*

### `_doc_start_flags(arr)`
- **Upstream Dependencies:** Wrapped in `_doc_start_udf`
- **Downstream Dependencies:** None

### `pack_source_to_shards(spark, raw_path, out_root, source_name, ctx, eos_token_id, order_col, seed, shard_rows)`
- **Upstream Dependencies:** `pack_all_configured_sources()` (within same file)
- **Downstream Dependencies:** `spark.read.parquet()`, `DataFrame.randomSplit()`, `_pack()` (inner function)

### `pack_all_configured_sources(spark, ctx, out_root)`
- **Upstream Dependencies:** None in this repository
- **Downstream Dependencies:** `pack_source_to_shards()`

---

## `plot_metrics.py`
### `load_events(metrics_path)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `open()`, `json.loads()`

### `plot_loss_and_lr(events, out_dir)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `matplotlib.pyplot.subplots()`, `matplotlib.pyplot.close()`, `os.path.join()`

### `plot_throughput(events, out_dir)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `matplotlib.pyplot.subplots()`, `matplotlib.pyplot.close()`, `os.path.join()`

### `plot_perplexity(events, out_dir)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `matplotlib.pyplot.subplots()`, `matplotlib.pyplot.close()`, `os.path.join()`

### `plot_moe_usage(events, out_dir)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `matplotlib.pyplot.subplots()`, `matplotlib.pyplot.close()`, `os.path.join()`

### `summarize(events)`
- **Upstream Dependencies:** `main()` (within same file)
- **Downstream Dependencies:** `min()`

### `main(out_dir)`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `os.path.join()`, `os.path.exists()`, `load_events()`, `summarize()`, `os.makedirs()`, `plot_loss_and_lr()`, `plot_throughput()`, `plot_perplexity()`, `plot_moe_usage()`

---

## `scheduler.py`
### `CosineLRScheduler.__init__(self, optimizer, warmup, total, base_lr, min_lr)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** None

### `CosineLRScheduler.step(self)`
- **Upstream Dependencies:** `train_one_epoch()` (in `train.py`)
- **Downstream Dependencies:** `max()`, `min()`, `math.cos()`

---

## `tokenizer.py`
### `load_cl100k_encoding(local_path)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** `open()`, `hashlib.sha256()`, `hashlib.sha256().hexdigest()`, `tiktoken.load.load_tiktoken_bpe()`, `tiktoken.Encoding()`, `tiktoken.get_encoding()`

---

## `train.py`
### `build_optimizer(model, lr, weight_decay, betas, eps)`
- **Upstream Dependencies:** `_run_arch()` (in `test_model_smoke.py`)
- **Downstream Dependencies:** `model.named_parameters()`, `torch.cuda.is_available()`, `torch.cuda.get_device_capability()`, `torch.optim.AdamW()`

### `_unpack_batch(batch, device)`
- **Upstream Dependencies:** `train_one_epoch()`, `evaluate()` (within same file)
- **Downstream Dependencies:** `torch.Tensor.to()`

### `_forward_loss(model, block, doc_start, pad_token_id, args)`
- **Upstream Dependencies:** `train_one_epoch()`, `evaluate()` (within same file)
- **Downstream Dependencies:** `build_doc_attention_mask()` (in `model.py`), `torch.amp.autocast()`, `torch.cuda.is_available()`, `model()` (forward pass), `torch.nn.functional.cross_entropy()`

### `_update_moe_bias(model)`
- **Upstream Dependencies:** `train_one_epoch()` (within same file), `_run_arch()` (in `test_model_smoke.py`)
- **Downstream Dependencies:** `getattr()`, `DeepSeekMoE.update_bias()` (in `moe.py`)

### `_moe_usage_snapshot(model)`
- **Upstream Dependencies:** `train_one_epoch()` (within same file)
- **Downstream Dependencies:** `getattr()`, `torch.Tensor.tolist()`

### `train_one_epoch(model, loader, optimizer, scaler, scheduler, args, micro, device, best_loss, step, pad_token_id, metrics)`
- **Upstream Dependencies:** `train_loop()` (within same file)
- **Downstream Dependencies:** `NullMetricsLogger()`, `model.train()`, `time.time()`, `_unpack_batch()`, `_forward_loss()`, `torch.cuda.amp.GradScaler.scale()`, `torch.Tensor.backward()`, `torch.cuda.amp.GradScaler.unscale_()`, `torch.nn.utils.clip_grad_norm_()`, `torch.optim.Optimizer.step()`, `torch.cuda.amp.GradScaler.step()`, `torch.cuda.amp.GradScaler.update()`, `_update_moe_bias()`, `torch.optim.Optimizer.zero_grad()`, `CosineLRScheduler.step()`, `MetricsLogger.log()`, `save_check_point()`, `_moe_usage_snapshot()`

### `evaluate(model, loader, device, args, pad_token_id, metrics, step)`
- **Upstream Dependencies:** `train_loop()` (within same file)
- **Downstream Dependencies:** `NullMetricsLogger()`, `model.eval()`, `_unpack_batch()`, `_forward_loss()`, `model.train()`, `math.exp()`, `MetricsLogger.log()`

### `save_check_point(model, optimizer, step, best_loss, args)`
- **Upstream Dependencies:** `train_one_epoch()`, `train_loop()` (within same file)
- **Downstream Dependencies:** `full_model_state_dict()` (in `distributed.py`), `full_optimizer_state_dict()` (in `distributed.py`), `is_main_process()` (in `distributed.py`), `os.makedirs()`, `os.path.join()`, `torch.save()`, `shutil.copyfile()`, `logging.Logger.warning()`

### `load_checkpoint(model, optimizer, path, device)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** `is_main_process()` (in `distributed.py`), `torch.load()`, `is_distributed()` (in `distributed.py`), `torch.distributed.broadcast_object_list()`, `load_full_model_state_dict()` (in `distributed.py`), `load_full_optimizer_state_dict()` (in `distributed.py`)

### `train_loop(model, train_loader, val_loader, optimizer, scaler, scheduler, micro, args, device, pad_token_id, step, best_loss, metrics)`
- **Upstream Dependencies:** `drive()` (in `main.py`)
- **Downstream Dependencies:** `MetricsLogger()` (in `metrics.py`), `is_main_process()` (in `distributed.py`), `NullMetricsLogger()` (in `metrics.py`), `os.path.join()`, `train_one_epoch()`, `evaluate()`, `save_check_point()`, `MetricsLogger.close()`

---

## `tests/test_distributed_smoke.py`
### `_worker(rank, world_size, zero_stage, tmp_dir, master_port)`
- **Upstream Dependencies:** `torch.multiprocessing.spawn()` from `run_stage()`
- **Downstream Dependencies:** `setup_distributed()`, `torch.device()`, `torch.manual_seed()`, `GPTModel()`, `wrap_model()`, `build_optimizer_for_zero()`, `torch.randint()`, `wrapped()` (forward), `torch.nn.functional.cross_entropy()`, `torch.Tensor.backward()`, `torch.optim.Optimizer.step()`, `torch.optim.Optimizer.zero_grad()`, `full_model_state_dict()`, `full_optimizer_state_dict()`, `os.path.join()`, `is_main_process()`, `torch.save()`, `torch.distributed.is_initialized()`, `torch.distributed.barrier()`, `torch.load()`, `torch.distributed.broadcast_object_list()`, `load_full_model_state_dict()`, `load_full_optimizer_state_dict()`, `torch.allclose()`, `torch.distributed.destroy_process_group()`

### `run_stage(zero_stage, world_size, port)`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `tempfile.TemporaryDirectory()`, `torch.multiprocessing.spawn()`

---

## `tests/test_model_smoke.py`
### `_run_arch(arch)`
- **Upstream Dependencies:** `test_dense()`, `test_deepseek()` (within same file)
- **Downstream Dependencies:** `torch.manual_seed()`, `GPTModel()` (in `model.py`), `build_optimizer()` (in `train.py`), `torch.randint()`, `torch.zeros()`, `build_doc_attention_mask()` (in `model.py`), `model()` (forward), `torch.nn.functional.cross_entropy()`, `torch.Tensor.backward()`, `torch.optim.Optimizer.step()`, `_update_moe_bias()` (in `train.py`), `torch.optim.Optimizer.zero_grad()`, `torch.Tensor.data_ptr()`

### `test_dense()`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `_run_arch()`

### `test_deepseek()`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `_run_arch()`

---

## `tests/test_quality_classifier_smoke.py`
### `test_scoring_mechanism()`
- **Upstream Dependencies:** `__main__` block (within same file)
- **Downstream Dependencies:** `tempfile.TemporaryDirectory()`, `os.path.join()`, `open()`, `transformers.BertTokenizer()`, `transformers.BertConfig()`, `transformers.BertForSequenceClassification()`, `model.eval()`, `score_batch()` (inner)