# Implementation notes (IG-3, IG-7)

Decisions on details the PRD leaves open, or where the PRD as written cannot be implemented
against the current `foundation_llm` code. Each entry: requirement, question, choice, source.

## Reference code consulted (IG-7)

None of the §21.4 repositories were read or ported for this build, so there is nothing to pin
and `NOTICE` lists no ported code. Formulas come from the PRD text itself (IG-1).

## Deviations forced by `foundation_llm` behavior (verified by running it, torch 2.14)

| Req | Problem | Choice |
|---|---|---|
| TR-5a | `wrap_model` builds FSDP with `use_orig_params=False`. For `arch: deepseek`, flattening fails: `DeepSeekMoE.routing_bias` has `requires_grad=False`, and FSDP raises "Must flatten tensors with uniform requires_grad". ZeRO 2/3 with deepseek cannot start. | `training/parallel.py` wraps ZeRO 2/3 itself with `FSDP(..., use_orig_params=True)` and the same Block auto-wrap policy. Tested 2-process gloo, ZeRO 0/2/3. |
| TR-3, TR-5a | Under FSDP, `build_optimizer_for_zero` sees only 1-D parameters (FlatParameters, or 1-D shard views with `use_orig_params`), so every weight lands in the no-decay group: weight decay is silently 0. Under ZeRO-1 it applies decay to norms too. | Decay/no-decay is decided from names and shapes captured before wrapping (`decay_param_names`), for every ZeRO stage; ZeRO-1 gets both groups via `add_param_group`. |
| TR-2, PT-9 | `DeepSeekMoE.forward`'s aux loss is built from token counts and has no autograd graph (`requires_grad=False`). Adding it to the objective changes the logged loss value but produces no gradient; `aux_loss_weight` has no training effect. | Replaced by a differentiable expert-level balance loss via forward hooks (`training/moe_balance.py`, prd_review #3); still added to the objective per TR-2. The reported `loss` field is the cross-entropy only; the aux term is in the gradient. |
| MD-3 | `foundation_llm` stores `d_latent=0` in `ckpt["args"]` meaning "derive as `max(d_model // 8, 64)`". A literal comparison with the configured 256 rejects every real base checkpoint. For dense checkpoints the deepseek knobs are unused CLI defaults. | `0` is resolved before comparing; deepseek-only knobs are compared only for `arch: deepseek`. |

## Choices on details the PRD does not fix (IG-2)

| Req | Question | Choice |
|---|---|---|
| TK-4 / TK-8 | Reasoning markers are configured as special tokens, but TK-4 encodes all dataset text with `encode_ordinary`, so the special ids would never appear in training data. | In assistant content only, the renderer emits the configured reasoning markers as their special ids; everything else stays ordinary text. User/system text never gets special ids. |
| SB-1 | Limits on open files and child processes have no configuration keys (CF-1). | Added `code_execution.max_open_files` and `code_execution.max_processes` (required). |
| DS-1 | `safe_label` (Appendix A, xstest) is not in DS-1; `chosen_rejected_messages` uses `role_key`/`content_key`, not listed in DS-2. | `eval_safety` datasets require `safe_label`; `role_key`/`content_key` are optional field_map keys for that adapter. |
| EV-1, CF-1 | "benchmarks listed" vs no defaults. | Every benchmark key is required; `null` leaves it out. |
| GN-3 | Meaning of `top_k: 0`, `top_p: 1.0`. | Disabled. Repetition penalty uses the CTRL rule (divide positive logits, multiply negative). |
| GN-1 | Contents of `completion_tokens`/`text` at a stop. | `completion_tokens` includes the stop token (it is trained, DP-7); `text` excludes it; stop strings trim `text` only. |
| MD-8 | Padding query rows have no visible key, giving NaN softmax. | Every query also sees itself; padding rows' outputs are discarded. |
| LR-1 | Warmup endpoint convention. | `lr(s) = lr * s / W` for `s < W` (step 0 has lr 0), HF convention. |
| RL-4 | Population or sample std. | Unbiased (Bessel), as in TRL's GRPOTrainer. |
| RW-2 | Plain numbers for `math_verify.parse`. | Wrapped in `$...$` before parsing; commas stripped from regex/last-number answers. |
| EX-2 | "match exactly" with `export.dtype: bf16`. | Exact for `fp32` export (tested); bf16 export matches a bf16 cast of the source. |

## Training loop, data and stage decisions (IG-2)

| Req | Question | Choice |
|---|---|---|
| TR-3, TR-5 | Which optimizer builder | `training/parallel.py` builds AdamW itself (decay/no-decay by parameter name captured before wrapping, every ZeRO stage). See prd_review #1, #2. |
| S1-1, S2-1 | LM normalisation | Both LM and SFT divide the summed cross-entropy by the global count of masked targets in the optimizer step and multiply by the world size (undoes DDP/FSDP averaging). |
| CK-3 | Optimizer state format | `named_full_v1`: full tensors keyed by parameter name for every ZeRO stage, assembled by hand under FSDP (prd_review #20). A resume under another world size restores model/optimizer/schedule exactly and rescales the data position. |
| DL-1 | Fewer shard files than ranks | Row-level sharding (`i % world == rank`); fewer rows than ranks: every rank reads all rows. Shuffle order depends on shard *name*, not URI. |
| DP-2, DP-5 | Memory | One dataset split's documents are held in memory for shuffled packing; very large corpora are bounded with `max_samples`. |
| DP-9 | Weights that do not sum to 1 | Normalised. Epoch stages: `rows_per_epoch = weight * total rows available`, interleaved in proportion to what is left of each quota. |
| DS-5 | What counts as a dropped record | Adapter failures and structural drops (too long for ctx, no assistant tokens, chosen == rejected). Decontamination drops are counted separately and do not trip `max_drop_fraction`. |
| DP-3, RL-1a | RL prompt rendering | Prompts are prepared with the stage's system message baked in (`prompts` kind: none), so training reads only prepared data. |
| adapters | Label for appended tests | `"Your code should pass these tests:"` is a fixed string in `data/adapters.py` (`TESTS_HEADER`), not configuration. |
| PO-5 | Mixture weight | `on_policy` joins the preference mixture with `on_policy.weight`; validation rejects listing it while disabled. Pair choice: highest score (first on ties) vs lowest; equal top and bottom skip the prompt. |
| DP-8 | Reference pass | Uses the training autocast dtype; rerun when the parent sha changes (`_ref.json`). |
| RL-2, RL-7a | Policy mode | RL and OPD forward passes run in eval mode (dropout off) so the sequence ratio is not perturbed. |
| RW-5 | `exclude` mode | Truncated responses get weight 0 instead of being removed, so every rank keeps the same micro-batch count. |
| RL-3a | No group kept anywhere | Update skipped on all ranks; LR schedule does not advance. |
| RL-7 | Step statistics | `grad_norm` is the mean over the step's mini-batch updates. |
| RL-10 vs RL-11 | Which checkpoint after an entropy stop | Best validation accuracy among checkpoints from before the collapse window; else the newest periodic checkpoint before it; else the parent. |
| RL-6 | Mismatch definition | prd_review #21. |
| PT-11 | Definition | Fraction of (sampled response token, MoE layer) pairs whose sorted top-k expert set differs between two forward passes on the same tokens, before and after the step's updates. |
| DI-1 | `max_prompts` across datasets | Round-robin over `prompts_datasets` until the total cap. |
| DI-2 | Held-out traces | Deterministic hash split by `prepared_data.holdout_fraction`; at least one trace goes to test when there are two or more. `teacher_traces` are packed like SFT data. |
| DI-7, RL-11 | `val` event fields | RL: `val_accuracy` (selection key) and `val_loss = 1 - accuracy`; OPD: `val_loss` = mean reverse KL. |
| EV-2 | Mode from lineage | No lineage (base) and stages `midtrain`, `midtrain_long` -> base; everything else -> chat. |
| B-1 | Scoring tokens | Sum of log-probs of all tokens after the first of `question\nAnswer: {choice}` (the shared prefix is a constant across choices). |
| B-3 | Base-mode MATH-500 | Same prompt file as a plain-text prompt. |
| B-9, B-10 | Mode and data | Chat only. B-10 uses the first prompts of the held-out split of `stages.sft.data.sources` (SFT enabled or not). |
| B-5 | IFEval checkers | `lm_eval.tasks.ifeval.utils.process_results`; needs NLTK data the first time. |
| PL-5 | Where the eval hash lives | `final/eval_report.json` (`eval_config_hash`); the gate is recomputed on every `pf gate`. |
| PL-7 | Launch | `torchrun` if on PATH else `python -m torch.distributed.run`; `PYTHONPATH` gets the package root; the subprocess always passes `--resume`. |
