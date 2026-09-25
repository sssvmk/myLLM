# PRD review: prd_postfoundation.md

Checked against the `foundation_llm` code (run, not just read) and for internal consistency.
Items 1–5, 19, 20 and 21 were reproduced by executing code; the rest are text-level.

## Blocking: the PRD as written cannot run

1. **DeepSeek + ZeRO 2/3 fails at wrap time (TR-5a).** `wrap_model`'s FSDP rejects
   `DeepSeekMoE` blocks because `routing_bias` has `requires_grad=False`. Appendix A uses ZeRO 2/3
   with `arch: deepseek` for midtrain, midtrain_long, sft, preference and distill_offpolicy.
   *Amend TR-5a:* ZeRO 2/3 are wrapped by Keystone with `use_orig_params=True`.
   This also means `foundation_llm` itself cannot have pretrained the DeepSeek arch at ZeRO 2/3.
2. **Weight decay is silently zero under FSDP (TR-3).** `build_optimizer_for_zero` classifies
   by `p.dim() < 2`, and every FSDP parameter is 1-D. *Amend TR-3/TR-5a:* decay groups are chosen
   by parameter name before wrapping; the optimizer is built by Keystone.
3. **MoE aux loss carries no gradient (TR-2, PT-9).** It is computed from routing counts.
   `aux_loss_weight` only changed the logged loss. **Fixed in Keystone** (`training/moe_balance.py`):
   forward hooks replace the returned aux loss with the DeepSeek-V2 expert-level balance loss
   `aux_loss_weight * n_routed * sum_i f_i * P_i` (f: assignment fractions, constant; P: mean router
   probability, differentiable). *Amend TR-2/PT-9 to say so.* Tested: value against an independent
   recomputation, gradient reaches the router, minimising it rebalances routing, and the
   optimizer step's gradient norm changes with `aux_loss_weight`. With weight 0 the module's own
   value is untouched.
4. **MD-3 rejects real base checkpoints.** `ckpt["args"]["d_latent"]` is `0` ("auto"), vs 256
   configured. *Amend MD-3:* resolve 0 as `max(d_model // 8, 64)`; compare deepseek-only knobs
   only for deepseek.
5. **EX-2 vs Appendix A.** Exact logit equality is impossible with `export.dtype: bf16` from an
   fp32 checkpoint. *Amend EX-2:* exact when dtypes match; otherwise equal to the source cast to
   the export dtype.

## Contradictions and gaps

6. **TK-4 vs TK-8/TK-2:** reasoning markers are special tokens, but dataset text is always
   `encode_ordinary`, so the `<think>` ids never reach training data. Say whether the renderer
   inserts them (implemented choice) or whether they are plain text (then drop the ids).
7. **RL-10 vs RL-11:** when the entropy floor stops Stage 4, is `final/model.pt` "the last
   checkpoint before the collapse window" or "the best validation accuracy"?
8. **SB-1 vs CF-1:** open-file and process limits have no config keys (added two keys).
9. **DS-1/DS-2 vs Appendix A:** `safe_label` is undefined; `role_key`/`content_key` for
   `chosen_rejected_messages` are not listed.
10. **Cross-mode gate comparisons (GT-1):** SFT's parent is scored in base mode (8-shot GSM8K),
    SFT in chat mode (0-shot). `gsm8k_acc` tolerance then compares two protocols. Compare only
    same-mode metrics, or also evaluate the parent in chat mode.
11. **B-9 length** is not marked chat-only; in base mode greedy runs to `max_new_tokens`, which
    makes the SFT length-ratio check meaningless.
12. **B-10** says "held-out SFT prompts" without naming the dataset, or what happens when SFT is
    disabled.
13. **PL-5** hashes the parent checkpoint on every resume check (~23 GB for the 1.3B DeepSeek
    fp32 checkpoint). Store the sha once in the parent's `_COMPLETE`/lineage and reuse it.
14. **TR-3 fp16:** FSDP needs `ShardedGradScaler`, not the plain scaler.
15. **DP-8:** reference log-probs must use the same autocast dtype as training, or AT-15's
    "log 2 at step 0" fails under bf16.
16. **S2-1:** normalizing by the global target count interacts with DDP gradient averaging
    (multiply by world size). Worth one sentence.
17. Small: `top_k: 0`/`top_p: 1.0` semantics (GN-3), sample vs population std (RL-4), and
    `finish_reason: "ctx"` being unreachable given GN-6.

18. **DP-4 vs Appendix A's GSM8K entry.** `gsm8k_test` exposes its *train* split as `fewshot`, so
    "every text field of every dataset in `against`" drops all of `gsm8k_train`. Implemented with an
    explicit `prepared_data.decontamination.splits` key (Appendix A: `[test]`).
19. **PT-9 under data parallelism.** `DeepSeekMoE.update_bias()` reads the local last micro-batch,
    so biases drift between ranks (they have `requires_grad=False`, so DDP/FSDP never synchronise
    them), and under FSDP `use_orig_params=True` the bias is sharded, so `update_bias()` raises on
    ranks that hold none of it. Fixed in `training/moe_balance.py`: usage is summed over all
    micro-batches of the step and over ranks, and `routing_bias` is kept out of FSDP
    (`ignored_states`). Reproduced with torch 2.14, 2 processes.
20. **CK-3 / TR-5 with ZeRO 2/3: the full optimizer state torch returns is wrong.** With
    `FSDP(use_orig_params=True)` both `FSDP.optim_state_dict` (what `full_optimizer_state_dict`
    uses) and `torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` return an
    Adam `exp_avg` that differs from the ground truth by as much as the values themselves
    (measured on torch 2.14, 2 CPU/gloo processes, on a two-layer `Linear`, so it is not specific
    to this model, to MoE, to tied weights or to parameter groups). With `use_orig_params=False`
    the same call is exact. Training under `use_orig_params=True` is correct (parameters and local
    moment shards match a plain AdamW step exactly); only the gather is wrong, so a checkpoint
    that used it would resume on wrong moments. **Fixed in Keystone:** `training/checkpoint.py`
    assembles the state itself (each rank's local piece of a parameter is a contiguous slice,
    slices follow rank order; the total is checked against the parameter's full shape) and
    stores it by parameter name (`named_full_v1`, used for every ZeRO stage). Tested against a plain
    AdamW step for ZeRO 2 and 3, resume exactness at ZeRO 0-3, and resume in one process from a
    two-process run (and the reverse). Because the state is portable, a resume under a different
    world size restores model, optimizer and LR schedule exactly; the data position is rescaled and
    the data order afterwards is a fresh seeded one (a message says so). Not checked on GPU/NCCL.
21. **RL-6 vs GN-4.** Generation log-probs come from the distribution with pad and undefined ids
    masked (GN-3 step 1), the training forward from the full softmax (TR-4). A model that puts mass
    on the ~98k unused rows of `vocab_rows` therefore shows a large systematic "mismatch" that has
    nothing to do with numerics. The metric compares the generation log-probs with the training
    forward's log-probs renormalised over the same allowed ids; the loss still uses the full
    softmax.
22. **PL-5 does not say what happens to a completed stage whose hash changed.** Its outputs would
    stay eligible as a parent. They are now removed (`_COMPLETE`, `_SKIPPED`, `final/`) before the
    stage retrains, and a checkpoint written under another hash is never resumed.
23. **RL-1 `skip` and PL-5.** The skip decision is recorded in `_SKIPPED` together with the
    configuration hash it was made under; otherwise a re-run would repeat the entry gate every time.
24. **DP-6 / DI-2 on tiny data.** A held-out split can legitimately be empty after packing (every
    conversation longer than `ctx + 1`). Such a split is left out rather than written empty, and
    validation uses the sources that have one.

## Not verifiable here

HuggingFace dataset paths and globs in Appendix A (no `huggingface.co` access from this
sandbox). `pf validate-config` checks them on the target environment (CF-7).
