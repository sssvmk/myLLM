# Keystone

Keystone is a post-training pipeline for causal language models produced by `foundation_llm`
(pretraining). It takes a pretrained checkpoint through mid-training, supervised fine-tuning,
preference optimization, reinforcement learning with verifiable rewards, and distillation, with an
evaluation suite and promotion gates between every stage. It is a from-scratch implementation of
the requirements in `prd_postfoundation.md`, built and validated against the real `foundation_llm`
code rather than against the PRD's prose alone.

`foundation_llm` itself is read-only: Keystone never edits it, and imports it through exactly one
module (`src/foundation/bridge.py`).

---

## 1. What it does

Given a base checkpoint (dense or DeepSeek-style MLA+MoE) and a set of datasets, Keystone runs up
to five post-training stages in order, each starting from the previous stage's output:

```
base checkpoint (foundation_llm)
   │
   ▼
Stage 1  midtrain          continued pretraining on a token-budgeted, weighted data mixture
   │      (optional Stage 1b: midtrain_long — same objective, longer context, RoPE theta changed)
   ▼
Stage 2  sft                supervised fine-tuning on chat-formatted conversations, loss on
   │                         assistant turns only
   ▼
Stage 3  preference          DPO or SimPO on preference pairs, optionally mixed with pairs the
   │                         model just generated and a judge just ranked (on-policy preference)
   ▼
Stage 4  rlvr                RL with verifiable rewards: GSPO objective, DAPO's dynamic sampling
   │                         and overlong handling, math and code reward functions
   ▼
Stage 5  distill             5a off-policy: train on a teacher model's responses to prompts
                              5b on-policy: student samples its own responses, reverse-KL to a
                              teacher's logits on those same responses
```

Every stage:
- resolves its own starting checkpoint (`init_from`: the base model, a named earlier stage, or an
  explicit checkpoint URI),
- prepares its own data once and caches the result (re-preparing only when an input changes),
- trains through one shared training loop with checkpointing, resume, and mixed precision,
- writes a full provenance record (parent checkpoint, data files used, config, git commit) next to
  its output,
- is evaluated by a benchmark suite and compared against its parent by a gate before its output can
  be used as the next stage's input or as the final model.

`pf run-pipeline` runs all of this end to end for a whole configuration file: skipping stages
that are disabled or already up to date, stopping or continuing past a failed gate as configured,
and exporting whichever stage's output is the last one to pass its gate.

## 2. Architecture support

Both of `foundation_llm`'s model architectures are supported, unmodified:

- **Dense** — standard multi-head attention with RoPE, GELU MLP.
- **DeepSeek-style** — Multi-head Latent Attention (MLA) + DeepSeekMoE (routed + shared experts,
  auxiliary-loss-free load balancing via a per-expert routing bias).

Keystone's own code (not `foundation_llm`'s) handles wrapping either architecture for distributed
training, because `foundation_llm`'s wrapper cannot run the DeepSeek architecture under FSDP (see
§7). It also fixes weight decay under FSDP and gives the MoE auxiliary loss an actual gradient —
also detailed in §7.

## 3. Components implemented

| Area | What's there |
|---|---|
| **Config** | Pydantic schema with no hidden defaults (every value must be in the config or the run refuses to start), OmegaConf-based YAML loading with `--set key=value` overrides, static and environment-reachability validation, secret redaction in saved configs. |
| **Storage** | One wrapper (`fsspec`-based) over local paths and `abfss://` / `s3://` / `gs://`, with staged uploads and local caching, so the rest of the code never branches on backend. |
| **Data pipeline** | 14 dataset kinds (text, packed tokens, conversations, preference pairs, prompts, math/code RL items, and 7 eval-only kinds), 16 field-mapping adapters over them, a registry that also derives held-out splits when a dataset has none, n-gram decontamination against eval sets, EOS-separated packing with document-boundary tracking, and a preparation step that tokenizes/packs/writes Parquet shards with input-hash caching so unchanged data is never reprocessed. |
| **Loading** | Streaming Parquet readers with rank-based sharding, resumable position tracking, and weighted multi-dataset mixture sampling (token-budget or epoch-based). |
| **Tokenizer & chat format** | Fully offline `cl100k_base` loading (no network dependency), a configurable chat template with special tokens for turns and an optional reasoning span. |
| **Model / inference** | KV-cached, explicit-position inference built on `foundation_llm`'s model code (not a copy of it), for both architectures, with batched sampling (temperature / top-k / top-p / repetition penalty) and multiple stopping conditions. |
| **Training loop** | One shared loop (`training/loop.py`) used by every stage: gradient accumulation, mixed precision (bf16/fp16/fp32), gradient clipping, MoE routing-bias synchronization across ranks, structured metrics (JSONL + optional TensorBoard), checkpoint/resume, and the final-artifact + lineage writer. Each stage supplies only its own loss ("objective"). |
| **Objectives** | LM cross-entropy, SFT (assistant-turn masking), DPO, SimPO, GSPO (with DAPO's clipping, dynamic sampling, overlong handling), reverse-KL on-policy distillation. |
| **Distributed training** | DDP (ZeRO stage 0), DDP + `ZeroRedundancyOptimizer` (stage 1), FSDP `SHARD_GRAD_OP` (stage 2), FSDP `FULL_SHARD` (stage 3) — Keystone's own wrapping, not `foundation_llm`'s (§7). |
| **Checkpointing** | Named, portable optimizer-state format that survives a change of ZeRO stage or process count, periodic + best + latest checkpoints, full lineage record, resumable at the exact training state. |
| **RL machinery** | Prompt pools, batched rollout generation, per-token log-prob recomputation for importance sampling, reward functions (math answer matching, code execution in a sandboxed subprocess, format checking, overlong penalty), a routing-change probe for MoE stability, an entry gate that can skip or stop Stage 4 if the base policy is too weak to learn from. |
| **Services** | OpenAI-compatible judge client (pairwise comparison, refusal classification, 1–10 scoring, response caching, retry-then-default-verdict) and teacher client (off-policy sampling), used by preference on-policy pairs, Stage 5's teacher traces, and two evaluation benchmarks. |
| **Evaluation** | One runner per benchmark — MMLU (+ calibration/ECE), GSM8K, MATH-500, HumanEval (pass@k), IFEval (via `lm-eval`'s checkers), AlpacaEval-style length-controlled win rate, safety refusal/compliance rates, response length, chat-template adherence — with automatic base-mode (few-shot, plain text) vs. chat-mode (zero-shot, templated) selection based on which stage produced the checkpoint. |
| **Gates** | Per-benchmark improvement/regression thresholds and absolute floors, comparing a stage's report against its resolved parent's, deciding whether the stage's output is eligible to be used further. |
| **Export** | A portable checkpoint (chosen dtype) plus its tokenizer, chat template, and lineage. |
| **CLI** | `pf validate-config`, `prepare-data`, `train`, `eval`, `gate`, `run-pipeline`, `generate`, `export` — see §6. |

178 tests exercise all of the above on CPU, including 2-process DDP/FSDP runs (see §8).

## 4. Repository layout

```
some-parent-dir/
├── foundation_llm/     read-only, never modified
└── keystone/           this project
    ├── src/            the code (import as `src`, e.g. `from src.config.loader import ...`)
    │   ├── config/          schema, YAML loading, validation
    │   ├── io/               storage backends
    │   ├── data/             adapters, registry, decontamination, packing, preparation, loaders, mixtures
    │   ├── tokenization/     offline tokenizer, chat template
    │   ├── modeling/         checkpoint loading, KV cache, inference
    │   ├── generation/       sampling and the batch generation API
    │   ├── training/         the shared loop, objectives, checkpoint/lineage, ZeRO wrapping, LR schedules
    │   ├── rl/                rollouts, reward functions, sandboxed code execution
    │   ├── services/          judge and teacher clients
    │   ├── stages/            one driver per pipeline stage, plus stage resolution and orchestration
    │   ├── eval/              benchmark runners, the suite, gates
    │   ├── export/            final checkpoint export
    │   ├── metrics/           JSONL/TensorBoard logger
    │   ├── foundation/        the only module that imports foundation_llm
    │   └── cli.py             the `pf` command
    ├── tests/               178 tests, one file per area
    ├── configs/             example_deepseek_1p3b.yaml (real-world shape), tiny_test.yaml (what tests run against)
    ├── prompts/             judge prompts, RL system prompts, few-shot/chat templates referenced by URI from configs
    ├── docs/
    │   ├── prd_review.md              problems found in the PRD, several reproduced by running foundation_llm
    │   ├── implementation_notes.md    every choice made where the PRD was silent, ambiguous, or wrong
    │   └── implementation_plan.md     how the build was split into chunks
    ├── NOTICE               no external repositories were read or ported for this build
    └── pyproject.toml
```

## 5. Configuration

Everything is one YAML file (OmegaConf-based, so `${oc.env:VAR}` interpolation and CLI `--set`
overrides both work) validated against a Pydantic schema with **no defaults anywhere** — every
value must be present in the file or under an environment variable it references, or the run
refuses to start. This is deliberate: nothing about a run is implicit.

Top-level sections:

| Section | Covers |
|---|---|
| `run` | name, seed, local/output paths, `device` (`auto`/`cpu`/`cuda`), `precision` (`bf16`/`fp16`/`fp32`) |
| `storage` | upload retries, per-protocol backend options |
| `foundation_llm` | `code_path` — where the read-only `foundation_llm` checkout lives |
| `base_model` | starting checkpoint URI and its architecture |
| `tokenizer` | ranks file URI, EOT/pad token ids, chat special tokens |
| `chat_template` | turn/generation-prompt/reasoning-span markers |
| `datasets` | one entry per dataset: URI, per-split file globs, format, adapter, field mapping, license, and whether it's third-party |
| `prepared_data` | where prepared shards go, worker count, shard size, holdout fraction, max allowed drop fraction, decontamination settings |
| `judge` / `teachers` | OpenAI-compatible endpoints for the judge and the two teacher roles, prompts, parsing regexes, retry/cache settings |
| `code_execution` | sandbox settings for code-reward execution |
| `generation` | default sampling settings used across stages |
| `stages` | one block per stage — see below |
| `eval` | which of the ten benchmarks are enabled and their settings (a benchmark set to `null` is skipped everywhere) |
| `gates` | improvement thresholds, tolerances, absolute floors, and whether a failed gate stops or just gets recorded |
| `export` | output location and dtype |
| `launcher` | `nnodes` / `nproc_per_node` for `torchrun`, extra torchrun args |
| `logging` | TensorBoard on/off, log/metric intervals |

**Every stage block** shares: `enabled`, `init_from` (`base`, another stage's name, `previous`, or an
explicit checkpoint URI), `output_uri`, optimizer settings, an LR schedule (warmup/stable/decay
fractions, linear/cosine/constant/WSD), batch size and gradient accumulation, ZeRO stage and
`find_unused_parameters`, MoE routing-bias settings, dropout, and checkpoint/eval intervals — then
its own data and objective settings on top:

- **`midtrain` / `midtrain_long`** — weighted dataset sources, a token budget, a retention-eval
  dataset (tracked separately so mixing in new data doesn't silently regress old capability).
  `midtrain_long` additionally overrides `ctx` and `rope_theta`.
- **`sft`** — weighted sources, epoch count.
- **`preference`** — `dpo` or `simpo`, `beta` (and `simpo_gamma` for SimPO), max total tokens per
  pair, epoch count, and an `on_policy` block (prompts dataset, samples per prompt, mixture weight,
  its own generation settings) to add judge-ranked self-generated pairs into the mix.
- **`rlvr`** — RL system prompts, an entry gate (sample size, k, minimum pass@k, skip-or-stop),
  prompts per step, group size, updates per step, total steps, GSPO clip epsilons, KL coefficient,
  DAPO settings (dynamic sampling, max resample rounds, overlong handling mode), reward weights,
  entropy floor (value + patience), routing-probe sample size.
- **`distill.offpolicy`** — prompt datasets, samples per prompt, max response length, epoch count.
- **`distill.onpolicy`** — prompts per step, total steps, its own generation settings.

`configs/example_deepseek_1p3b.yaml` is a complete, real-shaped example (Azure storage, a 1.3B
DeepSeek model, all five stages). `configs/tiny_test.yaml` is the minimal config the test suite
runs — a good second reference for what every key looks like at its smallest.

## 6. The `pf` CLI

| Command | Does |
|---|---|
| `pf validate-config -c FILE` | Schema + cross-field checks, then (unless `--skip-endpoints`) reachability checks: tokenizer file, base checkpoint, dataset globs, prompt files, a one-token ping to the judge/teacher endpoints |
| `pf prepare-data -c FILE --stage ID\|all [--force]` | Adapt, decontaminate, tokenize, pack one stage's (or every enabled stage's) datasets to Parquet; also produces on-policy preference pairs and teacher traces where a stage needs them |
| `pf train -c FILE --stage ID [--resume]` | Train one stage in the current process (launch with `torchrun` yourself for more than one process) |
| `pf eval -c FILE --checkpoint URI --out URI` | Run the evaluation suite on one checkpoint |
| `pf gate -c FILE --stage ID` | Compare a stage's evaluation report against its resolved parent's |
| `pf run-pipeline -c FILE` | prepare → train → eval → gate for every enabled stage, in order, then export the last stage that passed its gate. Resumable: stages whose inputs haven't changed are skipped (checked by a hash of everything that could affect them) |
| `pf generate -c FILE --checkpoint URI (--prompt TEXT \| --prompts-file FILE) [--chat]` | KV-cached generation for a quick manual check of a checkpoint |
| `pf export -c FILE [--checkpoint URI]` | Export a checkpoint (default: the pipeline's final model) to a portable format |

Every command takes `-c/--config FILE` and any number of `--set key.path=value` overrides.

## 7. Known limitations and fixes worth knowing before you run this

- **FSDP + DeepSeek arch, and weight decay under FSDP**: `foundation_llm`'s own model-wrapping and
  optimizer-building code cannot run the DeepSeek architecture under ZeRO 2/3, and silently zeroes
  weight decay under FSDP generally. Keystone does its own wrapping and decay-group selection
  (`src/training/parallel.py`) and is unaffected by either bug — but `foundation_llm` itself, run
  directly, still has them.
- **MoE auxiliary loss**: `foundation_llm`'s own aux loss carries no gradient (it's built from
  routing counts). Keystone replaces it with a differentiable DeepSeek-V2-style balance loss via
  forward hooks (`src/training/moe_balance.py`), so `aux_loss_weight` now actually affects
  training.
- **FSDP optimizer checkpoint/resume**: PyTorch's own full-optimizer-state gather under FSDP with
  `use_orig_params=True` returns wrong values (verified independently of this model or codebase).
  Keystone assembles the state itself and stores it by parameter name, which also means a run can
  resume under a *different* ZeRO stage or process count (the data read order after such a resume
  is freshly seeded rather than reproduced exactly).
- **GPU/NCCL is implemented but not run here.** This sandbox has no GPU. All 178 tests are CPU
  (`gloo`), including 2-process DDP and FSDP at every ZeRO stage. `device: cuda`, `bf16` on real
  hardware, and multi-GPU-per-node all use the same code paths, but haven't been verified on actual
  hardware.
- **Multi-node is rejected by `pf run-pipeline`** (`launcher.nnodes > 1` fails validation) — the
  pipeline command is scoped to one node; a single stage could still be launched multi-node by hand
  with `pf train`, but that path isn't tested here.
- Full details, including every other PRD ambiguity or contradiction found and how it was resolved,
  are in `docs/prd_review.md` and `docs/implementation_notes.md`.

## 8. Papers and how they contributed

Every non-obvious algorithm, formula, or design choice in Keystone traces back to a specific
paper, not to invention. The PRD this was built from cites 39 papers by ID (`R1`–`R39`) against
its requirements; the table below is the subset that actually became code, organized by where in
the pipeline it landed, with a one-line statement of what was taken and where to find it.

### 8.1 Architecture (reused from `foundation_llm`, exercised by Keystone's training/inference code)

| Paper | What was taken | Where |
|---|---|---|
| DeepSeek-AI, **DeepSeek-V2** (MLA), arXiv 2405.04434 | Multi-head Latent Attention: keys/values compressed into a shared low-rank latent, a small decoupled RoPE key carries position, so only the latent and the RoPE key need caching at inference. | `src/modeling/inference.py`'s KV cache; the architecture itself is `foundation_llm`'s. |
| DeepSeek-AI, **DeepSeek-V3**, arXiv 2412.19437 | Auxiliary-loss-free MoE load balancing: a per-expert bias added to routing scores for top-k selection only, nudged after every step by that expert's load. | `src/training/moe_balance.py` (usage summed across ranks and micro-batches before the update, fixing a drift bug in `foundation_llm`'s own version — see §7). |
| Xiong et al., **Effective Long-Context Scaling of Foundation Models**, arXiv 2309.16039 | Raise the RoPE base frequency and continue training on longer sequences to extend context. | Stage 1b (`midtrain_long`), which overrides `ctx`/`rope_theta` and nothing else. |
| Hu et al., **MiniCPM** (WSD schedule), arXiv 2404.06395 | Warmup–Stable–Decay: hold the LR constant, then decay over a short final phase for a large loss drop. | `src/training/schedules.py`, the `wsd` schedule type. |

### 8.2 Supervised fine-tuning (Stage 2)

| Paper | What was taken | Where |
|---|---|---|
| Lambert et al., **Tülu 3**, arXiv 2411.15124 | Stage order (SFT → preference → RLVR); assistant-only SFT loss; n-gram decontamination of training data against every eval set; on-policy judged preference pairs; the shape of the evaluation suite. | Pipeline stage order (`src/stages/pipeline.py`); `src/training/objectives/sft.py` (loss masking); `src/data/decontam.py`. |
| Zhou et al., **LIMA**, arXiv 2305.11206 | SFT is judged by behavior (template adherence, instruction following), not by pushing knowledge benchmarks up. | `eval/benchmarks` template-adherence check (B-10); the gate only requires knowledge benchmarks not to regress. |
| Gekhman et al., **Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?**, arXiv 2405.05904 | Stop at the checkpoint with the lowest validation loss, not the last step, to avoid memorizing facts the base model doesn't know. | The SFT objective's `best_mode="min"` selection in `src/training/loop.py` / `src/stages/common.py`. |

### 8.3 Preference optimization (Stage 3)

| Paper | What was taken | Where |
|---|---|---|
| Rafailov et al., **Direct Preference Optimization**, arXiv 2305.18290 | The closed-form optimal policy under a KL constraint turns into a classification loss on preference pairs — no reward model, no sampling. | `src/training/objectives/dpo.py`; reference log-probs precomputed once against the frozen parent (`stages/preference.py`'s `add_reference_logps`, DP-8). |
| Meng et al., **SimPO**, arXiv 2405.14734 | Length-normalized log-likelihood as an implicit, reference-free reward, plus a target margin γ. | `src/training/objectives/simpo.py`; selectable via `stages.preference.objective: simpo`. |
| Cui et al., **UltraFeedback**, arXiv 2310.01377; Bai et al., **Constitutional AI**, arXiv 2212.08073 | Several responses per prompt scored by a judge; build a preference pair from the scores (here: highest vs. lowest, for the largest gap — one deliberate difference from UltraFeedback's own top-vs-random pairing). | `stages/preference.py`'s `generate_on_policy_pairs`; the AI-judge-as-labeler idea generally (`services/judge.py`). |
| Zheng et al., **MT-Bench / LLM-as-a-Judge**, arXiv 2306.05685 | Judges have position and verbosity bias; mitigate by judging both answer orders and averaging. | `eval/suite.py`'s `run_alpaca`, which calls the judge with the model's response as both "A" and "B". |

### 8.4 RL with verifiable rewards (Stage 4)

| Paper | What was taken | Where |
|---|---|---|
| Shao et al., **DeepSeekMath** (GRPO), arXiv 2402.03300 | Sample a group of responses per prompt; advantage = reward normalized by the group's own mean/std; no learned value model. | `src/rl/rollouts.py` (group sampling), `src/training/objectives/gspo.py`'s `group_advantages`. |
| Liu et al., **Dr. GRPO**, arXiv 2503.20783 | Dividing advantages by the group standard deviation over-weights prompts that are almost always right or wrong; make that normalization switchable. | `advantage_std_normalization` in the RLVR config, read by `group_advantages`. |
| Zheng et al., **Group Sequence Policy Optimization**, arXiv 2507.18071 | Define the importance ratio over the *whole response* (not per-token) and clip at that level — token-level ratios are unstable for MoE models because expert routing shifts between updates. | `src/training/objectives/gspo.py` (`gspo_loss`, `gspo_per_sequence`); this is the loss Stage 4 actually trains on. |
| Yu et al., **DAPO**, arXiv 2503.14476 | Clip-Higher (asymmetric clip bounds so entropy doesn't collapse); dynamic sampling (drop and resample all-correct/all-wrong groups); soft overlong penalty near the length limit. | `src/stages/rlvr.py` (dynamic sampling, entropy monitoring), `src/rl/rewards/overlong.py`. |
| DeepSeek-AI, **DeepSeek-R1**, arXiv 2501.12948 | Rule-based rewards only (correctness + format), reasoning wrapped in `<think>` tags. | `src/rl/rewards/math.py`, `format.py`; the chat template's reasoning span. |
| Yue et al., **Does RL Really Incentivize Reasoning Capacity...**, arXiv 2504.13837 | RL mostly sharpens answers already reachable by the base model — so check the base model can reach them at all before spending RL compute. | The RLVR entry gate (`stages/rlvr.py`'s `entry_gate`), which requires a minimum pass@k before training starts. |
| Gao et al., **Scaling Laws for Reward Model Overoptimization**, arXiv 2210.10760 | A policy's true quality can fall even as its optimized (proxy) reward keeps rising — keep the checkpoint with the best *held-out* score, not the last one. | RLVR's `best_mode="max"` on validation accuracy (`training/loop.py`). |

### 8.5 Distillation (Stage 5)

| Paper | What was taken | Where |
|---|---|---|
| Qwen Team, **Qwen3 Technical Report**, arXiv 2505.09388 | Smaller models are distilled strong-to-weak: off-policy first, then on-policy. | Stage order 5a → 5b in `src/stages/pipeline.py`. |
| Agarwal et al., **On-Policy Distillation** (GKD), arXiv 2306.13649; Gu et al., **MiniLLM**, arXiv 2306.08543 | Train the student on its *own* generations, with the teacher scoring those same sequences token-by-token; reverse KL is mode-seeking and keeps the student from spreading mass over regions the teacher considers unlikely. | `src/training/objectives/opd.py`; `stages/distill.py`'s on-policy driver. |
| Thinking Machines Lab, **On-Policy Distillation**, Oct. 2025 | The exact per-token loss: `stopgrad(logp_student − logp_teacher) · logp_student`, averaged over response tokens — one student sample per prompt is enough. | `src/training/objectives/opd.py`'s loss function, used as-is. |

### 8.6 Evaluation

| Paper | What was taken | Where |
|---|---|---|
| Dubois et al., **Length-Controlled AlpacaEval**, arXiv 2404.04475 | A GLM predicting judge preference from a length-difference term; read off the win rate with that term zeroed to remove the length bias. | `eval/benchmarks/alpaca_lc.py`. |
| Zhou et al., **IFEval**, arXiv 2311.07911 | Programmatically checkable instructions (word counts, formats, keywords), scored strict/loose at prompt- and instruction-level. | `eval/benchmarks/ifeval.py`, using `lm-eval`'s own checkers directly rather than reimplementing them. |
| Chen et al., **HumanEval**, arXiv 2107.03374 | Functional correctness via unit-test execution; the unbiased pass@k estimator. | `eval/benchmarks/humaneval.py`; `rl/sandbox.py` for the isolated execution the paper calls for. |
| Röttger et al., **XSTest**, arXiv 2308.01263; Bianchi et al., **Safety-Tuned LLaMAs**, arXiv 2309.07875 | Measure both unsafe-prompt refusal *and* safe-lookalike compliance — a safety fix that only refuses more isn't actually safer. | `eval/suite.py`'s `run_safety`, which reports both rates. |
| Singhal et al., **A Long Way to Go: Length Correlations in RLHF**, arXiv 2310.03716 | Much of RLHF's apparent gain is just responses getting longer — track length directly and cap how much a stage is allowed to grow it. | The `length` benchmark and the gate's `length_ratio_max` threshold. |
| OpenAI, **GPT-4 Technical Report**, arXiv 2303.08774 | Post-training measurably degrades calibration on knowledge benchmarks — track it. | `eval/benchmarks/calibration.py` (expected calibration error on MMLU). |

### 8.7 Deliberately not implemented

A few well-known methods were read and left out on purpose, matching the PRD's stated scope:
**KTO** (arXiv 2402.01306, unpaired preference signals — every preference source here is paired),
**ORPO** (arXiv 2403.07691, merges SFT and preference into one stage — kept separate here so each
has its own gate), **process reward models** / *Let's Verify Step by Step* (arXiv 2305.20050,
needs step-level human labels this pipeline doesn't have), and **YaRN** (arXiv 2309.00071,
non-uniform RoPE interpolation — Stage 1b uses the simpler base-frequency change instead).

The full 39-paper list, including which ones only *guided* a threshold or metric rather than
supplying an algorithm, is in the original PRD's §20; `docs/implementation_notes.md` and
`docs/prd_review.md` cover every place an implementation had to depart from what a cited paper (or
the PRD's reading of it) technically specifies.

## 9. Running it

### 9.1 Setup

```bash
cd keystone
pip install -e ".[dev]"
export PF_FOUNDATION_LLM=/path/to/foundation_llm   # defaults to ../foundation_llm
```

`foundation_llm` is imported only through `src.foundation.bridge`, from `foundation_llm.code_path`
in the config.

### 9.2 Validate a config before doing anything else

```bash
pf validate-config -c configs/example_deepseek_1p3b.yaml
```

### 9.3 Run the whole pipeline

```bash
pf run-pipeline -c configs/example_deepseek_1p3b.yaml
```

This prepares data, trains, evaluates and gates every enabled stage in order, then exports the
final model. Re-running the same command later only redoes what changed (compared by a content
hash of the stage's config, its datasets, and its parent checkpoint).

### 9.4 Or run it stage by stage

```bash
pf prepare-data -c configs/example_deepseek_1p3b.yaml --stage midtrain
pf train -c configs/example_deepseek_1p3b.yaml --stage midtrain

# multi-GPU / multi-process, one node:
torchrun --nproc_per_node=8 -m src.cli train -c configs/example_deepseek_1p3b.yaml --stage midtrain --resume

pf eval -c configs/example_deepseek_1p3b.yaml --checkpoint <stage output>/final/model.pt --out <somewhere>
pf gate -c configs/example_deepseek_1p3b.yaml --stage midtrain
```

`pf run-pipeline` launches multi-process stages the same way automatically when
`launcher.nproc_per_node > 1` in the config.

### 9.5 A quick manual check of a checkpoint

```bash
pf generate -c configs/example_deepseek_1p3b.yaml --checkpoint <uri> --chat --prompt "..."
```

### 9.6 Tests

```bash
export PF_FOUNDATION_LLM=/path/to/foundation_llm
pytest -q
```

All 178 tests pass on CPU. Run in batches with `/tmp` cleared between them if disk space is tight
(each test's checkpoints and Parquet shards land in a temp directory not cleaned up until the
process exits; a single-process full run wants roughly 4-5 GB of temp space at once). IFEval
scoring needs the `lm-eval` package and downloads NLTK data on first use.
