# Implementation plan

The PRD is large, so the build is split into chunks. Each chunk ends with its own tests passing
before the next one starts; later chunks only depend on earlier ones. Status is tracked in
`README.md`, not here.

`foundation_llm` is read-only and only reachable through `src.foundation.bridge`
(FL-2..FL-5). The location comes from `foundation_llm.code_path` in the config (the
`..\foundation_llm` folder on Muni's machine; `PF_FOUNDATION_LLM` in tests).

| Chunk | Scope | Requirements | Files | Tests |
|---|---|---|---|---|
| 0 Foundations | config schema/loader/validation, storage, bridge, tokenizer and chat template, model loading, KV-cache inference, generation | CF-*, ST-*, FL-*, TK-*, MD-*, GN-*, DV-* | `config/`, `io/`, `foundation/`, `tokenization/`, `modeling/`, `generation/` | AT-1..AT-10, AT-26 |
| 1 Pure logic | objectives, rewards, sandbox, packing, decontamination, schedules, gates, metric maths, metrics logger, export | S1-1, S2-1, PO-3/4, RL-4/7, RW-*, DI-5, DP-4..6, LR-1, GT-*, EV-5, MT-L1a, EX-* | `training/objectives/`, `rl/`, `data/packing.py`, `data/decontam.py`, `eval/gates.py`, `eval/benchmarks/*` (maths), `export/`, `metrics/` | AT-11..AT-18, AT-22, AT-24, AT-28 |
| 2 Data registry, MoE/parallel | adapters, registry (DS-1..DS-5), ZeRO wrapping, optimizer groups, MoE routing-bias sync | DS-*, TR-3, TR-5, TR-5a, PT-9 | `data/adapters.py`, `data/registry.py`, `training/parallel.py`, `training/moe_balance.py` | registry, adapter, 2-process tests |
| 3 Data preparation and loaders | tokenize + pack + decontaminate into Parquet shards, prepared schemas, manifests, streaming loaders with rank sharding and resume, mixtures | DP-1..DP-7, DP-9, DL-1..DL-3 | `data/prepare.py`, `data/loaders.py`, `data/mixture.py` | prepared-schema, loader resume, mixture-count tests |
| 4 Training engine | one loop for every stage, checkpoints, lineage, resume, metrics events, LM and SFT stage drivers | TR-1..TR-7, CK-*, MT-L*, PT-9/10, S1-*, S1b-1, S2-* | `training/loop.py`, `training/checkpoint.py`, `training/lineage.py`, `stages/common.py`, `stages/midtrain.py`, `stages/sft.py` | AT-19 (midtrain, sft), AT-21 (bias switch) |
| 5 Services and preference | judge and teacher clients, cache, stub server for tests, DP-8 reference log-probs, PO-5 on-policy pairs, preference driver | JG-*, TC-1, DP-8, PO-* | `services/judge.py`, `services/teacher.py`, `stages/preference.py` | AT-19 (preference), judge cache/parse tests |
| 6 Generative stages | rollouts, dynamic sampling, GSPO training, entry gate, entropy floor, routing-change metric, RL-3a rank balancing, off- and on-policy distillation | RL-*, RW-*, PT-11, DI-*, TC-2 | `rl/rollouts.py`, `stages/rlvr.py`, `stages/distill.py` | AT-19 (rlvr, distill), AT-21, AT-23, AT-27 |
| 7 Evaluation | one runner per benchmark, suite, mode selection, `pf eval`, `pf gate` | EV-*, B-1..B-10, GT-* | `eval/suite.py`, `eval/benchmarks/*.py` | AT-22 (end to end) |
| 8 Pipeline | init_from resolution, config hash, resume, torchrun launch, final model selection, CLI commands | PL-*, CK-1, §15 | `stages/pipeline.py`, `cli.py` | AT-19, AT-20, AT-25, AT-29 |
| 9 Closing | README status, notes, PRD review update, packaging | IG-3, IG-7 | `README.md`, `docs/` | full suite |

## Order and dependencies

```
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
                     \___________/
                 4 needs 3 (loaders); 5 and 6 need 4 (loop); 8 needs everything
```

Chunk 7 could run before 5 and 6, but the pipeline test (AT-20) needs all of them, so the order
above keeps each chunk testable on its own.

## Ground rules used in every chunk

- No default values for configuration in code (CF-1). A number, token id, path or template that
  is not in the config comes from data or is derived from config values.
- Anything the PRD does not fix is decided, recorded in `docs/implementation_notes.md` and, when
  it changes behavior, covered by a test (IG-3).
- Anything the PRD gets wrong against the real `foundation_llm` code goes in `docs/prd_review.md`
  with how it was verified.
