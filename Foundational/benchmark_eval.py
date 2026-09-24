"""Downstream benchmark evaluation (MMLU/GSM8K/HumanEval-style), separate from train.evaluate's
held-out LM loss/perplexity.

Is this "part of foundational training"? Yes -- every major foundation-model report (GPT-3,
PaLM, LLaMA, DeepSeek, etc) reports benchmark scores like MMLU/GSM8K/HumanEval alongside
pretraining loss, precisely because loss/perplexity alone doesn't tell you what the model can
actually DO. It's standard practice, not an optional extra -- see prd.md.

What's actually implemented here vs. not, and why the split:
  - MMLU (`evaluate_mmlu`): multiple-choice, scored via log-likelihood comparison across the
    answer choices. This needs only a forward pass -- no sampling, no external tooling -- so
    it fits directly into this training-only codebase and IS implemented and tested below.
  - GSM8K: needs free-form generation (the model must produce a full solution) plus numeric-
    answer extraction and exact-match scoring. This codebase has no generation/sampling loop
    (train.py only ever computes loss on given token sequences) -- NOT implemented here.
  - HumanEval: needs generation plus SANDBOXED CODE EXECUTION for pass@k. Same generation gap
    as GSM8K, plus a code sandbox this codebase has no story for at all -- NOT implemented.
Adding real generation support (KV-cached autoregressive sampling) is the actual prerequisite
for GSM8K/HumanEval; once that exists, wiring in exact-match and pass@k scoring is comparatively
small additional work.

Dataset source: config.EVAL_BENCHMARK_SOURCES (same entries used by
data_quality.decontaminate_against_eval_sets for the raw text).
"""
from __future__ import annotations
import json
from typing import List, Optional
import torch
import torch.nn.functional as F

from config import EVAL_BENCHMARK_SOURCES


def _sequence_loglik(model, tokens: torch.Tensor, device) -> float:
  """Total log-likelihood the model assigns to `tokens` (1D long tensor, already on CPU),
  scored the standard multiple-choice-eval way: sum of per-token log p(token | prefix)."""
  if tokens.numel() < 2:
    return 0.0
  idx = tokens.unsqueeze(0).to(device)
  xb, yb = idx[:, :-1], idx[:, 1:]
  with torch.no_grad():
    logits, _ = model(xb)
    logprobs = F.log_softmax(logits, dim=-1)
    tok_logprobs = logprobs.gather(-1, yb.unsqueeze(-1)).squeeze(-1)
  return tok_logprobs.sum().item()


def evaluate_mmlu(model, tokenizer, examples: List[dict], device, max_examples: Optional[int] = None) -> dict:
  """Log-likelihood MMLU scoring: for each example, score `question + choice` for every
  choice and predict the highest-log-likelihood one. This is the standard lightweight way to
  MC-evaluate a base (non-instruction-tuned) LM without needing it to reliably follow an
  "answer with A/B/C/D" instruction.

  `examples`: list of {"question": str, "choices": [str, str, str, str], "answer": int}
  (matches cais/mmlu's schema once loaded via `datasets.load_dataset("cais/mmlu", "all")` --
  see config.EVAL_BENCHMARK_SOURCES["mmlu"]).

  Returns {"accuracy": float, "n": int, "correct": int}.
  """
  model.eval()
  if max_examples is not None:
    examples = examples[:max_examples]

  correct = 0
  for ex in examples:
    scores = []
    for choice in ex["choices"]:
      prompt = f"{ex['question']}\nAnswer: {choice}"
      ids = tokenizer.encode(prompt)
      scores.append(_sequence_loglik(model, torch.tensor(ids, dtype=torch.long), device))
    predicted = max(range(len(scores)), key=lambda i: scores[i])
    if predicted == ex["answer"]:
      correct += 1

  model.train()
  n = len(examples)
  return {"accuracy": correct / n if n else 0.0, "n": n, "correct": correct}


def load_local_mmlu(path: str) -> List[dict]:
  """Loads MMLU examples from a local JSONL file (one {"question","choices","answer"} object
  per line) -- for environments without direct HuggingFace access at eval time. Fill in
  config.EVAL_BENCHMARK_SOURCES["mmlu"]["local_path"] and point this at it. Convert from the
  HF dataset once, on a machine with access, via:
      from datasets import load_dataset
      ds = load_dataset("cais/mmlu", "all")["test"]
      with open("mmlu_test.jsonl", "w") as f:
          for ex in ds:
              f.write(json.dumps({"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]}) + "\\n")
  """
  examples = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        examples.append(json.loads(line))
  return examples
