"""Batched, KV-cached generation (GN-1..GN-8)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from ..modeling.inference import InferenceModel
from ..modeling.kv_cache import KVCache
from .sampler import sample_next

log = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    num_samples: int
    seed: int
    stop_token_ids: List[int]
    stop_strings: List[str]
    batch_size: int
    use_kv_cache: bool
    return_logprobs: bool
    overflow: str

    @classmethod
    def from_block(cls, block, *, num_samples: int, stop_token_ids: Sequence[int],
                   stop_strings: Sequence[str] = (), return_logprobs: bool = False, **overrides):
        d = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        d.update(overrides)
        return cls(num_samples=num_samples, stop_token_ids=list(stop_token_ids), stop_strings=list(stop_strings),
                   return_logprobs=return_logprobs, **d)


@dataclass
class GenerationResult:
    prompt_tokens: List[int]
    completion_tokens: List[int]
    text: str
    finish_reason: str                      # eos | stop | length | ctx
    logprobs: Optional[List[float]] = None
    truncated_prompt_tokens: int = 0


def _prepare_prompt(p: List[int], cfg: GenerationConfig, ctx: int) -> tuple:
    if len(p) + cfg.max_new_tokens <= ctx:
        return p, 0
    if cfg.overflow == "error":
        raise ValueError(f"prompt ({len(p)}) + max_new_tokens ({cfg.max_new_tokens}) > ctx ({ctx})")
    keep = ctx - cfg.max_new_tokens
    if keep <= 0:
        raise ValueError(f"max_new_tokens {cfg.max_new_tokens} leaves no room for a prompt at ctx {ctx}")
    dropped = len(p) - keep
    log.info("truncate_left: dropped %d prompt tokens", dropped)
    return p[-keep:], dropped


@torch.inference_mode()
def generate(model: InferenceModel, tokenizer, prompts: Sequence[List[int]], cfg: GenerationConfig,
             autocast_dtype: Optional[torch.dtype] = None) -> List[GenerationResult]:
    device = model.device
    model.model.eval()
    allowed = tokenizer.allowed_mask(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    expanded = [list(p) for p in prompts for _ in range(cfg.num_samples)]          # GN-7
    results: List[GenerationResult] = []
    for b0 in range(0, len(expanded), cfg.batch_size):
        batch = expanded[b0:b0 + cfg.batch_size]
        results.extend(_generate_batch(model, tokenizer, batch, cfg, allowed, gen, device, autocast_dtype))
    return results


def _generate_batch(model, tokenizer, raw_prompts, cfg, allowed, gen, device, autocast_dtype):
    ctx = model.ctx
    prepared = [_prepare_prompt(p, cfg, ctx) for p in raw_prompts]
    prompts = [p for p, _ in prepared]
    B = len(prompts)
    L = max(len(p) for p in prompts)
    pad = tokenizer.pad_token_id
    ids = torch.full((B, L), pad, dtype=torch.long, device=device)
    amask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):                                              # GN-2: left pad
        ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        amask[i, L - len(p):] = True
    pos = (amask.long().cumsum(1) - 1).clamp_min(0)
    seen = torch.zeros(B, tokenizer.vocab_rows, dtype=torch.bool, device=device)
    seen.scatter_(1, ids, amask)

    stop_ids = torch.tensor(sorted(set(cfg.stop_token_ids)), dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    finish = [None] * B
    out_tokens: List[List[int]] = [[] for _ in range(B)]
    out_lp: List[List[float]] = [[] for _ in range(B)]
    stop_text_cut: List[Optional[int]] = [None] * B
    ac_enabled = autocast_dtype is not None
    cache_dtype = autocast_dtype if ac_enabled else torch.float32               # DV-5

    def fwd(x, p, m, cache):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype or torch.float32, enabled=ac_enabled):
            return model(x, position_ids=p, attention_mask=m, kv_cache=cache)

    cache = KVCache(model.model, B, L + cfg.max_new_tokens, cache_dtype, device) if cfg.use_kv_cache else None
    all_ids, all_mask, all_pos = ids, amask, pos
    logits = fwd(ids, pos, amask, cache)[:, -1, :]
    last_pos = pos[:, -1]
    for step in range(cfg.max_new_tokens):
        nxt, lp = sample_next(logits, allowed, seen, cfg.temperature, cfg.top_k, cfg.top_p,
                              cfg.repetition_penalty, gen)
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        for i in range(B):
            if done[i]:
                continue
            t = int(nxt[i])
            out_tokens[i].append(t)
            out_lp[i].append(float(lp[i]))
            seen[i, t] = True
            if stop_ids.numel() and bool((stop_ids == t).any()):
                finish[i], done[i] = "eos", True
            elif cfg.stop_strings:
                text = tokenizer.decode(out_tokens[i])
                hits = [text.find(s) for s in cfg.stop_strings if s in text]
                if hits:
                    finish[i], done[i], stop_text_cut[i] = "stop", True, min(hits)
            if not done[i] and len(prompts[i]) + len(out_tokens[i]) >= ctx:
                finish[i], done[i] = "ctx", True
        if bool(done.all()):
            break
        new_pos = (last_pos + 1).clamp_max(ctx - 1)
        last_pos = new_pos
        if cache is not None:
            logits = fwd(nxt[:, None], new_pos[:, None], torch.ones(B, 1, dtype=torch.bool, device=device), cache)[:, -1, :]
        else:
            all_ids = torch.cat([all_ids, nxt[:, None]], dim=1)
            all_mask = torch.cat([all_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)
            all_pos = torch.cat([all_pos, new_pos[:, None]], dim=1)
            logits = fwd(all_ids, all_pos, all_mask, None)[:, -1, :]
    results = []
    for i in range(B):
        reason = finish[i] or "length"
        toks = out_tokens[i]
        body = toks[:-1] if reason == "eos" else toks
        text = tokenizer.decode(body)
        if stop_text_cut[i] is not None:
            text = text[:stop_text_cut[i]]
        results.append(GenerationResult(prompt_tokens=prompts[i], completion_tokens=toks, text=text,
                                        finish_reason=reason,
                                        logprobs=out_lp[i] if cfg.return_logprobs else None,
                                        truncated_prompt_tokens=prepared[i][1]))
    return results
