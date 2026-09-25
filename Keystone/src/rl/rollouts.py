"""Rollout machinery for the generative stages (RL-2, RL-6, PT-11, DI-4): prompt pools, group
generation, reward scoring, per-token log-prob passes and the routing-change probe."""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..data import loaders, prepare
from ..generation.api import GenerationConfig, generate
from ..modeling.inference import InferenceModel, routing_topk
from ..modeling.loading import moe_modules
from ..training.objectives.common import token_entropy, token_logps


@dataclass
class Rollout:
    item: Dict[str, Any]
    prompt: List[int]
    completion: List[int]
    logprobs: List[float]
    text: str
    finish: str


# --------------------------------------------------------------------------- prompts
def make_item(row: Dict[str, Any], kind: str) -> Dict[str, Any]:
    it = {"kind": kind, "prompt": row["prompt_tokens"], "source": row.get("source")}
    if kind == "rl_math":
        it["ground_truth"] = row["ground_truth"]
    elif kind == "rl_code":
        it["tests"], it["entry_point"] = row["tests"], row["entry_point"]
    return it


class PromptPool:
    """Prepared prompts of a stage held in memory. `draw` samples with replacement, weighted by
    dataset (RL-2); `fixed` returns a deterministic held-out list for entry gate and validation."""

    def __init__(self, storage, cfg, stage_id: str, sources: Sequence[Tuple[str, float]], split: str):
        self.names = [n for n, _ in sources]
        self.weights = [w for _, w in sources]
        self.kinds = [cfg.datasets[n].kind for n in self.names]
        self.rows: List[List[Dict[str, Any]]] = []
        keep = []
        for i, n in enumerate(self.names):
            uri = prepare.prepared_uri(cfg, stage_id, n, split)
            if prepare.read_marker(storage, uri) is None:
                if split == "train":
                    raise FileNotFoundError(f"{n}/{split} not prepared for stage {stage_id}")
                continue
            self.rows.append(list(prepare.read_rows(storage, uri)))
            keep.append(i)
        self.names = [self.names[i] for i in keep]
        self.weights = [self.weights[i] for i in keep]
        self.kinds = [self.kinds[i] for i in keep]
        if not self.rows:
            raise FileNotFoundError(f"no prepared {split} prompts for stage {stage_id}")

    def draw(self, rng: random.Random, n: int) -> List[Dict[str, Any]]:
        out = []
        for _ in range(n):
            d = rng.choices(range(len(self.names)), weights=self.weights, k=1)[0]
            out.append(make_item(rng.choice(self.rows[d]), self.kinds[d]))
        return out

    def fixed(self, total: int, rank: int, world: int) -> List[Dict[str, Any]]:
        """First `total` prompts, round-robin across datasets, then this rank's slice."""
        picked, pos = [], [0] * len(self.rows)
        while len(picked) < total and any(pos[i] < len(self.rows[i]) for i in range(len(self.rows))):
            for i in range(len(self.rows)):
                if pos[i] < len(self.rows[i]) and len(picked) < total:
                    picked.append(make_item(self.rows[i][pos[i]], self.kinds[i]))
                    pos[i] += 1
        return picked[rank::world]


# --------------------------------------------------------------------------- generation and rewards
def generate_rollouts(infer: InferenceModel, tok, items: List[Dict[str, Any]], num_samples: int, gen_block,
                      stop_ids: Sequence[int], autocast_dtype, return_logprobs: bool = True) -> List[List[Rollout]]:
    """One group of `num_samples` rollouts per item."""
    gcfg = GenerationConfig.from_block(gen_block, num_samples=num_samples, stop_token_ids=list(stop_ids),
                                       return_logprobs=return_logprobs)
    res = generate(infer, tok, [it["prompt"] for it in items], gcfg, autocast_dtype=autocast_dtype)
    groups = []
    for i, it in enumerate(items):
        groups.append([Rollout(it, r.prompt_tokens, r.completion_tokens, r.logprobs or [], r.text, r.finish_reason)
                       for r in res[i * num_samples:(i + 1) * num_samples]])
    return groups


def parallel_map(fn: Callable, xs: Sequence, workers: int) -> List:
    if workers <= 1 or len(xs) <= 1:
        return [fn(x) for x in xs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, xs))


# --------------------------------------------------------------------------- log-prob passes (RL-6)
@torch.no_grad()
def response_token_logps(forward_fn: Callable[[torch.Tensor], torch.Tensor], prompts: List[List[int]],
                         responses: List[List[int]], micro: int, pad: int, device, entropy: bool = False,
                         allowed: Optional[torch.Tensor] = None):
    """Per-response fp32 token log-probs (list of 1-D CPU tensors of length len(response)) computed
    with the training forward (right padding, no attention mask; DL-3). With `entropy`, also the
    per-token policy entropy on response tokens. With `allowed` (bool over the vocabulary), also the
    log-probs renormalised over the allowed ids only: that is the distribution generation samples
    from (GN-3 step 1 masks pad and undefined ids), so it is what RL-6's mismatch must compare with;
    the unmasked ones stay the training log-probs (TR-4)."""
    lps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    masked: List[torch.Tensor] = []
    for i in range(0, len(prompts), micro):
        ps, rs = prompts[i:i + micro], responses[i:i + micro]
        idx, _ = loaders.response_batch(ps, rs, pad)
        logits = forward_fn(idx)[:, :-1]
        lp = token_logps(logits, idx[:, 1:].to(device))
        ent = token_entropy(logits) if entropy else None
        mlp = token_logps(logits.float().masked_fill(~allowed.to(device), float("-inf")), idx[:, 1:].to(device)) \
            if allowed is not None else None
        for j, (p, r) in enumerate(zip(ps, rs)):
            lps.append(lp[j, len(p) - 1:len(p) - 1 + len(r)].detach().cpu())
            if ent is not None:
                ents.append(ent[j, len(p) - 1:len(p) - 1 + len(r)].detach().cpu())
            if mlp is not None:
                masked.append(mlp[j, len(p) - 1:len(p) - 1 + len(r)].detach().cpu())
    if allowed is not None:
        return lps, ents, masked
    return (lps, ents) if entropy else lps


def aligned(vectors: List[torch.Tensor], prompts: List[List[int]], T: int) -> torch.Tensor:
    """Places per-response vectors at their target positions in a (B, T) tensor (T = L - 1)."""
    out = torch.zeros(len(vectors), T)
    for i, (v, p) in enumerate(zip(vectors, prompts)):
        out[i, len(p) - 1:len(p) - 1 + len(v)] = v
    return out


# --------------------------------------------------------------------------- PT-11
class RoutingProbe:
    """Top-k expert sets of sampled response tokens, measured before and after a rollout batch's
    updates. Hidden states entering each DeepSeekMoE layer are captured with forward hooks and the
    selection rule of MD-9 (`routing_topk`) is applied to them."""

    def __init__(self, raw_model: torch.nn.Module, sample_tokens: int, rng: random.Random):
        self.mods = moe_modules(raw_model)
        self.n = sample_tokens
        self.rng = rng
        self.samples: List[Tuple[int, int]] = []          # (sequence index, input position)

    @property
    def active(self) -> bool:
        return bool(self.mods)

    def choose(self, prompts: List[List[int]], responses: List[List[int]]) -> None:
        cand = [(i, pos) for i, (p, r) in enumerate(zip(prompts, responses)) for pos in range(len(p) - 1, len(p) - 1 + len(r))]
        self.samples = self.rng.sample(cand, min(self.n, len(cand))) if cand else []

    @torch.no_grad()
    def measure(self, forward_fn, prompts, responses, micro: int, pad: int, device) -> List[torch.Tensor]:
        """Per layer: (S, top_k) expert indices (sorted within each token) for the chosen samples."""
        if not self.active or not self.samples:
            return []
        cache: Dict[Any, torch.Tensor] = {}
        hooks = [m.register_forward_hook(lambda mod, inp, out: cache.__setitem__(mod, inp[0].detach())) for m in self.mods]
        per_layer: List[List[torch.Tensor]] = [[] for _ in self.mods]
        order = sorted(range(len(self.samples)), key=lambda k: self.samples[k][0])
        try:
            seqs = sorted({s[0] for s in self.samples})
            for c in range(0, len(seqs), micro):
                chunk = seqs[c:c + micro]
                idx, _ = loaders.response_batch([prompts[i] for i in chunk], [responses[i] for i in chunk], pad)
                forward_fn(idx)
                where = {seq: b for b, seq in enumerate(chunk)}
                for k in order:
                    seq, pos = self.samples[k]
                    if seq not in where:
                        continue
                    for li, m in enumerate(self.mods):
                        h = cache[m][where[seq], pos][None]
                        per_layer[li].append((k, routing_topk(m, h)[0].sort().values.cpu()))
        finally:
            for h in hooks:
                h.remove()
        return [torch.stack([t for _, t in sorted(rows, key=lambda x: x[0])]) for rows in per_layer]

    @staticmethod
    def changed(before: List[torch.Tensor], after: List[torch.Tensor]) -> Tuple[float, float]:
        """(number of token-layer pairs whose expert set differs, number of pairs)."""
        diff = total = 0.0
        for b, a in zip(before, after):
            diff += float((b != a).any(dim=-1).sum())
            total += b.shape[0]
        return diff, total
