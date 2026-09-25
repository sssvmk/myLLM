"""Weighted mixtures over prepared datasets (DP-9) and the derivation of optimizer-step counts
(LR-2).

Token-budget stages (1, 1b): each draw picks a dataset with probability proportional to its weight;
every dataset stream cycles (upsampling) so none runs dry.

Epoch stages (2, 3, 5a): each dataset contributes `weight * total_examples` rows per epoch
(weights are normalised to sum to 1; `total_examples` is the sum of rows available across the
listed datasets). Draws inside an epoch are interleaved in proportion to what is left of each
dataset's quota, so the per-epoch counts are exact. Counts are returned so they can be written to
the manifest.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .loaders import RowStream


def normalise(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive number")
    return [w / total for w in weights]


def epoch_counts(available: Sequence[int], weights: Sequence[float]) -> List[int]:
    """DP-9: rows per dataset per epoch = weight * total available (>= 1 for a positive weight)."""
    total = sum(available)
    return [max(1, int(round(w * total))) for w in normalise(weights)]


def steps_for_budget(token_budget: float, micro: int, accum: int, world: int, ctx: int) -> int:
    """LR-2 for token-budget stages: each block carries `ctx` target tokens."""
    return max(1, math.ceil(token_budget / (micro * accum * world * ctx)))


def steps_for_epochs(epochs: float, counts: Sequence[int], micro: int, accum: int, world: int) -> int:
    """LR-2 for epoch stages."""
    return max(1, int(epochs * sum(counts) // (micro * accum * world)))


class MixtureStream:
    def __init__(self, streams: Sequence[RowStream], weights: Sequence[float], seed: int,
                 quotas: Optional[Sequence[int]] = None):
        assert len(streams) == len(weights)
        self.streams = list(streams)
        self.weights = normalise(weights)
        self.quotas = list(quotas) if quotas is not None else None
        self.seed = seed
        self.rng = random.Random(seed)
        self.remaining = list(self.quotas) if self.quotas is not None else None
        self.draws = 0

    def _choose(self) -> int:
        if self.remaining is None:
            return self.rng.choices(range(len(self.streams)), weights=self.weights, k=1)[0]
        if sum(self.remaining) == 0:
            self.remaining = list(self.quotas)
        idx = self.rng.choices(range(len(self.streams)), weights=self.remaining, k=1)[0]
        self.remaining[idx] -= 1
        return idx

    def next(self) -> Tuple[int, Dict[str, Any]]:
        idx = self._choose()
        row = self.streams[idx].next()
        self.draws += 1
        return idx, row

    def fast_forward(self, draws: int) -> None:
        """Replays the dataset choices of the first `draws` draws (cheap: no data is read), then
        skips each underlying stream by the number of rows it supplied."""
        per = [0] * len(self.streams)
        for _ in range(draws):
            per[self._choose()] += 1
        for s, n in zip(self.streams, per):
            if n:
                s.fast_forward(n)
        self.draws = draws

    def state(self) -> Dict[str, Any]:
        return {"draws": self.draws}
