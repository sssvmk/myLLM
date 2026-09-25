"""HumanEval pass@k estimator (B-4). Numerically stable product form of
1 - C(n-c, k) / C(n, k) (as in openai/human-eval evaluation.estimate_pass_at_k)."""
from __future__ import annotations

from typing import List

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(ns: List[int], cs: List[int], k: int) -> float:
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(ns, cs)])) if ns else 0.0
