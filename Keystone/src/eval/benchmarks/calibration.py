"""Expected calibration error over MMLU choice log-likelihoods (B-8)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def choice_confidence(choice_logliks: Sequence[float]) -> tuple:
    x = np.asarray(choice_logliks, dtype=np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    return float(p.max()), int(p.argmax())


def ece(confidences: List[float], correct: List[bool], bins: int) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    idx = np.minimum((conf * bins).astype(int), bins - 1)   # equal-width bins; conf=1.0 -> last bin
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.sum() / conf.size * abs(corr[sel].mean() - conf[sel].mean())
    return float(total)
