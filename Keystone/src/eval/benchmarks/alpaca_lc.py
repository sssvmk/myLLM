"""Length-controlled win rate (EV-5): fit logit P(win) = a + b * tanh(d / std(d)) by Newton's
method on per-instruction outcomes in {0, 0.5, 1}; LC win rate = sigmoid(a)."""
from __future__ import annotations

from typing import List

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lc_win_rate(outcomes: List[float], len_model: List[int], len_ref: List[int], max_iter: int) -> dict:
    y = np.asarray(outcomes, dtype=np.float64)
    d = np.asarray(len_model, dtype=np.float64) - np.asarray(len_ref, dtype=np.float64)
    sd = d.std()
    x = np.tanh(d / sd) if sd > 0 else np.zeros_like(d)
    X = np.stack([np.ones_like(x), x], axis=1)
    w = np.zeros(2)
    for _ in range(max_iter):
        p = _sigmoid(X @ w)
        grad = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, grad)
        w = w + step
        if np.abs(step).max() < 1e-10:
            break
    return {"alpaca_win_rate": float(y.mean()) if y.size else 0.0,
            "alpaca_lc_win_rate": float(_sigmoid(w[0])), "a": float(w[0]), "b": float(w[1])}
