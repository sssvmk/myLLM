"""Learning-rate schedules defined by fractions of total optimizer steps (LR-1).

Convention: `lr_at(s)` is the rate used for optimizer step s (0-based). Warmup is linear from 0
at s=0 to `lr` at s=W (W = round(warmup_fraction * total)); after the final step the rate is
`lr_min` (wsd/linear/cosine) or `lr` (constant)."""
from __future__ import annotations

import math


def _warm(s, W, lr):
    return lr * s / W


def lr_at(step: int, total: int, schedule, lr: float, lr_min: float) -> float:
    if total <= 0:
        raise ValueError("total optimizer steps must be positive")
    W = int(round(schedule.warmup_fraction * total))
    if step < W:
        return _warm(step, W, lr)
    if schedule.type == "constant":
        return lr
    if schedule.type == "wsd":
        S = int(round(schedule.stable_fraction * total))
        D = total - W - S
        if step < W + S:
            return lr
        if D <= 0:
            return lr_min
        t = min((step - W - S) / D, 1.0)
        if schedule.decay_shape == "linear":
            return lr_min + (lr - lr_min) * (1 - t)
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
    D = total - W
    t = min((step - W) / D, 1.0) if D > 0 else 1.0
    if schedule.type == "linear":
        return lr_min + (lr - lr_min) * (1 - t)
    if schedule.type == "cosine":
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
    raise ValueError(f"unknown schedule type {schedule.type!r}")


class Scheduler:
    def __init__(self, optimizer, total_steps: int, schedule, lr: float, lr_min: float, step: int = 0):
        self.optimizer, self.total, self.schedule, self.lr, self.lr_min = optimizer, total_steps, schedule, lr, lr_min
        self.step_num = step
        self._apply()

    def _apply(self):
        cur = lr_at(self.step_num, self.total, self.schedule, self.lr, self.lr_min)
        for g in self.optimizer.param_groups:
            g["lr"] = cur
        return cur

    def step(self):
        self.step_num += 1
        return self._apply()

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, d):
        self.step_num = d["step_num"]
        self._apply()
