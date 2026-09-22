"""Linear-warmup, cosine-decay LR schedule."""
from __future__ import annotations
import math


class CosineLRScheduler:
  def __init__(self, optimizer, warmup: int, total: int, base_lr: float, min_lr: float):
    self.optimizer = optimizer
    self.warmup = warmup
    self.total = total
    self.base_lr = base_lr
    self.min_lr = min_lr
    self.step_num = 0

  def step(self):
    self.step_num += 1
    if self.step_num < self.warmup:
      lr = self.base_lr * self.step_num / max(1, self.warmup)
    else:
      t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
      t = min(max(t, 0.0), 1.0)
      lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    for pg in self.optimizer.param_groups:
      pg["lr"] = lr
    return lr
