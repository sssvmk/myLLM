"""Metrics: foundation_llm's MetricsLogger for JSONL (always tensorboard=False, MT-L1a) plus a
generic TensorBoard writer for every numeric field of every event."""
from __future__ import annotations

import numbers
import os
from typing import Any

from ..foundation import bridge


class PFMetricsLogger:
    def __init__(self, out_dir: str, tensorboard: bool):
        bridge.require()
        self.out_dir = out_dir
        self._jsonl = bridge.fl_metrics.MetricsLogger(out_dir, tensorboard=False)
        self._tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))

    def log(self, event: str, **fields: Any):
        self._jsonl.log(event, **fields)
        if self._tb is None:
            return
        step = fields.get("step")
        if event == "moe_usage":
            for li, layer in enumerate(fields.get("usage_per_layer") or []):
                for ej, v in enumerate(layer):
                    self._tb.add_scalar(f"moe/layer{li}/expert_{ej}", v, step)
        else:
            for k, v in fields.items():
                if k != "step" and isinstance(v, numbers.Number) and not isinstance(v, bool):
                    self._tb.add_scalar(f"{event}/{k}", v, step)
        self._tb.flush()

    def close(self):
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()
