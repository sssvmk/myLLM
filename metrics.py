"""Structured training-metrics logging: JSONL always, TensorBoard optionally.

train.py writes one event per call to MetricsLogger.log(event, **fields) for train_step, eval,
and moe_usage events. Every event is appended to {out}/metrics.jsonl (see plot_metrics.py for
matplotlib charts from that file). When tensorboard=True, the same events are ALSO written as
TensorBoard scalars/histograms to {out}/tensorboard/, for anyone who prefers `tensorboard
--logdir` over the JSONL+matplotlib path -- both are populated from the same log() calls, so
they never drift out of sync with each other.
"""
from __future__ import annotations
import json, os, time
from typing import Any, Dict, Optional


class MetricsLogger:
  def __init__(self, out_dir: str, filename: str = "metrics.jsonl", tensorboard: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    self.path = os.path.join(out_dir, filename)
    self._fh = open(self.path, "a")
    self._tb = None
    if tensorboard:
      from torch.utils.tensorboard import SummaryWriter  # imported lazily: tensorboard is an
                                                            # optional dependency, only needed
                                                            # when this flag is actually set
      self._tb = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))

  def log(self, event: str, **fields: Any):
    record: Dict[str, Any] = {"event": event, "wall_time": time.time(), **fields}
    self._fh.write(json.dumps(record) + "\n")
    self._fh.flush()   # flushed per-write: metrics.jsonl is meant to be readable mid-run, not just after close()
    if self._tb is not None:
      self._log_tensorboard(event, fields)

  def _log_tensorboard(self, event: str, fields: Dict[str, Any]):
    step = fields.get("step")
    if event == "train_step":
      self._tb.add_scalar("train/loss", fields["loss"], step)
      self._tb.add_scalar("train/lr", fields["lr"], step)
      self._tb.add_scalar("train/tok_per_sec", fields["tok_per_sec"], step)
    elif event == "eval":
      self._tb.add_scalar("eval/val_loss", fields["val_loss"], step)
      self._tb.add_scalar("eval/perplexity", fields["perplexity"], step)
    elif event == "moe_usage":
      for layer_idx, layer_usage in enumerate(fields["usage_per_layer"]):
        self._tb.add_scalars(f"moe/layer{layer_idx}_usage",
                              {f"expert_{j}": v for j, v in enumerate(layer_usage)}, step)
    self._tb.flush()

  def close(self):
    self._fh.close()
    if self._tb is not None:
      self._tb.close()


class NullMetricsLogger:
  """No-op logger, used when metrics logging isn't wanted (e.g. tests/test_model_smoke.py,
  which calls train.py's building blocks directly without a MetricsLogger)."""
  def log(self, event: str, **fields: Any):
    pass

  def close(self):
    pass
