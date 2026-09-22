"""Structured training-metrics logging.

train.py writes one JSON line per event (train step, eval, MoE expert usage) to
{args.out}/metrics.jsonl via MetricsLogger. plot_metrics.py reads that file back and renders
charts with matplotlib -- run it any time during or after training (it just tails whatever
the file has so far).
"""
from __future__ import annotations
import json, os, time
from typing import Any, Dict, Optional


class MetricsLogger:
  def __init__(self, out_dir: str, filename: str = "metrics.jsonl"):
    os.makedirs(out_dir, exist_ok=True)
    self.path = os.path.join(out_dir, filename)
    self._fh = open(self.path, "a")

  def log(self, event: str, **fields: Any):
    record: Dict[str, Any] = {"event": event, "wall_time": time.time(), **fields}
    self._fh.write(json.dumps(record) + "\n")
    self._fh.flush()   # flushed per-write: metrics.jsonl is meant to be readable mid-run, not just after close()

  def close(self):
    self._fh.close()


class NullMetricsLogger:
  """No-op logger, used when metrics logging isn't wanted (e.g. tests/test_model_smoke.py,
  which calls train.py's building blocks directly without a MetricsLogger)."""
  def log(self, event: str, **fields: Any):
    pass

  def close(self):
    pass
