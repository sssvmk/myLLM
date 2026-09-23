"""Reads {out}/metrics.jsonl (written by train.MetricsLogger during training) and renders
PNG charts to {out}/plots/. Safe to run at any point during a live run -- it just reads
whatever's in the file so far, it doesn't need training to have finished.

Usage:
    python plot_metrics.py /path/to/args.out
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # headless -- this runs on a training box/notebook, not someone's desktop
import matplotlib.pyplot as plt


def load_events(metrics_path: str):
  events = defaultdict(list)
  with open(metrics_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      events[rec["event"]].append(rec)
  return events


def plot_loss_and_lr(events, out_dir: str):
  train_steps = [r["step"] for r in events.get("train_step", [])]
  train_loss = [r["loss"] for r in events.get("train_step", [])]
  lrs = [r["lr"] for r in events.get("train_step", [])]
  eval_steps = [r["step"] for r in events.get("eval", [])]
  eval_loss = [r["val_loss"] for r in events.get("eval", [])]

  if train_steps:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(train_steps, train_loss, label="train loss", color="tab:blue", linewidth=1)
    if eval_steps:
      ax1.plot(eval_steps, eval_loss, label="val loss", color="tab:red", marker="o", linewidth=1.5)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training / validation loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=130)
    plt.close(fig)

  if train_steps:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(train_steps, lrs, color="tab:green", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (as actually applied)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr_schedule.png"), dpi=130)
    plt.close(fig)


def plot_throughput(events, out_dir: str):
  rows = events.get("train_step", [])
  if not rows:
    return
  steps = [r["step"] for r in rows]
  tok_s = [r["tok_per_sec"] for r in rows]
  fig, ax = plt.subplots(figsize=(9, 3.5))
  ax.plot(steps, tok_s, color="tab:purple", linewidth=1)
  ax.set_xlabel("step")
  ax.set_ylabel("tokens / sec")
  ax.set_title("Training throughput")
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "throughput.png"), dpi=130)
  plt.close(fig)


def plot_perplexity(events, out_dir: str):
  rows = events.get("eval", [])
  if not rows:
    return
  steps = [r["step"] for r in rows]
  ppl = [r["perplexity"] for r in rows]
  fig, ax = plt.subplots(figsize=(9, 3.5))
  ax.plot(steps, ppl, color="tab:orange", marker="o", linewidth=1.5)
  ax.set_xlabel("step")
  ax.set_ylabel("perplexity")
  ax.set_title("Validation perplexity")
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "perplexity.png"), dpi=130)
  plt.close(fig)


def plot_moe_usage(events, out_dir: str):
  rows = events.get("moe_usage", [])
  if not rows:
    return   # dense arch, or deepseek arch that hasn't hit a ckpt_every boundary yet
  latest = rows[-1]
  usage_per_layer = latest["usage_per_layer"]
  n_layers = len(usage_per_layer)
  fig, axes = plt.subplots(1, n_layers, figsize=(3.5 * n_layers, 3.5), squeeze=False)
  for i, layer_usage in enumerate(usage_per_layer):
    ax = axes[0][i]
    ax.bar(range(len(layer_usage)), layer_usage, color="tab:cyan")
    ax.set_title(f"layer {i}")
    ax.set_xlabel("expert")
    if i == 0:
      ax.set_ylabel("tokens routed (last batch)")
  fig.suptitle(f"MoE expert usage at step {latest['step']} (balanced routing = flat bars)")
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "moe_usage.png"), dpi=130)
  plt.close(fig)


def summarize(events):
  train_rows = events.get("train_step", [])
  eval_rows = events.get("eval", [])
  print(f"train_step events: {len(train_rows)}   eval events: {len(eval_rows)}")
  if train_rows:
    print(f"latest train loss: {train_rows[-1]['loss']:.4f} @ step {train_rows[-1]['step']}")
  if eval_rows:
    best = min(eval_rows, key=lambda r: r["val_loss"])
    print(f"latest val loss:    {eval_rows[-1]['val_loss']:.4f} @ step {eval_rows[-1]['step']}")
    print(f"best val loss:      {best['val_loss']:.4f} @ step {best['step']} (perplexity {best['perplexity']:.2f})")


def main(out_dir: str):
  metrics_path = os.path.join(out_dir, "metrics.jsonl")
  if not os.path.exists(metrics_path):
    print(f"no metrics.jsonl found at {metrics_path} -- has training started and logged anything yet?")
    return

  events = load_events(metrics_path)
  summarize(events)

  plots_dir = os.path.join(out_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)
  plot_loss_and_lr(events, plots_dir)
  plot_throughput(events, plots_dir)
  plot_perplexity(events, plots_dir)
  plot_moe_usage(events, plots_dir)
  print(f"charts written to {plots_dir}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: python plot_metrics.py /path/to/args.out")
    sys.exit(1)
  main(sys.argv[1])
