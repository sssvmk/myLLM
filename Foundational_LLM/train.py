"""Training loop: gradient accumulation + mixed precision, cosine LR schedule, checkpointing,
and MoE aux-loss/bias-balance wiring.

Fixes applied relative to the original single-file version:
  - step and best_loss now actually propagate across epochs (train_loop reassigns both from
    train_one_epoch's return value instead of discarding it)
  - evaluate() runs under torch.no_grad()
  - dropout is applied (moved into model.Block; was constructed but never called before)
  - resume path: checkpoint save/load now go through distributed.py's collective-safe
    full_model_state_dict/full_optimizer_state_dict helpers (correct under ZeRO/FSDP, and a
    plain model.state_dict()/optimizer.load_state_dict() in the single-process case) -- this
    also fixes the original load_state_device (nonexistent method) and optimizer.to(device)
    (optimizers have no .to()) bugs, args.latest is gone (was referenced but never defined),
    and best_loss is restored from the checkpoint instead of being hardcoded back to inf
  - save_check_point creates the `latest/` subdir before copying into it, and no longer
    silently swallows the copy failure with a bare except
  - AdamW uses separate param groups so 1-D params (norms, the tied embedding) are excluded
    from weight decay
  - loss uses ignore_index=pad_token_id so padded block tails don't contribute to the loss
  - fused-AdamW capability check is guarded for CPU-only runs
  - the `if args.resume is None` dead branch (default is '', never None) is fixed to `if args.resume`
  - training now logs structured metrics (loss, lr, tok/s, eval loss, perplexity, MoE expert
    usage) to {args.out}/metrics.jsonl via metrics.MetricsLogger, instead of stdout-only
    printing -- see plot_metrics.py for charting them
"""
from __future__ import annotations
import os, time, shutil, logging, math
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import build_doc_attention_mask
from moe import DeepSeekMoE
from metrics import MetricsLogger, NullMetricsLogger
from distributed import (full_model_state_dict, full_optimizer_state_dict,
                          load_full_model_state_dict, load_full_optimizer_state_dict,
                          is_main_process, is_distributed)
import torch.distributed as dist

log = logging.getLogger(__name__)


def build_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
  decay, no_decay = [], []
  for _, p in model.named_parameters():
    if not p.requires_grad:
      continue
    (no_decay if p.dim() < 2 else decay).append(p)   # norms / 1-D params -> excluded from decay
  fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8   # fix: guarded for CPU
  return optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=lr, betas=betas, eps=eps, fused=fused,
  )


def _unpack_batch(batch, device):
  block, doc_start = batch
  return block.to(device, non_blocking=True), doc_start.to(device, non_blocking=True)


def _forward_loss(model, block, doc_start, pad_token_id, args):
  xb, yb = block[:, :-1], block[:, 1:]
  attn_mask = build_doc_attention_mask(doc_start[:, :-1]) if args.use_doc_mask else None
  with torch.amp.autocast("cuda", dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                           enabled=torch.cuda.is_available()):
    logits, aux_loss = model(xb, attn_mask=attn_mask)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1),
                            ignore_index=pad_token_id)      # fix: padded positions no longer count
    if aux_loss is not None:
      loss = loss + aux_loss
  return loss


def _update_moe_bias(model):
  blocks = getattr(model, "blocks", getattr(getattr(model, "_orig_mod", None), "blocks", None))
  if blocks is None:
    return
  for blk in blocks:
    if isinstance(blk.mlp, DeepSeekMoE):
      blk.mlp.update_bias()


def _moe_usage_snapshot(model):
  """Per-layer routed-expert usage counts from the most recent forward pass, or None for a
  dense model / a deepseek model that hasn't run forward yet. Logged periodically (not every
  step -- it's diagnostic, not needed at training granularity) so metrics.jsonl can chart
  whether routing stays balanced across training."""
  blocks = getattr(model, "blocks", getattr(getattr(model, "_orig_mod", None), "blocks", None))
  if blocks is None:
    return None
  usage = []
  for blk in blocks:
    if isinstance(blk.mlp, DeepSeekMoE) and blk.mlp._last_usage is not None:
      usage.append(blk.mlp._last_usage.tolist())
  return usage or None


def train_one_epoch(model, loader, optimizer, scaler, scheduler, args, micro, device,
                     best_loss, step, pad_token_id, metrics: MetricsLogger = None):
  metrics = metrics or NullMetricsLogger()
  model.train()
  running_loss = 0.0
  last_log = time.time()

  for batch in loader:
    block, doc_start = _unpack_batch(batch, device)
    loss = _forward_loss(model, block, doc_start, pad_token_id, args)
    loss = loss / args.accum
    (scaler.scale(loss) if scaler is not None else loss).backward()

    if (step + 1) % args.accum == 0:
      if args.grad_clip > 0:
        if scaler is not None:
          scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

      if scaler is None:
        optimizer.step()
      else:
        scaler.step(optimizer)
        scaler.update()

      _update_moe_bias(model)
      optimizer.zero_grad(set_to_none=True)
      scheduler.step()

    running_loss += loss.item() * args.accum
    step += 1

    if step % args.log_interval == 0:
      dt = time.time() - last_log
      tok = args.ctx * micro * args.log_interval
      avg_loss = running_loss / args.log_interval
      tok_per_sec = tok / max(dt, 1e-9)
      lr = optimizer.param_groups[0]['lr']
      print(f"step {step:06d} | loss {avg_loss:.4f} | lr {lr:.2e} | tok/s {tok_per_sec:,.0f}")
      metrics.log("train_step", step=step, loss=avg_loss, lr=lr, tok_per_sec=tok_per_sec)
      running_loss = 0.0
      last_log = time.time()

    if step % args.ckpt_every == 0:
      save_check_point(model, optimizer, step, best_loss, args)
      usage = _moe_usage_snapshot(model)
      if usage is not None:
        metrics.log("moe_usage", step=step, usage_per_layer=usage)

    if step > args.max_steps:
      return step, best_loss, True

  return step, best_loss, False


@torch.no_grad()                                            # fix: eval no longer builds a grad graph
def evaluate(model, loader, device, args, pad_token_id, metrics: MetricsLogger = None, step: int = None):
  metrics = metrics or NullMetricsLogger()
  model.eval()
  running_loss = 0.0
  num_batches = 0
  for batch in loader:
    block, doc_start = _unpack_batch(batch, device)
    loss = _forward_loss(model, block, doc_start, pad_token_id, args)
    running_loss += loss.item()
    num_batches += 1
  model.train()
  val_loss = running_loss / max(num_batches, 1)
  perplexity = math.exp(min(val_loss, 20))   # cap the exponent so a runaway early-training loss can't overflow
  if step is not None:
    metrics.log("eval", step=step, val_loss=val_loss, perplexity=perplexity)
  return val_loss


def save_check_point(model, optimizer, step, best_loss, args):
  # fix: under FSDP/ZeRO these two are COLLECTIVE -- every rank must call them even though
  # only rank 0's return value is populated, or the other ranks deadlock waiting on the
  # all-gather. Only the disk write itself is rank-0-only.
  full_msd = full_model_state_dict(model)
  full_osd = full_optimizer_state_dict(model, optimizer)
  if not is_main_process():
    return

  os.makedirs(args.out, exist_ok=True)
  latest_dir = os.path.join(args.out, "latest")
  os.makedirs(latest_dir, exist_ok=True)                     # fix: this dir was never created before

  save_path = os.path.join(args.out, f"{step:06d}.pt")
  latest_path = os.path.join(latest_dir, "latest_check.pt")
  torch.save({
    "model": full_msd,
    "optimizer": full_osd,
    "step": step,
    "best_loss": best_loss,
    "args": vars(args),
  }, save_path)

  try:
    shutil.copyfile(save_path, latest_path)
  except OSError as e:
    log.warning(f"failed to update latest checkpoint copy at {latest_path}: {e}")  # fix: no longer silently swallowed


def load_checkpoint(model, optimizer, path, device):
  # fix: broadcast from rank 0 rather than every rank independently re-reading the file --
  # under FSDP the subsequent load calls are collective and need the SAME object on every rank.
  if is_main_process():
    obj = torch.load(path, map_location='cpu', weights_only=False)
    payload = [obj['model'], obj['optimizer'], obj['step'], obj['best_loss']]
  else:
    payload = [None, None, None, None]
  if is_distributed():
    dist.broadcast_object_list(payload, src=0)
  full_msd, full_osd, step, best_loss = payload

  load_full_model_state_dict(model, full_msd)
  load_full_optimizer_state_dict(model, optimizer, full_osd)
  return step, best_loss


def train_loop(model, train_loader, val_loader, optimizer, scaler, scheduler, micro, args,
               device, pad_token_id, step=0, best_loss=float("inf"), metrics: MetricsLogger = None):
  own_metrics = metrics is None
  if metrics is None:
    metrics = MetricsLogger(args.out, tensorboard=getattr(args, "tensorboard", False)) if is_main_process() else NullMetricsLogger()
    # only rank 0 writes metrics.jsonl -- every rank owning a MetricsLogger would mean
    # multiple processes appending to the same file concurrently

  if args.resume:                                            # fix: was `if args.resume is None` (never true)
    print(f"Resumed from {os.path.join(args.out, args.resume)} at step {step}")
  else:
    print(f"Starting fresh training run at step {step}")

  try:
    for epoch in range(1, 10 ** 9):
      step, best_loss, done = train_one_epoch(                 # fix: step/best_loss now reassigned each epoch
        model, train_loader, optimizer, scaler, scheduler, args, micro, device,
        best_loss, step, pad_token_id, metrics=metrics,
      )

      if step % args.eval_every == 0 or done:    # fix: also eval on the final step, not just exact multiples --
                                                  # otherwise hitting max_steps between eval_every boundaries
                                                  # means eval (and best_loss) never runs before exit
        val_loss = evaluate(model, val_loader, device, args, pad_token_id, metrics=metrics, step=step)
        print(f"epoch {epoch} step {step} val_loss {val_loss:.4f}")
        if val_loss < best_loss:                                 # fix: best_loss now actually gets updated
          best_loss = val_loss
          save_check_point(model, optimizer, step, best_loss, args)

      if done:
        break
  finally:
    if own_metrics:
      metrics.close()

  return step, best_loss
