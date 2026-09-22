# Databricks notebook source
"""Driver: builds the model and data (single-source legacy path or multi-source mixture) and
runs training. Meant to be invoked from a Databricks notebook (import this module, or `%run`
it after `pip install`-ing this folder onto the cluster) or any plain Python environment for
the --data_root path, since ParquetTokenDataset just reads shards off disk/object storage and
doesn't need Spark at train time (Spark is only needed for the ETL/packing step -- see
packing.py and data_quality.py, run from a separate notebook/job).
"""
from __future__ import annotations
import os, random
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import tiktoken

from config import PRESETS, parse_args, pad_token_id_for
from model import GPTModel
from data import ParquetTokenDataset, build_mixture_dataset
from scheduler import CosineLRScheduler
from train import train_loop, load_checkpoint
from distributed import setup_distributed, wrap_model, build_optimizer_for_zero, is_main_process

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def set_seed(seed: int = 1337):
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def create_model(vocab_size: int, args) -> GPTModel:
  cfg = PRESETS[args.model]
  return GPTModel(
    vocab_size=vocab_size, d_model=cfg["d_model"], ctx=args.ctx, n_layers=cfg["n_layer"],
    d_ff=cfg["d_ff"], n_heads=cfg["n_heads"], dropout=args.dropout, arch=args.arch,
    d_latent=args.d_latent, d_rope=args.d_rope, n_routed_experts=args.n_routed_experts,
    n_shared_experts=args.n_shared_experts, moe_top_k=args.moe_top_k, rope_theta=args.rope_theta,
  )


def drive(args):
  rank, world_size, local_rank = setup_distributed()
  set_seed(args.seed)   # same seed on every rank -- model init must match before wrapping
  device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

  enc = tiktoken.get_encoding("cl100k_base")
  vocab_size = enc.n_vocab
  pad_token_id = pad_token_id_for(vocab_size)              # reserved id, one past the real vocab

  sources = [s.strip() for s in args.sources.split(',')] if args.sources else None
  if args.data_root:
    train_dataset = build_mixture_dataset(args.data_root, 'train', args.ctx, pad_token_id, sources=sources)
    test_dataset = build_mixture_dataset(args.data_root, 'test', args.ctx, pad_token_id, sources=sources, shuffle=False)
  elif args.train and args.test:
    train_dataset = ParquetTokenDataset(args.train, args.ctx, pad_token_id)
    test_dataset = ParquetTokenDataset(args.test, args.ctx, pad_token_id)
  else:
    raise ValueError("pass either --data_root (multi-source mixture) or both --train/--test (legacy single-source)")

  micro = max(1, args.batch_tokens // (args.ctx * args.accum))
  if is_main_process():
    print(f"Derived micro-batch size: {micro} (accum={args.accum}, ctx={args.ctx}, world_size={world_size})")
  train_dataloader = DataLoader(train_dataset, batch_size=micro, shuffle=False, num_workers=args.num_workers)
  val_dataloader = DataLoader(test_dataset, batch_size=micro, shuffle=False, num_workers=args.num_workers)

  model = create_model(vocab_size + 1, args)                # +1 for the reserved pad id
  model = wrap_model(model, args.zero_stage, device)        # ZeRO/FSDP wrapping (no-op for single-process runs)
  if args.compile:
    model = torch.compile(model)

  optimizer = build_optimizer_for_zero(model, args.zero_stage, args.lr, args.weight_decay)
  scheduler = CosineLRScheduler(optimizer, warmup=args.warmup, total=args.max_steps,
                                 base_lr=args.lr, min_lr=args.min_lr)
  scaler = None if args.use_bfloat else amp.GradScaler(enabled=torch.cuda.is_available())

  step, best_loss = 0, float("inf")
  if args.resume:
    ckpt_path = os.path.join(args.out, "latest", "latest_check.pt") if args.resume == "latest" else args.resume
    if os.path.exists(ckpt_path):
      step, best_loss = load_checkpoint(model, optimizer, ckpt_path, device)
    else:
      print(f"--resume given but {ckpt_path} does not exist; starting fresh")

  os.makedirs(args.out, exist_ok=True)
  train_loop(model, train_dataloader, val_dataloader, optimizer, scaler, scheduler, micro,
             args, device, pad_token_id, step=step, best_loss=best_loss)


# COMMAND ----------

if __name__ == '__main__':
  args, unknown = parse_args()
  drive(args)
