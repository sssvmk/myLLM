"""
Central config: model size presets, the multi-source data-mixture table, and the CLI.
Import from here rather than duplicating PRESETS/DATA_SOURCES/pad-id logic elsewhere.
"""
from __future__ import annotations
import argparse
import torch

# --- Model size presets ----------------------------------------------------
PRESETS = {
  "125M": dict(n_layer=12, d_model=768,  n_heads=12, d_ff=3072),
  "355M": dict(n_layer=12, d_model=1024, n_heads=16, d_ff=4096),
  "1.3B": dict(n_layer=24, d_model=2048, n_heads=16, d_ff=8192),
  "6.7B": dict(n_layer=32, d_model=4096, n_heads=32, d_ff=16384),
}

# --- Data mixture ------------------------------------------------------------
# Each source lives under {data_root}/{subdir}/{split}/shard_id=*/part-*.parquet (see
# packing.pack_source_to_shards). `weight` is the target share of *sampled blocks*
# (renormalized across whichever sources are active for a run), not raw token count.
# `cycle=True` sources are reshuffled and repeated for the life of the epoch so a small
# high-quality source (wiki/books/math) can be upsampled relative to its natural size
# instead of running out early and dropping from the mixture.
# These weights are a starting point -- tune them via small-scale ablations (see README).
DATA_SOURCES = [
  dict(name="web",       subdir="openwebtext", weight=0.65, cycle=False),
  dict(name="code",      subdir="code",         weight=0.15, cycle=False),
  dict(name="wikipedia", subdir="wikipedia",    weight=0.10, cycle=True),
  dict(name="books",     subdir="books",        weight=0.07, cycle=True),
  dict(name="math",      subdir="math",         weight=0.03, cycle=True),
]


def pad_token_id_for(vocab_size: int) -> int:
  """Reserved pad id, one past the real tokenizer vocab so it can never collide with a real
  token (the original version padded with id 0, which is a real cl100k_base token). The
  model's embedding table is built with vocab_size + 1 rows (see main.create_model) and this
  id is passed as F.cross_entropy's ignore_index so padded positions never contribute to loss.
  """
  return vocab_size


def parse_args():
  p = argparse.ArgumentParser()
  # paths
  p.add_argument('--out', type=str,
                  default='/dbfs/FileStore/shared_uploads/muni-kumar.x.schandra@gsk.com/models/toy_gpt35/')
  p.add_argument('--train', type=str, default='', help='Legacy single-source train path (used only if --data_root is unset)')
  p.add_argument('--test', type=str, default='', help='Legacy single-source test path (used only if --data_root is unset)')
  p.add_argument('--data_root', type=str, default='', help='Base "transformed" folder with one subdir per DATA_SOURCES entry')
  p.add_argument('--sources', type=str, default='', help='Comma-separated subset of DATA_SOURCES names (default: all)')

  # architecture
  p.add_argument('--model', choices=list(PRESETS.keys()), default="355M")
  p.add_argument('--arch', choices=["dense", "deepseek"], default="dense",
                  help="'dense': RoPE + standard MHA + GELU MLP. 'deepseek': Multi-head Latent "
                       "Attention + DeepSeekMoE (simplified reproduction -- see model.py/moe.py).")
  p.add_argument('--ctx', type=int, default=1024)
  p.add_argument('--rope_theta', type=float, default=10000.0)
  # deepseek-arch-only knobs (ignored for --arch dense)
  p.add_argument('--d_latent', type=int, default=0, help='0 = derive as d_model // 8')
  p.add_argument('--d_rope', type=int, default=32)
  p.add_argument('--n_routed_experts', type=int, default=8)
  p.add_argument('--n_shared_experts', type=int, default=1)
  p.add_argument('--moe_top_k', type=int, default=2)
  p.add_argument('--use_doc_mask', action='store_true', default=True,
                  help='Build a document-boundary attention mask from packed doc_start flags '
                       '(when present) so attention cannot cross concatenated-document seams')

  # optimization
  p.add_argument('--batch-tokens', type=int, default=262144)
  p.add_argument('--accum', type=int, default=8)
  p.add_argument('--lr', type=float, default=3e-4)
  p.add_argument('--min_lr', type=float, default=1.5e-7)
  p.add_argument('--warmup', type=int, default=2000)
  p.add_argument('--max_steps', type=int, default=100000)
  p.add_argument('--weight_decay', type=float, default=0.1)
  p.add_argument('--grad_clip', type=float, default=1.0)
  p.add_argument('--dropout', type=float, default=0.1)
  p.add_argument('--seed', type=int, default=1337)
  p.add_argument('--use_bfloat', type=bool,
                  default=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

  # runtime
  p.add_argument('--compile', action='store_true')
  p.add_argument('--zero_stage', type=int, choices=[0, 1, 2, 3], default=0,
                  help='ZeRO stage (see distributed.py): 0=plain DDP, 1=DDP+optimizer-state '
                       'sharding, 2=FSDP SHARD_GRAD_OP, 3=FSDP FULL_SHARD. Only takes effect '
                       'under torchrun (WORLD_SIZE>1); ignored for single-process runs.')
  p.add_argument('--num_workers', type=int, default=0)
  p.add_argument('--ckpt_every', type=int, default=1000)
  p.add_argument('--eval_every', type=int, default=1000)
  p.add_argument('--log_interval', type=int, default=50)
  p.add_argument('--resume', type=str, default='', help='"latest" to resume from --out/latest/latest_check.pt, or an explicit checkpoint path')

  return p.parse_known_args()
