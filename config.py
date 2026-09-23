"""
Loads editable settings from settings.yaml (path/model/data-source config -- edit that file,
not this one) and defines the CLI. See settings.yaml for PRESETS, TOKENIZER_SOURCES,
RAW_SOURCE_PATHS, DATA_SOURCES, EVAL_BENCHMARK_SOURCES, QUALITY_CLASSIFIER.
"""
from __future__ import annotations
import argparse
import os
import yaml
import torch

_SETTINGS_PATH = os.environ.get(
  "FOUNDATION_LLM_SETTINGS",
  os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.yaml"),
)

with open(_SETTINGS_PATH) as _f:
  _SETTINGS = yaml.safe_load(_f)

PRESETS = _SETTINGS["presets"]
TOKENIZER_SOURCES = _SETTINGS["tokenizer"]
RAW_SOURCE_PATHS = _SETTINGS["raw_source_paths"]
DATA_SOURCES = _SETTINGS["data_sources"]
EVAL_BENCHMARK_SOURCES = _SETTINGS["eval_benchmark_sources"]
QUALITY_CLASSIFIER = _SETTINGS["quality_classifier"]
_DEFAULT_OUT = _SETTINGS["output_dir"]


def apply_weight_overrides(data_sources: list, overrides_str: str) -> list:
  """Parses '--mixture_weights web=0.5,code=0.3,wikipedia=0.1,books=0.07,math=0.03' and
  returns a NEW list of DATA_SOURCES-shaped dicts with those weights substituted in (leaves
  `cycle`/`subdir` untouched, and leaves unmentioned sources at their settings.yaml default) --
  so a weight ablation is a CLI flag, not a file edit. Unknown names raise immediately rather
  than being silently ignored."""
  if not overrides_str:
    return data_sources
  overrides = {}
  for pair in overrides_str.split(','):
    pair = pair.strip()
    if not pair:
      continue
    name, _, val = pair.partition('=')
    if not _:
      raise ValueError(f"--mixture_weights entry {pair!r} is missing '=weight'")
    overrides[name.strip()] = float(val)
  known = {c["name"] for c in data_sources}
  unknown = set(overrides) - known
  if unknown:
    raise ValueError(f"--mixture_weights names {sorted(unknown)} are not in DATA_SOURCES {sorted(known)}")
  return [dict(c, weight=overrides.get(c["name"], c["weight"])) for c in data_sources]


def pad_token_id_for(vocab_size: int) -> int:
  """Reserved pad id, one past the real tokenizer vocab so it can never collide with a real
  token. The model's embedding table is built with vocab_size + 1 rows (see
  main.create_model) and this id is passed as F.cross_entropy's ignore_index so padded
  positions never contribute to loss."""
  return vocab_size


def parse_args():
  p = argparse.ArgumentParser()
  # paths
  p.add_argument('--out', type=str, default=_DEFAULT_OUT,
                  help='Checkpoint/output dir. Default comes from settings.yaml: output_dir')
  p.add_argument('--train', type=str, default='', help='Legacy single-source train path (used only if --data_root is unset)')
  p.add_argument('--test', type=str, default='', help='Legacy single-source test path (used only if --data_root is unset)')
  p.add_argument('--data_root', type=str, default='', help='Base "transformed" folder with one subdir per data_sources entry')
  p.add_argument('--sources', type=str, default='', help='Comma-separated subset of data_sources names (default: all)')
  p.add_argument('--mixture_weights', type=str, default='',
                  help='Override data_sources weights for this run, e.g. "web=0.5,code=0.3,wikipedia=0.1,books=0.07,math=0.03" '
                       '-- for ablations without editing settings.yaml. See apply_weight_overrides.')
  p.add_argument('--tokenizer_path', type=str, default='',
                  help='Local path to a cl100k_base.tiktoken ranks file, for fully offline tokenizer '
                       'loading (see tokenizer.py). Defaults to settings.yaml: tokenizer.cl100k_base.local_path '
                       'if set there; otherwise falls back to tiktoken.get_encoding (network on first use).')

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

  # observability
  p.add_argument('--tensorboard', action='store_true', help='Also log metrics to a TensorBoard event file at {out}/tensorboard/')
  p.add_argument('--eval_benchmarks', type=str, default='',
                  help='Comma-separated subset of eval_benchmark_sources to run periodically during '
                       'training via benchmark_eval.py (currently: mmlu; gsm8k/humaneval need generation '
                       'infra not yet in this codebase -- see benchmark_eval.py). Requires --tokenizer_path '
                       'or a reachable tiktoken and each benchmark\'s local_path filled in or HF access.')

  return p.parse_known_args()
