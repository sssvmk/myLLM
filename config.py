"""
Central config: model presets, data-mixture table, dataset/tokenizer/eval-set/quality-classifier
SOURCE LOCATIONS (all real, verified download URLs -- see prd.md for the documented version of
this same table), and the CLI.

Every "where do we get X" question in this project resolves to a dict in this file. Nothing
about *where data comes from* should live only in a docstring or a code comment -- if it's a
download source, it's here, and prd.md documents it in prose form.
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

# --- Tokenizer source --------------------------------------------------------
# tiktoken's cl100k_base normally fetches this file over the network at runtime (see
# tokenizer.py's docstring for why that can fail in a restricted environment, and how this
# entry's `local_path` is used to avoid it entirely).
TOKENIZER_SOURCES = {
  "cl100k_base": dict(
    official_url="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    # Mirrors (verify sha256 below regardless of source -- tokenizer.py does this automatically):
    mirrors=[
      "https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/cl100k_base.tiktoken",
      "https://huggingface.co/spaces/xu-song/tokenizer-arena/blob/main/vocab/gpt_35_turbo/cl100k_base.tiktoken",
    ],
    sha256="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    # Fill this in once the file has been placed somewhere this pipeline can read it (a shared
    # storage path, DBFS location, etc). --tokenizer_path on the CLI overrides this if set.
    local_path="",
  ),
}

# --- Raw (pre-packing) data source locations --------------------------------
# Where to actually download each DATA_SOURCES entry's raw corpus from, before running
# packing.pack_source_to_shards on it. All public, freely downloadable sources -- no
# organization-internal assumption. `raw_path` is where YOU put it after downloading (or
# mirroring into internal storage); fill it in once you've done that. packing.py reads this
# dict directly, so filling in `raw_path` here is the only step needed to make packing.py
# runnable against real data.
RAW_SOURCE_PATHS = {
  "web": dict(
    download_from="https://huggingface.co/datasets/Skylion007/openwebtext",
    homepage="https://skylion007.github.io/OpenWebTextCorpus/",
    license="Unspecified by curator; an open replication of OpenAI's WebText corpus -- review before commercial use",
    raw_path="",
  ),
  "code": dict(
    download_from="https://huggingface.co/datasets/bigcode/the-stack-v2",
    homepage="https://www.bigcode-project.org/",
    license="Per-file license from source repos, opt-out-respecting; see the dataset card",
    raw_path="",
  ),
  "wikipedia": dict(
    download_from="https://huggingface.co/datasets/wikimedia/wikipedia",
    homepage="https://dumps.wikimedia.org/",
    license="CC BY-SA 4.0 / GFDL",
    raw_path="",
  ),
  "books": dict(
    download_from="https://huggingface.co/datasets/deepmind/pg19",
    homepage="https://www.gutenberg.org/",
    license="Public domain (pre-1919 Project Gutenberg texts)",
    raw_path="",
  ),
  "math": dict(
    download_from="https://huggingface.co/datasets/open-web-math/open-web-math",
    homepage="https://github.com/EleutherAI/math-lm",
    license="ODC-By 1.0",
    raw_path="",
  ),
}

# --- Data mixture ------------------------------------------------------------
# Each source lives under {data_root}/{subdir}/{split}/shard_id=*/part-*.parquet (see
# packing.pack_source_to_shards). `weight` is the target share of *sampled blocks*
# (renormalized across whichever sources are active for a run), not raw token count.
# `cycle=True` sources are reshuffled and repeated for the life of the epoch so a small
# high-quality source (wiki/books/math) can be upsampled relative to its natural size
# instead of running out early and dropping from the mixture.
# These weights are a starting point -- override per-run with --mixture_weights (see
# apply_weight_overrides) for ablations, rather than editing this file each time.
DATA_SOURCES = [
  dict(name="web",       subdir="openwebtext", weight=0.65, cycle=False),
  dict(name="code",      subdir="code",         weight=0.15, cycle=False),
  dict(name="wikipedia", subdir="wikipedia",    weight=0.10, cycle=True),
  dict(name="books",     subdir="books",        weight=0.07, cycle=True),
  dict(name="math",      subdir="math",         weight=0.03, cycle=True),
]


def apply_weight_overrides(data_sources: list, overrides_str: str) -> list:
  """Parses '--mixture_weights web=0.5,code=0.3,wikipedia=0.1,books=0.07,math=0.03' and
  returns a NEW list of DATA_SOURCES-shaped dicts with those weights substituted in (leaves
  `cycle`/`subdir` untouched, and leaves unmentioned sources at their config default) -- so a
  weight ablation is a CLI flag, not a code edit. Unknown names raise immediately rather than
  being silently ignored."""
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


# --- Eval-set sources (decontamination + benchmark_eval.py) -----------------
# Public HuggingFace dataset paths for the standard pretraining-era benchmarks. Used both by
# data_quality.decontaminate_against_eval_sets (needs the raw question/context text) and by
# benchmark_eval.py (needs the same text plus answers, to actually score the model).
# `local_path`, once filled in, points at a downloaded copy (parquet/jsonl) for environments
# without direct HuggingFace access at load time -- mirrors the tokenizer's local-first pattern.
EVAL_BENCHMARK_SOURCES = {
  "mmlu": dict(
    hf_path="cais/mmlu", hf_config="all",
    homepage="https://github.com/hendrycks/test",
    paper="https://arxiv.org/abs/2009.03300",
    kind="multiple_choice",  # question + 4 choices + answer index -- see benchmark_eval.py
    local_path="",
  ),
  "gsm8k": dict(
    hf_path="openai/gsm8k", hf_config="main",
    homepage="https://github.com/openai/grade-school-math",
    kind="generation_exact_match",  # needs sampling + numeric-answer extraction, not yet implemented
    local_path="",
  ),
  "humaneval": dict(
    hf_path="openai/openai_humaneval", hf_config=None,
    homepage="https://github.com/openai/human-eval",
    kind="generation_code_exec",  # needs sampling + sandboxed code execution (pass@k), not yet implemented
    local_path="",
  ),
}

# --- Quality classifier source -----------------------------------------------
# data_quality.quality_filter_classifier loads this model (real, public, purpose-built for
# exactly this job) rather than a from-scratch classifier -- see data_quality.py for how it's
# used and prd.md for the fuller writeup of what it is and where it came from.
QUALITY_CLASSIFIER = dict(
  model_id="HuggingFaceFW/fineweb-edu-classifier",
  homepage="https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier",
  reference_annotations="HuggingFaceFW/fineweb-edu-llama3-annotations",
  description="BERT-like regression model (built on Snowflake-arctic-embed-m) scoring "
              "'educational value' 0-5; trained on 450k Llama3-70B-Instruct annotations of "
              "FineWeb web samples. This is the actual classifier HuggingFace used to build "
              "FineWeb-Edu -- not a toy model.",
  score_threshold=3.0,  # per the model card: score >= 3 is the keep/remove cutoff FineWeb-Edu used
  local_model_path="",  # fill in once mirrored/downloaded internally, else loaded by model_id directly
)


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
  p.add_argument('--mixture_weights', type=str, default='',
                  help='Override DATA_SOURCES weights for this run, e.g. "web=0.5,code=0.3,wikipedia=0.1,books=0.07,math=0.03" '
                       '-- for ablations without editing config.py. See apply_weight_overrides.')
  p.add_argument('--tokenizer_path', type=str, default='',
                  help='Local path to a cl100k_base.tiktoken ranks file, for fully offline tokenizer '
                       'loading (see tokenizer.py). Defaults to TOKENIZER_SOURCES["cl100k_base"]["local_path"] '
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
                  help='Comma-separated subset of EVAL_BENCHMARK_SOURCES to run periodically during '
                       'training via benchmark_eval.py (currently: mmlu; gsm8k/humaneval need generation '
                       'infra not yet in this codebase -- see benchmark_eval.py). Requires --tokenizer_path '
                       'or a reachable tiktoken and each benchmark\'s local_path filled in or HF access.')

  return p.parse_known_args()
