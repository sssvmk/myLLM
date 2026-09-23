"""Data loading.

ParquetTokenDataset streams fixed-length token blocks from `{path}/shard_id=*/part-*.parquet`
and fixes two bugs from the original version:
  - padding: short trailing blocks are padded with a *reserved* pad_token_id (the caller must
    pass config.pad_token_id_for(vocab_size), one past the real vocab) instead of id 0, which
    was a real cl100k_base token ("!"). Pass the same id as ignore_index to F.cross_entropy
    (train.py does this) so padded positions never contribute to the loss.
  - worker safety: with DataLoader(num_workers>0), each worker now only reads its own shard of
    `self.files` instead of every worker re-reading every file.

If the parquet schema has a `doc_start` column (bool, aligned with `tokens`; produced by
packing.pack_source_to_shards' EOS-insertion path) it's read and yielded alongside `tokens` so
the training loop can build a document-boundary attention mask (model.build_doc_attention_mask)
instead of letting attention cross a concatenated-document seam. Every item is always a
(tokens, doc_start) tuple for a consistent structure -- when the column is absent, doc_start is
synthesized as all-False except position 0 (equivalent to "whole block is one segment", i.e.
today's un-masked behavior).

MixtureIterableDataset samples blocks from several sources by weight (config.DATA_SOURCES);
build_mixture_dataset is the convenience constructor main.py uses.
"""
from __future__ import annotations
import os, glob, random
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

from config import DATA_SOURCES


class ParquetTokenDataset(torch.utils.data.IterableDataset):
  def __init__(self, path: str, ctx: int, pad_token_id: int, shuffle: bool = True):
    self.path = path
    self.block_size = ctx + 1
    self.files = sorted(glob.glob(f"{path}/shard_id=*/part-*.parquet"))
    if not self.files:
      raise FileNotFoundError(f"no shards found under {path}/shard_id=*/part-*.parquet")
    self.shuffle = shuffle
    self.pad_token_id = pad_token_id
    self.has_doc_starts = "doc_start" in pq.ParquetFile(self.files[0]).schema_arrow.names

  def _worker_files(self) -> List[str]:
    files = self.files[:]
    if self.shuffle:
      random.shuffle(files)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      files = files[worker_info.id::worker_info.num_workers]   # fix: was previously read in full by every worker
    return files

  def __iter__(self):
    cols = ["tokens", "doc_start"] if self.has_doc_starts else ["tokens"]
    for f in self._worker_files():
      table = pq.read_table(f, columns=cols, use_threads=True)
      for batch in table.to_batches():
        tok_col = batch["tokens"]
        ds_col = batch["doc_start"] if self.has_doc_starts else None
        for i in range(len(tok_col)):
          tokens = tok_col[i].as_py()
          n = len(tokens)
          if self.has_doc_starts:
            flags = ds_col[i].as_py()
          else:
            flags = [j == 0 for j in range(n)]

          if n < self.block_size:
            pad = self.block_size - n
            block = F.pad(torch.tensor(tokens, dtype=torch.long), (0, pad), "constant", self.pad_token_id)
            doc_start = F.pad(torch.tensor(flags, dtype=torch.bool), (0, pad), "constant", False)
          else:
            block = torch.tensor(tokens[:self.block_size], dtype=torch.long)
            doc_start = torch.tensor(flags[:self.block_size], dtype=torch.bool)

          yield block, doc_start


def _infinite_iter(dataset: "ParquetTokenDataset"):
  """Reshuffles and re-iterates `dataset` forever. Used for upsampled ('cycle') sources so a
  small high-quality corpus doesn't get exhausted early and silently drop out of the mixture
  while the larger driving source(s) are still going."""
  while True:
    for item in dataset:
      yield item


class MixtureIterableDataset(torch.utils.data.IterableDataset):
  """Samples blocks from several ParquetTokenDataset sources according to target mixture
  weights, so one pass over this dataset reflects the configured token mixture rather than
  each source's natural on-disk size.

  At least one source must have cycle=False (a "driving" source): the epoch ends once every
  driving source is exhausted. cycle=True sources repeat indefinitely and simply stop being
  drawn from at that point.
  """
  def __init__(self, datasets: List[ParquetTokenDataset], weights: List[float],
               cycle_flags: List[bool], names: Optional[List[str]] = None, seed: int = 1337):
    assert len(datasets) == len(weights) == len(cycle_flags)
    assert any(not c for c in cycle_flags), "at least one non-cycling (driving) source is required"
    total = sum(weights)
    self.datasets = datasets
    self.weights = [w / total for w in weights]
    self.cycle_flags = cycle_flags
    self.names = names or [f"source_{i}" for i in range(len(datasets))]
    self.seed = seed

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    seed = self.seed if worker_info is None else self.seed + worker_info.id
    rng = random.Random(seed)

    iters = [
      _infinite_iter(ds) if cyc else iter(ds)
      for ds, cyc in zip(self.datasets, self.cycle_flags)
    ]
    driving = {i for i, c in enumerate(self.cycle_flags) if not c}
    active = set(range(len(iters)))
    weights = self.weights[:]

    while active & driving:
      choices = list(active)
      idx = rng.choices(choices, weights=[weights[i] for i in choices], k=1)[0]
      try:
        block = next(iters[idx])
      except StopIteration:
        active.discard(idx)
        continue
      yield block


def build_mixture_dataset(data_root: str, split: str, ctx: int, pad_token_id: int,
                           sources: Optional[List[str]] = None, shuffle: bool = True,
                           source_configs: Optional[List[dict]] = None) -> MixtureIterableDataset:
  """Builds a MixtureIterableDataset for `split` ('train' or 'test') from DATA_SOURCES (or
  `source_configs`, if given -- e.g. config.apply_weight_overrides' output, for a per-run
  mixture-weight ablation without editing config.py), reading each source from
  {data_root}/{subdir}/{split}/. `sources`, if given, restricts to a subset of source names
  (e.g. for a quick ablation on 2-3 sources before committing to the full mix)."""
  base = source_configs if source_configs is not None else DATA_SOURCES
  cfgs = [c for c in base if sources is None or c["name"] in sources]
  if not cfgs:
    raise ValueError(f"no DATA_SOURCES match sources={sources}")

  datasets, weights, cycle_flags, names = [], [], [], []
  for cfg in cfgs:
    path = os.path.join(data_root, cfg["subdir"], split)
    datasets.append(ParquetTokenDataset(path, ctx, pad_token_id, shuffle=shuffle))
    weights.append(cfg["weight"])
    cycle_flags.append(cfg["cycle"])
    names.append(cfg["name"])

  print(f"[{split}] mixture sources: " +
        ", ".join(f"{n}(w={w:.2f}{',cycle' if c else ''})" for n, w, c in zip(names, weights, cycle_flags)))
  return MixtureIterableDataset(datasets, weights, cycle_flags, names=names)
