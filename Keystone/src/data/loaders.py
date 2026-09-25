"""Streaming Parquet loaders over prepared data (DL-1..DL-3).

* Sharding (DL-1): with at least `world` shard files, each file is read by exactly one rank
  (`files[rank::world]`). With fewer files than ranks, every rank reads every file but keeps rows
  `i % world == rank` of the file's shuffled order, so each row still goes to exactly one rank.
  (`num_workers` in DP-2 is the preparation pool; training reads in-process, so "per worker" is the
  per-rank rule above.)
* Shuffling is seeded per epoch and per file: same seed, epoch and shard list give the same order.
* Position is `consumed`, the number of rows returned since construction. `fast_forward(n)`
  reproduces a saved position by skipping whole files from Parquet metadata, so resume (CK-5) does
  not re-read skipped shards.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pyarrow.parquet as pq
import torch

from ..io.storage import Storage
from .prepare import SHARD_GLOB


class DataError(RuntimeError):
    pass


class RowStream:
    def __init__(self, storage: Storage, uri: str, columns: Sequence[str], rank: int, world: int, seed: int,
                 cycle: bool, shuffle: bool = True):
        self.storage, self.uri, self.columns = storage, uri, list(columns)
        self.rank, self.world, self.seed, self.cycle, self.shuffle = rank, world, seed, cycle, shuffle
        self.files = storage.glob(uri, SHARD_GLOB)
        if not self.files:
            raise DataError(f"no prepared shards under {uri}/{SHARD_GLOB}")
        self._nrows: Dict[str, int] = {}
        self.consumed = 0
        self.epoch = 0
        self._it: Optional[Iterator[Dict[str, Any]]] = None
        self._skip_pending = 0

    # ------------------------------------------------------------------ layout
    def _local(self, f: str) -> str:
        return self.storage.cached_local_path(f)

    def _count(self, f: str) -> int:
        if f not in self._nrows:
            self._nrows[f] = pq.ParquetFile(self._local(f)).metadata.num_rows
        return self._nrows[f]

    def _file_plan(self, epoch: int) -> List[tuple]:
        """[(file, row_stride_offset, row_stride)] for this rank in `epoch`."""
        files = list(self.files)
        if self.shuffle:
            random.Random(f"{self.seed}:{epoch}").shuffle(files)
        if len(files) >= self.world:
            return [(f, 0, 1) for f in files[self.rank::self.world]]
        if sum(self._count(f) for f in files) < self.world:
            return [(f, 0, 1) for f in files]      # degenerate: fewer rows than ranks, every rank reads all of them
        return [(f, self.rank, self.world) for f in files]

    def _rows_for(self, f: str, offset: int, stride: int) -> int:
        n = self._count(f)
        return (n - offset + stride - 1) // stride if n > offset else 0

    def rows_per_epoch(self) -> int:
        return sum(self._rows_for(f, o, s) for f, o, s in self._file_plan(0))

    def _read_file(self, f: str, epoch: int, offset: int, stride: int) -> List[Dict[str, Any]]:
        table = pq.read_table(self._local(f), columns=self.columns)
        data = {c: table.column(c).to_pylist() for c in self.columns}
        n = table.num_rows
        order = list(range(n))
        if self.shuffle:
            random.Random(f"{self.seed}:{epoch}:{f.rsplit('/', 1)[-1]}").shuffle(order)      # by shard name, not full URI: order must not depend on where the run lives
        order = order[offset::stride]
        return [{c: data[c][i] for c in self.columns} for i in order]

    # ------------------------------------------------------------------ iteration
    def _generate(self) -> Iterator[Dict[str, Any]]:
        while True:
            produced = 0
            for f, o, s in self._file_plan(self.epoch):
                n = self._rows_for(f, o, s)
                if self._skip_pending >= n:
                    self._skip_pending -= n
                    self.consumed += n
                    produced += n
                    continue
                rows = self._read_file(f, self.epoch, o, s)
                if self._skip_pending:
                    rows = rows[self._skip_pending:]
                    self.consumed += self._skip_pending
                    produced += self._skip_pending
                    self._skip_pending = 0
                for r in rows:
                    self.consumed += 1
                    produced += 1
                    yield r
            if not self.cycle:
                return
            if produced == 0:
                raise DataError(f"rank {self.rank}/{self.world} has no rows under {self.uri} (fewer rows than ranks?)")
            self.epoch += 1

    def fast_forward(self, n: int) -> None:
        if self._it is not None or self.consumed:
            raise DataError("fast_forward must be called on a fresh stream")
        self._skip_pending = n

    def next(self) -> Dict[str, Any]:
        if self._it is None:
            self._it = self._generate()
        try:
            return next(self._it)
        except StopIteration:
            raise

    def __iter__(self):
        while True:
            try:
                yield self.next()
            except StopIteration:
                return


# --------------------------------------------------------------------------- collation (DL-2, DL-3)
def collate_blocks(rows: List[Dict[str, Any]], pad_id: int, lm: bool) -> Dict[str, torch.Tensor]:
    """LM / SFT batch: tokens (B, ctx+1), doc_start, loss_mask. LM loss mask is True except padding."""
    tokens = torch.tensor([r["tokens"] for r in rows], dtype=torch.long)
    doc_start = torch.tensor([r["doc_start"] for r in rows], dtype=torch.bool)
    loss_mask = tokens != pad_id if lm else torch.tensor([r["loss_mask"] for r in rows], dtype=torch.bool)
    return {"tokens": tokens, "doc_start": doc_start, "loss_mask": loss_mask}


def pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def response_batch(prompts: List[List[int]], responses: List[List[int]], pad_id: int):
    """DL-3: right-padded `prompt + response` sequences and a mask over *targets* (positions
    1..L-1 of the sequence) that is True on response tokens."""
    seqs = [p + r for p, r in zip(prompts, responses)]
    idx = pad_sequences(seqs, pad_id)
    mask = torch.zeros(idx.shape, dtype=torch.bool)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        mask[i, len(p):len(p) + len(r)] = True
    return idx, mask[:, 1:]


def collate_pref(rows: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Chosen sequences first, then rejected, in one (2B, L) tensor; `resp_mask` is over targets."""
    prompts = [r["prompt_tokens"] for r in rows]
    idx, mask = response_batch(prompts + prompts, [r["chosen_tokens"] for r in rows] + [r["rejected_tokens"] for r in rows],
                               pad_id)
    out = {"idx": idx, "resp_mask": mask, "n": torch.tensor(len(rows)),
           "len_chosen": torch.tensor([len(r["chosen_tokens"]) for r in rows]),
           "len_rejected": torch.tensor([len(r["rejected_tokens"]) for r in rows])}
    if rows and "ref_logp_chosen" in rows[0]:
        out["ref_chosen"] = torch.tensor([r["ref_logp_chosen"] for r in rows], dtype=torch.float64)
        out["ref_rejected"] = torch.tensor([r["ref_logp_rejected"] for r in rows], dtype=torch.float64)
    return out
