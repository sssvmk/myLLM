"""Dataset registry (DS-1, DS-4, DS-5): resolves a dataset's split to files, streams raw records,
applies the adapter, and accounts for every dropped record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..io.storage import Storage
from .adapters import AdapterError, adapt_record


class DatasetError(RuntimeError):
    pass


class TooManyDropped(DatasetError):
    pass


@dataclass
class AdaptStats:
    """Filled while items are consumed; `check()` enforces DS-5 once the stream is exhausted."""
    name: str
    split: str
    read: int = 0
    kept: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def check(self, max_drop_fraction: float) -> None:
        if self.read and self.n_dropped / self.read > max_drop_fraction:
            raise TooManyDropped(
                f"dataset {self.name!r} split {self.split!r}: dropped {self.n_dropped}/{self.read} records "
                f"({self.n_dropped / self.read:.1%}) > prepared_data.max_drop_fraction={max_drop_fraction}; "
                f"reasons: {dict(sorted(self.dropped.items()))}")


def _file_stamp(info: Dict[str, Any]) -> Dict[str, Any]:
    """Size plus whichever change marker the filesystem offers (ETag, else modification time)."""
    stamp = info.get("ETag") or info.get("etag") or info.get("mtime") or info.get("LastModified") \
        or info.get("last_modified") or info.get("modified")
    return {"size": info.get("size"), "version": None if stamp is None else str(stamp)}


class DatasetRegistry:
    def __init__(self, datasets: Dict[str, Any], storage: Storage, seed: int, holdout_fraction: float,
                 max_drop_fraction: float):
        self.datasets = datasets
        self.storage = storage
        self.seed = seed
        self.holdout_fraction = holdout_fraction
        self.max_drop_fraction = max_drop_fraction

    @classmethod
    def from_config(cls, cfg, storage: Storage) -> "DatasetRegistry":
        return cls(cfg.datasets, storage, cfg.run.seed, cfg.prepared_data.holdout_fraction,
                   cfg.prepared_data.max_drop_fraction)

    # ------------------------------------------------------------------ splits and files
    def _ds(self, name: str):
        if name not in self.datasets:
            raise DatasetError(f"unknown dataset {name!r}; known: {sorted(self.datasets)}")
        return self.datasets[name]

    def has_split(self, name: str, split: str) -> bool:
        ds = self._ds(name)
        return split in ds.files or self._derived(ds, split)

    @staticmethod
    def _derived(ds, split: str) -> bool:
        """DS-4: with no configured test split, train and test are both derived from train."""
        return split in ("train", "test") and "train" in ds.files and "test" not in ds.files

    def resolve_files(self, name: str, split: str) -> List[str]:
        ds = self._ds(name)
        src = split if split in ds.files else ("train" if self._derived(ds, split) else None)
        if src is None:
            raise DatasetError(f"dataset {name!r} has no split {split!r}; configured splits: {sorted(ds.files)}")
        files = self.storage.glob(ds.uri, ds.files[src])
        if not files:
            raise DatasetError(f"dataset {name!r} split {src!r}: glob {ds.files[src]!r} under {ds.uri!r} matched no files")
        return files

    def file_manifest(self, name: str, split: str) -> List[Dict[str, Any]]:
        return [dict(path=f, **_file_stamp(self.storage.info(f))) for f in self.resolve_files(name, split)]

    # ------------------------------------------------------------------ raw records
    def _read_file(self, ds, uri: str) -> Iterator[Dict[str, Any]]:
        if ds.format in ("parquet", "packed_tokens"):
            import pyarrow.parquet as pq
            with self.storage.open(uri) as fh:
                for batch in pq.ParquetFile(fh).iter_batches(batch_size=1024):
                    yield from batch.to_pylist()
        elif ds.format == "jsonl":
            with self.storage.open(uri) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif ds.format == "json":
            with self.storage.open(uri) as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise DatasetError(f"{uri}: format json must be a top-level list of records")
            yield from data
        else:  # pragma: no cover -- schema restricts the literal
            raise DatasetError(f"unsupported format {ds.format!r}")

    def _is_holdout(self, record: Dict[str, Any]) -> bool:
        """DS-4: deterministic, hash of (seed, record); independent of file order and worker count."""
        blob = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)
        h = hashlib.sha256(f"{self.seed}|{blob}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64 < self.holdout_fraction

    def raw_records(self, name: str, split: str) -> Iterator[Dict[str, Any]]:
        ds = self._ds(name)
        files = self.resolve_files(name, split)
        # DS-4: with no configured test split BOTH train and test are derived from the train files,
        # so train must exclude the holdout too (otherwise the two overlap).
        derived = self._derived(ds, split)
        n = 0
        for uri in files:
            for rec in self._read_file(ds, uri):
                if derived and self._is_holdout(rec) != (split == "test"):
                    continue
                if ds.max_samples is not None and n >= ds.max_samples:
                    return
                n += 1
                yield rec

    # ------------------------------------------------------------------ adapted items
    def items(self, name: str, split: str, stats: Optional[AdaptStats] = None) -> Iterator[Any]:
        """Adapted items for one split. Dropped records are counted in `stats` by reason; DS-5 is
        enforced when the stream is exhausted."""
        ds = self._ds(name)
        stats = stats if stats is not None else AdaptStats(name, split)
        for rec in self.raw_records(name, split):
            stats.read += 1
            try:
                item = adapt_record(ds, rec)
            except AdapterError as e:
                stats.drop(e.reason)
                continue
            stats.kept += 1
            yield item
        stats.check(self.max_drop_fraction)

    def manifest_entry(self, name: str, split: str, stats: AdaptStats,
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ds = self._ds(name)
        entry = {"name": name, "split": split, "uri": ds.uri, "files": self.file_manifest(name, split),
                 "license": ds.license, "third_party_generated": ds.third_party_generated,
                 "records_read": stats.read, "records_kept": stats.kept, "dropped": dict(stats.dropped)}
        entry.update(extra or {})
        return entry
