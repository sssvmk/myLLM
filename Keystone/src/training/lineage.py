"""Lineage records and stage output artifacts (CK-1, CK-2).

`final/model.pt` = {"model": weights, "lineage": lineage}; `final/lineage.json` is the same record
plus `output_sha256` (the sha of model.pt, computed once here so children never re-hash a
multi-gigabyte parent, see docs/prd_review.md #13). `_COMPLETE` is written last; it holds the
configuration hash that PL-5 compares.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import torch

from ..config.loader import to_yaml
from ..io.storage import Storage
from .. import __version__


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_path(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_uri(storage: Storage, uri: str) -> str:
    """sha256 of a (possibly remote) file. Results are cached in local_work_dir keyed by URI, size
    and change marker, so a large base checkpoint is hashed once per machine."""
    info = storage.info(uri)
    stamp = str(info.get("ETag") or info.get("etag") or info.get("mtime") or info.get("LastModified")
                or info.get("last_modified") or "")
    key = hashlib.sha256(f"{uri}|{info.get('size')}|{stamp}".encode()).hexdigest()
    cache_dir = os.path.join(storage.local_work_dir, "sha_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, key)
    if os.path.exists(cache):
        return open(cache).read().strip()
    digest = sha256_path(storage.cached_local_path(uri))
    with open(cache, "w") as f:
        f.write(digest)
    return digest


def git_commit() -> Optional[str]:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       timeout=5).decode().strip()
    except Exception:  # noqa: BLE001 -- "when available"
        return None


def data_manifest(manifest: Dict[str, Dict[str, Any]], mixture: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CK-2: every dataset used, with files, license, third-party flag, counts after filtering and
    mixture weights. `manifest` is {dataset: {split: entry}} from preparation; `mixture` lists
    {dataset, weight, rows_per_epoch}."""
    out = []
    by_name = {m["dataset"]: m for m in mixture}
    for name, splits in sorted(manifest.items()):
        first = next(iter(splits.values()))
        out.append({"name": name, "uri": first.get("uri"), "license": first.get("license"),
                    "third_party_generated": first.get("third_party_generated"),
                    "splits": {s: {"files": e.get("files"), "records_read": e.get("records_read"),
                                   "records_kept": e.get("records_kept"), "dropped": e.get("dropped"),
                                   "rows": e.get("rows"), "decontamination": e.get("decontamination")}
                               for s, e in splits.items()},
                    "mixture": by_name.get(name)})
    return out


def build_lineage(*, stage: str, parent_uri: str, parent_sha256: str, parent_stage: Optional[str],
                  architecture: Dict[str, Any], tokenizer: Dict[str, Any], chat_template: Dict[str, Any],
                  data: List[Dict[str, Any]], config_hash: str, device: str, started: str, final_step: int,
                  services: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = {"stage": stage, "parent": {"uri": parent_uri, "sha256": parent_sha256, "stage": parent_stage},
           "architecture": architecture, "tokenizer": tokenizer, "chat_template": chat_template,
           "data": data, "services": services or {}, "config_hash": config_hash,
           "keystone_version": __version__, "git_commit": git_commit(), "device": device,
           "started_at": started, "ended_at": now_iso(), "final_step": final_step}
    rec.update(extra or {})
    return rec


def write_final(storage: Storage, out_uri: str, state_dict: Dict[str, torch.Tensor], lineage: Dict[str, Any],
                redacted_config: Dict[str, Any]) -> str:
    """Writes final/model.pt, lineage.json, resolved_config.yaml and returns model.pt's sha256.
    `_COMPLETE` is NOT written here; the caller writes it last (ST-4)."""
    final = Storage.join(out_uri, "final")
    os.makedirs(storage.local_work_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=storage.local_work_dir) as d:
        p = os.path.join(d, "model.pt")
        torch.save({"model": state_dict, "lineage": lineage}, p)
        sha = sha256_path(p)
        storage.put_file(p, Storage.join(final, "model.pt"))
    storage.write_text(Storage.join(final, "lineage.json"),
                       json.dumps(dict(lineage, output_sha256=sha), indent=2, sort_keys=True, default=str))
    storage.write_text(Storage.join(final, "resolved_config.yaml"), to_yaml(redacted_config))
    return sha


def write_complete(storage: Storage, out_uri: str, config_hash: str, output_sha256: str, final_step: int) -> None:
    storage.write_text(Storage.join(out_uri, "_COMPLETE"),
                       json.dumps({"config_hash": config_hash, "output_sha256": output_sha256,
                                   "final_step": final_step, "completed_at": now_iso()}, sort_keys=True))


def read_complete(storage: Storage, out_uri: str) -> Optional[Dict[str, Any]]:
    uri = Storage.join(out_uri, "_COMPLETE")
    return json.loads(storage.read_text(uri)) if storage.exists(uri) else None


def read_lineage(storage: Storage, out_uri: str) -> Optional[Dict[str, Any]]:
    uri = Storage.join(out_uri, "final", "lineage.json")
    return json.loads(storage.read_text(uri)) if storage.exists(uri) else None
