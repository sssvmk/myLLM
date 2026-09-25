"""All reads and writes go through here (ST-1..ST-4).

Location forms: plain local paths, /dbfs/... mounts, file://, abfss:// and az:// (adlfs),
s3:// (s3fs), gs:// (gcsfs), hf://... (huggingface_hub). Per-protocol options come from
`storage.protocols.<protocol>` and are passed to the fsspec filesystem constructor.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

import fsspec

log = logging.getLogger(__name__)

_LOCAL = ("", "file", "local")


class Storage:
    def __init__(self, protocols: Dict[str, Dict[str, Any]], upload_retries: int, local_work_dir: str,
                 retry_sleep_s: float = 1.0):
        self.protocols = protocols
        self.upload_retries = upload_retries
        self.local_work_dir = local_work_dir
        self.retry_sleep_s = retry_sleep_s
        self._fs_cache: Dict[str, fsspec.AbstractFileSystem] = {}

    @classmethod
    def from_config(cls, cfg) -> "Storage":
        return cls(cfg.storage.protocols, cfg.storage.upload_retries, cfg.run.local_work_dir)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def protocol_of(uri: str) -> str:
        return uri.split("://", 1)[0] if "://" in uri else ""

    @staticmethod
    def is_local(uri: str) -> bool:
        return Storage.protocol_of(uri) in _LOCAL

    @staticmethod
    def local_path(uri: str) -> str:
        return uri[len("file://"):] if uri.startswith("file://") else uri

    def fs(self, uri: str):
        proto = self.protocol_of(uri)
        if proto in _LOCAL:
            return fsspec.filesystem("file"), self.local_path(uri)
        if proto not in self._fs_cache:
            opts = dict(self.protocols.get(proto, {}))
            self._fs_cache[proto] = fsspec.filesystem(proto, **opts)
        fs = self._fs_cache[proto]
        return fs, fs._strip_protocol(uri)

    @staticmethod
    def join(uri: str, *parts: str) -> str:
        out = uri.rstrip("/")
        for p in parts:
            out = out + "/" + p.strip("/")
        return out

    # ------------------------------------------------------------------ reads
    def exists(self, uri: str) -> bool:
        fs, p = self.fs(uri)
        return fs.exists(p)

    def glob(self, root_uri: str, pattern: str) -> List[str]:
        fs, root = self.fs(root_uri)
        matches = sorted(fs.glob(root.rstrip("/") + "/" + pattern))
        proto = self.protocol_of(root_uri)
        if proto in _LOCAL:
            return matches
        return [f"{proto}://{m}" for m in matches]

    def info(self, uri: str) -> Dict[str, Any]:
        fs, p = self.fs(uri)
        return fs.info(p)

    def read_bytes(self, uri: str) -> bytes:
        fs, p = self.fs(uri)
        with fs.open(p, "rb") as f:
            return f.read()

    def read_text(self, uri: str) -> str:
        return self.read_bytes(uri).decode("utf-8")

    def open(self, uri: str, mode: str = "rb"):
        if "r" not in mode:
            raise ValueError("Storage.open is read-only; use write_bytes/put_file for writes (ST-3)")
        fs, p = self.fs(uri)
        return fs.open(p, mode)

    def cached_local_path(self, uri: str) -> str:
        """Local path for a (possibly remote) file. Remote files are cached under
        local_work_dir/cache/, keyed by URI and ETag/modification time (ST-3)."""
        if self.is_local(uri):
            return self.local_path(uri)
        info = self.info(uri)
        version = str(info.get("ETag") or info.get("etag") or info.get("last_modified")
                      or info.get("mtime") or info.get("LastModified") or info.get("size"))
        key = hashlib.sha256(f"{uri}|{version}".encode()).hexdigest()
        cache_dir = os.path.join(self.local_work_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, key + "_" + os.path.basename(uri.rstrip("/")))
        if not os.path.exists(dest):
            fs, p = self.fs(uri)
            tmp = dest + ".part"
            fs.get(p, tmp)
            os.replace(tmp, dest)
        return dest

    # ------------------------------------------------------------------ writes
    def _put(self, fs, local_file: str, remote_path: str):
        """Single upload attempt. Separate method so tests can inject failures (AT-2)."""
        fs.put(local_file, remote_path)

    def put_file(self, local_file: str, uri: str) -> None:
        """Upload a local file to `uri`, retrying, and verify size before returning (ST-3, ST-4)."""
        size = os.path.getsize(local_file)
        if self.is_local(uri):
            dest = self.local_path(uri)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            tmp = dest + ".tmp"
            shutil.copyfile(local_file, tmp)
            os.replace(tmp, dest)
            if os.path.getsize(dest) != size:
                raise IOError(f"size mismatch after write to {uri}")
            return
        fs, p = self.fs(uri)
        last: Optional[Exception] = None
        for attempt in range(self.upload_retries + 1):
            try:
                self._put(fs, local_file, p)
                remote_size = fs.info(p)["size"]
                if remote_size != size:
                    raise IOError(f"size mismatch after upload to {uri}: {remote_size} != {size}")
                return
            except Exception as e:  # noqa: BLE001 -- any transport error is retried
                last = e
                log.warning("upload attempt %d/%d to %s failed: %s", attempt + 1, self.upload_retries + 1, uri, e)
                time.sleep(self.retry_sleep_s)
        raise IOError(f"upload to {uri} failed after {self.upload_retries + 1} attempts: {last}")

    def write_bytes(self, uri: str, data: bytes) -> None:
        staging = os.path.join(self.local_work_dir, "staging")
        os.makedirs(staging, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=staging)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            self._put_with_local_fallback(tmp, uri)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _put_with_local_fallback(self, tmp: str, uri: str):
        self.put_file(tmp, uri)

    def write_text(self, uri: str, text: str) -> None:
        self.write_bytes(uri, text.encode("utf-8"))

    def put_dir(self, local_dir: str, uri: str) -> None:
        for root, _, files in os.walk(local_dir):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, local_dir).replace(os.sep, "/")
                self.put_file(full, self.join(uri, rel))

    def delete(self, uri: str) -> None:
        fs, p = self.fs(uri)
        if fs.exists(p):
            fs.rm(p, recursive=True)
