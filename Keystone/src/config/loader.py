"""Loads one YAML file with OmegaConf, applies `--set` dot-list overrides, resolves
interpolations, and validates with the Pydantic schema (CF-2, CF-3). Secrets are redacted in the
copy that gets saved anywhere (CF-4, CF-5)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from .schema import PFConfig

REDACTED = "***REDACTED***"
_SECRET_KEY = re.compile(r"(^|_)(key|token|secret|password|credential|credentials|sas_token|connection_string)$", re.IGNORECASE)
_ENV_INTERP = re.compile(r"\$\{oc\.env:")


class ConfigError(ValueError):
    """Raised with every problem found, one per line, each starting with the dotted key path."""


@dataclass
class LoadedConfig:
    cfg: PFConfig
    resolved: Dict[str, Any]      # contains secrets -- never persist
    redacted: Dict[str, Any]      # safe to write to disk / logs
    path: str
    overrides: List[str]


def _format_pydantic(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        kind = e["type"]
        if kind == "missing":
            lines.append(f"{loc}: missing required key")
        elif kind == "extra_forbidden":
            lines.append(f"{loc}: unknown key")
        else:
            lines.append(f"{loc}: {e['msg']}")
    return "\n".join(lines)


def _redact(raw_node, resolved_node, path=()):
    """Walks the unresolved OmegaConf container alongside the resolved one. A value is redacted
    when it came from an environment-variable interpolation under a secret-looking key, or sits
    anywhere under storage.protocols (all fsspec credentials)."""
    if isinstance(resolved_node, dict):
        out = {}
        for k, v in resolved_node.items():
            raw_v = raw_node.get(k) if isinstance(raw_node, dict) else None
            out[k] = _redact(raw_v, v, path + (str(k),))
        return out
    if isinstance(resolved_node, list):
        raw_list = raw_node if isinstance(raw_node, list) else [None] * len(resolved_node)
        return [_redact(r, v, path + (str(i),)) for i, (r, v) in enumerate(zip(raw_list, resolved_node))]
    under_protocols = len(path) >= 3 and path[0] == "storage" and path[1] == "protocols"
    from_env = isinstance(raw_node, str) and bool(_ENV_INTERP.search(raw_node))
    secret_key = bool(path) and bool(_SECRET_KEY.search(path[-1])) and not path[-1].endswith("_env")
    if resolved_node is not None and (under_protocols or secret_key or (from_env and secret_key)):
        return REDACTED
    return resolved_node


def load_config(path: str, overrides: Sequence[str] = ()) -> LoadedConfig:
    try:
        base = OmegaConf.load(path)
        for ov in overrides:
            if "=" not in ov:
                raise ConfigError(f"--set {ov!r}: expected key=value")
            key, value = ov.split("=", 1)
            parsed = OmegaConf.from_dotlist([f"v={value}"])._get_node("v")._value()  # YAML-typed value
            OmegaConf.update(base, key.strip(), parsed, merge=True, force_add=True)
        raw = OmegaConf.to_container(base, resolve=False)
        resolved = OmegaConf.to_container(base, resolve=True, throw_on_missing=True)
    except OmegaConfBaseException as e:
        key = getattr(e, "full_key", None) or ""
        raise ConfigError(f"{key}: {e.__class__.__name__}: {getattr(e, 'msg', e)}") from None
    try:
        cfg = PFConfig.model_validate(resolved)
    except ValidationError as e:
        raise ConfigError(_format_pydantic(e)) from None
    return LoadedConfig(cfg=cfg, resolved=resolved, redacted=_redact(raw, resolved),
                        path=path, overrides=list(overrides))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def to_yaml(obj: Dict[str, Any]) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(obj))
