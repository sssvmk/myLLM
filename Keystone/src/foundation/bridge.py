"""The only place foundation_llm modules are imported (FL-3..FL-7).

`load(code_path)` puts the foundation_llm directory at the front of sys.path, imports the
allowed modules with importlib, and checks every allowed name exists. After `load`, other
Keystone modules use `bridge.fl_model`, `bridge.fl_moe`, ... -- never `import model`.
"""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Dict, List, Optional

ALLOWED: Dict[str, List[str]] = {
    "model": ["GPTModel", "Block", "RMSNorm", "CausalSelfAttention", "MultiHeadLatentAttention",
              "build_rope_cache", "rotate_half", "build_doc_attention_mask"],
    "moe": ["DeepSeekMoE"],
    "tokenizer": ["_PAT_STR", "_SPECIAL_TOKENS"],
    "distributed": ["setup_distributed", "is_distributed", "is_main_process", "wrap_model",
                    "build_optimizer_for_zero", "full_model_state_dict", "full_optimizer_state_dict",
                    "load_full_model_state_dict", "load_full_optimizer_state_dict"],
    "metrics": ["MetricsLogger"],
}
FORBIDDEN = {"config", "data", "train", "main", "packing", "data_quality", "benchmark_eval",
             "scheduler", "plot_metrics"}

fl_model: Optional[ModuleType] = None
fl_moe: Optional[ModuleType] = None
fl_tokenizer: Optional[ModuleType] = None
fl_distributed: Optional[ModuleType] = None
fl_metrics: Optional[ModuleType] = None
_loaded_from: Optional[str] = None


class BridgeError(ImportError):
    pass


def import_foundation_module(name: str) -> ModuleType:
    if name in FORBIDDEN or name not in ALLOWED:
        raise BridgeError(f"foundation_llm module {name!r} may not be imported (FL-3/FL-4); allowed: {sorted(ALLOWED)}")
    return importlib.import_module(name)


def load(code_path: str) -> None:
    global fl_model, fl_moe, fl_tokenizer, fl_distributed, fl_metrics, _loaded_from
    code_path = os.path.abspath(code_path)
    if _loaded_from == code_path:
        return
    if _loaded_from is not None:
        raise BridgeError(f"bridge already loaded from {_loaded_from}; cannot reload from {code_path}")
    if not os.path.isfile(os.path.join(code_path, "model.py")):
        raise BridgeError(f"foundation_llm.code_path {code_path} has no model.py")
    if code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.insert(0, code_path)
    mods = {name: import_foundation_module(name) for name in ["model", "moe", "tokenizer", "distributed", "metrics"]}
    missing = [f"{m}.{n}" for m, names in ALLOWED.items() for n in names if not hasattr(mods[m], n)]
    if missing:
        raise BridgeError(f"foundation_llm at {code_path} is missing required names: {missing}")
    for m, mod in mods.items():
        if not os.path.abspath(mod.__file__).startswith(code_path):
            raise BridgeError(f"module {m!r} resolved to {mod.__file__}, not under {code_path} (shadowed?)")
    fl_model, fl_moe, fl_tokenizer = mods["model"], mods["moe"], mods["tokenizer"]
    fl_distributed, fl_metrics = mods["distributed"], mods["metrics"]
    _loaded_from = code_path


def require() -> None:
    if _loaded_from is None:
        raise BridgeError("src.foundation.bridge.load(code_path) has not been called")
