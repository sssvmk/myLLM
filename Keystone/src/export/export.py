"""Final model export and reload (EX-1, EX-2)."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from ..modeling.loading import build_model, strip_prefixes, unwrap

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
RANKS_NAME = "cl100k_base.tiktoken"


def export_model(model, architecture: Dict[str, Any], storage, out_uri: str, dtype: str,
                 ranks_file_uri: str, tokenizer_record: Dict[str, Any], chat_template: Dict[str, Any],
                 lineage: Optional[Dict[str, Any]] = None, eval_report: Optional[Dict[str, Any]] = None) -> None:
    sd = strip_prefixes(unwrap(model).state_dict())
    sd.pop("lm_head.weight", None)                                   # tied to tok_emb.weight
    sd = {k: v.detach().to("cpu", DTYPES[dtype]).contiguous() for k, v in sd.items()}
    arch = dict(architecture, tied_embeddings=True)
    tok = dict(tokenizer_record, ranks_file=RANKS_NAME, chat_template=chat_template)
    with tempfile.TemporaryDirectory() as d:
        st = os.path.join(d, "model.safetensors")
        save_file(sd, st, metadata={"dtype": dtype})
        storage.put_file(st, storage.join(out_uri, "model.safetensors"))
    storage.write_text(storage.join(out_uri, "architecture.json"), json.dumps(arch, indent=2, sort_keys=True))
    storage.write_text(storage.join(out_uri, "tokenizer.json"), json.dumps(tok, indent=2, sort_keys=True))
    storage.write_bytes(storage.join(out_uri, RANKS_NAME), storage.read_bytes(ranks_file_uri))
    if lineage is not None:
        storage.write_text(storage.join(out_uri, "lineage.json"), json.dumps(lineage, indent=2, sort_keys=True, default=str))
    if eval_report is not None:
        storage.write_text(storage.join(out_uri, "eval_report.json"), json.dumps(eval_report, indent=2, sort_keys=True))


def load_exported(uri: str, storage, device: torch.device, compute_dtype: Optional[torch.dtype] = None):
    """Rebuilds (model, architecture, tokenizer_record) from export files alone. Weights stay in
    the exported dtype unless `compute_dtype` is given."""
    arch = json.loads(storage.read_text(storage.join(uri, "architecture.json")))
    tok = json.loads(storage.read_text(storage.join(uri, "tokenizer.json")))
    sd = load_file(storage.cached_local_path(storage.join(uri, "model.safetensors")))
    model_arch = {k: v for k, v in arch.items() if k != "tied_embeddings"}
    model = build_model(model_arch)
    if compute_dtype is None:
        compute_dtype = next(iter(sd.values())).dtype
    model = model.to(compute_dtype)
    sd["lm_head.weight"] = sd["tok_emb.weight"]
    model.load_state_dict({k: v.to(compute_dtype) for k, v in sd.items()}, strict=True)
    model.lm_head.weight = model.tok_emb.weight
    return model.to(device).eval(), arch, tok
