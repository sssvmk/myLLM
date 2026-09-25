"""Model construction and checkpoint loading (MD-1..MD-4)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..foundation import bridge

_PREFIXES = ("_orig_mod.", "module.", "_fsdp_wrapped_module.")
DEEPSEEK_ONLY = ("d_latent", "d_rope", "n_routed_experts", "n_shared_experts", "moe_top_k")


def arch_dict(architecture) -> Dict[str, Any]:
    return architecture.model_dump() if hasattr(architecture, "model_dump") else dict(architecture)


def build_model(architecture) -> nn.Module:
    bridge.require()
    a = arch_dict(architecture)
    return bridge.fl_model.GPTModel(
        vocab_size=a["vocab_rows"], d_model=a["d_model"], ctx=a["ctx"], n_layers=a["n_layer"],
        d_ff=a["d_ff"], n_heads=a["n_heads"], dropout=0.0, arch=a["arch"], d_latent=a["d_latent"],
        d_rope=a["d_rope"], n_routed_experts=a["n_routed_experts"], n_shared_experts=a["n_shared_experts"],
        moe_top_k=a["moe_top_k"], rope_theta=a["rope_theta"],
    )


def strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for p in _PREFIXES:
                if k.startswith(p):
                    k, changed = k[len(p):], True
            k = k.replace("._fsdp_wrapped_module.", ".").replace("._checkpoint_wrapped_module.", ".")
        out[k] = v
    return out


def is_base_checkpoint(ckpt: Dict[str, Any]) -> bool:
    return {"model", "optimizer", "step", "best_loss", "args"} <= set(ckpt)


def validate_base_args(ckpt_args: Dict[str, Any], architecture) -> None:
    """MD-3. foundation_llm stores d_latent=0 to mean 'derive as max(d_model // 8, 64)'
    (model.GPTModel), so 0 is resolved before comparing. Deepseek-only knobs are compared only
    for arch=deepseek; for dense they are unused CLI defaults."""
    a = arch_dict(architecture)
    keys = ["arch", "ctx", "rope_theta"] + (list(DEEPSEEK_ONLY) if a["arch"] == "deepseek" else [])
    errors = []
    for k in keys:
        if k not in ckpt_args:
            continue
        got = ckpt_args[k]
        if k == "d_latent" and got == 0:
            got = max(a["d_model"] // 8, 64)
        if (float(got) != float(a[k])) if k == "rope_theta" else (got != a[k]):
            errors.append(f"{k}: checkpoint args={got!r} configured={a[k]!r}")
    if errors:
        raise ValueError("base checkpoint does not match base_model.architecture (MD-3):\n" + "\n".join(errors))


def load_state(ckpt_path: str) -> Dict[str, Any]:
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def load_model(ckpt: Dict[str, Any], architecture, device: torch.device,
               validate_base: bool = True) -> nn.Module:
    """MD-1/MD-2: `ckpt` is a loaded checkpoint dict (base or Keystone format)."""
    a = arch_dict(architecture)
    sd = strip_prefixes(ckpt["model"])
    emb = sd.get("tok_emb.weight")
    if emb is None:
        raise ValueError("checkpoint has no tok_emb.weight")
    if emb.shape[0] != a["vocab_rows"]:
        raise ValueError(f"embedding rows: checkpoint={emb.shape[0]} configured vocab_rows={a['vocab_rows']} (MD-3)")
    if validate_base and is_base_checkpoint(ckpt):
        validate_base_args(ckpt["args"], architecture)
    model = build_model(architecture)
    if "lm_head.weight" not in sd:          # exported / tied-only state dicts
        sd["lm_head.weight"] = sd["tok_emb.weight"]
    model.load_state_dict(sd, strict=True)
    model.lm_head.weight = model.tok_emb.weight    # keep the tie after load
    return model.to(device)


def set_dropout(model: nn.Module, p: float) -> None:
    """TR-7."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def moe_modules(model: nn.Module):
    bridge.require()
    return [m for m in model.modules() if isinstance(m, bridge.fl_moe.DeepSeekMoE)]


def configure_moe(model: nn.Module, bias_update_rate: float, aux_loss_weight: float) -> None:
    """PT-9: attribute assignment on the reused module."""
    for m in moe_modules(model):
        m.bias_update_rate = bias_update_rate
        m.aux_loss_weight = aux_loss_weight


def unwrap(model: nn.Module) -> nn.Module:
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            return model
