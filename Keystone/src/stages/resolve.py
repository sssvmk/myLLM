"""Where a stage starts from and whether its output is still valid (PL-2, PL-3, PL-5, PL-6).

Status files under each stage's `output_uri`:
  `_COMPLETE`           training finished; holds the configuration hash and output sha256
  `_SKIPPED`            the stage decided not to run (RL-1 entry gate); holds the reason
  `final/gate_result.json`   written by `pf gate`; `passed: false` makes the output ineligible for
                        `init_from: previous` and as the final model (GT-3)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config.loader import sha256_json
from ..config.schema import STAGE_IDS, PFConfig
from ..io.storage import Storage
from ..training.lineage import read_complete, read_lineage, sha256_uri


@dataclass
class ParentRef:
    kind: str                       # base | stage | uri
    uri: str                        # checkpoint file to load
    sha256: str
    stage: Optional[str]            # stage id when the parent is a stage output
    architecture: Dict[str, Any]
    lineage: Optional[Dict[str, Any]] = None


def out_uri(cfg: PFConfig, stage_id: str) -> str:
    return cfg.stage_block(stage_id).output_uri


def final_model_uri(cfg: PFConfig, stage_id: str) -> str:
    return Storage.join(out_uri(cfg, stage_id), "final", "model.pt")


def is_skipped(storage: Storage, cfg: PFConfig, stage_id: str) -> Optional[Dict[str, Any]]:
    u = Storage.join(out_uri(cfg, stage_id), "_SKIPPED")
    return json.loads(storage.read_text(u)) if storage.exists(u) else None


def gate_passed(storage: Storage, cfg: PFConfig, stage_id: str) -> Optional[bool]:
    u = Storage.join(out_uri(cfg, stage_id), "final", "gate_result.json")
    return bool(json.loads(storage.read_text(u))["passed"]) if storage.exists(u) else None


def eligible(storage: Storage, cfg: PFConfig, stage_id: str) -> bool:
    """A stage output that can be a parent or the final model: enabled, complete, not skipped, and
    its gate (when one has been recorded) did not fail."""
    return (cfg.stage_enabled(stage_id) and read_complete(storage, out_uri(cfg, stage_id)) is not None
            and is_skipped(storage, cfg, stage_id) is None and gate_passed(storage, cfg, stage_id) is not False)


def previous_stage(storage: Storage, cfg: PFConfig, stage_id: str) -> Optional[str]:
    """PL-2 `previous`: nearest earlier stage that completed and did not fail its gate."""
    for sid in reversed(STAGE_IDS[:STAGE_IDS.index(stage_id)]):
        if eligible(storage, cfg, sid):
            return sid
    return None


def final_stage(storage: Storage, cfg: PFConfig) -> Optional[str]:
    """PL-6: output of the last stage that completed and passed its gate."""
    for sid in reversed(STAGE_IDS):
        if eligible(storage, cfg, sid):
            return sid
    return None


def planned_architecture(cfg: PFConfig, stage_id: str) -> Dict[str, Any]:
    """Architecture a stage will have according to the configuration alone (used before any parent
    exists, e.g. by `pf prepare-data --stage all`): base architecture, with Stage 1b's override
    applied to every stage from 1b on when 1b is enabled."""
    a = cfg.base_model.architecture.model_dump()
    if cfg.stage_enabled("midtrain_long") and STAGE_IDS.index(stage_id) >= STAGE_IDS.index("midtrain_long"):
        o = cfg.stages.midtrain_long.architecture_override
        a.update(ctx=o.ctx, rope_theta=o.rope_theta)
    return a


def resolve_parent(storage: Storage, cfg: PFConfig, stage_id: str) -> ParentRef:
    init = cfg.stage_block(stage_id).init_from
    if init == "previous":
        prev = previous_stage(storage, cfg, stage_id)
        init = prev if prev is not None else "base"
    if init == "base":
        uri = cfg.base_model.checkpoint_uri
        return ParentRef("base", uri, sha256_uri(storage, uri), None, cfg.base_model.architecture.model_dump())
    if init in STAGE_IDS:
        if not eligible(storage, cfg, init):
            raise FileNotFoundError(f"stages.{stage_id}.init_from: stage {init!r} has no complete, gate-passing output "
                                    f"at {out_uri(cfg, init)} (PL-2)")
        uri = final_model_uri(cfg, init)
        lin = read_lineage(storage, out_uri(cfg, init))
        return ParentRef("stage", uri, lin["output_sha256"], init, dict(lin["architecture"]), lin)
    # explicit checkpoint URI: a sibling lineage.json (final/ of a stage output) describes it
    sibling = init.rsplit("/", 1)[0] + "/lineage.json"
    if storage.exists(sibling):
        lin = json.loads(storage.read_text(sibling))
        return ParentRef("uri", init, lin.get("output_sha256") or sha256_uri(storage, init), lin.get("stage"),
                         dict(lin["architecture"]), lin)
    return ParentRef("uri", init, sha256_uri(storage, init), None, cfg.base_model.architecture.model_dump())


def stage_architecture(cfg: PFConfig, stage_id: str, parent: ParentRef) -> Dict[str, Any]:
    """PL-3: the parent's architecture, except Stage 1b which overrides ctx and rope_theta."""
    a = dict(parent.architecture)
    if stage_id == "midtrain_long":
        o = cfg.stages.midtrain_long.architecture_override
        a.update(ctx=o.ctx, rope_theta=o.rope_theta)
    return a


def referenced_datasets(cfg: PFConfig, stage_id: str) -> List[str]:
    block = cfg.stage_block(stage_id)
    names = {s.dataset for s in block.data.sources}
    if hasattr(block.data, "retention_eval_dataset"):
        names.add(block.data.retention_eval_dataset)
    if stage_id == "preference" and block.on_policy.enabled:
        names.add(block.on_policy.prompts_dataset)
    if stage_id == "distill_offpolicy":
        names.update(block.prompts_datasets)
    d = cfg.prepared_data.decontamination
    names.update(d.against)
    return sorted(n for n in names if n in cfg.datasets)


def stage_config_hash(cfg: PFConfig, stage_id: str, architecture: Dict[str, Any], parent_sha256: str) -> str:
    """PL-5: sha256 of canonical JSON of the stage block, the datasets it references, tokenizer,
    chat_template, resolved architecture, run.seed, run.precision and the parent's sha256."""
    return sha256_json({
        "stage": stage_id,
        "block": cfg.stage_block(stage_id).model_dump(),
        "datasets": {n: cfg.datasets[n].model_dump() for n in referenced_datasets(cfg, stage_id)},
        "tokenizer": cfg.tokenizer.model_dump(),
        "chat_template": cfg.chat_template.model_dump(),
        "architecture": architecture,
        "seed": cfg.run.seed,
        "precision": cfg.run.precision,
        "parent_sha256": parent_sha256,
    })


def stage_up_to_date(storage: Storage, cfg: PFConfig, stage_id: str, config_hash: str) -> bool:
    c = read_complete(storage, out_uri(cfg, stage_id))
    return c is not None and c["config_hash"] == config_hash
