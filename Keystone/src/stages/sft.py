"""Stage 2 (sft) and Stage 5a (distill_offpolicy): assistant-only cross-entropy (S2-1, DI-3).

The best checkpoint by held-out SFT loss becomes final/model.pt (S2-3).
"""
from __future__ import annotations

from ..data import prepare
from ..training.loop import TrainResult
from .common import BlockObjective, StageContext, run_training

STAGES = ("sft", "distill_offpolicy")


def prepare_stage(sctx: StageContext, pctx: prepare.PrepContext):
    return prepare.prepare_stage_datasets(pctx, sctx.stage_id, sctx.ctx_len)


def build_objective(sctx: StageContext) -> BlockObjective:
    return BlockObjective(sctx, "sft")


def train(sctx: StageContext) -> TrainResult:
    return run_training(sctx, build_objective(sctx))
