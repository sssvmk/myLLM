"""Stage 1 (midtrain) and Stage 1b (midtrain_long): LM objective on a weighted mix (S1-1..S1b-1).

Stage 1b differs only in its architecture (ctx, rope_theta from `architecture_override`, applied
by resolve.stage_architecture) and in that its data is prepared at the new ctx.
"""
from __future__ import annotations

from ..data import prepare
from ..training.loop import TrainResult
from .common import BlockObjective, StageContext, run_training

STAGES = ("midtrain", "midtrain_long")


def prepare_stage(sctx: StageContext, pctx: prepare.PrepContext):
    return prepare.prepare_stage_datasets(pctx, sctx.stage_id, sctx.ctx_len)


def build_objective(sctx: StageContext) -> BlockObjective:
    return BlockObjective(sctx, "lm")


def train(sctx: StageContext) -> TrainResult:
    return run_training(sctx, build_objective(sctx))
