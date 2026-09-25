"""Total reward (RW-1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .code import build_program, last_code_block
from .format import format_ok
from .math import extract_answer, math_equal
from .overlong import overlong_penalty


@dataclass
class RewardBreakdown:
    reward: float
    correct: int
    format_ok: int
    overlong: float
    excluded: bool           # RW-5 exclude mode: drop from the loss


def correctness(item: dict, text: str, template, final_answer_regex: str, sandbox=None) -> int:
    """RW-2 / RW-4 correctness only (also used to filter teacher traces, DI-2)."""
    stripped = template.strip_reasoning(text)
    if stripped is None:
        return 0
    if item["kind"] == "rl_math":
        return int(math_equal(item["ground_truth"], extract_answer(stripped, final_answer_regex)))
    code = last_code_block(stripped)
    if code is None:
        return 0
    if sandbox is None:
        raise ValueError("rl_code rewards require a sandbox")
    return int(sandbox.run(build_program(code, item["tests"], item.get("entry_point"))).passed)


def correctness(item: dict, text: str, template, final_answer_regex: str, sandbox=None) -> int:
    """RW-2 / RW-4 correctness of one response text (reasoning stripped first, TK-8)."""
    stripped = template.strip_reasoning(text)
    if stripped is None:
        return 0
    if item["kind"] == "rl_math":
        return int(math_equal(item["ground_truth"], extract_answer(stripped, final_answer_regex)))
    code = last_code_block(stripped)
    if code is None:
        return 0
    if sandbox is None:
        raise ValueError("rl_code rewards require a sandbox")
    return int(sandbox.run(build_program(code, item["tests"], item.get("entry_point"))).passed)


def score_response(item: dict, text: str, finish_reason: str, n_tokens: int, template, rewards_cfg,
                   dapo_overlong, max_new_tokens: int, sandbox=None) -> RewardBreakdown:
    """`item` is a prepared RL record: {"kind": "rl_math"|"rl_code", "ground_truth"|("tests","entry_point")}."""
    stripped = template.strip_reasoning(text)
    correct = 0
    if stripped is not None:
        if item["kind"] == "rl_math":
            correct = int(math_equal(item["ground_truth"], extract_answer(stripped, rewards_cfg.math.final_answer_regex)))
        else:
            code = last_code_block(stripped)
            if code is not None:
                if sandbox is None:
                    raise ValueError("rl_code rewards require a sandbox")
                correct = int(sandbox.run(build_program(code, item["tests"], item.get("entry_point"))).passed)
    fmt = int(format_ok(text, finish_reason, template.reasoning_open, template.reasoning_close))
    truncated = finish_reason == "length"
    pen, excluded = 0.0, False
    if dapo_overlong.mode == "soft_penalty":
        pen = overlong_penalty(n_tokens, truncated, max_new_tokens, dapo_overlong.buffer_tokens, dapo_overlong.penalty_factor)
    else:
        excluded = truncated
    r = rewards_cfg.correctness_weight * correct + rewards_cfg.format_weight * fmt + pen
    return RewardBreakdown(r, correct, fmt, pen, excluded)
