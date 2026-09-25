"""DAPO overlong shaping (RW-5)."""
from __future__ import annotations


def overlong_penalty(length: int, truncated: bool, max_new_tokens: int, buffer_tokens: int, penalty_factor: float) -> float:
    if truncated:
        return -penalty_factor
    soft_start = max_new_tokens - buffer_tokens
    if length <= soft_start:
        return 0.0
    return (soft_start - length) / buffer_tokens * penalty_factor
