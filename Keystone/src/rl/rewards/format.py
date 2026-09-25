"""Format reward (RW-3)."""
from __future__ import annotations


def format_ok(text: str, finish_reason: str, reasoning_open: str, reasoning_close: str) -> bool:
    if finish_reason != "eos":
        return False
    n_open, n_close = text.count(reasoning_open), text.count(reasoning_close)
    if n_open == 0 and n_close == 0:
        return True
    return n_open == 1 and n_close == 1 and text.index(reasoning_open) < text.index(reasoning_close)
