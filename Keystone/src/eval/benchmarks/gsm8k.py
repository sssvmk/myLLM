"""GSM8K answer parsing (B-2): first match of answer_regex, else last number; numeric compare."""
from __future__ import annotations

import re
from typing import Optional

from ...rl.rewards.math import last_number


def parse_gsm8k(text: str, answer_regex: str) -> Optional[float]:
    m = re.search(answer_regex, text)
    s = m.group(1) if m else last_number(text)
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_correct(pred_text: str, gold_answer_field: str, answer_regex: str) -> bool:
    p, g = parse_gsm8k(pred_text, answer_regex), parse_gsm8k(gold_answer_field, answer_regex)
    return p is not None and g is not None and abs(p - g) < 1e-6
