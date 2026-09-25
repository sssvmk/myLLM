"""Math answer extraction and correctness (DS-3a, RW-2)."""
from __future__ import annotations

import re
from typing import Optional

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} with balanced braces (also accepts \\fbox)."""
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx == -1:
        return None
    i = text.index("{", idx) + 1
    depth, start = 1, i
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def last_number(text: str) -> Optional[str]:
    nums = _NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else None


def extract_ground_truth(answer_field: str, mode: str, regex: Optional[str]) -> Optional[str]:
    """DS-3a, applied at preparation time."""
    if mode == "raw":
        return answer_field.strip()
    if mode == "boxed":
        return last_boxed(answer_field)
    m = re.search(regex, answer_field)
    return m.group(1).replace(",", "").strip() if m else None


def extract_answer(stripped_text: str, final_answer_regex: str) -> Optional[str]:
    """RW-2 order: last \\boxed{}; else first match of the configured regex; else last number."""
    b = last_boxed(stripped_text)
    if b is not None:
        return b
    m = re.search(final_answer_regex, stripped_text)
    if m:
        return m.group(1).strip()
    return last_number(stripped_text)


def _wrap(s: str) -> str:
    s = s.strip()
    return s if s.startswith("$") or "\\boxed" in s else f"${s}$"


def math_equal(ground_truth: str, answer: Optional[str]) -> bool:
    if answer is None or ground_truth is None:
        return False
    from math_verify import parse, verify
    try:
        gold = parse(_wrap(ground_truth))
        pred = parse(_wrap(answer))
        return bool(gold) and bool(pred) and bool(verify(gold, pred))
    except Exception:  # noqa: BLE001 -- unparseable answers are simply wrong
        return False
