"""Code extraction and program assembly (RW-4)."""
from __future__ import annotations

import re
from typing import List, Optional, Union

_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def last_code_block(text: str) -> Optional[str]:
    blocks = _FENCE.findall(text)
    return blocks[-1] if blocks else None


def build_program(code: str, tests: Union[str, List[str]], entry_point: Optional[str]) -> str:
    test_src = tests if isinstance(tests, str) else "\n".join(tests)
    prog = code + "\n" + test_src + "\n"
    if entry_point and re.search(r"^def check\(", test_src, re.MULTILINE):
        prog += f"check({entry_point})\n"
    return prog
