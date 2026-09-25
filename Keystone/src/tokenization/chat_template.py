"""Chat template rendering with an assistant-only loss mask (TK-5..TK-8)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .tokenizer import ChatTokenizer


@dataclass
class Rendered:
    tokens: List[int]
    loss_mask: List[bool]


class ChatTemplate:
    def __init__(self, tok: ChatTokenizer, turn_start: str, turn_end: str, generation_prompt: str,
                 reasoning_open: str, reasoning_close: str):
        self.tok = tok
        self.turn_start = turn_start
        self.turn_end = turn_end
        self.generation_prompt = generation_prompt
        self.reasoning_open = reasoning_open
        self.reasoning_close = reasoning_close
        self._special_texts = set(tok.special_text_to_id)
        self._reasoning_specials = {s for s in (reasoning_open, reasoning_close) if s in self._special_texts}

    @classmethod
    def from_config(cls, cfg, tok: ChatTokenizer) -> "ChatTemplate":
        c = cfg.chat_template
        return cls(tok, c.turn_start, c.turn_end, c.generation_prompt, c.reasoning_open, c.reasoning_close)

    def _piece(self, s: str) -> List[int]:
        """Configuration string: special-token substrings -> ids, rest -> ordinary text (TK-5)."""
        return self.tok.encode_with_specials(s, allowed=self._special_texts)

    def _assistant_content(self, content: str) -> List[int]:
        """Assistant content: reasoning markers that are configured special tokens are emitted as
        their ids (renderer-inserted, TK-4); every other character is ordinary text.
        See docs/implementation_notes.md (TK-4 vs TK-8)."""
        if not self._reasoning_specials:
            return self.tok.encode_ordinary(content)
        pattern = "(" + "|".join(re.escape(s) for s in sorted(self._reasoning_specials, key=len, reverse=True)) + ")"
        out: List[int] = []
        for part in re.split(pattern, content):
            if not part:
                continue
            if part in self._reasoning_specials:
                out.append(self.tok.special_text_to_id[part])
            else:
                out.extend(self.tok.encode_ordinary(part))
        return out

    def render(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> Rendered:
        tokens: List[int] = []
        mask: List[bool] = []
        im_end = self.tok.im_end_id
        for m in messages:
            role, content = m["role"], m["content"]
            start = self._piece(self.turn_start.format(role=role))
            tokens += start
            mask += [False] * len(start)
            is_asst = role == "assistant"
            body = self._assistant_content(content) if is_asst else self.tok.encode_ordinary(content)
            tokens += body
            mask += [is_asst] * len(body)
            end = self._piece(self.turn_end)
            tokens += end
            mask += [is_asst and t == im_end for t in end]   # TK-6: only <|im_end|> of assistant turns
        if add_generation_prompt:
            gp = self._piece(self.generation_prompt)
            tokens += gp
            mask += [False] * len(gp)
        return Rendered(tokens, mask)

    def render_prompt(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.render(messages, add_generation_prompt=True).tokens

    def render_response(self, content: str) -> List[int]:
        """Assistant response tokens as trained in preference/RL: content + <|im_end|> (DP-7)."""
        return self._assistant_content(content) + [self.tok.im_end_id]

    def strip_reasoning(self, text: str) -> Optional[str]:
        """TK-8: drop everything up to and including the last reasoning_close. An unclosed
        reasoning_open means there is no answer (returns None)."""
        last_open = text.rfind(self.reasoning_open)
        last_close = text.rfind(self.reasoning_close)
        if last_open != -1 and last_open > last_close:
            return None
        if last_close == -1:
            return text
        return text[last_close + len(self.reasoning_close):]
