"""cl100k_base encoding built from a ranks file plus configured chat special tokens (TK-1..TK-4)."""
from __future__ import annotations

import base64
import hashlib
from typing import Dict, List, Set

import tiktoken

from ..foundation import bridge


def parse_ranks(data: bytes) -> Dict[bytes, int]:
    ranks = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


class ChatTokenizer:
    def __init__(self, ranks_bytes: bytes, expected_sha256: str, eot_token_id: int, pad_token_id: int,
                 chat_special_tokens: Dict[str, dict], vocab_rows: int):
        bridge.require()
        actual = hashlib.sha256(ranks_bytes).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"tokenizer ranks sha256 {actual} != configured {expected_sha256}")
        self.ranks_sha256 = actual
        ranks = parse_ranks(ranks_bytes)
        base_specials = dict(bridge.fl_tokenizer._SPECIAL_TOKENS)
        self.base = tiktoken.Encoding(name="pf_base", pat_str=bridge.fl_tokenizer._PAT_STR,
                                      mergeable_ranks=ranks, special_tokens=base_specials)
        self.chat_tokens = {k: (v["text"], int(v["id"])) for k, v in chat_special_tokens.items()}
        self.eot_token_id = eot_token_id
        self.pad_token_id = pad_token_id
        self.vocab_rows = vocab_rows
        self.validate_special_ids()
        specials = dict(base_specials)
        specials.update({text: tid for text, tid in self.chat_tokens.values()})
        self.enc = tiktoken.Encoding(name="pf_chat", pat_str=bridge.fl_tokenizer._PAT_STR,
                                     mergeable_ranks=ranks, special_tokens=specials)
        self.special_text_to_id = specials
        self.defined_ids: Set[int] = set(ranks.values()) | set(specials.values())

    @classmethod
    def from_config(cls, cfg, storage) -> "ChatTokenizer":
        t = cfg.tokenizer
        return cls(storage.read_bytes(t.ranks_file_uri), t.ranks_sha256, t.eot_token_id, t.pad_token_id,
                   {k: v.model_dump() for k, v in t.chat_special_tokens.items()},
                   cfg.base_model.architecture.vocab_rows)

    def validate_special_ids(self) -> None:
        """TK-3: each chat id below vocab_rows, != pad, not an existing special, undefined in base."""
        errors = []
        base_special_ids = set(self.base._special_tokens.values())
        seen = {}
        if self.eot_token_id not in base_special_ids:
            errors.append(f"eot_token_id {self.eot_token_id} is not a base special token id")
        if not (self.base.n_vocab <= self.pad_token_id < self.vocab_rows):
            errors.append(f"pad_token_id {self.pad_token_id} must be >= base n_vocab {self.base.n_vocab} and < vocab_rows {self.vocab_rows}")
        for name, (text, tid) in self.chat_tokens.items():
            if tid in seen:
                errors.append(f"chat token {name} id {tid} duplicates {seen[tid]}")
            seen[tid] = name
            if not (0 <= tid < self.vocab_rows):
                errors.append(f"chat token {name} id {tid} not below vocab_rows {self.vocab_rows}")
            if tid == self.pad_token_id:
                errors.append(f"chat token {name} id {tid} equals pad_token_id")
            if tid in base_special_ids:
                errors.append(f"chat token {name} id {tid} is an existing special-token id")
            try:
                self.base.decode_single_token_bytes(tid)
                errors.append(f"chat token {name} id {tid} is already defined in the base encoding")
            except KeyError:
                pass
        if errors:
            raise ValueError("TK-3 special-token validation failed:\n" + "\n".join(errors))

    # --------------------------------------------------------------- encode / decode
    def encode_ordinary(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def encode_with_specials(self, text: str, allowed: Set[str]) -> List[int]:
        """Only for configuration strings (template pieces): the listed special texts map to ids."""
        return self.enc.encode(text, allowed_special=allowed, disallowed_special=())

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode([i for i in ids if i in self.defined_ids])

    def id_of(self, name: str) -> int:
        return self.chat_tokens[name][1]

    @property
    def im_end_id(self) -> int:
        return self.id_of("im_end")

    def allowed_mask(self, device=None):
        """Bool (vocab_rows,) -- True where sampling is allowed (GN-3 step 1)."""
        import torch
        mask = torch.zeros(self.vocab_rows, dtype=torch.bool, device=device)
        ids = torch.tensor(sorted(i for i in self.defined_ids if i < self.vocab_rows), dtype=torch.long, device=device)
        mask[ids] = True
        mask[self.pad_token_id] = False
        return mask
