"""Block packing for LM (DP-5, DP-5a) and SFT (DP-6)."""
from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

Block = Tuple[List[int], List[bool]]                     # tokens, doc_start
SftBlock = Tuple[List[int], List[bool], List[bool]]      # tokens, doc_start, loss_mask


def _cut_stream(stream: Iterable[Tuple[int, bool]], block_len: int, pad: int) -> Iterator[Block]:
    toks: List[int] = []
    starts: List[bool] = []
    for t, s in stream:
        toks.append(t)
        starts.append(s)
        if len(toks) == block_len:
            yield toks, starts
            toks, starts = [], []
    if toks:
        n = block_len - len(toks)
        yield toks + [pad] * n, starts + [False] * n


def pack_lm(documents: Sequence[List[int]], ctx: int, eot: int, pad: int, seed: int) -> Iterator[Block]:
    """DP-5: documents already tokenized with encode_ordinary; EOT appended; shuffled order."""
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)

    def stream():
        for i in order:
            doc = list(documents[i]) + [eot]
            for j, t in enumerate(doc):
                yield t, j == 0
    return _cut_stream(stream(), ctx + 1, pad)


def repack_packed(blocks: Iterable[Tuple[List[int], Optional[List[bool]]]], ctx: int, pad: int) -> Iterator[Block]:
    """DP-5a: drop padding, concatenate in stored order, carry doc_start (first token of the
    stream only, when absent), re-cut to ctx+1. No EOT is inserted."""
    def stream():
        first = True
        for toks, starts in blocks:
            for j, t in enumerate(toks):
                if t == pad:
                    continue
                s = bool(starts[j]) if starts is not None else False
                yield t, (s or first)
                first = False
    return _cut_stream(stream(), ctx + 1, pad)


def pack_sft(conversations: Sequence[Tuple[List[int], List[bool]]], ctx: int, pad: int, seed: int,
             dropped: Optional[list] = None) -> Iterator[SftBlock]:
    """DP-6: conversations never split; longer than ctx+1 are dropped; greedy fill in shuffled order."""
    block_len = ctx + 1
    order = list(range(len(conversations)))
    random.Random(seed).shuffle(order)
    toks: List[int] = []
    starts: List[bool] = []
    mask: List[bool] = []

    def flush():
        n = block_len - len(toks)
        return toks + [pad] * n, starts + [False] * n, mask + [False] * n

    for i in order:
        ct, cm = conversations[i]
        if len(ct) > block_len:
            if dropped is not None:
                dropped.append(i)
            continue
        if len(toks) + len(ct) > block_len:
            yield flush()
            toks, starts, mask = [], [], []
        toks = toks + list(ct)
        starts = starts + [True] + [False] * (len(ct) - 1)
        mask = mask + list(cm)
    if toks:
        yield flush()
