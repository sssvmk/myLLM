"""AT-10 tokenizer/template, AT-11 packing, AT-12 decontamination, AT-24 re-packing."""
import hashlib

import pytest

from conftest import VOCAB_ROWS, ranks_bytes
from src.data.decontam import build_eval_ngrams, is_contaminated
from src.data.packing import pack_lm, pack_sft, repack_packed
from src.tokenization.tokenizer import ChatTokenizer

PAD, EOT = 100277, 100257


def _tok(chat):
    d = ranks_bytes()
    return ChatTokenizer(d, hashlib.sha256(d).hexdigest(), EOT, PAD, chat, VOCAB_ROWS)


def test_special_id_validation():
    for bad in (100257, 100277, 100300, 65):          # existing special, pad, >= vocab_rows, defined byte
        with pytest.raises(ValueError, match="TK-3"):
            _tok({"im_start": {"text": "<|im_start|>", "id": bad}, "im_end": {"text": "<|im_end|>", "id": 100265}})
    with pytest.raises(ValueError, match="sha256"):
        ChatTokenizer(ranks_bytes(), "0" * 64, EOT, PAD, {}, VOCAB_ROWS)


def test_literal_special_text_is_ordinary(tok):
    ids = tok.encode_ordinary("<|im_start|> hi")
    assert 100264 not in ids and len(ids) > 3


def test_loss_mask_exact(tok, template):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"},
            {"role": "assistant", "content": "<think>r</think>yes"}]
    r = template.render(msgs)
    masked = [t for t, m in zip(r.tokens, r.loss_mask) if m]
    expected = (tok.encode_ordinary("ok") + [100265] +
                [100266] + tok.encode_ordinary("r") + [100267] + tok.encode_ordinary("yes") + [100265])
    assert masked == expected
    assert r.tokens.count(100264) == 5 and r.tokens.count(100265) == 5
    nl = tok.encode_ordinary("\n")[0]
    for i, t in enumerate(r.tokens[:-1]):                   # newline after <|im_end|> is never trained
        if t == 100265:
            assert r.tokens[i + 1] == nl and not r.loss_mask[i + 1]


def test_strip_reasoning(template):
    assert template.strip_reasoning("<think>a</think> 42") == " 42"
    assert template.strip_reasoning("<think>a") is None
    assert template.strip_reasoning("plain") == "plain"


def test_lm_packing():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    blocks = list(pack_lm(docs, ctx=4, eot=EOT, pad=PAD, seed=0))
    assert all(len(t) == 5 and len(s) == 5 for t, s in blocks)
    flat = [x for t, _ in blocks for x in t if x != PAD]
    assert sorted(flat) == sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, EOT, EOT, EOT])
    starts = [x for t, s in blocks for x, f in zip(t, s) if f]
    assert sorted(starts) == [1, 4, 6]


def test_sft_packing_no_split(template):
    convs = []
    for i in range(6):
        r = template.render([{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a" * (i + 1)}])
        convs.append((r.tokens, r.loss_mask))
    too_long = ([1] * 200, [True] * 200)
    dropped = []
    blocks = list(pack_sft(convs + [too_long], ctx=40, pad=PAD, seed=1, dropped=dropped))
    assert dropped == [6]
    seen = []
    for toks, starts, mask in blocks:
        assert len(toks) == len(starts) == len(mask) == 41
        idx = [i for i, s in enumerate(starts) if s] + [len([t for t in toks if t != PAD])]
        for a, b in zip(idx, idx[1:]):
            seg = (toks[a:b], mask[a:b])
            assert seg in [(c[0], c[1]) for c in convs]          # whole conversation, aligned mask
            seen.append(seg)
    assert len(seen) == 6


def test_repack_17_to_33_preserves_stream():
    import random
    rng = random.Random(0)
    stream, starts = [], []
    for d in range(10):
        n = rng.randint(3, 12)
        stream += [rng.randint(0, 255) for _ in range(n)]
        starts += [True] + [False] * (n - 1)
    blocks17 = []
    for i in range(0, len(stream), 17):
        t, s = stream[i:i + 17], starts[i:i + 17]
        blocks17.append((t + [PAD] * (17 - len(t)), s + [False] * (17 - len(s))))
    out = list(repack_packed(blocks17, ctx=32, pad=PAD))
    assert all(len(t) == 33 for t, _ in out)
    t_out = [x for t, _ in out for x in t if x != PAD]
    s_out = [f for t, s in out for x, f in zip(t, s) if x != PAD]
    assert t_out == stream and s_out == starts


def test_decontamination():
    ev = build_eval_ngrams(["Natalia sold clips to 48 of her friends in April, and then she sold half as many."], 5)
    assert is_contaminated("QUESTION: Natalia sold clips to 48 of her friends in April, and then she sold half as many!", ev, 5, 0.5)
    assert not is_contaminated("The mitochondria is the powerhouse of the cell and makes energy.", ev, 5, 0.5)
