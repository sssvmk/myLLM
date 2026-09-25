"""JG-1..JG-3, TC-1: judge parsing, retries, caching; teacher sampling."""
import os

import pytest

from src.config.loader import load_config
from src.io.storage import Storage
from src.services.judge import Judge, last_match
from src.services.teacher import Teacher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def make(tiny_root, stub, extra=()):
    lc = load_config(TINY, stub.overrides() + [f"prepared_data.root_uri=file://{tiny_root}/prep_{id(stub)}", *extra])
    st = Storage.from_config(lc.cfg)
    return lc.cfg, st, Judge.from_config(lc.cfg, st)


def test_last_match():
    assert last_match(r"VERDICT:\s*(A|B|TIE)", "VERDICT: A is what I would say\nVERDICT: B") == "B"
    assert last_match(r"x", "y") is None


def test_pairwise_refusal_score_parse(tiny_root, stub):
    cfg, st, j = make(tiny_root, stub)
    assert j.pairwise("i", "a", "b") in ("A", "B", "TIE")
    assert j.refusal("p", "r") in ("compliance", "refusal", "partial_refusal")
    s = j.score("i", "r")
    assert isinstance(s, int) and 1 <= s <= 10
    # the pairwise prompt the stub received is the rendered template, not the raw one
    model, msg = stub.requests[0]
    assert model == "stub-judge" and "[Response A]\na" in msg


def test_cache_hit_makes_no_request(tiny_root, stub):
    cfg, st, j = make(tiny_root, stub)
    first = j.score("instr", "resp")
    n = len(stub.requests)
    assert j.score("instr", "resp") == first and len(stub.requests) == n                 # JG-3
    _, _, j2 = make(tiny_root, stub)                                                       # a new Judge shares the cache dir
    assert j2.score("instr", "resp") == first and len(stub.requests) == n
    assert j2.score("instr", "other") is not None and len(stub.requests) == n + 1


def test_unparseable_retried_then_defaults(tiny_root, stub):
    cfg, st, j = make(tiny_root, stub)                                     # max_retries = 1 -> 2 attempts
    stub.garbled_next = 2
    assert j.pairwise("i", "x", "y") == "TIE"                              # JG-2: ties
    assert len(stub.requests) == 2
    stub.garbled_next = 2
    assert j.refusal("p", "z") == "partial_refusal"
    stub.garbled_next = 2
    assert j.score("i", "w") is None                                       # dropped
    stub.garbled_next = 1
    assert j.pairwise("i", "x2", "y2") in ("A", "B", "TIE") and stub.garbled_next == 0     # second attempt parses
    assert not any("cannot tell" in open(os.path.join(st.local_path(j.cache_uri), f)).read()
                   for f in os.listdir(st.local_path(j.cache_uri)))         # unparseable answers are never cached


def test_transport_errors_count_as_attempts(tiny_root, stub):
    cfg, st, j = make(tiny_root, stub)
    stub.fail_next = 1
    assert j.score("i", "e") is not None                                   # first attempt 500, second succeeds
    stub.fail_next = 5
    assert j.score("i", "e2") is None


def test_concurrent_map(tiny_root, stub):
    cfg, st, j = make(tiny_root, stub)
    out = j.map(lambda i: j.score("i", f"resp{i}"), range(6))
    assert len(out) == 6 and all(1 <= s <= 10 for s in out)


def test_missing_api_key_is_a_clear_error(tiny_root, stub, monkeypatch):
    monkeypatch.delenv("PF_TINY_JUDGE_KEY")
    from src.services.judge import ServiceError
    with pytest.raises(ServiceError, match="PF_TINY_JUDGE_KEY"):
        make(tiny_root, stub)


def test_teacher_samples(tiny_root, stub):
    lc = load_config(TINY, stub.overrides())
    t = Teacher.from_config(lc.cfg)
    out = t.complete([[{"role": "user", "content": "3+1?"}], [{"role": "user", "content": "hello"}]], 2)
    assert [len(o) for o in out] == [2, 2]
    assert out[0][0] == "<think>add one</think>\\boxed{4}" and out[1][0] == "Sure thing."
    assert t.identity()["model"] == "stub-teacher" and "key" not in str(t.identity()).lower()
