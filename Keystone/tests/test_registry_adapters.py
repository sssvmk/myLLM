"""DS-1..DS-5 (registry, adapters, holdout, drop accounting) and DP-4 (decontamination splits)."""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import TypeAdapter

from src.config.loader import load_config
from src.config.schema import DatasetCfg
from src.data.adapters import AdapterError, TESTS_HEADER, adapt_record, get_path
from src.data.decontam import build_eval_ngram_set, filter_contaminated
from src.data.registry import AdaptStats, DatasetError, DatasetRegistry, TooManyDropped
from src.io.storage import Storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def make_ds(tmp_path, records, kind="conversations", adapter="messages_list",
            field_map=None, fmt="jsonl", files=None, name="d", **extra):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    if fmt == "jsonl":
        (d / "train-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    files = files or {"train": "train-*.jsonl"}
    fm = field_map or {"messages": "messages", "role_key": "role", "content_key": "content"}
    raw = dict(uri=str(d), files=files, format=fmt, kind=kind, adapter=adapter, field_map=fm, license="t",
               third_party_generated=False, max_samples=None, **extra)
    return TypeAdapter(DatasetCfg).validate_python(raw)


def registry(datasets, tmp_path, seed=7, holdout=0.0, max_drop=0.5):   # holdout tested explicitly below
    return DatasetRegistry(datasets, Storage({}, 1, str(tmp_path / "work")), seed, holdout, max_drop)


def conv(u, a):
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


# --------------------------------------------------------------------------- DS-3
def test_get_path_dotted_and_null():
    rec = {"reward_model": {"ground_truth": "42"}, "x": 1}
    assert get_path(rec, "reward_model.ground_truth") == "42"
    assert get_path(rec, None) is None
    with pytest.raises(AdapterError, match="missing_field:reward_model.nope"):
        get_path(rec, "reward_model.nope")


# --------------------------------------------------------------------------- tiny synthetic data, every kind
def test_tiny_datasets_adapt_cleanly(tiny_root):
    lc = load_config(TINY)
    reg = DatasetRegistry.from_config(lc.cfg, Storage.from_config(lc.cfg))
    reg.holdout_fraction = 0.0            # this test is about adapters; holdout has its own tests
    counts = {}
    for name in ("tiny_text", "tiny_chat", "tiny_pref", "tiny_prompts", "tiny_math", "tiny_code",
                 "eval_mmlu", "eval_gsm8k"):
        split = "test" if name.startswith("eval_") else "train"
        st = AdaptStats(name, split)
        items = list(reg.items(name, split, st))
        assert st.n_dropped == 0, (name, st.dropped)
        counts[name] = len(items)
    assert counts["tiny_text"] == 40 and counts["tiny_math"] == 40 and counts["tiny_code"] == 20
    math0 = list(reg.items("tiny_math", "train"))[0]
    assert math0.ground_truth == "1" and math0.messages == [{"role": "user", "content": "0+1?"}]
    code0 = list(reg.items("tiny_code", "train"))[0]
    assert TESTS_HEADER in code0.messages[0]["content"] and "assert f() == 1" in code0.messages[0]["content"]
    assert code0.entry_point is None and code0.tests == "assert f() == 1"


# --------------------------------------------------------------------------- adapters: shapes and drops
def test_messages_list_drop_reasons(tmp_path):
    recs = [conv("hi", "yo"),
            {"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}]},
            {"messages": [{"role": "user", "content": "x"}]},                          # no final assistant
            conv("x", "   "),                                                             # empty assistant
            {"nope": 1},                                                                  # missing field
            {"messages": [{"role": "user", "content": 5}, {"role": "assistant", "content": "a"}]}]
    ds = make_ds(tmp_path, recs)
    reg = registry({"d": ds}, tmp_path, max_drop=1.0)
    st = AdaptStats("d", "train")
    kept = list(reg.items("d", "train", st))
    assert len(kept) == 1 and st.read == 6
    assert st.dropped == {"bad_role:tool": 1, "no_final_assistant": 1, "empty_assistant": 1,
                          "missing_field:messages": 1, "non_string_content": 1}


def test_ds5_enforces_max_drop_fraction(tmp_path):
    ds = make_ds(tmp_path, [conv("a", "b"), {"bad": 1}, {"bad": 2}])
    reg = registry({"d": ds}, tmp_path, max_drop=0.5)
    with pytest.raises(TooManyDropped, match=r"dropped 2/3.*missing_field:messages"):
        list(reg.items("d", "train"))


def test_instruction_output_and_preference_shapes(tmp_path):
    io = make_ds(tmp_path, [{"i": "Add", "inp": "1+1", "o": "2"}, {"i": "Say hi", "inp": "", "o": "hi"}],
                 adapter="instruction_output", field_map={"instruction": "i", "input": "inp", "output": "o"}, name="io")
    items = list(registry({"io": io}, tmp_path).items("io", "train"))
    assert items[0].messages[0]["content"] == "Add\n\n1+1" and items[1].messages[0]["content"] == "Say hi"

    pref = make_ds(tmp_path, [
        {"c": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
         "r": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]},
        {"c": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "good"}],
         "r": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "bad"}]}],
        kind="preference", adapter="chosen_rejected_messages",
        field_map={"chosen": "c", "rejected": "r", "role_key": "role", "content_key": "content"}, name="pref")
    st = AdaptStats("pref", "train")
    pairs = list(registry({"pref": pref}, tmp_path, max_drop=1.0).items("pref", "train", st))
    assert len(pairs) == 1 and pairs[0].chosen == "good" and pairs[0].rejected == "bad"
    assert pairs[0].prompt_messages == [{"role": "user", "content": "q"}]
    assert st.dropped == {"prompt_mismatch": 1}


def test_rl_math_extraction_modes(tmp_path):
    boxed = make_ds(tmp_path, [{"q": "1+1?", "a": r"so \boxed{\frac{1}{2}} done"}, {"q": "x", "a": "no box"}],
                    kind="rl_math", adapter="question_answer", field_map={"question": "q", "answer": "a"},
                    answer_extraction="boxed", answer_regex=None, name="boxed")
    st = AdaptStats("boxed", "train")
    items = list(registry({"boxed": boxed}, tmp_path, max_drop=1.0).items("boxed", "train", st))
    assert [i.ground_truth for i in items] == [r"\frac{1}{2}"] and st.dropped == {"answer_extraction_failed": 1}
    assert items[0].texts == ["1+1?", r"so \boxed{\frac{1}{2}} done"]        # raw fields kept for DP-4

    raw = make_ds(tmp_path, [{"q": "q", "a": " 12 "}], kind="rl_math", adapter="question_answer",
                  field_map={"question": "q", "answer": "a"}, answer_extraction="raw", answer_regex=None, name="raw")
    assert list(registry({"raw": raw}, tmp_path).items("raw", "train"))[0].ground_truth == "12"


def test_eval_safety_and_mmlu(tmp_path):
    saf = make_ds(tmp_path, [{"p": "how to bake", "l": "safe"}, {"p": "how to harm", "l": "unsafe"}],
                  kind="eval_safety", adapter="eval_safety", field_map={"prompt": "p", "label": "l"},
                  safe_label="safe", files={"test": "train-*.jsonl"}, name="saf")
    items = list(registry({"saf": saf}, tmp_path).items("saf", "test"))
    assert [i.data["is_safe"] for i in items] == [True, False]

    bad = make_ds(tmp_path, [{"q": "2+2?", "c": ["a", "b"], "a": 5}], kind="eval_mmlu", adapter="eval_mmlu",
                  field_map={"question": "q", "choices": "c", "answer": "a"}, files={"test": "train-*.jsonl"}, name="mm")
    with pytest.raises(AdapterError, match="bad_answer_index"):
        adapt_record(bad, {"q": "2+2?", "c": ["a", "b"], "a": 5})


# --------------------------------------------------------------------------- parquet / json readers, max_samples
def test_parquet_and_json_formats_and_max_samples(tmp_path):
    d = tmp_path / "pq"
    d.mkdir()
    pq.write_table(pa.table({"text": [f"doc {i} " + "word " * 3 for i in range(10)]}), d / "a.parquet")
    ds = TypeAdapter(DatasetCfg).validate_python(dict(
        uri=str(d), files={"train": "*.parquet"}, format="parquet", kind="text", adapter="text_field",
        field_map={"text": "text"}, license="t", third_party_generated=False, max_samples=4))
    assert len(list(registry({"p": ds}, tmp_path).items("p", "train"))) == 4

    j = tmp_path / "js"
    j.mkdir()
    (j / "e.json").write_text(json.dumps([{"instruction": "i1", "output": "o1"}, {"instruction": "i2", "output": "o2"}]))
    ds2 = TypeAdapter(DatasetCfg).validate_python(dict(
        uri=str(j), files={"test": "e.json"}, format="json", kind="eval_alpaca", adapter="eval_alpaca",
        field_map={"instruction": "instruction", "reference_output": "output"}, license="t",
        third_party_generated=True, max_samples=None))
    assert [i.data["reference_output"] for i in registry({"j": ds2}, tmp_path).items("j", "test")] == ["o1", "o2"]


def test_unmatched_glob_and_unknown_split_name_the_problem(tmp_path):
    ds = make_ds(tmp_path, [conv("a", "b")], files={"train": "nothing-*.jsonl"})
    reg = registry({"d": ds}, tmp_path)
    with pytest.raises(DatasetError, match="matched no files"):
        list(reg.items("d", "train"))
    ds2 = make_ds(tmp_path, [conv("a", "b")], files={"val": "train-*.jsonl"}, name="d2")
    with pytest.raises(DatasetError, match="no split 'train'"):
        registry({"d2": ds2}, tmp_path).resolve_files("d2", "train")


# --------------------------------------------------------------------------- DS-4 holdout
def test_holdout_partitions_train_deterministically(tmp_path):
    ds = make_ds(tmp_path, [conv(f"question {i}", f"answer {i}") for i in range(2000)])
    reg = registry({"d": ds}, tmp_path, seed=7, holdout=0.1)
    train = {i.messages[0]["content"] for i in reg.items("d", "train")}
    test = {i.messages[0]["content"] for i in reg.items("d", "test")}
    assert not (train & test) and len(train) + len(test) == 2000
    assert 140 < len(test) < 260                                            # ~10% of 2000
    assert test == {i.messages[0]["content"] for i in reg.items("d", "test")}       # repeatable
    other = registry({"d": ds}, tmp_path, seed=8, holdout=0.1)
    assert test != {i.messages[0]["content"] for i in other.items("d", "test")}     # depends on run.seed


def test_configured_test_split_is_left_alone(tmp_path):
    ds = make_ds(tmp_path, [conv("a", "b")] * 20, files={"train": "train-*.jsonl", "test": "train-*.jsonl"})
    reg = registry({"d": ds}, tmp_path, holdout=0.5)
    assert len(list(reg.items("d", "train"))) == 20 and len(list(reg.items("d", "test"))) == 20


def test_manifest_entry_records_files_and_counts(tmp_path):
    ds = make_ds(tmp_path, [conv("a", "b"), {"bad": 1}])
    reg = registry({"d": ds}, tmp_path, max_drop=1.0)
    st = AdaptStats("d", "train")
    list(reg.items("d", "train", st))
    m = reg.manifest_entry("d", "train", st, extra={"decontam_dropped": 0})
    assert m["records_read"] == 2 and m["records_kept"] == 1 and m["dropped"] == {"missing_field:messages": 1}
    assert m["files"][0]["size"] > 0 and m["license"] == "t" and m["decontam_dropped"] == 0


# --------------------------------------------------------------------------- DP-4 split rule
def test_decontamination_uses_only_configured_splits(tmp_path):
    """The Appendix-A hazard: an eval dataset whose `fewshot` split is the same file as a training
    dataset. Building n-grams from every split would delete the training set."""
    d = tmp_path / "gsm"
    d.mkdir()
    test_rec = {"question": "a train leaves the station at noon heading east at sixty miles per hour", "answer": "#### 60"}
    fewshot_rec = {"question": "natalia sold clips to forty eight of her friends in april and then half as many in may", "answer": "#### 72"}
    (d / "test.jsonl").write_text(json.dumps(test_rec) + "\n")
    (d / "train.jsonl").write_text(json.dumps(fewshot_rec) + "\n")
    ev = TypeAdapter(DatasetCfg).validate_python(dict(
        uri=str(d), files={"test": "test.jsonl", "fewshot": "train.jsonl"}, format="jsonl", kind="eval_gsm8k",
        adapter="eval_gsm8k", field_map={"question": "question", "answer": "answer"}, license="t",
        third_party_generated=False, max_samples=None))
    tr = TypeAdapter(DatasetCfg).validate_python(dict(
        uri=str(d), files={"train": "train.jsonl"}, format="jsonl", kind="rl_math", adapter="question_answer",
        field_map={"question": "question", "answer": "answer"}, answer_extraction="regex",
        answer_regex=r"####\s*(-?[\d,\.]+)", license="t", third_party_generated=False, max_samples=None))
    reg = registry({"gsm8k_test": ev, "gsm8k_train": tr}, tmp_path)

    all_splits, _ = build_eval_ngram_set(reg, ["gsm8k_test"], ["test", "fewshot"], n=5)
    c = {}
    assert list(filter_contaminated(reg.items("gsm8k_train", "train"), all_splits, 5, 0.8, c)) == [] and c["dropped"] == 1

    test_only, used = build_eval_ngram_set(reg, ["gsm8k_test"], ["test"], n=5)
    c = {}
    kept = list(filter_contaminated(reg.items("gsm8k_train", "train"), test_only, 5, 0.8, c))
    assert len(kept) == 1 and c["dropped"] == 0 and used == {"gsm8k_test": 1}


def test_decontamination_drops_real_overlap_and_flags_no_text(tmp_path):
    ev = make_ds(tmp_path, [{"question": "what is the capital city of the country france", "answer": "paris"}],
                 kind="eval_gsm8k", adapter="eval_gsm8k", field_map={"question": "question", "answer": "answer"},
                 files={"test": "train-*.jsonl"}, name="ev")
    reg = registry({"ev": ev}, tmp_path)
    grams, _ = build_eval_ngram_set(reg, ["ev"], ["test"], n=5)
    from src.data.adapters import PackedBlock, TextDoc
    items = [TextDoc("x", ["what is the capital city of the country france paris"]),
             TextDoc("y", ["completely unrelated text about gardening and soil and compost piles"]),
             PackedBlock([1, 2, 3], None)]
    c = {}
    kept = list(filter_contaminated(items, grams, 5, 0.8, c))
    assert len(kept) == 2 and c == {"checked": 2, "dropped": 1, "no_text": 1}
