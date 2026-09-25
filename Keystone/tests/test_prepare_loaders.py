"""Chunk 3: DP-1..DP-7, DP-9, DL-1..DL-3 on the tiny synthetic datasets."""
import os

import pyarrow.parquet as pq
import pytest

from src.config.loader import load_config
from src.data import loaders, mixture, prepare
from src.data.registry import DatasetRegistry
from src.io.storage import Storage
from src.tokenization.chat_template import ChatTemplate
from src.tokenization.tokenizer import ChatTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")
PAD, EOT, IM_END = 100277, 100257, 100265


@pytest.fixture()
def pctx(tiny_root, tmp_path):
    lc = load_config(TINY, [f"run.name=t{abs(hash(str(tmp_path))) % 10**8}"])
    cfg = lc.cfg
    st = Storage.from_config(cfg)
    tok = ChatTokenizer.from_config(cfg, st)
    ctx = prepare.PrepContext(cfg, st, tok, ChatTemplate.from_config(cfg, tok), DatasetRegistry.from_config(cfg, st),
                              log=lambda *_: None)
    return ctx


def rows_of(ctx, stage, name, split):
    return list(prepare.read_rows(ctx.storage, prepare.prepared_uri(ctx.cfg, stage, name, split)))


# --------------------------------------------------------------------------- LM (DP-5, DP-5a)
def test_lm_text_blocks(pctx):
    e = prepare.prepare_lm(pctx, "midtrain", "tiny_text", "train", 64)
    rows = rows_of(pctx, "midtrain", "tiny_text", "train")
    assert e["rows"] == len(rows) > 0
    for r in rows:
        assert len(r["tokens"]) == 65 and len(r["doc_start"]) == 65
    flat = [t for r in rows for t in r["tokens"] if t != PAD]
    assert flat.count(EOT) == e["records_kept"]                     # one EOT per document
    assert sum(sum(r["doc_start"]) for r in rows) == e["records_kept"]
    # padding only in the tail of the last block, doc_start False there
    last = rows[-1]
    assert all(not s for t, s in zip(last["tokens"], last["doc_start"]) if t == PAD)


def test_lm_packed_repack_and_skip_when_unchanged(pctx):
    e = prepare.prepare_lm(pctx, "midtrain_long", "tiny_packed", "train", 32)      # block 17 -> 33 (AT-24)
    rows = rows_of(pctx, "midtrain_long", "tiny_packed", "train")
    assert all(len(r["tokens"]) == 33 for r in rows)
    assert [t for r in rows for t in r["tokens"] if t != PAD] == [1] * 17 + [2] * 17
    marker = prepare.read_marker(pctx.storage, prepare.prepared_uri(pctx.cfg, "midtrain_long", "tiny_packed", "train"))
    e2 = prepare.prepare_lm(pctx, "midtrain_long", "tiny_packed", "train", 32)
    assert e2["prepared_uri"] == e["prepared_uri"] and marker["input_hash"]
    files = pctx.storage.glob(e["prepared_uri"], prepare.SHARD_GLOB)
    m = [os.path.getmtime(pctx.storage.local_path(f)) for f in files]
    prepare.prepare_lm(pctx, "midtrain_long", "tiny_packed", "train", 32)
    assert m == [os.path.getmtime(pctx.storage.local_path(f)) for f in files]        # cache hit: not rewritten
    prepare.prepare_lm(pctx, "midtrain_long", "tiny_packed", "train", 16)            # different ctx -> rebuilt
    assert all(len(r["tokens"]) == 17 for r in rows_of(pctx, "midtrain_long", "tiny_packed", "train"))


def test_decontamination_counter_in_manifest(pctx):
    e = prepare.prepare_lm(pctx, "midtrain", "tiny_text", "test", 64)
    assert set(e["decontamination"]) == {"checked", "dropped", "no_text"}
    assert e["decontamination"]["checked"] == e["records_kept"]


# --------------------------------------------------------------------------- SFT (DP-6)
def test_sft_blocks_align_and_never_split(pctx):
    e = prepare.prepare_sft(pctx, "sft", "tiny_chat", "train", 64)
    rows = rows_of(pctx, "sft", "tiny_chat", "train")
    assert e["conversations"] == sum(sum(r["doc_start"]) for r in rows)
    for r in rows:
        assert len(r["tokens"]) == len(r["doc_start"]) == len(r["loss_mask"]) == 65
        assert all(not m for t, m in zip(r["tokens"], r["loss_mask"]) if t == PAD)
        assert sum(r["loss_mask"]) > 0
    # every masked token belongs to an assistant turn: masks end exactly on <|im_end|> tokens
    for r in rows:
        toks, mask = r["tokens"], r["loss_mask"]
        for i in range(65):
            if mask[i] and (i + 1 == 65 or not mask[i + 1]):
                assert toks[i] == IM_END


def test_sft_over_long_conversations_are_dropped_and_counted(pctx):
    pctx.cfg.prepared_data.max_drop_fraction = 1.0
    with pytest.raises(ValueError, match="no SFT blocks"):          # everything longer than ctx 12 -> nothing to train on
        prepare.prepare_sft(pctx, "sft", "tiny_chat", "train", 12)
    e = prepare.prepare_sft(pctx, "sft", "tiny_chat", "train", 48)
    assert e["dropped"].get("longer_than_ctx", 0) == 0 and e["conversations"] == e["records_kept"]


def test_sft_drop_limit_trips_ds5(pctx):
    pctx.cfg.prepared_data.max_drop_fraction = 0.1
    with pytest.raises(Exception, match="max_drop_fraction"):
        prepare.prepare_sft(pctx, "sft", "tiny_chat", "train", 12)


# --------------------------------------------------------------------------- preference (DP-7)
def test_pref_rows_and_filters(pctx):
    e = prepare.prepare_pref(pctx, "preference", "tiny_pref", "train", 60)
    rows = rows_of(pctx, "preference", "tiny_pref", "train")
    assert e["rows"] == len(rows) > 0
    for r in rows:
        assert r["chosen_tokens"][-1] == IM_END and r["rejected_tokens"][-1] == IM_END
        assert len(r["prompt_tokens"]) + max(len(r["chosen_tokens"]), len(r["rejected_tokens"])) <= 60
    tight = prepare.prepare_pref(pctx, "preference", "tiny_pref", "test", 60)
    assert tight["rows"] > 0
    pctx.cfg.prepared_data.max_drop_fraction = 1.0
    totals = sorted(len(r["prompt_tokens"]) + max(len(r["chosen_tokens"]), len(r["rejected_tokens"])) for r in rows)
    assert totals[0] < totals[-1]
    small = prepare.prepare_pref(pctx, "preference", "tiny_pref", "train", totals[0])
    assert small["dropped"].get("longer_than_max_total_tokens", 0) > 0 and 0 < small["rows"] < len(rows)


# --------------------------------------------------------------------------- RL prompts (RL-1a)
def test_rl_prompts_carry_system_message(pctx, template):
    prepare.prepare_prompts_like(pctx, "rlvr", "tiny_math", "train")
    rows = rows_of(pctx, "rlvr", "tiny_math", "train")
    assert rows and all(r["ground_truth"].strip() for r in rows)
    sysmsg = pctx.storage.read_text(pctx.cfg.stages.rlvr.system_prompts.rl_math).strip()
    for r in rows[:3]:
        assert r["prompt_tokens"][-len(template.render_prompt([])):] == template.render_prompt([])   # ends with generation prompt
        assert pctx.tok.encode_ordinary(sysmsg)[:3] == r["prompt_tokens"][3:6] or sysmsg[:5] in pctx.tok.decode(r["prompt_tokens"])
    prepare.prepare_prompts_like(pctx, "rlvr", "tiny_code", "train")
    code = rows_of(pctx, "rlvr", "tiny_code", "train")
    assert code and "assert f() == 1" in code[0]["tests"] and code[0]["entry_point"] is None
    prepare.prepare_prompts_like(pctx, "distill_onpolicy", "tiny_prompts", "train")
    pr = rows_of(pctx, "distill_onpolicy", "tiny_prompts", "train")
    assert sysmsg[:5] not in pctx.tok.decode(pr[0]["prompt_tokens"])      # plain prompts have no system message


def test_stage_plan_and_manifest(pctx):
    plan = dict(prepare.stage_dataset_plan(pctx.cfg, "midtrain"))
    assert plan["tiny_text"] == ["train", "test"] and "test" in plan["tiny_packed"]
    out = prepare.prepare_stage_datasets(pctx, "midtrain", 64)
    man = prepare.read_stage_manifest(pctx.storage, pctx.cfg, "midtrain")
    assert set(man) == set(out) == {"tiny_text", "tiny_packed"}
    assert prepare.prepared_rows(pctx.storage, pctx.cfg, "midtrain", "tiny_text", "train") == out["tiny_text"]["train"]["rows"]
    with pytest.raises(FileNotFoundError):
        prepare.prepared_rows(pctx.storage, pctx.cfg, "sft", "tiny_chat", "train")


def test_produced_datasets_written(pctx):
    from src.data.adapters import Conversation, PrefPair
    convs = [Conversation([{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}])
             for i in range(6)]
    e = prepare.prepare_produced_sft(pctx, "distill_offpolicy", "teacher_traces", "train", convs, 64, True, {"n": 6})
    assert e["third_party_generated"] is True and e["conversations"] == 6
    pairs = [PrefPair([{"role": "user", "content": "q"}], "yes", "no")] * 3
    e2 = prepare.prepare_produced_pref(pctx, "preference", "on_policy", "train", pairs, 60, {})
    assert e2["rows"] == 3 and e2["third_party_generated"] is False


# --------------------------------------------------------------------------- loaders (DL-1)
def write_toy(pctx, n_rows, shard_rows, name="toy"):
    uri = prepare.prepared_uri(pctx.cfg, "midtrain", name, "train")
    rows = [{"tokens": [i] * 5, "doc_start": [True] + [False] * 4} for i in range(n_rows)]
    prepare.write_shards(pctx.storage, uri, "lm", rows, shard_rows)
    return uri


def ids(rows):
    return [r["tokens"][0] for r in rows]


@pytest.mark.parametrize("n_rows,shard_rows", [(20, 5), (20, 20)])         # file-level and row-level sharding
def test_rank_sharding_is_a_partition(pctx, n_rows, shard_rows):
    uri = write_toy(pctx, n_rows, shard_rows)
    got = []
    for rank in range(2):
        s = loaders.RowStream(pctx.storage, uri, ["tokens"], rank, 2, seed=3, cycle=False)
        got.append(ids(list(s)))
    assert sorted(got[0] + got[1]) == list(range(n_rows)) and not set(got[0]) & set(got[1])


def test_shuffle_is_seeded_and_changes_per_epoch(pctx):
    uri = write_toy(pctx, 12, 4)
    a = loaders.RowStream(pctx.storage, uri, ["tokens"], 0, 1, seed=1, cycle=True)
    e0 = [a.next() for _ in range(12)]
    e1 = [a.next() for _ in range(12)]
    b = loaders.RowStream(pctx.storage, uri, ["tokens"], 0, 1, seed=1, cycle=True)
    assert ids(e0) == ids([b.next() for _ in range(12)])
    assert ids(e0) != ids(e1) and sorted(ids(e0)) == sorted(ids(e1))


@pytest.mark.parametrize("consumed", [0, 3, 7, 12, 19, 30])
def test_fast_forward_matches_continuation(pctx, consumed):
    uri = write_toy(pctx, 12, 4)
    ref = loaders.RowStream(pctx.storage, uri, ["tokens"], 0, 1, seed=5, cycle=True)
    for _ in range(consumed):
        ref.next()
    expect = ids([ref.next() for _ in range(15)])
    res = loaders.RowStream(pctx.storage, uri, ["tokens"], 0, 1, seed=5, cycle=True)
    res.fast_forward(consumed)
    assert ids([res.next() for _ in range(15)]) == expect


def test_fewer_rows_than_ranks_every_rank_reads_all(pctx):
    uri = write_toy(pctx, 1, 1)
    for rank in range(2):
        s = loaders.RowStream(pctx.storage, uri, ["tokens"], rank, 2, seed=1, cycle=True)
        assert ids([s.next(), s.next()]) == [0, 0]          # cycles over its single row


def test_collation(pctx):
    rows = [{"tokens": [1, 2, PAD], "doc_start": [True, False, False], "loss_mask": [True, True, False]}]
    lm = loaders.collate_blocks(rows, PAD, lm=True)
    assert lm["loss_mask"].tolist() == [[True, True, False]]
    sft = loaders.collate_blocks(rows, PAD, lm=False)
    assert sft["loss_mask"].tolist() == [[True, True, False]]
    idx, mask = loaders.response_batch([[7, 8], [9]], [[1, 2, 3], [4]], PAD)
    assert idx.tolist() == [[7, 8, 1, 2, 3], [9, 4, PAD, PAD, PAD]]
    assert mask.tolist() == [[False, True, True, True], [True, False, False, False]]
    pref = loaders.collate_pref([{"prompt_tokens": [7], "chosen_tokens": [1, 2], "rejected_tokens": [3],
                                  "ref_logp_chosen": -1.0, "ref_logp_rejected": -2.0}], PAD)
    assert pref["idx"].shape == (2, 3) and pref["len_chosen"].tolist() == [2] and pref["ref_chosen"].tolist() == [-1.0]


# --------------------------------------------------------------------------- mixtures (DP-9)
def test_epoch_counts_and_steps():
    assert mixture.epoch_counts([100, 10], [0.7, 0.3]) == [77, 33]
    assert mixture.epoch_counts([1, 1], [0.99, 0.01])[1] >= 1
    assert mixture.steps_for_budget(2000, micro=2, accum=2, world=1, ctx=64) == 8
    assert mixture.steps_for_epochs(2, [10, 6], micro=2, accum=2, world=2) == 4


def test_quota_mixture_exact_counts_and_resume(pctx):
    ua, ub = write_toy(pctx, 8, 4, "a"), write_toy(pctx, 4, 4, "b")

    def build():
        s = [loaders.RowStream(pctx.storage, u, ["tokens"], 0, 1, seed=9, cycle=True) for u in (ua, ub)]
        return mixture.MixtureStream(s, [0.75, 0.25], seed=11, quotas=[6, 2])
    m = build()
    epoch = [m.next()[0] for _ in range(8)]
    assert epoch.count(0) == 6 and epoch.count(1) == 2                 # exact per-epoch counts
    ref = build()
    for _ in range(11):
        ref.next()
    expect = [ref.next() for _ in range(9)]
    res = build()
    res.fast_forward(11)
    assert [res.next() for _ in range(9)] == expect


def test_weighted_mixture_proportions(pctx):
    ua, ub = write_toy(pctx, 8, 4, "a"), write_toy(pctx, 8, 4, "b")
    s = [loaders.RowStream(pctx.storage, u, ["tokens"], 0, 1, seed=9, cycle=True) for u in (ua, ub)]
    m = mixture.MixtureStream(s, [0.7, 0.3], seed=2)
    picks = [m.next()[0] for _ in range(2000)]
    assert 0.65 < picks.count(0) / 2000 < 0.75
