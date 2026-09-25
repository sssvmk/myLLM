import base64
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
FOUNDATION = os.environ.get("PF_FOUNDATION_LLM", os.path.join(os.path.dirname(ROOT), "foundation_llm"))
os.environ["PF_FOUNDATION_LLM"] = FOUNDATION
os.environ["PF_PROMPTS_DIR"] = os.path.join(ROOT, "prompts")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from src.foundation import bridge  # noqa: E402

bridge.load(FOUNDATION)

import torch  # noqa: E402

VOCAB_ROWS = 100278
N = 40          # rows per synthetic training dataset: enough for non-empty derived holdouts
TINY = dict(d_model=32, n_layer=2, n_heads=4, d_ff=64, ctx=64, rope_theta=10000.0, d_latent=64,
            d_rope=8, n_routed_experts=4, n_shared_experts=1, moe_top_k=2, vocab_rows=VOCAB_ROWS)


def arch(kind):
    return dict(TINY, arch=kind)


def ranks_bytes():
    return b"".join(base64.b64encode(bytes([i])) + b" " + str(i).encode() + b"\n" for i in range(256))


def make_tokenizer():
    from src.tokenization.tokenizer import ChatTokenizer
    data = ranks_bytes()
    return ChatTokenizer(data, hashlib.sha256(data).hexdigest(), 100257, 100277,
                         {"im_start": {"text": "<|im_start|>", "id": 100264},
                          "im_end": {"text": "<|im_end|>", "id": 100265},
                          "think_open": {"text": "<think>", "id": 100266},
                          "think_close": {"text": "</think>", "id": 100267}}, VOCAB_ROWS)


def make_template(tok):
    from src.tokenization.chat_template import ChatTemplate
    return ChatTemplate(tok, "<|im_start|>{role}\n", "<|im_end|>\n", "<|im_start|>assistant\n", "<think>", "</think>")


def make_model(kind, seed=0):
    from src.modeling.loading import build_model
    torch.manual_seed(seed)
    m = build_model(arch(kind)).eval()
    # foundation init leaves tiny random weights; scale up the embedding so logits are not all ties
    with torch.no_grad():
        m.tok_emb.weight.mul_(20)
    return m


@pytest.fixture(scope="session")
def tok():
    return make_tokenizer()


@pytest.fixture(scope="session")
def template(tok):
    return make_template(tok)


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    os.environ["PF_TINY_ROOT"] = str(root)
    data = ranks_bytes()
    os.environ["PF_TINY_RANKS_SHA"] = hashlib.sha256(data).hexdigest()
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tiny.tiktoken").write_bytes(data)
    # base checkpoint in foundation_llm format (keys model/optimizer/step/best_loss/args)
    (root / "base").mkdir()
    m = make_model("deepseek")
    torch.save({"model": m.state_dict(), "optimizer": {}, "step": 10, "best_loss": 3.0,
                "args": {"arch": "deepseek", "ctx": 64, "rope_theta": 10000.0, "d_latent": 0, "d_rope": 8,
                         "n_routed_experts": 4, "n_shared_experts": 1, "moe_top_k": 2}},
               root / "base" / "latest_check.pt")
    d = root / "data"

    def jl(sub, name, rows):
        (d / sub).mkdir(parents=True, exist_ok=True)
        with open(d / sub / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    jl("text", "train-0.jsonl", [{"text": f"document number {i} about cats and dogs"} for i in range(N)])
    jl("chat", "train-0.jsonl", [{"messages": [{"role": "user", "content": f"hi {i}"},
                                               {"role": "assistant", "content": f"hello {i}"}]} for i in range(N)])
    jl("pref", "train-0.jsonl", [{"prompt": f"q{i}", "chosen": "good", "rejected": "bad"} for i in range(N)])
    jl("prompts", "train-0.jsonl", [{"prompt": f"say {i}"} for i in range(N)])
    jl("math", "train-0.jsonl", [{"question": f"{i}+1?", "answer": f"it is #### {i + 1}"} for i in range(N)])
    jl("code", "train-0.jsonl", [{"text": f"write f{i}", "test_list": ["assert f() == 1"]} for i in range(N // 2)])
    jl("eval", "mmlu.jsonl", [{"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}])
    jl("eval", "gsm8k.jsonl", [{"question": "1+1?", "answer": "#### 2"}])
    import pyarrow as pa
    import pyarrow.parquet as pq
    for split in ("train", "test"):
        p = d / "packed" / split / "shard_id=0"
        p.mkdir(parents=True)
        pq.write_table(pa.table({"tokens": [[1] * 17, [2] * 17], "doc_start": [[True] + [False] * 16] * 2}),
                       p / "part-0.parquet")
    return root


@pytest.fixture()
def stub(monkeypatch):
    from stub_server import Stub
    monkeypatch.setenv("PF_TINY_JUDGE_KEY", "k")
    monkeypatch.setenv("PF_TINY_TEACHER_KEY", "k")
    s = Stub().start()
    yield s
    s.stop()
