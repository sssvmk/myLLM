"""AT-4..AT-9 (loading, inference equivalence, KV cache, positions, batching, sampling)."""
import pytest
import torch

from conftest import arch, make_model
from src.generation.api import GenerationConfig, generate
from src.generation.sampler import sample_next
from src.modeling.inference import InferenceModel, routing_topk
from src.modeling.loading import load_model, validate_base_args

ARCHS = ["dense", "deepseek"]


def gcfg(**kw):
    base = dict(max_new_tokens=32, temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0, num_samples=1,
                seed=7, stop_token_ids=[], stop_strings=[], batch_size=4, use_kv_cache=True,
                return_logprobs=True, overflow="error")
    base.update(kw)
    return GenerationConfig(**base)


def test_strict_load_with_prefixes():
    m = make_model("deepseek")
    sd = {"_orig_mod.module." + k: v for k, v in m.state_dict().items()}
    m2 = load_model({"model": sd}, arch("deepseek"), torch.device("cpu"))
    assert torch.equal(m2.tok_emb.weight, m.tok_emb.weight)
    assert m2.lm_head.weight is m2.tok_emb.weight


def test_arch_mismatch_raises():
    m = make_model("deepseek")
    base = {"model": m.state_dict(), "optimizer": {}, "step": 0, "best_loss": 0.0,
            "args": {"arch": "deepseek", "ctx": 64, "rope_theta": 10000.0, "d_latent": 0, "d_rope": 8,
                     "n_routed_experts": 4, "n_shared_experts": 1, "moe_top_k": 2}}
    load_model(base, arch("deepseek"), torch.device("cpu"))                 # d_latent 0 resolves to 64
    base["args"]["moe_top_k"] = 1
    with pytest.raises(ValueError, match="moe_top_k"):
        load_model(base, arch("deepseek"), torch.device("cpu"))
    with pytest.raises(ValueError, match="vocab_rows"):
        load_model({"model": m.state_dict()}, dict(arch("deepseek"), vocab_rows=100279), torch.device("cpu"))


def test_dense_ignores_deepseek_knobs():
    validate_base_args({"arch": "dense", "ctx": 64, "rope_theta": 1e4, "d_latent": 0, "n_routed_experts": 8},
                       arch("dense"))


@pytest.mark.parametrize("kind", ARCHS)
def test_inference_equivalence(kind):                                          # AT-5 / MD-11
    m = make_model(kind)
    idx = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        ref, _ = m(idx)
        got = InferenceModel(m)(idx)
    assert (ref - got).abs().max().item() < 1e-5


@pytest.mark.parametrize("kind", ARCHS)
def test_cached_matches_uncached(kind, tok):                                   # AT-6
    im = InferenceModel(make_model(kind))
    prompt = [[5, 6, 7, 8, 9]]
    a = generate(im, tok, prompt, gcfg(use_kv_cache=True))[0]
    b = generate(im, tok, prompt, gcfg(use_kv_cache=False))[0]
    assert a.completion_tokens == b.completion_tokens and len(a.completion_tokens) == 32


@pytest.mark.parametrize("kind", ARCHS)
def test_positions(kind):                                                      # AT-7
    from src.modeling.kv_cache import KVCache
    im = InferenceModel(make_model(kind))
    idx = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        full = im(idx)
        cache = KVCache(im.model, 1, 12, torch.float32, torch.device("cpu"))
        for t in range(12):
            step = im(idx[:, t:t + 1], position_ids=torch.tensor([[t]]), kv_cache=cache)
            assert (step[0, 0] - full[0, t]).abs().max() < 1e-4


@pytest.mark.parametrize("kind", ARCHS)
def test_batched_left_padded_equals_single(kind, tok):                         # AT-8
    im = InferenceModel(make_model(kind))
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    batched = generate(im, tok, prompts, gcfg(max_new_tokens=12, batch_size=3))
    for p, r in zip(prompts, batched):
        single = generate(im, tok, [p], gcfg(max_new_tokens=12, batch_size=1))[0]
        assert single.completion_tokens == r.completion_tokens


def test_pad_and_undefined_never_sampled_and_seeded(tok):                      # AT-9
    allowed = tok.allowed_mask()
    logits = torch.zeros(4, allowed.numel())
    logits[:, tok.pad_token_id] = 1e4
    logits[:, 100261] = 1e4                                                     # undefined id
    seen = torch.zeros_like(logits, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        nxt, _ = sample_next(logits, allowed, seen, 1.0, 0, 1.0, 1.0, g)
        assert allowed[nxt].all()
    im = InferenceModel(make_model("dense"))
    c = gcfg(temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=10, num_samples=3)
    r1 = generate(im, tok, [[1, 2, 3]], c)
    r2 = generate(im, tok, [[1, 2, 3]], c)
    assert [r.completion_tokens for r in r1] == [r.completion_tokens for r in r2]
    assert len(r1) == 3                                                          # GN-7


def test_stop_and_overflow(tok):
    im = InferenceModel(make_model("dense"))
    first = generate(im, tok, [[1, 2, 3]], gcfg(max_new_tokens=5))[0].completion_tokens[0]
    r = generate(im, tok, [[1, 2, 3]], gcfg(max_new_tokens=5, stop_token_ids=[first]))[0]
    assert r.finish_reason == "eos" and r.completion_tokens == [first]
    with pytest.raises(ValueError, match="ctx"):
        generate(im, tok, [list(range(60))], gcfg(max_new_tokens=10, overflow="error"))
    r = generate(im, tok, [list(range(60))], gcfg(max_new_tokens=10, overflow="truncate_left"))[0]
    assert r.truncated_prompt_tokens == 6 and len(r.prompt_tokens) == 54


def test_logprobs_match_training_forward(tok):
    """GN-4 / RL-6: generation log-probs equal the training forward's log-probs (fp32, CPU)."""
    m = make_model("deepseek")
    im = InferenceModel(m)
    r = generate(im, tok, [[3, 4, 5, 6]], gcfg(temperature=1.0, max_new_tokens=10))[0]
    seq = torch.tensor([r.prompt_tokens + r.completion_tokens])
    with torch.no_grad():
        logits, _ = m(seq[:, :-1])
    lp = torch.log_softmax(logits.float().masked_fill(~tok.allowed_mask()[None, None], float("-inf")), -1)
    tgt = seq[0, 4:]
    train_lp = lp[0, 3:].gather(-1, tgt[:, None]).squeeze(-1)
    assert (train_lp - torch.tensor(r.logprobs)).abs().max() < 1e-4


def test_routing_topk_matches_moe():
    m = make_model("deepseek")
    moe = m.blocks[0].mlp
    h = torch.randn(2, 5, 32)
    moe.routing_bias.data = torch.tensor([0.5, -0.5, 0.0, 0.2])
    expected = (moe.router(h.reshape(-1, 32)) + moe.routing_bias).topk(2, -1).indices
    assert torch.equal(routing_topk(moe, h), expected)
