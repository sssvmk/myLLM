"""TR-2 / PT-9: the MoE auxiliary loss has a gradient (docs/prd_review.md #3)."""
import os
import uuid

import pytest
import torch

from conftest import make_model
from src.training.moe_balance import install_balance_loss, moe_modules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "configs", "tiny_test.yaml")


def batch(seed=0, n=4, t=24):
    return torch.randint(0, 256, (n, t), generator=torch.Generator().manual_seed(seed))


def manual_aux(model, idx, weight):
    """Independent recomputation from each MoE layer's own input, router and bias."""
    inputs = []
    hooks = [m.register_forward_hook(lambda mod, i, o, store=inputs: store.append((mod, i[0].detach()))) for m in moe_modules(model)]
    with torch.no_grad():
        model(idx)
    for h in hooks:
        h.remove()
    total = 0.0
    for mod, x in inputs:
        flat = x.reshape(-1, x.shape[-1])
        scores = mod.router(flat)
        p = torch.softmax(scores, -1)
        sel = (scores + mod.routing_bias).topk(mod.top_k, -1).indices
        f = torch.bincount(sel.reshape(-1), minlength=mod.n_routed).float() / sel.numel()
        total += weight * mod.n_routed * float((f * p.mean(0)).sum())
    return total


def test_value_matches_independent_formula():
    m = make_model("deepseek")
    for mod in moe_modules(m):
        mod.aux_loss_weight = 0.5
    assert install_balance_loss(m) == 2 and install_balance_loss(m) == 0          # idempotent
    idx = batch()
    _, aux = m(idx)
    assert aux.requires_grad
    assert float(aux) == pytest.approx(manual_aux(m, idx, 0.5), rel=1e-5)


def test_uniform_router_gives_weight_times_one():
    m = make_model("deepseek")
    for mod in moe_modules(m):
        mod.aux_loss_weight = 0.3
        torch.nn.init.zeros_(mod.router.weight)                # all experts equally likely: P_i = 1/N
        mod.routing_bias.data = torch.arange(mod.n_routed).float() * 1e-6      # only breaks top-k ties
    install_balance_loss(m)
    _, aux = m(batch(t=64))
    n_layers = len(moe_modules(m))
    # f_i is exactly uniform only in expectation, but with P uniform the loss is weight * N * (1/N) * sum f = weight
    assert float(aux) == pytest.approx(0.3 * n_layers, rel=1e-5)


def test_gradient_reaches_the_router_and_zero_weight_leaves_the_module_alone():
    m = make_model("deepseek")
    for mod in moe_modules(m):
        mod.aux_loss_weight = 0.4
    install_balance_loss(m)
    _, aux = m(batch())
    aux.backward()
    assert all(mod.router.weight.grad is not None and float(mod.router.weight.grad.abs().sum()) > 0 for mod in moe_modules(m))
    m0 = make_model("deepseek")
    for mod in moe_modules(m0):
        mod.aux_loss_weight = 0.0
    install_balance_loss(m0)
    _, aux0 = m0(batch())
    assert not aux0.requires_grad                              # foundation_llm's own count-based value, as before


def test_minimising_the_loss_rebalances_routing():
    m = make_model("deepseek")
    mods = moe_modules(m)
    for mod in mods:
        mod.aux_loss_weight = 1.0
        mod.routing_bias.data.zero_()
        mod.router.weight.data[0] += 3.0 * mod.router.weight.data.std()      # expert 0 starts over-used
    install_balance_loss(m)
    for p in m.parameters():
        p.requires_grad_(False)
    router_params = [mod.router.weight for mod in mods]
    for p in router_params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(router_params, lr=2.0)
    idx = batch(n=8, t=32)
    first = None
    for _ in range(60):
        opt.zero_grad()
        _, aux = m(idx)
        first = float(aux) if first is None else first
        aux.backward()
        opt.step()
    assert float(aux) < first and float(aux) < 2.05 * 1.0           # 2 layers: perfectly balanced = 2.0


def test_trainer_step_changes_with_the_weight(tiny_root):
    from src.config.loader import load_config
    from src.data import prepare
    from src.data.registry import DatasetRegistry
    from src.io.storage import Storage
    from src.stages import common, midtrain
    from src.tokenization.chat_template import ChatTemplate
    from src.tokenization.tokenizer import ChatTokenizer
    from src.training.loop import Trainer
    norms = {}
    for w in (0.0, 5.0):
        lc = load_config(TINY, [f"run.name=b{uuid.uuid4().hex[:6]}", f"stages.midtrain.moe.aux_loss_weight={w}",
                                "stages.midtrain.moe.update_routing_bias=false"])
        st = Storage.from_config(lc.cfg)
        tok = ChatTokenizer.from_config(lc.cfg, st)
        prepare.prepare_stage_datasets(prepare.PrepContext(lc.cfg, st, tok, ChatTemplate.from_config(lc.cfg, tok),
                                                            DatasetRegistry.from_config(lc.cfg, st), log=lambda *_: None),
                                       "midtrain", 64)
        sctx = common.make_stage_context(lc, "midtrain", False, st, (0, 1, 0))
        obj = midtrain.build_objective(sctx)
        t = Trainer(sctx, common.load_stage_model(sctx), obj)
        obj.setup(t)
        norms[w] = obj.train_step(t, 0)["grad_norm"]
    assert norms[0.0] != pytest.approx(norms[5.0], rel=1e-3)          # aux_loss_weight now changes the gradient
