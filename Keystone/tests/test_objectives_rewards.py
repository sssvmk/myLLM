"""AT-13..AT-18 schedules, losses, rewards."""
import math
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from src.rl.rewards import score_response
from src.rl.rewards.code import build_program, last_code_block
from src.rl.rewards.format import format_ok
from src.rl.rewards.math import extract_answer, extract_ground_truth, last_boxed, math_equal
from src.rl.rewards.overlong import overlong_penalty
from src.rl.sandbox import Sandbox
from src.training.objectives.dpo import dpo_loss
from src.training.objectives.gspo import group_advantages, gspo_loss
from src.training.objectives.opd import opd_loss
from src.training.objectives.sft import sft_loss_sum
from src.training.objectives.simpo import simpo_loss
from src.training.schedules import lr_at


def test_schedules():
    wsd = NS(type="wsd", warmup_fraction=0.1, stable_fraction=0.5, decay_fraction=0.4, decay_shape="linear")
    assert lr_at(0, 100, wsd, 1.0, 0.1) == 0.0
    assert lr_at(5, 100, wsd, 1.0, 0.1) == pytest.approx(0.5)
    assert lr_at(10, 100, wsd, 1.0, 0.1) == 1.0 and lr_at(59, 100, wsd, 1.0, 0.1) == 1.0
    assert lr_at(80, 100, wsd, 1.0, 0.1) == pytest.approx(0.1 + 0.9 * 0.5)
    assert lr_at(100, 100, wsd, 1.0, 0.1) == pytest.approx(0.1)
    cos = NS(type="wsd", warmup_fraction=0.0, stable_fraction=0.0, decay_fraction=1.0, decay_shape="cosine")
    assert lr_at(50, 100, cos, 1.0, 0.0) == pytest.approx(0.5)
    lin = NS(type="linear", warmup_fraction=0.1)
    assert lr_at(55, 100, lin, 1.0, 0.0) == pytest.approx(0.5)
    c = NS(type="cosine", warmup_fraction=0.0)
    assert lr_at(100, 100, c, 1.0, 0.2) == pytest.approx(0.2)
    k = NS(type="constant", warmup_fraction=0.2)
    assert lr_at(1, 10, k, 2.0, 0.0) == pytest.approx(1.0) and lr_at(7, 10, k, 2.0, 0.0) == 2.0


def test_sft_loss_hand_computed():
    logits = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    tgt = torch.tensor([[0, 2]])
    mask = torch.tensor([[True, False]])
    s, n = sft_loss_sum(logits, tgt, mask)
    assert n.item() == 1
    assert s.item() == pytest.approx(-math.log(math.exp(2) / (math.exp(2) + 2)), rel=1e-6)


def test_dpo_simpo():
    z = torch.tensor([-5.0, -3.0])
    loss, st = dpo_loss(z, z - 1, z, z - 1, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))                     # model == reference
    loss, _ = dpo_loss(torch.tensor([-1.0]), torch.tensor([-4.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), 0.5)
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-0.5 * 3))))
    loss, _ = simpo_loss(torch.tensor([-4.0]), torch.tensor([-9.0]), torch.tensor([2]), torch.tensor([3]), 2.0, 0.5)
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-(2 * -2 - 2 * -3 - 0.5)))))


def test_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    a = group_advantages(r, 3, True, 1e-6)
    std = torch.tensor([1.0, 0.0, 0.0]).std(unbiased=True)
    assert torch.allclose(a[:3], (torch.tensor([1.0, 0, 0]) - 1 / 3) / (std + 1e-6))
    assert torch.allclose(a[3:], torch.zeros(3))
    assert torch.allclose(group_advantages(r, 3, False, 1e-6)[:3], torch.tensor([2 / 3, -1 / 3, -1 / 3]))


def test_gspo_hand_computed():
    old = torch.zeros(2, 3)
    new = torch.tensor([[0.3, 0.3, 0.0], [-0.001, -0.001, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    adv = torch.tensor([1.0, -1.0])
    loss, st = gspo_loss(new, old, mask, adv, eps_low=0.1, eps_high=0.2)
    s1, s2 = math.exp(0.3), math.exp(-0.001)
    l1 = -min(s1 * 1, min(max(s1, 0.9), 1.2) * 1)
    l2 = -min(s2 * -1, min(max(s2, 0.9), 1.2) * -1)
    assert loss.item() == pytest.approx((l1 + l2) / 2, rel=1e-6)
    assert st["clip_fraction"] == pytest.approx(0.5)


def test_opd():
    ls = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    lt = torch.tensor([[-1.5, -1.0]])
    mask = torch.tensor([[True, True]])
    loss, st = opd_loss(ls, lt, mask)
    assert loss.item() == pytest.approx((0.5 * -1 + -1 * -2) / 2)
    assert st["reverse_kl_per_token"] == pytest.approx(-0.25)
    ls2 = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    opd_loss(ls2, ls2.detach().clone(), mask)[0].backward()
    assert torch.equal(ls2.grad, torch.zeros_like(ls2))


def test_math_rewards():
    assert last_boxed(r"x \boxed{\frac{1}{2}} y \boxed{3}") == "3"
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    rx = r"(?i)final answer\s*[:：]\s*(.+)"
    assert extract_answer("so Final answer: 12", rx) == "12"
    assert extract_answer("we get 3 then 1,234", rx) == "1234"
    assert extract_ground_truth("steps #### 1,234", "regex", r"####\s*(-?[\d,\.]+)") == "1234"
    assert math_equal("0.5", r"\frac{1}{2}")
    assert math_equal("1234", "1234.0")
    assert not math_equal("3", "4")


def test_format_and_overlong():
    assert format_ok("<think>a</think>b", "eos", "<think>", "</think>")
    assert format_ok("b", "eos", "<think>", "</think>")
    assert not format_ok("<think>a</think>b", "length", "<think>", "</think>")
    assert not format_ok("<think>a<think></think>", "eos", "<think>", "</think>")
    assert overlong_penalty(10, False, 20, 4, 1.0) == 0.0
    assert overlong_penalty(18, False, 20, 4, 1.0) == pytest.approx(-0.5)
    assert overlong_penalty(20, True, 20, 4, 1.0) == -1.0


def _sandbox():
    return Sandbox("python3", 2, 512, 64, 4096, 2, ["PATH"], True)


def test_code_rewards():
    sb = _sandbox()
    text = "<think>plan</think>\n```python\ndef f():\n    return 1\n```"
    code = last_code_block(text)
    assert sb.run(build_program(code, ["assert f() == 1"], None)).status == "pass"
    assert sb.run(build_program(code, ["assert f() == 2"], None)).status == "fail"
    assert sb.run("while True: pass").status == "timeout"
    prog = build_program("def g(x): return x", "def check(c):\n    assert c(3) == 3", "g")
    assert prog.endswith("check(g)\n") and sb.run(prog).passed
    with pytest.raises(PermissionError):
        Sandbox("python3", 1, 256, 64, 64, 1, [], False)


def test_total_reward(template):
    rewards = NS(correctness_weight=1.0, format_weight=0.1, math=NS(final_answer_regex=r"(?i)final answer\s*[:：]\s*(.+)"))
    over = NS(mode="soft_penalty", buffer_tokens=4, penalty_factor=1.0)
    r = score_response({"kind": "rl_math", "ground_truth": "7"}, "<think>x</think> \\boxed{7}", "eos", 5,
                       template, rewards, over, 20)
    assert (r.correct, r.format_ok, r.reward) == (1, 1, pytest.approx(1.1))
    r = score_response({"kind": "rl_math", "ground_truth": "7"}, "<think>x \\boxed{7}", "length", 20,
                       template, rewards, NS(mode="exclude", buffer_tokens=4, penalty_factor=1.0), 20)
    assert (r.correct, r.format_ok, r.excluded) == (0, 0, True)
