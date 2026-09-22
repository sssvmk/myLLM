"""Synthetic-data smoke test -- no Spark, no real corpus needed. Run this first after
installing the package's requirements, before pointing anything at real data, to confirm the
model, both architectures, and the loss path run end to end (forward, backward, optimizer
step) without shape errors.

    python -m pytest foundation_llm/tests/test_model_smoke.py -q
  or just:
    python foundation_llm/tests/test_model_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from model import GPTModel, build_doc_attention_mask
from train import build_optimizer, _update_moe_bias


def _run_arch(arch: str):
  torch.manual_seed(0)
  vocab_size, pad_id = 300, 300   # tiny synthetic vocab + reserved pad id, mirrors main.py's +1
  B, T = 2, 16

  kwargs = dict(vocab_size=vocab_size + 1, d_model=32, ctx=T, n_layers=2, d_ff=64, n_heads=4,
                dropout=0.1, arch=arch)
  if arch == "deepseek":
    kwargs.update(d_latent=8, d_rope=8, n_routed_experts=4, n_shared_experts=1, moe_top_k=2)

  model = GPTModel(**kwargs)
  optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)

  idx = torch.randint(0, vocab_size, (B, T + 1))
  # fake one document boundary partway through, to exercise the doc-mask path
  doc_start = torch.zeros(B, T, dtype=torch.bool)
  doc_start[:, 0] = True
  doc_start[:, T // 2] = True

  xb, yb = idx[:, :-1], idx[:, 1:]
  attn_mask = build_doc_attention_mask(doc_start)
  logits, aux_loss = model(xb, attn_mask=attn_mask)
  assert logits.shape == (B, T, vocab_size + 1), logits.shape

  loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), ignore_index=pad_id)
  if aux_loss is not None:
    loss = loss + aux_loss
  loss.backward()
  optimizer.step()
  _update_moe_bias(model)   # no-op for dense arch, exercises the bias update for deepseek
  optimizer.zero_grad()

  # tied weights must still be the same tensor after a step
  assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
  print(f"[{arch}] OK  loss={loss.item():.4f}  logits={tuple(logits.shape)}")


def test_dense():
  _run_arch("dense")


def test_deepseek():
  _run_arch("deepseek")


if __name__ == "__main__":
  test_dense()
  test_deepseek()
  print("all smoke tests passed")
