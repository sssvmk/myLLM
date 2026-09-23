"""
Model architectures.

arch='dense'    : RoPE + standard multi-head causal attention + GELU MLP. Replaces the
                   original's learned absolute position embedding (dropped) and fixes the
                   previously-unused `rope` flag by actually applying RoPE, and the previously
                   dead `nn.Dropout` (now applied in Block.forward).
arch='deepseek' : Multi-head Latent Attention (MLA) + DeepSeekMoE per block.

Both share RMSNorm and a weight-tied LM head, with the tie fixed to happen *after*
initialization (the original applied Xavier init to the tied Linear after tying, which
silently clobbered the embedding's intended normal(0, 0.02) init).
"""
from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import DeepSeekMoE


class RMSNorm(nn.Module):
  def __init__(self, d, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d))

  def forward(self, x):
    norm = x.pow(2).mean(-1, keepdim=True)
    x = x / torch.sqrt(norm + self.eps)
    return self.weight * x


def build_rope_cache(seq_len: int, dim: int, theta: float = 10000.0, device=None, dtype=torch.float32):
  """cos/sin tables for rotary position embeddings, shape (seq_len, dim) each."""
  inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
  t = torch.arange(seq_len, device=device, dtype=dtype)
  freqs = torch.outer(t, inv_freq)            # (T, dim/2)
  emb = torch.cat([freqs, freqs], dim=-1)      # (T, dim)
  return emb.cos(), emb.sin()


def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
  """x: (B, n_heads, T, head_dim). cos/sin: (T, head_dim)."""
  cos = cos[None, None, :, :].to(x.dtype)
  sin = sin[None, None, :, :].to(x.dtype)
  return x * cos + rotate_half(x) * sin


def build_doc_attention_mask(doc_start: torch.Tensor) -> torch.Tensor:
  """doc_start: (B, T) bool, True marks the first token of a document within a packed block
  (see packing.pack_source_to_shards / data.ParquetTokenDataset). Returns (B, 1, T, T) bool
  where True = attention is allowed (causal AND same document) -- passed straight to
  F.scaled_dot_product_attention's attn_mask. Prevents a packed block from silently attending
  across a concatenated-document seam."""
  B, T = doc_start.shape
  seg_id = torch.cumsum(doc_start.long(), dim=1)               # (B, T)
  same_seg = seg_id.unsqueeze(2) == seg_id.unsqueeze(1)          # (B, T, T)
  causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=doc_start.device))
  mask = same_seg & causal.unsqueeze(0)
  return mask.unsqueeze(1)                                       # (B, 1, T, T)


class MLP(nn.Module):
  """Standard GELU MLP, used by the dense baseline."""
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()
    self.fc1 = nn.Linear(d_model, d_ff, bias=False)
    self.fc2 = nn.Linear(d_ff, d_model, bias=False)
    self.act = nn.GELU(approximate='tanh')

  def forward(self, x):
    return self.fc2(self.act(self.fc1(x)))


class CausalSelfAttention(nn.Module):
  """Standard multi-head causal self-attention with RoPE (dense baseline). Accepts an
  optional document-boundary attn_mask (see build_doc_attention_mask)."""
  def __init__(self, d_model: int, n_heads: int, rope_theta: float = 10000.0):
    super().__init__()
    assert d_model % n_heads == 0
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.rope_theta = rope_theta
    self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
    self.proj = nn.Linear(d_model, d_model, bias=False)

  def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
    B, T, C = x.size()
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    cos, sin = build_rope_cache(T, self.head_dim, self.rope_theta, device=x.device)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if attn_mask is not None:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    else:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.proj(out)


class MultiHeadLatentAttention(nn.Module):
  """Simplified Multi-head Latent Attention (DeepSeek-V2/V3 style).

  Compresses K/V into a shared low-rank latent (d_latent << d_model) and reconstructs
  per-head K/V from it, plus a small decoupled RoPE-only query/key path concatenated onto the
  "content" head dims. Query is also low-rank projected, mirroring DeepSeek's design.

  NOTE: this reproduces MLA's *shapes and information flow* so it's structurally correct for
  training; it doesn't implement incremental KV caching (the actual inference-time memory
  win), since this codebase only runs full-sequence training forward passes. `c_kv` below is
  what you'd cache if this were later wired up for generation.
  """
  def __init__(self, d_model: int, n_heads: int, d_latent: int, d_rope: int = 32,
               rope_theta: float = 10000.0):
    super().__init__()
    assert d_model % n_heads == 0
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.d_rope = d_rope
    self.rope_theta = rope_theta

    # KV low-rank path
    self.kv_down = nn.Linear(d_model, d_latent, bias=False)
    self.kv_norm = RMSNorm(d_latent)
    self.kv_up = nn.Linear(d_latent, n_heads * self.head_dim * 2, bias=False)   # -> k_content, v
    self.k_rope = nn.Linear(d_model, n_heads * d_rope, bias=False)               # decoupled RoPE key

    # Query low-rank path
    self.q_down = nn.Linear(d_model, d_latent, bias=False)
    self.q_norm = RMSNorm(d_latent)
    self.q_up = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)
    self.q_rope = nn.Linear(d_latent, n_heads * d_rope, bias=False)

    self.proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

  def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
    B, T, C = x.size()
    H, Dh, Dr = self.n_heads, self.head_dim, self.d_rope

    c_kv = self.kv_norm(self.kv_down(x))                            # (B,T,d_latent) -- cache this for inference
    kv = self.kv_up(c_kv).view(B, T, H, 2 * Dh).transpose(1, 2)       # (B,H,T,2*Dh)
    k_content, v = kv.split(Dh, dim=-1)
    k_rope = self.k_rope(x).view(B, T, H, Dr).transpose(1, 2)         # (B,H,T,Dr)

    c_q = self.q_norm(self.q_down(x))
    q_content = self.q_up(c_q).view(B, T, H, Dh).transpose(1, 2)
    q_rope = self.q_rope(c_q).view(B, T, H, Dr).transpose(1, 2)

    cos, sin = build_rope_cache(T, Dr, self.rope_theta, device=x.device)
    q_rope = apply_rope(q_rope, cos, sin)
    k_rope = apply_rope(k_rope, cos, sin)

    q = torch.cat([q_content, q_rope], dim=-1)     # (B,H,T,Dh+Dr)
    k = torch.cat([k_content, k_rope], dim=-1)

    if attn_mask is not None:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    else:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
    return self.proj(out)


class Block(nn.Module):
  """Pre-norm transformer block. `attn` and `mlp` are injected so the same wiring serves both
  the dense baseline and the deepseek arch. Dropout is applied here (fix: the original
  constructed nn.Dropout but never called it)."""
  def __init__(self, d_model: int, attn: nn.Module, mlp: nn.Module, dropout: float = 0.0):
    super().__init__()
    self.norm1 = RMSNorm(d_model)
    self.attn = attn
    self.norm2 = RMSNorm(d_model)
    self.mlp = mlp
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
    x = x + self.dropout(self.attn(self.norm1(x), attn_mask=attn_mask))
    mlp_out = self.mlp(self.norm2(x))
    if isinstance(mlp_out, tuple):    # MoE layers return (output, aux_load_balance_loss)
      mlp_out, aux_loss = mlp_out
    else:
      aux_loss = None
    x = x + self.dropout(mlp_out)
    return x, aux_loss


class GPTModel(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, ctx: int, n_layers: int, d_ff: int,
               n_heads: int, dropout: float, arch: str = "dense",
               d_latent: int = 0, d_rope: int = 32,
               n_routed_experts: int = 8, n_shared_experts: int = 1, moe_top_k: int = 2,
               rope_theta: float = 10000.0):
    super().__init__()
    self.ctx = ctx
    self.arch = arch
    self.tok_emb = nn.Embedding(vocab_size, d_model)

    self.blocks = nn.ModuleList()
    for _ in range(n_layers):
      if arch == "deepseek":
        dl = d_latent or max(d_model // 8, 64)
        attn = MultiHeadLatentAttention(d_model, n_heads, d_latent=dl, d_rope=d_rope, rope_theta=rope_theta)
        mlp = DeepSeekMoE(d_model, d_ff, n_routed_experts=n_routed_experts,
                           n_shared_experts=n_shared_experts, top_k=moe_top_k)
      elif arch == "dense":
        attn = CausalSelfAttention(d_model, n_heads, rope_theta=rope_theta)
        mlp = MLP(d_model, d_ff)
      else:
        raise ValueError(f"unknown arch {arch!r}, expected 'dense' or 'deepseek'")
      self.blocks.append(Block(d_model, attn, mlp, dropout=dropout))

    self.norm = RMSNorm(d_model)
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    self.apply(self._init)              # fix: init BEFORE tying so tying isn't overwritten
    self.lm_head.weight = self.tok_emb.weight

  def _init(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Embedding):
      nn.init.normal_(m.weight, mean=0, std=0.02)

  def forward(self, idx, attn_mask: Optional[torch.Tensor] = None):
    idx = idx.to(self.tok_emb.weight.device)
    B, T = idx.shape
    assert T <= self.ctx, (T, self.ctx)
    x = self.tok_emb(idx)
    aux_losses: List[torch.Tensor] = []
    for blk in self.blocks:
      x, aux = blk(x, attn_mask=attn_mask)
      if aux is not None:
        aux_losses.append(aux)
    x = self.norm(x)
    logits = self.lm_head(x)
    aux_loss = torch.stack(aux_losses).sum() if aux_losses else None
    return logits, aux_loss
