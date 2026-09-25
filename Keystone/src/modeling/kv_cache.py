"""Preallocated per-layer KV cache (MD-10). Dense layers store post-RoPE k and v
(B, H, T, Dh); MLA layers store c_kv (B, T, d_latent) and post-RoPE k_rope (B, H, T, d_rope)."""
from __future__ import annotations

from typing import List

import torch


class KVCache:
    def __init__(self, model, batch: int, max_len: int, dtype: torch.dtype, device: torch.device):
        from ..foundation import bridge
        MLA = bridge.fl_model.MultiHeadLatentAttention
        self.max_len = max_len
        self.batch = batch
        self.length = 0                                                  # shared write position
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)  # real tokens per row
        self.key_valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
        self.layers: List[dict] = []
        for blk in model.blocks:
            attn = blk.attn
            if isinstance(attn, MLA):
                d_latent = attn.kv_down.out_features
                self.layers.append({
                    "kind": "mla",
                    "c_kv": torch.zeros(batch, max_len, d_latent, dtype=dtype, device=device),
                    "k_rope": torch.zeros(batch, attn.n_heads, max_len, attn.d_rope, dtype=dtype, device=device),
                })
            else:
                self.layers.append({
                    "kind": "dense",
                    "k": torch.zeros(batch, attn.n_heads, max_len, attn.head_dim, dtype=dtype, device=device),
                    "v": torch.zeros(batch, attn.n_heads, max_len, attn.head_dim, dtype=dtype, device=device),
                })

    def reserve(self, attention_mask: torch.Tensor) -> int:
        """Marks the next T key slots with this step's validity; returns the start index."""
        T = attention_mask.shape[1]
        start = self.length
        if start + T > self.max_len:
            raise ValueError(f"KV cache overflow: {start}+{T} > {self.max_len}")
        self.key_valid[:, start:start + T] = attention_mask
        self.lengths += attention_mask.sum(dim=1)
        return start

    def advance(self, T: int) -> None:
        self.length += T
