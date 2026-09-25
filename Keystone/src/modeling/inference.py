"""Inference forward with explicit positions and an optional KV cache, computed over the reused
foundation_llm module parameters without modifying them (MD-4a..MD-9, MD-11)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..foundation import bridge
from .kv_cache import KVCache
from .loading import unwrap


class InferenceModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        bridge.require()
        self.model = unwrap(model)
        self.ctx = self.model.ctx
        self._MLA = bridge.fl_model.MultiHeadLatentAttention
        self._rope: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        first = self.model.blocks[0].attn
        self.rope_theta = first.rope_theta
        self.rope_dim = first.d_rope if isinstance(first, self._MLA) else first.head_dim

    @property
    def device(self):
        return self.model.tok_emb.weight.device

    def _rope_tables(self, device):
        """MD-6: computed once per model/device with foundation_llm's build_rope_cache."""
        key = (str(device),)
        if key not in self._rope:
            self._rope[key] = bridge.fl_model.build_rope_cache(self.ctx, self.rope_dim, self.rope_theta, device=device)
        return self._rope[key]

    def _apply_rope(self, x, cos_t, sin_t):
        # x: (B, H, T, D); cos_t/sin_t: (B, T, D) gathered by position_ids
        cos = cos_t[:, None].to(x.dtype)
        sin = sin_t[:, None].to(x.dtype)
        return x * cos + bridge.fl_model.rotate_half(x) * sin

    @staticmethod
    def _build_mask(key_valid: torch.Tensor, start: int, T: int) -> torch.Tensor:
        """MD-8: (B, 1, T_q, T_k) bool. Key j visible to query i iff j <= start+i and key j is
        not padding. Each query also sees itself, so padding query rows are never all-False
        (their outputs are ignored; this only avoids NaNs from an empty softmax)."""
        Tk = start + T
        q_pos = torch.arange(start, start + T, device=key_valid.device)[:, None]
        k_pos = torch.arange(Tk, device=key_valid.device)[None, :]
        causal = k_pos <= q_pos                                           # (T, Tk)
        mask = causal[None] & key_valid[:, None, :Tk]                     # (B, T, Tk)
        mask = mask | (k_pos == q_pos)[None]
        return mask[:, None]

    def _attn_dense(self, attn, h, cos_t, sin_t, mask, cache_layer, start):
        B, T, C = h.shape
        H, Dh = attn.n_heads, attn.head_dim
        q, k, v = attn.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q = self._apply_rope(q, cos_t, sin_t)
        k = self._apply_rope(k, cos_t, sin_t)
        if cache_layer is not None:
            cache_layer["k"][:, :, start:start + T] = k.to(cache_layer["k"].dtype)
            cache_layer["v"][:, :, start:start + T] = v.to(cache_layer["v"].dtype)
            k = cache_layer["k"][:, :, :start + T].to(q.dtype)
            v = cache_layer["v"][:, :, :start + T].to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return attn.proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def _attn_mla(self, attn, h, cos_t, sin_t, mask, cache_layer, start):
        B, T, _ = h.shape
        H, Dh, Dr = attn.n_heads, attn.head_dim, attn.d_rope
        c_kv = attn.kv_norm(attn.kv_down(h))                                   # (B, T, dl)
        k_rope = self._apply_rope(attn.k_rope(h).view(B, T, H, Dr).transpose(1, 2), cos_t, sin_t)
        if cache_layer is not None:
            cache_layer["c_kv"][:, start:start + T] = c_kv.to(cache_layer["c_kv"].dtype)
            cache_layer["k_rope"][:, :, start:start + T] = k_rope.to(cache_layer["k_rope"].dtype)
            c_kv = cache_layer["c_kv"][:, :start + T].to(c_kv.dtype)
            k_rope = cache_layer["k_rope"][:, :, :start + T].to(k_rope.dtype)
        Tk = c_kv.shape[1]
        kv = attn.kv_up(c_kv).view(B, Tk, H, 2 * Dh).transpose(1, 2)
        k_content, v = kv.split(Dh, dim=-1)
        c_q = attn.q_norm(attn.q_down(h))
        q_content = attn.q_up(c_q).view(B, T, H, Dh).transpose(1, 2)
        q_rope = self._apply_rope(attn.q_rope(c_q).view(B, T, H, Dr).transpose(1, 2), cos_t, sin_t)
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope.to(k_content.dtype)], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return attn.proj(out.transpose(1, 2).contiguous().view(B, T, H * Dh))

    def forward(self, idx: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None, return_hidden: bool = False):
        m = self.model
        idx = idx.to(self.device)
        B, T = idx.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        attention_mask = attention_mask.to(idx.device).bool()
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(1) - 1).clamp_min(0)
        position_ids = position_ids.to(idx.device)
        if int(position_ids.max()) >= self.ctx:
            raise ValueError(f"position {int(position_ids.max())} >= ctx {self.ctx}")
        cos_tab, sin_tab = self._rope_tables(idx.device)
        cos_t, sin_t = cos_tab[position_ids], sin_tab[position_ids]

        if kv_cache is not None:
            start = kv_cache.reserve(attention_mask)
            key_valid = kv_cache.key_valid
        else:
            start, key_valid = 0, attention_mask
        mask = self._build_mask(key_valid, start, T)

        x = m.tok_emb(idx)
        hiddens = []
        for li, blk in enumerate(m.blocks):
            h = blk.norm1(x)
            layer_cache = kv_cache.layers[li] if kv_cache is not None else None
            if isinstance(blk.attn, self._MLA):
                a = self._attn_mla(blk.attn, h, cos_t, sin_t, mask, layer_cache, start)
            else:
                a = self._attn_dense(blk.attn, h, cos_t, sin_t, mask, layer_cache, start)
            x = x + a
            h2 = blk.norm2(x)
            if return_hidden:
                hiddens.append(h2)
            mlp_out = blk.mlp(h2)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            x = x + mlp_out
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = m.lm_head(m.norm(x))
        return (logits, hiddens) if return_hidden else logits


@torch.no_grad()
def routing_topk(moe_module, h: torch.Tensor) -> torch.Tensor:
    """MD-9: exactly DeepSeekMoE.forward's selection rule."""
    flat = h.reshape(-1, h.shape[-1])
    scores = moe_module.router(flat) + moe_module.routing_bias
    return scores.topk(moe_module.top_k, dim=-1).indices
