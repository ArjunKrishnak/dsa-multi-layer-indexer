"""A single DeepSeek Sparse Attention transformer block (pure PyTorch).

The attention is Multi-head Latent Attention in the absorbed MQA form (matching
the TileLang ``sparse_mla_fwd`` reference), preceded by a lightning indexer that
selects the sparse key set. The block is pre-norm with a SwiGLU MLP so hidden
states evolve meaningfully across layers (needed for the divergence study).

Key structural point for the lag idea: both the attention queries and the
indexer queries are derived from the *same* query latent
``c_q = q_a_layernorm(q_a_proj(x_norm))``. The fused path computes ``c_q`` once
and feeds both consumers; the reference path computes it independently for each.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention_core import (
    apply_causal_mask,
    dense_mla_core,
    indexer_logits,
    select_topk,
    sparse_mla_core,
)
from .config import DSAConfig
from .rope import apply_rotary


@dataclass
class RopeCaches:
    """Precomputed rotary caches shared across layers for one forward pass."""

    mla_cos: torch.Tensor
    mla_sin: torch.Tensor
    idx_cos: torch.Tensor
    idx_sin: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


class DSALayer(nn.Module):
    def __init__(self, config: DSAConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        h = c.num_attention_heads

        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.ffn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

        # MLA query path (low-rank).
        self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, h * (c.qk_nope_head_dim + c.qk_rope_head_dim), bias=False)

        # MLA key/value path (low-rank latent + decoupled rope key).
        self.kv_a_proj = nn.Linear(c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c.rms_norm_eps)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, h * (c.qk_nope_head_dim + c.v_head_dim), bias=False)
        self.o_proj = nn.Linear(h * c.v_head_dim, c.hidden_size, bias=False)

        # Lightning indexer.
        self.index_q_proj = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
        self.index_k_proj = nn.Linear(c.hidden_size, c.index_head_dim, bias=False)
        self.index_weights_proj = nn.Linear(c.q_lora_rank, c.index_n_heads, bias=False)

        # SwiGLU MLP.
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    # --- shared query latent -------------------------------------------------
    def q_latent(self, x_norm: torch.Tensor) -> torch.Tensor:
        """``c_q = q_a_layernorm(q_a_proj(x_norm))`` -- shared by attn and indexer."""
        return self.q_a_layernorm(self.q_a_proj(x_norm))

    # --- MLA projections ------------------------------------------------------
    def _mla_qkv(
        self, x_norm: torch.Tensor, c_q: torch.Tensor, rope: RopeCaches
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        b, s, _ = x_norm.shape
        h = c.num_attention_heads

        q = self.q_b_proj(c_q).view(b, s, h, c.qk_nope_head_dim + c.qk_rope_head_dim)
        q_nope, q_rope = torch.split(q, [c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary(q_rope.transpose(1, 2), rope.mla_cos, rope.mla_sin).transpose(1, 2)

        kv_a = self.kv_a_proj(x_norm)
        c_kv, k_rope = torch.split(kv_a, [c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        k_rope = apply_rotary(k_rope.unsqueeze(1), rope.mla_cos, rope.mla_sin).squeeze(1)

        # Absorb the K up-projection into the query: q_nope @ W_UK^T -> latent space.
        kv_b_w = self.kv_b_proj.weight.view(h, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        w_uk = kv_b_w[:, : c.qk_nope_head_dim, :]  # [H, D_nope, L_kv]
        q_absorbed = torch.einsum("bshd,hdl->bshl", q_nope, w_uk)  # [B, S, H, L_kv]

        q = torch.cat([q_absorbed, q_rope], dim=-1)  # [B, S, H, D_qk]
        kv = torch.cat([c_kv, k_rope], dim=-1)  # [B, S, D_qk]
        return q, kv

    def _project_output(self, o_latent: torch.Tensor) -> torch.Tensor:
        """Apply W_UV (latent->value) and the output projection."""
        c = self.config
        h = c.num_attention_heads
        kv_b_w = self.kv_b_proj.weight.view(h, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        w_uv = kv_b_w[:, c.qk_nope_head_dim :, :]  # [H, D_v, L_kv]
        o = torch.einsum("bshl,hdl->bshd", o_latent, w_uv)  # [B, S, H, D_v]
        b, s = o.shape[0], o.shape[1]
        return self.o_proj(o.reshape(b, s, h * c.v_head_dim))

    # --- indexer --------------------------------------------------------------
    def _indexer_qkw(
        self, x_norm: torch.Tensor, c_q: torch.Tensor, rope: RopeCaches
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        b, s, _ = x_norm.shape
        interleave = c.indexer_rope_interleave
        index_q = self.index_q_proj(c_q).view(b, s, c.index_n_heads, c.index_head_dim)
        index_q = apply_rotary(
            index_q.transpose(1, 2), rope.idx_cos, rope.idx_sin, interleave=interleave
        ).transpose(1, 2)
        index_k = self.index_k_proj(x_norm)  # [B, S, D_idx]
        index_k = apply_rotary(
            index_k.unsqueeze(1), rope.idx_cos, rope.idx_sin, interleave=interleave
        ).squeeze(1)
        weights = self.index_weights_proj(c_q)  # [B, S, H_idx]
        return index_q, index_k, weights

    def indexer_indices(
        self, x_norm: torch.Tensor, rope: RopeCaches, c_q: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the indexer's top-k selected key indices for each query."""
        if c_q is None:
            c_q = self.q_latent(x_norm)
        index_q, index_k, weights = self._indexer_qkw(x_norm, c_q, rope)
        logits = apply_causal_mask(indexer_logits(index_q, index_k, weights))
        return select_topk(logits, self.config.index_topk)

    # --- attention ------------------------------------------------------------
    def attention_output(
        self,
        x_norm: torch.Tensor,
        rope: RopeCaches,
        indices: torch.Tensor | None,
        c_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention hidden output. ``indices=None`` -> dense causal (cold start)."""
        if c_q is None:
            c_q = self.q_latent(x_norm)
        q, kv = self._mla_qkv(x_norm, c_q, rope)
        if indices is None:
            o_latent = dense_mla_core(q, kv, self.config.kv_lora_rank)
        else:
            o_latent = sparse_mla_core(q, kv, indices, self.config.kv_lora_rank)
        return self._project_output(o_latent)

    def fused_attention_and_indices(
        self,
        x_norm: torch.Tensor,
        rope: RopeCaches,
        attn_indices: torch.Tensor | None,
        produce_indexer: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single shared read of ``x_norm``: attention output + next index set.

        Pure-Python analogue of the fused kernel -- ``c_q`` is computed once and
        used for both the attention queries and the indexer queries.

        ``produce_indexer=False`` skips the indexer (returns ``None`` for the
        index set): used by DSA layers whose successor is not a DSA layer, and by
        the final layer, where the lagged indexer would have no consumer.
        """
        c_q = self.q_latent(x_norm)
        attn_hidden = self.attention_output(x_norm, rope, attn_indices, c_q=c_q)
        next_indices = self.indexer_indices(x_norm, rope, c_q=c_q) if produce_indexer else None
        return attn_hidden, next_indices

    # --- MLP ------------------------------------------------------------------
    def mlp(self, h_norm: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(h_norm)) * self.up_proj(h_norm))
