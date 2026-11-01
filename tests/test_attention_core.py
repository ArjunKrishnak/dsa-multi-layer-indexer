"""Validate the core DSA ops against CPU ports of the TileLang reference.

The reference functions are faithful ports of ``ref_fp8_mqa_logits`` and
``ref_sparse_mla_fwd_interface`` from
``src/lagged_indexer_dsa/tilelang_dsa/{fp8_lighting_indexer,sparse_mla_fwd}.py``,
with the CUDA/FP8/bf16 specifics removed so they run on CPU in fp32.
"""

from __future__ import annotations

import torch

from lagged_indexer_dsa.attention_core import (
    indexer_logits,
    select_topk,
    sparse_mla_core,
)


def ref_indexer_logits(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Port of ``ref_fp8_mqa_logits`` (no fp8, no cu_seqlen ranges), batched.

    q: [B, S, H, D]; k: [B, S_kv, D]; weights: [B, S, H] -> logits [B, S, S_kv].
    """
    q = q.float()
    k = k.float()
    score = torch.einsum("bmhd,bnd->bhmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(1, 2)).sum(dim=1)
    return logits


def ref_sparse_mla(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, value_dim: int) -> torch.Tensor:
    """Port of ``ref_sparse_mla_fwd_interface`` for a single kv group (CPU/fp32).

    q: [B, S, H, D_qk]; kv: [B, S_kv, D_qk]; indices: [B, S, topk] (sentinel=S_kv).
    """
    q = q.float()
    kv = kv.float()
    b, sq, h, dim_q = q.shape
    sk = kv.shape[1]
    k = kv
    v = kv[..., :value_dim]

    causal = torch.arange(sq).view(-1, 1) >= torch.arange(0, sk).view(1, -1)  # [S, S_kv]
    membership = q.new_zeros(b, sq, sk + 1, dtype=torch.bool)
    membership.scatter_(2, indices.clamp(max=sk).long(), True)
    membership = membership[..., :sk]
    mask = membership & causal.view(1, sq, sk)

    score = torch.einsum("bmhd,bnd->bhmn", q, k)  # [B, H, S, S_kv]
    sm_scale = dim_q**-0.5
    score = score.masked_fill(~mask.unsqueeze(1), float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    o = torch.einsum("bhmn,bnd->bmhd", p, v)  # [B, S, H, value_dim]
    return o


def _random_causal_indices(batch: int, seq_len: int, topk: int, seq_len_kv: int) -> torch.Tensor:
    indices = torch.full((batch, seq_len, topk), seq_len_kv, dtype=torch.int64)
    for b in range(batch):
        for t in range(seq_len):
            valid = torch.randperm(t + 1)[:topk]
            indices[b, t, : len(valid)] = valid
    return indices


def test_indexer_logits_matches_reference() -> None:
    torch.manual_seed(0)
    b, s, h, d, skv = 2, 12, 8, 16, 12
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, skv, d)
    weights = torch.randn(b, s, h)

    got = indexer_logits(q, k, weights)
    expected = ref_indexer_logits(q, k, weights)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_sparse_mla_matches_reference() -> None:
    torch.manual_seed(0)
    b, s, h, value_dim, rope = 2, 16, 8, 64, 16
    dim_qk = value_dim + rope
    topk = 6
    q = torch.randn(b, s, h, dim_qk)
    kv = torch.randn(b, s, dim_qk)
    indices = _random_causal_indices(b, s, topk, seq_len_kv=s)

    got = sparse_mla_core(q, kv, indices, value_dim=value_dim)
    expected = ref_sparse_mla(q, kv, indices, value_dim=value_dim)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_select_topk_is_causal_and_sized() -> None:
    torch.manual_seed(0)
    b, s, skv = 2, 10, 10
    logits = torch.randn(b, s, skv)
    # Causal-mask then select.
    from lagged_indexer_dsa.attention_core import apply_causal_mask

    masked = apply_causal_mask(logits)
    topk = 4
    indices = select_topk(masked, topk)
    assert indices.shape == (b, s, topk)
    # Every selected, non-sentinel index must be causal (<= query position).
    positions = torch.arange(s).view(1, s, 1)
    non_sentinel = indices < skv
    assert torch.all((indices[non_sentinel] <= positions.expand_as(indices)[non_sentinel]))
