"""Rotary embedding conventions.

Covers both layouts -- rotate-half (DeepSeek-V3.2) and interleaved (GPT-J / GLM-5)
-- against an independent closed-form reference, verifies the defining
relative-position property of RoPE for each, and proves that the indexer path
actually honors :attr:`DSAConfig.indexer_rope_interleave` end to end.
"""

from __future__ import annotations

import pytest
import torch

from lagged_indexer_dsa import DSAConfig, DSAModel
from lagged_indexer_dsa.attention_core import indexer_logits
from lagged_indexer_dsa.rope import apply_rotary, build_rope_cache

THETA = 10_000.0


def _reference_rotary(
    x: torch.Tensor, positions: torch.Tensor, theta: float, interleave: bool
) -> torch.Tensor:
    """Closed-form RoPE via explicit per-pair 2x2 rotations (float64).

    Independent of the production cache/rotate implementation: each rotated pair
    ``(a, b)`` becomes ``(a*cos - b*sin, a*sin + b*cos)``. The pairing differs by
    convention: rotate-half pairs ``(x_i, x_{i+half})``; interleaved pairs
    ``(x_{2i}, x_{2i+1})``. Both pairs use frequency ``i``.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float64) / half))
    angles = positions.to(torch.float64)[:, None] * inv_freq[None, :]  # [S, half]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    xd = x.to(torch.float64)
    out = torch.empty_like(xd)
    if interleave:
        a, b = xd[..., 0::2], xd[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = xd[..., :half], xd[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:] = a * sin + b * cos
    return out


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("head_dim", [8, 16, 64])
def test_apply_rotary_matches_closed_form(interleave: bool, head_dim: int) -> None:
    torch.manual_seed(0)
    seq_len = 12
    x = torch.randn(2, 3, seq_len, head_dim, dtype=torch.float64)
    cos, sin = build_rope_cache(
        seq_len, head_dim, THETA, torch.device("cpu"), torch.float64, interleave=interleave
    )
    got = apply_rotary(x, cos, sin, interleave=interleave)
    expected = _reference_rotary(x, torch.arange(seq_len), THETA, interleave)
    # build_rope_cache computes the trig in float32 then casts, so the cache
    # carries ~1e-7 precision; a convention error would be O(1), not ~1e-6.
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)


def test_conventions_differ() -> None:
    """The two layouts must produce different rotations for the same input."""
    torch.manual_seed(1)
    seq_len, head_dim = 10, 16
    x = torch.randn(1, 2, seq_len, head_dim, dtype=torch.float64)
    rh_cos, rh_sin = build_rope_cache(
        seq_len, head_dim, THETA, torch.device("cpu"), torch.float64, interleave=False
    )
    il_cos, il_sin = build_rope_cache(
        seq_len, head_dim, THETA, torch.device("cpu"), torch.float64, interleave=True
    )
    rotate_half = apply_rotary(x, rh_cos, rh_sin, interleave=False)
    interleaved = apply_rotary(x, il_cos, il_sin, interleave=True)
    # Position 0 is the identity rotation under both, so compare position >= 1.
    assert not torch.allclose(rotate_half[..., 1:, :], interleaved[..., 1:, :], atol=1e-6)


def _float64_cache(max_pos: int, head_dim: int, interleave: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-precision (cos, sin) cache mirroring build_rope_cache's layout.

    build_rope_cache computes its trig in float32 (and casts), which is fine for
    the production indexer but breaks the *exact* relative-position algebra at
    large positions; this test isolates apply_rotary's rotation math instead.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) / half))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float64), inv_freq)
    emb = freqs.repeat_interleave(2, dim=-1) if interleave else torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("interleave", [False, True])
def test_relative_position_invariance(interleave: bool) -> None:
    """<R(q, m), R(k, n)> depends only on the offset (m - n) -- the property
    that makes RoPE a relative encoding. Verified for several (m, n) pairs."""
    torch.manual_seed(2)
    head_dim, max_pos = 16, 32
    q = torch.randn(head_dim, dtype=torch.float64)
    k = torch.randn(head_dim, dtype=torch.float64)
    cos, sin = _float64_cache(max_pos, head_dim, interleave)

    def rotated_dot(m: int, n: int) -> torch.Tensor:
        qm = apply_rotary(q.view(1, head_dim), cos[m : m + 1], sin[m : m + 1], interleave=interleave)
        kn = apply_rotary(k.view(1, head_dim), cos[n : n + 1], sin[n : n + 1], interleave=interleave)
        return (qm * kn).sum()

    for offset in (0, 1, 5, 13):
        base = rotated_dot(offset, 0)
        for shift in (2, 7, 18):
            assert torch.allclose(rotated_dot(offset + shift, shift), base, atol=1e-10, rtol=0)


def _tiny_interleave_config(interleave: bool) -> DSAConfig:
    return DSAConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=48,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=24,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        intermediate_size=256,
        first_k_dense_replace=0,
        indexer_rope_interleave=interleave,
    )


def test_indexer_honors_interleave_flag() -> None:
    """End-to-end wiring: with identical weights and input, flipping the config
    flag changes the indexer logits, proving the flag is actually consumed."""
    seq_len = 16

    def indexer_logits_for(interleave: bool) -> torch.Tensor:
        torch.manual_seed(7)  # identical weight init for both configs
        model = DSAModel(_tiny_interleave_config(interleave)).eval().double()
        torch.manual_seed(123)
        x_norm = torch.randn(1, seq_len, model.config.hidden_size, dtype=torch.float64)
        rope = model._build_rope(seq_len, x_norm.device, x_norm.dtype)
        layer = model.layers[0]
        with torch.no_grad():
            c_q = layer.q_latent(x_norm)
            index_q, index_k, weights = layer._indexer_qkw(x_norm, c_q, rope)
            return indexer_logits(index_q, index_k, weights)

    rotate_half = indexer_logits_for(interleave=False)
    interleaved = indexer_logits_for(interleave=True)
    assert rotate_half.shape == interleaved.shape
    assert not torch.allclose(rotate_half, interleaved, atol=1e-6)
