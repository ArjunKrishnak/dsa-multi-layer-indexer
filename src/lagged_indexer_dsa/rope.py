"""Rotary position embeddings.

Two layouts are supported, selected by an ``interleave`` flag that must match
between the cache builder and the application:

* **non-interleaved** ("rotate-half", DeepSeek-V3.2): the head dim is split into
  two contiguous halves ``[a | b]`` and rotated as ``[-b | a]``; the ``(cos, sin)``
  cache repeats each frequency across the two halves (``cat((freqs, freqs))``).
* **interleaved** (GPT-J / GLM-5): consecutive even/odd elements form the rotated
  pairs ``(x_{2i}, x_{2i+1})``; the cache repeats each frequency per pair
  (``repeat_interleave(freqs, 2)``).

The two conventions are *not* interchangeable: a cache built for one layout
produces wrong results if applied with the other, so the same ``interleave``
value must be passed to :func:`build_rope_cache` and :func:`apply_rotary`. The
flag is plumbed from :attr:`DSAConfig.indexer_rope_interleave` for the indexer
path (GLM-5 uses the interleaved layout; DeepSeek uses rotate-half).
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
    interleave: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` caches of shape ``[seq_len, head_dim]``.

    ``interleave`` selects the frequency layout and must match the value passed to
    :func:`apply_rotary`.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, half]
    if interleave:
        # Pair (2i, 2i+1) shares frequency i.
        emb = freqs.repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    else:
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive even/odd pairs: ``(a, b) -> (-b, a)`` per pair."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleave: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape ``[..., seq_len, head_dim]``.

    ``cos``/``sin`` have shape ``[seq_len, head_dim]`` and are broadcast over the
    leading dimensions. ``interleave`` must match the value used to build the
    cache (see module docstring).
    """
    seq_len = x.shape[-2]
    if cos.shape[-2] < seq_len:
        raise ValueError(f"rope cache too short: have {cos.shape[-2]}, need {seq_len}")
    cos_b = cos[:seq_len].to(x.dtype)
    sin_b = sin[:seq_len].to(x.dtype)
    rotate = _rotate_interleaved if interleave else _rotate_half
    return x * cos_b + rotate(x) * sin_b
