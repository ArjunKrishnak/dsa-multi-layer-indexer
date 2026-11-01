"""Core DSA tensor ops in plain PyTorch (CPU-friendly, fp32).

These mirror the semantics of the TileLang reference kernels:

* :func:`indexer_logits` matches ``ref_fp8_mqa_logits`` (lightning indexer:
  ReLU'd MQA dot products, weighted head reduction).
* :func:`sparse_mla_core` matches ``ref_sparse_mla_fwd_interface`` (sparse MLA
  in the absorbed MQA form: scores over the selected + causal set, softmax,
  value-weighted sum).

Both are validated against ports of those reference functions in the tests.
The implementations are intentionally simple and dense (no gather/paging); the
high-performance fused kernel is Phase 3 work.
"""

from __future__ import annotations

import torch

# Sentinel used for padding selected-index tensors. Equal to the kv length, it
# is always out of range and therefore masked out by the causal check, matching
# the TileLang reference which pads with ``seq_len_kv``.
def pad_index(seq_len_kv: int) -> int:
    return seq_len_kv


def indexer_logits(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Lightning-indexer relevance scores.

    Args:
        index_q: ``[B, S, H_idx, D_idx]`` indexer queries.
        index_k: ``[B, S_kv, D_idx]`` shared (MQA) indexer keys.
        weights: ``[B, S, H_idx]`` per-head learned weights.

    Returns:
        ``[B, S, S_kv]`` index scores ``I_{t,s} = sum_h w_{t,h} ReLU(q_{t,h}.k_s)``.
    """
    if index_q.dim() != 4:
        raise ValueError(f"index_q must be [B,S,H,D], got {tuple(index_q.shape)}")
    if index_k.dim() != 3:
        raise ValueError(f"index_k must be [B,S_kv,D], got {tuple(index_k.shape)}")
    score = torch.einsum("bshd,bnd->bshn", index_q, index_k)
    score = torch.relu(score)
    logits = (score * weights.unsqueeze(-1)).sum(dim=2)
    return logits


def apply_causal_mask(logits: torch.Tensor) -> torch.Tensor:
    """Mask out non-causal positions (key index > query index) with ``-inf``.

    Assumes the query and key sequences are aligned (self-attention), i.e. query
    position ``s`` may attend to key positions ``0..s``.
    """
    seq_len = logits.shape[1]
    seq_len_kv = logits.shape[2]
    q_pos = torch.arange(seq_len, device=logits.device).unsqueeze(1)
    k_pos = torch.arange(seq_len_kv, device=logits.device).unsqueeze(0)
    causal = k_pos <= q_pos  # [S, S_kv]
    return logits.masked_fill(~causal.unsqueeze(0), float("-inf"))


def select_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Select the top-``k`` key indices per query from causal-masked logits.

    Args:
        logits: ``[B, S, S_kv]`` index scores (already causal-masked to ``-inf``).
        topk: number of indices to select per query.

    Returns:
        ``[B, S, topk]`` int64 indices. Slots without a finite score (fewer than
        ``topk`` causal tokens available) are filled with the out-of-range
        sentinel ``S_kv`` so they are masked downstream.
    """
    seq_len_kv = logits.shape[2]
    sentinel = pad_index(seq_len_kv)
    k_eff = min(topk, seq_len_kv)
    values, indices = torch.topk(logits, k_eff, dim=-1)
    indices = indices.masked_fill(~torch.isfinite(values), sentinel)
    if k_eff < topk:
        pad = torch.full(
            (logits.shape[0], logits.shape[1], topk - k_eff),
            sentinel,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices = torch.cat([indices, pad], dim=-1)
    return indices


def _selection_mask(indices: torch.Tensor, seq_len_kv: int) -> torch.Tensor:
    """Boolean ``[B, S, S_kv]`` membership mask from selected ``indices``.

    Mirrors the scatter trick in ``ref_sparse_mla_fwd_interface``: scatter into
    ``S_kv + 1`` columns (the extra column absorbs the sentinel) then drop it.
    """
    batch, seq_len, _ = indices.shape
    mask = torch.zeros(batch, seq_len, seq_len_kv + 1, dtype=torch.bool, device=indices.device)
    mask.scatter_(2, indices.clamp(max=seq_len_kv), True)
    return mask[..., :seq_len_kv]


def _causal_kv_mask(seq_len: int, seq_len_kv: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(seq_len_kv, device=device).unsqueeze(0)
    return k_pos <= q_pos  # [S, S_kv]


def _attend(q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor, sm_scale: float, value_dim: int) -> torch.Tensor:
    """Softmax attention given an explicit boolean key mask ``[B, S, S_kv]``."""
    score = torch.einsum("bshd,bnd->bshn", q, kv)  # [B, S, H, S_kv]
    score = score.masked_fill(~mask.unsqueeze(2), float("-inf")) * sm_scale
    probs = torch.softmax(score, dim=-1)
    value = kv[..., :value_dim]
    return torch.einsum("bshn,bnd->bshd", probs, value)  # [B, S, H, value_dim]


def sparse_mla_core(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over the selected + causal key set.

    Args:
        q: ``[B, S, H, D_qk]`` absorbed queries (latent + rope dims concatenated).
        kv: ``[B, S_kv, D_qk]`` shared latent keys (value = first ``value_dim``).
        indices: ``[B, S, topk]`` selected key indices (sentinel = ``S_kv``).
        value_dim: latent value dimension (``kv_lora_rank``).
        sm_scale: softmax scale; defaults to ``D_qk ** -0.5``.

    Returns:
        ``[B, S, H, value_dim]`` latent attention output.
    """
    seq_len_kv = kv.shape[1]
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    selection = _selection_mask(indices, seq_len_kv)
    causal = _causal_kv_mask(q.shape[1], seq_len_kv, q.device).unsqueeze(0)
    mask = selection & causal
    return _attend(q, kv, mask, sm_scale, value_dim)


def dense_mla_core(
    q: torch.Tensor,
    kv: torch.Tensor,
    value_dim: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Dense causal MLA attention (used for the layer-0 cold start)."""
    seq_len_kv = kv.shape[1]
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    mask = _causal_kv_mask(q.shape[1], seq_len_kv, q.device).unsqueeze(0)
    return _attend(q, kv, mask, sm_scale, value_dim)
