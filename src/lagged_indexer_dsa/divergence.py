"""Divergence metrics for same-layer vs one-layer-lagged indexing.

Two questions this quantifies:

1. How similar are the index sets of adjacent layers? -- :func:`cross_layer_overlap`
   computes the per-query Jaccard overlap between layer ``l`` and layer ``l-1``'s
   selected indices. High overlap means the one-layer lag is a small perturbation.
2. How different is the model output? -- :func:`output_divergence` compares final
   hidden states between the baseline and a lagged schedule.

The index-overlap helpers take raw index tensors, so they are reused unchanged in
Phase 4 on index sets captured from a pretrained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ColdStartPolicy, ScheduleMode
from .model import DSAModel


def _membership(indices: torch.Tensor, seq_len_kv: int) -> torch.Tensor:
    """Boolean ``[B, S, S_kv]`` membership mask (sentinel/out-of-range dropped)."""
    batch, seq_len, _ = indices.shape
    mask = torch.zeros(batch, seq_len, seq_len_kv + 1, dtype=torch.bool, device=indices.device)
    mask.scatter_(2, indices.clamp(max=seq_len_kv), True)
    return mask[..., :seq_len_kv]


def jaccard(idx_a: torch.Tensor, idx_b: torch.Tensor, seq_len_kv: int) -> torch.Tensor:
    """Per-query Jaccard overlap of two index sets. Returns ``[B, S]`` in [0, 1]."""
    a = _membership(idx_a, seq_len_kv)
    b = _membership(idx_b, seq_len_kv)
    inter = (a & b).sum(dim=-1).float()
    union = (a | b).sum(dim=-1).float()
    # Queries whose union is empty (no causal tokens, e.g. position 0 edge cases)
    # are defined as perfectly overlapping.
    return torch.where(union > 0, inter / union, torch.ones_like(union))


def cross_layer_overlap(
    produced_indices: list[torch.Tensor | None], seq_len_kv: int
) -> list[float]:
    """Mean Jaccard overlap between consecutive layers' produced index sets.

    Only adjacent layer pairs where *both* layers ran an indexer contribute; pairs
    involving a non-producing layer (full-attention or non-producing lagged layer,
    recorded as ``None``) are skipped. Entry order follows the layer pairs.
    """
    overlaps: list[float] = []
    for layer_idx in range(1, len(produced_indices)):
        current = produced_indices[layer_idx]
        previous = produced_indices[layer_idx - 1]
        if current is None or previous is None:
            continue
        ov = jaccard(current, previous, seq_len_kv)
        overlaps.append(ov.mean().item())
    return overlaps


@dataclass
class OutputDivergence:
    mse: float
    relative_l2: float
    mean_cosine: float


def output_divergence(hidden_a: torch.Tensor, hidden_b: torch.Tensor) -> OutputDivergence:
    """Compare two ``[B, S, H]`` hidden-state tensors."""
    diff = hidden_a - hidden_b
    mse = diff.pow(2).mean().item()
    relative_l2 = (diff.norm() / hidden_a.norm().clamp_min(1e-12)).item()
    cosine = torch.nn.functional.cosine_similarity(hidden_a, hidden_b, dim=-1).mean().item()
    return OutputDivergence(mse=mse, relative_l2=relative_l2, mean_cosine=cosine)


@dataclass
class DivergenceReport:
    cross_layer_overlap: list[float]
    mean_cross_layer_overlap: float
    baseline_vs_lagged: OutputDivergence


def compute_divergence(
    model: DSAModel,
    inputs_embeds: torch.Tensor,
    cold_start: ColdStartPolicy = ColdStartPolicy.DENSE,
) -> DivergenceReport:
    """Run baseline and lagged schedules; report overlap and output divergence."""
    seq_len_kv = inputs_embeds.shape[1]
    with torch.no_grad():
        baseline = model(inputs_embeds, ScheduleMode.BASELINE, cold_start, record=True)
        lagged = model(inputs_embeds, ScheduleMode.LAGGED_FUSED, cold_start, record=True)

    overlaps = cross_layer_overlap(baseline.indices_produced, seq_len_kv)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0
    out_div = output_divergence(baseline.final_hidden, lagged.final_hidden)
    return DivergenceReport(
        cross_layer_overlap=overlaps,
        mean_cross_layer_overlap=mean_overlap,
        baseline_vs_lagged=out_div,
    )
