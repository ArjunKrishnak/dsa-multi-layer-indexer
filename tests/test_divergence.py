"""Tests for the divergence harness.

The reusable Phase-4 piece is the index-overlap metric, so we validate it
directly on synthetic index sets with known overlap. With random, independent
per-layer indexers the full-model overlap sits near chance and is not meant to be
representative; real cross-layer overlap is measured on a pretrained checkpoint in
Phase 4 using these same helpers.
"""

from __future__ import annotations

import pytest
import torch

from lagged_indexer_dsa import (
    ColdStartPolicy,
    DSAConfig,
    DSAModel,
    compute_divergence,
    cross_layer_overlap,
    jaccard,
)


def _config() -> DSAConfig:
    return DSAConfig(
        hidden_size=128,
        num_hidden_layers=6,
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
    )


def test_jaccard_bounds_and_extremes() -> None:
    seq_len_kv = 10
    idx = torch.tensor([[[0, 1, 2, 3]]])  # [1, 1, 4]
    # Identical sets -> overlap 1.
    assert torch.allclose(jaccard(idx, idx, seq_len_kv), torch.ones(1, 1))
    # Disjoint sets -> overlap 0.
    other = torch.tensor([[[4, 5, 6, 7]]])
    assert torch.allclose(jaccard(idx, other, seq_len_kv), torch.zeros(1, 1))


def _share(previous: list[int], shared: int, fresh_start: int, k: int) -> list[int]:
    """A length-``k`` index list sharing exactly ``shared`` entries with ``previous``.

    The remaining ``k - shared`` entries are drawn from a disjoint high range.
    """
    kept = previous[:shared]
    fresh = list(range(fresh_start, fresh_start + (k - shared)))
    return kept + fresh


def test_cross_layer_overlap_recovers_known_overlap() -> None:
    seq_len_kv = 1000
    k = 10
    l0 = list(range(0, k))
    l1 = _share(l0, shared=2, fresh_start=100, k=k)  # disjoint fresh range
    l2 = _share(l1, shared=8, fresh_start=200, k=k)

    def as_tensor(idx: list[int]) -> torch.Tensor:
        return torch.tensor([[idx]], dtype=torch.int64)

    overlaps = cross_layer_overlap([as_tensor(l0), as_tensor(l1), as_tensor(l2)], seq_len_kv)
    # Jaccard with disjoint replacement is shared / (2k - shared).
    assert overlaps[0] == pytest.approx(2 / (2 * k - 2), rel=1e-5)
    assert overlaps[1] == pytest.approx(8 / (2 * k - 8), rel=1e-5)
    assert overlaps[1] > overlaps[0]


def test_divergence_report_is_well_formed() -> None:
    torch.manual_seed(0)
    config = _config()
    model = DSAModel(config).eval()
    inputs = torch.randn(2, 24, config.hidden_size)
    report = compute_divergence(model, inputs, ColdStartPolicy.DENSE)

    assert len(report.cross_layer_overlap) == config.num_hidden_layers - 1
    assert all(0.0 <= ov <= 1.0 for ov in report.cross_layer_overlap)
    assert report.baseline_vs_lagged.mse >= 0.0
    assert -1.0 <= report.baseline_vs_lagged.mean_cosine <= 1.0
    # Lagging changes the output, so it must not be a perfect match.
    assert report.baseline_vs_lagged.mse > 0.0
