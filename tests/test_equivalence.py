"""Phase 1 gate: the lagged-fused schedule is numerically equivalent to the
lagged-reference schedule, and the lagged schedule preserves causality.

Equivalence proves the reformulation that the fused kernel will rely on -- that
computing the attention (over the previous layer's indices) and the next layer's
indexer from a single shared read of the layer input gives the same result as
computing them as two independent passes.
"""

from __future__ import annotations

import pytest
import torch

from lagged_indexer_dsa import ColdStartPolicy, DSAConfig, DSAModel, ScheduleMode


def _tiny_config() -> DSAConfig:
    return DSAConfig(
        hidden_size=128,
        num_hidden_layers=5,
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


def _build(seed: int) -> tuple[DSAModel, torch.Tensor]:
    torch.manual_seed(seed)
    config = _tiny_config()
    model = DSAModel(config).eval().double()
    inputs = torch.randn(2, 20, config.hidden_size, dtype=torch.float64)
    return model, inputs


@pytest.mark.parametrize("cold_start", [ColdStartPolicy.DENSE, ColdStartPolicy.EAGER])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_fused_equals_reference(cold_start: ColdStartPolicy, seed: int) -> None:
    model, inputs = _build(seed)
    with torch.no_grad():
        ref = model(inputs, ScheduleMode.LAGGED_REFERENCE, cold_start, record=True)
        fused = model(inputs, ScheduleMode.LAGGED_FUSED, cold_start, record=True)

    assert torch.allclose(ref.final_hidden, fused.final_hidden, atol=1e-10, rtol=0)

    # The index sets used and produced must be identical layer by layer.
    assert len(ref.indices_produced) == len(fused.indices_produced)
    for ref_idx, fused_idx in zip(ref.indices_produced, fused.indices_produced):
        if ref_idx is None:
            assert fused_idx is None
        else:
            assert torch.equal(ref_idx, fused_idx)
    for ref_used, fused_used in zip(ref.indices_used, fused.indices_used):
        if ref_used is None:
            assert fused_used is None
        else:
            assert torch.equal(ref_used, fused_used)


@pytest.mark.parametrize("cold_start", [ColdStartPolicy.DENSE, ColdStartPolicy.EAGER])
def test_lagged_preserves_causality(cold_start: ColdStartPolicy) -> None:
    model, inputs = _build(0)
    seq_len = inputs.shape[1]
    with torch.no_grad():
        trace = model(inputs, ScheduleMode.LAGGED_FUSED, cold_start, record=True)

    positions = torch.arange(seq_len).view(1, seq_len, 1)
    for used in trace.indices_used:
        if used is None:  # dense cold-start layer
            continue
        non_sentinel = used < seq_len
        assert torch.all(used[non_sentinel] <= positions.expand_as(used)[non_sentinel])


def test_schedules_actually_differ() -> None:
    """Sanity: baseline and lagged are not accidentally the same computation."""
    model, inputs = _build(0)
    with torch.no_grad():
        baseline = model(inputs, ScheduleMode.BASELINE)
        lagged = model(inputs, ScheduleMode.LAGGED_FUSED, ColdStartPolicy.DENSE)
    assert not torch.allclose(baseline, lagged, atol=1e-6)
