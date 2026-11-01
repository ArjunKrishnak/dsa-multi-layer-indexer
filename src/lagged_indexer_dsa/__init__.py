"""Multi-layer Indexer (MLI) for DeepSeek Sparse Attention.

Top-level exports are the CPU-only pure-PyTorch reference: a faithful DSA layer
and a stack runnable under three schedules (``baseline``, ``lagged_reference``,
``lagged_fused``), plus the cross-layer divergence harness. Distillation lives in
:mod:`lagged_indexer_dsa.distill`; the GPU checkpoint loader and reference
forward live in :mod:`lagged_indexer_dsa.inference`. The serving implementation
is the SGLang overlay in ``sglang_overlay/``.
"""

from __future__ import annotations

from .config import (
    GLM5_1_CONFIG,
    GLM5_CONFIG,
    SMALL_CONFIG,
    AttentionType,
    ColdStartPolicy,
    DSAConfig,
    ScheduleMode,
    dense_prefix_attention_types,
    named_config,
)
from .divergence import (
    DivergenceReport,
    OutputDivergence,
    compute_divergence,
    cross_layer_overlap,
    jaccard,
    output_divergence,
)
from .dsa_layer import DSALayer, RopeCaches
from .model import DSAModel, ForwardTrace

__all__ = [
    "GLM5_1_CONFIG",
    "GLM5_CONFIG",
    "SMALL_CONFIG",
    "AttentionType",
    "ColdStartPolicy",
    "DSAConfig",
    "ScheduleMode",
    "dense_prefix_attention_types",
    "named_config",
    "DSALayer",
    "RopeCaches",
    "DSAModel",
    "ForwardTrace",
    "DivergenceReport",
    "OutputDivergence",
    "compute_divergence",
    "cross_layer_overlap",
    "jaccard",
    "output_divergence",
]
