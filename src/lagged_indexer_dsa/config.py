"""Configuration for the DeepSeek Sparse Attention (DSA) reference model.

The default dimensions mirror the released DeepSeek-V3.2-Exp checkpoint
(`DeepseekV32Config`). A deliberately small preset is also provided for fast,
CPU-only Phase 1 equivalence and divergence work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AttentionType(str, Enum):
    """Per-layer attention kind.

    FULL: ordinary dense causal attention over the whole prefix, no indexer
        needed for its own attention (DeepSeek/GLM full-attention layer; also the
        dense warm-up layers).
    DSA: DeepSeek Sparse Attention -- the indexer selects a top-k key set and the
        main attention runs only over it.
    """

    FULL = "full"
    DSA = "dsa"


def dense_prefix_attention_types(num_hidden_layers: int, num_dense: int) -> tuple[AttentionType, ...]:
    """``num_dense`` leading FULL-attention layers, then DSA for the rest.

    Mirrors the common "dense-attention prefix then sparse" layout (and the
    DeepSeek dense warm-up). With the lagged schedule the last FULL layer of the
    prefix is the one that must additionally emit the indexer that seeds the first
    DSA layer.
    """
    if not 0 <= num_dense <= num_hidden_layers:
        raise ValueError(f"num_dense must be in [0, {num_hidden_layers}], got {num_dense}")
    return tuple(
        AttentionType.FULL if i < num_dense else AttentionType.DSA for i in range(num_hidden_layers)
    )


class ColdStartPolicy(str, Enum):
    """How a DSA layer at index 0 behaves when there is no previous index set.

    Only relevant when the very first layer is a DSA layer (no predecessor to
    produce its lagged indices). A dense-attention prefix removes the need for a
    cold start: the last FULL layer's boundary indexer seeds the first DSA layer.

    DENSE: layer 0 runs ordinary dense causal attention (no sparsity).
    EAGER: layer 0 computes its own indexer and attends same-layer (no lag),
        i.e. it falls back to the baseline behaviour for that single layer.
    """

    DENSE = "dense"
    EAGER = "eager"


class ScheduleMode(str, Enum):
    """Execution schedules compared in Phase 1.

    BASELINE: standard DSA. Layer L attends over the index set that layer L's
        own indexer produces (strict intra-layer producer->consumer chain).
    LAGGED_REFERENCE: layer L attends over layer L-1's index set; the indexer
        and the attention are computed as two separate passes over the layer
        input (the "unfused" reference).
    LAGGED_FUSED: identical math to LAGGED_REFERENCE, but the attention and the
        next-layer indexer are computed from a single shared read of the layer
        input -- the pure-Python analogue of the fused kernel.
    """

    BASELINE = "baseline"
    LAGGED_REFERENCE = "lagged_reference"
    LAGGED_FUSED = "lagged_fused"


@dataclass(frozen=True)
class DSAConfig:
    """Dimensions of a DSA attention stack.

    Attributes mirror `DeepseekV32Config` names where they exist so that the
    reference implementation can later be checked against real checkpoints.
    """

    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128

    # MLA latent dimensions.
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # Lightning indexer dimensions.
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    # MLP.
    intermediate_size: int = 18432
    # Dense-FFN prefix (MoE models use a dense FFN for the first k layers). This
    # is the *FFN* distinction; it is orthogonal to attention type and the lag,
    # and is recorded only to avoid conflating it with the attention prefix.
    first_k_dense_replace: int = 3

    # Per-layer attention type. ``None`` -> every layer is DSA (the released
    # DeepSeek-V3.2-Exp / GLM-5 layout). Provide a tuple (e.g. via
    # :func:`dense_prefix_attention_types`) to model a full-attention prefix.
    attention_layer_types: tuple[AttentionType, ...] | None = field(default=None)

    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # Indexer RoPE convention. DeepSeek-V3.2 applies the half-rotation ("rotate
    # half") form; GLM-5 interleaves even/odd pairs. Latency-neutral, but it
    # changes the indexer's rotation, so it is plumbed into the indexer RoPE path
    # (see rope.py / DSALayer._indexer_qkw). Indexer-scoped: the MLA RoPE path is
    # unaffected.
    indexer_rope_interleave: bool = False

    @property
    def qk_head_dim(self) -> int:
        """Combined per-head query/key dimension in the absorbed MQA form."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def resolved_attention_types(self) -> tuple[AttentionType, ...]:
        """Per-layer attention type, defaulting to all-DSA."""
        if self.attention_layer_types is None:
            return tuple(AttentionType.DSA for _ in range(self.num_hidden_layers))
        return tuple(AttentionType(t) for t in self.attention_layer_types)

    def __post_init__(self) -> None:
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError(
                f"first_k_dense_replace must be in [0, {self.num_hidden_layers}], "
                f"got {self.first_k_dense_replace}"
            )
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for rotary embeddings")
        if self.index_head_dim % 2 != 0:
            raise ValueError("index_head_dim must be even for rotary embeddings")
        if self.attention_layer_types is not None:
            resolved = self.resolved_attention_types  # validates membership
            if len(resolved) != self.num_hidden_layers:
                raise ValueError(
                    f"attention_layer_types has length {len(resolved)}, "
                    f"expected num_hidden_layers={self.num_hidden_layers}"
                )


# GLM-5 (released layout): MLA + lightning indexer, same absorbed-MQA key/value
# dims as DeepSeek-V3.2 (qk_head_dim = 512 + 64 = 576, d_v = 512), so the sparse
# attention kernels are dimensionally unchanged. The deltas are the indexer
# (32 heads vs 64), the interleaved indexer RoPE, and rope_theta. Dims mirror the
# canonical GLM-5 HF config (hidden_size, num_hidden_layers, index_n_heads, etc.).
GLM5_CONFIG = DSAConfig(
    hidden_size=6144,
    num_hidden_layers=78,
    num_attention_heads=64,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    intermediate_size=12288,
    first_k_dense_replace=3,
    rope_theta=1_000_000.0,
    indexer_rope_interleave=True,
)

# GLM-5.1 shares GLM-5's architecture (different checkpoint/training only).
GLM5_1_CONFIG = GLM5_CONFIG

# Small preset for CPU-only Phase 1 work: faithful structure, tiny dimensions.
SMALL_CONFIG = DSAConfig(
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    q_lora_rank=96,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    index_n_heads=8,
    index_head_dim=16,
    index_topk=16,
    intermediate_size=512,
)


def named_config(name: str) -> DSAConfig:
    """Look up a model preset by name (for benchmark/CLI selection)."""
    presets = {
        "deepseek": DSAConfig(),  # DeepSeek-V3.2-Exp defaults
        "glm5": GLM5_CONFIG,
        "glm5.1": GLM5_1_CONFIG,
        "small": SMALL_CONFIG,
    }
    if name not in presets:
        raise ValueError(f"unknown config '{name}'; choose from {sorted(presets)}")
    return presets[name]
