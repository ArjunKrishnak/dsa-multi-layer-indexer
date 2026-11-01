from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from lagged_indexer_dsa.inference.modeling.kernels import act_quant, fp8_gemm, fp8_index
except ImportError:
    act_quant = fp8_gemm = fp8_index = None

from lagged_indexer_dsa.inference.modeling.fused_decode import (
    HEAD_BLOCK,
    FusedDecodeState,
    ReuseFusedState,
    fused_cascade_select,
    pad_heads,
    pad_index_operands,
    reuse_fused_slab_need,
    sink_duplicate_mask,
)
from lagged_indexer_dsa.ckv_block_store import derive_key as _ckv_derive_key
from lagged_indexer_dsa.indexer_source import (
    SourceConfig,
    fp8_dequantize,
    fp8_quantize_per_key,
    gaussian_projection,
    identity_projection,
    single_stage_topk,
    two_stage_select,
    weighted_relu_scores,
)
from lagged_indexer_dsa.tap import (
    VARIANT_NAMES,
    attend_softmax_stats,
    block_summary_scores,
    block_token_candidates,
    cascade_topk,
    entries_head_features,
    entries_head_scores,
    neighbor_candidates,
    recency_candidates,
    stripe_candidates,
    tap_logits,
    variant_local_scores,
)


world_size = 1
rank = 0
block_size = 128


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: str | None = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    # multi-model support
    gate_bias: bool = True
    indexer_rope_interleave: bool = False
    rope_type: str = "yarn"


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        if vocab_size % world_size != 0:
            raise ValueError(
                f"Vocabulary size ({vocab_size}) must be divisible by world size ({world_size})"
            )
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_fmt: str | None = None,
) -> torch.Tensor:
    """Apply a (potentially fp8-quantised) matmul ``y = x @ weight^T``.

    The helper does not handle bias terms — callers that need a bias
    add it post-hoc (see :class:`Linear.forward`,
    :class:`ColumnParallelLinear.forward`,
    :class:`RowParallelLinear.forward`). This keeps the function a
    pure matmul: the fp8 path goes through the quantised gemm kernel,
    the bf16/fp32 path goes through ``F.linear`` without bias, and
    bias addition is uniformly the caller's responsibility.

    Args:
        x: Input tensor.
        weight: Weight tensor. May be fp8-quantised; the function
            auto-detects via ``weight.dtype`` and routes through
            ``fp8_gemm`` if so.
        scale_fmt: Format of fp8 scaling factors.

    Returns:
        Output tensor without any bias term.
    """
    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    if act_quant is None or fp8_gemm is None:
        raise RuntimeError(
            "FP8 weights require CUDA kernels (act_quant, fp8_gemm). "
            "Install with: uv sync --extra cuda"
        )
    x, scale = act_quant(x, block_size, scale_fmt)
    return fp8_gemm(x, scale, weight, weight.scale)


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    dtype = torch.bfloat16
    scale_fmt: str | None = None

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # ``scale_fmt`` is set as a class attribute by
        # ``Transformer.__init__`` so that all ``Linear`` instances
        # built during a given construction share the same FP8 scale
        # format. Snapshotting it onto the instance here insulates
        # already-built modules from later mutations of the class
        # attribute (e.g. a second ``Transformer`` constructed with a
        # different ``args.scale_fmt`` in the same process).
        self.scale_fmt = Linear.scale_fmt
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
        )
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        if bias:
            # Match the bias dtype to the weight (or the surrounding
            # compute dtype when weights are fp8) so bias addition
            # doesn't upcast bf16 activations to fp32.
            bias_dtype = (
                self.weight.dtype
                if self.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else (dtype or Linear.dtype)
            )
            self.bias = nn.Parameter(torch.empty(out_features, dtype=bias_dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        y = linear(x, self.weight, self.scale_fmt)
        if self.bias is not None:
            # Force the activation dtype in case a state_dict load
            # mutated the bias dtype.
            y = (y + self.bias).type_as(x)
        return y


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by "
                f"world_size={world_size} for column-parallel linear."
            )
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.scale_fmt)
        if self.bias is not None:
            # See ``Linear.forward``: keep bias addition in the input
            # dtype so a stale fp32 bias does not upcast bf16 activations.
            y = (y + self.bias).type_as(x)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        reduce_output=True,
        dtype=None,
    ):
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by "
                f"world_size={world_size} for row-parallel linear."
            )
        self.part_in_features = in_features // world_size
        self.reduce_output = reduce_output
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight, self.scale_fmt)
        if self.reduce_output and world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min_val (float): Minimum value for the ramp function.
            max_val (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if args.rope_type != "default" and seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def _hadamard_transform_pt(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform (pure PyTorch fallback, butterfly algorithm)."""
    n = x.shape[-1]
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Hadamard fallback requires a power-of-two last dim, got {n}"
        )
    orig_shape = x.shape
    batch = x.numel() // n
    x = x.float().view(batch, n)
    h = 1
    while h < n:
        x_view = x.view(batch, n // (2 * h), 2, h)
        even = x_view[:, :, 0, :].clone()
        odd = x_view[:, :, 1, :].clone()
        x_view[:, :, 0, :] = even + odd
        x_view[:, :, 1, :] = even - odd
        h *= 2
    return (x * scale).view(orig_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    try:
        from fast_hadamard_transform import hadamard_transform  # type: ignore[import-unresolved]
    except Exception:
        # Not just ImportError: a broken CUDA extension can raise OSError /
        # RuntimeError on import, and we still want the PyTorch fallback.
        hadamard_transform = None

    original_dtype = x.dtype
    if original_dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    hidden_size = x.size(-1)
    # The fast-hadamard extension is CUDA-only; a CPU tensor must take the
    # PyTorch butterfly even when the extension imports (the import guard
    # alone does not cover CPU callers, e.g. the CPU-tested source path).
    if hadamard_transform is not None and x.is_cuda:
        y = hadamard_transform(x, scale=hidden_size**-0.5)
    else:
        y = _hadamard_transform_pt(x.float(), scale=hidden_size**-0.5)
    if y.dtype != original_dtype:
        y = y.to(original_dtype)
    return y


def refresh_stripe_spans(
    step: int,
    period: int,
    slice_width: int,
    end_pos: int,
    fresh_tail: int,
) -> list[tuple[int, int]]:
    """Key spans a temporally-striped refresh scan scores at one decode step.

    Temporal striping (track SCHED, mechanism f) amortizes a refresh
    producer's O(L) lightning-indexer scan across decode steps: each step
    scores one rotating contiguous slice of the prefix (``O(L/period)``)
    plus a fresh tail of the newest ``fresh_tail`` positions, and a running
    per-layer score cache keeps every other key's last score.  The slice
    width is FROZEN by the caller (derived from the allocated cache length,
    not the live prefix), so a key's slice index never changes and every
    cached key is re-scored exactly once per ``period`` steps.

    Args:
        step: Striped-step counter since the cache was seeded (the seed step
            itself scores the full prefix and does not come through here).
        period: Stripe period ``s`` — the number of steps a full rotation
            takes.
        slice_width: Frozen slice width ``ceil(alloc_len / period)``.
        end_pos: Current prefix length; spans never extend past it.
        fresh_tail: Newest positions scored EVERY step regardless of the
            rotation, so a freshly generated key is never stuck at its
            seed/-inf score while waiting for its slice to come around
            (``fresh_tail >= period`` closes that gap completely).

    Returns:
        Disjoint, sorted, non-empty ``[lo, hi)`` spans within
        ``[0, end_pos)``.  The rotating slice may be empty when it lies
        beyond the live prefix (only the tail is scored that step).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if slice_width < 1:
        raise ValueError(f"slice_width must be >= 1, got {slice_width}")
    if end_pos < 1:
        raise ValueError(f"end_pos must be >= 1, got {end_pos}")
    if fresh_tail < 1:
        raise ValueError(f"fresh_tail must be >= 1, got {fresh_tail}")
    slice_idx = step % period
    slice_lo = slice_idx * slice_width
    slice_hi = min(slice_lo + slice_width, end_pos)
    tail_lo = max(end_pos - fresh_tail, 0)
    spans: list[tuple[int, int]] = []
    if slice_lo < slice_hi:
        spans.append((slice_lo, slice_hi))
    if tail_lo < end_pos:
        spans.append((tail_lo, end_pos))
    spans.sort()
    # Merge overlapping/adjacent spans so each key column is scored once.
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if act_quant is None or fp8_index is None:
            raise RuntimeError(
                "Indexer requires FP8 CUDA kernels (act_quant, fp8_index) from "
                "the 'cuda' extras. Install with: uv sync --extra cuda"
            )
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = args.scale_fmt
        self.indexer_rope_interleave: bool = args.indexer_rope_interleave

        self._cache_max_bsz = args.max_batch_size
        self._cache_block_size = block_size
        # Allocate with seqlen=1; use reset_kv_cache() before each eval run.
        self.register_buffer(
            "k_cache",
            torch.zeros(
                args.max_batch_size, 1, self.head_dim, dtype=torch.float8_e4m3fn,
            ),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(
                args.max_batch_size, 1, self.head_dim // block_size, dtype=torch.float32,
            ),
            persistent=False,
        )

        # SOURCE-track indexer redesign (docs/track_source_design.md).  Mode
        # state, not checkpoint weights: configured via
        # Transformer.set_indexer_source(); None = the unchanged base path.
        self._src_cfg: SourceConfig | None = None
        self._src_wide_proj: torch.Tensor | None = None  # [d_in, r]
        self._src_ckv_map: torch.Tensor | None = None    # [head_dim, d_ckv]
        self._src_code_cache: torch.Tensor | None = None
        self._src_code_scale: torch.Tensor | None = None

        # Opt-in compressed-latent SELECTION key source (ckv ΔPPL gate,
        # docs/prereg_ckv_glm_ppl.md).  None = the certified base wk key path
        # (default).  "ckv" derives the indexer selection key from the MLA
        # joint cache row via the fitted per-layer M_derot map ``_select_m0``
        # ([576,128]) instead of the base ``wk`` projection, feeding the
        # IDENTICAL fp8 quant + cache + score path.  Orthogonal to the
        # SOURCE track (_src_cfg) and to cascade/lag; a pure key-source swap.
        # Model-agnostic: the output R_idx rope layout follows
        # ``indexer_rope_interleave`` (DeepSeek non-interleaved / GLM-5
        # interleaved).
        self._select_key_source: str | None = None
        self._select_m0: torch.Tensor | None = None

    def reset_kv_cache(self, bsz: int, seqlen: int) -> None:
        dev = self.k_cache.device
        self.k_cache = torch.zeros(bsz, seqlen, self.head_dim, dtype=torch.float8_e4m3fn, device=dev)
        self.k_scale_cache = torch.zeros(bsz, seqlen, self.head_dim // self._cache_block_size, dtype=torch.float32, device=dev)
        self._reset_source_code_cache(bsz, seqlen, dev)

    # --- SOURCE-track: compressed / c_KV-sourced indexer key path ----------
    def _reset_source_code_cache(
        self, bsz: int, seqlen: int, dev: torch.device
    ) -> None:
        cfg = self._src_cfg
        if cfg is None or not cfg.uses_code_cache:
            self._src_code_cache = None
            self._src_code_scale = None
            return
        if cfg.code_dtype == "bf16":
            self._src_code_cache = torch.zeros(
                bsz, seqlen, cfg.rank, dtype=torch.bfloat16, device=dev
            )
            self._src_code_scale = None
        else:  # fp8 codes + per-key fp32 scale
            self._src_code_cache = torch.zeros(
                bsz, seqlen, cfg.rank, dtype=torch.float8_e4m3fn, device=dev
            )
            self._src_code_scale = torch.zeros(
                bsz, seqlen, 1, dtype=torch.float32, device=dev
            )

    def configure_source(
        self,
        cfg: SourceConfig | None,
        wide_proj: torch.Tensor | None = None,
        ckv_map: torch.Tensor | None = None,
    ) -> None:
        """Install (or clear, ``cfg=None``) an indexer-source mechanism.

        ``wide_proj``/``ckv_map`` are the projection tensors the mechanism
        scores through (see ``Transformer.set_indexer_source`` for their
        provenance).  Code caches are (re)allocated by ``reset_kv_cache``;
        configuring a source invalidates any existing code cache.
        """
        if cfg is None:
            self._src_cfg = None
            self._src_wide_proj = None
            self._src_ckv_map = None
            self._src_code_cache = None
            self._src_code_scale = None
            return
        needs_wide = cfg.uses_code_cache
        needs_map = cfg.source in ("ckv", "ckv_lowrank")
        if needs_wide and wide_proj is None:
            raise ValueError(f"source {cfg.source!r} requires wide_proj")
        if needs_map and ckv_map is None:
            raise ValueError(f"source {cfg.source!r} requires ckv_map")
        if wide_proj is not None and wide_proj.shape[1] != cfg.rank:
            raise ValueError(
                f"wide_proj columns {wide_proj.shape[1]} != rank {cfg.rank}"
            )
        if ckv_map is not None and ckv_map.shape[0] != self.head_dim:
            raise ValueError(
                f"ckv_map rows {ckv_map.shape[0]} != index head_dim "
                f"{self.head_dim}"
            )
        dev = self.weights_proj.weight.device
        self._src_cfg = cfg
        self._src_wide_proj = (
            wide_proj.to(device=dev, dtype=torch.float32) if needs_wide else None
        )
        self._src_ckv_map = (
            ckv_map.to(device=dev, dtype=torch.float32) if needs_map else None
        )
        self._src_code_cache = None
        self._src_code_scale = None

    def _append_source_codes(
        self, feats: torch.Tensor, start_pos: int, end_pos: int
    ) -> None:
        """Project new keys to r-dim codes and append them to the code cache.

        ``feats`` is ``[B, S, d_in]``: the roped indexer key (``base_eager``/
        ``lowrank``) or the joint ``[c_KV | k_pe]`` cache row
        (``ckv_lowrank``).  One tiny GEMM per new token — the wide-stage
        scan then reads only these codes.
        """
        cfg = self._src_cfg
        if cfg is None or not cfg.uses_code_cache:
            raise RuntimeError("_append_source_codes called without a code cache")
        if self._src_code_cache is None:
            raise RuntimeError(
                "source code cache not allocated; call reset_kv_cache() after "
                "set_indexer_source()"
            )
        bsz = feats.size(0)
        if (
            self._src_code_cache.size(0) < bsz
            or self._src_code_cache.size(1) < end_pos
        ):
            raise RuntimeError(
                f"source code cache too small (need bsz={bsz}, seqlen>={end_pos}; "
                f"have {tuple(self._src_code_cache.shape[:2])}); call "
                "reset_kv_cache() after set_indexer_source()"
            )
        proj = self._src_wide_proj.to(feats.dtype)
        if feats.shape[-1] != proj.shape[0]:
            raise ValueError(
                f"source feature dim {feats.shape[-1]} != wide_proj rows "
                f"{proj.shape[0]}"
            )
        codes = feats @ proj  # [B, S, r]
        if cfg.code_dtype == "bf16":
            self._src_code_cache[:bsz, start_pos:end_pos] = codes.to(torch.bfloat16)
        else:
            codes_fp8, scale = fp8_quantize_per_key(codes)
            self._src_code_cache[:bsz, start_pos:end_pos] = codes_fp8
            self._src_code_scale[:bsz, start_pos:end_pos] = scale

    def _source_codes_slice(
        self, bsz: int, end_pos: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantized ``[B, end_pos, r]`` view/copy of the code cache."""
        cfg = self._src_cfg
        codes = self._src_code_cache[:bsz, :end_pos]
        if cfg.code_dtype == "bf16":
            return codes
        return fp8_dequantize(
            codes, self._src_code_scale[:bsz, :end_pos], dtype
        )

    def _forward_source(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        kv_joint: torch.Tensor | None,
    ):
        """Decode-step selection under the configured source mechanism.

        Same contract as the base decode path: returns
        ``(topk_indices [B, 1, min(topk, end_pos)], index_score [B, 1,
        end_pos])``.  ``index_score`` is ``-inf`` outside the fine-scored
        set for two-stage sources.  No fp8 GEMM kernels: the whole path is
        eager torch, so the eager ``base_eager`` control is the
        apples-to-apples speed reference (see docs/track_source_design.md).
        """
        cfg = self._src_cfg
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if seqlen != 1:
            raise RuntimeError(
                f"_forward_source is a decode-step path (seqlen=1), got "
                f"seqlen={seqlen}"
            )
        if cfg.source in ("ckv", "ckv_lowrank") and kv_joint is None:
            raise RuntimeError(
                f"source {cfg.source!r} needs the MLA joint KV cache; the "
                "caller must pass kv_joint"
            )

        # Indexer queries: identical to the base path minus the fp8
        # quantization and Hadamard rotation (orthogonal — score-preserving;
        # it exists only for fp8 quantization robustness).
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        weights = (
            self.weights_proj(x.float())
            * self.n_heads**-0.5
            * self.softmax_scale
        )

        if cfg.source in ("base_eager", "lowrank"):
            # Key side unchanged from base (wk + norm + rope); the new
            # token's full-width fp8 key row is still cached because the
            # lowrank fine stage gathers from it on later steps.
            k = self.k_norm(self.wk(x))
            k_pe, k_nope = torch.split(
                k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            k_pe = apply_rotary_emb(
                k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave
            ).squeeze(2)
            k = torch.cat([k_pe, k_nope], dim=-1)
            if cfg.source == "lowrank":
                if self.k_scale_cache.shape[-1] != 1:
                    raise RuntimeError(
                        "source 'lowrank' fine stage assumes one fp8 scale "
                        "per key (head_dim == quant block size); got "
                        f"{self.k_scale_cache.shape[-1]} scale groups"
                    )
                k_rot = rotate_activation(k)
                k_fp8, k_scale = fp8_quantize_per_key(k_rot)
                self.k_cache[:bsz, start_pos:end_pos] = k_fp8
                self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
            self._append_source_codes(k, start_pos, end_pos)
            q_wide = q @ self._src_wide_proj.to(q.dtype)
            key_codes = self._source_codes_slice(bsz, end_pos, q.dtype)
            if cfg.source == "base_eager":
                scores = weighted_relu_scores(q_wide, key_codes, weights)
                topk_indices, index_score = single_stage_topk(
                    scores, self.index_topk
                )
            else:
                # Fine stage: exact full-width re-score of the survivors,
                # gathered from the fp8 key cache (Hadamard-rotated rows, so
                # the fine query is rotated to match).
                q_fine = rotate_activation(q)

                def _gather_fine(idx: torch.Tensor) -> torch.Tensor:
                    # Gather the C survivor rows FIRST (fp8 via a uint8
                    # view — torch.gather has no float8 kernel), then
                    # dequantize only those rows.  The SRC-A run (job
                    # 24576) predated this fix and dequantized the ENTIRE
                    # [L, 128] cache slice per layer-step (~84 MB traffic
                    # + a 67 MB fp32 allocation at 128K) before gathering
                    # — part of src_lowrank's measured eager overhead
                    # (docs/track_source_cells.md, SRC-A verdict).
                    flat = idx.reshape(bsz, -1)  # [B, S*C]
                    rows_fp8 = torch.gather(
                        self.k_cache[:bsz, :end_pos].view(torch.uint8),
                        1,
                        flat.unsqueeze(-1).expand(-1, -1, self.head_dim),
                    ).view(torch.float8_e4m3fn)
                    scales = torch.gather(
                        self.k_scale_cache[:bsz, :end_pos], 1,
                        flat.unsqueeze(-1).expand(
                            -1, -1, self.k_scale_cache.shape[-1]
                        ),
                    )
                    rows = rows_fp8.to(torch.float32) * scales
                    return rows.view(*idx.shape, self.head_dim).to(q_fine.dtype)

                topk_indices, index_score = two_stage_select(
                    q_wide, key_codes, q_fine, _gather_fine, weights,
                    survivors=cfg.survivors, topk=self.index_topk,
                )
        elif cfg.source == "ckv":
            # Direct full-rank scan of the attention KV cache with the
            # query absorbed through the (fitted) ckv map — no indexer key
            # projection, cache, or code cache at all.  Scan-cost NEGATIVE
            # control (576 > 128 dims); the win is cache elimination.
            q_hat = torch.einsum(
                "bshd,dc->bshc", q.float(), self._src_ckv_map
            ).to(q.dtype)
            joint = kv_joint[:bsz, :end_pos]
            scores = weighted_relu_scores(q_hat, joint, weights)
            topk_indices, index_score = single_stage_topk(scores, self.index_topk)
        elif cfg.source == "ckv_lowrank":
            # Codes are appended from the joint cache row the attention path
            # already wrote this step (MLA writes kv/pe before the indexer
            # runs); the fine stage gathers survivor rows from the same
            # cache.  The indexer key path (wk, k_norm, fp8 k cache) is
            # never touched.
            joint = kv_joint[:bsz, :end_pos]
            self._append_source_codes(
                kv_joint[:bsz, start_pos:end_pos], start_pos, end_pos
            )
            q_hat = torch.einsum(
                "bshd,dc->bshc", q.float(), self._src_ckv_map
            ).to(q.dtype)
            q_wide = q_hat @ self._src_wide_proj.to(q_hat.dtype)
            key_codes = self._source_codes_slice(bsz, end_pos, q.dtype)

            def _gather_joint(idx: torch.Tensor) -> torch.Tensor:
                d_ckv = joint.shape[-1]
                flat = idx.reshape(bsz, -1)
                rows = torch.gather(
                    joint, 1, flat.unsqueeze(-1).expand(-1, -1, d_ckv)
                )
                return rows.view(*idx.shape, d_ckv)

            topk_indices, index_score = two_stage_select(
                q_wide, key_codes, q_hat, _gather_joint, weights,
                survivors=cfg.survivors, topk=self.index_topk,
            )
        else:  # pragma: no cover - SourceConfig validates
            raise RuntimeError(f"unknown source {cfg.source!r}")

        if dist.is_initialized() and dist.get_world_size() > 1:
            # Same determinism tail as the base path: rank 0's selection is
            # authoritative (the indexer is replicated, not sharded).
            topk_indices_ = topk_indices.clone()
            dist.broadcast(topk_indices_, src=0)
            if not torch.equal(topk_indices, topk_indices_):
                topk_indices = topk_indices_
            index_score_ = index_score.clone()
            dist.broadcast(index_score_, src=0)
            index_score = index_score_
        return topk_indices, index_score

    def stage_fp8_operands(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Front half of :meth:`forward` for an in-kernel scoring pass.

        Computes the fp8 indexer queries and folded per-head weights,
        and appends the new token's fp8 index key (+ scale) to the key
        cache — everything ``forward`` does BEFORE its ``fp8_index``
        call.  The fused reuse path then scores the full prefix inside
        the attention kernel (``with_select=1``) instead.

        The op sequence deliberately DUPLICATES ``forward``'s front half
        instead of refactoring it (existing paths stay bit-for-bit); the
        two are cross-certified by the ``--mini`` step-0 selection-
        equality gate (fused refresh selection vs the eager indexer on
        identical state), which runs exactly this staging against
        ``forward``'s own scoring.

        Returns:
            ``(q_fp8 [B, S, H_idx, 128], weights [B, S, H_idx] float32)``
            with ``weights = weights_proj * H^-0.5 * q_scale *
            softmax_scale`` — the kernel's ``sel_w`` folding.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale).squeeze(-1)
        return q_fp8, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        kv_joint: torch.Tensor | None = None,
    ):
        if self._src_cfg is not None and mask is None:
            # SOURCE-track decode-step selection (mask is None exactly on the
            # single-token MQA decode path).  Prefill below stays on the base
            # fp8 path and only appends wide-stage codes.
            return self._forward_source(x, qr, start_pos, freqs_cis, kv_joint)
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        if self._select_key_source == "ckv":
            # Opt-in key-source swap (ckv ΔPPL gate): derive the selection key
            # from the MLA joint cache row via the fitted M_derot map, instead
            # of the base wk projection.  derive_key applies R_idx internally,
            # so k̂ is already idx-rope'd and in the base [pe(64)|nope(64)]
            # layout — it feeds the identical rotate/fp8/cache/score path.
            if kv_joint is None:
                raise RuntimeError(
                    "select_key_source='ckv' needs the MLA joint KV cache; "
                    "the caller must pass kv_joint"
                )
            if self._select_m0 is None:
                raise RuntimeError(
                    "select_key_source='ckv' requires a per-layer M0; call "
                    "Transformer.set_select_key_source(..., m0_by_layer=...)"
                )
            joint_rows = kv_joint[:bsz, start_pos:end_pos].to(torch.float32)
            k = _ckv_derive_key(
                joint_rows, freqs_cis, self._select_m0,
                idx_interleave=self.indexer_rope_interleave,
            ).to(x.dtype)
        else:
            k = self.wk(x)
            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave).squeeze(2)
            k = torch.cat([k_pe, k_nope], dim=-1)
        if self._src_cfg is not None and self._src_cfg.uses_code_cache:
            # SOURCE-track prefill append: one r-dim projection per new key
            # so decode-time wide scans cover the whole prefix.  Codes come
            # from the un-rotated key space (base_eager/lowrank) or the
            # joint [c_KV | k_pe] cache row (ckv_lowrank) that MLA wrote
            # just before invoking the indexer.
            if self._src_cfg.source in ("base_eager", "lowrank"):
                self._append_source_codes(k, start_pos, end_pos)
            elif self._src_cfg.source == "ckv_lowrank":
                if kv_joint is None:
                    raise RuntimeError(
                        "source 'ckv_lowrank' needs the MLA joint KV cache; "
                        "the caller must pass kv_joint"
                    )
                self._append_source_codes(
                    kv_joint[:bsz, start_pos:end_pos], start_pos, end_pos
                )
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale).squeeze(-1)
        k_fp8_slice = torch.empty(
            bsz, end_pos, self.head_dim, dtype=torch.float8_e4m3fn, device=x.device
        )
        k_fp8_slice.copy_(self.k_cache[:bsz, :end_pos])
        k_scale_slice = torch.empty(bsz, end_pos, dtype=torch.float32, device=x.device)
        k_scale_slice.copy_(self.k_scale_cache[:bsz, :end_pos].squeeze(-1))
        index_score = fp8_index(
            q_fp8.contiguous(), weights.contiguous(), k_fp8_slice, k_scale_slice
        )
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        if dist.is_initialized() and dist.get_world_size() > 1:
            topk_indices_ = topk_indices.clone()
            dist.broadcast(topk_indices_, src=0)
            if not torch.equal(topk_indices, topk_indices_):
                topk_indices = topk_indices_
            index_score_ = index_score.clone()
            dist.broadcast(index_score_, src=0)
            index_score = index_score_
        return topk_indices, index_score

    def forward_striped(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        spans: list[tuple[int, int]],
        score_row: torch.Tensor,
    ):
        """Single-token decode scan restricted to ``spans`` of the prefix.

        Temporal-striping counterpart of :meth:`forward` (track SCHED
        mechanism f): projects/quantizes q and k and writes the new key into
        the cache EXACTLY like ``forward``, but runs ``fp8_index`` only over
        the given key spans, merging their scores into the caller's running
        ``score_row`` cache in place.  Columns outside the spans keep their
        previous scores — computed against an OLDER query (the staleness the
        mechanism trades for the O(L/period) per-step cost); never-scored
        columns hold ``-inf`` and cannot be selected.

        The projection/caching block below mirrors ``forward`` statement for
        statement; keep the two in sync.

        Args:
            x: ``[B, 1, dim]`` single-token decode input.
            qr: Shared low-rank query activations (same as ``forward``).
            spans: Disjoint sorted ``[lo, hi)`` key spans within
                ``[0, end_pos)`` to score this step
                (see :func:`refresh_stripe_spans`).
            score_row: ``[B, alloc_len]`` float32 running score cache;
                updated in place at the span columns and broadcast-synced
                across ranks so every rank threads the same scores.

        Returns:
            ``(topk_indices, index_score)`` with the same shapes/semantics
            as ``forward`` (``index_score`` is the merged ``[B, 1, end_pos]``
            row, partially stale by construction).
        """
        bsz, seqlen, _ = x.size()
        if seqlen != 1:
            raise ValueError(
                f"forward_striped is a single-token decode path, got seqlen={seqlen}"
            )
        end_pos = start_pos + seqlen
        if score_row.shape[0] != bsz or score_row.size(1) < end_pos:
            raise ValueError(
                f"score_row must be [B={bsz}, >= {end_pos}], got "
                f"{tuple(score_row.shape)}"
            )
        for lo, hi in spans:
            if not 0 <= lo < hi <= end_pos:
                raise ValueError(
                    f"span ({lo}, {hi}) outside the causal prefix [0, {end_pos})"
                )
        # --- projection + cache write: mirrors forward exactly ---
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale).squeeze(-1)
        # --- span-restricted scoring into the running cache ---
        q_fp8 = q_fp8.contiguous()
        weights = weights.contiguous()
        for lo, hi in spans:
            k_fp8_slice = torch.empty(
                bsz, hi - lo, self.head_dim,
                dtype=torch.float8_e4m3fn, device=x.device,
            )
            k_fp8_slice.copy_(self.k_cache[:bsz, lo:hi])
            k_scale_slice = torch.empty(
                bsz, hi - lo, dtype=torch.float32, device=x.device
            )
            k_scale_slice.copy_(self.k_scale_cache[:bsz, lo:hi].squeeze(-1))
            span_score = fp8_index(q_fp8, weights, k_fp8_slice, k_scale_slice)
            score_row[:, lo:hi] = span_score[:, 0].float()
        # Clone: the returned row must not alias the cache (consumers such as
        # the refresh prior hold it across layers while later steps mutate
        # the cache in place).
        index_score = score_row[:, :end_pos].unsqueeze(1).clone()  # [B, 1, T]
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Mirror forward's rank-0 broadcast, and fold the agreed row back
            # into the cache so stale columns can never drift across ranks.
            dist.broadcast(index_score, src=0)
            score_row[:, :end_pos] = index_score[:, 0]
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        if dist.is_initialized() and dist.get_world_size() > 1:
            topk_indices_ = topk_indices.clone()
            dist.broadcast(topk_indices_, src=0)
            if not torch.equal(topk_indices, topk_indices_):
                topk_indices = topk_indices_
        return topk_indices, index_score


def weight_dequant(weight, scale):
    shape = weight.shape
    if weight.dim() != 2:
        raise ValueError(
            f"weight_dequant expects a 2-D weight tensor, got {weight.dim()}-D"
        )
    weight = (
        weight.view(
            shape[0] // block_size, block_size, shape[1] // block_size, block_size
        )
        .transpose(1, 2)
        .contiguous()
        .view(-1, block_size * block_size)
    )
    weight = (
        (weight.float() * scale.view(-1, 1).float())
        .to(torch.get_default_dtype())
        .view(shape[0] // block_size, shape[1] // block_size, block_size, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(shape)
    )
    return weight


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.qk_head_dim
        )
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim**-0.5
        self.scale_fmt = args.scale_fmt
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self._cache_max_bsz = args.max_batch_size
        # Allocate with seqlen=1; use reset_kv_cache() before each eval run.
        # kv_cache and pe_cache are views into one joint [B, T, 512+64]
        # buffer whose rows are the fused decode kernel's KV layout
        # ([kv_lora | rope] contiguous per position).  All existing slice
        # reads and writes go through the views unchanged; the views are
        # re-derived by reset_kv_cache, which every eval calls first (a
        # .to()/.cuda() after init moves only _kv_joint, so the views are
        # stale until then — same lifecycle as the old seqlen=1 buffers).
        self.register_buffer(
            "_kv_joint",
            torch.zeros(
                args.max_batch_size, 1, self.kv_lora_rank + self.qk_rope_head_dim
            ),
            persistent=False,
        )
        self.kv_cache = self._kv_joint[..., : self.kv_lora_rank]
        self.pe_cache = self._kv_joint[..., self.kv_lora_rank :]
        self.dequant_wkv_b = None

        self._collect_attn_mass: bool = False
        self._attn_mass_window: int = 128
        self._n_window_queries: int | None = None
        # When True the indexer routing mask is NOT applied during single-token
        # decode.  Set via Transformer.set_indexer_frozen(); used by the
        # hub_frozen eval to attend densely over all retained cache positions
        # without further dynamic indexer re-selection.
        self._disable_indexer_mask: bool = False
        # One-layer lag: when set, this layer's sparse attention uses the
        # top-k indices supplied here instead of its own indexer output.
        # The indexer still runs so last_topk_indices is populated for the
        # next layer's lag_external_topk.  Set per-layer by Transformer
        # before each forward when _lag_mode is True.
        self._lag_external_topk: torch.Tensor | None = None
        self.last_index_score: torch.Tensor | None = None
        self.last_topk_indices: torch.Tensor | None = None
        self.last_raw_qk_scores: torch.Tensor | None = None
        self._capture_raw_qk: bool = False
        # Selection tap / cascade (observation harness).  When _cascade_mode
        # is set, this layer emits last_cascade_topk — the tap re-rank of the
        # keys it actually attended plus a recency/sink admission window —
        # which the Transformer threads into the NEXT layer's
        # _lag_external_topk (same plumbing as the one-layer lag).  When
        # _collect_tap_topk is set, the layer also captures last_tap_topk,
        # the dense-tap top-k over ALL causal keys, for fidelity metrics
        # against the lightning indexer.  Both consume the raw pre-softmax
        # attention scores before any masking.  Set via
        # Transformer.set_cascade_mode() / forward(collect_tap_topk=...).
        self._cascade_mode: bool = False
        self._cascade_window: int = 128
        self._cascade_n_sink: int = 0
        self._cascade_head_weights: torch.Tensor | None = None
        # Pool/attend decoupling: when > 0, the cascade threads a candidate
        # POOL of this size to the next layer while attention still uses only
        # the leading index_topk entries (every threaded set is score-sorted,
        # so the attend set is the pool's leading slice).  A key then survives
        # between refreshes as long as it stays in the tap's top-pool rather
        # than its top-k, raising the eviction threshold that binds pool
        # recall.  0 = disabled (pool == attend == index_topk).
        self._cascade_pool_topk: int = 0
        # Design-space levers (set_cascade_mode): phase-rotating stripe over
        # the pre-window region, selection-only incumbent bias, and the
        # running max-ever-tap hub set (per-sequence state, reset with the
        # KV cache; distinct from hub_marginalization's analysis-time
        # aggregates).
        self._cascade_stripe: int = 0
        # Temporal refresh striping (track SCHED, mechanism f) — distinct
        # from _cascade_stripe (the ADMISSION stripe over candidate keys
        # within one step).  When _refresh_stripe_period > 0 this layer's
        # decode-time lightning-indexer scan scores only a rotating
        # 1/period slice of the prefix per step (plus a fresh tail),
        # merging into a running per-layer score cache; the top-k threads
        # a partially stale score row.  Set by set_cascade_mode(...,
        # refresh_stripe_period=...) on layers that run the indexer in
        # cascade decode (layer 0 + refresh producers).  Eager-only.
        self._refresh_stripe_period: int = 0
        self._refresh_stripe_fresh_tail: int = 0
        # Running score cache [B, alloc_len] fp32 (per-sequence state, reset
        # with the KV cache), frozen slice width, striped-step counter, and
        # the expected end_pos of the next striped step (any discontinuity —
        # prefill boundary, cache reset — forces a full-scan re-seed).
        self._refresh_stripe_cache: torch.Tensor | None = None
        self._refresh_stripe_width: int = 0
        self._refresh_stripe_step: int = 0
        self._refresh_stripe_expected_end: int = -1
        # Debug/validation hooks (bench/track_sched_mini.py): the spans the
        # last striped step scored, and — when _refresh_stripe_selfcheck is
        # set — a comparison of the merged row against a same-state full
        # scan (fresh columns must match exactly; stale columns measure the
        # staleness the mechanism introduces).
        self._refresh_stripe_selfcheck: bool = False
        self.last_refresh_stripe_spans: list[tuple[int, int]] | None = None
        self.last_refresh_stripe_selfcheck: dict | None = None
        self._cascade_hysteresis: float = 0.0
        # Same-layer scored admission (V3 semantics, prereg_scored_admission /
        # tap_fidelity_pool follow-up): selection over incumbent ∪ admission
        # is ranked by THIS layer's own lightning-indexer scores instead of
        # the (one-layer-lagged) tap, and the layer ATTENDS that same-layer
        # selection.  The tap is still computed for the hub's 'max' statistic
        # (control parity).  Prefill-only: the decode cascade site raises.
        self._cascade_same_layer: bool = False
        self._cascade_same_layer_selects: int = 0
        # Fused-decode hysteresis in TAP-PLANE units: the eager delta
        # converted by set_cascade_fused_decode(...,
        # hysteresis_scale_matched=True).  Kept separate from
        # _cascade_hysteresis because the two selects score on
        # incommensurate scales (planes = per-head ReLU sums of raw QK;
        # eager tap = head mean/weighted sum of softmax_scale-multiplied
        # scores) — see the H1' registration in design_space_prereg.md.
        self._cascade_hysteresis_fused: float = 0.0
        # b' (select-only prefill scope, prereg SS9 quality replay): when True,
        # only refresh/select layers run own-select attention + fold the hub;
        # non-refresh skip layers reuse-copy (thread the incumbent) and do not
        # fold. Mirrors the serving _index_prefill_full=select-only fix. Default
        # False = certified realization b (own-select + hub at every layer).
        self._cascade_prefill_select_only: bool = False
        self._cascade_hub_size: int = 0
        self._cascade_hub_stat: str = "max"
        self._cascade_hub_decay: float = 0.9
        self._cascade_hub_scores: torch.Tensor | None = None
        # Hub-membership freeze (prereg_freeze_ablation): when set, the hub
        # statistic stops FOLDING for any forward whose chunk starts at or
        # beyond this absolute position — the serving design's frozen
        # selection-history set (membership fixed at prefill end).  Reads
        # (hub candidate admission in _cascade_select) are unaffected: the
        # frozen buffer keeps serving its top-N.  None = mutable (research
        # hub, updates every chunk).
        self._cascade_hub_freeze_pos: int | None = None
        # Coverage-track admission levers (set_cascade_mode; both stateless
        # and eager-only, following the stripe pattern):
        # - neighbor expansion (g): admit positional neighbors +/-1..delta of
        #   the incumbent/pool set, phase-subsampled to neighbor_budget slots
        #   per query (tap.neighbor_candidates).
        # - block coarse-to-fine (e): reduce the key axis to
        #   ceil(T/block_size) per-block summary scores every layer
        #   (tap.block_summary_scores; "mean" = the exact mean-latent
        #   summary-key score by bilinearity), tap-reduce them, and admit
        #   every token of the top block_admit blocks
        #   (tap.block_token_candidates).
        self._cascade_neighbor_delta: int = 0
        self._cascade_neighbor_budget: int = 0
        self._cascade_block_size: int = 0
        self._cascade_block_admit: int = 0
        self._cascade_block_stat: str = "mean"
        # Entries-head (docs/prereg_entries_head.md): learned block-admission
        # ranker riding the block lever.  _cascade_block_head holds THIS
        # layer's fitted head payload (mu/sd/w1/b1/w2/b2) or None;
        # _cascade_block_stash makes _block_candidates retain last_block_tap
        # so the forward loop can thread it to the NEXT layer's
        # _cascade_prev_block_tap (per-chunk transient, prior_row pattern).
        self._cascade_block_head: dict | None = None
        self._cascade_block_stash: bool = False
        self._cascade_prev_block_tap: torch.Tensor | None = None
        self.last_block_tap: torch.Tensor | None = None
        # Scoring-function variant for the cascade's tap re-rank (axis V
        # stage 3): None = the classic sum_h w_h ReLU(S_h).  Uniform-weight
        # variants only (the screen-validated arms); mutually exclusive
        # with distilled head weights and the fused decode path.
        self._cascade_scoring_variant: str | None = None
        # Refresh-prior blend (axis-7 source S7e): consumer taps inside a
        # refresh block add a decaying, scale-matched additive prior gathered
        # from the block producer's lightning-indexer scores.
        # _cascade_prior_row holds the producer's [B, S, T] score row for the
        # CURRENT forward chunk only (threaded by the Transformer loop and
        # dropped right after this layer runs — never retained across chunks);
        # _cascade_prior_dist is this layer's distance from the producer
        # (1 = immediate consumer).  Set via set_cascade_mode(...,
        # refresh_prior_lambda=..., refresh_prior_decay=...).
        self._cascade_prior_lambda: float = 0.0
        self._cascade_prior_decay: float = 1.0
        self._cascade_prior_row: torch.Tensor | None = None
        self._cascade_prior_dist: int = 0
        # Serving mode: during single-token MQA decode this layer skips the
        # lightning indexer entirely and attends via the cascade set threaded
        # from the previous layer.  Set per-layer by set_cascade_mode(...,
        # skip_indexer_decode=True) on non-refresh consumer layers only —
        # layer 0 and refresh producers still run the indexer.
        self._cascade_skip_indexer_decode: bool = False
        # ρ_admit freeze: set per-layer by set_cascade_mode; when True this
        # layer skips _cascade_select and threads the incumbent forward.
        self._cascade_freeze_admit: bool = False
        # Latent-refresh: at refresh layers, replace the indexer's O(L) key
        # scan with a cheap dot-product over the compressed MLA latents
        # already in kv_cache.  Scores = q_latent @ kv_cache[:, :pos].T
        # (512-dim matmul, no learned projection).  Set per-layer by
        # set_cascade_mode(..., latent_refresh=True) on refresh producers.
        self._cascade_latent_refresh: bool = False
        # Adaptive refresh: content-driven refresh triggering.  When the
        # jaccard similarity between this layer's current and previous
        # decode step's produced selection drops below 1 - threshold, the
        # layer is promoted to a dynamic refresh at the NEXT step (its
        # indexer runs, tap production is suppressed, and the fresh set
        # replaces the stale cascade).  Model-agnostic and self-tuning:
        # noisy models (GLM-5) trigger more refreshes; stable models (DS)
        # fewer.  Set per-layer by set_cascade_mode.
        self._cascade_adaptive_threshold: float = 0.0
        self._cascade_prev_selection: torch.Tensor | None = None
        self._cascade_adaptive_pending: bool = False
        # Fused-kernel decode (set via Transformer.set_cascade_fused_decode):
        # when _fused_decode_state holds a FusedDecodeState, single-token MQA
        # decode runs sparse attention over the layer's index set through the
        # fused split-K kernel instead of the full-prefix einsum, and layers
        # whose cascade set the next layer consumes (_fused_produce_tap) emit
        # the tap planes + next index set from the same launch.
        self._fused_decode_state = None
        # reuse_fused schedule (lag + IndexCache composition): fused
        # select-and-attend on refresh layers only; see
        # Transformer.set_reuse_fused_decode.
        self._reuse_fused_state = None
        self._reuse_fused_rowloop = False
        self._fused_produce_tap: bool = False
        # Local (rank-sliced, zero-padded to the kernel's 16-head block)
        # float32 tap weights, or None for the uniform tap.
        self._fused_tapw: torch.Tensor | None = None
        self._collect_tap_topk: bool = False
        self._tap_query_chunk: int = 1024
        self.last_cascade_topk: torch.Tensor | None = None
        self.last_tap_topk: torch.Tensor | None = None
        # Tap-weight fitting (distillation): when _collect_tap_features is set
        # and the current chunk ends at _tap_feature_end, store the raw
        # per-head ReLU'd scores at _tap_feature_rows (chunk-relative), plus
        # the lightning indexer's scores at the same rows (the teacher).
        # Offloaded to CPU so all layers of a long forward fit in host memory.
        self._collect_tap_features: bool = False
        self._tap_feature_end: int = -1
        self._tap_feature_rows: torch.Tensor | None = None
        self.last_tap_features: torch.Tensor | None = None
        self.last_teacher_scores: torch.Tensor | None = None
        # On-policy pool capture: the index set this layer actually attended
        # at the sampled rows (the cascade's incumbent pool), so the fit can
        # restrict to keys the deployed cascade could really score.
        self.last_tap_pool: torch.Tensor | None = None
        # Scoring-variant screen (design-space axis V): when
        # _collect_variant_topk is set and the current chunk ends at
        # _tap_feature_end, store each variant's dense causal top-k at
        # _tap_feature_rows, this layer's indexer top-k at the same rows
        # (the adjacent pair's one-hop teacher), and the head-summed dense
        # attention probs there (the mass endpoint).  All CPU-resident so a
        # full-depth screen fits in host memory.
        self._collect_variant_topk: bool = False
        self._variant_weights: torch.Tensor | None = None
        self._variant_lse_taus: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
        self.last_variant_topk: dict[str, torch.Tensor] | None = None
        self.last_variant_indexer_rows: torch.Tensor | None = None
        self.last_attn_probs_row: torch.Tensor | None = None
        # One-hop tap-lag (axis V stage 2): when _tap_lag_variant is set,
        # this layer computes the variant's DENSE causal top-k over its own
        # pre-softmax scores and threads it to the next layer through the
        # lag plumbing.  Every layer still runs its true indexer, so — like
        # the plain lag and unlike the cascade — selection errors cannot
        # compound across depth.
        self._tap_lag_variant: str | None = None
        self._tap_lag_weights: torch.Tensor | None = None
        # Temperature for the v4b_temp tap-lag arm (1.0 for every other
        # variant): v4b(tau) == v4 on S/tau with stats computed on S/tau.
        self._tap_lag_tau: float = 1.0
        self.last_tap_lag_topk: torch.Tensor | None = None
        self.last_attn_mass: torch.Tensor | None = None
        self.last_attn_mass_window: torch.Tensor | None = None
        self.last_dense_attn_mass: torch.Tensor | None = None
        self.last_dense_attn_mass_window: torch.Tensor | None = None

    def reset_kv_cache(self, bsz: int, seqlen: int) -> None:
        dev = self._kv_joint.device
        dtype = self._kv_joint.dtype
        self._kv_joint = torch.zeros(
            bsz, seqlen, self.kv_lora_rank + self.qk_rope_head_dim,
            device=dev, dtype=dtype,
        )
        self.kv_cache = self._kv_joint[..., : self.kv_lora_rank]
        self.pe_cache = self._kv_joint[..., self.kv_lora_rank :]
        # Cascade-hub scores are per-sequence state: a new cache = new docs.
        self._cascade_hub_scores = None
        # So is the temporal-stripe score cache (and its step bookkeeping).
        self._refresh_stripe_cache = None
        self._refresh_stripe_step = 0
        self._refresh_stripe_expected_end = -1
        self.last_refresh_stripe_spans = None
        self.last_refresh_stripe_selfcheck = None
        self.indexer.reset_kv_cache(bsz, seqlen)

    @staticmethod
    def _accumulate_mass(
        buf: torch.Tensor | None,
        new: torch.Tensor | None,
        end_pos: int,
    ) -> torch.Tensor | None:
        """Accumulate per-chunk attention mass into a (possibly growing) buffer.

        During chunked prefill the same ``last_*_attn_mass`` field can be
        written once per chunk.  ``new`` is the current chunk's contribution,
        shaped ``(bsz, end_pos)`` where ``end_pos`` is the absolute key-dim
        length seen so far.  ``buf`` is either ``None`` (first write), or a
        previous chunk's buffer whose key-dim may be strictly smaller than
        ``end_pos``.  We grow ``buf`` with zero-padding when needed, then
        in-place add ``new``.

        Callers that reset ``buf`` to ``None`` between chunks (the public
        ``collect_attn_mass=True`` aggregation path does) see identical
        behaviour; callers that keep ``buf`` across chunks now accumulate
        correctly instead of raising a shape mismatch.
        """
        if new is None:
            return buf
        if buf is None:
            return new
        if buf.size(1) < end_pos:
            grown = new.new_zeros(buf.size(0), end_pos)
            grown[:, : buf.size(1)] = buf
            buf = grown
        buf[:, :end_pos] += new
        return buf

    def _tap_reduce(self, scores_chunk: torch.Tensor) -> torch.Tensor:
        """Tap logits for a ``[B, s, H_local, T]`` score chunk, TP-correct.

        Under tensor parallelism each rank holds a contiguous slice of the
        attention heads, so the local weighted ReLU head-sum is a partial
        result; all-reducing completes it and — just as important — makes
        every rank select the *same* set downstream.  (With uniform weights
        the reduced value is a positive rescale of the true head-sum, which
        preserves ranking.)
        """
        weights = self._cascade_head_weights
        if weights is not None:
            n_local = scores_chunk.size(2)
            weights = weights.to(scores_chunk.device)[
                rank * n_local : (rank + 1) * n_local
            ]
        tap = tap_logits(scores_chunk, weights)
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.all_reduce(tap)
        return tap


    def _capture_tap_features(self, raw_scores: torch.Tensor) -> None:
        """Store ReLU'd per-head scores + teacher indexer scores at sampled rows.

        ``raw_scores`` is ``[B, S, H_local, T]`` pre-softmax and pre-mask;
        rows are chunk-relative.  Under tensor parallelism the head dimension
        is all-gathered so the fit sees all ``H`` heads.  Offloaded to CPU
        float32 so every layer of a long forward fits in host memory.

        Under pool/attend decoupling ``last_tap_pool`` holds the widened
        candidate pool (width ``pool_topk``, including keys this layer never
        attended), not the attend set: fits collected with and without
        ``pool_topk`` are not comparable.
        """
        if self._tap_feature_rows is None:
            raise ValueError("_collect_tap_features requires _tap_feature_rows")
        rows = self._tap_feature_rows.to(raw_scores.device)
        feats = raw_scores[:, rows].float().relu()  # [B, r, H_local, T]
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gathered = [torch.empty_like(feats) for _ in range(world_size)]
            dist.all_gather(gathered, feats)
            feats = torch.cat(gathered, dim=2)
        self.last_tap_features = feats.cpu()
        if self.last_index_score is not None:
            self.last_teacher_scores = self.last_index_score[:, rows].float().cpu()
        attend = (
            self._lag_external_topk
            if self._lag_external_topk is not None
            else self.last_topk_indices
        )
        self.last_tap_pool = (
            attend[:, rows].detach().cpu() if attend is not None else None
        )

    def _variant_combine(
        self, partial: torch.Tensor, combine: str, tau: float
    ) -> torch.Tensor:
        """Complete a rank-local variant reduction across TP head shards."""
        if not (dist.is_available() and dist.is_initialized() and world_size > 1):
            return partial
        if combine == "sum":
            dist.all_reduce(partial)
            return partial
        if combine == "max":
            dist.all_reduce(partial, op=dist.ReduceOp.MAX)
            return partial
        if combine == "lse":
            # Shifted exp-sum: LSE over global heads from per-rank LSEs.
            shift = partial.clone()
            dist.all_reduce(shift, op=dist.ReduceOp.MAX)
            merged = ((partial - shift) / tau).exp()
            dist.all_reduce(merged)
            return shift + tau * merged.log()
        raise ValueError(f"unknown combine {combine!r}")

    def _capture_variant_scores(
        self,
        raw_scores: torch.Tensor,
        nope_scores: torch.Tensor | None = None,
    ) -> None:
        """One-hop scoring-variant screen: dense variant top-k at sampled rows.

        For every scoring variant in ``tap.VARIANT_NAMES`` (uniform weights
        and, when ``_variant_weights`` holds V0-fitted head weights, fitted),
        stores the dense causal top-``index_topk`` of the variant's reduction
        of this layer's signed post-scale scores at ``_tap_feature_rows`` —
        plus this layer's indexer top-k at the same rows (the adjacent
        pair's one-hop teacher) and the head-summed dense attention probs
        (the mass endpoint).  ``v3_lse`` is evaluated at every temperature
        in ``_variant_lse_taus`` (keyed ``v3_lse@<tau>``); ``v4``/``v5`` use
        the ATTEND slice's per-head softmax statistics — the only
        statistics an in-kernel probability tap would have.

        Everything lands on CPU: at 64K a full-depth screen stores ~25 MB
        per layer, so all adjacent pairs are covered without the
        ``collect_topk_indices`` full-sequence buffers.
        """
        if self._tap_feature_rows is None:
            raise ValueError("_collect_variant_topk requires _tap_feature_rows")
        if raw_scores.size(0) != 1:
            raise ValueError(
                f"variant capture requires bsz=1, got {raw_scores.size(0)}"
            )
        rows = self._tap_feature_rows.to(raw_scores.device)
        end_pos = raw_scores.size(-1)
        chunk_start = end_pos - raw_scores.size(1)
        qpos = chunk_start + rows  # [r] absolute query positions
        sel = raw_scores[:, rows].float()  # [1, r, H_local, T] signed post-scale
        key_range = torch.arange(end_pos, device=sel.device)
        acausal = (
            key_range.view(1, 1, 1, -1) > qpos.view(1, -1, 1, 1)
        )  # [1, r, 1, T]

        # Mass endpoint: dense causal softmax, head-summed, TP-completed.
        probs = sel.masked_fill(acausal, float("-inf")).softmax(dim=-1).sum(dim=2)
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.all_reduce(probs)
        self.last_attn_probs_row = probs.cpu()
        del probs

        # This layer's indexer top-k at the same rows: the one-hop teacher
        # for the (previous layer -> this layer) pair.  Score-sorted, -1
        # sentinels already mapped.
        self.last_variant_indexer_rows = self.last_topk_indices[:, rows].cpu()

        attend_stats = attend_softmax_stats(sel, self.last_topk_indices[:, rows])

        # Per-head value norms of every cached key (query-independent):
        # v = wkv_b[:, -v_head_dim:] applied to the compressed KV latents.
        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = (
            self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        )
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        vnorm = torch.einsum(
            "tc,hdc->htd",
            self.kv_cache[0, :end_pos].float(),
            wkv_b[:, -self.v_head_dim :].float(),
        ).norm(dim=-1)  # [H_local, T]

        weights = self._variant_weights
        if weights is not None:
            n_local = sel.size(2)
            weights = weights.to(sel.device)[
                rank * n_local : (rank + 1) * n_local
            ]

        k = min(self.indexer.index_topk, end_pos)
        results: dict[str, torch.Tensor] = {}
        for wname, wt in (("uniform", None), ("fitted", weights)):
            if wname == "fitted" and wt is None:
                continue
            for variant in VARIANT_NAMES:
                if variant == "v3_lse":
                    taus = self._variant_lse_taus
                elif variant == "v4b_temp":
                    taus = (0.5, 2.0)
                else:
                    taus = (1.0,)
                for tau in taus:
                    if variant == "v4b_temp":
                        # Tempered softmax = v4 on S/tau with stats on S/tau
                        # (variant_local_scores requires the pairing).
                        scaled = sel / tau
                        partial, combine = variant_local_scores(
                            scaled, variant, head_weights=wt, tau=tau,
                            attend_stats=attend_softmax_stats(
                                scaled, self.last_topk_indices[:, rows]
                            ),
                        )
                    else:
                        partial, combine = variant_local_scores(
                            sel, variant, head_weights=wt, tau=tau,
                            attend_stats=attend_stats, value_norms=vnorm,
                            causal_mask=acausal,
                        )
                    full = self._variant_combine(partial, combine, tau)
                    full = full.masked_fill(acausal.squeeze(2), float("-inf"))
                    top_scores, top_idx = full.topk(k, dim=-1)
                    top_idx = torch.where(
                        torch.isfinite(top_scores),
                        top_idx,
                        torch.full_like(top_idx, -1),
                    )
                    key = (
                        f"{variant}@{tau:g}/{wname}"
                        if variant in ("v3_lse", "v4b_temp")
                        else f"{variant}/{wname}"
                    )
                    results[key] = top_idx.to(torch.int32).cpu()

        if nope_scores is not None:
            # S7a tap-source arms (K1): the content/position decomposition
            # is the only categorically distinct tap point in the data path
            # — both branches compute the nope and pe einsums separately
            # before summing.  v4-on-nope uses attend stats computed over
            # the NOPE component: the statistics an in-kernel content-tap
            # would have.
            sel_nope = nope_scores[:, rows].float()
            sel_pe = sel - sel_nope
            nope_stats = attend_softmax_stats(
                sel_nope, self.last_topk_indices[:, rows]
            )
            source_arms: tuple[
                tuple[str, torch.Tensor, str, tuple | None], ...
            ] = (
                ("s7a_nope_v0", sel_nope, "v0_relu_sum", None),
                ("s7a_pe_v0", sel_pe, "v0_relu_sum", None),
                ("s7a_nope_v4", sel_nope, "v4_probs", nope_stats),
            )
            for arm_name, src, variant, stats in source_arms:
                partial, combine = variant_local_scores(
                    src, variant, head_weights=None, tau=1.0,
                    attend_stats=stats, causal_mask=acausal,
                )
                full = self._variant_combine(partial, combine, 1.0)
                full = full.masked_fill(acausal.squeeze(2), float("-inf"))
                top_scores, top_idx = full.topk(k, dim=-1)
                top_idx = torch.where(
                    torch.isfinite(top_scores),
                    top_idx,
                    torch.full_like(top_idx, -1),
                )
                results[f"{arm_name}/uniform"] = top_idx.to(torch.int32).cpu()
        self.last_variant_topk = results

    def _tap_lag_select(
        self,
        raw_scores: torch.Tensor,
        attend_idx: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Dense variant-tap top-k for the tap-lag handoff (axis V stage 2).

        Scores every causal key of ``raw_scores`` (``[B, S, H_local, T]``
        pre-softmax, pre-mask) with ``_tap_lag_variant`` and returns the
        top-``index_topk`` per row — the set the NEXT layer attends.  Same
        semantics as the one-hop screen: no admission machinery, dense
        causal coverage.  The v4 family (``v4_probs``, ``v5_probs_vnorm``,
        ``v4b_temp``, ``v4a_exp_nonorm``) uses the per-head softmax
        statistics of ``attend_idx`` (the keys this layer actually
        attends — the only statistics an in-kernel tap would have).  Query
        rows run in subchunks so the ``[B, s, T]`` float32 tap bounds peak
        memory; TP partials merge via ``_variant_combine``.
        """
        variant = self._tap_lag_variant
        if variant is None:
            raise RuntimeError("_tap_lag_select requires _tap_lag_variant")
        end_pos = raw_scores.size(-1)
        k = min(self.indexer.index_topk, end_pos)
        key_range = torch.arange(end_pos, device=raw_scores.device)
        weights = self._tap_lag_weights
        if weights is not None:
            n_local = raw_scores.size(2)
            weights = weights.to(raw_scores.device)[
                rank * n_local : (rank + 1) * n_local
            ]
        vnorm = None
        if variant == "v5_probs_vnorm":
            if raw_scores.size(0) != 1:
                raise ValueError(
                    "tap-lag v5 requires bsz=1 (per-key value norms are "
                    "taken from batch row 0)"
                )
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(
                    self.wkv_b.weight, self.wkv_b.scale
                )
            wkv_b = (
                self.wkv_b.weight
                if self.dequant_wkv_b is None
                else self.dequant_wkv_b
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            vnorm = torch.einsum(
                "tc,hdc->htd",
                self.kv_cache[0, :end_pos].float(),
                wkv_b[:, -self.v_head_dim :].float(),
            ).norm(dim=-1)  # [H_local, T]
        chunks: list[torch.Tensor] = []
        for lo in range(0, raw_scores.size(1), self._tap_query_chunk):
            hi = min(lo + self._tap_query_chunk, raw_scores.size(1))
            sel = raw_scores[:, lo:hi].float()
            if variant == "v4b_temp":
                sel = sel / self._tap_lag_tau
            qpos = query_positions[lo:hi]
            acausal = (
                key_range.view(1, 1, 1, -1) > qpos.view(1, -1, 1, 1)
            )
            # The whole v4 family consumes attend-slice softmax statistics
            # (v4a uses only the max m; the others also need Z).  For
            # v4b_temp the stats are computed on the ALREADY tau-scaled
            # scores — variant_local_scores assigns both rescalings to the
            # caller, and tempered-v4 semantics require the pairing.
            stats = (
                attend_softmax_stats(sel, attend_idx[:, lo:hi])
                if variant in (
                    "v4_probs", "v5_probs_vnorm", "v4b_temp", "v4a_exp_nonorm"
                )
                else None
            )
            partial, combine = variant_local_scores(
                sel, variant, head_weights=weights,
                attend_stats=stats, value_norms=vnorm, causal_mask=acausal,
            )
            full = self._variant_combine(partial, combine, 1.0)
            full = full.masked_fill(acausal.squeeze(2), float("-inf"))
            top_scores, top_idx = full.topk(k, dim=-1)
            chunks.append(torch.where(
                torch.isfinite(top_scores),
                top_idx,
                torch.full_like(top_idx, -1),
            ))
        return torch.cat(chunks, dim=1)

    def _capture_tap_topk(
        self, raw_scores: torch.Tensor, query_positions: torch.Tensor
    ) -> None:
        """Store the dense-tap top-k over all causal keys in ``last_tap_topk``.

        ``raw_scores`` is ``[B, S, H, T]`` pre-softmax and PRE-mask (no ``-inf``
        entries).  Runs in query subchunks so the ``[B, s, T]`` float32 tap
        tensor — not the full ``[B, S, T]`` — bounds peak memory.
        """
        n_keys = raw_scores.size(-1)
        k = min(self.indexer.index_topk, n_keys)
        key_range = torch.arange(n_keys, device=raw_scores.device)
        chunks: list[torch.Tensor] = []
        for lo in range(0, raw_scores.size(1), self._tap_query_chunk):
            hi = min(lo + self._tap_query_chunk, raw_scores.size(1))
            tap = self._tap_reduce(raw_scores[:, lo:hi])  # [B, s, T]
            acausal = key_range.view(1, 1, -1) > query_positions[lo:hi].view(1, -1, 1)
            tap.masked_fill_(acausal, float("-inf"))
            top_scores, top_idx = tap.topk(k, dim=-1)
            top_idx = torch.where(
                torch.isfinite(top_scores), top_idx, torch.full_like(top_idx, -1)
            )
            chunks.append(top_idx)
        self.last_tap_topk = torch.cat(chunks, dim=1)

    def _cascade_select(
        self,
        raw_scores: torch.Tensor,
        incumbent: torch.Tensor,
        query_positions: torch.Tensor,
        same_layer_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce the next layer's index set: tap re-rank of incumbent ∪ admission.

        ``raw_scores`` is ``[B, S, H, T]`` pre-softmax and PRE-mask;
        ``incumbent`` is this layer's candidate pool ``[B, S, k]`` or
        ``[B, S, pool_topk]`` (``-1`` sentinels allowed); ``query_positions``
        is ``[S]`` absolute positions.  With pool/attend decoupling the
        output is the score-sorted top-``pool_topk`` pool, whose leading
        ``index_topk`` slice is the next layer's attend set.

        ``same_layer_scores`` (``[B, S, T]``, this layer's own lightning-
        indexer score row): when given, the SELECTION is ranked by these
        same-layer scores gathered at the candidate columns instead of the
        lagged tap; the tap is still computed so the hub's tap-max statistic
        stays identical to the control arm.  Invalid (``-1``) slots rank
        ``-inf``.

        Admission is the recency window ∪ sinks, optionally widened by a
        phase-rotating stripe over the pre-window region and by this layer's
        running hub set (top keys by max-ever tap score).  With
        ``select_hysteresis`` the incumbent slots get an additive bias for
        the selection only; the incumbent copy of a duplicated key survives
        ``cascade_topk``'s stable dedup, so the bias lands deterministically.
        """
        bsz, seqlen = raw_scores.size(0), raw_scores.size(1)
        blend_prior = (
            self._cascade_prior_lambda > 0 and self._cascade_prior_row is not None
        )
        if blend_prior:
            pr = self._cascade_prior_row
            # Prior rows are per-chunk transient: a row from another chunk
            # would silently mis-gather, so mismatched shapes fail loudly.
            if pr.dim() != 3 or pr.shape[:2] != (bsz, seqlen) or (
                pr.size(-1) != raw_scores.size(-1)
            ):
                raise RuntimeError(
                    f"refresh-prior row shape {tuple(pr.shape)} does not match "
                    f"this chunk's scores [B={bsz}, S={seqlen}, "
                    f"T={raw_scores.size(-1)}]; the prior must come from a "
                    "refresh producer of the SAME forward chunk"
                )
        variant_vnorm: torch.Tensor | None = None
        if self._cascade_scoring_variant == "v5_probs_vnorm":
            if bsz != 1:
                raise ValueError(
                    "cascade v5 scoring requires bsz=1 (per-key value norms "
                    "are taken from batch row 0)"
                )
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(
                    self.wkv_b.weight, self.wkv_b.scale
                )
            wkv_b = (
                self.wkv_b.weight
                if self.dequant_wkv_b is None
                else self.dequant_wkv_b
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            variant_vnorm = torch.einsum(
                "tc,hdc->htd",
                self.kv_cache[0, : raw_scores.size(-1)].float(),
                wkv_b[:, -self.v_head_dim :].float(),
            ).norm(dim=-1)  # [H_local, T]
        admission = recency_candidates(
            query_positions, self._cascade_window, self._cascade_n_sink
        )  # [S, A]
        parts = [incumbent, admission.unsqueeze(0).expand(bsz, -1, -1)]
        if self._cascade_stripe > 0:
            stripe = stripe_candidates(
                query_positions, self._cascade_stripe, self._cascade_window
            )
            parts.append(stripe.unsqueeze(0).expand(bsz, -1, -1))
        if self._cascade_hub_size > 0 and self._cascade_hub_scores is not None:
            # Hub candidates from state as of the PREVIOUS forward: within
            # one call the chunk loop below updates the running scores, but
            # the candidate set is fixed at entry (one hub set per step).
            if self._cascade_hub_stat == "mass_decay":
                # Gamma ticks once per layer-forward (per decode step, per
                # prefill chunk — K3 registered semantics).  Uniform positive
                # scaling, so the top-k below is order-invariant to it.
                self._cascade_hub_scores.mul_(self._cascade_hub_decay)
            n_hub = min(self._cascade_hub_size, self._cascade_hub_scores.size(1))
            hub_vals, hub_idx = self._cascade_hub_scores[:bsz].topk(n_hub, dim=-1)
            hub_valid = (
                torch.isfinite(hub_vals)
                if self._cascade_hub_stat in ("max", "idx_max")
                # Mass buffers init to 0; a key with zero accumulated mass
                # was never a scored candidate and cannot be a hub.
                else hub_vals > 0
            )
            hub_idx = torch.where(
                hub_valid, hub_idx, torch.full_like(hub_idx, -1)
            )
            parts.append(hub_idx.unsqueeze(1).expand(-1, seqlen, -1))
        if self._cascade_neighbor_budget > 0:
            parts.append(
                neighbor_candidates(
                    incumbent,
                    query_positions,
                    self._cascade_neighbor_delta,
                    self._cascade_neighbor_budget,
                )
            )
        if self._cascade_block_size > 0:
            parts.append(
                self._block_candidates(raw_scores, query_positions, incumbent)
            )
        candidates = torch.cat(parts, dim=-1)  # [B, S, C]
        tap_hub_fold = (
            self._cascade_hub_size > 0 and self._cascade_hub_stat != "idx_max"
        )
        # Chunk start for the hub-freeze gate: query_positions are absolute
        # on every caller (prefill chunks and decode single rows alike).
        chunk_start = int(query_positions[0]) if tap_hub_fold else 0
        selected: list[torch.Tensor] = []
        for lo in range(0, seqlen, self._tap_query_chunk):
            hi = min(lo + self._tap_query_chunk, seqlen)
            cand = candidates[:, lo:hi]  # [B, s, C]
            gather_idx = (
                cand.clamp(min=0).unsqueeze(2).expand(-1, -1, raw_scores.size(2), -1)
            )
            cand_scores = raw_scores[:, lo:hi].gather(-1, gather_idx)  # [B, s, H, C]
            if self._cascade_scoring_variant is not None:
                tap = self._cascade_variant_reduce(
                    cand_scores, cand,
                    raw_scores[:, lo:hi],
                    incumbent[:, lo:hi],
                    variant_vnorm,
                )
            else:
                tap = self._tap_reduce(cand_scores)  # [B, s, C]
            if blend_prior:
                # S7e blend applies on top of EITHER scoring path; it lands
                # before the hub fold and hysteresis so the prior is part of
                # the tap score proper, not a selection-only bias.
                tap = self._apply_refresh_prior(tap, cand, lo, hi)
            if tap_hub_fold:
                self._cascade_hub_update(
                    cand, tap, raw_scores.size(-1), chunk_start=chunk_start
                )
            if same_layer_scores is not None:
                # V3 semantics: rank by THIS layer's own indexer scores at
                # the candidate columns; the tap above still fed the hub.
                gathered = same_layer_scores[:, lo:hi].float().gather(
                    -1, cand.clamp(min=0)
                )
                sel = torch.where(
                    cand >= 0,
                    gathered,
                    torch.full_like(gathered, float("-inf")),
                )
                self._cascade_same_layer_selects += 1
            else:
                sel = tap
            if self._cascade_hysteresis > 0.0:
                # Selection-only bias on the incumbent slots (the leading
                # columns of the candidate concat).
                sel = sel.clone()
                sel[..., : incumbent.size(-1)] += self._cascade_hysteresis
            selected.append(
                cascade_topk(
                    cand,
                    sel,
                    self._cascade_pool_topk or self.indexer.index_topk,
                    query_positions=query_positions[lo:hi],
                )
            )
        return torch.cat(selected, dim=1)

    def _apply_refresh_prior(
        self, tap: torch.Tensor, cand: torch.Tensor, lo: int, hi: int
    ) -> torch.Tensor:
        """Blend the refresh producer's indexer scores into the tap (S7e).

        Scale-matched additive prior, applied per query row over the VALID
        candidate slots (``cand >= 0``)::

            tap' = tap + lam * decay^(d-1) * std_row(tap) * z_row(prior)

        where ``prior`` is the block producer's lightning-indexer score row
        (``_cascade_prior_row``, ``[B, S, T]``, head-free) gathered at the
        candidate columns, ``z_row`` z-normalizes over the valid slots
        (mean/std over ``cand >= 0``; std clamped at 1e-6), ``std_row(tap)``
        matches the prior to the tap's own per-row spread, and ``d`` is this
        layer's distance from the producer (``_cascade_prior_dist``, 1 =
        immediate consumer).  Invalid (``-1`` sentinel) slots receive 0.

        ``tap``/``cand`` are the ``[B, s, C]`` chunk-loop slices of
        ``_cascade_select``; ``lo:hi`` selects the matching prior rows.
        """
        prior_row = self._cascade_prior_row
        if prior_row is None:
            raise RuntimeError(
                "_apply_refresh_prior requires _cascade_prior_row; the "
                "Transformer forward loop threads it from the block producer"
            )
        if self._cascade_prior_dist < 1:
            raise RuntimeError(
                "refresh-prior blend requires _cascade_prior_dist >= 1, got "
                f"{self._cascade_prior_dist}; the Transformer loop sets 1 at "
                "the producer's immediate consumer"
            )
        # Same gather pattern as cand_scores, but the prior row is head-free:
        # clamp the -1 sentinels for the gather, zero their contribution after
        # normalization.
        prior_c = prior_row[:, lo:hi].float().gather(-1, cand.clamp(min=0))
        invalid = cand < 0
        n = (~invalid).sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        p = prior_c.masked_fill(invalid, 0.0)
        p_mean = p.sum(dim=-1, keepdim=True) / n
        p_var = (
            (p - p_mean).masked_fill(invalid, 0.0).pow(2).sum(dim=-1, keepdim=True) / n
        )
        z = (prior_c - p_mean) / p_var.sqrt().clamp(min=1e-6)
        tap32 = tap.float()
        t = tap32.masked_fill(invalid, 0.0)
        t_mean = t.sum(dim=-1, keepdim=True) / n
        t_var = (
            (t - t_mean).masked_fill(invalid, 0.0).pow(2).sum(dim=-1, keepdim=True) / n
        )
        lam = self._cascade_prior_lambda * (
            self._cascade_prior_decay ** (self._cascade_prior_dist - 1)
        )
        add = (lam * t_var.sqrt()) * z
        return tap32 + add.masked_fill(invalid, 0.0)

    def _cascade_variant_reduce(
        self,
        cand_scores: torch.Tensor,
        cand: torch.Tensor,
        chunk_scores: torch.Tensor,
        incumbent: torch.Tensor,
        value_norms: torch.Tensor | None,
    ) -> torch.Tensor:
        """Variant tap over gathered candidate scores (axis V stage 3).

        Drop-in replacement for ``_tap_reduce`` inside ``_cascade_select``
        when ``_cascade_scoring_variant`` is set.  ``cand_scores`` is
        ``[B, s, H_local, C]`` at the candidate columns; ``chunk_scores``
        the full-width ``[B, s, H_local, T]`` chunk (v4/v5 softmax stats
        come from the ATTEND slice — the leading ``index_topk`` incumbent
        columns — exactly the statistics a fused kernel has);
        ``value_norms`` is the per-key ``[H_local, T]`` map (v5 only).
        Uniform head weights (the screen-validated arms); v6's per-head
        statistics run over the valid candidate slots (duplicate admission
        columns mildly skew them — accepted).  TP: partial head-sums
        all-reduce, mirroring ``_tap_reduce``.
        """
        variant = self._cascade_scoring_variant
        s = cand_scores.float()
        if variant == "v0_relu_sum":
            partial = s.relu().sum(dim=2)
        elif variant == "v1_signed_sum":
            partial = s.sum(dim=2)
        elif variant in ("v4_probs", "v5_probs_vnorm"):
            attend = incumbent[..., : self.indexer.index_topk]
            m, z = attend_softmax_stats(chunk_scores.float(), attend)
            probs = (s - m.unsqueeze(-1)).exp() / z.clamp(min=1e-20).unsqueeze(-1)
            if variant == "v5_probs_vnorm":
                if value_norms is None:
                    raise RuntimeError("cascade v5 scoring requires value_norms")
                vn = value_norms[:, cand.clamp(min=0)]  # [H, B, s, C]
                probs = probs * vn.permute(1, 2, 0, 3)
            partial = probs.sum(dim=2)
        elif variant == "v6_zscore":
            invalid = (cand < 0).unsqueeze(2)  # [B, s, 1, C]
            masked = s.masked_fill(invalid, 0.0)
            n = (~invalid).float().sum(dim=-1, keepdim=True)
            mean = masked.sum(dim=-1, keepdim=True) / n.clamp(min=1)
            var = (
                (masked - mean).masked_fill(invalid, 0.0).pow(2)
                .sum(dim=-1, keepdim=True) / n.clamp(min=1)
            )
            zed = (s - mean) / var.sqrt().clamp(min=1e-6)
            partial = zed.masked_fill(invalid, 0.0).sum(dim=2)
        else:
            raise ValueError(
                f"cascade scoring variant {variant!r} is not supported "
                "(v2/v3 were killed at the screen; nothing dead is wired)"
            )
        return self._variant_combine(partial, "sum", 1.0)

    def _hub_fold_frozen(self, chunk_start: int) -> bool:
        """True when hub folds are frozen for a chunk starting at ``chunk_start``.

        The freeze-ablation knob (docs/prereg_freeze_ablation.md): with
        ``_cascade_hub_freeze_pos`` set, membership statistics accumulated
        from chunks starting at or beyond that absolute position are
        dropped — the buffer state at the freeze boundary is what every
        later selection consumes (the serving frozen-set semantics).
        """
        freeze_pos = self._cascade_hub_freeze_pos
        return freeze_pos is not None and chunk_start >= freeze_pos

    def _cascade_hub_update(
        self,
        cand: torch.Tensor,
        tap: torch.Tensor,
        end_pos: int,
        chunk_start: int = 0,
    ) -> None:
        """Fold this chunk's tap scores into the running per-key hub stat.

        ``cand``/``tap`` are ``[B, s, C]`` aligned candidate indices and tap
        logits.  Aggregators (K3, S7b):

        - ``max`` (the v1 aggregator, adopted at S7b-a): running per-key
          max of tap logits.  Buffer init ``-inf`` — keys never scored can
          never become hubs; a key that once scored high stays admissible
          for the rest of the sequence.
        - ``mass`` / ``mass_decay``: H2O-style accumulated attention mass
          computed on the tap's own candidate distribution — per query row
          the tap logits are softmaxed over valid candidate slots and
          scatter-ADDED.  Buffer init 0.  ``mass_decay`` additionally
          multiplies the buffer by ``_cascade_hub_decay`` once per
          layer-forward (applied at candidate-read time in
          ``_cascade_select``).  Duplicate candidate slots (window keys
          re-listed in admission) double-count by construction — the
          registered definition; such keys are admission-guaranteed
          anyway, so hub selection among NON-window keys is unaffected.

        ``chunk_start`` is the chunk's absolute start position, consumed by
        the freeze gate only (``_hub_fold_frozen``): a frozen fold is a
        no-op, leaving the boundary state to serve all later selections.
        """
        if self._hub_fold_frozen(chunk_start):
            return
        bsz = cand.size(0)
        stat = self._cascade_hub_stat
        init_val = float("-inf") if stat == "max" else 0.0
        buf = self._cascade_hub_scores
        if buf is None or buf.size(0) < bsz or buf.size(1) < end_pos:
            fresh = torch.full(
                (max(bsz, buf.size(0) if buf is not None else 0), end_pos),
                init_val, dtype=torch.float32, device=cand.device,
            )
            if buf is not None:
                fresh[: buf.size(0), : buf.size(1)] = buf
            self._cascade_hub_scores = buf = fresh
        idx = cand.clamp(min=0).reshape(bsz, -1)
        masked = torch.where(
            cand >= 0, tap.float(), torch.full_like(tap.float(), float("-inf"))
        )
        if stat == "max":
            buf[:bsz].scatter_reduce_(
                1, idx, masked.reshape(bsz, -1), reduce="amax",
                include_self=True,
            )
        else:
            # Invalid slots carry -inf logits -> exactly 0 mass after the
            # softmax, so their clamp(min=0) scatter target receives 0.
            mass = masked.softmax(dim=-1)
            buf[:bsz].scatter_add_(1, idx, mass.reshape(bsz, -1))

    def _cascade_hub_update_idx(self, end_pos: int, chunk_start: int = 0) -> None:
        """Fold this layer's FULL lightning-indexer score row into the hub.

        The serving-realizable statistic (``hub_stat="idx_max"``): during
        serving prefill the stock indexer runs at every layer, so the frozen
        selection-history set can only be built from indexer scores (the
        QK^T tap does not exist on the prefill kernel path).  Running per-key
        max over the layer's full score row, reduced over the chunk's query
        rows — richer than the tap statistic (all keys, not just candidates)
        but from a DIFFERENT score space; whether the membership transfers
        is exactly what the pre-registered arm measures.  Buffer init
        ``-inf`` (masked/acausal columns stay ``-inf`` and never become
        hubs).  Called AFTER ``_cascade_select`` so the candidate set the
        layer consumed is the state at entry (research-hub convention).

        ``chunk_start`` is the chunk's absolute start position, consumed by
        the freeze gate only (``_hub_fold_frozen``).
        """
        if self._hub_fold_frozen(chunk_start):
            return
        row = self.last_index_score
        if row is None:
            raise RuntimeError(
                "hub_stat='idx_max' requires the lightning indexer to have "
                "run this forward (prefill-only; incompatible with "
                "skip_indexer_decode consumers)"
            )
        bsz = row.size(0)
        buf = self._cascade_hub_scores
        if buf is None or buf.size(0) < bsz or buf.size(1) < end_pos:
            fresh = torch.full(
                (max(bsz, buf.size(0) if buf is not None else 0), end_pos),
                float("-inf"), dtype=torch.float32, device=row.device,
            )
            if buf is not None:
                fresh[: buf.size(0), : buf.size(1)] = buf
            self._cascade_hub_scores = buf = fresh
        folded = row[..., :end_pos].float().amax(dim=1)  # [B, T]
        buf[:bsz, :end_pos] = torch.maximum(buf[:bsz, :end_pos], folded)
    def _block_candidates(
        self,
        raw_scores: torch.Tensor,
        query_positions: torch.Tensor,
        incumbent: torch.Tensor,
    ) -> torch.Tensor:
        """Coarse-to-fine block admission part ``[B, S, block_admit * block_size]``.

        Per query chunk (bounding the float32 masked copy of the score row to
        ``_tap_query_chunk`` rows, the same budget as the dense-tap path):
        reduce the key axis to per-head block summaries
        (``tap.block_summary_scores``), collapse heads with the SAME tap
        reduction as candidate scoring (``_tap_reduce`` — TP all-reduced, so
        every rank admits the same blocks), and expand the top
        ``block_admit`` blocks to token candidates.

        Ranking function: the raw tap summary, or — when this layer carries
        a fitted entries-head AND the previous layer's summary was threaded
        (``_cascade_prev_block_tap``) — the head's learned ranking over
        NON-incumbent causal blocks (the training universe; incumbent-covered
        blocks are masked out).  ``block_head=None`` is bit-for-bit the raw
        ranking.  With ``_cascade_block_stash`` the reduced summaries are
        retained on ``last_block_tap`` for the forward loop to thread.

        The ``"mean"`` summary is exactly the score of the block's mean
        cached key (bilinearity), so a serving realization scores a
        ``[n_blocks, kv_lora_rank + rope]`` mean-latent store with the same
        MQA primitive; this eager emulation reduces the already-materialized
        score row instead.  The classic weighted-ReLU tap is the only
        head reduction wired here — ``set_cascade_mode`` rejects the
        combination with ``scoring_variant``.
        """
        head = self._cascade_block_head
        prev_full = self._cascade_prev_block_tap
        use_head = head is not None and prev_full is not None
        out: list[torch.Tensor] = []
        taps: list[torch.Tensor] = []
        for lo in range(0, raw_scores.size(1), self._tap_query_chunk):
            hi = min(lo + self._tap_query_chunk, raw_scores.size(1))
            summary = block_summary_scores(
                raw_scores[:, lo:hi],
                query_positions[lo:hi],
                self._cascade_block_size,
                self._cascade_block_stat,
            )  # [B, s, H_local, n_blocks]
            block_tap = self._tap_reduce(summary)  # [B, s, n_blocks]
            if self._cascade_block_stash:
                taps.append(block_tap)
            if use_head:
                features, covered = entries_head_features(
                    block_tap,
                    prev_full[:, lo:hi],
                    incumbent[:, lo:hi],
                    query_positions[lo:hi],
                    self._cascade_block_size,
                )
                rank_scores = entries_head_scores(features, head)
                rank_scores = rank_scores.masked_fill(covered, float("-inf"))
            else:
                rank_scores = block_tap
            out.append(
                block_token_candidates(
                    rank_scores,
                    query_positions[lo:hi],
                    self._cascade_block_size,
                    self._cascade_block_admit,
                    raw_scores.size(-1),
                )
            )
        self.last_block_tap = torch.cat(taps, dim=1) if taps else None
        return torch.cat(out, dim=1)

    def _indexer_pool(self) -> torch.Tensor:
        """Score-sorted top-``pool_topk`` of the lightning indexer's scores.

        Pool/attend counterpart of ``last_topk_indices``: on layers where the
        true indexer ran (layer 0, refresh producers), the threaded pool is
        the indexer's own top-``pool_topk`` — the widest exact candidate set
        available — rather than its top-``index_topk``.  ``-1``-padded where
        fewer finite-scored keys exist.
        """
        if not self._cascade_pool_topk:
            raise RuntimeError("_indexer_pool requires _cascade_pool_topk > 0")
        if self.last_index_score is None:
            raise RuntimeError(
                "_indexer_pool requires the lightning indexer's scores; "
                "this layer did not run the indexer"
            )
        pool = self._cascade_pool_topk
        n_keys = self.last_index_score.size(-1)
        scores, idx = self.last_index_score.topk(min(pool, n_keys), dim=-1)
        idx = torch.where(
            torch.isfinite(scores), idx, torch.full_like(idx, -1)
        )
        if idx.size(-1) < pool:
            pad = torch.full(
                (*idx.shape[:-1], pool - idx.size(-1)), -1,
                dtype=idx.dtype, device=idx.device,
            )
            idx = torch.cat([idx, pad], dim=-1)
        return idx

    def _latent_refresh_score(
        self,
        bsz: int,
        end_pos: int,
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score ALL keys via compressed-latent dot product (cost-per-score).

        At refresh layers, replaces the full indexer with a single matmul
        over the 512-dim compressed MLA latents already in ``kv_cache``.

        Args:
            bsz: Batch size.
            end_pos: One past the last populated position in the KV cache.
            query_positions: ``[S]`` int tensor of query token positions
                (for decode S=1, for prefill S=chunk_len).

        Returns:
            topk_indices: ``[B, S, index_topk]`` int64 top-k key indices.
            full_scores: ``[B, S, L_max]`` float32 score row (padded to the
                indexer's ``max_len`` with ``-inf`` beyond ``end_pos``).
        """
        # Query latents: kv_cache at query positions [B, S, kv_lora_rank].
        q_lat = self.kv_cache[:bsz, query_positions]  # [B, S, 512]
        # All key latents up to end_pos [B, end_pos, 512].
        k_lat = self.kv_cache[:bsz, :end_pos]
        # Score: [B, S, end_pos] via matmul (no learned projection).
        scores = torch.bmm(
            q_lat.float(), k_lat.float().transpose(1, 2)
        )  # [B, S, end_pos]

        # Pad to the indexer's max_len so last_index_score has a consistent
        # shape for downstream consumers (tap, _cascade_select, tracer).
        max_len = self.kv_cache.size(1)
        if end_pos < max_len:
            pad = torch.full(
                (bsz, scores.size(1), max_len - end_pos),
                float("-inf"),
                dtype=scores.dtype,
                device=scores.device,
            )
            full_scores = torch.cat([scores, pad], dim=-1)
        else:
            full_scores = scores

        # Top-k selection.
        k = self.indexer.index_topk
        n_keys = min(end_pos, scores.size(-1))
        topk_k = min(k, n_keys)
        topk_scores, topk_indices = scores[..., :end_pos].topk(topk_k, dim=-1)
        topk_indices = topk_indices.to(torch.int64)
        # Pad top-k if fewer keys than index_topk.
        if topk_k < k:
            pad_idx = torch.full(
                (*topk_indices.shape[:-1], k - topk_k),
                -1,
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
            topk_indices = torch.cat([topk_indices, pad_idx], dim=-1)

        return topk_indices, full_scores

    def _indexer_striped_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Temporally-striped refresh scan for one eager decode step.

        Drives :meth:`Indexer.forward_striped` with the rotating spans of
        :func:`refresh_stripe_spans` over a running per-layer score cache.
        The FIRST striped step (or any discontinuity: fresh KV cache,
        changed batch size, a prefill chunk between decode steps) re-seeds
        the cache with a full O(L) scan — one full-refresh cost — so the
        stripe never threads a score row with untracked holes; steady-state
        steps then score O(L/period + fresh_tail) keys.

        Staleness contract: column ``j`` of the threaded score row equals
        ``s(q_t_j, k_j)`` where ``t_j`` is the last step that scored ``j``
        (at most ``period - 1`` steps old; tail columns are always current).
        Keys are immutable once cached — only the QUERY is stale.
        """
        period = self._refresh_stripe_period
        if period <= 0:
            raise RuntimeError(
                "_indexer_striped_decode requires _refresh_stripe_period > 0"
            )
        bsz = x.size(0)
        alloc_len = self.indexer.k_cache.size(1)
        cache = self._refresh_stripe_cache
        need_seed = (
            cache is None
            or cache.size(0) != bsz
            or cache.size(1) != alloc_len
            or cache.device != x.device
            or end_pos != self._refresh_stripe_expected_end
        )
        if need_seed:
            cache = torch.full(
                (bsz, alloc_len), float("-inf"),
                dtype=torch.float32, device=x.device,
            )
            self._refresh_stripe_cache = cache
            # Frozen width: derived from the ALLOCATED length, so a key's
            # slice index never changes as the prefix grows and every key is
            # revisited exactly once per period.
            self._refresh_stripe_width = -(-alloc_len // period)  # ceil div
            self._refresh_stripe_step = 0
            spans = [(0, end_pos)]  # cold-start full scan seeds the cache
        else:
            spans = refresh_stripe_spans(
                self._refresh_stripe_step,
                period,
                self._refresh_stripe_width,
                end_pos,
                self._refresh_stripe_fresh_tail,
            )
        topk_indices, index_score = self.indexer.forward_striped(
            x, qr, start_pos, freqs_cis, spans, cache
        )
        self.last_refresh_stripe_spans = spans
        self._refresh_stripe_step += 1
        self._refresh_stripe_expected_end = end_pos + 1
        if self._refresh_stripe_selfcheck:
            # Validation-only (bench/track_sched_mini.py): a same-state full
            # scan must agree exactly on the columns this step scored; the
            # stale-column gap measures the staleness the stripe introduces.
            # The reference forward rewrites the same k-cache row with
            # identical values, so the model state is unchanged.
            _, ref_score = self.indexer(x, qr, start_pos, freqs_cis, None)
            fresh = torch.zeros(end_pos, dtype=torch.bool, device=x.device)
            for lo, hi in spans:
                fresh[lo:hi] = True
            diff = (index_score - ref_score).abs()[:, 0]  # [B, end_pos]
            finite_stale = ~fresh & torch.isfinite(index_score[:, 0]).all(0)
            self.last_refresh_stripe_selfcheck = {
                "fresh_max_abs": float(diff[:, fresh].max().item())
                if bool(fresh.any()) else 0.0,
                "stale_max_abs": float(diff[:, finite_stale].max().item())
                if bool(finite_stale.any()) else 0.0,
                "n_fresh": int(fresh.sum().item()),
                "n_stale_finite": int(finite_stale.sum().item()),
                "n_unscored": int(
                    (~torch.isfinite(index_score[:, 0])).sum().item()
                ),
            }
        return topk_indices, index_score

    def _use_fused_decode(self, bsz: int, end_pos: int) -> bool:
        """Whether this single-token decode step takes the fused-kernel path."""
        if self._fused_decode_state is None:
            return False
        if self._disable_indexer_mask:
            raise RuntimeError(
                "fused cascade decode is incompatible with set_indexer_frozen(True)"
            )
        if (
            self._collect_tap_topk
            or self._collect_attn_mass
            or self._collect_tap_features
        ):
            raise RuntimeError(
                "fused cascade decode does not support instrumentation "
                "collectors; disable them or call set_cascade_fused_decode(None)"
            )
        if self._kv_joint.dtype != torch.bfloat16:
            raise RuntimeError(
                "fused cascade decode requires a bfloat16 KV cache, got "
                f"{self._kv_joint.dtype}"
            )
        # The kernel path needs a full candidate pool of distinct causal keys
        # plus a complete admission window; shorter prefixes stay on the
        # eager path (where selection is trivially near-dense anyway).
        pool = self._cascade_pool_topk or self.indexer.index_topk
        need = pool + self._cascade_window + self._cascade_n_sink
        return end_pos - 1 >= need

    def _mqa_decode_fused(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        end_pos: int,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token sparse decode through the fused split-K kernel.

        Replaces the eager full-prefix score einsum + index-mask softmax + PV
        with one fused-kernel launch: sparse attention over the threaded index
        set, and — on layers whose cascade set the next layer consumes — the
        tap planes over incumbents + admission window emitted from the same
        QK^T pass (``with_select=3``).  Tensor parallelism shards the heads,
        so the per-rank planes are partial head-sums: the cascade epilogue
        collapses them locally, the sink candidates are appended, and one
        all-reduce over the ``[topk+win+nsink]`` candidate logits precedes the
        final top-k (the TP-correct counterpart of the in-kernel select fold).

        At ``bsz > 1`` each batch row is an independent sequence: the kernel's
        batched entry maps query row ``b`` to its own KV cache row via
        ``kv_stride`` and every row decodes at the same position
        ``end_pos-1``; the selection math downstream is row-generic.  The
        ``bsz == 1`` path launches the original single-sequence entries
        unchanged.  Returns the pre-``wo`` attention output
        ``[bsz, 1, h_local, v]``.
        """
        st = self._fused_decode_state
        bsz = x.size(0)
        topk = self.indexer.index_topk
        pool = self._cascade_pool_topk or topk
        win = self._cascade_window
        n_sink = self._cascade_n_sink
        q_pos = end_pos - 1

        # Producer layers (layer 0, periodic refresh) still run the lightning
        # indexer to hand an exact set to the next layer; consumers skip it.
        if self._cascade_skip_indexer_decode:
            if self._lag_external_topk is None:
                raise RuntimeError(
                    "skip_indexer_decode requires a cascade set from the "
                    "previous layer; layer 0 / refresh producers must not "
                    "set _cascade_skip_indexer_decode"
                )
            topk_indices = None
            self.last_index_score = None
            self.last_topk_indices = None
        else:
            topk_indices, self.last_index_score = self.indexer(
                x, qr, start_pos, freqs_cis, None
            )
            topk_scores = self.last_index_score.gather(
                -1, topk_indices.clamp(min=0)
            )
            self.last_topk_indices = torch.where(
                torch.isfinite(topk_scores),
                topk_indices,
                torch.full_like(topk_indices, -1),
            )

        # The candidate pool this layer consumes: the threaded set from the
        # previous layer, or — on cold-start producers — the indexer's own
        # top-pool (its top-k when decoupling is off).  Every pool is
        # score-sorted, so the attend set is its leading index_topk slice.
        cand_pool = (
            self._lag_external_topk
            if self._lag_external_topk is not None
            else (self._indexer_pool() if pool != topk else topk_indices)
        )
        if cand_pool.size(-1) != pool:
            raise RuntimeError(
                f"candidate pool has {cand_pool.size(-1)} slots, expected "
                f"pool={pool}"
            )
        # Kernel sentinel: invalid slots use Skv (= end_pos), acausal for the
        # query and safe to load (masked lanes read row 0 instead).
        idx32 = cand_pool.view(bsz, pool).int()
        idx32 = torch.where(
            idx32 < 0, torch.full_like(idx32, end_pos), idx32
        ).contiguous()

        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = (
            self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        )
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_abs = torch.einsum(
            "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
        )
        q_cat = torch.cat([q_abs, q_pe], dim=-1)  # [bsz, 1, h_local, 576] bf16
        # The kernel reads Q as [S, H, dqk] rows; a [bsz, 1, ...] tensor has
        # the identical layout viewed as [1, bsz, ...].
        q_rows = q_cat.view(1, bsz, self.n_local_heads, -1)

        h_local = self.n_local_heads
        h_pad = pad_heads(h_local)
        buf = st.scratch(q_cat.device, h_pad, pool, win, bsz)
        if h_pad != h_local:
            buf["q_pad"][..., :h_local, :].copy_(q_rows)
            q_kernel = buf["q_pad"]
        else:
            q_kernel = q_rows.contiguous()

        # bsz == 1 keeps the certified single-sequence entries; bsz > 1 uses
        # the batched entry, whose per-row KV base is the joint cache's row
        # stride (elements) — the whole [B, T_alloc, 576] buffer is passed so
        # row b's keys live at b*kv_stride.
        kv = self._kv_joint[:1, :end_pos] if bsz == 1 else self._kv_joint
        kv_stride = self._kv_joint.size(1) * self._kv_joint.size(2)
        nblk = h_pad // HEAD_BLOCK
        tapw = self._fused_tapw
        if tapw is not None and tapw.device != q_cat.device:
            tapw = self._fused_tapw = tapw.to(q_cat.device)

        if self._fused_produce_tap:
            variant = self._cascade_scoring_variant
            if variant == "v4_probs":
                # In-kernel exp-tap (mode 5): per-head raw-score planes + exp
                # epilogue reproduce the eager v4 scoring — per-head m/Z over
                # the attend slice are the kernel's own softmax statistics,
                # and the epilogue output is exactly this rank's partial
                # head-sum of normalized probs, so the existing TP all-reduce
                # of union logits stays correct.  Uniform weights only (the
                # verified arm; set_cascade_mode already rejects
                # head_weights + scoring_variant).  bsz > 1 routes through
                # the batched exp-tap entry (per-row KV via kv_stride, every
                # row at position end_pos-1; certified bitwise row-equal by
                # validate_exptap_batched).
                if pool != topk:
                    raise RuntimeError(
                        "fused v4_probs decode does not support pool/attend "
                        "decoupling: the exp-tap entry attends the full "
                        "candidate set (attend_k == topk)"
                    )
                if bsz == 1:
                    st.ext.launch_fused_sparse_splitk_exptap(
                        q_kernel, kv, idx32,
                        buf["idx_q"], buf["idx_k"], buf["idx_w"],
                        buf["idx_scale"],
                        buf["part_acc"], buf["part_m"], buf["part_l"],
                        buf["sel_slab_h"], buf["hist"], buf["out"],
                        1, end_pos, h_pad, HEAD_BLOCK, topk, st.nsplit,
                        self.softmax_scale, win,
                    )
                    st.ext.launch_cascade_exp_epilogue(
                        buf["sel_slab_h"], buf["part_m"], buf["part_l"],
                        idx32, buf["tap_out"], buf["exp_stats"], 1, nblk,
                        topk, win, end_pos, st.nsplit,
                    )
                else:
                    st.ext.launch_fused_sparse_splitk_exptap_batch(
                        q_kernel, kv, idx32,
                        buf["idx_q"], buf["idx_k"], buf["idx_w"],
                        buf["idx_scale"],
                        buf["part_acc"], buf["part_m"], buf["part_l"],
                        buf["sel_slab_h"], buf["hist"], buf["out"],
                        bsz, end_pos, h_pad, HEAD_BLOCK, topk, st.nsplit,
                        self.softmax_scale, win, kv_stride,
                    )
                    st.ext.launch_cascade_exp_epilogue_batch(
                        buf["sel_slab_h"], buf["part_m"], buf["part_l"],
                        idx32, buf["tap_out"], buf["exp_stats"], bsz, nblk,
                        topk, win, end_pos, st.nsplit,
                    )
            elif bsz > 1:
                # One batched launch covers all three single-sequence
                # variants: attend_k=topk handles both the decoupled pool
                # (pool > topk) and the plain cascade (pool == topk), and the
                # entry takes explicit weights (ones = uniform).
                st.ext.launch_fused_sparse_splitk_batch(
                    q_kernel, kv, idx32,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"],
                    tapw if tapw is not None else buf["tapw_ones"],
                    buf["out"],
                    bsz, end_pos, h_pad, HEAD_BLOCK, pool, topk, st.nsplit,
                    self.softmax_scale, 3, win, kv_stride,
                )
                st.ext.launch_cascade_epilogue_batch(
                    buf["slab"], idx32, buf["tap_out"], bsz, nblk, pool, win,
                    end_pos,
                )
            elif pool != topk:
                # Decoupled pool: attention over the leading attend_k=topk
                # columns, tap planes over the whole pool.  TapW is required
                # by the entry; set_cascade_fused_decode installs ones for
                # the uniform tap.
                if tapw is None:
                    raise RuntimeError(
                        "pool/attend decoupling requires _fused_tapw; "
                        "reconfigure via set_cascade_fused_decode"
                    )
                st.ext.launch_fused_sparse_splitk_pool(
                    q_kernel, kv, idx32,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"], tapw, buf["out"],
                    1, end_pos, h_pad, HEAD_BLOCK, pool, topk, st.nsplit,
                    self.softmax_scale, win,
                )
            elif tapw is not None:
                st.ext.launch_fused_sparse_splitk_tapw(
                    q_kernel, kv, idx32,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"], tapw, buf["out"],
                    1, end_pos, h_pad, HEAD_BLOCK, topk, st.nsplit,
                    self.softmax_scale, win,
                )
            else:
                st.ext.launch_fused_sparse_splitk(
                    q_kernel, kv, idx32,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"], buf["out"],
                    1, end_pos, h_pad, HEAD_BLOCK, topk, st.nsplit,
                    self.softmax_scale, 3, 1, win,
                )
            if bsz == 1 and variant != "v4_probs":
                st.ext.launch_cascade_epilogue(
                    buf["slab"], idx32, buf["tap_out"], 1, nblk, pool, win,
                    end_pos,
                )
            cand = buf["tap_out"]
            if n_sink > 0:
                # Sink candidates ride the same all-reduce: their tap logits
                # use the identical per-head weighting (and, like the kernel
                # planes, no softmax_scale factor — a positive rescale that
                # cannot change the selected set).  Under v4 the scale is NOT
                # free (it sits inside the exp), so the v4 branch matches the
                # kernel exactly instead.
                if variant == "v4_probs" and bsz == 1:
                    # Eager v4 scores sinks through the same exp-prob
                    # reduction as every other candidate, so the fused path
                    # must too: score them against the epilogue's exported
                    # per-head stats, in the kernel's log2-scaled domain
                    # (score * softmax_scale * log2e — the kernel folds
                    # softmax_scale and the log2 conversion into s2, whereas
                    # the ReLU-sum path above applies no scale at all).
                    # q_kernel includes the zero-pad head rows, so sinks
                    # receive the same uniform pad-head mass (1/Z_pad) every
                    # in-plane candidate carries — commensurate by
                    # construction; the per-rank head-sum all-reduces
                    # exactly like the plane columns.
                    k_sink = kv[0, :n_sink].float()  # [n_sink, 576]
                    q_heads = q_kernel[0, 0].float()  # [h_pad, 576]
                    s2_sink = (q_heads @ k_sink.t()) * (
                        self.softmax_scale * 1.4426950408889634
                    )  # [h_pad, n_sink], log2 domain
                    m2g = buf["exp_stats"][0, 0].view(-1, 1)   # [h_pad, 1]
                    invz = buf["exp_stats"][0, 1].view(-1, 1)  # [h_pad, 1]
                    sink_vals = torch.where(
                        invz > 0,
                        torch.exp2(s2_sink - m2g) * invz,
                        torch.zeros_like(s2_sink),
                    ).sum(0, keepdim=True)  # [1, n_sink]
                elif variant == "v4_probs":
                    # Batched exp-tap sinks: identical recipe, one KV row and
                    # one [2, h_pad] ExpStats block per sequence (the batched
                    # epilogue exports per-row stats).
                    k_sink = self._kv_joint[:bsz, :n_sink].float()  # [B, n_sink, 576]
                    q_heads = q_kernel[0].float()  # [B, h_pad, 576]
                    s2_sink = torch.einsum(
                        "bhd,bnd->bhn", q_heads, k_sink
                    ) * (
                        self.softmax_scale * 1.4426950408889634
                    )  # [B, h_pad, n_sink], log2 domain
                    m2g = buf["exp_stats"][:, 0, :].unsqueeze(-1)   # [B, h_pad, 1]
                    invz = buf["exp_stats"][:, 1, :].unsqueeze(-1)  # [B, h_pad, 1]
                    sink_vals = torch.where(
                        invz > 0,
                        torch.exp2(s2_sink - m2g) * invz,
                        torch.zeros_like(s2_sink),
                    ).sum(1)  # [B, n_sink]
                elif bsz == 1:
                    k_sink = kv[0, :n_sink].float()  # [n_sink, 576]
                    q_heads = q_cat[0, 0].float()  # [h_local, 576]
                    s_sink = (q_heads @ k_sink.t()).relu()  # [h_local, n_sink]
                    w_real = (
                        tapw[:h_local].view(-1, 1)
                        if tapw is not None
                        else s_sink.new_ones(h_local, 1)
                    )
                    sink_vals = (s_sink * w_real).sum(0, keepdim=True)  # [1, n_sink]
                else:
                    k_sink = self._kv_joint[:bsz, :n_sink].float()  # [B, n_sink, 576]
                    q_heads = q_cat[:, 0].float()  # [B, h_local, 576]
                    s_sink = torch.einsum(
                        "bhd,bnd->bhn", q_heads, k_sink
                    ).relu()  # [B, h_local, n_sink]
                    w_real = (
                        tapw[:h_local].view(1, -1, 1)
                        if tapw is not None
                        else s_sink.new_ones(1, h_local, 1)
                    )
                    sink_vals = (s_sink * w_real).sum(1)  # [B, n_sink]
                w_lo = max(q_pos + 1 - win, 0)
                dup = sink_duplicate_mask(idx32, n_sink, w_lo)
                sink_vals = sink_vals.masked_fill(dup, float("-inf"))
                cand = torch.cat([cand, sink_vals], dim=1)
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                dist.all_reduce(cand)
            if self._cascade_hysteresis_fused > 0.0:
                # Selection-only incumbent bias, in tap-plane units.  The
                # eager delta is NOT commensurate with these planes (per-head
                # ReLU sums of raw QK vs the eager tap's head mean/weighted
                # sum of softmax_scale-multiplied scores), so this attr holds
                # the converted value installed by set_cascade_fused_decode
                # under the explicit hysteresis_scale_matched opt-in (H1',
                # design_space_prereg.md).  Applied AFTER the all-reduce (the
                # reduce sums per-rank partial head-sums; a per-rank bias
                # would scale with world size).  The epilogue already -inf'd
                # window copies of incumbent keys, so the bias lands on the
                # surviving copy.
                cand = cand.clone()
                cand[:, :pool] += self._cascade_hysteresis_fused
            next_idx = fused_cascade_select(cand, idx32, q_pos, pool, win, n_sink)
            self.last_cascade_topk = next_idx.view(bsz, 1, pool)
        else:
            # Refresh producers and the final layer: sparse attention only
            # (with_select=0) over the attend slice; the next set, if any,
            # is the indexer's own.
            idx_attend = idx32[:, :topk].contiguous()
            if bsz > 1:
                st.ext.launch_fused_sparse_splitk_batch(
                    q_kernel, kv, idx_attend,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"], buf["tapw_ones"], buf["out"],
                    bsz, end_pos, h_pad, HEAD_BLOCK, topk, topk, st.nsplit,
                    self.softmax_scale, 0, 0, kv_stride,
                )
            else:
                st.ext.launch_fused_sparse_splitk(
                    q_kernel, kv, idx_attend,
                    buf["idx_q"], buf["idx_k"], buf["idx_w"], buf["idx_scale"],
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    buf["slab"], buf["hist"], buf["out"],
                    1, end_pos, h_pad, HEAD_BLOCK, topk, st.nsplit,
                    self.softmax_scale, 0, 1, 0,
                )
        # [1, bsz, h_pad, 512] -> [bsz, 1, h_local, 512] (pure view + slice;
        # identical values to the legacy [1,1,...] expression at bsz == 1).
        out = buf["out"][0, :, :h_local].unsqueeze(1)
        return torch.einsum("bshc,hdc->bshd", out, wkv_b[:, -self.v_head_dim :])

    def _reuse_fused_active(self, bsz: int, end_pos: int) -> bool:
        """Whether this decode step routes through the reuse_fused kernel path.

        True only on refresh PRODUCERS with an incoming set: layer 0
        (cold start, attends its own fresh set — a chicken-and-egg for a
        select-and-attend launch) and all verbatim consumers stay on the
        eager path, which reuse_fused leaves untouched.
        """
        if self._reuse_fused_state is None:
            return False
        if self._cascade_skip_indexer_decode:
            return False  # verbatim consumer: eager propagation path
        if self._lag_external_topk is None:
            return False  # layer-0 cold start: eager indexer + attention
        if (
            self._collect_tap_topk
            or self._collect_attn_mass
            or self._collect_tap_features
        ):
            raise RuntimeError(
                "reuse_fused decode does not support instrumentation "
                "collectors; disable them or call "
                "set_reuse_fused_decode(None)"
            )
        if self._kv_joint.dtype != torch.bfloat16:
            raise RuntimeError(
                "reuse_fused decode requires a bfloat16 KV cache, got "
                f"{self._kv_joint.dtype}"
            )
        # Short prefixes stay eager (the kernel path wants a full set of
        # distinct causal keys; eager reuse is semantically identical).
        return end_pos - 1 >= self.indexer.index_topk

    def _mqa_decode_reuse_fused(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        end_pos: int,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Refresh-layer decode: fused select-and-attend (``with_select=1``).

        One kernel launch per sequence row runs sparse attention over the
        INCOMING verbatim set while the in-kernel fp8 indexer scores the
        full prefix into ``SelSlab``; the host top-k of the slab is this
        layer's fresh set (``last_topk_indices``), which the next layer
        consumes via ``_cascade_produced`` — exactly eager ``reuse``'s
        lag semantics with the serial [indexer -> attention] pair
        replaced by one launch.  Returns the pre-``wo`` attention output
        ``[bsz, 1, h_local, v]``.
        """
        st = self._reuse_fused_state
        bsz = x.size(0)
        topk = self.indexer.index_topk
        q_pos = end_pos - 1

        # fp8 indexer operands: the front half of Indexer.forward (appends
        # the new token's index key), padded to the kernel's 16-head block
        # (zero-weight pad heads are exact under the ReLU-weighted sum).
        idx_q_fp8, idx_w = self.indexer.stage_fp8_operands(
            x, qr, start_pos, freqs_cis
        )
        idx_q_fp8, idx_w, h_idx_pad = pad_index_operands(
            idx_q_fp8, idx_w.float()
        )
        idx_w = idx_w.contiguous()
        idx_q_fp8 = idx_q_fp8.contiguous()

        incoming = self._lag_external_topk
        if incoming.size(-1) != topk:
            raise RuntimeError(
                f"reuse_fused incoming set has {incoming.size(-1)} slots, "
                f"expected index_topk={topk} (pool/attend decoupling is not "
                "supported under verbatim reuse)"
            )
        idx32 = incoming.view(bsz, topk).int()
        idx32 = torch.where(
            idx32 < 0, torch.full_like(idx32, end_pos), idx32
        ).contiguous()

        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = (
            self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        )
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_abs = torch.einsum(
            "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
        )
        q_cat = torch.cat([q_abs, q_pe], dim=-1)  # [bsz, 1, h_local, 576] bf16
        q_rows = q_cat.view(1, bsz, self.n_local_heads, -1)

        h_local = self.n_local_heads
        h_pad = pad_heads(h_local)
        t_alloc = self._kv_joint.size(1)
        rowloop = bsz > 1 and self._reuse_fused_rowloop
        buf = st.scratch(
            q_cat.device, h_pad, t_alloc, bsz=1 if rowloop or bsz == 1 else bsz
        )
        # SelSlab sizing is host-asserted too (belt and braces with the
        # launcher's with_select 1/2 check); fail loud, never OOB.  The
        # requirement is PATH-AWARE: the rowloop reuses one [end_pos]
        # slice per row, the batched entry packs [bsz, end_pos] (job
        # 24699: the unconditional bsz*end_pos assert wrongly failed the
        # rowloop arm against its correctly-sized bsz=1 scratch).
        slab_need = reuse_fused_slab_need(bsz, end_pos, rowloop)
        if buf["sel_slab"].numel() < slab_need:
            raise RuntimeError(
                f"reuse_fused SelSlab has {buf['sel_slab'].numel()} slots, "
                f"needs {slab_need} (bsz={bsz}, end_pos={end_pos}, "
                f"rowloop={rowloop}); call reset_kv_cache first"
            )

        k_cache = self.indexer.k_cache
        k_scales = self.indexer.k_scale_cache
        if k_scales.shape[-1] != 1:
            raise RuntimeError(
                "reuse_fused expects one fp8 scale per index key "
                f"(head_dim == quant block size); got {k_scales.shape[-1]}"
            )

        if bsz == 1 or rowloop:
            # Single row, or the R2a per-row control arm (bitwise-equal to
            # the batched rows by the kernel's row-slice contract, at a
            # priced per-launch overhead).
            sel_slab = buf["sel_slab"][:end_pos]
            outs = []
            fresh_sets = []
            slab_rows = []
            for b in range(bsz):
                if h_pad != h_local:
                    buf["q_pad"][0, 0, :h_local].copy_(q_rows[0, b])
                    q_kernel = buf["q_pad"][:, :1]
                else:
                    q_kernel = q_rows[:, b : b + 1].contiguous()
                # Row b's caches: linear [T, ...] layouts from the row base
                # (contiguous slices of the [B, T, ...] buffers).
                kv_b = self._kv_joint[b : b + 1, :end_pos]
                k_b = k_cache[b, :end_pos]
                scale_b = k_scales[b, :end_pos].reshape(end_pos).contiguous()
                st.ext.launch_fused_sparse_splitk(
                    q_kernel, kv_b, idx32[b : b + 1],
                    idx_q_fp8[b, 0], k_b, idx_w[b, 0], scale_b,
                    buf["part_acc"], buf["part_m"], buf["part_l"],
                    sel_slab, buf["hist"], buf["out"][:, :1],
                    1, end_pos, h_pad, h_idx_pad, topk, st.nsplit,
                    self.softmax_scale, 1, 1, 0,
                )
                outs.append(buf["out"][0, :1, :h_local].clone())
                fresh_sets.append(sel_slab.topk(topk)[1])
                slab_rows.append(sel_slab.clone())
            out_rows = torch.cat(outs, dim=0)  # [bsz, h_local, 512]
            fresh_idx = torch.stack(fresh_sets).view(bsz, 1, topk)
            slab_view = torch.stack(slab_rows)  # [bsz, end_pos]
        else:
            # R2b batched entry: one launch for all rows, per-row
            # IndexK/IndexKScale strides (certified bitwise row-equal to
            # the legacy launches by validate_batched's idx-select gate).
            if h_pad != h_local:
                buf["q_pad"][0, :, :h_local].copy_(q_rows[0])
                q_kernel = buf["q_pad"]
            else:
                q_kernel = q_rows.contiguous()
            kv_stride = self._kv_joint.size(1) * self._kv_joint.size(2)
            idx_k_stride = k_cache.size(1) * k_cache.size(2)
            idx_scale_stride = k_scales.size(1) * k_scales.size(2)
            sel_slab = buf["sel_slab"][: bsz * end_pos]
            st.ext.launch_fused_sparse_splitk_batch_idx(
                q_kernel, self._kv_joint, idx32,
                idx_q_fp8.view(bsz, h_idx_pad, -1), k_cache,
                idx_w.view(bsz, h_idx_pad),
                k_scales.reshape(k_scales.size(0), -1),
                buf["part_acc"], buf["part_m"], buf["part_l"],
                sel_slab, buf["hist"], buf["out"],
                bsz, end_pos, h_pad, h_idx_pad, topk, st.nsplit,
                self.softmax_scale, 1, 1, 0,
                kv_stride, idx_k_stride, idx_scale_stride,
            )
            slab_view = sel_slab.view(bsz, end_pos)
            fresh_idx = slab_view.topk(topk, dim=-1)[1].view(bsz, 1, topk)
            slab_view = slab_view.clone()
            out_rows = buf["out"][0, :, :h_local].clone()
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Same determinism tail as the eager indexer: rank 0's
            # selection is authoritative (idx operands are replicated).
            fresh_bcast = fresh_idx.clone()
            dist.broadcast(fresh_bcast, src=0)
            if not torch.equal(fresh_idx, fresh_bcast):
                fresh_idx = fresh_bcast
        self.last_topk_indices = fresh_idx
        self.last_index_score = slab_view.view(bsz, 1, end_pos)

        out = out_rows.view(bsz, 1, h_local, 512)
        return torch.einsum("bshc,hdc->bshd", out, wkv_b[:, -self.v_head_dim :])

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if act_quant is not None:
            kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
            kv = (
                (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1))
                .to(kv.dtype)
                .view_as(kv)
            )
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if (
            mask is not None and start_pos == 0
        ):  # MHA prefill (first chunk or full sequence)
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(
                bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

            if self._collect_tap_topk:
                self._capture_tap_topk(
                    scores, torch.arange(seqlen, device=x.device)
                )

            if self._collect_attn_mass:
                dense_scores = (scores + mask.unsqueeze(0).unsqueeze(2)).softmax(dim=-1)
                self.last_dense_attn_mass = dense_scores.sum(dim=2).sum(dim=1)
                nwq = self._n_window_queries
                if nwq is not None:
                    if nwq > 0:
                        self.last_dense_attn_mass_window = (
                            dense_scores[:, -nwq:, :, :].sum(dim=2).sum(dim=1)
                        )
                    # nwq == 0 → chunk is outside window, don't set windowed mass
                else:
                    w = self._attn_mass_window
                    if seqlen > w:
                        self.last_dense_attn_mass_window = (
                            dense_scores[:, -w:, :, :].sum(dim=2).sum(dim=1)
                        )
                    else:
                        self.last_dense_attn_mass_window = self.last_dense_attn_mass.clone()
                del dense_scores

            # indexer
            if self._cascade_latent_refresh:
                # Cost-per-score: compressed-latent dot product replaces the
                # full indexer on this refresh layer (MHA first chunk).
                topk_indices, self.last_index_score = (
                    self._latent_refresh_score(
                        bsz, end_pos,
                        torch.arange(start_pos, end_pos, device=x.device),
                    )
                )
            else:
                topk_indices, self.last_index_score = self.indexer(
                    x, qr, start_pos, freqs_cis, mask, kv_joint=self._kv_joint
                )
            topk_scores = self.last_index_score.gather(
                -1, topk_indices.clamp(min=0)
            )
            self.last_topk_indices = torch.where(
                torch.isfinite(topk_scores),
                topk_indices,
                torch.full_like(topk_indices, -1),
            )
            if (
                self._collect_tap_features
                and start_pos + seqlen == self._tap_feature_end
            ):
                self._capture_tap_features(scores)
            if (
                self._collect_variant_topk
                and start_pos + seqlen == self._tap_feature_end
            ):
                # S7a (K1) source components: the content-only plane from
                # the un-absorbed heads — capture-only cost, final chunk.
                variant_nope = torch.einsum(
                    "bshd,bthd->bsht", q_nope, k_nope
                ).mul_(self.softmax_scale)
                self._capture_variant_scores(scores, variant_nope)
                del variant_nope
            if not self._disable_indexer_mask:
                candidate_pool = (
                    self._lag_external_topk
                    if self._lag_external_topk is not None
                    else topk_indices
                )
                if self._cascade_mode:
                    if self._cascade_pool_topk and self._lag_external_topk is None:
                        # Cold-start producer: widen the incumbent to the
                        # indexer's own top-pool so the threaded pool is
                        # full width from the first layer.
                        candidate_pool = self._indexer_pool()
                    # Raw (pre-mask) scores: the tap re-ranks the keys this
                    # layer attends, plus the admission window, for layer L+1.
                    if self._cascade_same_layer and self.last_index_score is None:
                        raise RuntimeError(
                            "same-layer cascade scoring requires the "
                            "lightning indexer to have run this forward"
                        )
                    if self._cascade_freeze_admit:
                        # ρ_admit freeze: skip admission; _cascade_produced
                        # threads the incumbent (candidate_pool) forward.
                        self.last_cascade_topk = None
                    else:
                        self.last_cascade_topk = self._cascade_select(
                            scores, candidate_pool,
                            torch.arange(seqlen, device=x.device),
                            same_layer_scores=(
                                self.last_index_score
                                if self._cascade_same_layer else None
                            ),
                        )
                    if self._cascade_hub_size > 0 and (
                        self._cascade_hub_stat == "idx_max"
                    ):
                        # start_pos == 0 on this branch (MHA first chunk).
                        self._cascade_hub_update_idx(seqlen, chunk_start=start_pos)
                    if self._cascade_same_layer:
                        # V3: attend the same-layer selection NOW (scores and
                        # attention from the same layer); it is also the set
                        # threaded to layer L+1 as the incumbent.
                        candidate_pool = self.last_cascade_topk
                if self._tap_lag_variant is not None:
                    self.last_tap_lag_topk = self._tap_lag_select(
                        scores,
                        candidate_pool[..., : self.indexer.index_topk],
                        torch.arange(seqlen, device=x.device),
                    )
                # Attention uses only the leading index_topk slice of the
                # (score-sorted) pool; without pool/attend decoupling the
                # slice is the whole set.
                attend_indices = candidate_pool[..., : self.indexer.index_topk]
                # Route -1 sentinels (non-finite score slots) to a dummy key
                # column that is sliced off, so invalid slots are masked out
                # instead of clamped into real attention on token 0.
                valid_attend = attend_indices >= 0
                attend_indices = torch.where(
                    valid_attend,
                    attend_indices.clamp(min=0, max=seqlen - 1),
                    torch.full_like(attend_indices, seqlen),
                )
                index_mask = torch.full(
                    (bsz, seqlen, seqlen + 1), float("-inf"), device=x.device
                ).scatter_(-1, attend_indices, 0)[..., :seqlen]
                scores += (index_mask + mask).unsqueeze(2)
            else:
                scores += mask.unsqueeze(0).unsqueeze(2)

            scores = scores.softmax(dim=-1)
            if self._collect_attn_mass:
                self.last_attn_mass = scores.sum(dim=2).sum(dim=1)
                nwq = self._n_window_queries
                if nwq is not None:
                    if nwq > 0:
                        self.last_attn_mass_window = (
                            scores[:, -nwq:, :, :].sum(dim=2).sum(dim=1)
                        )
                else:
                    w = self._attn_mass_window
                    if seqlen > w:
                        self.last_attn_mass_window = (
                            scores[:, -w:, :, :].sum(dim=2).sum(dim=1)
                        )
                    else:
                        self.last_attn_mass_window = self.last_attn_mass.clone()
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        elif (
            mask is not None
        ):  # chunked prefill — attend to full KV cache in compressed space
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = (
                self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            if self._capture_raw_qk:
                self.last_raw_qk_scores = scores[:, -1, :, :].detach().clone()
            if self._collect_tap_topk:
                self._capture_tap_topk(
                    scores,
                    start_pos + torch.arange(seqlen, device=x.device),
                )

            if self._collect_attn_mass:
                dense_scores = (scores + mask.unsqueeze(0).unsqueeze(2)).softmax(dim=-1)
                dmass = dense_scores.sum(dim=2).sum(dim=1)
                self.last_dense_attn_mass = self._accumulate_mass(
                    self.last_dense_attn_mass, dmass, end_pos,
                )
                nwq = self._n_window_queries
                if nwq is not None and nwq > 0:
                    dwmass = dense_scores[:, -nwq:, :, :].sum(dim=2).sum(dim=1)
                elif nwq is not None:
                    dwmass = None
                else:
                    w = self._attn_mass_window
                    dwmass = (
                        dense_scores[:, -w:, :, :].sum(dim=2).sum(dim=1)
                        if seqlen > w
                        else dmass
                    )
                self.last_dense_attn_mass_window = self._accumulate_mass(
                    self.last_dense_attn_mass_window, dwmass, end_pos,
                )
                del dense_scores

            # indexer
            if self._cascade_latent_refresh:
                # Cost-per-score: compressed-latent dot product replaces the
                # full indexer on this refresh layer (chunked prefill).
                topk_indices, self.last_index_score = (
                    self._latent_refresh_score(
                        bsz, end_pos,
                        torch.arange(start_pos, end_pos, device=x.device),
                    )
                )
            else:
                topk_indices, self.last_index_score = self.indexer(
                    x, qr, start_pos, freqs_cis, mask, kv_joint=self._kv_joint
                )
            topk_scores = self.last_index_score.gather(
                -1, topk_indices.clamp(min=0)
            )
            self.last_topk_indices = torch.where(
                torch.isfinite(topk_scores),
                topk_indices,
                torch.full_like(topk_indices, -1),
            )
            if (
                self._collect_tap_features
                and start_pos + seqlen == self._tap_feature_end
            ):
                self._capture_tap_features(scores)
            if (
                self._collect_variant_topk
                and start_pos + seqlen == self._tap_feature_end
            ):
                # S7a (K1) source components: q_nope here is ABSORBED
                # (q W_kv-projected), so its einsum against the kv-latent
                # cache IS the content term of the summed score — one extra
                # einsum on the final chunk only.
                variant_nope = torch.einsum(
                    "bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]
                ).mul_(self.softmax_scale)
                self._capture_variant_scores(scores, variant_nope)
                del variant_nope
            if not self._disable_indexer_mask:
                candidate_pool = (
                    self._lag_external_topk
                    if self._lag_external_topk is not None
                    else topk_indices
                )
                if self._cascade_mode:
                    if self._cascade_pool_topk and self._lag_external_topk is None:
                        # Cold-start producer: widen the incumbent to the
                        # indexer's own top-pool so the threaded pool is
                        # full width from the first layer.
                        candidate_pool = self._indexer_pool()
                    # Raw (pre-mask) scores: the tap re-ranks the keys this
                    # layer attends, plus the admission window, for layer L+1.
                    if self._cascade_same_layer and self.last_index_score is None:
                        raise RuntimeError(
                            "same-layer cascade scoring requires the "
                            "lightning indexer to have run this forward"
                        )
                    self.last_cascade_topk = self._cascade_select(
                        scores, candidate_pool,
                        start_pos + torch.arange(seqlen, device=x.device),
                        same_layer_scores=(
                            self.last_index_score
                            if self._cascade_same_layer else None
                        ),
                    )
                    if self._cascade_hub_size > 0 and (
                        self._cascade_hub_stat == "idx_max"
                    ):
                        self._cascade_hub_update_idx(end_pos, chunk_start=start_pos)
                    if self._cascade_same_layer:
                        # V3: attend the same-layer selection NOW (scores and
                        # attention from the same layer); it is also the set
                        # threaded to layer L+1 as the incumbent.
                        candidate_pool = self.last_cascade_topk
                if self._tap_lag_variant is not None:
                    self.last_tap_lag_topk = self._tap_lag_select(
                        scores,
                        candidate_pool[..., : self.indexer.index_topk],
                        start_pos + torch.arange(seqlen, device=x.device),
                    )
                # Attention uses only the leading index_topk slice of the
                # (score-sorted) pool; without pool/attend decoupling the
                # slice is the whole set.
                attend_indices = candidate_pool[..., : self.indexer.index_topk]
                valid_attend = attend_indices >= 0
                attend_indices = torch.where(
                    valid_attend,
                    attend_indices.clamp(min=0, max=end_pos - 1),
                    torch.full_like(attend_indices, end_pos),
                )
                index_mask = torch.full(
                    (bsz, seqlen, end_pos + 1), float("-inf"), device=x.device
                ).scatter_(-1, attend_indices, 0)[..., :end_pos]
                scores += (index_mask + mask).unsqueeze(2)
            else:
                scores += mask.unsqueeze(0).unsqueeze(2)

            scores = scores.softmax(dim=-1)
            if self._collect_attn_mass:
                mass = scores.sum(dim=2).sum(dim=1)
                self.last_attn_mass = self._accumulate_mass(
                    self.last_attn_mass, mass, end_pos,
                )
                nwq = self._n_window_queries
                if nwq is not None and nwq > 0:
                    wmass = scores[:, -nwq:, :, :].sum(dim=2).sum(dim=1)
                elif nwq is not None:
                    wmass = None
                else:
                    w = self._attn_mass_window
                    wmass = (
                        scores[:, -w:, :, :].sum(dim=2).sum(dim=1) if seqlen > w else mass
                    )
                self.last_attn_mass_window = self._accumulate_mass(
                    self.last_attn_mass_window, wmass, end_pos,
                )
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        else:  # MQA decode
            if self._use_fused_decode(bsz, end_pos):
                x = self._mqa_decode_fused(
                    x, qr, start_pos, freqs_cis, end_pos, q_nope, q_pe
                )
                return self.wo(x.flatten(2))
            if self._reuse_fused_active(bsz, end_pos):
                x = self._mqa_decode_reuse_fused(
                    x, qr, start_pos, freqs_cis, end_pos, q_nope, q_pe
                )
                return self.wo(x.flatten(2))
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = (
                self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            if self._capture_raw_qk:
                self.last_raw_qk_scores = scores[:, -1, :, :].detach().clone()
            if self._collect_tap_topk:
                self._capture_tap_topk(
                    scores, torch.tensor([end_pos - 1], device=x.device)
                )

            # indexer
            if self._cascade_skip_indexer_decode:
                # Serving cascade: selection is the tap over these attention
                # scores; the lightning indexer never runs on this layer.
                if self._lag_external_topk is None:
                    raise RuntimeError(
                        "skip_indexer_decode requires a cascade set from the "
                        "previous layer; layer 0 / refresh producers must not "
                        "set _cascade_skip_indexer_decode"
                    )
                if self._disable_indexer_mask:
                    raise RuntimeError(
                        "skip_indexer_decode is incompatible with "
                        "set_indexer_frozen(True)"
                    )
                topk_indices = None
                self.last_index_score = None
                self.last_topk_indices = None
            elif self._cascade_latent_refresh:
                # Cost-per-score: compressed-latent dot product replaces the
                # full indexer on this refresh layer (MQA decode).
                topk_indices, self.last_index_score = (
                    self._latent_refresh_score(
                        bsz, end_pos,
                        torch.tensor([end_pos - 1], device=x.device),
                    )
                )
                topk_scores = self.last_index_score.gather(-1, topk_indices)
                self.last_topk_indices = torch.where(
                    torch.isfinite(topk_scores),
                    topk_indices,
                    torch.full_like(topk_indices, -1),
                )
            else:
                if self._refresh_stripe_period > 0:
                    topk_indices, self.last_index_score = (
                        self._indexer_striped_decode(
                            x, qr, start_pos, freqs_cis, end_pos
                        )
                    )
                else:
                    topk_indices, self.last_index_score = self.indexer(
                        x, qr, start_pos, freqs_cis, mask,
                        kv_joint=self._kv_joint,
                    )
                topk_scores = self.last_index_score.gather(-1, topk_indices)
                self.last_topk_indices = torch.where(
                    torch.isfinite(topk_scores),
                    topk_indices,
                    torch.full_like(topk_indices, -1),
                )
            if not self._disable_indexer_mask:
                candidate_pool = (
                    self._lag_external_topk
                    if self._lag_external_topk is not None
                    else topk_indices
                )
                if self._cascade_mode:
                    if self._cascade_same_layer or (
                        self._cascade_hub_size > 0
                        and self._cascade_hub_stat == "idx_max"
                    ):
                        # Both new modes are registered prefill-only (the
                        # chunked-prefill PPL cell); decode semantics — which
                        # layer's indexer runs at decode, statistic freezing
                        # — are the SERVING design's job, not this harness's.
                        raise NotImplementedError(
                            "same_layer_indexer / hub_stat='idx_max' are "
                            "prefill-only cascade modes (decode reached)"
                        )
                    if self._cascade_pool_topk and self._lag_external_topk is None:
                        # Cold-start producer: widen the incumbent to the
                        # indexer's own top-pool so the threaded pool is
                        # full width from the first layer.
                        candidate_pool = self._indexer_pool()
                    if self._cascade_freeze_admit:
                        # ρ_admit freeze: skip admission; _cascade_produced
                        # threads the incumbent (candidate_pool) forward.
                        self.last_cascade_topk = None
                    else:
                        self.last_cascade_topk = self._cascade_select(
                            scores, candidate_pool,
                            torch.tensor([end_pos - 1], device=x.device),
                        )
                if self._tap_lag_variant is not None:
                    self.last_tap_lag_topk = self._tap_lag_select(
                        scores,
                        candidate_pool[..., : self.indexer.index_topk],
                        torch.tensor([end_pos - 1], device=x.device),
                    )
                # Attention uses only the leading index_topk slice of the
                # (score-sorted) pool; without pool/attend decoupling the
                # slice is the whole set.
                attend_indices = candidate_pool[..., : self.indexer.index_topk]
                valid_attend = attend_indices >= 0
                attend_indices = torch.where(
                    valid_attend,
                    attend_indices.clamp(min=0, max=end_pos - 1),
                    torch.full_like(attend_indices, end_pos),
                )
                index_mask = torch.full(
                    (bsz, 1, end_pos + 1), float("-inf"), device=x.device
                ).scatter_(-1, attend_indices, 0)[..., :end_pos]
                scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim, reduce_output=reduce_output)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if args.gate_bias
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x.float(), self.weight.float())
        scores = scores.softmax(dim=-1) if self.score_func == "softmax" else scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        if args.n_routed_experts % world_size != 0:
            raise ValueError(
                f"n_routed_experts={args.n_routed_experts} must be "
                f"divisible by world_size={world_size}."
            )
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = MLP(
            args.dim, args.n_shared_experts * args.moe_inter_dim, reduce_output=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        return x, residual


def _cat_index_chunks(
    chunks_by_layer: dict[int, list[torch.Tensor]], device: torch.device
) -> dict[int, torch.Tensor] | None:
    """Concatenate per-chunk ``[B, s, k]`` index tensors along the query dim.

    Early chunks may hold fewer candidates than ``index_topk`` (when
    ``end_pos < index_topk``), so chunks are right-padded with the ``-1``
    sentinel to the widest ``k`` before concatenation.
    """
    if not chunks_by_layer:
        return None
    combined: dict[int, torch.Tensor] = {}
    for layer, chunks in chunks_by_layer.items():
        max_topk = max(c.size(2) for c in chunks)
        padded = []
        for c in chunks:
            if c.size(2) < max_topk:
                pad = torch.full(
                    (c.size(0), c.size(1), max_topk - c.size(2)),
                    fill_value=-1,
                    dtype=c.dtype,
                    device=c.device,
                )
                c = torch.cat([c, pad], dim=2)
            padded.append(c)
        combined[layer] = torch.cat(padded, dim=1).to(device)
    return combined


@dataclass
class ForwardResult:
    """Result of an instrumented Transformer forward pass."""

    logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor] | None = None
    index_scores: dict[int, torch.Tensor] | None = None
    topk_indices: dict[int, torch.Tensor] | None = None
    tap_topk: dict[int, torch.Tensor] | None = None
    mla_latents: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    hub_scores: dict[int, torch.Tensor] | None = None
    h2o_scores: torch.Tensor | None = None
    snapkv_scores: torch.Tensor | None = None
    dense_h2o_scores: torch.Tensor | None = None
    dense_snapkv_scores: torch.Tensor | None = None


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        # One-layer lag schedule: layer L attends using layer (L-1)'s top-k.
        # Layer 0 uses its own indexer (eager cold start).  Toggle with
        # set_lag_mode(); applies to both chunked prefill and decode.
        self._lag_mode: bool = False
        # One-hop tap-lag schedule (axis V stage 2); see set_tap_lag_mode().
        self._tap_lag_mode: bool = False
        # Tap-cascade schedule; see set_cascade_mode().  Mutually exclusive
        # with _lag_mode and _tap_lag_mode (all three thread a produced set
        # into the next layer's _lag_external_topk slot).
        self._cascade_mode: bool = False
        self._cascade_refresh_every: int = 0
        self._cascade_refresh_layers: set[int] | None = None
        self._cascade_adaptive_refresh_threshold: float = 0.0
        self._cascade_adaptive_refresh_set: set[int] = set()
        # Staggered refresh: divide refresh layers into S groups; only
        # one group is active per decode step (rotating), amortizing
        # refresh cost across steps while maintaining full coverage.
        self._cascade_stagger_groups: int = 1
        self._cascade_stagger_step: int = 0
        self._cascade_stagger_group_map: dict[int, int] = {}
        # Refresh-prior blend (S7e): > 0 makes the forward loops thread the
        # block producer's indexer score row to its consumers.
        self._cascade_prior_lambda: float = 0.0
        # Entries-head threading: when True the forward loops carry each
        # layer's block summaries to the next layer's _cascade_prev_block_tap
        # (per-chunk transient, prior_row pattern).
        self._cascade_block_head_active: bool = False
        # SOURCE-track indexer redesign; see set_indexer_source().  Mutually
        # exclusive with the lag/tap-lag/cascade schedules (it replaces the
        # indexer's own scan under the unchanged 'same' schedule).
        self._indexer_source: str | None = None

    def reset_kv_cache(self, bsz: int, seqlen: int) -> None:
        """Reallocate all per-layer KV caches for a new (bsz, seqlen).

        Call this before each eval run to avoid the ~12 GB pre-allocation at
        max_seq_len=128K. Init allocates caches with seqlen=1 to save memory;
        this call materialises the full-size buffers right before they are used.
        """
        for layer in self.layers:
            if hasattr(layer.attn, "reset_kv_cache"):
                layer.attn.reset_kv_cache(bsz, seqlen)

    def _ensure_kv_cache_capacity(self, bsz: int, end_pos: int, start_pos: int) -> None:
        """Guard the public forwards against undersized lazy caches.

        The model boots with seqlen=1 caches (see ``reset_kv_cache``). A caller
        that forgets ``reset_kv_cache`` or underestimates the decode length would
        otherwise fault deep inside an attention cache write. Auto-size when it
        is safe to start fresh (``start_pos == 0``); otherwise raise a clear
        error before any partial writes happen.
        """
        attn = next(
            (
                layer.attn
                for layer in self.layers
                if getattr(getattr(layer, "attn", None), "kv_cache", None) is not None
            ),
            None,
        )
        if attn is None:
            return
        cache_bsz, cache_len = attn.kv_cache.size(0), attn.kv_cache.size(1)
        if bsz <= cache_bsz and end_pos <= cache_len:
            return
        if start_pos == 0:
            self.reset_kv_cache(bsz, end_pos)
            return
        raise ValueError(
            f"KV cache too small for this continuation (need bsz={bsz}, "
            f"seqlen>={end_pos}; have bsz={cache_bsz}, seqlen={cache_len}); call "
            "reset_kv_cache(bsz, seqlen) before prefill/decode."
        )

    def set_lag_mode(self, enabled: bool) -> None:
        """Enable or disable the one-layer lag schedule.

        When enabled, each layer's sparse attention (MHA prefill, chunked
        prefill, and single-token MQA decode) uses the previous layer's
        indexer top-k set.  Layer 0 always uses its own indexer (eager cold
        start), keeping the ``PPL(lag) - PPL(same)`` delta an apples-to-apples
        isolation of the L>=1 lag effect.
        """
        if enabled and self._indexer_source is not None:
            raise ValueError(
                "lag mode is incompatible with an active indexer source; call "
                "set_indexer_source(None) first"
            )
        if enabled and self._cascade_mode:
            raise ValueError(
                "lag and cascade modes are mutually exclusive; call "
                "set_cascade_mode(False) first"
            )
        if enabled and self._tap_lag_mode:
            raise ValueError(
                "lag and tap-lag modes are mutually exclusive; call "
                "set_tap_lag_mode(None) first"
            )
        self._lag_mode = enabled
        self._cascade_refresh_layers: set[int] | None = None
        if not enabled:
            for layer in self.layers:
                layer.attn._lag_external_topk = None

    def set_tap_lag_mode(
        self,
        variant: str | None,
        head_weights: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> None:
        """Enable the one-hop tap-lag schedule (axis V stage 2), or disable.

        The lag structure with the handoff payload swapped: every layer
        still runs its true lightning indexer, but layer ``L+1`` attends
        the top-k of layer ``L``'s *scoring-variant tap* over ALL causal
        keys (dense — no admission machinery) instead of layer ``L``'s
        indexer top-k.  Layer 0 attends its own indexer set.  Selection
        errors cannot compound: each handoff is one hop from exact
        indexer-driven scores, so ``PPL(tap_lag) - PPL(same)`` isolates
        the scoring function's quality the way ``PPL(lag) - PPL(same)``
        isolates the lag premise.

        Args:
            variant: A name from ``tap.VARIANT_NAMES``, or ``None`` to
                disable the mode.
            head_weights: Optional ``[H]`` global head weights (``None`` =
                uniform, the screen-validated arm for v4/v5/v6).
        """
        enabled = variant is not None
        if enabled and variant not in VARIANT_NAMES:
            raise ValueError(
                f"unknown scoring variant {variant!r}; expected one of "
                f"{VARIANT_NAMES}"
            )
        if variant == "v4b_temp" and tau == 1.0:
            raise ValueError(
                "v4b_temp at tau=1 is exactly v4_probs; pass the tau that "
                "won its screen cell (or use v4_probs)"
            )
        if tau <= 0 or not math.isfinite(tau):
            raise ValueError(f"tau must be finite and > 0, got {tau}")
        if enabled and tau != 1.0 and variant not in ("v4b_temp", "v3_lse"):
            # A tau on a variant that ignores it would be recorded as a
            # distinct schedule while executing identically (CodeRabbit
            # PR-29).
            raise ValueError(
                f"tau is a v4b_temp/v3_lse lever; {variant} ignores it — "
                "drop the tau"
            )
        if enabled and self._indexer_source is not None:
            raise ValueError(
                "tap-lag mode is incompatible with an active indexer source; call "
                "set_indexer_source(None) first"
            )
        if enabled and self._lag_mode:
            raise ValueError(
                "tap-lag and lag modes are mutually exclusive; call "
                "set_lag_mode(False) first"
            )
        if enabled and self._cascade_mode:
            raise ValueError(
                "tap-lag and cascade modes are mutually exclusive; call "
                "set_cascade_mode(False) first"
            )
        self._tap_lag_mode = enabled
        for layer in self.layers:
            layer.attn._tap_lag_variant = variant
            layer.attn._tap_lag_weights = head_weights if enabled else None
            layer.attn._tap_lag_tau = tau if enabled else 1.0
            layer.attn._lag_external_topk = None
            layer.attn.last_tap_lag_topk = None

    def set_cascade_mode(
        self,
        enabled: bool,
        window: int = 128,
        n_sink: int = 1,
        head_weights: torch.Tensor | dict[int, torch.Tensor] | None = None,
        refresh_every: int = 0,
        refresh_layers: set[int] | None = None,
        reuse: bool = False,
        skip_indexer_decode: bool = False,
        pool_topk: int | None = None,
        admission_stripe: int = 0,
        select_hysteresis: float = 0.0,
        hub_size: int = 0,
        hub_stat: str = "max",
        hub_decay: float = 0.9,
        scoring_variant: str | None = None,
        refresh_prior_lambda: float = 0.0,
        refresh_prior_decay: float = 0.5,
        same_layer_indexer: bool = False,
        hub_freeze_pos: int | None = None,
        neighbor_delta: int = 0,
        neighbor_budget: int = 0,
        block_size: int = 0,
        block_admit: int = 0,
        block_stat: str = "mean",
        block_head: dict[int, dict[str, torch.Tensor]] | None = None,
        prefill_select_only: bool = False,
        admit_every: int = 1,
        latent_refresh: bool = False,
        adaptive_refresh_threshold: float = 0.0,
        stagger_groups: int = 1,
        refresh_stripe_period: int = 0,
        refresh_stripe_fresh_tail: int | None = None,
    ) -> None:
        """Enable or disable the tap-cascade schedule.

        When enabled, each layer's sparse attention uses the *cascade* set the
        previous layer produced: the tap re-rank (``sum_h w_h ReLU(S_h)`` of
        the previous layer's own pre-softmax attention scores) over the keys
        it attended plus an admission window of the ``window`` most recent
        positions and the first ``n_sink`` positions.  Layer 0 uses its own
        indexer (eager cold start), mirroring ``set_lag_mode``.  The lightning
        indexer still runs every layer so fidelity metrics can compare
        against it.

        Args:
            enabled: Turn the cascade schedule on or off.
            window: Recency-window size of the admission set.
            n_sink: Leading sink positions always admitted.
            head_weights: Optional per-attention-head tap weights: a single
                ``[H]`` tensor shared by all layers, a ``{layer_idx: [H]}``
                dict for per-layer (distilled) weights, or ``None`` = uniform.
            refresh_every: When > 0, every ``refresh_every``-th layer hands
                the lightning indexer's own top-k forward instead of the
                cascade set (periodic exact refresh); 0 = pure cascade.
            refresh_layers: Explicit set of *producer* layer indices that
                hand the indexer's own top-k forward (layer ``L`` in the set
                refreshes the set consumed by layer ``L+1``).  Overrides
                ``refresh_every``; use to concentrate refreshes where tap
                fidelity is weakest (early layers) instead of uniformly.
            reuse: IndexCache-style baseline — non-refresh layers propagate
                the incoming index set VERBATIM (no tap re-rank, no
                admission); refresh layers hand the indexer's set forward as
                usual.  Isolates the value of per-layer re-ranking at a
                matched indexer-retention ratio.
            skip_indexer_decode: Serving mode — during single-token MQA
                decode, layers that consume a threaded set (every layer
                except layer 0) and are not refresh producers skip the
                lightning indexer entirely instead of running it for
                metrics.  This realizes the cascade's decode-time saving:
                selection becomes a second consumer of attention's own
                QK^T scores rather than a separate scoring pass.  Prefill
                is unaffected (the indexer still runs everywhere there).
            pool_topk: Pool/attend decoupling — when set (> index_topk),
                each layer threads a score-sorted candidate pool of this
                size to the next layer while attention itself uses only the
                pool's leading ``index_topk`` entries.  A key is then
                evicted from the cascade only when it falls below rank
                ``pool_topk`` of the tap, not rank ``index_topk``, directly
                raising the coverage ceiling that binds pure-cascade
                quality.  Layers that run the true indexer (layer 0,
                refresh producers) seed the pool with the indexer's own
                top-``pool_topk``.  ``None`` disables decoupling.
            admission_stripe: When > 0, each query also scores this many
                phase-rotating equally spaced candidates over the region
                older than the recency window (``tap.stripe_candidates``) —
                a re-admission chance for evicted distant keys, at the same
                candidate budget as widening the window by the same count.
                Eager-only (the fused kernel scores window tiles, not
                stripes).
            select_hysteresis: When > 0, incumbent candidates get this
                additive tap-logit bias during selection only (threaded
                scores stay unbiased): a challenger must beat an incumbent
                by this margin to evict it, damping near-tie selection
                churn.  Eager-only for now: the fused kernel's tap planes
                are unweighted per-head ReLU sums of raw QK — an
                incommensurate scale for an absolute bias tuned on the
                eager tap (head mean of softmax_scale-multiplied scores) —
                so ``set_cascade_fused_decode`` rejects a nonzero value
                (see docs/pr29_review_findings.md).
            hub_size: When > 0, each layer tracks the running max tap score
                per key over every key it has ever scored and always admits
                the top-``hub_size`` such keys as candidates — persistent
                re-admission for keys that once scored high (targets
                distant-evidence eviction).  Eager-only.
            refresh_prior_lambda: When > 0, consumer taps inside each refresh
                block add a decaying prior from the block producer's
                lightning-indexer scores (axis-7 source S7e):
                ``tap' = tap + lam * decay^(d-1) * std_row(tap) *
                z_row(prior)`` at the candidate columns, where ``d`` is the
                layer distance from the producer.  Layer 0's cold-start
                indexer row seeds the FIRST block (S7e' amendment): layer 0
                always runs its indexer and hands its set forward, so it is
                that block's producer.  Requires a refresh schedule (no
                later block has a producer otherwise) and per-layer
                re-ranking (incompatible with ``reuse``).  Eager-only.
            refresh_prior_decay: Per-layer decay of the prior, in (0, 1].
            hub_freeze_pos: When set (absolute token position), the hub
                membership statistic stops FOLDING for any chunk starting
                at or beyond this position; every later selection consumes
                the frozen boundary state (the serving frozen
                selection-history set — docs/prereg_freeze_ablation.md).
                Reads are unaffected.  Requires ``hub_size > 0`` and a
                monotone max statistic (``"max"`` / ``"idx_max"``): the
                mass aggregators' read-time decay semantics under a freeze
                are unregistered.  ``None`` = mutable membership.
            neighbor_delta: Neighbor-expansion reach (coverage lever g):
                admission candidates at ``incumbent +/- 1..neighbor_delta``,
                phase-subsampled to ``neighbor_budget`` slots per query
                (``tap.neighbor_candidates``).  Content-adaptive — the
                anchors are wherever selection currently looks — stateless,
                and eager-only.  Requires ``neighbor_budget > 0``.
            neighbor_budget: Candidate slots per query for the neighbor
                expansion; 0 disables (then ``neighbor_delta`` must be 0
                too).
            block_size: Block coarse-to-fine admission (coverage lever e):
                every layer reduces the whole prefix to
                ``ceil(T / block_size)`` per-block summary scores, tap-
                reduces them, and admits all tokens of the top
                ``block_admit`` blocks for exact re-rank scoring — the only
                admission lever that sees the ENTIRE prefix content-
                adaptively every layer, at O(T / block_size) coarse cost.
                Stateless, eager-only, classic-tap only (rejected with
                ``scoring_variant``).  Requires ``block_admit > 0``.
            block_admit: Blocks admitted per query (token budget =
                ``block_admit * block_size``); 0 disables (then
                ``block_size`` must be 0 too).
            block_stat: Block summary reduction: ``"mean"`` (default; the
                exact mean-latent summary-key score by bilinearity — the
                serving-realizable summary) or ``"amax"`` (per-head max —
                score-space upper envelope, diagnostic only).
            block_head: Entries-head payloads keyed by PRODUCER layer index
                (docs/prereg_entries_head.md): ``{layer: {mu, sd, w1, b1,
                w2, b2}}`` fitted by ``bench/delta_probe_analysis.py
                --export-head``.  A layer with a head ranks admission blocks
                by the learned MLP over {this summary, recency, incumbent-
                block distance, previous layer's summary} restricted to
                non-incumbent causal blocks; layers without a head (or
                without a threaded previous summary — e.g. the first
                cascade consumer) keep the raw summary ranking.  Requires
                the block lever (``block_size``/``block_admit`` > 0).
            adaptive_refresh_threshold: When > 0, enable content-adaptive
                refresh: after each decode step, measure the jaccard churn
                (``1 - |A ∩ B| / |A ∪ B|``) between this step's and the
                previous step's produced selection at each non-refresh layer.
                When churn exceeds the threshold, promote that layer to a
                dynamic refresh at the NEXT decode step (its indexer runs,
                tap production is suppressed, and the fresh set replaces
                the stale cascade).  Model-agnostic: noisy models trigger
                more refreshes automatically.  Composes with
                ``refresh_every`` / ``refresh_layers`` (which set the
                minimum refresh cadence) and the fused decode path (flags
                are toggled per-step).  0 = disabled (static schedule only).
            refresh_stripe_period: Temporal refresh striping (track SCHED,
                mechanism f).  When > 0, layers that run the lightning
                indexer during cascade decode (layer 0 and refresh
                producers) score only a rotating ``1/period`` contiguous
                slice of the prefix per decode step — O(L/period) instead
                of O(L) — merging into a running per-layer score cache; the
                threaded top-k reads the merged, partially stale row.  A
                fresh tail of the newest positions is re-scored every step,
                and the first striped step (or any position discontinuity,
                e.g. the chunked-prefill boundary) re-seeds the cache with
                one full scan.  Max staleness of any cached score:
                ``period - 1`` steps (the QUERY is stale; keys are
                immutable).  Eager-only; 0 = off.  Incompatible with
                ``reuse`` (it would silently change the IndexCache-style
                baseline's refresh sets).
            refresh_stripe_fresh_tail: Newest positions re-scored every
                striped step (>= 1).  ``None`` = ``refresh_stripe_period``,
                which guarantees a freshly generated key is scored at every
                step until its slice's first rotation reaches it.
        """
        if not 0.0 <= adaptive_refresh_threshold < 1.0:
            raise ValueError(
                f"adaptive_refresh_threshold must be in [0, 1), got "
                f"{adaptive_refresh_threshold}"
            )
        if refresh_prior_lambda < 0:
            raise ValueError(
                f"refresh_prior_lambda must be >= 0, got {refresh_prior_lambda}"
            )
        if not 0.0 < refresh_prior_decay <= 1.0:
            raise ValueError(
                f"refresh_prior_decay must be in (0, 1], got {refresh_prior_decay}"
            )
        if refresh_prior_lambda > 0:
            if reuse:
                raise ValueError(
                    "refresh_prior_lambda requires per-layer re-ranking; there "
                    "is no tap to blend the prior into under verbatim reuse"
                )
            if not (refresh_every > 0 or refresh_layers):
                raise ValueError(
                    "refresh_prior_lambda > 0 requires a refresh schedule "
                    "(refresh_every > 0 or a non-empty refresh_layers); the "
                    "prior is the block PRODUCER's indexer scores and a pure "
                    "cascade has no producers"
                )
        if admission_stripe < 0 or hub_size < 0:
            raise ValueError(
                f"admission_stripe and hub_size must be >= 0, got "
                f"{admission_stripe}, {hub_size}"
            )
        if refresh_stripe_period < 0:
            raise ValueError(
                f"refresh_stripe_period must be >= 0, got {refresh_stripe_period}"
            )
        if refresh_stripe_fresh_tail is not None:
            if refresh_stripe_period == 0:
                raise ValueError(
                    "refresh_stripe_fresh_tail is a temporal-striping lever; "
                    "it requires refresh_stripe_period > 0"
                )
            if refresh_stripe_fresh_tail < 1:
                raise ValueError(
                    f"refresh_stripe_fresh_tail must be >= 1, got "
                    f"{refresh_stripe_fresh_tail}"
                )
        if refresh_stripe_period > 0 and reuse:
            raise ValueError(
                "refresh_stripe_period would change the refresh sets of the "
                "verbatim-reuse (IndexCache-style) baseline; temporal "
                "striping is a cascade lever only"
            )
        if neighbor_delta < 0 or neighbor_budget < 0:
            raise ValueError(
                f"neighbor_delta and neighbor_budget must be >= 0, got "
                f"{neighbor_delta}, {neighbor_budget}"
            )
        if (neighbor_delta > 0) != (neighbor_budget > 0):
            raise ValueError(
                "neighbor expansion needs both knobs: neighbor_delta (reach) "
                f"and neighbor_budget (slots) — got delta={neighbor_delta}, "
                f"budget={neighbor_budget}; set both > 0 or both 0"
            )
        if block_size < 0 or block_admit < 0:
            raise ValueError(
                f"block_size and block_admit must be >= 0, got "
                f"{block_size}, {block_admit}"
            )
        if (block_size > 0) != (block_admit > 0):
            raise ValueError(
                "block coarse-to-fine needs both knobs: block_size (keys per "
                f"block) and block_admit (blocks admitted) — got size="
                f"{block_size}, admit={block_admit}; set both > 0 or both 0"
            )
        if block_stat not in ("mean", "amax"):
            raise ValueError(
                f"block_stat must be 'mean' or 'amax', got {block_stat!r}"
            )
        if block_stat != "mean" and block_size == 0:
            raise ValueError(
                "block_stat is a block coarse-to-fine lever; it requires "
                "block_size > 0"
            )
        if scoring_variant is not None and block_size > 0:
            raise ValueError(
                "block coarse-to-fine scores its summaries with the classic "
                "weighted-ReLU tap (_tap_reduce); mixing it with a "
                "scoring_variant would rank blocks and candidates in "
                "incommensurate score spaces — drop one of the two"
            )
        if block_head is not None:
            if block_size == 0:
                raise ValueError(
                    "block_head rides the block lever; it requires "
                    "block_size/block_admit > 0"
                )
            required = ("mu", "sd", "w1", "b1", "w2", "b2")
            for layer_idx, payload in block_head.items():
                missing = [k for k in required if k not in payload]
                if missing:
                    raise ValueError(
                        f"block_head[{layer_idx}] is missing keys {missing}; "
                        "expected a bench/delta_probe_analysis.py "
                        "--export-head payload"
                    )
        if hub_stat not in ("max", "mass", "mass_decay", "idx_max"):
            raise ValueError(
                f"hub_stat must be one of ('max', 'mass', 'mass_decay', "
                f"'idx_max'), got {hub_stat!r}"
            )
        if same_layer_indexer:
            # Registered scope (prereg_history_set_scoring): the same-layer
            # V3-semantics arm composes only with the plain cascade + hub.
            incompatible = {
                "reuse": reuse,
                "pool_topk": pool_topk is not None,
                "admission_stripe": admission_stripe > 0,
                "scoring_variant": scoring_variant is not None,
                "refresh_prior_lambda": refresh_prior_lambda > 0,
            }
            bad = [k for k, v in incompatible.items() if v]
            if bad:
                raise ValueError(
                    f"same_layer_indexer is incompatible with {bad} "
                    "(registered prefill-only scope)"
                )
        if hub_stat != "max" and hub_size == 0:
            raise ValueError(
                "hub_stat is a hub-set lever (K3, S7b aggregator); it "
                "requires hub_size > 0"
            )
        if hub_freeze_pos is not None:
            if hub_size == 0:
                raise ValueError(
                    "hub_freeze_pos freezes the hub membership statistic; "
                    "it requires hub_size > 0"
                )
            if hub_freeze_pos <= 0:
                raise ValueError(
                    f"hub_freeze_pos must be a positive absolute position "
                    f"(a freeze at <= 0 leaves the hub permanently empty), "
                    f"got {hub_freeze_pos}"
                )
            if hub_stat not in ("max", "idx_max"):
                raise ValueError(
                    "hub_freeze_pos is defined for the monotone max "
                    "statistics ('max', 'idx_max'); the mass aggregators' "
                    f"read-time decay under a freeze is unregistered, got "
                    f"hub_stat={hub_stat!r}"
                )
        if not 0.0 < hub_decay <= 1.0:
            raise ValueError(
                f"hub_decay must be in (0, 1], got {hub_decay}"
            )
        if select_hysteresis < 0.0:
            raise ValueError(
                f"select_hysteresis must be >= 0, got {select_hysteresis}"
            )
        if reuse and (
            admission_stripe or hub_size or select_hysteresis
            or neighbor_budget or block_size
        ):
            raise ValueError(
                "admission_stripe/select_hysteresis/hub_size/neighbor_*/"
                "block_* require per-layer re-ranking; they are meaningless "
                "under verbatim reuse"
            )
        supported_variants = (
            "v0_relu_sum", "v1_signed_sum", "v4_probs",
            "v5_probs_vnorm", "v6_zscore",
        )
        if scoring_variant is not None:
            if scoring_variant not in supported_variants:
                raise ValueError(
                    f"cascade scoring_variant {scoring_variant!r} not "
                    f"supported; choose from {supported_variants}"
                )
            if reuse:
                raise ValueError(
                    "scoring_variant requires per-layer re-ranking; it is "
                    "meaningless under verbatim reuse"
                )
            if head_weights is not None:
                raise ValueError(
                    "scoring_variant runs uniform head weights (the "
                    "screen-validated arms); drop head_weights"
                )
        if enabled and self._indexer_source is not None:
            raise ValueError(
                "cascade mode is incompatible with an active indexer source; call "
                "set_indexer_source(None) first"
            )
        if enabled and self._lag_mode:
            raise ValueError(
                "cascade and lag modes are mutually exclusive; call "
                "set_lag_mode(False) first"
            )
        if enabled and self._tap_lag_mode:
            raise ValueError(
                "cascade and tap-lag modes are mutually exclusive; call "
                "set_tap_lag_mode(None) first"
            )
        if refresh_layers is not None and refresh_every:
            raise ValueError("pass refresh_every or refresh_layers, not both")
        if pool_topk is not None:
            index_topk = self.layers[0].attn.indexer.index_topk
            if pool_topk < index_topk:
                raise ValueError(
                    f"pool_topk={pool_topk} must be >= index_topk={index_topk}"
                )
            if reuse:
                raise ValueError(
                    "pool/attend decoupling requires per-layer re-ranking; "
                    "it is meaningless under verbatim reuse (the attended "
                    "slice of a verbatim pool never changes)"
                )
        self._cascade_mode = enabled
        self._cascade_refresh_every = refresh_every
        self._cascade_refresh_layers = refresh_layers
        self._cascade_reuse = reuse
        self._cascade_prefill_select_only = bool(enabled and prefill_select_only)
        if admit_every < 1:
            raise ValueError(f"admit_every must be >= 1, got {admit_every}")
        # ρ_admit: LAYER cadence for admission scoring, decoupled from the
        # refresh (rescan) cadence. admit_every=1 admits on every select layer
        # (realization b, the default). admit_every=k freezes the threaded set
        # on non-admit, non-refresh, non-zero layers (like prefill_select_only),
        # so only ~1/k of the select layers pay the _cascade_select admission
        # cost — the offline analog of the serving HH_TAP_ADMIT_EVERY knob.
        self._cascade_admit_every = int(admit_every) if enabled else 1
        self._cascade_adaptive_refresh_threshold = (
            adaptive_refresh_threshold if enabled else 0.0
        )
        self._cascade_adaptive_refresh_set: set[int] = set()
        if stagger_groups < 1:
            raise ValueError(
                f"stagger_groups must be >= 1, got {stagger_groups}"
            )
        self._cascade_stagger_groups = int(stagger_groups) if enabled else 1
        self._cascade_stagger_step = 0
        self._cascade_stagger_group_map = {}
        if enabled and stagger_groups > 1:
            refresh_list = sorted(
                refresh_layers
                if refresh_layers is not None
                else [
                    i
                    for i in range(len(self.layers))
                    if refresh_every > 0 and (i + 1) % refresh_every == 0
                ]
            )
            for rank, layer_idx in enumerate(refresh_list):
                self._cascade_stagger_group_map[layer_idx] = rank % stagger_groups
        self._cascade_prior_lambda = refresh_prior_lambda if enabled else 0.0
        self._cascade_block_head_active = (
            enabled and block_head is not None and block_size > 0
        )
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            # Verbatim-reuse layers never compute the tap.
            layer.attn._cascade_mode = enabled and not reuse
            layer.attn._cascade_window = window
            layer.attn._cascade_n_sink = n_sink
            layer.attn._cascade_pool_topk = (
                pool_topk if enabled and pool_topk is not None else 0
            )
            layer.attn._cascade_stripe = admission_stripe if enabled else 0
            layer.attn._cascade_hysteresis = (
                select_hysteresis if enabled else 0.0
            )
            layer.attn._cascade_neighbor_delta = (
                neighbor_delta if enabled else 0
            )
            layer.attn._cascade_neighbor_budget = (
                neighbor_budget if enabled else 0
            )
            layer.attn._cascade_block_size = block_size if enabled else 0
            layer.attn._cascade_block_admit = block_admit if enabled else 0
            layer.attn._cascade_block_stat = block_stat if enabled else "mean"
            layer.attn._cascade_block_head = (
                block_head.get(i) if enabled and block_head else None
            )
            # Stash summaries on every block-lever layer whenever ANY head
            # is configured: layer L's summary is layer L+1's prev feature.
            layer.attn._cascade_block_stash = (
                enabled and block_head is not None and block_size > 0
            )
            # Threaded transients must never survive a schedule change.
            layer.attn._cascade_prev_block_tap = None
            layer.attn.last_block_tap = None
            layer.attn._cascade_hub_size = hub_size if enabled else 0
            layer.attn._cascade_hub_stat = hub_stat if enabled else "max"
            layer.attn._cascade_hub_decay = hub_decay
            layer.attn._cascade_hub_freeze_pos = (
                hub_freeze_pos if enabled else None
            )
            layer.attn._cascade_same_layer = (
                enabled and same_layer_indexer and not reuse
            )
            layer.attn._cascade_same_layer_selects = 0
            # Cascade-hub state is per-sequence; any schedule change
            # invalidates it (distinct from hub_marginalization's
            # analysis-time hub_scores).
            layer.attn._cascade_hub_scores = None
            layer.attn._cascade_scoring_variant = (
                scoring_variant if enabled else None
            )
            layer.attn._cascade_prior_lambda = (
                refresh_prior_lambda if enabled else 0.0
            )
            layer.attn._cascade_prior_decay = (
                refresh_prior_decay if enabled else 1.0
            )
            # Prior rows are per-forward transients; a schedule change must
            # never leave a stale row behind.
            layer.attn._cascade_prior_row = None
            layer.attn._cascade_prior_dist = 0
            layer.attn._cascade_head_weights = (
                head_weights.get(i) if isinstance(head_weights, dict)
                else head_weights
            )
            refresh = self._cascade_is_static_refresh(i)
            # b' select-only prefill scope: non-refresh skip layers do NOT run
            # own-select attention (they reuse-copy the incumbent, see
            # _cascade_produced) and do NOT fold the hub (hub_size=0 gates both
            # _cascade_hub_update_idx fold sites AND admission). Active layers
            # (refresh producers + layer 0) keep own-select + hub unchanged.
            if enabled and prefill_select_only and not (refresh or i == 0):
                layer.attn._cascade_same_layer = False
                layer.attn._cascade_hub_size = 0
            # The indexer must still run on layer 0 (cold-start consumer of
            # its own set) and on refresh producers whose set the NEXT layer
            # consumes; the final layer produces for nobody.
            needs_indexer = i == 0 or (refresh and i < n_layers - 1)
            layer.attn._cascade_skip_indexer_decode = (
                enabled and skip_indexer_decode and not needs_indexer
            )
            # ρ_admit (admit_every) LAYER cadence: freeze the threaded incumbent
            # forward on non-admit, non-refresh, non-zero layers (their
            # _cascade_select is skipped in MLA.forward and _cascade_produced
            # threads the incumbent unchanged). Layer 0 (cold start) and refresh
            # producers always admit; admit_every=1 admits on every select layer.
            layer.attn._cascade_freeze_admit = (
                enabled
                and admit_every > 1
                and i != 0
                and not refresh
                and (i % admit_every) != 0
            )
            # Latent-refresh: on refresh-producer layers, replace the full
            # indexer with a compressed-latent dot product.  Layer 0 keeps
            # the real indexer (cold start — no threaded set to fall back on).
            layer.attn._cascade_latent_refresh = (
                enabled and latent_refresh and refresh and i != 0
            )
            layer.attn._cascade_adaptive_threshold = (
                adaptive_refresh_threshold if enabled else 0.0
            )
            layer.attn._cascade_prev_selection = None
            layer.attn._cascade_adaptive_pending = False
            # Temporal striping applies exactly where the O(L) refresh cost
            # lives: layers whose indexer must run during cascade decode.
            # Consumer layers that run the indexer only for observation
            # (skip_indexer_decode=False harness modes) keep full scans so
            # their metrics stay exact.
            stripe_on = enabled and refresh_stripe_period > 0 and needs_indexer
            layer.attn._refresh_stripe_period = (
                refresh_stripe_period if stripe_on else 0
            )
            layer.attn._refresh_stripe_fresh_tail = (
                (refresh_stripe_fresh_tail
                 if refresh_stripe_fresh_tail is not None
                 else refresh_stripe_period)
                if stripe_on else 0
            )
            # Stripe caches are schedule-derived per-sequence state; any
            # schedule change invalidates them.
            layer.attn._refresh_stripe_cache = None
            layer.attn._refresh_stripe_step = 0
            layer.attn._refresh_stripe_expected_end = -1
            layer.attn.last_refresh_stripe_spans = None
            layer.attn.last_refresh_stripe_selfcheck = None
            if not enabled:
                layer.attn._lag_external_topk = None
                layer.attn.last_cascade_topk = None
        # A fused-decode configuration is derived from one specific cascade
        # schedule (refresh roles, head weights); any schedule change drops
        # the kernel path until set_cascade_fused_decode is called again.
        self.set_cascade_fused_decode(None)
        # Same invalidation for the reuse_fused composition path.
        self.set_reuse_fused_decode(None)

    def _cascade_is_refresh(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` is a refresh producer of the current schedule.

        The single source of truth for the refresh-role arithmetic:
        ``set_cascade_mode``, ``set_cascade_fused_decode``,
        ``_cascade_produced`` and the refresh-prior threading all consult it
        so the roles can never drift apart.

        Also returns True for layers in the adaptive refresh set — those
        flagged for dynamic refresh this decode step due to selection churn
        exceeding ``adaptive_refresh_threshold`` at the previous step.
        """
        if layer_idx in self._cascade_adaptive_refresh_set:
            return True
        if self._cascade_refresh_layers is not None:
            is_static = layer_idx in self._cascade_refresh_layers
        else:
            is_static = (
                self._cascade_refresh_every > 0
                and (layer_idx + 1) % self._cascade_refresh_every == 0
            )
        if not is_static:
            return False
        if self._cascade_stagger_groups > 1:
            group = self._cascade_stagger_group_map.get(layer_idx, 0)
            return (self._cascade_stagger_step % self._cascade_stagger_groups) == group
        return True

    def _cascade_is_static_refresh(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` is in the static refresh schedule (ignoring stagger)."""
        if self._cascade_refresh_layers is not None:
            return layer_idx in self._cascade_refresh_layers
        return (
            self._cascade_refresh_every > 0
            and (layer_idx + 1) % self._cascade_refresh_every == 0
        )

    def set_cascade_fused_decode(
        self, ext, nsplit: int = 32, hysteresis_scale_matched: bool = False
    ) -> None:
        """Route single-token decode through the fused split-K kernel.

        Call after ``set_cascade_mode(True, ...)``.  Each MLA layer's MQA
        decode then runs sparse attention over its threaded index set via
        ``launch_fused_sparse_splitk`` instead of the full-prefix einsum;
        layers whose cascade set the next layer consumes emit the tap planes
        from the same launch and select the next set through the cascade
        epilogue + all-reduce + top-k chain (see ``fused_decode``).  Prefill
        and all instrumentation paths are unaffected.

        Args:
            ext: The compiled ``cuda_fused_sparse_splitk`` extension (built by
                the caller via ``load_inline``), or ``None`` to restore the
                eager decode path.
            nsplit: Split-K factor for the kernel's key-dimension split.
            hysteresis_scale_matched: Explicit opt-in (H1' registration,
                design_space_prereg.md) that converts a nonzero eager
                ``select_hysteresis`` into tap-plane units instead of
                rejecting it: the planes omit ``softmax_scale`` and reduce
                heads by SUM, so ``delta_fused = delta_eager /
                softmax_scale`` when distilled head weights fold into TapW
                (identical weights on both paths) and ``delta_eager *
                n_local_heads / softmax_scale`` for the uniform tap (the
                eager tap is a local-head MEAN all-reduced across ranks).
                Classic weighted-ReLU tap only — a scoring variant scores in
                a different space and is rejected.
        """
        if ext is None:
            for layer in self.layers:
                layer.attn._fused_decode_state = None
                layer.attn._fused_produce_tap = False
                layer.attn._fused_tapw = None
                layer.attn._cascade_hysteresis_fused = 0.0
            return
        if not self._cascade_mode:
            raise ValueError(
                "fused decode rides the cascade schedule; call "
                "set_cascade_mode(True, ...) first"
            )
        if self._cascade_reuse:
            raise ValueError(
                "fused decode implements the tap cascade, not verbatim reuse"
            )
        attn0 = self.layers[0].attn
        # v4_probs is folded into the kernel (mode-5 exp-tap + exp epilogue:
        # per-head m/Z over the attend slice ARE the kernel's own softmax
        # statistics); every other variant stays eager-only.
        if attn0._cascade_scoring_variant not in (None, "v4_probs"):
            raise ValueError(
                "scoring_variant is an eager-only stage-3 lever: the fused "
                "kernel computes the classic weighted-ReLU tap planes; fold "
                "the winning variant into the kernel before fusing "
                "(v4_probs is the only variant folded so far)"
            )
        if attn0._cascade_stripe > 0 or attn0._cascade_hub_size > 0:
            raise ValueError(
                "admission_stripe and hub_size are eager-only levers: the "
                "fused kernel scores incumbent + window tiles, not stripe or "
                "hub candidates"
            )
        if attn0._cascade_neighbor_budget > 0 or attn0._cascade_block_size > 0:
            raise ValueError(
                "neighbor expansion and block coarse-to-fine are eager-only "
                "levers: the fused kernel scores incumbent + window tiles "
                "only; fold the winning mechanism into the kernel before "
                "fusing"
            )
        if any(layer.attn._refresh_stripe_period > 0 for layer in self.layers):
            raise ValueError(
                "refresh_stripe_period is an eager-only lever: the fused "
                "decode path runs the full lightning-indexer scan on refresh "
                "producers; disable temporal striping or stay on the eager "
                "decode path"
            )
        hysteresis_factor: float | None = None
        if any(layer.attn._cascade_hysteresis > 0 for layer in self.layers):
            if not hysteresis_scale_matched:
                raise ValueError(
                    "select_hysteresis is eager-only by default: the fused "
                    "tap planes are per-head ReLU SUMS of raw QK while the "
                    "eager tap is a head MEAN of softmax_scale-multiplied "
                    "scores, so the same absolute bias sits on an "
                    "incommensurate scale between the two selects (see "
                    "docs/pr29_review_findings.md, transformer.py:1838); "
                    "disable the hysteresis, stay on the eager decode path, "
                    "or opt in to the registered H1' conversion via "
                    "hysteresis_scale_matched=True"
                )
            if attn0._cascade_scoring_variant is not None:
                raise ValueError(
                    "hysteresis_scale_matched is defined for the classic "
                    "weighted-ReLU tap only: scoring variant "
                    f"{attn0._cascade_scoring_variant!r} scores in a "
                    "different space (e.g. v4_probs is post-softmax "
                    "probability mass), so no scalar unit conversion exists"
                )
            if attn0._cascade_head_weights is not None:
                # The same distilled weights fold into the kernel's TapW
                # planes, so the only unit difference is the softmax_scale
                # the planes omit.
                hysteresis_factor = 1.0 / attn0.softmax_scale
            else:
                # Uniform eager tap = LOCAL-head mean, all-reduced across
                # ranks (sum of per-rank means) vs the planes' plain head
                # sum: mean->sum contributes n_local_heads, the omitted
                # softmax_scale the rest.
                hysteresis_factor = attn0.n_local_heads / attn0.softmax_scale
        if any(layer.attn._cascade_prior_lambda > 0 for layer in self.layers):
            raise ValueError(
                "refresh_prior_lambda is an eager-only Phase-A lever: the "
                "fused kernel's cascade epilogue re-ranks its own tap planes "
                "and never gathers the producer's indexer score row; disable "
                "the refresh prior or stay on the eager decode path"
            )
        if attn0.kv_lora_rank != 512 or attn0.qk_rope_head_dim != 64:
            raise ValueError(
                "the fused decode kernel is compiled for the full model's "
                "576-wide joint KV rows (kv_lora_rank=512, rope=64); got "
                f"kv_lora_rank={attn0.kv_lora_rank}, "
                f"qk_rope_head_dim={attn0.qk_rope_head_dim}"
            )
        state = FusedDecodeState(ext, nsplit=nsplit)
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            refresh = self._cascade_is_refresh(i)
            attn = layer.attn
            attn._fused_decode_state = state
            attn._cascade_hysteresis_fused = (
                attn._cascade_hysteresis * hysteresis_factor
                if hysteresis_factor is not None
                else 0.0
            )
            # A layer tap-produces iff the next layer consumes its cascade
            # set: refresh producers hand the indexer's set instead, the
            # final layer produces for nobody, and frozen-admit layers
            # thread the incumbent forward (_cascade_produced discards the
            # tap), so the kernel runs attention-only (mode 0) on them.
            attn._fused_produce_tap = (
                (not refresh)
                and (i < n_layers - 1)
                and (not attn._cascade_freeze_admit)
            )
            weights = attn._cascade_head_weights
            if weights is None:
                if attn._cascade_pool_topk and (
                    attn._cascade_pool_topk != attn.indexer.index_topk
                ):
                    # The pool entry requires a TapW tensor; ones on the real
                    # heads reproduce the uniform tap exactly (pad heads stay
                    # zero, matching the weighted path's convention).
                    h_local = attn.n_local_heads
                    # Explicit float32: the loader leaves bf16 as the global
                    # default dtype, and the kernel requires float32 TapW.
                    ones = torch.zeros(pad_heads(h_local), dtype=torch.float32)
                    ones[:h_local] = 1.0
                    attn._fused_tapw = ones.contiguous()
                else:
                    attn._fused_tapw = None
            else:
                h_local = attn.n_local_heads
                local = weights.float()[rank * h_local : (rank + 1) * h_local]
                padded = local.new_zeros(pad_heads(h_local))
                padded[:h_local] = local
                attn._fused_tapw = padded.contiguous()

    def set_reuse_fused_decode(
        self,
        ext,
        nsplit: int = 32,
        batched_rowloop: bool = False,
    ) -> None:
        """Route reuse-schedule REFRESH layers through the fused kernel.

        The lag + IndexCache composition (mode ``reuse_fused``): verbatim
        propagation everywhere (identical to ``reuse``), except refresh
        producers replace their serial [eager indexer -> eager attention]
        with ONE fused split-K launch — sparse attention over the
        incoming set while the in-kernel fp8 indexer scores the full
        prefix (``with_select=1``); the host top-k of the emitted slab is
        the fresh set the next layer consumes.

        SET-IDENTICAL to eager ``reuse`` (kernel numerics aside): eager
        reuse's refresh layers already attend the INCOMING set under the
        one-layer lag, with the fresh set taking effect next layer.  The
        boundary-shift disclosure both modes carry vs canonical
        IndexCache (refresh-then-attend-own-set): ~1/rho of layers attend
        a one-layer-older set (docs/track_source_cells.md).

        Call after ``set_cascade_mode(True, reuse=True, refresh_...)``.
        Guard relaxation is EXACTLY this composition: the tap-cascade
        fused path (``set_cascade_fused_decode``) still rejects reuse,
        and this path rejects everything that is not verbatim reuse.

        Args:
            ext: The compiled ``cuda_fused_sparse_splitk`` extension, or
                ``None`` to restore the eager decode path.
            nsplit: Split-K factor.
            batched_rowloop: Opt-in for ``bsz > 1`` — the batched kernel
                entry rejects the fp8-indexer modes (IndexK carries no
                per-row stride), so batch rows loop through single-row
                launches (bitwise-equal per the kernel's row-slice
                contract) at a priced per-launch overhead (R2a,
                docs/track_source_cells.md).  Without the flag, bsz > 1
                decode fails fast.
        """
        if ext is None:
            for layer in self.layers:
                layer.attn._reuse_fused_state = None
                layer.attn._reuse_fused_rowloop = False
            return
        if not self._cascade_mode or not self._cascade_reuse:
            raise ValueError(
                "reuse_fused rides the verbatim-reuse schedule; call "
                "set_cascade_mode(True, reuse=True, ...) first"
            )
        if not (
            self._cascade_refresh_every > 0
            or self._cascade_refresh_layers
        ):
            raise ValueError(
                "reuse_fused needs refresh producers (refresh_every > 0 or "
                "refresh_layers); a pure verbatim schedule has no indexer "
                "work to fuse"
            )
        if any(
            layer.attn._fused_decode_state is not None for layer in self.layers
        ):
            raise ValueError(
                "reuse_fused conflicts with the tap-cascade fused decode; "
                "call set_cascade_fused_decode(None) first"
            )
        if self._indexer_source is not None:
            raise ValueError(
                "reuse_fused is incompatible with an active indexer source; "
                "call set_indexer_source(None) first"
            )
        attn0 = self.layers[0].attn
        if attn0.kv_lora_rank != 512 or attn0.qk_rope_head_dim != 64:
            raise ValueError(
                "the fused decode kernel is compiled for the full model's "
                "576-wide joint KV rows (kv_lora_rank=512, rope=64); got "
                f"kv_lora_rank={attn0.kv_lora_rank}, "
                f"qk_rope_head_dim={attn0.qk_rope_head_dim}"
            )
        if attn0.indexer.head_dim != 128:
            raise ValueError(
                "the fused fp8 indexer path is compiled for "
                f"index_head_dim=128, got {attn0.indexer.head_dim}"
            )
        state = ReuseFusedState(ext, nsplit=nsplit)
        for layer in self.layers:
            # State on every layer; routing itself is gated by the
            # per-layer skip/lag flags (_reuse_fused_active), so only
            # refresh producers with an incoming set launch the kernel.
            layer.attn._reuse_fused_state = state
            layer.attn._reuse_fused_rowloop = batched_rowloop

    def _cascade_produced(
        self,
        layer_idx: int,
        layer: Block,
        incoming: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The index set layer ``layer_idx`` hands to the next layer in cascade mode.

        Returns the layer's cascade set, or the lightning indexer's own top-k
        on periodic-refresh layers (and as a fallback when no cascade set was
        produced, e.g. under a dense schedule).  In verbatim-reuse mode
        (IndexCache-style baseline) non-refresh layers hand ``incoming`` —
        the set this layer itself consumed — forward unchanged.  Clears
        ``last_cascade_topk`` so per-layer tensors do not accumulate.
        """
        produced = layer.attn.last_cascade_topk
        layer.attn.last_cascade_topk = None
        refresh = self._cascade_is_refresh(layer_idx)
        # b' select-only scope: non-refresh skip layers thread the incumbent
        # forward unchanged (reuse-copy), exactly like verbatim reuse, so those
        # layers are copy-attention (their own-select was disabled in
        # set_cascade_mode). Refresh producers + layer 0 fall through to produce
        # their own set as in realization b.
        if self._cascade_prefill_select_only and not refresh and layer_idx != 0:
            return incoming if incoming is not None else layer.attn.last_topk_indices
        # ρ_admit cadence: frozen (non-admit) layers thread the incumbent forward
        # unchanged — their _cascade_select was skipped in MLA.forward, so there
        # is no produced set to hand on.
        if layer.attn._cascade_freeze_admit:
            return incoming if incoming is not None else layer.attn.last_topk_indices
        if refresh and layer_idx in self._cascade_adaptive_refresh_set:
            self._cascade_adaptive_refresh_set.discard(layer_idx)
        if self._cascade_adaptive_refresh_threshold > 0 and not refresh:
            self._cascade_track_adaptive_churn(layer_idx, layer, produced)
        if self._cascade_reuse:
            if refresh or incoming is None:
                return layer.attn.last_topk_indices
            return incoming
        if refresh or produced is None:
            # With pool/attend decoupling the exact refresh also refreshes
            # the pool: the indexer's top-pool_topk, not just its top-k.
            # The final layer under skip_indexer_decode reaches here with no
            # indexer scores AND no consumer of the returned value (the loop
            # is ending); fall through to the None its undecoupled
            # counterpart returns rather than raising inside _indexer_pool.
            if (
                layer.attn._cascade_pool_topk
                and layer.attn.last_index_score is not None
            ):
                return layer.attn._indexer_pool()
            return layer.attn.last_topk_indices
        return produced

    def _cascade_apply_adaptive(
        self, layer_idx: int, layer: Block,
    ) -> tuple[bool, bool, bool] | None:
        """Toggle flags for an adaptively-promoted refresh layer.

        Returns the saved state tuple to pass to ``_cascade_restore_adaptive``,
        or ``None`` if no toggling was needed.
        """
        if layer_idx not in self._cascade_adaptive_refresh_set:
            return None
        attn = layer.attn
        saved = (
            attn._fused_produce_tap,
            attn._cascade_skip_indexer_decode,
            attn._cascade_freeze_admit,
        )
        attn._fused_produce_tap = False
        attn._cascade_skip_indexer_decode = False
        attn._cascade_freeze_admit = False
        return saved

    @staticmethod
    def _cascade_restore_adaptive(
        layer: Block, saved: tuple[bool, bool, bool] | None,
    ) -> None:
        """Restore flags after an adaptive refresh step."""
        if saved is None:
            return
        attn = layer.attn
        attn._fused_produce_tap = saved[0]
        attn._cascade_skip_indexer_decode = saved[1]
        attn._cascade_freeze_admit = saved[2]

    def _cascade_apply_stagger(
        self, layer_idx: int, layer: Block,
    ) -> tuple[bool, bool] | None:
        """Demote a stagger-off refresh layer to non-refresh behavior.

        When staggering is active and this layer is in the static refresh
        schedule but NOT in the active stagger group this step, temporarily
        toggle its flags so it behaves like a non-refresh layer (skips
        indexer, respects ae cadence).
        """
        if self._cascade_stagger_groups <= 1:
            return None
        if not self._cascade_is_static_refresh(layer_idx):
            return None
        if self._cascade_is_refresh(layer_idx):
            return None
        attn = layer.attn
        saved = (
            attn._cascade_skip_indexer_decode,
            attn._cascade_freeze_admit,
        )
        attn._cascade_skip_indexer_decode = True
        attn._cascade_freeze_admit = (
            self._cascade_admit_every > 1
            and layer_idx != 0
            and (layer_idx % self._cascade_admit_every) != 0
        )
        return saved

    @staticmethod
    def _cascade_restore_stagger(
        layer: Block, saved: tuple[bool, bool] | None,
    ) -> None:
        """Restore flags after a stagger-demoted layer."""
        if saved is None:
            return
        attn = layer.attn
        attn._cascade_skip_indexer_decode = saved[0]
        attn._cascade_freeze_admit = saved[1]

    def _cascade_track_adaptive_churn(
        self,
        layer_idx: int,
        layer: Block,
        produced: torch.Tensor | None,
    ) -> None:
        """Track selection churn and flag layers for adaptive refresh.

        Called from ``_cascade_produced`` on non-refresh layers when
        ``adaptive_refresh_threshold > 0``.  Computes jaccard between the
        current and previous step's produced selection; when churn
        (``1 - jaccard``) exceeds the threshold, the layer is added to
        ``_cascade_adaptive_refresh_set`` so the NEXT decode step treats
        it as a refresh producer.
        """
        attn = layer.attn
        if produced is None:
            attn._cascade_prev_selection = None
            return
        current = produced.view(-1)
        prev = attn._cascade_prev_selection
        attn._cascade_prev_selection = current.detach()
        if prev is None:
            return
        valid_cur = current[current >= 0]
        valid_prev = prev[prev >= 0]
        if valid_cur.numel() == 0 or valid_prev.numel() == 0:
            return
        set_cur = set(valid_cur.tolist())
        set_prev = set(valid_prev.tolist())
        intersection = len(set_cur & set_prev)
        union = len(set_cur | set_prev)
        if union == 0:
            return
        churn = 1.0 - intersection / union
        if churn > self._cascade_adaptive_refresh_threshold:
            self._cascade_adaptive_refresh_set.add(layer_idx)

    def set_indexer_frozen(self, frozen: bool) -> None:
        """Enable or disable the indexer routing mask during single-token decode.

        When ``frozen=True``, each MLA layer skips applying the ``index_mask``
        in the MQA decode branch — the model attends densely to **all** tokens
        in the current KV cache rather than the top-k selected by the indexer.
        This is used by the ``hub_frozen`` evaluation to measure hub-eviction
        quality independently of the indexer's own dynamic re-selection.
        """
        for layer in self.layers:
            layer.attn._disable_indexer_mask = frozen

    def set_indexer_source(
        self,
        source: str | None,
        rank: int = 32,
        survivors: int = 8192,
        code_dtype: str = "bf16",
        proj_seed: int = 0,
        proj_weights: dict | None = None,
    ) -> None:
        """Install a SOURCE-track indexer mechanism, or clear it (``None``).

        Replaces the lightning indexer's decode-step key scan under the
        unchanged ``same`` schedule (every layer selects every step);
        prefill selection stays on the base fp8 path, which additionally
        appends wide-stage codes so decode scans cover the whole prefix.
        Mutually exclusive with the lag/tap-lag/cascade schedules.

        Sources (docs/track_source_design.md):

        - ``"base_eager"``: baseline indexer semantics through the eager
          bf16 einsum path — the unconfounded speed control for the
          mechanisms below (fp8 TileLang kernel vs eager torch is an
          implementation confound, not a mechanism difference).
        - ``"lowrank"``: (h) two-stage — r-dim coarse scan of ALL keys over
          a cached code, full-width re-rank of the top-``survivors``.
        - ``"ckv"``: (d) alone — full-rank scan of the attention's joint
          [c_KV | k_pe] cache with the query absorbed through a linear map
          (scan-cost negative control; the win is cache elimination).
        - ``"ckv_lowrank"``: (d)+(h) — r-dim codes derived from the joint
          cache, fine stage gathers survivor rows from the same cache; the
          indexer key path (wk / k_norm / fp8 k cache) is never used at
          decode.

        Args:
            source: One of the above, or ``None`` to disable.
            rank: Wide-stage code dimension ``r`` (two-stage sources).
            survivors: Coarse-stage survivor count ``C`` (two-stage sources).
            code_dtype: ``"bf16"`` or ``"fp8"`` code-cache storage (the
                precision-reduction variant of the wide stage).
            proj_seed: Seed of the deterministic random projections (speed
                screening is shape-, not weight-dependent; all TP ranks
                derive identical tensors from the seed).
            proj_weights: Optional fitted projections, ``{"wide_proj":
                {layer_idx: [d_in, r]}, "ckv_map": {layer_idx: [d_h,
                d_ckv]}}`` — the distillation path (ridge/PCA fits from the
                base indexer; see docs/track_source_design.md).
        """
        if source is None:
            self._indexer_source = None
            for layer in self.layers:
                layer.attn.indexer.configure_source(None)
            return
        if self._cascade_mode or self._lag_mode or self._tap_lag_mode:
            raise ValueError(
                "set_indexer_source requires the plain 'same' schedule; "
                "disable cascade/lag/tap-lag modes first"
            )
        if code_dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"code_dtype must be 'bf16' or 'fp8', got {code_dtype!r}"
            )
        attn0 = self.layers[0].attn
        d_h = attn0.indexer.head_dim
        d_ckv = attn0.kv_lora_rank + attn0.qk_rope_head_dim
        index_topk = attn0.indexer.index_topk
        two_stage = source in ("lowrank", "ckv_lowrank")
        if two_stage and survivors < index_topk:
            raise ValueError(
                f"survivors={survivors} must be >= index_topk={index_topk}"
            )
        for i, layer in enumerate(self.layers):
            if two_stage:
                cfg = SourceConfig(
                    source=source, rank=rank, survivors=survivors,
                    code_dtype=code_dtype,
                )
            elif source == "base_eager":
                # Full-width single stage: the 'code' is the key itself.
                cfg = SourceConfig(
                    source=source, rank=d_h, survivors=None,
                    code_dtype=code_dtype,
                )
            elif source == "ckv":
                cfg = SourceConfig(
                    source=source, rank=1, survivors=None, code_dtype="bf16",
                )
            else:
                raise ValueError(
                    "unknown indexer source "
                    f"{source!r}; choose from ('base_eager', 'ckv', "
                    "'lowrank', 'ckv_lowrank') or None"
                )
            wide_proj = None
            ckv_map = None
            d_in = d_ckv if source == "ckv_lowrank" else d_h
            if proj_weights is not None:
                if cfg.uses_code_cache and source != "base_eager":
                    wide_proj = proj_weights["wide_proj"][i].float()
                if source in ("ckv", "ckv_lowrank"):
                    ckv_map = proj_weights["ckv_map"][i].float()
            if cfg.uses_code_cache and wide_proj is None:
                wide_proj = (
                    identity_projection(d_h) if source == "base_eager"
                    else gaussian_projection(d_in, rank, seed=proj_seed * 10007 + i)
                )
            if source in ("ckv", "ckv_lowrank") and ckv_map is None:
                # Random map at the screen; a ridge fit (k_idx ~= M @ c_hat)
                # loads through proj_weights for quality work.
                gen = torch.Generator().manual_seed(proj_seed * 20011 + i)
                ckv_map = torch.randn(
                    d_h, d_ckv, generator=gen, dtype=torch.float32
                ) * (d_ckv ** -0.5)
            layer.attn.indexer.configure_source(cfg, wide_proj, ckv_map)
        self._indexer_source = source

    def set_select_key_source(
        self,
        source: str | None,
        m0_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Install (or clear) the opt-in compressed-latent SELECTION key
        source (ckv ΔPPL gate, docs/prereg_ckv_glm_ppl.md).

        ``source=None`` (default) is the certified base ``wk`` key path,
        untouched.  ``source="ckv"`` makes every indexer derive its selection
        key from the MLA joint cache row via the fitted per-layer M_derot map
        (``m0_by_layer[i]`` is the ``[576,128]`` de-rotated-frame ridge W),
        instead of the base ``wk`` projection — a PURE key-source swap that
        feeds the identical fp8 quant + cache + score path.  Orthogonal to
        cascade/same (it is applied in whichever schedule runs); mutually
        exclusive with the SOURCE track (``set_indexer_source``), which also
        rewrites the key path.  Model-agnostic: the output R_idx rope layout
        follows each indexer's ``indexer_rope_interleave``.
        """
        if source is None:
            for layer in self.layers:
                idx = getattr(layer.attn, "indexer", None)
                if idx is not None:
                    idx._select_key_source = None
                    idx._select_m0 = None
            self._select_key_source = None
            return
        if source != "ckv":
            raise ValueError(
                f"select_key_source must be None or 'ckv', got {source!r}"
            )
        if getattr(self, "_indexer_source", None) is not None:
            raise ValueError(
                "set_select_key_source is mutually exclusive with "
                "set_indexer_source; clear the SOURCE track first"
            )
        if m0_by_layer is None:
            raise ValueError(
                "select_key_source='ckv' requires m0_by_layer="
                "{layer_idx: [576,128]} fitted M_derot maps"
            )
        n_set = 0
        for i, layer in enumerate(self.layers):
            idx = getattr(layer.attn, "indexer", None)
            if idx is None:
                continue
            if i not in m0_by_layer:
                raise ValueError(
                    f"m0_by_layer missing indexer layer {i} (a per-layer M0 "
                    "is required for every indexer-bearing layer)"
                )
            m0 = m0_by_layer[i]
            if tuple(m0.shape) != (576, 128):
                raise ValueError(
                    f"layer {i} M0 must be [576, 128], got {tuple(m0.shape)}"
                )
            dev = idx.wk.weight.device
            idx._select_m0 = m0.to(device=dev, dtype=torch.float32)
            idx._select_key_source = "ckv"
            n_set += 1
        self._select_key_source = "ckv"
        logging.getLogger(__name__).info(
            "select_key_source='ckv' installed on %d indexer layers", n_set
        )

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        *,
        position_ids: torch.Tensor | None = None,
        collect_hidden_layers: set[int] | None = None,
        collect_index_scores: bool = False,
        collect_topk_indices: bool = False,
        collect_tap_topk: bool = False,
        collect_index_layers: set[int] | None = None,
        collect_mla_latents: bool = False,
        collect_attn_mass: bool = False,
        attn_mass_window: int = 128,
        full_logits: bool = False,
        prefill_chunk_size: int | None = None,
        hub_aggregator: str = "softmax_topk",
        n_sink: int = 0,
        patch_hidden: dict[int, list[tuple[int, torch.Tensor]]] | None = None,
        logits_from_pos: int | None = None,
        eviction_mask: torch.Tensor | None = None,
    ) -> ForwardResult:
        """
        Forward pass for the Transformer model.

        Args:
            tokens: Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos: Starting position in the sequence for rotary embeddings.
            position_ids: Explicit per-token position indices of shape
                (seq_len,) used to index ``freqs_cis``. When provided,
                ``start_pos`` is still used for KV-cache management but
                RoPE embeddings are derived from these positions, allowing
                non-contiguous (gapped) position schemes.  Incompatible
                with ``prefill_chunk_size``.
            collect_hidden_layers: If provided, collect the pre-norm hidden
                state (h + residual) at these layer indices.
            collect_index_scores: If True, collect per-layer routing scores
                from the Indexer.
            collect_topk_indices: If True, collect per-layer top-k token
                indices selected by the Indexer.
            collect_tap_topk: If True, collect per-layer top-k token indices
                selected by the *selection tap* (``sum_h w_h ReLU(S_h)`` over
                the layer's own pre-softmax attention scores) — the fidelity
                counterpart to ``collect_topk_indices``.  Honors
                ``collect_index_layers``.
            collect_index_layers: If provided, restrict index score and top-k
                collection to these layer indices only. When ``None`` and
                either flag is True, all layers are collected.
            collect_mla_latents: If True, collect per-layer MLA compressed
                KV latents (kv_cache and pe_cache slices).
            full_logits: If True, return logits for all positions
                (batch, seq, vocab) instead of only the last token
                (batch, vocab).
            prefill_chunk_size: When set, process the input in chunks of this
                size. The first chunk uses the standard dense prefill path;
                subsequent chunks use the compressed-space KV cache path.
                This trades compute for memory, reducing peak attention memory
                from O(seq^2) to O(chunk x seq). Hub scores are accumulated
                incrementally and returned in ``hub_scores`` (the full
                ``index_scores`` matrices are NOT stored).
            hub_aggregator: Per-chunk reducer applied when the chunked
                path emits ``hub_scores`` (only meaningful when
                ``collect_index_scores`` is True). Defaults to
                ``"softmax_topk"`` — the gauge under which the indexer
                is trained. Other accepted values are ``"frequency"``
                and ``"intensity"``; see
                :mod:`lagged_indexer_dsa.inference.modeling.hub_marginalization`
                for definitions and the choice rationale.
            n_sink: Number of leading absolute key positions to exclude
                from per-chunk hub-score accumulation. The first few
                token positions accumulate disproportionate hub scores
                regardless of content; the mask folds into the scatter
                validity term so output positions ``[0, n_sink)`` end
                up at zero. Only meaningful with the chunked path;
                using ``n_sink > 0`` on the dense prefill (which emits
                ``index_scores`` rather than aggregated hub scores)
                raises.
            eviction_mask: Optional bool tensor of shape ``(max_seq_len,)``
                where ``True`` marks a position as logically evicted.
                Evicted positions receive ``-inf`` in the attention prefix
                mask so queries cannot attend to their KV entries, even
                though those entries still occupy cache slots.  Only
                meaningful for the chunked path (``prefill_chunk_size`` must
                be set); enables streaming KV-cache compression where
                positions are logically evicted between chunks without
                physical compaction of the cache.
            patch_hidden: Patchscopes-style hidden state injection. Maps
                ``layer_idx -> [(position, hidden_state), ...]``. After the
                specified layer completes, the pre-norm hidden state
                ``(h + residual)`` at each listed position is replaced with
                the provided tensor. Incompatible with ``prefill_chunk_size``.
            logits_from_pos: When set with ``full_logits=True``, only compute
                logits for positions >= this absolute position. Skips the LM
                head projection and cross-GPU all-gather for earlier chunks,
                reducing both compute and communication. The returned logits
                tensor starts at ``logits_from_pos``.

        Returns:
            ForwardResult with logits and optional collected intermediates.
        """
        bsz, seqlen = tokens.shape

        if patch_hidden and prefill_chunk_size is not None:
            raise ValueError(
                "patch_hidden is incompatible with prefill_chunk_size; "
                "run the target prompt without chunked prefill."
            )

        if position_ids is not None and prefill_chunk_size is not None:
            raise ValueError(
                "position_ids is incompatible with prefill_chunk_size; "
                "non-contiguous positions require the dense prefill path."
            )

        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError(
                f"prefill_chunk_size must be a positive integer, got {prefill_chunk_size!r}"
            )

        cascade_refresh = getattr(self, "_cascade_refresh_layers", None)
        if prefill_chunk_size is not None and cascade_refresh is not None:
            raise ValueError(
                "cascade mode is incompatible with chunked prefill; "
                "use the dense prefill path for cascade evaluation."
            )

        use_chunked = prefill_chunk_size is not None and seqlen > prefill_chunk_size
        if use_chunked:
            return self._forward_chunked(
                tokens,
                start_pos=start_pos,
                chunk_size=prefill_chunk_size,
                collect_hidden_layers=collect_hidden_layers,
                collect_index_scores=collect_index_scores,
                collect_topk_indices=collect_topk_indices,
                collect_tap_topk=collect_tap_topk,
                collect_index_layers=collect_index_layers,
                collect_mla_latents=collect_mla_latents,
                collect_attn_mass=collect_attn_mass,
                attn_mass_window=attn_mass_window,
                full_logits=full_logits,
                logits_from_pos=logits_from_pos,
                hub_aggregator=hub_aggregator,
                n_sink=n_sink,
                eviction_mask=eviction_mask,
            )

        if eviction_mask is not None:
            raise ValueError(
                "eviction_mask requires the chunked prefill path "
                "(set prefill_chunk_size so seqlen > prefill_chunk_size)."
            )

        if n_sink != 0:
            # The dense path returns raw ``index_scores``, not aggregated
            # hub scores; sink masking is meaningful only on the
            # aggregator output. Force the chunked path explicitly
            # rather than silently dropping the kwarg.
            raise ValueError(
                "n_sink > 0 requires the chunked prefill path "
                "(set prefill_chunk_size <= seq_len so the model emits "
                "aggregated hub scores)."
            )

        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"start_pos + seqlen ({start_pos} + {seqlen} = {end_pos}) exceeds "
                f"max_seq_len={self.max_seq_len}; truncate the input or rebuild the "
                "model with a larger max_seq_len."
            )
        self._ensure_kv_cache_capacity(bsz, end_pos, start_pos)
        if position_ids is not None:
            freqs_cis = self.freqs_cis[position_ids]
        else:
            freqs_cis = self.freqs_cis[start_pos:end_pos]
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            ).triu_(1)
            if start_pos > 0:
                # Prepend zeros for prior KV cache positions so mask shape
                # matches index_score of shape (bsz, seqlen, end_pos).
                # Mirrors the same logic in _forward_chunked.
                prefix = torch.zeros(seqlen, start_pos, device=tokens.device)
                mask = torch.cat([prefix, mask], dim=1)
        else:
            mask = None

        hidden_states: dict[int, torch.Tensor] = {}
        index_scores_map: dict[int, torch.Tensor] = {}
        topk_indices_map: dict[int, torch.Tensor] = {}
        tap_topk_map: dict[int, torch.Tensor] = {}
        mla_latents_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        collecting_index = collect_index_scores or collect_topk_indices

        # Always assign (not just when True) so a stale flag from a previous
        # instrumented forward cannot leak into this one.
        for layer in self.layers:
            layer.attn._collect_tap_topk = collect_tap_topk
            layer.attn.last_tap_topk = None

        if collect_attn_mass:
            for layer in self.layers:
                layer.attn._collect_attn_mass = True
                layer.attn._attn_mass_window = attn_mass_window
                layer.attn._n_window_queries = None
                layer.attn.last_attn_mass = None
                layer.attn.last_attn_mass_window = None
                layer.attn.last_dense_attn_mass = None
                layer.attn.last_dense_attn_mass_window = None

        h, residual = self.embed(tokens), None
        lag_prev_topk: torch.Tensor | None = None
        # Refresh-prior threading (S7e): carry the block producer's indexer
        # score row + the consumer's layer distance from it alongside the
        # threaded index set.  Per-forward transient — one [B, S, T] fp32 row
        # alive at a time, never retained across chunks.
        prior_active = self._cascade_mode and self._cascade_prior_lambda > 0
        prior_row: torch.Tensor | None = None
        prior_dist = 0
        # Entries-head threading: previous layer's block summaries (per-
        # forward transient, one [B, S, nb] fp32 alive at a time).
        head_active = self._cascade_mode and self._cascade_block_head_active
        block_tap_prev: torch.Tensor | None = None
        if self._cascade_mode and seqlen == 1 and self._cascade_stagger_groups > 1:
            self._cascade_stagger_step += 1
        for layer_idx, layer in enumerate(self.layers):
            if self._lag_mode or self._cascade_mode or self._tap_lag_mode:
                layer.attn._lag_external_topk = lag_prev_topk
            if prior_active:
                layer.attn._cascade_prior_row = prior_row
                layer.attn._cascade_prior_dist = prior_dist
            if head_active:
                layer.attn._cascade_prev_block_tap = block_tap_prev
            _adaptive_saved = self._cascade_apply_adaptive(layer_idx, layer)
            _stagger_saved = self._cascade_apply_stagger(layer_idx, layer)
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)
            self._cascade_restore_adaptive(layer, _adaptive_saved)
            if head_active:
                block_tap_prev = layer.attn.last_block_tap
                layer.attn.last_block_tap = None
                layer.attn._cascade_prev_block_tap = None
            if self._lag_mode:
                layer.attn._lag_external_topk = None
                lag_prev_topk = layer.attn.last_topk_indices
            elif self._tap_lag_mode:
                layer.attn._lag_external_topk = None
                lag_prev_topk = layer.attn.last_tap_lag_topk
                layer.attn.last_tap_lag_topk = None
            elif self._cascade_mode:
                layer.attn._lag_external_topk = None
                lag_prev_topk = self._cascade_produced(layer_idx, layer, lag_prev_topk)
                if prior_active:
                    layer.attn._cascade_prior_row = None
                    layer.attn._cascade_prior_dist = 0
                    if (
                        layer_idx == 0 or self._cascade_is_refresh(layer_idx)
                    ) and layer.attn.last_index_score is not None:
                        prior_row = layer.attn.last_index_score.detach()
                        prior_dist = 1
                    else:
                        prior_dist += 1
            self._cascade_restore_stagger(layer, _stagger_saved)
            if collect_hidden_layers is not None and layer_idx in collect_hidden_layers:
                hidden_states[layer_idx] = (h + residual).detach()
            if patch_hidden is not None and layer_idx in patch_hidden:
                seq_len_h = h.size(1)
                for pos, patch_tensor in patch_hidden[layer_idx]:
                    if not (0 <= pos < seq_len_h):
                        raise IndexError(
                            f"patch_hidden: pos={pos} out of range for "
                            f"layer {layer_idx} with seq_len={seq_len_h}"
                        )
                    h[:, pos, :] = 0.0
                    residual[:, pos, :] = patch_tensor.to(
                        dtype=residual.dtype, device=residual.device
                    )
            if collecting_index:
                layer_wanted = (
                    collect_index_layers is None or layer_idx in collect_index_layers
                )
                if layer_wanted:
                    if collect_index_scores:
                        index_scores_map[layer_idx] = (
                            layer.attn.last_index_score.detach()
                        )
                    if collect_topk_indices:
                        topk_indices_map[layer_idx] = (
                            layer.attn.last_topk_indices.detach()
                        )
            if collect_tap_topk:
                tap_wanted = (
                    collect_index_layers is None or layer_idx in collect_index_layers
                )
                if tap_wanted and layer.attn.last_tap_topk is not None:
                    tap_topk_map[layer_idx] = layer.attn.last_tap_topk.detach()
                layer.attn.last_tap_topk = None
            if collect_mla_latents:
                mla_latents_map[layer_idx] = (
                    layer.attn.kv_cache[:bsz, start_pos:end_pos].detach().clone(),
                    layer.attn.pe_cache[:bsz, start_pos:end_pos].detach().clone(),
                )
            # Free large per-layer index tensors between layers when they are not
            # being collected; _ppl drives long sequences via repeated
            # full-logits forwards and would otherwise retain O(layers x seq)
            # fp32 score tensors. lag_prev_topk already holds the reference the
            # next layer needs, so clearing last_topk_indices here is safe.
            layer_wanted_for_index = (
                collect_index_layers is None or layer_idx in collect_index_layers
            )
            keep_index_score = collect_index_scores and layer_wanted_for_index
            keep_topk_indices = collect_topk_indices and layer_wanted_for_index
            if not keep_index_score:
                layer.attn.last_index_score = None
            if not keep_topk_indices:
                layer.attn.last_topk_indices = None
            layer.attn.last_raw_qk_scores = None

        h2o_scores = None
        snapkv_scores = None
        dense_h2o_scores = None
        dense_snapkv_scores = None
        if collect_attn_mass:
            for layer in self.layers:
                if layer.attn.last_attn_mass is not None:
                    if h2o_scores is None:
                        h2o_scores = layer.attn.last_attn_mass.detach().clone()
                    else:
                        h2o_scores += layer.attn.last_attn_mass.detach()
                if layer.attn.last_attn_mass_window is not None:
                    if snapkv_scores is None:
                        snapkv_scores = (
                            layer.attn.last_attn_mass_window.detach().clone()
                        )
                    else:
                        snapkv_scores += layer.attn.last_attn_mass_window.detach()
                if layer.attn.last_dense_attn_mass is not None:
                    if dense_h2o_scores is None:
                        dense_h2o_scores = (
                            layer.attn.last_dense_attn_mass.detach().clone()
                        )
                    else:
                        dense_h2o_scores += layer.attn.last_dense_attn_mass.detach()
                if layer.attn.last_dense_attn_mass_window is not None:
                    if dense_snapkv_scores is None:
                        dense_snapkv_scores = (
                            layer.attn.last_dense_attn_mass_window.detach().clone()
                        )
                    else:
                        dense_snapkv_scores += (
                            layer.attn.last_dense_attn_mass_window.detach()
                        )
                layer.attn._collect_attn_mass = False
                layer.attn._n_window_queries = None
                layer.attn.last_attn_mass = None
                layer.attn.last_attn_mass_window = None
                layer.attn.last_dense_attn_mass = None
                layer.attn.last_dense_attn_mass_window = None

        h, _ = self.norm(h, residual)
        if full_logits:
            if logits_from_pos is not None:
                seqlen = h.size(1)
                offset = min(max(0, logits_from_pos - start_pos), seqlen)
                logits = self.head(h[:, offset:].float())
            else:
                logits = self.head(h.float())
        else:
            logits = self.head(h[:, -1].float())
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        return ForwardResult(
            logits=logits,
            hidden_states=hidden_states if collect_hidden_layers is not None else None,
            index_scores=index_scores_map if collect_index_scores else None,
            topk_indices=topk_indices_map if collect_topk_indices else None,
            tap_topk=tap_topk_map if collect_tap_topk else None,
            mla_latents=mla_latents_map if collect_mla_latents else None,
            h2o_scores=h2o_scores,
            snapkv_scores=snapkv_scores,
            dense_h2o_scores=dense_h2o_scores,
            dense_snapkv_scores=dense_snapkv_scores,
        )

    @torch.inference_mode()
    def _forward_chunked(
        self,
        tokens: torch.Tensor,
        *,
        chunk_size: int,
        start_pos: int = 0,
        collect_hidden_layers: set[int] | None = None,
        collect_index_scores: bool = False,
        collect_topk_indices: bool = False,
        collect_tap_topk: bool = False,
        collect_index_layers: set[int] | None = None,
        collect_mla_latents: bool = False,
        collect_attn_mass: bool = False,
        attn_mass_window: int = 128,
        full_logits: bool = False,
        logits_from_pos: int | None = None,
        hub_aggregator: str = "softmax_topk",
        n_sink: int = 0,
        eviction_mask: torch.Tensor | None = None,
    ) -> ForwardResult:
        """Process a long sequence in memory-efficient chunks.

        The first chunk uses the standard dense prefill path.  Subsequent
        chunks use the compressed-space KV cache path (same math as the
        decode path but with multiple query positions), keeping attention
        memory at O(chunk x end_pos) instead of O(seq^2).

        Index scores cannot be stored as full (seq, seq) matrices in
        chunked mode. Instead, per-layer **hub scores** are accumulated
        incrementally via the streaming reducer
        :func:`lagged_indexer_dsa.inference.modeling.hub_marginalization.chunk_hub_contribution`.
        The ``hub_aggregator`` argument selects the reducer; see
        :mod:`lagged_indexer_dsa.inference.modeling.hub_marginalization` for the
        full menu (``"softmax_topk"``, ``"frequency"``, ``"intensity"``)
        and the choice rationale.

        ``n_sink`` is forwarded to ``chunk_hub_contribution`` so each
        chunk's contribution already excludes the sink window; the
        global accumulator therefore stays at zero across the first
        ``n_sink`` absolute key positions without any post-hoc copy.
        """
        from lagged_indexer_dsa.inference.modeling.hub_marginalization import (
            CHUNKED_HUB_MODES,
            chunk_hub_contribution,
        )

        if hub_aggregator not in CHUNKED_HUB_MODES:
            raise ValueError(
                f"hub_aggregator={hub_aggregator!r} is not one of "
                f"{CHUNKED_HUB_MODES}."
            )
        bsz, seqlen = tokens.shape
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"start_pos + seqlen ({start_pos} + {seqlen} = {start_pos + seqlen}) "
                f"exceeds max_seq_len={self.max_seq_len}; truncate the input or "
                "rebuild the model with a larger max_seq_len."
            )
        self._ensure_kv_cache_capacity(bsz, start_pos + seqlen, start_pos)
        if eviction_mask is not None:
            total_len = start_pos + seqlen
            if eviction_mask.dtype is not torch.bool:
                raise ValueError("eviction_mask must be a bool tensor")
            if eviction_mask.device != tokens.device:
                raise ValueError(
                    "eviction_mask must be on the same device as tokens "
                    f"({eviction_mask.device} != {tokens.device})"
                )
            if eviction_mask.numel() < total_len:
                raise ValueError(
                    f"eviction_mask length must be at least {total_len}, "
                    f"got {eviction_mask.numel()}"
                )
        collecting_index = collect_index_scores or collect_topk_indices

        for layer in self.layers:
            layer.attn._collect_tap_topk = collect_tap_topk
            layer.attn.last_tap_topk = None

        hidden_states_chunks: dict[int, list[torch.Tensor]] = {}
        hub_scores_map: dict[int, torch.Tensor] = {}
        topk_indices_chunks: dict[int, list[torch.Tensor]] = {}
        tap_topk_chunks: dict[int, list[torch.Tensor]] = {}
        mla_latents_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        logits_chunks: list[torch.Tensor] = []

        h2o_acc: torch.Tensor | None = None
        snapkv_acc: torch.Tensor | None = None
        dense_h2o_acc: torch.Tensor | None = None
        dense_snapkv_acc: torch.Tensor | None = None
        if collect_attn_mass:
            total_len = start_pos + seqlen
            h2o_acc = torch.zeros(bsz, total_len, device=tokens.device)
            snapkv_acc = torch.zeros(bsz, total_len, device=tokens.device)
            dense_h2o_acc = torch.zeros(bsz, total_len, device=tokens.device)
            dense_snapkv_acc = torch.zeros(bsz, total_len, device=tokens.device)
            for layer in self.layers:
                layer.attn._collect_attn_mass = True
                layer.attn._attn_mass_window = attn_mass_window
                layer.attn.last_attn_mass = None
                layer.attn.last_attn_mass_window = None
                layer.attn.last_dense_attn_mass = None
                layer.attn.last_dense_attn_mass_window = None

        window_start = max(0, seqlen - attn_mass_window) if collect_attn_mass else 0

        for chunk_start in range(0, seqlen, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seqlen)
            clen = chunk_end - chunk_start
            abs_start = start_pos + chunk_start
            abs_end = start_pos + chunk_end

            if collect_attn_mass:
                nwq = max(0, chunk_end - max(chunk_start, window_start))
                for layer in self.layers:
                    layer.attn._n_window_queries = nwq

            chunk_tokens = tokens[:, chunk_start:chunk_end]
            freqs_cis = self.freqs_cis[abs_start:abs_end]

            # Causal mask: within-chunk causal + full access to prior cache.
            # eviction_mask (bool, shape >= abs_end) marks logically-evicted
            # positions as True; those get -inf so queries cannot attend to
            # them even though their KV entries still occupy cache slots.
            chunk_mask = torch.full(
                (clen, clen), float("-inf"), device=tokens.device
            ).triu_(1)
            if abs_start > 0:
                if eviction_mask is not None:
                    prefix = torch.where(
                        eviction_mask[:abs_start].unsqueeze(0).expand(clen, -1),
                        torch.full(
                            (clen, abs_start), float("-inf"), device=tokens.device
                        ),
                        torch.zeros(clen, abs_start, device=tokens.device),
                    )
                else:
                    prefix = torch.zeros(clen, abs_start, device=tokens.device)
                chunk_mask = torch.cat([prefix, chunk_mask], dim=1)

            h, residual = self.embed(chunk_tokens), None
            lag_prev_topk: torch.Tensor | None = None
            # Refresh-prior threading (S7e), reset per chunk: carry the block
            # producer's indexer score row + the consumer's layer distance
            # alongside the threaded index set.  One [B, S, T] fp32 row alive
            # at a time, never retained across chunks.
            prior_active = self._cascade_mode and self._cascade_prior_lambda > 0
            prior_row: torch.Tensor | None = None
            prior_dist = 0
            # Entries-head threading, reset per chunk (same-chunk rows only).
            head_active = self._cascade_mode and self._cascade_block_head_active
            block_tap_prev: torch.Tensor | None = None
            if self._cascade_mode and clen == 1 and self._cascade_stagger_groups > 1:
                self._cascade_stagger_step += 1
            for layer_idx, layer in enumerate(self.layers):
                if self._lag_mode or self._cascade_mode or self._tap_lag_mode:
                    layer.attn._lag_external_topk = lag_prev_topk
                if prior_active:
                    layer.attn._cascade_prior_row = prior_row
                    layer.attn._cascade_prior_dist = prior_dist
                if head_active:
                    layer.attn._cascade_prev_block_tap = block_tap_prev
                _adaptive_saved = self._cascade_apply_adaptive(layer_idx, layer)
                _stagger_saved = self._cascade_apply_stagger(layer_idx, layer)
                h, residual = layer(h, residual, abs_start, freqs_cis, chunk_mask)
                self._cascade_restore_adaptive(layer, _adaptive_saved)
                if head_active:
                    block_tap_prev = layer.attn.last_block_tap
                    layer.attn.last_block_tap = None
                    layer.attn._cascade_prev_block_tap = None
                if self._lag_mode:
                    layer.attn._lag_external_topk = None
                    lag_prev_topk = layer.attn.last_topk_indices
                elif self._tap_lag_mode:
                    layer.attn._lag_external_topk = None
                    lag_prev_topk = layer.attn.last_tap_lag_topk
                    layer.attn.last_tap_lag_topk = None
                elif self._cascade_mode:
                    layer.attn._lag_external_topk = None
                    lag_prev_topk = self._cascade_produced(layer_idx, layer, lag_prev_topk)
                    if prior_active:
                        # Drop this layer's reference immediately (the select
                        # has already consumed it) so at most one prior row is
                        # alive.
                        layer.attn._cascade_prior_row = None
                        layer.attn._cascade_prior_dist = 0
                        # S7e' amendment: layer 0 is the FIRST block's
                        # producer — it always runs its indexer and hands its
                        # set forward — so its score row seeds the prior for
                        # layers 1..first refresh, exactly like a refresh
                        # producer's row does.
                        if (
                            layer_idx == 0 or self._cascade_is_refresh(layer_idx)
                        ) and layer.attn.last_index_score is not None:
                            prior_row = layer.attn.last_index_score.detach()
                            prior_dist = 1
                        else:
                            prior_dist += 1
                self._cascade_restore_stagger(layer, _stagger_saved)

                if (
                    collect_hidden_layers is not None
                    and layer_idx in collect_hidden_layers
                ):
                    hs = (h + residual).detach()
                    hidden_states_chunks.setdefault(layer_idx, []).append(hs)

                if collecting_index:
                    layer_wanted = (
                        collect_index_layers is None
                        or layer_idx in collect_index_layers
                    )
                    if layer_wanted:
                        if collect_index_scores:
                            chunk_score = layer.attn.last_index_score.detach()
                            chunk_topk_indices = layer.attn.last_topk_indices.detach()
                            contribution = chunk_hub_contribution(
                                mode=hub_aggregator,
                                chunk_score=chunk_score,
                                chunk_topk_indices=chunk_topk_indices,
                                total_key_len=abs_end,
                                n_sink=n_sink,
                            )
                            if layer_idx not in hub_scores_map:
                                hub_scores_map[layer_idx] = torch.zeros(
                                    bsz,
                                    start_pos + seqlen,
                                    device=tokens.device,
                                    dtype=contribution.dtype,
                                )
                            hub_scores_map[layer_idx][:, :abs_end] += contribution
                        if collect_topk_indices:
                            topk_indices_chunks.setdefault(layer_idx, []).append(
                                layer.attn.last_topk_indices.detach().cpu()
                            )

                if collect_tap_topk:
                    tap_wanted = (
                        collect_index_layers is None
                        or layer_idx in collect_index_layers
                    )
                    if tap_wanted and layer.attn.last_tap_topk is not None:
                        tap_topk_chunks.setdefault(layer_idx, []).append(
                            layer.attn.last_tap_topk.detach().cpu()
                        )
                    layer.attn.last_tap_topk = None

                # Free large fp32 index-score tensors that accumulate to
                # ~70 GiB across 78 layers at seq=128K if not cleared.
                # lag_prev_topk already holds the reference needed for the
                # next layer, so freeing last_topk_indices here is safe.
                layer.attn.last_index_score = None
                layer.attn.last_topk_indices = None

                if collect_mla_latents:
                    mla_latents_chunks = mla_latents_map.setdefault(layer_idx, ([], []))
                    mla_latents_chunks[0].append(
                        layer.attn.kv_cache[:bsz, abs_start:abs_end].detach().clone()
                    )
                    mla_latents_chunks[1].append(
                        layer.attn.pe_cache[:bsz, abs_start:abs_end].detach().clone()
                    )

                if collect_attn_mass:
                    if layer.attn.last_attn_mass is not None:
                        h2o_acc[:, : layer.attn.last_attn_mass.size(1)] += (
                            layer.attn.last_attn_mass.detach()
                        )
                        layer.attn.last_attn_mass = None
                    if layer.attn.last_attn_mass_window is not None:
                        snapkv_acc[:, : layer.attn.last_attn_mass_window.size(1)] += (
                            layer.attn.last_attn_mass_window.detach()
                        )
                        layer.attn.last_attn_mass_window = None
                    if layer.attn.last_dense_attn_mass is not None:
                        dense_h2o_acc[:, : layer.attn.last_dense_attn_mass.size(1)] += (
                            layer.attn.last_dense_attn_mass.detach()
                        )
                        layer.attn.last_dense_attn_mass = None
                    if layer.attn.last_dense_attn_mass_window is not None:
                        dense_snapkv_acc[
                            :, : layer.attn.last_dense_attn_mass_window.size(1)
                        ] += layer.attn.last_dense_attn_mass_window.detach()
                        layer.attn.last_dense_attn_mass_window = None

            h, _ = self.norm(h, residual)
            if full_logits:
                skip_chunk_logits = (
                    logits_from_pos is not None and abs_end <= logits_from_pos
                )
                if not skip_chunk_logits:
                    logits_offset = (
                        max(0, logits_from_pos - abs_start)
                        if logits_from_pos is not None
                        else 0
                    )
                    chunk_logits = self.head(h[:, logits_offset:].float())
                    if world_size > 1:
                        all_logits = [
                            torch.empty_like(chunk_logits)
                            for _ in range(world_size)
                        ]
                        dist.all_gather(all_logits, chunk_logits)
                        chunk_logits = torch.cat(all_logits, dim=-1)
                    logits_chunks.append(chunk_logits)

        if full_logits:
            if logits_chunks:
                logits = torch.cat(logits_chunks, dim=1)
            else:
                # ``logits_from_pos`` lies past the end of the
                # sequence; emit an empty ``(batch, 0, vocab)`` so
                # downstream consumers see a zero-length slice
                # rather than a ``torch.cat`` on an empty list.
                logits = self.head(h[:, :0].float())
                if world_size > 1:
                    all_logits = [
                        torch.empty_like(logits) for _ in range(world_size)
                    ]
                    dist.all_gather(all_logits, logits)
                    logits = torch.cat(all_logits, dim=-1)
        else:
            logits = self.head(h[:, -1].float())
            if world_size > 1:
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)

        # Assemble collected hidden states by concatenating chunk slices
        hidden_states_combined: dict[int, torch.Tensor] | None = None
        if collect_hidden_layers is not None:
            hidden_states_combined = {
                layer: torch.cat(chunks, dim=1)
                for layer, chunks in hidden_states_chunks.items()
            }

        topk_indices_combined = _cat_index_chunks(topk_indices_chunks, tokens.device)
        tap_topk_combined = _cat_index_chunks(tap_topk_chunks, tokens.device)

        mla_latents_combined: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        if collect_mla_latents and mla_latents_map:
            mla_latents_combined = {
                layer: (
                    torch.cat(kv_chunks, dim=1),
                    torch.cat(pe_chunks, dim=1),
                )
                for layer, (kv_chunks, pe_chunks) in mla_latents_map.items()
            }

        if collect_attn_mass:
            for layer in self.layers:
                layer.attn._collect_attn_mass = False
                layer.attn._n_window_queries = None
                layer.attn.last_attn_mass = None
                layer.attn.last_attn_mass_window = None
                layer.attn.last_dense_attn_mass = None
                layer.attn.last_dense_attn_mass_window = None

        return ForwardResult(
            logits=logits,
            hidden_states=hidden_states_combined,
            index_scores=None,
            topk_indices=topk_indices_combined,
            tap_topk=tap_topk_combined,
            mla_latents=mla_latents_combined,
            hub_scores=hub_scores_map if hub_scores_map else None,
            h2o_scores=h2o_acc,
            snapkv_scores=snapkv_acc,
            dense_h2o_scores=dense_h2o_acc,
            dense_snapkv_scores=dense_snapkv_acc,
        )


if __name__ == "__main__":
    # Interactive smoke check: `python -m lagged_indexer_dsa.inference.modeling.transformer`
    # builds a Transformer with default ``ModelArgs`` and logs the
    # logits shape of a random-token forward pass. The entry point
    # runs outside the test/CLI surface, so we configure logging
    # here (``basicConfig`` is a no-op if the host process has
    # already set up handlers) to keep the project's "logging, not
    # ``print``" rule consistent even for direct-invocation smoke
    # checks.
    logging.basicConfig(level=logging.INFO)
    smoke_logger = logging.getLogger(__name__)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    smoke_logger.info("logits shape: %s", model(x).logits.size())
