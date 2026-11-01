"""Differentiable reference implementation of the lightning-indexer score.

The shipped indexer
(:class:`lagged_indexer_dsa.inference.modeling.transformer.Indexer`) computes
its score through non-differentiable FP8 TileLang kernels (``act_quant`` /
``fp8_index``) followed by a hard ``topk``; ``Transformer`` forwards are
additionally wrapped in ``@torch.inference_mode()``. None of that admits
gradients, so distilling the indexer requires a faithful, fully differentiable
re-implementation of its score.

This module provides exactly that. The score follows DeepSeek-V3.2-Exp Eq. 1::

    I_{t,s} = sum_{j=1..H} w_{t,j} * ReLU(q_{t,j} . k_s)

with ``w_{t,j} = weights_proj(x_t)_j * H^{-0.5} * softmax_scale``.

Two deliberate departures from the inference path, both exact in real
arithmetic:

* The Hadamard ``rotate_activation`` applied to ``q`` and ``k`` before the dot
  product is an orthonormal transform on the head dimension, so it cancels in
  ``q . k`` and is omitted here. (It exists in inference only to spread energy
  for FP8 quantization.)
* The FP8 quantization of ``q``/``k`` is dropped; we compute the dot product in
  full precision. The trained weights are validated against the FP8 kernel's
  top-k selection (the G0 faithfulness gate) before any result is trusted.

RoPE is retained because it encodes relative position and does *not* cancel.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagged_indexer_dsa.inference.modeling.transformer import (
    apply_rotary_emb,
    weight_dequant,
)

logger = logging.getLogger(__name__)


def _plain_weight(linear: nn.Module) -> torch.Tensor:
    """Return a dense fp32 weight matrix from a ``Linear``.

    Handles both bf16/fp32 weights and block-wise FP8 weights (which carry a
    companion ``scale`` buffer and must be dequantized via ``weight_dequant``).
    """
    weight = linear.weight
    if weight.dtype == torch.float8_e4m3fn:
        # Inference reads the FP8 block-scale from the weight parameter itself;
        # fall back to a module-level buffer for older checkpoints.
        scale = getattr(weight, "scale", None)
        if scale is None:
            scale = getattr(linear, "scale", None)
        if scale is None:
            raise ValueError(
                "FP8 Linear is missing its 'scale' buffer; cannot dequantize "
                "indexer weights for the differentiable reference."
            )
        weight = weight_dequant(weight, scale)
    return weight.detach().float().clone()


class IndexerRef(nn.Module):
    """Differentiable lightning-indexer score with trainable projections.

    Parameters mirror :class:`Indexer`: ``wq_b`` (query projection), ``wk`` (key
    projection), ``k_norm`` (LayerNorm over the key head dim) and
    ``weights_proj`` (per-head scalar weights). All are kept in fp32 and marked
    trainable; the backbone that produces the inputs ``x``/``qr`` stays frozen.
    """

    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        indexer_rope_interleave: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rope_head_dim > head_dim:
            raise ValueError(
                f"rope_head_dim ({rope_head_dim}) cannot exceed head_dim ({head_dim})"
            )
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.indexer_rope_interleave = indexer_rope_interleave
        self.eps = eps
        self.softmax_scale = head_dim**-0.5

        # Force fp32 explicitly: the shipped model is often loaded with the
        # process default dtype pinned to bf16 (FP8 checkpoints keep it that
        # way), and constructing these params under that default would make the
        # student score bf16 while its inputs are cast to fp32 -- a dtype
        # mismatch in F.linear. The reference score is defined in fp32.
        self.wq_b_weight = nn.Parameter(torch.empty(n_heads * head_dim, q_lora_rank, dtype=torch.float32))
        self.wk_weight = nn.Parameter(torch.empty(head_dim, dim, dtype=torch.float32))
        self.weights_proj_weight = nn.Parameter(torch.empty(n_heads, dim, dtype=torch.float32))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=torch.float32))
        self.k_norm_bias = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        # Initialize projections so a standalone module (not loaded from a
        # shipped Indexer) is not left with uninitialized memory.
        nn.init.kaiming_uniform_(self.wq_b_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.wk_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.weights_proj_weight, a=5**0.5)

    @torch.no_grad()
    def load_from_indexer(self, indexer: nn.Module) -> None:
        """Copy (and dequantize if needed) weights from a shipped ``Indexer``."""
        self.wq_b_weight.copy_(_plain_weight(indexer.wq_b))
        self.wk_weight.copy_(_plain_weight(indexer.wk))
        self.weights_proj_weight.copy_(_plain_weight(indexer.weights_proj))
        self.k_norm_weight.copy_(indexer.k_norm.weight.detach().float())
        self.k_norm_bias.copy_(indexer.k_norm.bias.detach().float())

    def _project_keys(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Compute RoPE'd key vectors for all positions: ``(b, Sk, head_dim)``."""
        k = F.linear(x.float(), self.wk_weight)
        k = F.layer_norm(k, (self.head_dim,), self.k_norm_weight, self.k_norm_bias, self.eps)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave
        ).squeeze(2)
        return torch.cat([k_pe, k_nope], dim=-1)

    def _project_queries(
        self, xq: torch.Tensor, qrq: torch.Tensor, freqs_cis_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE'd query vectors and per-head weights for query rows.

        Returns ``q`` of shape ``(b, Q, n_heads, head_dim)`` and ``w`` of shape
        ``(b, Q, n_heads)``.
        """
        bsz, q_len, _ = qrq.shape
        q = F.linear(qrq.float(), self.wq_b_weight)
        q = q.view(bsz, q_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis_q, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        w = F.linear(xq.float(), self.weights_proj_weight) * self.n_heads**-0.5
        w = w * self.softmax_scale
        return q, w

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        query_idx: torch.Tensor | None = None,
        query_chunk: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute indexer scores for selected query rows over all key positions.

        Args:
            x: layer input hidden states, ``(b, Sk, dim)``.
            qr: compressed query (MLA ``q_norm(wq_a(x))``), ``(b, Sk, q_lora_rank)``.
            freqs_cis: per-position rotary table for the current positions, ``(Sk, rope_head_dim/2)`` complex.
            query_idx: optional 1-D long tensor of query positions to score; if
                ``None``, all positions are used.
            query_chunk: number of query rows processed per chunk (bounds the
                ``(b, chunk, n_heads, Sk)`` intermediate).

        Returns:
            ``(index_score, query_pos)`` where ``index_score`` is
            ``(b, Q, Sk)`` fp32 (no causal mask applied) and ``query_pos`` is the
            ``(Q,)`` long tensor of scored query positions.
        """
        bsz, seq_k, _ = x.shape
        if query_idx is None:
            query_pos = torch.arange(seq_k, device=x.device)
        else:
            query_pos = query_idx.to(device=x.device, dtype=torch.long)

        if query_chunk <= 0:
            raise ValueError(f"query_chunk must be positive, got {query_chunk}")

        k = self._project_keys(x, freqs_cis)  # (b, Sk, head_dim)

        scores = x.new_empty((bsz, query_pos.numel(), seq_k), dtype=torch.float32)
        for start in range(0, query_pos.numel(), query_chunk):
            sel = query_pos[start : start + query_chunk]
            xq = x.index_select(1, sel)
            qrq = qr.index_select(1, sel)
            fq = freqs_cis.index_select(0, sel)
            q, w = self._project_queries(xq, qrq, fq)  # (b,Cq,H,hd), (b,Cq,H)
            qk = torch.einsum("bqjd,bsd->bqjs", q, k)
            qk = torch.relu(qk)
            chunk_score = torch.einsum("bqjs,bqj->bqs", qk, w)
            scores[:, start : start + sel.numel()] = chunk_score
        return scores, query_pos


def indexer_ref_from_module(indexer: nn.Module) -> IndexerRef:
    """Construct an :class:`IndexerRef` initialized from a shipped ``Indexer``.

    Dimensions are read from the ``Indexer`` module itself (``dim``, ``n_heads``,
    ``head_dim``, ``rope_head_dim``, ``q_lora_rank``, ``indexer_rope_interleave``),
    so no ``ModelArgs`` handle is required.
    """
    ref = IndexerRef(
        dim=indexer.dim,
        q_lora_rank=indexer.q_lora_rank,
        n_heads=indexer.n_heads,
        head_dim=indexer.head_dim,
        rope_head_dim=indexer.rope_head_dim,
        indexer_rope_interleave=indexer.indexer_rope_interleave,
    )
    ref.load_from_indexer(indexer)
    logger.info(
        "Built IndexerRef (n_heads=%d head_dim=%d rope_head_dim=%d) from shipped Indexer",
        indexer.n_heads,
        indexer.head_dim,
        indexer.rope_head_dim,
    )
    return ref
