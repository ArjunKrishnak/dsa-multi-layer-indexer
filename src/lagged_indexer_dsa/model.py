"""A stack of DSA layers runnable under the three Phase 1 schedules.

The model operates directly on hidden states (``inputs_embeds``); no vocabulary
is needed for the equivalence / divergence studies. The same weights are run
under :class:`ScheduleMode` baseline, lagged-reference and lagged-fused.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import AttentionType, ColdStartPolicy, DSAConfig, ScheduleMode
from .dsa_layer import DSALayer, RMSNorm, RopeCaches
from .logging_utils import get_logger
from .rope import build_rope_cache

logger = get_logger("model")


@dataclass
class ForwardTrace:
    """Per-layer record used by the causality and divergence tooling."""

    final_hidden: torch.Tensor
    # Index set actually used for attention at each layer (``None`` == dense).
    indices_used: list[torch.Tensor | None]
    # Index set produced by each layer's indexer (``None`` when the layer runs no
    # indexer: full-attention layers in baseline, and lagged layers whose indexer
    # would have no downstream consumer).
    indices_produced: list[torch.Tensor | None]


class DSAModel(nn.Module):
    def __init__(self, config: DSAConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DSALayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _build_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> RopeCaches:
        c = self.config
        mla_cos, mla_sin = build_rope_cache(seq_len, c.qk_rope_head_dim, c.rope_theta, device, dtype)
        idx_cos, idx_sin = build_rope_cache(
            seq_len, c.index_head_dim, c.rope_theta, device, dtype, interleave=c.indexer_rope_interleave
        )
        return RopeCaches(mla_cos, mla_sin, idx_cos, idx_sin)

    def _cold_start_indices(
        self, layer: DSALayer, x_norm: torch.Tensor, rope: RopeCaches, cold_start: ColdStartPolicy
    ) -> torch.Tensor | None:
        """Attention index set for a DSA layer at index 0 (no previous indexer)."""
        if cold_start is ColdStartPolicy.DENSE:
            return None
        if cold_start is ColdStartPolicy.EAGER:
            return layer.indexer_indices(x_norm, rope)
        raise ValueError(f"unknown cold start policy: {cold_start}")

    def _lagged_indexer_layers(self, types: tuple[AttentionType, ...]) -> list[bool]:
        """Which layers run an indexer under the lagged schedule.

        The lag shifts indexer *production* one layer earlier: layer ``L`` must
        emit an indexer iff layer ``L+1`` attends sparsely (is a DSA layer). This
        is why the last full-attention layer of a dense prefix gains an indexer
        (the boundary indexer that seeds the first DSA layer), and why a DSA layer
        followed by a non-DSA layer -- or the final layer -- runs none.
        """
        n = len(types)
        return [(i + 1 < n and types[i + 1] is AttentionType.DSA) for i in range(n)]

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        mode: ScheduleMode,
        cold_start: ColdStartPolicy = ColdStartPolicy.DENSE,
        record: bool = False,
    ) -> torch.Tensor | ForwardTrace:
        if inputs_embeds.dim() != 3:
            raise ValueError(f"inputs_embeds must be [B,S,H], got {tuple(inputs_embeds.shape)}")
        _, seq_len, _ = inputs_embeds.shape
        rope = self._build_rope(seq_len, inputs_embeds.device, inputs_embeds.dtype)

        types = self.config.resolved_attention_types
        lagged_produces = self._lagged_indexer_layers(types)

        x = inputs_embeds
        indices_prev: torch.Tensor | None = None
        used: list[torch.Tensor | None] = []
        produced: list[torch.Tensor | None] = []

        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, DSALayer)
            x_norm = layer.attn_norm(x)
            is_dsa = types[layer_idx] is AttentionType.DSA

            if mode is ScheduleMode.BASELINE:
                # Standard DSA: each sparse layer computes and consumes its own
                # indexer; full-attention layers attend densely and run none.
                if is_dsa:
                    same_layer_idx = layer.indexer_indices(x_norm, rope)
                    attn_indices: torch.Tensor | None = same_layer_idx
                    produced_idx: torch.Tensor | None = same_layer_idx
                else:
                    attn_indices = None
                    produced_idx = None
                attn_hidden = layer.attention_output(x_norm, rope, attn_indices)
            elif mode in (ScheduleMode.LAGGED_REFERENCE, ScheduleMode.LAGGED_FUSED):
                # Attention consumes the previous layer's indexer output. A full
                # layer attends densely; a DSA layer at index 0 cold-starts.
                if not is_dsa:
                    attn_indices = None
                elif layer_idx == 0:
                    attn_indices = self._cold_start_indices(layer, x_norm, rope, cold_start)
                else:
                    attn_indices = indices_prev
                produce = lagged_produces[layer_idx]

                if mode is ScheduleMode.LAGGED_REFERENCE:
                    # Two independent passes over the layer input (unfused).
                    attn_hidden = layer.attention_output(x_norm, rope, attn_indices)
                    produced_idx = layer.indexer_indices(x_norm, rope) if produce else None
                else:
                    # Single shared read of the layer input (fused analogue).
                    attn_hidden, produced_idx = layer.fused_attention_and_indices(
                        x_norm, rope, attn_indices, produce_indexer=produce
                    )
            else:
                raise ValueError(f"unknown schedule mode: {mode}")

            h = x + attn_hidden
            h_norm = layer.ffn_norm(h)
            x = h + layer.mlp(h_norm)

            indices_prev = produced_idx
            if record:
                used.append(attn_indices)
                produced.append(produced_idx)

        x = self.final_norm(x)
        if record:
            return ForwardTrace(final_hidden=x, indices_used=used, indices_produced=produced)
        return x
