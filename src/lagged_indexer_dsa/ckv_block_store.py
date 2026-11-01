"""Compressed-latent block-admission store (CPU-friendly numeric core).

Implements the store that the ckv-adequacy verdict authorizes
(``docs/prereg_ckv_block_admission_store.md``, job 24929 ADEQUATE): a
running block-mean of the indexer key DERIVED from the attention cache's
compressed MLA joint row ``ĉ_s = [c_KV(512) | k_pe(64)]`` via the fitted
map ``M_derot``, instead of a second dedicated per-token indexer-key store.

The deployable map is applied at KV-APPEND (positions are known there), so
it is position-dependent by construction::

    k̂_s = R_idx(s) · M0 · R_mla(s)^{-1} · ĉ_s          (dim 128)

with a STATIC per-layer ``M0 ∈ R^{128×576}`` fit offline in the de-rotated
frame (the adequacy PRIMARY variant).  Because ``M_derot`` carries the
position ``s`` in its two 64-dim rope rotations, the store MUST be
**map-then-mean**: derive ``k̂_s`` per token, THEN accumulate into the
block's running mean.  Applying a static ``M`` to a raw block MEAN would
reintroduce the rope confound the adequacy screen's M_full control
exhibited on unseen positions (prereg §1.1).  This module's
``block_ov32`` decision statistic was itself measured map-then-mean in
``bench/ckv_adequacy_analysis.py`` (per-token ``M_derot`` then
``block_means``), so the deployable store matches the certified screen.

Rope layouts (validated against ``apply_rotary_emb`` 2026-08-11, mirrored
from ``bench/ckv_adequacy_analysis.py``): DS idx-rope is NON-interleaved
(pair ``j`` = dims ``(j, j+32)``); MLA rope is INTERLEAVED (pair ``i`` =
dims ``(2i, 2i+1)``).  Both carry the same 32 yarn frequencies.

Everything here is plain PyTorch (CPU-safe); the kernel-cost instrument for
the append/select budget lives in ``bench/cuda_srck_microbench.py``
(``--arm-ckv``).  Quality is DEFERRED to the distillation pass by
construction — this module makes no recall/PPL claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .logging_utils import get_logger

logger = get_logger("ckv_block_store")

D_PE = 64          # rope block width (both sides)
D_KEY = 128        # indexer key dim
D_CKV = 512        # MLA compressed-KV latent dim
D_LATENT = 576     # joint row width (c_KV 512 + k_pe 64)
BLOCK = 64         # admission block size (tokens)


# ---------------------------------------------------------------------------
# Rope helpers (identical conventions to bench/ckv_adequacy_analysis.py)
# ---------------------------------------------------------------------------

def pairs_noninterleaved(x: torch.Tensor) -> torch.Tensor:
    """[..., 64] real (DS idx-rope layout) -> [..., 32] complex."""
    lead = x.shape[:-1]
    return torch.view_as_complex(
        x.float().reshape(*lead, 2, D_PE // 2).transpose(-1, -2).contiguous()
    )


def unpairs_noninterleaved(z: torch.Tensor) -> torch.Tensor:
    """[..., 32] complex -> [..., 64] real (DS idx-rope layout)."""
    y = torch.view_as_real(z)
    return torch.cat([y[..., 0], y[..., 1]], dim=-1)


def pairs_interleaved(x: torch.Tensor) -> torch.Tensor:
    """[..., 64] real (MLA rope layout) -> [..., 32] complex."""
    lead = x.shape[:-1]
    return torch.view_as_complex(x.float().reshape(*lead, D_PE // 2, 2).contiguous())


def unpairs_interleaved(z: torch.Tensor) -> torch.Tensor:
    """[..., 32] complex -> [..., 64] real (MLA rope layout)."""
    return torch.view_as_real(z).flatten(-2)


# ---------------------------------------------------------------------------
# The deployable map M_derot (map-then-mean building block)
# ---------------------------------------------------------------------------

def derive_key(
    joint: torch.Tensor,   # [..., 576] float (ckv | kpe, exactly as cached)
    freqs: torch.Tensor,   # [..., 32] complex (per-position rope phase)
    m0: torch.Tensor,      # [576, 128] float (static, fit in the de-rotated frame)
    idx_interleave: bool = False,
) -> torch.Tensor:
    """Per-token derived indexer key k̂_s = R_idx(s) · M0 · R_mla(s)^{-1} · ĉ_s.

    This is the EXACT pipeline the adequacy analysis used for the M_derot
    variant (``build_variants``: de-rotate the joint pe block by the known
    phase, apply the static map, re-rotate the key pe block).  ``m0`` is the
    right-multiplied form (``joint_derot @ m0``) so ``k̂ = joint_derot @ m0``.

    ``idx_interleave`` selects the OUTPUT (indexer-key) rope layout, matching
    ``DSAConfig.indexer_rope_interleave``: DeepSeek-V3.2 is non-interleaved
    (rotate-half, the default), GLM-5 is interleaved.  The MLA joint side is
    ALWAYS interleaved (the MLA rope path is model-invariant), so only the
    output ``R_idx`` re-rotation switches.  Supports batched leading dims
    ([T, .] or [B, S, .]) — the pair/unpair helpers are leading-dim generic.
    """
    if joint.shape[-1] != D_LATENT:
        raise ValueError(f"joint must be [..., {D_LATENT}], got {tuple(joint.shape)}")
    if m0.shape != (D_LATENT, D_KEY):
        raise ValueError(f"m0 must be [{D_LATENT}, {D_KEY}], got {tuple(m0.shape)}")
    idx_pairs = pairs_interleaved if idx_interleave else pairs_noninterleaved
    idx_unpairs = unpairs_interleaved if idx_interleave else unpairs_noninterleaved
    ckv, kpe = joint[..., :D_CKV], joint[..., D_CKV:]
    # R_mla(s)^{-1}: de-rotate the interleaved MLA pe block by the phase conj.
    kpe_derot = unpairs_interleaved(pairs_interleaved(kpe) * freqs.conj())
    joint_derot = torch.cat([ckv.float(), kpe_derot], dim=-1)
    pred_derot = joint_derot @ m0.float()
    # R_idx(s): re-rotate the idx-rope pe block of the output (layout per model).
    pred_pe = idx_unpairs(idx_pairs(pred_derot[..., :D_PE]) * freqs)
    return torch.cat([pred_pe, pred_derot[..., D_PE:]], dim=-1)


# ---------------------------------------------------------------------------
# The store: online map-then-mean over 64-token blocks
# ---------------------------------------------------------------------------

@dataclass
class BlockStoreSlab:
    """The store artifacts (prereg §1.2), per select-layer per rank.

    * ``mean``  [nb, 128] bf16  — the coarse-scoring slab read at decode.
    * ``sum``   [nb, 128] fp32  — exact online accumulator (bf16 running
      means drift).
    * ``count`` [nb]      int32 — block denominator (divide-at-score).

    ``mean`` is materialized on demand from ``sum / count``; a fused kernel
    may instead divide at score time and skip the bf16 slab (a kernel
    choice, not a semantics change — prereg §1.2).
    """

    sum: torch.Tensor
    count: torch.Tensor

    @property
    def n_blocks(self) -> int:
        return int(self.count.numel())

    def mean(self) -> torch.Tensor:
        denom = self.count.clamp_min(1).to(torch.float32).unsqueeze(-1)
        return (self.sum / denom).to(torch.bfloat16)


class CkvBlockStore:
    """Running block-mean of the M_derot-derived key (deployable store).

    ``m0`` is the static per-layer map (deployment artifact, fit offline).
    Appends derive the key per token then accumulate into its block; the
    coarse slab is ``block_means()``.  Disabled/empty until the first
    append, at which point it is a strict addition to the decode path (it
    never touches the attention or base-key computation — prereg §1.1).
    """

    def __init__(self, n_blocks: int, m0: torch.Tensor, *, block: int = BLOCK):
        if m0.shape != (D_LATENT, D_KEY):
            raise ValueError(f"m0 must be [{D_LATENT}, {D_KEY}], got {tuple(m0.shape)}")
        self.block = block
        self.m0 = m0.float()
        self.slab = BlockStoreSlab(
            sum=torch.zeros(n_blocks, D_KEY, dtype=torch.float32),
            count=torch.zeros(n_blocks, dtype=torch.int32),
        )

    def append(
        self, joint_row: torch.Tensor, pos: int, freqs_row: torch.Tensor
    ) -> torch.Tensor:
        """Derive k̂ for one token at ``pos`` and accumulate it into its block.

        ``joint_row`` [576], ``freqs_row`` [32] complex.  Returns the derived
        key (also the append arm's timed output).
        """
        khat = derive_key(
            joint_row.unsqueeze(0), freqs_row.unsqueeze(0), self.m0
        )[0]
        b = pos // self.block
        self.slab.sum[b] += khat
        self.slab.count[b] += 1
        return khat

    def append_chunk(
        self, joint: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized append of a contiguous chunk (prefill / batched decode).

        ``joint`` [n, 576], ``positions`` [n] long, ``freqs`` [n, 32] complex.
        Equivalent to looping ``append`` — map-then-mean, scatter-add into
        the fp32 accumulator.  Returns the derived keys [n, 128].
        """
        khat = derive_key(joint, freqs, self.m0)
        blocks = (positions // self.block).long()
        self.slab.sum.index_add_(0, blocks, khat.float())
        self.slab.count.index_add_(
            0, blocks, torch.ones_like(blocks, dtype=torch.int32)
        )
        return khat

    def block_means(self) -> torch.Tensor:
        """[nb, 128] bf16 coarse-scoring slab (divide-at-score materialized)."""
        return self.slab.mean()


def dedicated_block_means(
    keys: torch.Tensor, n_blocks: int, *, block: int = BLOCK
) -> torch.Tensor:
    """Reference block-mean of TRUE per-token indexer keys (dedicated store).

    The uncompressed-source ceiling the store is compared against: block
    means of the base indexer's own keys.  Used by the CPU parity tests and
    as the dedicated-key-store baseline the kernel gate subtracts against.
    """
    kb = keys[: n_blocks * block].reshape(n_blocks, block, -1).float().mean(1)
    return kb.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Admission (C = n_admit blocks); C = 0 is the disabled / no-op path
# ---------------------------------------------------------------------------

def admit_blocks(block_scores: torch.Tensor, n_admit: int) -> torch.Tensor:
    """Top-``n_admit`` block ids by coarse score.  ``n_admit == 0`` -> empty.

    C = 0 is the DISABLED store: no candidate blocks are admitted, so the
    downstream widen contributes nothing and the selection is exactly the
    incumbent set (the store is a strict no-op when off — prereg preflight).
    """
    if n_admit == 0:
        return torch.empty(0, dtype=torch.int64, device=block_scores.device)
    n_admit = min(n_admit, block_scores.numel())
    return block_scores.topk(n_admit).indices.to(torch.int64)


def widen_candidates(
    incumbents: torch.Tensor, admitted_blocks: torch.Tensor, *, block: int = BLOCK
) -> torch.Tensor:
    """Append the admitted blocks' token ids after the incumbent set.

    With ``admitted_blocks`` empty (C = 0) the result is exactly
    ``incumbents`` — the no-op identity the preflight asserts.
    """
    if admitted_blocks.numel() == 0:
        return incumbents.clone()
    offs = torch.arange(block, dtype=torch.int64, device=incumbents.device)
    tok = (admitted_blocks.view(-1, 1) * block + offs).reshape(-1)
    return torch.cat([incumbents, tok])


# ---------------------------------------------------------------------------
# Cost-model instrumentation (analytic append budget, prereg §2.2)
# ---------------------------------------------------------------------------

def predict_append_op_counts(
    *, d_ckv: int = D_CKV, d_pe: int = D_PE, d_key: int = D_KEY
) -> dict[str, int]:
    """Per-token append cost of the ckv store (prereg §2.2).

    ``m0`` matmul MACs + the two 64-dim rope rotations + the block-mean
    accumulate (fp32 add + count).  Reported for the mini self-test and the
    kernel gate's K2 pricing.  Compared against the base indexer's per-token
    ``wk`` key projection (7168×128 = 917,504 MACs) the store REPLACES.
    """
    d_latent = d_ckv + d_pe
    m0_macs = d_latent * d_key                    # 576 * 128 = 73,728
    rot_macs = 2 * (d_pe // 2) * 4                 # two complex 32-pair rotations
    accum_ops = d_key                             # fp32 running-sum add
    base_wk_macs = 7168 * d_key                    # the projection this replaces
    return {
        "m0_macs": m0_macs,
        "rope_rotation_macs": rot_macs,
        "block_accumulate_ops": accum_ops,
        "append_macs_total": m0_macs + rot_macs,
        "base_wk_macs_replaced": base_wk_macs,
    }
