"""Fused-kernel MQA decode for the tap cascade: shared state and selection math.

During single-token decode the cascade replaces, per layer, the full-prefix
attention einsum + index-mask softmax + host-side tap select with one launch
of the fused split-K kernel (``bench/cuda_fused_sparse_splitk.py``): sparse
attention over the layer's index set, with the tap planes for the next
layer's selection emitted from the same QK^T pass (``with_select=3``).

Under tensor parallelism each rank holds a contiguous slice of the attention
heads, so a rank's tap planes are a partial head-sum.  The TP-correct chain
per producing layer is therefore

    fused kernel (mode 3)  ->  cascade epilogue (local plane sum, dedupe)
    ->  concat sink candidates  ->  all-reduce  ->  ``fused_cascade_select``

rather than the single-GPU in-kernel select fold.  The final top-k runs over
``topk + window + n_sink`` candidates, matching ``MLA._cascade_select``'s
candidate set (incumbent set, recency window, leading sinks) exactly; the
tap values differ from the eager path only by the positive ``softmax_scale``
factor, which cannot change the selected set.

The compiled extension object is injected by the caller (e.g.
``bench/decode_throughput.py`` builds it via ``load_inline``); this module
never imports or compiles CUDA itself.
"""

from __future__ import annotations

from typing import Any

import torch

# The kernel processes heads in blocks of 16 (HB); a rank whose local head
# count is smaller (e.g. GLM-5 at world size 32: 64/32 = 2 heads) pads its
# query rows with zeros.  Zero queries give zero scores, zero ReLU tap
# contributions, and discarded attention rows, and the KV tiles are loaded
# once per 16-head block regardless, so the padding costs almost nothing.
HEAD_BLOCK = 16


def pad_heads(n_local_heads: int) -> int:
    """Local head count padded up to the kernel's 16-head block size."""
    return ((n_local_heads + HEAD_BLOCK - 1) // HEAD_BLOCK) * HEAD_BLOCK


class FusedDecodeState:
    """Launch state shared by every MLA layer within a fused decode step.

    All layers of one model see the same shapes (``bsz`` query rows — one per
    batched sequence — ``h_pad`` padded local heads, model-wide top-k and
    admission window), so one set of scratch tensors serves the whole stack;
    layers execute sequentially.
    """

    def __init__(self, ext: Any, nsplit: int = 32):
        if nsplit <= 0:
            raise ValueError(f"nsplit must be positive, got {nsplit}")
        self.ext = ext
        self.nsplit = nsplit
        self._key: tuple | None = None
        self._buf: dict[str, torch.Tensor] = {}

    def scratch(
        self, device: torch.device, h_pad: int, topk: int, win: int,
        bsz: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Scratch tensors for the given decode shape (allocated once)."""
        key = (str(device), h_pad, topk, win, bsz)
        if self._key != key:
            if h_pad % HEAD_BLOCK != 0:
                raise ValueError(f"h_pad must be a multiple of 16, got {h_pad}")
            nblk = h_pad // HEAD_BLOCK
            nb = bsz * nblk * self.nsplit
            ntap = topk + win
            f32 = {"dtype": torch.float32, "device": device}
            self._buf = {
                "part_acc": torch.zeros(nb, 16, 512, **f32),
                "part_m": torch.zeros(nb, 16, **f32),
                "part_l": torch.zeros(nb, 16, **f32),
                # Tap planes, viewed by the kernel as [nblk, S=bsz, topk+win].
                "slab": torch.zeros(nblk * bsz * ntap, **f32),
                # with_select=2 histogram slot; unused in cascade mode but the
                # launcher dereferences the pointer, so keep it real.
                "hist": torch.zeros(bsz, 256, dtype=torch.int32, device=device),
                "tap_out": torch.zeros(bsz, ntap, **f32),
                "out": torch.zeros(
                    1, bsz, h_pad, 512, dtype=torch.bfloat16, device=device
                ),
                # Zero-padded query staging for ranks whose local head count
                # is below the 16-head block (the pad rows stay zero).
                "q_pad": torch.zeros(
                    1, bsz, h_pad, 576, dtype=torch.bfloat16, device=device
                ),
                # fp8 indexer operands: never read in cascade mode (the fp8
                # warps idle), but the launcher validates shape/dtype.
                "idx_q": torch.zeros(
                    1, HEAD_BLOCK, 128, dtype=torch.float8_e4m3fn, device=device
                ),
                "idx_k": torch.zeros(1, 128, dtype=torch.float8_e4m3fn, device=device),
                "idx_w": torch.zeros(1, HEAD_BLOCK, **f32),
                "idx_scale": torch.zeros(1, **f32),
                # Uniform tap weights for the batched entry, which requires a
                # weight vector in cascade mode (ones = the legacy uniform
                # plane sum, a positive rescale that never changes the set).
                "tapw_ones": torch.ones(h_pad, **f32),
                # Per-head exp-tap planes (kernel mode 5 / v4_probs), viewed
                # by the kernel as [nblk*HB, S=bsz, topk+win].
                "sel_slab_h": torch.zeros(
                    nblk * HEAD_BLOCK * bsz * ntap, **f32
                ),
                # Per-head global exp stats from the exp epilogue, one
                # [2, h_pad] block per row (row 0: m2_g in the log2 domain,
                # row 1: 1/Z): the host scores the out-of-plane sink
                # candidates against these so the union top-k compares
                # commensurate v4 probabilities.
                "exp_stats": torch.zeros(bsz, 2, h_pad, **f32),
            }
            self._key = key
        return self._buf


def pad_index_operands(
    idx_q_fp8: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad fp8 indexer operands to the kernel's 16-head block multiple.

    The fused kernel requires ``Hidx % 16 == 0``; models whose
    ``index_n_heads`` is smaller (the mini's 8) pad with zero query rows
    and ZERO weights — a zero-weight head contributes ``w * ReLU(s) = 0``
    to the per-key sum, so the padded scoring is EXACT, not approximate.

    Args:
        idx_q_fp8: ``[B, 1, Hidx, 128]`` fp8 indexer queries.
        weights: ``[B, 1, Hidx]`` float32 folded weights
            (``weights_proj * H^-0.5 * q_scale * softmax_scale``).

    Returns:
        ``(idx_q_padded, weights_padded, h_idx_pad)`` — the input tensors
        unchanged when ``Hidx`` is already a multiple of 16.
    """
    if idx_q_fp8.dim() != 4 or weights.dim() != 3:
        raise ValueError(
            f"expected idx_q [B,1,H,128] and weights [B,1,H], got "
            f"{tuple(idx_q_fp8.shape)} / {tuple(weights.shape)}"
        )
    h_idx = idx_q_fp8.size(2)
    if weights.size(2) != h_idx:
        raise ValueError(
            f"weights heads {weights.size(2)} != idx_q heads {h_idx}"
        )
    h_pad = pad_heads(h_idx)
    if h_pad == h_idx:
        return idx_q_fp8, weights, h_idx
    b, s = idx_q_fp8.size(0), idx_q_fp8.size(1)
    q_pad = torch.zeros(
        b, s, h_pad, idx_q_fp8.size(3),
        dtype=idx_q_fp8.dtype, device=idx_q_fp8.device,
    )
    q_pad[:, :, :h_idx] = idx_q_fp8
    w_pad = torch.zeros(
        b, s, h_pad, dtype=weights.dtype, device=weights.device
    )
    w_pad[:, :, :h_idx] = weights
    return q_pad, w_pad, h_pad


def reuse_fused_slab_need(bsz: int, end_pos: int, rowloop: bool) -> int:
    """SelSlab slots one reuse_fused decode step needs.

    The batched entry packs [bsz, end_pos] rows into the slab; the
    single-row / R2a rowloop path reuses ONE [end_pos] slice per
    sequential launch, so it needs only end_pos slots regardless of bsz.
    (Job 24699 regression: asserting bsz*end_pos unconditionally failed
    the rowloop arm against its correctly-sized bsz=1 scratch.)
    """
    if bsz < 1 or end_pos < 1:
        raise ValueError(f"bsz and end_pos must be >= 1, got {bsz}, {end_pos}")
    return end_pos if (bsz == 1 or rowloop) else bsz * end_pos


class ReuseFusedState:
    """Launch state for the reuse_fused schedule's refresh layers.

    reuse_fused = verbatim-reuse (IndexCache-style) with the refresh
    layers' serial [eager indexer -> eager attention] replaced by ONE
    fused split-K launch (``with_select=1``): sparse attention over the
    INCOMING set while the in-kernel fp8 indexer scores the FULL prefix
    into ``SelSlab [S, Skv]``; the host top-k of that slab is the fresh
    set the next layer consumes.  Set-identical to eager ``reuse``
    (refresh layers already attend the incoming set under the lag — see
    docs/track_source_cells.md, reuse_fused semantics note).

    bsz > 1 uses the R2b batched entry
    (``launch_fused_sparse_splitk_batch_idx``: per-row IndexK/IndexKScale
    strides, certified bitwise row-equal by ``validate_batched``);
    ``batched_rowloop`` opts into the R2a per-row legacy launches instead
    (the control arm, with its priced per-launch overhead recorded in the
    bench output metadata).
    """

    def __init__(self, ext: Any, nsplit: int = 32):
        if nsplit <= 0:
            raise ValueError(f"nsplit must be positive, got {nsplit}")
        self.ext = ext
        self.nsplit = nsplit
        self._key: tuple | None = None
        self._buf: dict[str, torch.Tensor] = {}

    def scratch(
        self, device: torch.device, h_pad: int, t_alloc: int, bsz: int = 1
    ) -> dict[str, torch.Tensor]:
        """Scratch for one launch at KV capacity ``t_alloc``.

        ``bsz`` sizes the buffers for the R2b batched entry (S = bsz rows
        in one launch); the rowloop path reuses the same buffers with
        S=1 semantics per row (part buffers are per-launch scratch, and
        sequential launches on one stream never overlap).
        """
        key = (str(device), h_pad, t_alloc, bsz)
        if self._key != key:
            if h_pad % HEAD_BLOCK != 0:
                raise ValueError(f"h_pad must be a multiple of 16, got {h_pad}")
            if bsz < 1:
                raise ValueError(f"bsz must be >= 1, got {bsz}")
            nb = bsz * (h_pad // HEAD_BLOCK) * self.nsplit
            f32 = {"dtype": torch.float32, "device": device}
            self._buf = {
                "part_acc": torch.zeros(nb, 16, 512, **f32),
                "part_m": torch.zeros(nb, 16, **f32),
                "part_l": torch.zeros(nb, 16, **f32),
                # Full-prefix indexer logits, packed [S, Skv] at the
                # CURRENT Skv (= end_pos) row stride per launch; the
                # launcher validates numel >= S*Skv (with_select 1/2).
                "sel_slab": torch.zeros(bsz * t_alloc, **f32),
                "hist": torch.zeros(bsz, 256, dtype=torch.int32,
                                    device=device),
                "out": torch.zeros(
                    1, bsz, h_pad, 512, dtype=torch.bfloat16, device=device
                ),
                "q_pad": torch.zeros(
                    1, bsz, h_pad, 576, dtype=torch.bfloat16, device=device
                ),
            }
            self._key = key
        return self._buf


def fused_cascade_select(
    cand_scores: torch.Tensor,
    incumbents: torch.Tensor,
    q_pos: int,
    topk: int,
    window: int,
    n_sink: int,
) -> torch.Tensor:
    """Top-k over the deduped candidate union, mapped to absolute key indices.

    Args:
        cand_scores: ``[S, topk + window + n_sink]`` float32 tap logits — the
            (all-reduced) cascade-epilogue output over incumbent and window
            slots, with the sink candidates' logits appended.  Duplicate,
            acausal, and never-written slots hold ``-inf`` (the epilogue and
            the caller's sink-dedup mask guarantee this).
        incumbents: ``[S, topk]`` int32/int64 attended key set; slots whose
            score is ``-inf`` are never selected, so sentinel values are safe.
        q_pos: Absolute position of the (single) query row.
        topk: Number of keys to select (equals the incumbent slot count).
        window: Admission-window slot count.
        n_sink: Sink candidate count.

    Returns:
        ``[S, topk]`` int64 absolute key indices, ``-1``-padded where fewer
        than ``topk`` finite candidates exist.
    """
    S, C = cand_scores.shape
    if C != topk + window + n_sink:
        raise ValueError(
            f"cand_scores has {C} slots, expected topk+window+n_sink = "
            f"{topk + window + n_sink}"
        )
    if incumbents.shape != (S, topk):
        raise ValueError(
            f"incumbents must be [{S}, {topk}], got {tuple(incumbents.shape)}"
        )
    vals, slot = cand_scores.topk(topk, dim=-1)
    w_lo = max(q_pos + 1 - window, 0)
    in_inc = slot < topk
    in_win = (slot >= topk) & (slot < topk + window)
    abs_idx = torch.where(
        in_inc,
        incumbents.long().gather(1, slot.clamp(max=topk - 1)),
        torch.where(in_win, w_lo + (slot - topk), slot - (topk + window)),
    )
    return torch.where(
        torch.isfinite(vals), abs_idx, torch.full_like(abs_idx, -1)
    )


def sink_duplicate_mask(
    incumbents: torch.Tensor, n_sink: int, w_lo: int
) -> torch.Tensor:
    """Mask of sink candidates already present in the incumbent or window slots.

    A key must enter the top-k exactly once; the epilogue already ``-inf``s
    window slots that duplicate incumbents, and this mask does the same for
    the appended sink slots (sink position covered by the window, or sink
    position already in the incumbent set).

    Args:
        incumbents: ``[S, topk]`` int attended key set (sentinel values must
            be outside ``[0, n_sink)``, which ``Skv``-style sentinels are).
        n_sink: Number of leading sink positions.
        w_lo: First window position (``max(q_pos + 1 - window, 0)``).

    Returns:
        ``[S, n_sink]`` bool, True where the sink slot must be ``-inf``.
    """
    sinks = torch.arange(n_sink, device=incumbents.device, dtype=incumbents.dtype)
    in_incumbent = (incumbents.unsqueeze(-1) == sinks.view(1, 1, -1)).any(dim=1)
    in_window = (sinks >= w_lo).view(1, -1).expand_as(in_incumbent)
    return in_incumbent | in_window
