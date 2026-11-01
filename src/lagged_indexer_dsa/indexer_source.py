"""SOURCE-track compression mechanisms for the lightning indexer (CPU-friendly core).

The lightning indexer scores ALL ``L`` prefix keys per layer at full width
(``index_head_dim`` = 128, fp8) to pick top-k for sparse attention.  This
module implements the numeric core of two speed mechanisms that shrink that
O(L) scan, plus the eager reference they are compared against:

* **(h) low-rank wide stage** (``lowrank``): keys are projected ONCE (at
  KV-append time) to ``r`` = 16-32 dims and the O(L) scan runs in that tiny
  dim as a *coarse* stage; only the top-``C`` survivors are re-scored at the
  full 128-dim width.  Scan MACs shrink by ``128/r``; key bytes by the code
  size ratio.
* **(d) c_KV source** (``ckv`` / ``ckv_lowrank``): the key-side input is the
  MLA compressed latent the attention path ALREADY caches (``kv_cache`` +
  ``pe_cache``, jointly 512+64 dims), absorbed into the query via a fitted
  linear map — the indexer's own ``wk`` projection and its separate fp8 key
  cache disappear.  Alone (``ckv``) this scans at 576 dims (a scan-cost
  LOSS, kept as the sign-check control); composed with (h)
  (``ckv_lowrank``) the wide stage runs on r-dim codes derived from c_KV
  and the fine stage gathers survivor rows from the attention cache — the
  indexer key path costs nothing beyond the r-dim code append.
* ``base_eager``: the baseline indexer's exact functional form
  (``sum_h w_h ReLU(q_h . k_s)``) computed through the same eager bf16
  einsum path as the mechanisms — the unconfounded speed control (the
  production fp8 TileLang kernel vs eager torch is an implementation
  confound, not a mechanism difference).

Everything here is plain PyTorch (CPU-safe, no fp8 GEMM kernels) so the
mechanisms validate on dummy weights in seconds; the serving wiring lives in
``inference/modeling/transformer.py`` (``Indexer._forward_source``) and
delegates to these functions.

Op accounting: every scan/gather helper can append ``{op, macs, key_bytes}``
records to a caller-supplied ``oplog`` list; ``predict_decode_op_counts``
gives the analytic model the mini self-test asserts against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .logging_utils import get_logger

logger = get_logger("indexer_source")

VALID_SOURCES = ("base_eager", "ckv", "lowrank", "ckv_lowrank")
CODE_DTYPES = ("bf16", "fp8")

# fp8 e4m3 finite range used for per-key absmax scaling of code caches.
_FP8_MAX = 448.0


@dataclass(frozen=True)
class SourceConfig:
    """Configuration of one indexer-source mechanism.

    Args:
        source: One of :data:`VALID_SOURCES`.
        rank: Wide-stage code dimension ``r`` (equals the full key width for
            ``base_eager``; unused for the direct-scan ``ckv``).
        survivors: Coarse-stage survivor count ``C`` for the two-stage
            sources, or ``None`` for single-stage sources.
        code_dtype: Storage dtype of the wide-stage code cache.
    """

    source: str
    rank: int
    survivors: int | None
    code_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"unknown indexer source {self.source!r}; choose from {VALID_SOURCES}"
            )
        if self.code_dtype not in CODE_DTYPES:
            raise ValueError(
                f"code_dtype must be one of {CODE_DTYPES}, got {self.code_dtype!r}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        two_stage = self.source in ("lowrank", "ckv_lowrank")
        if two_stage and self.survivors is None:
            raise ValueError(f"source {self.source!r} requires survivors (C)")
        if not two_stage and self.survivors is not None:
            raise ValueError(
                f"source {self.source!r} is single-stage; survivors must be None"
            )
        if self.survivors is not None and self.survivors < 1:
            raise ValueError(f"survivors must be >= 1, got {self.survivors}")

    @property
    def uses_code_cache(self) -> bool:
        """Whether this source maintains a wide-stage code cache."""
        return self.source != "ckv"


def gaussian_projection(d_in: int, rank: int, seed: int) -> torch.Tensor:
    """Deterministic random projection ``[d_in, rank]`` with orthonormal columns.

    Speed screening is shape- not weight-dependent, so the screen uses a
    random (but reproducible) projection; a fitted projection (PCA of
    captured base-indexer keys / ridge, see docs/track_source_design.md) is
    loaded through the same tensor slot for quality work later.
    """
    if not 1 <= rank <= d_in:
        raise ValueError(f"rank must be in [1, {d_in}], got {rank}")
    gen = torch.Generator().manual_seed(seed)
    gauss = torch.randn(d_in, rank, generator=gen, dtype=torch.float32)
    # QR gives orthonormal columns: the projection preserves energy in the
    # retained subspace, so identity-at-full-rank sanity checks stay exact.
    q, _ = torch.linalg.qr(gauss)
    return q[:, :rank].contiguous()


def identity_projection(d: int) -> torch.Tensor:
    """``[d, d]`` identity — the ``base_eager`` control's 'projection'."""
    return torch.eye(d, dtype=torch.float32)


def fp8_quantize_per_key(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-key absmax quantization to float8_e4m3fn.

    Args:
        x: ``[..., D]`` float tensor (one key vector per trailing row).

    Returns:
        ``(codes_fp8 [..., D], scales [..., 1] float32)`` with
        ``codes * scales ~= x``.
    """
    absmax = x.detach().abs().amax(dim=-1, keepdim=True).float()
    scale = (absmax / _FP8_MAX).clamp_min(1e-12)
    codes = (x.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def fp8_dequantize(
    codes: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of :func:`fp8_quantize_per_key`."""
    return (codes.to(torch.float32) * scales).to(out_dtype)


def _log(oplog: list | None, op: str, macs: int, key_bytes: int) -> None:
    if oplog is not None:
        oplog.append({"op": op, "macs": int(macs), "key_bytes": int(key_bytes)})


def weighted_relu_scores(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    oplog: list | None = None,
    op_name: str = "scan",
) -> torch.Tensor:
    """Lightning-indexer reduction ``I_{t,s} = sum_h w_{t,h} ReLU(q_{t,h} . k_s)``.

    Args:
        q: ``[B, S, H, D]`` (already roped/absorbed as the source requires).
        keys: ``[B, T, D]`` shared MQA keys (codes, full keys, or c_KV rows).
        weights: ``[B, S, H]`` per-head weights (softmax_scale folded in).

    Returns:
        ``[B, S, T]`` float32 scores.
    """
    if q.dim() != 4 or keys.dim() != 3 or weights.dim() != 3:
        raise ValueError(
            f"expected q [B,S,H,D], keys [B,T,D], weights [B,S,H]; got "
            f"{tuple(q.shape)} / {tuple(keys.shape)} / {tuple(weights.shape)}"
        )
    if q.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"q dim {q.shape[-1]} != key dim {keys.shape[-1]} "
            "(projection/source mismatch)"
        )
    # The scan GEMM runs in the inputs' native dtype (bf16 on GPU — the
    # same class of GEMM the eager MLA decode path uses, keeping eager
    # timings comparable); the ReLU/weighted head reduction is fp32.
    per_head = torch.einsum("bshd,btd->bsht", q, keys.to(q.dtype))
    scores = (per_head.float().relu_() * weights.float().unsqueeze(-1)).sum(dim=2)
    b, s, h, d = q.shape
    t = keys.shape[1]
    _log(oplog, op_name, macs=b * s * h * t * d,
         key_bytes=keys.numel() * keys.element_size())
    return scores


def single_stage_topk(
    scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over a full score row (baseline decode semantics).

    Returns ``(indices [B, S, min(topk, T)], scores [B, S, T])``.
    """
    t = scores.shape[-1]
    indices = scores.topk(min(topk, t), dim=-1)[1]
    return indices, scores


def two_stage_select(
    q_wide: torch.Tensor,
    key_codes: torch.Tensor,
    q_fine: torch.Tensor,
    gather_fine_keys: Callable[[torch.Tensor], torch.Tensor],
    weights: torch.Tensor,
    survivors: int,
    topk: int,
    oplog: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarse r-dim scan over ALL keys, then full-width re-rank of survivors.

    Args:
        q_wide: ``[B, S, H, r]`` coarse queries (projected).
        key_codes: ``[B, T, r]`` cached coarse key codes (dequantized).
        q_fine: ``[B, S, H, Df]`` fine queries (full width / absorbed c_KV).
        gather_fine_keys: ``indices [B, S, C] -> keys [B, S, C, Df]`` — the
            survivor gather from whichever cache backs the fine stage.
        weights: ``[B, S, H]`` per-head weights (shared by both stages).
        survivors: Coarse survivor count ``C`` (clipped to ``T``).
        topk: Final selection size (clipped to ``T``).

    Returns:
        ``(topk_indices [B, S, min(topk, T)], index_score [B, S, T])`` where
        ``index_score`` holds fine scores at survivor columns and ``-inf``
        elsewhere (non-survivors were never fine-scored, so their score is
        unknown and must never win a downstream comparison).
    """
    b, s, h, r = q_wide.shape
    t = key_codes.shape[1]
    if topk > survivors:
        raise ValueError(
            f"topk={topk} > survivors={survivors}: the fine stage cannot "
            "select more keys than the coarse stage passes through"
        )
    coarse = weighted_relu_scores(
        q_wide, key_codes, weights, oplog=oplog, op_name="wide_scan"
    )
    c_eff = min(survivors, t)
    surv_idx = coarse.topk(c_eff, dim=-1)[1]  # [B, S, C]

    fine_keys = gather_fine_keys(surv_idx)  # [B, S, C, Df]
    if fine_keys.shape[:3] != (b, s, c_eff) or fine_keys.dim() != 4:
        raise ValueError(
            f"gather_fine_keys returned {tuple(fine_keys.shape)}, expected "
            f"[{b}, {s}, {c_eff}, Df]"
        )
    df = fine_keys.shape[-1]
    if q_fine.shape[-1] != df:
        raise ValueError(
            f"fine q dim {q_fine.shape[-1]} != fine key dim {df}"
        )
    per_head = torch.einsum("bshd,bscd->bshc", q_fine, fine_keys.to(q_fine.dtype))
    fine = (per_head.float().relu_() * weights.float().unsqueeze(-1)).sum(dim=2)  # [B,S,C]
    _log(oplog, "fine_rerank", macs=b * s * h * c_eff * df,
         key_bytes=fine_keys.numel() * fine_keys.element_size())

    index_score = torch.full(
        (b, s, t), float("-inf"), dtype=torch.float32, device=q_wide.device
    )
    index_score.scatter_(-1, surv_idx, fine)
    topk_indices = index_score.topk(min(topk, t), dim=-1)[1]
    return topk_indices, index_score


def code_bytes_per_key(rank: int, code_dtype: str) -> int:
    """Wide-stage cache bytes per key: bf16 codes, or fp8 codes + fp32 scale."""
    if code_dtype == "bf16":
        return 2 * rank
    if code_dtype == "fp8":
        return rank + 4
    raise ValueError(f"unknown code_dtype {code_dtype!r}")


def predict_decode_op_counts(
    cfg: SourceConfig,
    *,
    seq_len: int,
    n_heads: int,
    d_fine: int,
    d_ckv: int,
    topk: int,
    batch: int = 1,
) -> dict[str, int]:
    """Analytic per-layer decode-step cost of the key-side scan(s).

    Counts only the O(L)/O(C) key-side work (the quantity the mechanisms
    change); constant query-side projections are excluded.  MACs are
    multiply-accumulates (FLOPs = 2x); ``key_bytes`` is the cache traffic a
    fused kernel would read (bf16 codes = 2r B/key, fp8 codes = r+4 B/key,
    fp8 full keys = d+4 B/key, c_KV joint rows = 2*(512+64) B/key).  The
    eager harness additionally materializes the [H, T] per-head score
    tensor (rank-independent) — see docs/track_source_design.md for why the
    eager timing understates the bandwidth win.
    """
    t = seq_len
    h = n_heads
    if cfg.source == "base_eager":
        macs = batch * h * t * d_fine
        kbytes = batch * t * code_bytes_per_key(d_fine, cfg.code_dtype)
    elif cfg.source == "ckv":
        macs = batch * h * t * d_ckv
        kbytes = batch * t * 2 * d_ckv  # joint cache rows are bf16
    elif cfg.source in ("lowrank", "ckv_lowrank"):
        c_eff = min(cfg.survivors, t)
        df = d_fine if cfg.source == "lowrank" else d_ckv
        fine_bytes = (d_fine + 4) if cfg.source == "lowrank" else 2 * d_ckv
        macs = batch * h * (t * cfg.rank + c_eff * df)
        kbytes = batch * (
            t * code_bytes_per_key(cfg.rank, cfg.code_dtype) + c_eff * fine_bytes
        )
    else:  # pragma: no cover - VALID_SOURCES is closed
        raise ValueError(f"unknown source {cfg.source!r}")
    return {"macs": macs, "key_bytes": kbytes}
