"""Global hub marginalization.

Aggregates per-layer Indexer outputs across query positions to produce a
per-key importance signal, then selects the top-k positions.

Three aggregators are first-class:

* :func:`hub_score_frequency` (C) — count of queries whose top-``k_softmax``
  set contained the key. Uses only the discrete decision the model
  consumes at inference (the per-query top-k mask added to the attention
  logits).

* :func:`hub_score_softmax_topk` (F_1) — sum of per-query softmax mass on
  the key, restricted to the top-``k_softmax`` support per query. This is
  the post-hoc analogue of the indexer's *training target* — DeepSeek-V3.2
  Eq. 4 distils ``softmax(I_{t, S_t})`` against the dense main-attention
  distribution restricted to the top-k set ``S_t``.

* :func:`hub_score_intensity` (F_int) — F_1 / C, the mean softmax mass on
  the key conditional on selection.

The decomposition ``F_1 = C * F_int`` is exact by construction.

The ``k_softmax`` knob (post-hoc, applied at analysis time) controls
which support each aggregator uses. Principled values lie in
``[k_inference, k_train]`` where ``k_train = 2048`` for DeepSeek-V3.2;
values above ``k_train`` extend into the warm-up-only region where the
indexer received only ~2.1B tokens of distillation pressure, vs the
~943.7B tokens the sparse stage applied within the top-``k_train``
support.

Each aggregator accepts an optional ``n_sink`` kwarg that excludes the
first ``n_sink`` key positions from the per-key accumulator entirely.
The first few token positions accumulate disproportionate hub scores
regardless of content (sink ratios of 100-43,000x in layers 9+); the
mask is folded into the existing scatter validity term so the output
tensor is constructed with sink positions already at zero — no clone
or post-hoc mutation. ``n_sink`` propagates through
:func:`chunk_hub_contribution` and the chunked forward path
(:meth:`lagged_indexer_dsa.inference.modeling.Transformer.forward`).

"""

from __future__ import annotations

import torch

#: Default smoothing constant for reciprocal rank fusion. Matches the value
#: used by the original RRF paper (Cormack, Clarke & Buettcher, SIGIR 2009)
#: and is also the default in Pyserini/Anserini.
DEFAULT_RRF_K: int = 60


def _assert_topk_shapes(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor | None,
) -> None:
    """Validate the shapes of the top-K tensors produced by the extractor.

    The indexer emits ``(batch, query_len, K_stored)`` index and score
    tensors per layer. This helper enforces that contract — both rank 3,
    matching shapes when the score tensor is supplied — and raises
    :class:`ValueError` on any deviation. ``topk_scores`` is allowed to
    be ``None`` for aggregators that consume only the discrete decision
    (``hub_score_frequency``).
    """
    if topk_indices.dim() != 3:
        raise ValueError(
            "topk_indices must be 3-D (batch, query, topk), "
            f"got {topk_indices.dim()}-D"
        )
    if topk_scores is None:
        return
    if topk_scores.dim() != 3:
        raise ValueError(
            "topk_scores must be 3-D (batch, query, topk), "
            f"got {topk_scores.dim()}-D"
        )
    if topk_scores.shape != topk_indices.shape:
        raise ValueError(
            "topk_scores and topk_indices must have the same shape, "
            f"got {tuple(topk_scores.shape)} and {tuple(topk_indices.shape)}"
        )


def _resolve_k_softmax(K_stored: int, k_softmax: int | None) -> int:
    """Resolve the per-query softmax support against the stored ``K``.

    Returns ``K_stored`` when *k_softmax* is ``None`` (use every stored
    slot); otherwise returns *k_softmax* after checking it is a positive
    integer no greater than *K_stored*.
    """
    if k_softmax is None:
        return K_stored
    if not isinstance(k_softmax, int) or k_softmax < 1:
        raise ValueError(f"k_softmax must be a positive int, got {k_softmax!r}")
    if k_softmax > K_stored:
        raise ValueError(
            f"k_softmax={k_softmax} exceeds stored K={K_stored}; "
            "extract with a larger K_storage or reduce k_softmax"
        )
    return k_softmax


def _validate_n_sink(n_sink: int, seq_len: int) -> None:
    """Validate ``n_sink`` against ``seq_len``.

    ``n_sink == 0`` is the no-mask default. Negative values reject any
    caller passing a sentinel-style "off" of ``-1`` (use ``0`` instead).
    ``n_sink >= seq_len`` would zero every output position and is almost
    certainly a configuration error.
    """
    if not isinstance(n_sink, int) or n_sink < 0:
        raise ValueError(f"n_sink must be a non-negative int, got {n_sink!r}")
    if n_sink >= seq_len:
        raise ValueError(
            f"n_sink={n_sink} would zero every output position "
            f"(seq_len={seq_len})"
        )


def _validate_topk_bounds(topk_indices: torch.Tensor, seq_len: int, name: str) -> None:
    """Reject positive top-k indices that fall outside ``[0, seq_len)``.

    Negative sentinel values (e.g. ``-1`` for empty slots) are allowed;
    positive indices must refer to valid key positions so the downstream
    scatter/gather cannot read or write out of bounds.
    """
    too_large = topk_indices >= seq_len
    if too_large.any():
        n_bad = int(too_large.sum().item())
        raise ValueError(
            f"{name}: {n_bad} top-k indice(s) are >= seq_len={seq_len}; "
            "negative sentinel values are allowed, but positive indices must "
            "refer to valid key positions."
        )


def hub_score_frequency(
    topk_indices: torch.Tensor,
    seq_len: int,
    *,
    k_softmax: int | None = None,
    n_sink: int = 0,
) -> torch.Tensor:
    """Per-key inclusion-count aggregator (C).

    Counts how many query positions selected each key position in their
    top-``k_softmax`` set. Uses *only* the discrete decision the model
    itself consumes at inference (the per-query top-k mask added to the
    attention logits).

    The ``topk_indices`` tensor may use ``-1`` (or any negative value) as
    a sentinel for invalid slots (e.g. queries near the start of the
    sequence whose causal frontier is smaller than the stored ``K``).
    Invalid slots contribute zero count.

    Args:
        topk_indices: ``(batch, query_len, K_stored)`` int tensor — per-query
            stored top-``K_stored`` indices, sorted in score-descending
            order so that truncating to the first ``k_softmax`` entries
            yields the per-query top-``k_softmax`` set.
        seq_len: Total sequence length (output tensor sizing).
        k_softmax: Number of top-k slots per query to include. ``None``
            uses all stored ``K_stored`` slots.
        n_sink: Number of leading key positions to exclude from the
            accumulator. Slots whose stored index is ``< n_sink`` are
            dropped from the count alongside the ``-1`` sentinels, so
            output positions ``[0, n_sink)`` end up at zero without any
            post-hoc masking step. ``0`` (default) preserves every slot.

    Returns:
        ``(batch, seq_len)`` float tensor of inclusion counts.
    """
    _assert_topk_shapes(topk_indices, None)
    k = _resolve_k_softmax(topk_indices.size(-1), k_softmax)
    _validate_n_sink(n_sink, seq_len)
    _validate_topk_bounds(topk_indices, seq_len, "hub_score_frequency")

    sliced = topk_indices[..., :k]
    bsz = sliced.size(0)
    counts = torch.zeros(bsz, seq_len, device=sliced.device, dtype=torch.float32)
    flat_idx = sliced.flatten(1, 2)
    # ``flat_idx >= n_sink`` collapses two filters into one: the ``-1``
    # sentinel (always negative, so always < n_sink) and the sink window
    # ``[0, n_sink)`` are both rejected. ``n_sink == 0`` reduces this to
    # the original ``flat_idx >= 0`` validity test, which is why folding
    # the sink mask in here costs nothing for the no-mask default.
    valid = flat_idx >= n_sink
    # ``scatter_add_`` rejects negative indices, so the sentinel ``-1``
    # entries in ``flat_idx`` would crash before the ``valid`` mask
    # zeros their contribution. Clamping them to ``0`` makes the scatter
    # bounds-safe while ``valid`` ensures invalid slots add zero.
    safe_idx = flat_idx.clamp(min=0)
    counts.scatter_add_(1, safe_idx, valid.to(counts.dtype))
    return counts


def hub_score_softmax_topk(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_len: int,
    *,
    k_softmax: int | None = None,
    n_sink: int = 0,
) -> torch.Tensor:
    """Per-key softmax-mass aggregator (F_1), restricted to top-``k_softmax``.

    For each query position ``t``, computes ``softmax(I_{t, S_t})`` where
    ``S_t`` is the per-query top-``k_softmax`` index set restricted to
    indices ``>= n_sink``, then sums the resulting mass into a per-key
    accumulator. The output for key ``s`` is::

        hub[s] = sum_t  softmax(I_{t, S_t \\ [0, n_sink)})[s]
                        * 1[s in S_t \\ [0, n_sink)]

    By construction ``hub[s] >= 0`` and ``hub[s] == 0`` for every
    ``s < n_sink``. The softmax is renormalised over the surviving
    support, so each query contributes total mass ``1`` whenever at
    least one of its top-K slots lies outside the sink window;
    ``sum_s hub[s]`` therefore equals the number of queries with at
    least one surviving slot (so it equals ``query_len`` in the typical
    case and drops by one for every query whose entire top-K landed
    inside ``[0, n_sink)``).

    This is the post-hoc analogue of the trained quantity: DeepSeek-V3.2
    sparse-stage loss (Eq. 4) optimises ``softmax(I_{t, S_t})`` to match
    a target distribution per query; summing across queries produces a
    per-key total trained-mass score that respects the indexer's
    softmax-shift-invariance.

    Args:
        topk_scores: ``(batch, query_len, K_stored)`` float tensor — raw
            indexer logits at the stored top-``K_stored`` indices.
        topk_indices: ``(batch, query_len, K_stored)`` int tensor — see
            :func:`hub_score_frequency`. Invalid slots are excluded from
            both the softmax denominator (``-inf`` mask) and the
            accumulator (zero contribution).
        seq_len: Total sequence length.
        k_softmax: Per-query softmax support (truncates the stored
            ``K_stored`` to the first ``k_softmax`` slots, which are
            the largest by score). ``None`` uses all stored slots.
        n_sink: Number of leading key positions to exclude from the
            accumulator. Slots whose stored index is ``< n_sink`` are
            removed from the per-query softmax support along with the
            ``-1`` sentinels. Per-query rows whose entire top-K falls
            inside the sink window are legitimate near-start queries
            and contribute zero (no mass accumulates anywhere in the
            output for that query). ``0`` (default) preserves every
            non-sentinel slot.

    Returns:
        ``(batch, seq_len)`` float tensor of accumulated softmax mass.
    """
    _assert_topk_shapes(topk_indices, topk_scores)
    k = _resolve_k_softmax(topk_indices.size(-1), k_softmax)
    _validate_n_sink(n_sink, seq_len)
    _validate_topk_bounds(topk_indices, seq_len, "hub_score_softmax_topk")

    scores_k = topk_scores[..., :k].float()
    indices_k = topk_indices[..., :k]

    # Producer-bug check: an all-sentinel row can only come from a
    # buggy extractor (the chunked forward guarantees at least one
    # finite slot per query). We surface that distinct from the
    # legitimate "all top-K landed in [0, n_sink)" case below.
    sentinel_invalid = indices_k < 0
    if sentinel_invalid.all(dim=-1).any():
        n_bad = int(sentinel_invalid.all(dim=-1).sum().item())
        raise ValueError(
            "hub_score_softmax_topk: encountered "
            f"{n_bad} query position(s) whose entire top-K slot row is "
            "invalid (all indices < 0). This points at a producer bug "
            "in the indexer top-K extraction path; refusing to silently "
            "zero the softmax of an all -inf row."
        )

    # Combined sentinel + sink filter (see ``hub_score_frequency`` for
    # why this single threshold collapses both rejections; reduces to
    # ``indices_k >= 0`` when n_sink == 0).
    valid = indices_k >= n_sink

    masked = torch.where(
        valid, scores_k, torch.full_like(scores_k, float("-inf"))
    )
    softmax_mass = torch.softmax(masked, dim=-1)
    # Zero any row that is entirely -inf after masking.  Two legitimate
    # causes:
    #   1. All top-K indices fell in the sink window  (original case).
    #   2. With n_sink > 0, near-start queries (positions 0..n_sink-1)
    #      have ALL their visible keys in [0, n_sink); the remaining
    #      top-K slots are filled with causally-masked future positions
    #      that the indexer returned with score -inf.  Those positions
    #      pass the ``valid`` filter (index >= n_sink) but still carry
    #      -inf, so ``masked`` ends up all-inf → NaN softmax.
    # Detecting "all -inf" subsumes both cases without changing the
    # math for rows that have at least one finite entry.
    all_neginf = (masked == float("-inf")).all(dim=-1)
    if all_neginf.any():
        softmax_mass = torch.where(
            all_neginf.unsqueeze(-1),
            torch.zeros_like(softmax_mass),
            softmax_mass,
        )
    if not torch.isfinite(softmax_mass).all():
        raise ValueError(
            "hub_score_softmax_topk: softmax produced non-finite values. "
            "This typically means topk_scores contains NaN/inf entries. "
            "Inspect the upstream indexer extraction; the aggregator "
            "refuses to silently coerce non-finite entries to zero."
        )
    softmax_mass = softmax_mass * valid.to(softmax_mass.dtype)

    bsz = indices_k.size(0)
    out = torch.zeros(bsz, seq_len, device=indices_k.device, dtype=softmax_mass.dtype)
    # See the comment in ``hub_score_frequency`` for why we clamp the
    # sentinel-bearing rows before scattering.
    flat_idx = indices_k.clamp(min=0).flatten(1, 2)
    flat_mass = softmax_mass.flatten(1, 2)
    out.scatter_add_(1, flat_idx, flat_mass)
    return out


def hub_score_intensity(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_len: int,
    *,
    k_softmax: int | None = None,
    n_sink: int = 0,
) -> torch.Tensor:
    """Per-key intensity-per-inclusion aggregator (F_int = F_1 / C).

    Returns the mean softmax mass on each key, conditional on the key
    being selected by at least one query. Keys with zero inclusion count
    map to zero (rather than NaN) so the output is safe to use as a
    weight signal.

    Equivalent to ``hub_score_softmax_topk(...) / hub_score_frequency(...)``
    on the support where ``C > 0`` and the same ``k_softmax`` / ``n_sink``.

    Args:
        topk_scores: see :func:`hub_score_softmax_topk`.
        topk_indices: see :func:`hub_score_softmax_topk`.
        seq_len: Total sequence length.
        k_softmax: Per-query softmax support; must match across F_1 and
            C for the decomposition to hold.
        n_sink: see :func:`hub_score_frequency`. Threaded through to
            both the F_1 and C calls so the decomposition still holds
            on the masked output.

    Returns:
        ``(batch, seq_len)`` float tensor of intensity scores.
    """
    f1 = hub_score_softmax_topk(
        topk_scores, topk_indices, seq_len,
        k_softmax=k_softmax, n_sink=n_sink,
    )
    counts = hub_score_frequency(
        topk_indices, seq_len,
        k_softmax=k_softmax, n_sink=n_sink,
    )
    return torch.where(counts > 0, f1 / counts.clamp(min=1.0), torch.zeros_like(f1))


# ---------------------------------------------------------------------------
# Compatibility dispatch by mode
# ---------------------------------------------------------------------------

#: Mode strings the dispatch helpers accept. Kept as a single source of
#: truth so :func:`compute_hub_scores` and
#: :func:`chunk_hub_contribution` validate against the same surface.
CHUNKED_HUB_MODES = ("frequency", "softmax_topk", "intensity")


def compute_hub_scores(
    *,
    mode: str,
    topk_indices: torch.Tensor,
    seq_len: int,
    topk_scores: torch.Tensor | None = None,
    k_softmax: int | None = None,
) -> torch.Tensor:
    """Dispatch to one of the canonical aggregators by name.

    Thin wrapper around :func:`hub_score_frequency`,
    :func:`hub_score_softmax_topk`, and :func:`hub_score_intensity`
    so callers that drive an aggregator grid (vary ``mode`` /
    ``k_softmax`` over a config matrix) can dispatch by string
    label without re-implementing the if/elif chain.

    The ``topk_scores`` argument is mandatory for the
    score-consuming modes and is rejected with ``ValueError`` when
    omitted; ``"frequency"`` ignores it because it only counts
    indices.
    """
    if mode == "frequency":
        return hub_score_frequency(topk_indices, seq_len, k_softmax=k_softmax)
    if mode == "softmax_topk":
        if topk_scores is None:
            raise ValueError(
                "topk_scores is required for mode='softmax_topk'"
            )
        return hub_score_softmax_topk(
            topk_scores, topk_indices, seq_len, k_softmax=k_softmax
        )
    if mode == "intensity":
        if topk_scores is None:
            raise ValueError(
                "topk_scores is required for mode='intensity'"
            )
        return hub_score_intensity(
            topk_scores, topk_indices, seq_len, k_softmax=k_softmax
        )
    raise ValueError(
        f"Unknown aggregator mode: {mode!r}. "
        f"Expected one of {CHUNKED_HUB_MODES}."
    )


# ---------------------------------------------------------------------------
# Streaming reducer for ``Transformer._forward_chunked``
# ---------------------------------------------------------------------------


def chunk_hub_contribution(
    *,
    mode: str,
    chunk_score: torch.Tensor,
    chunk_topk_indices: torch.Tensor | None,
    total_key_len: int,
    k_softmax: int | None = None,
    n_sink: int = 0,
) -> torch.Tensor:
    """Reduce a single chunk's index logits into per-key hub contributions.

    Designed for streaming use inside the chunked forward path: each
    chunk produces an additive ``(batch, total_key_len)`` contribution
    that the caller scatters into the global hub-score accumulator. The
    output positions are the *absolute* key positions referenced by the
    indexer's top-K (or column-sum support); positions outside the
    chunk's causal frontier receive zero contribution naturally.

    Args:
        mode: One of :data:`CHUNKED_HUB_MODES`. ``"softmax_topk"`` is
            the recommended default; ``"frequency"`` is gauge-correct
            without storing scores; ``"intensity"`` is the per-selection
            mean softmax mass.
        chunk_score: ``(batch, q_chunk, total_key_len)`` dense indexer
            logits for this chunk. May contain ``-inf`` causal-mask
            entries.
        chunk_topk_indices: ``(batch, q_chunk, K_topk)`` per-query top-K
            indices. Required by every aggregator.
        total_key_len: Total key dimension of ``chunk_score`` (== global
            ``abs_end`` at chunk evaluation time).
        k_softmax: Optional truncation of the per-query top-K used by
            ``"softmax_topk"`` / ``"intensity"`` (defaults to all
            stored slots).
        n_sink: Number of leading absolute key positions to exclude
            from the contribution. Threaded through to the per-mode
            aggregator. ``0`` (default) preserves every position.

    Returns:
        ``(batch, total_key_len)`` float tensor of per-key contributions
        from this chunk.
    """
    if chunk_topk_indices is None:
        raise ValueError(
            f"chunk_hub_contribution(mode={mode!r}) requires "
            "chunk_topk_indices; every aggregator depends on the "
            "stored top-K indices."
        )
    _validate_topk_bounds(chunk_topk_indices, total_key_len, "chunk_hub_contribution")

    if mode == "frequency":
        return hub_score_frequency(
            chunk_topk_indices, total_key_len,
            k_softmax=k_softmax, n_sink=n_sink,
        )

    # ``softmax_topk`` / ``intensity`` need top-K *scores*. Gather them
    # from the dense chunk logits at the stored indices. Invalid slots
    # (sentinel ``-1`` in ``chunk_topk_indices``) are clamped to ``0``
    # for the gather and re-marked invalid by the aggregator's own
    # validity mask.
    safe_idx = chunk_topk_indices.clamp(min=0)
    chunk_topk_scores = chunk_score.gather(2, safe_idx)

    if mode == "softmax_topk":
        return hub_score_softmax_topk(
            chunk_topk_scores,
            chunk_topk_indices,
            total_key_len,
            k_softmax=k_softmax,
            n_sink=n_sink,
        )
    if mode == "intensity":
        return hub_score_intensity(
            chunk_topk_scores,
            chunk_topk_indices,
            total_key_len,
            k_softmax=k_softmax,
            n_sink=n_sink,
        )

    raise ValueError(
        f"Unknown chunked hub aggregator mode: {mode!r}. "
        f"Expected one of {CHUNKED_HUB_MODES}."
    )


def query_probe_scores(
    topk_indices: torch.Tensor,
    seq_len: int,
    *,
    n_sink: int = 0,
) -> torch.Tensor:
    """Compute scores from the last query position's top-k selection only.

    Diagnostic variant: instead of marginalizing over all queries, uses
    only the final query position's routing decisions. When the input
    ends with a task-oriented suffix (e.g. "Summary:"), this captures
    which keys the model actually routes to for that task.

    Args:
        topk_indices: Shape ``(batch, query_len, topk)`` — per-query
            routing decisions from the Indexer.
        seq_len: Total sequence length (for output tensor sizing).
        n_sink: Number of leading positions to exclude. Slots whose
            stored index is ``< n_sink`` are dropped from the scatter
            alongside the ``-1`` sentinels. ``0`` (default) preserves
            every slot.

    Returns:
        Binary scores of shape ``(batch, seq_len)`` — 1.0 for positions
        selected by the last query, 0.0 otherwise.
    """
    if topk_indices.dim() != 3:
        raise ValueError(
            f"topk_indices must be 3-D (batch, query, topk), got {topk_indices.dim()}-D"
        )
    _validate_n_sink(n_sink, seq_len)
    _validate_topk_bounds(topk_indices, seq_len, "query_probe_scores")
    bsz = topk_indices.size(0)
    last_query_topk = topk_indices[:, -1, :]  # (batch, topk)
    # Single threshold filters both sentinels (always negative) and the
    # sink window in one pass; reduces to ``>= 0`` for n_sink == 0.
    valid = last_query_topk >= n_sink
    last_query_topk = last_query_topk.clamp(min=0)
    scores = torch.zeros(bsz, seq_len, device=topk_indices.device)
    # ``scatter_add_`` (not ``scatter_``) so a clamped sentinel landing on
    # index 0 cannot overwrite a genuine selection of key 0; clamp back to
    # a 0/1 membership indicator afterwards.
    scores.scatter_add_(1, last_query_topk, valid.to(scores.dtype))
    return scores.clamp_(max=1.0)


def _min_max_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Normalize scores to [0, 1] per batch element via min-max scaling."""
    if scores.dim() == 1:
        lo, hi = scores.min(), scores.max()
        denom = hi - lo
        return (scores - lo) / denom if denom > 0 else torch.zeros_like(scores)
    lo = scores.min(dim=-1, keepdim=True).values
    hi = scores.max(dim=-1, keepdim=True).values
    denom = hi - lo
    return torch.where(denom > 0, (scores - lo) / denom, torch.zeros_like(scores))


#: Canonical cross-layer fusion methods accepted by :func:`fuse_hub_scores`.
#: Surface as ``choices=FUSION_METHODS`` in argparse so typos fail at
#: parse time rather than at the bottom of an eval loop.
FUSION_METHODS: tuple[str, ...] = (
    "mean",
    "rrf",
    "geomean",
    "cross_intersect",
    "cross_mean_rank",
    "cross_group",
)


def fuse_hub_scores(
    hub_scores_by_layer: dict[int, torch.Tensor],
    method: str = "mean",
    rrf_k: int = DEFAULT_RRF_K,
    budget: float | None = None,
) -> torch.Tensor:
    """Combine hub scores from multiple layers into a single ranking.

    Args:
        hub_scores_by_layer: Mapping from layer index to hub scores of
            shape ``(seq_len,)`` or ``(batch, seq_len)``.
        method: Fusion strategy — one of :data:`FUSION_METHODS`.
        rrf_k: Smoothing constant for reciprocal rank fusion (default 60).
        budget: Retention ratio required by ``cross_intersect`` and
            ``cross_group`` (used to determine per-layer top-k).

    Returns:
        Fused scores with the same shape as the input tensors.
    """
    if not hub_scores_by_layer:
        raise ValueError("hub_scores_by_layer must not be empty")
    if method not in FUSION_METHODS:
        raise ValueError(
            f"Unknown fusion method: {method!r}. Expected one of {FUSION_METHODS}."
        )

    layers = sorted(hub_scores_by_layer.keys())
    tensors = [hub_scores_by_layer[layer] for layer in layers]

    ref_shape = tensors[0].shape
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: layer {layers[0]} has shape {ref_shape} "
                f"but layer {layers[i]} has shape {t.shape}"
            )

    # Reject non-finite hub scores at the boundary so they cannot
    # propagate through ``_min_max_normalize`` / ``mean`` / ``geomean``
    # / RRF and emit a NaN-poisoned fused score downstream.
    for layer, t in zip(layers, tensors, strict=True):
        finite_mask = torch.isfinite(t)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum().item())
            raise ValueError(
                f"fuse_hub_scores: layer {layer} has {n_bad} of {t.numel()} "
                "non-finite hub-score entries (NaN / +Inf / -Inf). The "
                "canonical aggregators emit finite scores; a non-finite "
                "entry indicates an upstream scoring bug."
            )

    if len(tensors) == 1:
        return tensors[0].clone()

    if method == "mean":
        normed = torch.stack([_min_max_normalize(t) for t in tensors])
        return normed.mean(dim=0)

    if method == "rrf":
        # ``rank == 0`` is a valid output of argsort (the top-ranked
        # position), so the RRF denominator ``rrf_k + rank`` is zero
        # iff ``rrf_k == 0``. The classic Cormack et al. (2009)
        # default is 60; we reject ``rrf_k <= 0`` rather than letting
        # the top-ranked position return ``inf`` and dominate the
        # fused score.
        if rrf_k <= 0:
            raise ValueError(
                f"rrf_k must be > 0 for method='rrf' to avoid a zero "
                f"denominator at rank 0, got rrf_k={rrf_k!r}"
            )
        ranks = []
        for t in tensors:
            order = t.argsort(dim=-1, descending=True)
            rank = torch.zeros_like(t)
            if t.dim() == 1:
                rank[order] = torch.arange(len(t), dtype=t.dtype, device=t.device)
            else:
                for b in range(t.size(0)):
                    rank[b, order[b]] = torch.arange(
                        t.size(1), dtype=t.dtype, device=t.device,
                    )
            ranks.append(rank)
        fused = torch.zeros_like(tensors[0])
        for rank in ranks:
            fused += 1.0 / (rrf_k + rank)
        return fused

    if method == "geomean":
        normed = torch.stack([_min_max_normalize(t) for t in tensors])
        eps = 1e-8
        log_sum = (normed + eps).log().sum(dim=0)
        return (log_sum / len(tensors)).exp()

    if method == "cross_intersect":
        if budget is None:
            raise ValueError("cross_intersect requires a budget parameter")
        return _cross_layer_intersect_count(hub_scores_by_layer, budget)

    if method == "cross_mean_rank":
        return _cross_layer_mean_rank(hub_scores_by_layer)

    if method == "cross_group":
        if budget is None:
            raise ValueError("cross_group requires a budget parameter")
        return _cross_layer_group_intersect(hub_scores_by_layer, budget)

    # Unreachable on a consistent build: the membership check above
    # already rejects anything outside ``FUSION_METHODS``. The explicit
    # raise here surfaces drift if a method is added to
    # ``FUSION_METHODS`` without a matching dispatch branch.
    raise AssertionError(
        f"FUSION_METHODS lists {method!r} but fuse_hub_scores has no branch for it"
    )


def _per_batch_intersect_count(
    per_layer_1d: dict[int, torch.Tensor],
    budget: float,
) -> torch.Tensor:
    """1-D cross-layer intersect-count (single batch entry).

    Operates on a per-layer mapping of 1-D ``(seq_len,)`` tensors and
    returns a 1-D ``(seq_len,)`` count vector. The 1-D / 2-D dispatch
    happens in :func:`_cross_layer_intersect_count`.
    """
    layers = sorted(per_layer_1d.keys())
    first = per_layer_1d[layers[0]]
    seq_len = first.shape[-1]
    counts = torch.zeros(seq_len, device=first.device, dtype=first.dtype)
    for layer in layers:
        scores = per_layer_1d[layer]
        k = resolve_budget(seq_len, budget)
        _, topk_idx = scores.topk(k)
        counts[topk_idx] += 1.0
    return counts


def _cross_layer_intersect_count(
    hub_scores_by_layer: dict[int, torch.Tensor],
    budget: float,
) -> torch.Tensor:
    """Score each position by how many layers include it in their top-k.

    Preserves the input batch dimension: ``(batch, seq_len)`` inputs
    yield a ``(batch, seq_len)`` count tensor, ``(seq_len,)`` inputs
    yield ``(seq_len,)``.
    """
    first = hub_scores_by_layer[next(iter(hub_scores_by_layer))]
    if first.dim() == 1:
        return _per_batch_intersect_count(hub_scores_by_layer, budget)
    if first.dim() == 2:
        batch_size = first.size(0)
        per_batch = []
        for b in range(batch_size):
            sliced = {layer: t[b] for layer, t in hub_scores_by_layer.items()}
            per_batch.append(_per_batch_intersect_count(sliced, budget))
        return torch.stack(per_batch, dim=0)
    raise ValueError(
        f"cross_intersect requires 1-D or 2-D tensors, got {first.dim()}-D"
    )


def _per_batch_mean_rank(
    per_layer_1d: dict[int, torch.Tensor],
) -> torch.Tensor:
    """1-D cross-layer mean-rank (single batch entry)."""
    layers = sorted(per_layer_1d.keys())
    first = per_layer_1d[layers[0]]
    seq_len = first.shape[-1]
    rank_sum = torch.zeros(seq_len, device=first.device, dtype=first.dtype)
    for layer in layers:
        scores = per_layer_1d[layer]
        order = scores.argsort(descending=True)
        rank = torch.zeros_like(scores)
        rank[order] = torch.arange(
            seq_len, dtype=scores.dtype, device=scores.device,
        )
        rank_sum += 1.0 - (rank / seq_len)
    return rank_sum / len(layers)


def _cross_layer_mean_rank(
    hub_scores_by_layer: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Score each position by its mean normalized rank across layers.

    Preserves the input batch dimension; see
    :func:`_cross_layer_intersect_count` for the 1-D vs 2-D contract.
    """
    first = hub_scores_by_layer[next(iter(hub_scores_by_layer))]
    if first.dim() == 1:
        return _per_batch_mean_rank(hub_scores_by_layer)
    if first.dim() == 2:
        batch_size = first.size(0)
        per_batch = []
        for b in range(batch_size):
            sliced = {layer: t[b] for layer, t in hub_scores_by_layer.items()}
            per_batch.append(_per_batch_mean_rank(sliced))
        return torch.stack(per_batch, dim=0)
    raise ValueError(
        f"cross_mean_rank requires 1-D or 2-D tensors, got {first.dim()}-D"
    )


def _per_batch_group_intersect(
    per_layer_1d: dict[int, torch.Tensor],
    budget: float,
    group_boundaries: tuple[int, int],
) -> torch.Tensor:
    """1-D cross-layer group-intersect (single batch entry)."""
    layers = sorted(per_layer_1d.keys())
    first = per_layer_1d[layers[0]]
    seq_len = first.shape[-1]
    b0, b1 = group_boundaries
    groups: dict[str, list[int]] = {"early": [], "mid": [], "late": []}
    for layer in layers:
        if layer < b0:
            groups["early"].append(layer)
        elif layer < b1:
            groups["mid"].append(layer)
        else:
            groups["late"].append(layer)
    scores = torch.zeros(seq_len, device=first.device, dtype=first.dtype)
    for group_layers in groups.values():
        if not group_layers:
            continue
        group_mask = torch.zeros(seq_len, dtype=torch.bool, device=first.device)
        for layer in group_layers:
            k = resolve_budget(seq_len, budget)
            _, topk_idx = per_layer_1d[layer].topk(k)
            group_mask[topk_idx] = True
        scores += group_mask.to(scores.dtype)
    return scores


def _cross_layer_group_intersect(
    hub_scores_by_layer: dict[int, torch.Tensor],
    budget: float,
    group_boundaries: tuple[int, int] = (20, 40),
) -> torch.Tensor:
    """Score positions by presence across early/mid/late layer groups.

    Max score = 3 (present in all three groups). Preserves the input
    batch dimension; see :func:`_cross_layer_intersect_count`.
    """
    first = hub_scores_by_layer[next(iter(hub_scores_by_layer))]
    if first.dim() == 1:
        return _per_batch_group_intersect(
            hub_scores_by_layer, budget, group_boundaries,
        )
    if first.dim() == 2:
        batch_size = first.size(0)
        per_batch = []
        for b in range(batch_size):
            sliced = {layer: t[b] for layer, t in hub_scores_by_layer.items()}
            per_batch.append(
                _per_batch_group_intersect(sliced, budget, group_boundaries),
            )
        return torch.stack(per_batch, dim=0)
    raise ValueError(
        f"cross_group requires 1-D or 2-D tensors, got {first.dim()}-D"
    )


def resolve_budget(
    seq_len: int,
    budget: int | float,
    *,
    denominator: int | None = None,
) -> int:
    """Resolve a budget specification to an absolute token count.

    Different evaluation pipelines apply ``budget`` against different
    denominators on purpose:

    * **Total** (``denominator=None``, default) — ``budget`` is a fraction
      of the entire sequence. Used by intrinsic / degradation /
      ``select_top_tokens`` callers, where every position is in scope.
    * **Prefix** (``denominator=prefix_len`` where ``prefix_len = seq_len
      - protect_last``) — ``budget`` is a fraction of the *evictable*
      prefix only. Used by KV-cache eviction so a fixed budget label
      means the same eviction aggressiveness regardless of how big the
      protected recency window is.
    * **Compressible** (``denominator=cutoff``) — ``budget`` is a fraction
      of the compressible prefix that precedes a fixed evaluation
      runway. Used by the token-level NTP pipeline so budget labels
      stay invariant in the runway length.

    All three call sites share this helper to keep the int → round
    convention and the ``[1, seq_len]`` clamp consistent. Each call site
    states the semantic explicitly via ``denominator``.

    Args:
        seq_len: Total number of tokens in the sequence. Used as the
            upper clamp on the returned count.
        budget: If a float in ``(0, 1]``, treated as a retention ratio
            against ``denominator`` (or ``seq_len`` when ``denominator``
            is ``None``). If an int ``>= 1``, treated as an absolute
            count. ``1.0`` is treated as the ratio "keep 100%".
        denominator: Optional explicit denominator for the ratio.
            Defaults to ``seq_len``. Must be ``>= 1`` when provided.

    Returns:
        The number of tokens to retain, clamped to ``[1, seq_len]``.
    """
    if denominator is not None and denominator < 1:
        raise ValueError(
            f"denominator must be >= 1 when provided, got {denominator!r}"
        )
    # ``isinstance(True, int)`` is True in Python — booleans inherit
    # from int — so we have to reject them explicitly to match the
    # contract that ``baselines._validate_budget`` already enforces.
    # Otherwise ``resolve_budget(seq_len, True)`` silently retains 1
    # token, which is almost never what the caller meant.
    if isinstance(budget, bool):
        raise ValueError(
            f"budget must be a float in (0,1] or an int >= 1, "
            f"got bool ({budget!r})"
        )
    base = seq_len if denominator is None else denominator
    if isinstance(budget, float) and 0 < budget <= 1:
        k = max(1, round(base * budget))
    elif isinstance(budget, int) and budget >= 1:
        k = budget
    else:
        raise ValueError(
            f"budget must be a float in (0,1] or an int >= 1, got {budget!r}"
        )
    return min(k, seq_len)


def select_top_tokens(
    hub_scores: torch.Tensor,
    budget: int | float,
) -> torch.Tensor:
    """Select top-scoring token positions under a retention budget.

    Args:
        hub_scores: Shape ``(batch, seq_len)`` — per-token hub scores.
        budget: Retention budget — see ``resolve_budget`` for semantics.

    Returns:
        Sorted token indices of shape ``(batch, k)`` where *k* is the
        resolved budget.
    """
    if hub_scores.dim() != 2:
        raise ValueError(
            f"hub_scores must be 2-D (batch, seq_len), got {hub_scores.dim()}-D"
        )
    # Mirror the fail-fast contract that
    # ``baselines.selection._topk_indices`` enforces: refuse to pick a
    # selection from a row that has no positive mass at all. A flat
    # zero / negative row would still produce ``k`` indices via
    # implementation-defined topk tie-breaking, masking an upstream
    # scoring bug under what looks like a successful selection.
    row_max = hub_scores.max(dim=-1).values
    if not torch.all(row_max > 0):
        n_bad = int((row_max <= 0).sum().item())
        raise ValueError(
            f"select_top_tokens: {n_bad} of {hub_scores.size(0)} batch row(s) "
            "have no positive hub mass (max <= 0). The canonical hub "
            "aggregators emit non-negative scores with at least one "
            "positive entry per non-degenerate input; an all-zero / "
            "non-positive row indicates an upstream scoring bug."
        )
    k = resolve_budget(hub_scores.size(1), budget)

    _, top_indices = hub_scores.topk(k, dim=-1)
    return top_indices.sort(dim=-1).values
