"""Selection tap and cascade: selection as a second consumer of attention's scores.

The lightning indexer scores keys with its own projections:
``logit[m, n] = sum_h ReLU(q^I_h . k^I_n) * w[m, h]``.  The *tap* replaces those
projections with attention's own pre-softmax scores ``S_h[m, n]`` — the bilinear
the attention layer has already computed —

    ``tap[m, n] = sum_h w_h * ReLU(S_h[m, n])``

so selection becomes a second consumer of the QK^T primitive instead of a
separate computation.  The one-layer lag is what makes the reuse legal: layer
``L``'s attention scores seed layer ``L+1``'s index set.

Sparse layers only materialize scores at the keys they attend, so the tap alone
cannot admit new keys.  The *cascade* closes that: the next layer's index set is
the top-k of the tap logits over

    incumbent set (this layer's attended keys)  ∪  admission candidates,

where the admission candidates are cheap-to-obtain keys the incumbent set cannot
contain yet — a recency window of the newest positions plus (optionally) the
leading sink positions.  Steady state then needs no full-prefix indexer scan.

Everything here is pure tensor logic (CPU-runnable); score capture lives in
``inference.modeling.transformer`` and the measurement harness in ``bench/``.
"""

from __future__ import annotations

import torch


def tap_logits(
    raw_scores: torch.Tensor, head_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Reduce per-head pre-softmax attention scores to per-key selection logits.

    Args:
        raw_scores: ``[B, S, H, T]`` pre-softmax attention scores (before any
            causal or index masking; ``-inf`` entries would poison the ReLU sum).
        head_weights: Optional ``[H]`` per-head weights. ``None`` averages the
            heads uniformly (weight ``1/H``).

    Returns:
        ``[B, S, T]`` float32 tap logits ``sum_h w_h * ReLU(S_h)``.
    """
    if raw_scores.dim() != 4:
        raise ValueError(f"raw_scores must be [B, S, H, T], got {tuple(raw_scores.shape)}")
    n_heads = raw_scores.size(2)
    scores = raw_scores.float().relu()
    if head_weights is None:
        return scores.mean(dim=2)
    if head_weights.shape != (n_heads,):
        raise ValueError(
            f"head_weights must be [{n_heads}] to match raw_scores heads, "
            f"got {tuple(head_weights.shape)}"
        )
    return torch.einsum("bsht,h->bst", scores, head_weights.float())


def recency_candidates(
    query_positions: torch.Tensor, window: int, n_sink: int = 0
) -> torch.Tensor:
    """Admission candidates for each query: trailing window plus leading sinks.

    Args:
        query_positions: ``[S]`` absolute key-space position of each query row
            (a query at position ``p`` may attend keys ``0..p``).
        window: Number of most-recent positions ``p, p-1, ..., p-window+1``.
        n_sink: Leading absolute positions ``0..n_sink-1`` always eligible
            (attention-sink keys accumulate mass regardless of content).

    Returns:
        ``[S, n_sink + window]`` int64 candidate indices; slots that would be
        negative or acausal (``> p``) hold the ``-1`` sentinel.
    """
    if query_positions.dim() != 1:
        raise ValueError(f"query_positions must be [S], got {tuple(query_positions.shape)}")
    if window < 0 or n_sink < 0:
        raise ValueError(f"window and n_sink must be >= 0, got {window}, {n_sink}")
    pos = query_positions.long().unsqueeze(1)  # [S, 1]
    device = query_positions.device
    parts: list[torch.Tensor] = []
    if n_sink > 0:
        sinks = torch.arange(n_sink, device=device).unsqueeze(0).expand(pos.size(0), -1)
        parts.append(torch.where(sinks <= pos, sinks, torch.full_like(sinks, -1)))
    if window > 0:
        offsets = torch.arange(window, device=device).unsqueeze(0)  # 0 = newest
        recent = pos - offsets
        parts.append(torch.where(recent >= 0, recent, torch.full_like(recent, -1)))
    if not parts:
        return torch.empty(pos.size(0), 0, dtype=torch.int64, device=device)
    return torch.cat(parts, dim=1)


def stripe_candidates(
    query_positions: torch.Tensor, count: int, exclude_window: int = 0
) -> torch.Tensor:
    """Phase-rotating strided admission candidates over the pre-window prefix.

    The recency window gives every key exactly one scoring chance as it ages
    past the query; a key older than the window can re-enter the cascade only
    while it survives in the threaded pool.  The stripe restores a scoring
    chance to the old region ``[0, p - exclude_window]``: ``count`` equally
    spaced positions whose phase is a multiplicative-hash function of the
    query position (a low-discrepancy sweep of ``[0, stride)`` — a plain
    ``p % stride`` phase barely moves when the stride itself grows with the
    prefix), so consecutive positions keep sampling fresh old-region keys at
    ~``count`` per step.  Cost-wise this is budget-matched to widening the
    window by ``count`` slots — the A/B against that control isolates
    candidate *placement* from candidate *budget*.

    Args:
        query_positions: ``[S]`` absolute query positions.
        count: Stripe slot count per query.
        exclude_window: Newest positions already covered by the recency
            window; the stripe only samples strictly older keys.

    Returns:
        ``[S, count]`` int64 candidate indices; ``-1`` sentinels where the
        old region is shorter than the stripe.
    """
    if query_positions.dim() != 1:
        raise ValueError(
            f"query_positions must be [S], got {tuple(query_positions.shape)}"
        )
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if exclude_window < 0:
        raise ValueError(f"exclude_window must be >= 0, got {exclude_window}")
    pos = query_positions.long().unsqueeze(1)  # [S, 1]
    span_end = pos - exclude_window  # newest strippable position, inclusive
    span = (span_end + 1).clamp(min=0)
    stride = torch.div(span + count - 1, count, rounding_mode="floor").clamp(min=1)
    # Knuth multiplicative hash of the position, scaled to [0, stride):
    # low-discrepancy in p, independent of the stride's own drift.
    phase = ((pos * 2654435761) % 2**32) * stride // 2**32
    offsets = torch.arange(count, device=query_positions.device).unsqueeze(0)
    cand = phase + offsets * stride
    return torch.where(
        (cand <= span_end) & (span > 0), cand, torch.full_like(cand, -1)
    )


def neighbor_candidates(
    incumbent: torch.Tensor,
    query_positions: torch.Tensor,
    delta: int,
    budget: int,
) -> torch.Tensor:
    """Positional neighbors of the incumbent set, subsampled to a budget.

    Coverage lever (g): the incumbent set marks where attention currently
    looks; keys at ``incumbent +/- 1..delta`` are its positional neighbors —
    likely-relevant keys the window cannot reach and the stripe only samples
    uniformly.  The full neighbor multiset has ``k * 2*delta`` slots; a
    low-discrepancy strided subsample (same Knuth-hash phase rotation as
    :func:`stripe_candidates`) reduces it to ``budget`` candidates per query,
    so consecutive positions sweep different (slot, offset) pairs and the
    whole neighborhood gets scored over ``ceil(k*2*delta / budget)`` steps.
    Content-adaptive by construction — the anchor set is whatever selection
    currently holds — at a flat O(budget) admission cost and no state.

    Args:
        incumbent: ``[B, S, k]`` int64 incumbent (or pool) key indices;
            ``-1`` sentinel slots yield sentinel neighbors.
        query_positions: ``[S]`` absolute query positions.
        delta: Maximum neighbor distance (offsets ``-delta..-1, +1..+delta``).
        budget: Candidate slots per query row.

    Returns:
        ``[B, S, budget]`` int64 candidate indices; ``-1`` sentinels where
        the anchor slot is empty, the offset walks off the prefix, or the
        neighbor would be acausal.
    """
    if incumbent.dim() != 3:
        raise ValueError(f"incumbent must be [B, S, k], got {tuple(incumbent.shape)}")
    if query_positions.dim() != 1 or query_positions.size(0) != incumbent.size(1):
        raise ValueError(
            f"query_positions must be [{incumbent.size(1)}], "
            f"got {tuple(query_positions.shape)}"
        )
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    bsz, seqlen, k = incumbent.shape
    device = incumbent.device
    two_delta = 2 * delta
    virtual = k * two_delta  # full neighbor-multiset size per query
    pos = query_positions.long().unsqueeze(1)  # [S, 1]
    stride = max((virtual + budget - 1) // budget, 1)
    # Knuth multiplicative hash of the position scaled to [0, stride) —
    # the stripe's low-discrepancy phase convention.
    phase = ((pos * 2654435761) % 2**32) * stride // 2**32  # [S, 1]
    v = phase + torch.arange(budget, device=device).unsqueeze(0) * stride  # [S, budget]
    in_range = v < virtual
    v = v.clamp(max=max(virtual - 1, 0))
    slot = v // two_delta  # which incumbent column anchors this candidate
    o = v % two_delta
    # Offset indices 0..2*delta-1 map to -delta..-1, +1..+delta (0 skipped:
    # the anchor itself is already a candidate as the incumbent copy).
    offset = torch.where(o < delta, o - delta, o - delta + 1)
    base = incumbent.long().gather(2, slot.unsqueeze(0).expand(bsz, -1, -1))
    cand = base + offset.unsqueeze(0)
    invalid = (
        (base < 0)
        | ~in_range.unsqueeze(0)
        | (cand < 0)
        | (cand > pos.unsqueeze(0))
    )
    return torch.where(invalid, torch.full_like(cand, -1), cand)


def block_summary_scores(
    raw_scores: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
    stat: str = "mean",
) -> torch.Tensor:
    """Per-head, per-block causal summary scores for coarse-to-fine admission.

    Coverage lever (e): reduce the key axis to ``ceil(T / block_size)``
    blocks so ALL of the prefix can be scored every layer at O(T/block_size).

    - ``"mean"``: per-head mean of the block's causal key scores.  Because
      attention scores are bilinear in the cached key, this is EXACTLY the
      score of a single summary key equal to the block's mean cached key
      (mean MLA latent ``(c, pe)`` rows) — the kernel-realizable summary a
      serving path would score with the same MQA primitive.  The eager
      emulation here reduces the already-materialized score row instead.
    - ``"amax"``: per-head max over the block's causal keys — the score-space
      upper envelope.  No single summary key realizes it; diagnostic /
      eager-only by nature.

    Args:
        raw_scores: ``[B, S, H, T]`` signed pre-softmax, PRE-mask scores.
        query_positions: ``[S]`` absolute query positions (keys ``> p`` are
            excluded from every summary — the row beyond the causal frontier
            holds garbage scores).
        block_size: Keys per block (> 0).
        stat: ``"mean"`` or ``"amax"``.

    Returns:
        ``[B, S, H, ceil(T / block_size)]`` float32 block scores; blocks with
        no causal key hold ``-inf``.
    """
    if raw_scores.dim() != 4:
        raise ValueError(f"raw_scores must be [B, S, H, T], got {tuple(raw_scores.shape)}")
    if query_positions.dim() != 1 or query_positions.size(0) != raw_scores.size(1):
        raise ValueError(
            f"query_positions must be [{raw_scores.size(1)}], "
            f"got {tuple(query_positions.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if stat not in ("mean", "amax"):
        raise ValueError(f"stat must be 'mean' or 'amax', got {stat!r}")
    bsz, seqlen, n_heads, n_keys = raw_scores.shape
    n_blocks = (n_keys + block_size - 1) // block_size
    pad = n_blocks * block_size - n_keys
    device = raw_scores.device
    key_idx = torch.arange(n_keys, device=device)
    acausal = key_idx.unsqueeze(0) > query_positions.long().unsqueeze(1)  # [S, T]
    s = raw_scores.float()
    if stat == "mean":
        s = s.masked_fill(acausal.view(1, seqlen, 1, n_keys), 0.0)
        if pad:
            s = torch.nn.functional.pad(s, (0, pad))
        sums = s.view(bsz, seqlen, n_heads, n_blocks, block_size).sum(dim=-1)
        starts = torch.arange(n_blocks, device=device) * block_size  # [nb]
        # Causal keys in block b for query p: min(p+1 - start, block_size,
        # keys physically present in the tail block), floored at 0.
        counts = (
            (query_positions.long().unsqueeze(1) + 1 - starts.unsqueeze(0))
            .clamp(min=0, max=block_size)
        )
        present = (n_keys - starts).clamp(max=block_size)
        counts = torch.minimum(counts, present.unsqueeze(0))  # [S, nb]
        out = sums / counts.clamp(min=1).view(1, seqlen, 1, n_blocks)
        return torch.where(
            counts.view(1, seqlen, 1, n_blocks) > 0,
            out,
            torch.full_like(out, float("-inf")),
        )
    s = s.masked_fill(acausal.view(1, seqlen, 1, n_keys), float("-inf"))
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    return s.view(bsz, seqlen, n_heads, n_blocks, block_size).amax(dim=-1)


def block_token_candidates(
    block_scores: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
    n_admit: int,
    total_len: int,
) -> torch.Tensor:
    """Expand the top-scoring blocks into token-level admission candidates.

    The fine stage of coarse-to-fine admission: rank blocks by their reduced
    tap score, take the top ``n_admit`` causal blocks per query, and admit
    every token of those blocks for exact scoring by the cascade re-rank.

    Args:
        block_scores: ``[B, S, n_blocks]`` reduced (post-tap) block logits.
        query_positions: ``[S]`` absolute query positions.
        block_size: Keys per block (must match the summary construction).
        n_admit: Blocks to admit per query (> 0).
        total_len: Key-axis length ``T`` (tail-block tokens ``>= total_len``
            are sentinels).

    Returns:
        ``[B, S, n_admit * block_size]`` int64 candidate indices with ``-1``
        sentinels (empty blocks, acausal tokens, tail overhang).
    """
    if block_scores.dim() != 3:
        raise ValueError(
            f"block_scores must be [B, S, n_blocks], got {tuple(block_scores.shape)}"
        )
    if query_positions.dim() != 1 or query_positions.size(0) != block_scores.size(1):
        raise ValueError(
            f"query_positions must be [{block_scores.size(1)}], "
            f"got {tuple(query_positions.shape)}"
        )
    if block_size <= 0 or n_admit <= 0:
        raise ValueError(
            f"block_size and n_admit must be positive, got {block_size}, {n_admit}"
        )
    bsz, seqlen, n_blocks = block_scores.shape
    device = block_scores.device
    pos = query_positions.long()
    starts = torch.arange(n_blocks, device=device) * block_size
    # A block strictly beyond the causal frontier must never be admitted,
    # whatever its (garbage or ReLU-zeroed) score.
    acausal_block = starts.view(1, 1, -1) > pos.view(1, -1, 1)
    scores = block_scores.float().masked_fill(acausal_block, float("-inf"))
    n_sel = min(n_admit, n_blocks)
    top_scores, top_blk = scores.topk(n_sel, dim=-1)  # [B, S, n_sel]
    top_blk = torch.where(
        torch.isfinite(top_scores), top_blk, torch.full_like(top_blk, -1)
    )
    offsets = torch.arange(block_size, device=device)
    toks = top_blk.unsqueeze(-1) * block_size + offsets  # [B, S, n_sel, bs]
    invalid = (
        (top_blk < 0).unsqueeze(-1)
        | (toks >= total_len)
        | (toks > pos.view(1, -1, 1, 1))
    )
    toks = torch.where(invalid, torch.full_like(toks, -1), toks)
    toks = toks.view(bsz, seqlen, n_sel * block_size)
    if n_sel < n_admit:
        pad = torch.full(
            (bsz, seqlen, (n_admit - n_sel) * block_size), -1,
            dtype=toks.dtype, device=device,
        )
        toks = torch.cat([toks, pad], dim=-1)
    return toks


def incumbent_block_mask_and_distance(
    incumbent: torch.Tensor,
    block_size: int,
    n_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block incumbent coverage mask and distance to the nearest covered block.

    Entries-head features (docs/prereg_entries_head.md): a candidate block's
    distance (in blocks) to the nearest block containing at least one
    incumbent key, plus the coverage mask itself (covered blocks are
    excluded from the head's ranking universe — the training semantics).

    Args:
        incumbent: ``[B, S, k]`` incumbent key indices (``-1`` sentinels).
        block_size: Keys per block.
        n_blocks: Number of blocks on the key axis.

    Returns:
        ``(covered [B, S, nb] bool, distance [B, S, nb] float32)``; rows with
        no valid incumbents get distance ``n_blocks`` everywhere.
    """
    if incumbent.dim() != 3:
        raise ValueError(f"incumbent must be [B, S, k], got {tuple(incumbent.shape)}")
    if block_size <= 0 or n_blocks <= 0:
        raise ValueError(
            f"block_size and n_blocks must be positive, got {block_size}, {n_blocks}"
        )
    bsz, seqlen, _ = incumbent.shape
    device = incumbent.device
    blocks = torch.div(
        incumbent.long().clamp(min=0), block_size, rounding_mode="floor"
    )
    valid = incumbent >= 0
    # Route sentinel slots to a dummy column so a False can never overwrite
    # a True at block 0 (scatter is last-write-wins).
    target = torch.where(
        valid, blocks.clamp(max=n_blocks - 1),
        torch.full_like(blocks, n_blocks),
    )
    covered_ext = torch.zeros(
        bsz, seqlen, n_blocks + 1, dtype=torch.bool, device=device
    )
    covered_ext.scatter_(2, target, torch.ones_like(target, dtype=torch.bool))
    covered = covered_ext[..., :n_blocks]
    idx = torch.arange(n_blocks, device=device).view(1, 1, -1).expand_as(covered)
    # Forward pass: index of the last covered block at or before j.
    last_before = torch.cummax(
        torch.where(covered, idx, torch.full_like(idx, -1)), dim=2
    ).values
    dist_fwd = torch.where(
        last_before >= 0,
        (idx - last_before).float(),
        torch.full_like(idx, n_blocks, dtype=torch.float32),
    )
    # Backward pass on the flipped axis: nearest covered block at or after j.
    rev = torch.flip(covered, dims=[2])
    last_rev = torch.cummax(
        torch.where(rev, idx, torch.full_like(idx, -1)), dim=2
    ).values
    dist_bwd = torch.flip(
        torch.where(
            last_rev >= 0,
            (idx - last_rev).float(),
            torch.full_like(idx, n_blocks, dtype=torch.float32),
        ),
        dims=[2],
    )
    return covered, torch.minimum(dist_fwd, dist_bwd)


def entries_head_features(
    block_tap: torch.Tensor,
    prev_block_tap: torch.Tensor,
    incumbent: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the entries-head feature tensor (docs/prereg_entries_head.md).

    Feature order is FIXED to the training layout: 0 this layer's block mean
    summary, 1 ``log1p(recency in blocks)``, 2 ``log1p(distance to nearest
    incumbent-covered block)``, 3 previous layer's block mean summary.

    Args:
        block_tap: ``[B, S, nb]`` this layer's reduced block summaries.
        prev_block_tap: ``[B, S, nb]`` previous layer's summaries (threaded).
        incumbent: ``[B, S, k]`` incumbent key indices.
        query_positions: ``[S]`` absolute query positions.
        block_size: Keys per block.

    Returns:
        ``(features [B, S, nb, 4] float32, covered [B, S, nb] bool)`` where
        ``covered`` marks incumbent-covered blocks (to exclude from ranking).
    """
    if block_tap.dim() != 3:
        raise ValueError(f"block_tap must be [B, S, nb], got {tuple(block_tap.shape)}")
    if prev_block_tap.shape != block_tap.shape:
        raise ValueError(
            f"prev_block_tap {tuple(prev_block_tap.shape)} must match "
            f"block_tap {tuple(block_tap.shape)}; the threaded summary must "
            "come from the SAME forward chunk"
        )
    if query_positions.dim() != 1 or query_positions.size(0) != block_tap.size(1):
        raise ValueError(
            f"query_positions must be [{block_tap.size(1)}], "
            f"got {tuple(query_positions.shape)}"
        )
    n_blocks = block_tap.size(2)
    covered, dist = incumbent_block_mask_and_distance(
        incumbent, block_size, n_blocks
    )
    max_blk = torch.div(query_positions.long(), block_size, rounding_mode="floor")
    idx = torch.arange(n_blocks, device=block_tap.device)
    recency = (max_blk.view(1, -1, 1) - idx.view(1, 1, -1)).clamp(min=0).float()
    features = torch.stack([
        block_tap.float(),
        torch.log1p(recency).expand_as(block_tap),
        torch.log1p(dist),
        prev_block_tap.float(),
    ], dim=-1)
    return features, covered


def entries_head_scores(
    features: torch.Tensor, head: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply the tiny entries-head MLP to a ``[..., 4]`` feature tensor.

    ``head`` holds ``mu``/``sd`` ``[4]`` standardization constants and the
    MLP parameters ``w1 [4, H]``, ``b1 [H]``, ``w2 [H]``, ``b2 []``.  Used
    identically at training-export time and inside the cascade, so the two
    can never diverge.  Returns ``[...]`` float32 logits.
    """
    required = ("mu", "sd", "w1", "b1", "w2", "b2")
    missing = [k for k in required if k not in head]
    if missing:
        raise ValueError(f"entries head payload missing keys {missing}")
    n_feat = head["w1"].size(0)
    if features.size(-1) != n_feat:
        raise ValueError(
            f"features last dim {features.size(-1)} != head n_feat {n_feat}"
        )
    x = (features.float() - head["mu"].float()) / head["sd"].float().clamp(min=1e-6)
    hidden = torch.relu(x @ head["w1"].float() + head["b1"].float())
    return hidden @ head["w2"].float() + head["b2"].float()


def cascade_topk(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    k: int,
    query_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k key selection over a candidate pool with duplicates and sentinels.

    The pool is the concatenation of the incumbent set and the admission
    candidates, so the same key may appear more than once (identical score both
    times — the score is a function of the key alone); duplicates are collapsed
    to a single entry before selection.  ``-1`` sentinel candidates are ignored.

    Args:
        candidate_indices: ``[B, S, C]`` int64 key indices (``-1`` = empty slot).
        candidate_scores: ``[B, S, C]`` tap logits aligned with the indices.
        k: Number of keys to select per query.
        query_positions: Optional ``[S]`` absolute query positions; when given,
            candidates with ``index > position`` are dropped (causality guard).

    Returns:
        ``[B, S, k]`` int64 selected indices, ``-1``-padded where fewer than
        ``k`` distinct valid candidates exist.
    """
    if candidate_indices.shape != candidate_scores.shape:
        raise ValueError(
            f"indices {tuple(candidate_indices.shape)} and scores "
            f"{tuple(candidate_scores.shape)} must have identical shapes"
        )
    if candidate_indices.dim() != 3:
        raise ValueError(
            f"candidate_indices must be [B, S, C], got {tuple(candidate_indices.shape)}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    indices = candidate_indices.long()
    if query_positions is not None:
        if query_positions.shape != (indices.size(1),):
            raise ValueError(
                f"query_positions must be [{indices.size(1)}], "
                f"got {tuple(query_positions.shape)}"
            )
        acausal = indices > query_positions.long().view(1, -1, 1)
        indices = torch.where(acausal, torch.full_like(indices, -1), indices)

    # Sort by key index so duplicates become adjacent (sentinels sort first),
    # then keep one entry per distinct key.  Duplicate entries carry identical
    # scores, so which one survives is immaterial.
    order = indices.argsort(dim=-1, stable=True)
    idx_sorted = indices.gather(-1, order)
    score_sorted = candidate_scores.float().gather(-1, order)
    duplicate = torch.zeros_like(idx_sorted, dtype=torch.bool)
    duplicate[..., 1:] = idx_sorted[..., 1:] == idx_sorted[..., :-1]
    dropped = duplicate | (idx_sorted < 0)
    score_sorted = score_sorted.masked_fill(dropped, float("-inf"))

    n_select = min(k, idx_sorted.size(-1))
    top_scores, top_pos = score_sorted.topk(n_select, dim=-1)
    selected = idx_sorted.gather(-1, top_pos)
    selected = torch.where(
        torch.isfinite(top_scores), selected, torch.full_like(selected, -1)
    )
    if n_select < k:
        pad = torch.full(
            (*selected.shape[:-1], k - n_select), -1,
            dtype=selected.dtype, device=selected.device,
        )
        selected = torch.cat([selected, pad], dim=-1)
    return selected


def mass_capture(probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Fraction of attention mass an index set captures, per query.

    Args:
        probs: ``[B, S, T]`` attention probabilities (post-softmax, e.g. the
            dense reference distribution), non-negative.
        indices: ``[B, S, k]`` selected key indices (``-1`` = empty slot).

    Returns:
        ``[B, S]`` captured-mass fraction in ``[0, 1]``; rows with zero total
        mass are defined as fully captured.
    """
    if probs.dim() != 3 or indices.dim() != 3:
        raise ValueError(
            f"probs must be [B, S, T] and indices [B, S, k], got "
            f"{tuple(probs.shape)} and {tuple(indices.shape)}"
        )
    valid = indices >= 0
    safe = indices.clamp(min=0)
    captured = probs.gather(-1, safe).masked_fill(~valid, 0.0).sum(dim=-1)
    total = probs.sum(dim=-1)
    return torch.where(total > 0, captured / total, torch.ones_like(total))


# ---------------------------------------------------------------------------
# Scoring-function variants (design-space axis V, docs/design_space_prereg.md)
# ---------------------------------------------------------------------------

# Every variant reduces the same [B, S, H, T] signed post-scale scores to
# [B, S, T] selection logits.  Under tensor parallelism each rank reduces its
# local head slice; the ``combine`` tag says how partial results merge across
# ranks ("sum" / "max" / "lse" — the caller owns the collective).
VARIANT_NAMES: tuple[str, ...] = (
    "v0_relu_sum",
    "v1_signed_sum",
    "v2_max",
    "v3_lse",
    "v4_probs",
    "v5_probs_vnorm",
    "v6_zscore",
    # v4 adjacents (verification campaign, prereg section V-verification):
    "v4a_exp_nonorm",   # exp(S - m), no Z — is normalization load-bearing?
    "v4b_temp",         # tempered softmax: v4 on S/tau with stats at S/tau
    "v4c_selfnorm",     # per-head softmax over the scored row itself
    "v4d_zexp",         # per-head z-score, then exp — calibration + shape
)


def attend_softmax_stats(
    raw_scores: torch.Tensor, attend_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head softmax statistics over the attend slice (kernel semantics).

    The fused kernel only ever softmaxes the keys it attends, so the
    statistics available to an in-kernel probability tap are the per-head
    max ``m`` and denominator ``Z`` over the attend set — computed here and
    applied to *every* candidate by the probability variants.

    Args:
        raw_scores: ``[B, S, H, T]`` signed post-scale scores.
        attend_idx: ``[B, S, k]`` attended key indices (``-1`` = empty slot).

    Returns:
        ``(m, Z)`` each ``[B, S, H]`` float32: row-max and ``sum exp(S - m)``
        over the valid attend slots.
    """
    if raw_scores.dim() != 4 or attend_idx.dim() != 3:
        raise ValueError(
            f"raw_scores must be [B, S, H, T] and attend_idx [B, S, k], got "
            f"{tuple(raw_scores.shape)} and {tuple(attend_idx.shape)}"
        )
    valid = attend_idx >= 0  # [B, S, k]
    safe = attend_idx.clamp(min=0).unsqueeze(2).expand(
        -1, -1, raw_scores.size(2), -1
    )  # [B, S, H, k]
    sel = raw_scores.float().gather(-1, safe)
    sel = sel.masked_fill(~valid.unsqueeze(2), float("-inf"))
    m = sel.amax(dim=-1)  # [B, S, H]
    z = (sel - m.unsqueeze(-1)).exp().sum(dim=-1)
    return m, z


def variant_local_scores(
    raw_scores: torch.Tensor,
    variant: str,
    head_weights: torch.Tensor | None = None,
    tau: float = 1.0,
    attend_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_norms: torch.Tensor | None = None,
    causal_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Rank-local partial reduction of one scoring variant.

    Args:
        raw_scores: ``[B, S, H, T]`` signed post-scale scores (this rank's
            head slice under TP).
        variant: One of ``VARIANT_NAMES``.
        head_weights: Optional ``[H]`` weights for this rank's heads
            (``None`` = uniform ``1/H_global`` handled by the caller's
            normalization; here ``None`` means unit weights).
        tau: LSE temperature (``v3_lse`` only).
        attend_stats: ``(m, Z)`` from :func:`attend_softmax_stats`
            (``v4_probs`` / ``v5_probs_vnorm`` only).
        value_norms: ``[H, T]`` per-head value norms of every cached key
            (``v5_probs_vnorm`` only; query-independent).
        causal_mask: ``[B, S, 1, T]`` bool, True at ACAUSAL slots
            (``v6_zscore`` only — its per-head statistics must exclude keys
            the query cannot attend).

    Returns:
        ``([B, S, T] float32 partial, combine)`` where ``combine`` in
        ``{"sum", "max", "lse"}`` names the cross-rank reduction that
        completes the variant.
    """
    if raw_scores.dim() != 4:
        raise ValueError(
            f"raw_scores must be [B, S, H, T], got {tuple(raw_scores.shape)}"
        )
    n_heads = raw_scores.size(2)
    # Promote to at least float32; float64 inputs keep their precision (the
    # LSE limit tau*log(H) + mean(x)/H needs the headroom in tests).
    s = raw_scores if raw_scores.dtype == torch.float64 else raw_scores.float()
    if head_weights is not None and head_weights.shape != (n_heads,):
        raise ValueError(
            f"head_weights must be [{n_heads}] to match raw_scores heads, "
            f"got {tuple(head_weights.shape)}"
        )
    w = None if head_weights is None else head_weights.float()

    def _weighted(x: torch.Tensor) -> torch.Tensor:
        return x if w is None else x * w.view(1, 1, -1, 1)

    if variant == "v0_relu_sum":
        return _weighted(s.relu()).sum(dim=2), "sum"
    if variant == "v1_signed_sum":
        return _weighted(s).sum(dim=2), "sum"
    if variant == "v2_max":
        return _weighted(s.relu()).amax(dim=2), "max"
    if variant == "v3_lse":
        if tau <= 0:
            raise ValueError(f"v3_lse requires tau > 0, got {tau}")
        return tau * torch.logsumexp(_weighted(s.relu()) / tau, dim=2), "lse"
    if variant == "v4b_temp":
        # Tempered attend-slice softmax == v4 applied to S/tau with stats
        # computed on S/tau; the CALLER owns both rescalings (this keeps
        # the (scores, stats) pairing honest at every temperature).
        if attend_stats is None:
            raise ValueError("v4b_temp requires attend_stats computed on S/tau")
        m, z = attend_stats
        probs = (s - m.unsqueeze(-1)).exp() / z.clamp(min=1e-20).unsqueeze(-1)
        return _weighted(probs).sum(dim=2), "sum"
    if variant == "v4a_exp_nonorm":
        if attend_stats is None:
            raise ValueError("v4a_exp_nonorm requires attend_stats (m, _)")
        m, _ = attend_stats
        return _weighted((s - m.unsqueeze(-1)).exp()).sum(dim=2), "sum"
    if variant == "v4c_selfnorm":
        if causal_mask is None:
            raise ValueError("v4c_selfnorm requires causal_mask [B, S, 1, T]")
        probs = s.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
        return _weighted(probs).sum(dim=2), "sum"
    if variant == "v4d_zexp":
        if causal_mask is None:
            raise ValueError("v4d_zexp requires causal_mask [B, S, 1, T]")
        masked = s.masked_fill(causal_mask, 0.0)
        n = (~causal_mask).float().sum(dim=-1, keepdim=True)
        mean = masked.sum(dim=-1, keepdim=True) / n.clamp(min=1)
        var = (
            (masked - mean).masked_fill(causal_mask, 0.0).pow(2)
            .sum(dim=-1, keepdim=True) / n.clamp(min=1)
        )
        zed = (s - mean) / var.sqrt().clamp(min=1e-6)
        return _weighted(zed.exp().masked_fill(causal_mask, 0.0)).sum(dim=2), "sum"
    if variant in ("v4_probs", "v5_probs_vnorm"):
        if attend_stats is None:
            raise ValueError(f"{variant} requires attend_stats (m, Z)")
        m, z = attend_stats
        # exp(S - m) / Z with the ATTEND slice's statistics applied to all
        # keys: candidates hotter than anything attended exceed 1 by design.
        probs = (s - m.unsqueeze(-1)).exp() / z.clamp(min=1e-20).unsqueeze(-1)
        if variant == "v5_probs_vnorm":
            if value_norms is None:
                raise ValueError("v5_probs_vnorm requires value_norms [H, T]")
            if value_norms.shape != (n_heads, raw_scores.size(-1)):
                raise ValueError(
                    f"value_norms must be [{n_heads}, {raw_scores.size(-1)}], "
                    f"got {tuple(value_norms.shape)}"
                )
            probs = probs * value_norms.float().view(1, 1, n_heads, -1)
        return _weighted(probs).sum(dim=2), "sum"
    if variant == "v6_zscore":
        if causal_mask is None:
            raise ValueError("v6_zscore requires causal_mask [B, S, 1, T]")
        masked = s.masked_fill(causal_mask, 0.0)
        n = (~causal_mask).float().sum(dim=-1, keepdim=True)  # [B, S, 1, 1]
        mean = masked.sum(dim=-1, keepdim=True) / n.clamp(min=1)
        var = (
            (masked - mean).masked_fill(causal_mask, 0.0).pow(2).sum(dim=-1, keepdim=True)
            / n.clamp(min=1)
        )
        z = (s - mean) / var.sqrt().clamp(min=1e-6)
        return _weighted(z.masked_fill(causal_mask, 0.0)).sum(dim=2), "sum"
    raise ValueError(f"unknown scoring variant {variant!r}; expected one of {VARIANT_NAMES}")
