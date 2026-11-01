"""Distillation losses and selection-overlap metric for the lagged indexer.

Two interchangeable objectives train the producing (layer ``L-1``) indexer to
reproduce the consuming layer ``L``'s selection (the same-layer choice the
one-layer lag would ideally reuse):

* :func:`cross_layer_kl` -- match the full score *distribution* over keys
  (soft-label KL). Causal positions are masked with a large *finite* negative
  constant rather than ``-inf`` so softmax mass on disallowed keys is
  effectively zero without introducing NaNs in the backward pass.
* :func:`cross_layer_rank` -- match the teacher's *ordering* of keys with a
  pairwise :class:`~torch.nn.MarginRankingLoss`-style hinge over sampled causal
  key pairs. Optimises rank agreement rather than the exact distribution, which
  may generalise / transfer better across domains.
* :func:`cross_layer_list_mle` -- listwise Plackett-Luce MLE over the teacher's
  top-``C`` candidates per query row. Concentrates the whole gradient budget on
  the keys that top-k selection actually depends on (unlike uniform pairwise
  sampling), needs no margin, and retains a natural score scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG = -1.0e9


def causal_mask(query_pos: torch.Tensor, seq_k: int) -> torch.Tensor:
    """Boolean ``(Q, Sk)`` mask, True where key position <= query position."""
    key_pos = torch.arange(seq_k, device=query_pos.device)
    return key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)


def _apply_causal(scores: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    """Replace non-causal entries with a finite large-negative value."""
    return torch.where(causal, scores, scores.new_full((), _NEG))


def cross_layer_kl(
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(softmax(teacher) || softmax(student)) over query rows.

    Args:
        teacher_score: ``(b, Q, Sk)`` frozen consuming-layer indexer scores.
        student_score: ``(b, Q, Sk)`` trainable producing-layer scores.
        causal: ``(Q, Sk)`` boolean mask (broadcast over batch).

    Both are masked to the causal region with a finite constant before softmax.
    """
    mask = causal.unsqueeze(0)
    t = _apply_causal(teacher_score, mask)
    s = _apply_causal(student_score, mask)
    log_p = F.log_softmax(t, dim=-1)
    p = log_p.exp()
    log_q = F.log_softmax(s, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)  # (b, Q)
    return kl.mean()


def cross_layer_rank(
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    causal: torch.Tensor,
    num_pairs: int,
    margin: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean pairwise MarginRanking hinge aligning student key order to teacher's.

    For each query row we sample ``num_pairs`` random causal key pairs ``(i, j)``,
    take the target sign from the teacher (``+1`` if the teacher scores ``i`` above
    ``j``, ``-1`` otherwise), and penalise the student when it disagrees:
    ``max(0, margin - target * (s_i - s_j))``. Only causal keys are sampled, so no
    explicit score masking is needed. Teacher ties (equal scores) are dropped.

    Args:
        teacher_score: ``(b, Q, Sk)`` frozen consuming-layer indexer scores.
        student_score: ``(b, Q, Sk)`` trainable producing-layer scores.
        causal: ``(Q, Sk)`` boolean mask (broadcast over batch); ``True`` where a
            key position is <= its query position.
        num_pairs: key pairs sampled per query row.
        margin: ranking margin (``0.0`` is a pure order hinge, scale-free).
        generator: optional RNG so pair sampling is reproducible and identical
            across tensor-parallel ranks.
    """
    b, q, sk = teacher_score.shape
    # Number of causal keys per row (= query_pos + 1); sample uniformly in
    # [0, valid) so every drawn index is causal by construction.
    valid = causal.sum(dim=-1).clamp(min=1).to(dtype=torch.float32).view(1, q, 1)
    shape = (b, q, num_pairs)
    dev = teacher_score.device
    # Force fp32 sampling: under a bf16 default dtype (as in training) bf16 rand
    # scaled by a long sequence length would collapse index precision.
    i = (torch.rand(shape, device=dev, dtype=torch.float32, generator=generator) * valid).long().clamp_(max=sk - 1)
    j = (torch.rand(shape, device=dev, dtype=torch.float32, generator=generator) * valid).long().clamp_(max=sk - 1)
    t_i, t_j = teacher_score.gather(-1, i), teacher_score.gather(-1, j)
    s_i, s_j = student_score.gather(-1, i), student_score.gather(-1, j)
    target = torch.sign(t_i - t_j)
    hinge = torch.clamp(margin - target * (s_i - s_j), min=0.0)
    # Drop teacher ties (incl. i == j) via a masked mean rather than boolean
    # indexing, so peak memory stays O(b * Q * num_pairs) with no extra copy and
    # the graph is grad-connected even when every pair is a tie.
    keep = (target != 0).to(hinge.dtype)
    denom = keep.sum().clamp(min=1.0)
    return (hinge * keep).sum() / denom


def cross_layer_list_mle(
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    causal: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:
    """Listwise Plackett-Luce MLE on the teacher's top-``C`` keys per query row.

    Gathers the teacher's top-``C`` candidates (already the region top-k
    selection cares about), then maximises the likelihood of the teacher's
    descending order under the student scores:
    ``sum_i [ s_(i) - logsumexp(s_(i), ..., s_(C-1)) ]`` where ``(i)`` indexes
    keys in teacher-descending order. Unlike the pairwise hinge this spends the
    entire budget on boundary-relevant keys and needs no margin.

    Args:
        teacher_score: ``(b, Q, Sk)`` frozen consuming-layer indexer scores.
        student_score: ``(b, Q, Sk)`` trainable producing-layer scores.
        causal: ``(Q, Sk)`` boolean mask (broadcast over batch).
        num_candidates: candidate pool size ``C`` (use ~2-4x the model top-k).
    """
    mask = causal.unsqueeze(0)
    t = _apply_causal(teacher_score, mask)
    s = _apply_causal(student_score, mask)
    c = min(num_candidates, t.shape[-1])
    # topk returns candidates already sorted by descending teacher score, so the
    # gathered student scores are in the teacher's oracle order (no argsort).
    top = torch.topk(t, c, dim=-1)
    idx = top.indices  # (b, Q, C)
    t_sorted = top.values  # (b, Q, C), descending
    s_sorted = s.gather(-1, idx)  # (b, Q, C) student in teacher order
    # Short rows: non-causal keys pull into the tail as _NEG; drop them from the
    # mean. Their ~0 exp mass leaves the logcumsumexp of kept positions intact.
    valid = t_sorted > (_NEG / 2)
    # fp32 logcumsumexp: bf16 accumulation over C candidates is unstable.
    s_fp32 = s_sorted.to(torch.float32)
    log_cumsum = torch.logcumsumexp(s_fp32.flip(-1), dim=-1).flip(-1)
    log_probs = s_fp32 - log_cumsum
    denom = valid.sum().clamp(min=1)
    return -(log_probs * valid).sum() / denom


@torch.no_grad()
def topk_overlap(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    k: int,
    causal: torch.Tensor,
) -> float:
    """Mean Jaccard of causal top-k key sets between two ``(b, Q, Sk)`` matrices."""
    mask = causal.unsqueeze(0)
    a = _apply_causal(score_a, mask)  # (B, Q, Sk)
    b = _apply_causal(score_b, mask)
    valid = causal.sum(dim=-1)  # (Q,)
    valid_list = valid.tolist()
    jacc: list[float] = []
    for batch_a, batch_b in zip(a, b, strict=True):
        for row_a, row_b, n_valid in zip(batch_a, batch_b, valid_list, strict=True):
            kk = min(k, int(n_valid))
            if kk <= 0:
                continue
            set_a = set(torch.topk(row_a, kk).indices.tolist())
            set_b = set(torch.topk(row_b, kk).indices.tolist())
            union = len(set_a | set_b)
            jacc.append(len(set_a & set_b) / union if union else 1.0)
    return sum(jacc) / len(jacc) if jacc else float("nan")
