"""Tests for the distillation objectives (cross-layer KL and pairwise ranking)."""

from __future__ import annotations

import torch

from lagged_indexer_dsa.distill.losses import (
    causal_mask,
    cross_layer_kl,
    cross_layer_list_mle,
    cross_layer_rank,
    topk_overlap,
)


def _causal(q: int, sk: int, lo: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Query positions in ``[lo, sk)`` and their causal ``(Q, Sk)`` mask."""
    query_pos = torch.arange(lo, sk)
    return query_pos, causal_mask(query_pos, sk)


def test_rank_loss_zero_when_student_matches_teacher_order() -> None:
    torch.manual_seed(0)
    _, causal = _causal(q=8, sk=32, lo=8)
    teacher = torch.randn(1, causal.shape[0], 32)
    gen = torch.Generator().manual_seed(1)
    # Student that preserves the teacher's order exactly -> no rank violations.
    loss = cross_layer_rank(teacher, teacher.clone(), causal, num_pairs=64, generator=gen)
    assert loss.item() == 0.0


def test_rank_loss_positive_when_order_reversed() -> None:
    torch.manual_seed(0)
    _, causal = _causal(q=8, sk=32, lo=8)
    teacher = torch.randn(1, causal.shape[0], 32)
    gen = torch.Generator().manual_seed(1)
    loss = cross_layer_rank(teacher, -teacher, causal, num_pairs=256, generator=gen)
    assert loss.item() > 0.0


def test_rank_loss_is_deterministic_with_generator() -> None:
    _, causal = _causal(q=8, sk=48, lo=8)
    teacher = torch.randn(1, causal.shape[0], 48)
    student = torch.randn(1, causal.shape[0], 48)
    a = cross_layer_rank(teacher, student, causal, num_pairs=128,
                         generator=torch.Generator().manual_seed(5))
    b = cross_layer_rank(teacher, student, causal, num_pairs=128,
                         generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_rank_loss_only_samples_causal_keys() -> None:
    # A student that is correct on the causal region but adversarial on the
    # (never-selectable) future keys must still incur zero loss, proving those
    # non-causal columns are never sampled.
    _, causal = _causal(q=4, sk=16, lo=4)
    q = causal.shape[0]
    teacher = torch.zeros(1, q, 16)
    student = torch.zeros(1, q, 16)
    for row in range(q):
        qpos = row + 4
        teacher[0, row, : qpos + 1] = torch.arange(qpos + 1, dtype=torch.float32)
        student[0, row, : qpos + 1] = torch.arange(qpos + 1, dtype=torch.float32)
        # Corrupt the non-causal tail with an order-violating pattern.
        student[0, row, qpos + 1 :] = 1e3
        teacher[0, row, qpos + 1 :] = -1e3
    loss = cross_layer_rank(teacher, student, causal, num_pairs=512,
                            generator=torch.Generator().manual_seed(3))
    assert loss.item() == 0.0


def test_rank_loss_backward_flows_to_student_only() -> None:
    _, causal = _causal(q=6, sk=24, lo=6)
    q = causal.shape[0]
    teacher = torch.randn(1, q, 24)
    student = torch.randn(1, q, 24, requires_grad=True)
    loss = cross_layer_rank(teacher, student, causal, num_pairs=128,
                            generator=torch.Generator().manual_seed(2))
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_kl_loss_still_zero_at_identity() -> None:
    _, causal = _causal(q=8, sk=32, lo=8)
    teacher = torch.randn(1, causal.shape[0], 32)
    loss = cross_layer_kl(teacher, teacher.clone(), causal)
    assert abs(loss.item()) < 1e-6


def test_rank_loss_under_bf16_default_dtype() -> None:
    # Training sets the default dtype to bf16; the loss must not crash and its
    # fp32-pinned pair sampling must stay in range for a long sequence.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        sk = 4096
        _, causal = _causal(q=8, sk=sk, lo=sk - 8)
        teacher = torch.randn(1, causal.shape[0], sk)
        student = torch.randn(1, causal.shape[0], sk, requires_grad=True)
        loss = cross_layer_rank(teacher, student, causal, num_pairs=256,
                                generator=torch.Generator().manual_seed(4))
        assert loss.dtype == torch.bfloat16
        assert torch.isfinite(loss).all()
        loss.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()
    finally:
        torch.set_default_dtype(prev)


def test_rank_loss_trains_student_toward_teacher_order() -> None:
    # Gradient-direction check: optimising a free student score with the rank
    # objective must reduce the loss and raise top-k overlap above its random
    # init (absolute overlap is not asserted -- this adversarial toy has free
    # per-row params and a structureless random teacher).
    from lagged_indexer_dsa.distill.losses import topk_overlap
    torch.manual_seed(0)
    _, causal = _causal(q=16, sk=64, lo=16)
    q = causal.shape[0]
    teacher = torch.randn(1, q, 64)
    student = torch.nn.Parameter(torch.randn(1, q, 64))
    init_overlap = topk_overlap(teacher, student.detach(), k=8, causal=causal)
    opt = torch.optim.SGD([student], lr=0.5)
    gen = torch.Generator().manual_seed(11)
    first = last = None
    for step in range(500):
        opt.zero_grad(set_to_none=True)
        loss = cross_layer_rank(teacher, student, causal, num_pairs=512, generator=gen)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
    assert topk_overlap(teacher, student.detach(), k=8, causal=causal) > init_overlap


def test_rank_loss_memory_is_independent_of_seq_len() -> None:
    # A 128K-key sequence: any O(Sk^2) or dense (Q, Sk, Sk) materialisation would
    # OOM instantly. Completing with a modest num_pairs proves peak memory scales
    # with the pair budget, not the sequence length.
    sk = 131072
    _, causal = _causal(q=4, sk=sk, lo=sk - 4)
    teacher = torch.randn(1, causal.shape[0], sk)
    student = torch.randn(1, causal.shape[0], sk, requires_grad=True)
    loss = cross_layer_rank(teacher, student, causal, num_pairs=512,
                            generator=torch.Generator().manual_seed(6))
    loss.backward()
    assert torch.isfinite(loss).all()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_list_mle_minimised_when_student_matches_teacher() -> None:
    torch.manual_seed(0)
    _, causal = _causal(q=8, sk=64, lo=8)
    teacher = torch.randn(1, causal.shape[0], 64)
    matched = cross_layer_list_mle(teacher, teacher.clone(), causal, num_candidates=32)
    scrambled = cross_layer_list_mle(teacher, torch.randn_like(teacher), causal, num_candidates=32)
    # Matching the teacher's order gives a lower NLL than a random student.
    assert matched.item() < scrambled.item()


def test_list_mle_causal_only_and_short_context() -> None:
    # First query rows have fewer valid keys than num_candidates: padded
    # non-causal tail must be dropped and no grad may flow through it.
    _, causal = _causal(q=2, sk=32, lo=2)
    teacher = torch.randn(1, causal.shape[0], 32)
    student = torch.randn(1, causal.shape[0], 32, requires_grad=True)
    loss = cross_layer_list_mle(teacher, student, causal, num_candidates=32)
    loss.backward()
    assert torch.isfinite(loss).all()
    non_causal = ~causal.unsqueeze(0)
    assert torch.allclose(student.grad[non_causal], torch.zeros_like(student.grad[non_causal]))


def test_list_mle_under_bf16_default_dtype() -> None:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        sk = 4096
        _, causal = _causal(q=8, sk=sk, lo=sk - 8)
        teacher = torch.randn(1, causal.shape[0], sk)
        student = torch.randn(1, causal.shape[0], sk, requires_grad=True)
        loss = cross_layer_list_mle(teacher, student, causal, num_candidates=1024)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()
    finally:
        torch.set_default_dtype(prev)


def test_list_mle_memory_is_independent_of_seq_len() -> None:
    # 128K keys with a modest candidate pool: any O(Sk^2) materialisation OOMs.
    sk = 131072
    _, causal = _causal(q=4, sk=sk, lo=sk - 4)
    teacher = torch.randn(1, causal.shape[0], sk)
    student = torch.randn(1, causal.shape[0], sk, requires_grad=True)
    loss = cross_layer_list_mle(teacher, student, causal, num_candidates=512)
    loss.backward()
    assert torch.isfinite(loss).all()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_list_mle_trains_student_toward_teacher_order() -> None:
    torch.manual_seed(0)
    _, causal = _causal(q=16, sk=64, lo=16)
    q = causal.shape[0]
    teacher = torch.randn(1, q, 64)
    student = torch.nn.Parameter(torch.randn(1, q, 64))
    init_overlap = topk_overlap(teacher, student.detach(), k=8, causal=causal)
    opt = torch.optim.SGD([student], lr=0.5)
    first = last = None
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        loss = cross_layer_list_mle(teacher, student, causal, num_candidates=32)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
    assert topk_overlap(teacher, student.detach(), k=8, causal=causal) > init_overlap
