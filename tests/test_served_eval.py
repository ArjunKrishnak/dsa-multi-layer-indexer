"""Unit tests for the served-eval helpers (no server, no model)."""
from __future__ import annotations

import pytest

from lagged_indexer_dsa.served_eval import (
    aggregate_ruler,
    extract_mcq_answer,
    middle_truncate_tokens,
)


class _CharTokenizer:
    """One token per character: truncation semantics are visible in the text."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The correct answer is (B).", "B"),
        ("the correct answer is C", "C"),
        ("Reasoning... The correct answer is (d)", "D"),
        ("I think (A) fits best.", "A"),
        ("I choose (b).", "B"),        # lowercase parenthesized
        ("d) is my pick", "D"),        # lowercase leading letter
        ("B", "B"),
        ("  C) because", "C"),
        ("No idea.", None),
        ("The correct answer is (E)", None),
        ("", None),
    ],
)
def test_extract_mcq_answer(text, expected):
    assert extract_mcq_answer(text) == expected


def test_middle_truncate_keeps_head_and_tail():
    tok = _CharTokenizer()
    text = "0123456789"
    out, n, truncated = middle_truncate_tokens(tok, text, 4)
    assert (out, n, truncated) == ("0189", 10, True)


def test_middle_truncate_odd_budget_favors_tail():
    tok = _CharTokenizer()
    out, _, truncated = middle_truncate_tokens(tok, "abcdefgh", 5)
    assert truncated and out == "abfgh"


def test_middle_truncate_noop_when_short():
    tok = _CharTokenizer()
    out, n, truncated = middle_truncate_tokens(tok, "abc", 3)
    assert (out, n, truncated) == ("abc", 3, False)


def test_middle_truncate_rejects_tiny_budget():
    with pytest.raises(ValueError):
        middle_truncate_tokens(_CharTokenizer(), "abc", 1)


def test_aggregate_ruler_means_over_tasks_then_lengths():
    scores = [
        ("niah_single", 4096, 1.0), ("niah_single", 4096, 0.0),
        ("vt", 4096, 1.0),
        ("niah_single", 8192, 0.5),
        # vt missing at 8192: the 8192 mean is over the present task only
    ]
    overall, by_length, by_tl = aggregate_ruler(scores, ["niah_single", "vt"], [4096, 8192, 16384])
    assert by_tl["niah_single"]["4096"] == 0.5 and by_tl["vt"]["4096"] == 1.0
    assert by_length["4096"] == pytest.approx(0.75)
    assert by_length["8192"] == pytest.approx(0.5)
    assert by_length["16384"] is None
    assert overall == pytest.approx((0.75 + 0.5) / 2)


def test_aggregate_ruler_empty():
    overall, by_length, by_tl = aggregate_ruler([], ["vt"], [4096])
    assert overall == 0.0 and by_length == {"4096": None} and by_tl == {}
