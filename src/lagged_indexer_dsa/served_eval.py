"""Pure helpers shared by the served-eval clients (eval/eval_*_served.py).

Kept in the package so they are importable without path hacks and unit-testable
without a server: answer-letter parsing for multiple-choice suites, token-level
middle truncation, and the RULER aggregation (mean over tasks per length, then
mean over lengths).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Protocol, Sequence


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


_MCQ_ANSWER_RE = re.compile(r"The correct answer is\s*\(?([A-D])\)?", re.IGNORECASE)
_PAREN_LETTER_RE = re.compile(r"\(([A-D])\)", re.IGNORECASE)
_LEADING_LETTER_RE = re.compile(r"\s*([A-D])\b", re.IGNORECASE)


def extract_mcq_answer(text: str) -> str | None:
    """Parse the LongBench v2 answer letter.

    Priority: the requested format ``The correct answer is (X)``, then the
    first parenthesized letter, then a bare leading letter. ``None`` when the
    response carries no recognizable choice (counted as wrong by the caller).
    """
    m = _MCQ_ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _PAREN_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _LEADING_LETTER_RE.match(text)
    return m.group(1).upper() if m else None


def middle_truncate_tokens(tok: TokenizerLike, text: str, max_tokens: int) -> tuple[str, int, bool]:
    """Keep the first and last halves of the token stream when it exceeds ``max_tokens``.

    Returns ``(text, n_tokens_before, truncated)``. Mirrors the official
    LongBench v1/v2 rule (head + tail, middle dropped).
    """
    if max_tokens < 2:
        raise ValueError(f"max_tokens must be >= 2, got {max_tokens}")
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, len(ids), False
    half = max_tokens // 2
    kept = list(ids[:half]) + list(ids[-(max_tokens - half):])
    return tok.decode(kept, skip_special_tokens=True), len(ids), True


def aggregate_ruler(
    scores: Iterable[tuple[str, int, float]], tasks: Sequence[str], lengths: Sequence[int]
) -> tuple[float, dict[str, float | None], dict[str, dict[str, float]]]:
    """RULER aggregation: per (task, length) mean -> per length mean over tasks -> overall.

    ``scores`` are ``(task, length, score)`` triples. Lengths with no scored
    task map to ``None`` and are excluded from the overall mean.
    """
    cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for task, length, s in scores:
        cell[(task, length)].append(float(s))
    by_task_length: dict[str, dict[str, float]] = {}
    for (task, length), vals in sorted(cell.items()):
        by_task_length.setdefault(task, {})[str(length)] = sum(vals) / len(vals)
    by_length: dict[str, float | None] = {}
    for length in lengths:
        vals = [by_task_length[t][str(length)] for t in tasks if str(length) in by_task_length.get(t, {})]
        by_length[str(length)] = sum(vals) / len(vals) if vals else None
    present = [v for v in by_length.values() if v is not None]
    overall = sum(present) / len(present) if present else 0.0
    return overall, by_length, by_task_length
