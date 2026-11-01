"""RULER-style synthetic long-context task generators and scorers (CPU-only).

Reimplements a subset of RULER's procedurally generated tasks in-repo (RULER
is not a library; its tasks are synthetic, so we regenerate them from a seed
rather than depending on an external eval framework).  Everything is
deterministic given ``(seed, task, context_length, depth_percent, index)``.

Tasks and metrics
-----------------
``niah_single``
    One needle sentence ``"The special magic number for {word} is {number}."``
    hidden in essay-like filler; the query asks for the number.
    Metric ``needle_substring_acc``: 1.0 iff the gold number appears as an
    exact substring of the prediction, else 0.0.
``niah_multikey``
    Three needles with distinct key words and distinct 7-digit numbers; the
    query asks for one key's number (the other two needles are distractors).
    The queried needle sits at the swept depth; distractors at seeded-uniform
    random depths.  Metric ``needle_substring_acc`` on the queried number.
``niah_multivalue``
    One key word with four distinct values scattered through the context (the
    first value at the swept depth, the other three at seeded-uniform random
    depths); the model must return all four.  Metric ``value_recall``: the
    fraction of the four gold numbers appearing as exact substrings of the
    prediction (set match with partial credit).
``vt``
    Variable tracking: one assignment ``"The value of variable {A} is {v}."``
    followed by four copy hops ``"The value of variable {B} equals the value
    of variable {A}."`` (:data:`VT_HOPS` = 4 hops, 5 statements).  Statements
    are placed in chain order, the first at the swept depth and the rest
    spread evenly up to 95% depth, so the chain always reads forward.  The
    query gives the VALUE and asks for all variable names in the chain (the
    canonical RULER vt direction: "Find all variables that are assigned the
    value {v}").  Gold answers are the chain's 5 variable names; metric
    ``name_recall``: the fraction of the names present in the prediction as
    case-insensitive substrings (canonical ``string_match_all`` semantics,
    partial credit).
``cwe``
    Common-words extraction (lite): the context is a long word list in which
    10 seeded target words each appear once per round while background words
    appear roughly once per five rounds (deterministic cyclic sampling, so
    targets are strictly the 10 most frequent — asserted at generation time).
    The query asks for the 10 most common words.  Depth does not apply
    (``depth_percent`` must be ``None``); metric ``word_set_recall``: the
    fraction of the 10 gold words present in the prediction as
    case-insensitive whole words (set match with partial credit).

Token-budget construction
-------------------------
Haystacks are built by *token* count against the caller's tokenizer: the
context is trimmed so the full prompt (preamble + haystack + query) is exactly
``context_length - GEN_TOKEN_SLACK`` tokens (``cwe`` may undershoot by a few
tokens because it trims at word granularity).  The slack reserves room for
generation (every :data:`MAX_NEW_BY_TASK` budget fits inside it), so a model
loaded with ``max_seq_len = context_length`` can prefill and decode without
overflow.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Any, Protocol, Sequence


class TokenizerLike(Protocol):
    """The tokenizer surface the generators need (HF tokenizers satisfy it)."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


RULER_TASKS: tuple[str, ...] = (
    "niah_single", "niah_multikey", "niah_multivalue", "vt", "cwe",
)

# Tasks whose generator takes a needle depth; ``cwe`` has no needle position.
DEPTH_SWEPT_TASKS: frozenset[str] = frozenset(
    {"niah_single", "niah_multikey", "niah_multivalue", "vt"}
)

DEFAULT_DEPTHS: tuple[int, ...] = (10, 25, 50, 75, 90)

# Tokens reserved out of ``context_length`` so prompt + greedy generation fit
# in a KV cache sized to the nominal context length.
GEN_TOKEN_SLACK: int = 64

METRIC_BY_TASK: dict[str, str] = {
    "niah_single": "needle_substring_acc",
    "niah_multikey": "needle_substring_acc",
    "niah_multivalue": "value_recall",
    "vt": "name_recall",
    "cwe": "word_set_recall",
}

MAX_NEW_BY_TASK: dict[str, int] = {
    "niah_single": 32,
    "niah_multikey": 32,
    "niah_multivalue": 64,
    "vt": 32,
    "cwe": 64,
}

VT_HOPS: int = 4

# Chain/scatter placements never go past this depth so the query stays last.
_MAX_INSERT_FRAC: float = 0.95
_MIN_FILLER_TOKENS: int = 32
_FILLER_SENTENCES_PER_CHUNK: int = 20
_CWE_BG_PER_ROUND: int = 20

# ---------------------------------------------------------------------------
# Fixed word lists (the needles' vocabulary is disjoint from anything the
# filler can produce: filler is lowercase pastoral prose with no digits, so a
# 7-digit number or "special magic number for <fruit>" can only come from a
# planted needle).
# ---------------------------------------------------------------------------

_FILLER_ADJECTIVES: tuple[str, ...] = (
    "quiet", "golden", "distant", "ancient", "weathered", "patient",
    "gentle", "narrow", "crowded", "empty", "fragrant", "shaded",
    "winding", "sturdy", "faded", "bustling",
)

_FILLER_NOUNS: tuple[str, ...] = (
    "meadow", "river", "lantern", "orchard", "valley", "breeze", "garden",
    "harvest", "traveler", "village", "mountain", "forest", "pathway",
    "bridge", "harbor", "market", "festival", "letter", "journey", "season",
    "courtyard", "windmill", "shoreline", "cottage", "granary", "stable",
    "chapel", "mill",
)

_FILLER_TEMPLATES: tuple[str, ...] = (
    "The {a1} {n1} lay beyond the {a2} {n2}, and the {n3} kept its slow rhythm.",
    "Past the {n1}, a {a1} {n2} opened onto the {a2} {n3}.",
    "Every {n1} in the {a1} {n2} seemed smaller beside the {a2} {n3}.",
    "A {a1} {n1} followed the {n2} down toward the {a2} {n3}.",
    "In the {a1} {n2}, the {n1} waited while the {n3} slowly changed.",
    "The {n1} near the {a1} {n2} carried stories of the {a2} {n3}.",
    "Beyond the {a2} {n3}, the {a1} {n1} met the {n2} at dusk.",
    "Through the {n2} and the {a1} {n3}, the {n1} wandered without hurry.",
)

NIAH_KEY_WORDS: tuple[str, ...] = (
    "apple", "banana", "cherry", "dragonfruit", "elderberry", "fig", "grape",
    "hazelnut", "kiwi", "lemon", "mango", "nectarine", "olive", "papaya",
    "quince", "raspberry", "strawberry", "tangerine", "walnut", "apricot",
    "blueberry", "coconut", "date", "grapefruit", "lime", "melon", "peach",
    "pear", "plum", "pomegranate",
)

CWE_TARGET_POOL: tuple[str, ...] = (
    "falcon", "ember", "willow", "copper", "cedar", "raven", "amber",
    "slate", "maple", "heron", "aspen", "coral", "tiger", "otter", "walrus",
    "condor", "ivory", "cobalt", "onyx", "jasper", "linen", "saffron",
    "indigo", "crimson", "viola", "tundra", "prairie", "lagoon", "glacier",
    "canyon",
)

_CWE_BG_PREFIXES: tuple[str, ...] = (
    "north", "south", "east", "west", "over", "under", "out", "up", "down",
    "back",
)
_CWE_BG_SUFFIXES: tuple[str, ...] = (
    "land", "field", "stream", "light", "stone", "wood", "gate", "port",
    "ridge", "brook",
)
CWE_BACKGROUND_POOL: tuple[str, ...] = tuple(
    p + s for p in _CWE_BG_PREFIXES for s in _CWE_BG_SUFFIXES
)

_NIAH_PREAMBLE = (
    "Some special magic numbers are hidden in the following text. Make sure "
    "to memorize them. I will quiz you about the numbers afterwards.\n\n"
)
_VT_PREAMBLE = (
    "Memorize and track the chained variable assignments hidden in the "
    "following text.\n\n"
)
_CWE_PREAMBLE = (
    "Below is a long list of words. Memorize how often each word appears.\n\n"
)

_MAGIC_SENTENCE = "The special magic number for {word} is {number}."


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _filler_sentence(rng: random.Random) -> str:
    template = rng.choice(_FILLER_TEMPLATES)
    n1, n2, n3 = rng.sample(_FILLER_NOUNS, 3)
    return template.format(
        a1=rng.choice(_FILLER_ADJECTIVES),
        a2=rng.choice(_FILLER_ADJECTIVES),
        n1=n1, n2=n2, n3=n3,
    )


def _build_filler_ids(
    tokenizer: TokenizerLike, rng: random.Random, budget: int
) -> list[int]:
    """Exactly ``budget`` token ids of seeded essay-like filler.

    Sentences are encoded a paragraph at a time (single encode call per ~20
    sentences) because per-sentence encode calls dominate generation time at
    64k-token contexts; the trailing paragraph is trimmed at token
    granularity so the budget is exact.
    """
    ids: list[int] = []
    while len(ids) < budget:
        paragraph = " ".join(
            _filler_sentence(rng) for _ in range(_FILLER_SENTENCES_PER_CHUNK)
        )
        ids.extend(_encode(tokenizer, " " + paragraph))
    return ids[:budget]


def _assemble_prompt(
    tokenizer: TokenizerLike,
    rng: random.Random,
    context_length: int,
    preamble: str,
    query: str,
    needles: list[tuple[float, str]],
) -> tuple[list[int], list[float]]:
    """Splice needle sentences into token-budgeted filler.

    ``needles`` is a list of ``(depth_fraction, sentence)``.  Returns the full
    prompt ids (preamble + context + query, exactly
    ``context_length - GEN_TOKEN_SLACK`` tokens) and each needle's realised
    start position as a fraction of the full prompt length, in the same order
    as ``needles``.
    """
    preamble_ids = _encode(tokenizer, preamble)
    query_ids = _encode(tokenizer, query)
    needle_ids = [_encode(tokenizer, " " + sentence) for _, sentence in needles]

    context_budget = (
        context_length - GEN_TOKEN_SLACK - len(preamble_ids) - len(query_ids)
    )
    filler_budget = context_budget - sum(len(n) for n in needle_ids)
    if filler_budget < _MIN_FILLER_TOKENS:
        raise ValueError(
            f"context_length={context_length} leaves only {filler_budget} "
            f"filler tokens after preamble/query/needles/slack; too small"
        )
    for frac, _ in needles:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"needle depth fraction {frac} outside [0, 1]")

    filler = _build_filler_ids(tokenizer, rng, filler_budget)

    # Splice in ascending-depth order via segment assembly so realised token
    # offsets are exact (no post-hoc shift bookkeeping).
    order = sorted(range(len(needles)), key=lambda i: needles[i][0])
    out = list(preamble_ids)
    starts: list[int] = [0] * len(needles)
    prev = 0
    for i in order:
        pos = round(needles[i][0] * filler_budget)
        out.extend(filler[prev:pos])
        starts[i] = len(out)
        out.extend(needle_ids[i])
        prev = pos
    out.extend(filler[prev:])
    out.extend(query_ids)

    total = len(out)
    assert total == context_length - GEN_TOKEN_SLACK, (total, context_length)
    fracs = [start / total for start in starts]
    return out, fracs


def _scatter_fracs(rng: random.Random, n: int) -> list[float]:
    return [rng.uniform(0.05, _MAX_INSERT_FRAC) for _ in range(n)]


# ---------------------------------------------------------------------------
# Per-task generators
# ---------------------------------------------------------------------------

def _gen_niah_single(
    tokenizer: TokenizerLike, rng: random.Random, context_length: int, depth: float
) -> dict[str, Any]:
    word = rng.choice(NIAH_KEY_WORDS)
    number = rng.randint(1_000_000, 9_999_999)
    needle = _MAGIC_SENTENCE.format(word=word, number=number)
    query = (
        f"\n\nQuestion: What is the special magic number for {word} "
        f"mentioned in the text?\nAnswer: The special magic number for "
        f"{word} is"
    )
    ids, fracs = _assemble_prompt(
        tokenizer, rng, context_length, _NIAH_PREAMBLE, query, [(depth, needle)]
    )
    return {
        "ids": ids,
        "answers": [str(number)],
        "meta": {"needle_fracs": fracs, "primary_frac": fracs[0], "key_word": word},
    }


def _gen_niah_multikey(
    tokenizer: TokenizerLike, rng: random.Random, context_length: int, depth: float
) -> dict[str, Any]:
    words = rng.sample(NIAH_KEY_WORDS, 3)
    numbers = rng.sample(range(1_000_000, 10_000_000), 3)
    # words[0] is the queried key at the swept depth; the rest are distractors.
    placements = [depth] + _scatter_fracs(rng, 2)
    needles = [
        (placements[j], _MAGIC_SENTENCE.format(word=words[j], number=numbers[j]))
        for j in range(3)
    ]
    query = (
        f"\n\nQuestion: What is the special magic number for {words[0]} "
        f"mentioned in the text?\nAnswer: The special magic number for "
        f"{words[0]} is"
    )
    ids, fracs = _assemble_prompt(
        tokenizer, rng, context_length, _NIAH_PREAMBLE, query, needles
    )
    return {
        "ids": ids,
        "answers": [str(numbers[0])],
        "meta": {
            "needle_fracs": fracs,
            "primary_frac": fracs[0],
            "key_word": words[0],
            "distractor_words": words[1:],
            "distractor_numbers": [str(n) for n in numbers[1:]],
        },
    }


def _gen_niah_multivalue(
    tokenizer: TokenizerLike, rng: random.Random, context_length: int, depth: float
) -> dict[str, Any]:
    word = rng.choice(NIAH_KEY_WORDS)
    numbers = rng.sample(range(1_000_000, 10_000_000), 4)
    # First value at the swept depth, the other three scattered.
    placements = [depth] + _scatter_fracs(rng, 3)
    needles = [
        (placements[j], _MAGIC_SENTENCE.format(word=word, number=numbers[j]))
        for j in range(4)
    ]
    query = (
        f"\n\nQuestion: What are all the special magic numbers for {word} "
        f"mentioned in the text?\nAnswer: The special magic numbers for "
        f"{word} are"
    )
    ids, fracs = _assemble_prompt(
        tokenizer, rng, context_length, _NIAH_PREAMBLE, query, needles
    )
    return {
        "ids": ids,
        "answers": [str(n) for n in numbers],
        "meta": {"needle_fracs": fracs, "primary_frac": fracs[0], "key_word": word},
    }


def _gen_vt(
    tokenizer: TokenizerLike, rng: random.Random, context_length: int, depth: float
) -> dict[str, Any]:
    n_vars = VT_HOPS + 1
    names: list[str] = []
    while len(names) < n_vars:
        cand = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        if cand not in names:
            names.append(cand)
    value = rng.randint(10_000, 99_999)

    statements = [f"The value of variable {names[0]} is {value}."]
    for k in range(1, n_vars):
        statements.append(
            f"The value of variable {names[k]} equals the value of "
            f"variable {names[k - 1]}."
        )
    # Chain order is preserved in depth: first assignment at the swept depth,
    # hops spread evenly up to _MAX_INSERT_FRAC so the chain reads forward.
    step = max(0.0, _MAX_INSERT_FRAC - depth) / (n_vars - 1)
    needles = [(depth + k * step, statements[k]) for k in range(n_vars)]
    # Canonical RULER vt query direction: give the value, ask for ALL
    # variable names in the chain (RULER data/synthetic/constants.py).
    query = (
        f"\n\nQuestion: Find all variables that are assigned the value "
        f"{value} in the text above.\nAnswer: According to the chain of "
        f"variable assignment in the text above, {n_vars} variables are "
        f"assigned the value {value}, they are:"
    )
    ids, fracs = _assemble_prompt(
        tokenizer, rng, context_length, _VT_PREAMBLE, query, needles
    )
    return {
        "ids": ids,
        "answers": list(names),
        "meta": {
            "needle_fracs": fracs,
            "primary_frac": fracs[0],
            "chain_names": names,
            "value": str(value),
        },
    }


def _gen_cwe(
    tokenizer: TokenizerLike, rng: random.Random, context_length: int
) -> dict[str, Any]:
    targets = rng.sample(CWE_TARGET_POOL, 10)
    background = list(CWE_BACKGROUND_POOL)
    rng.shuffle(background)

    query = (
        "\n\nQuestion: What are the 10 most frequently occurring words in "
        "the above list?\nAnswer: The 10 most common words are"
    )
    preamble_ids = _encode(tokenizer, _CWE_PREAMBLE)
    query_ids = _encode(tokenizer, query)
    context_budget = (
        context_length - GEN_TOKEN_SLACK - len(preamble_ids) - len(query_ids)
    )
    if context_budget < _MIN_FILLER_TOKENS:
        raise ValueError(
            f"context_length={context_length} too small for cwe preamble/query"
        )

    # Each round holds every target once plus a cyclic 20-word slice of the
    # 100-word background pool, so after R full rounds every target appears
    # R times while each background word appears about R/5 times — the
    # frequency gap that makes the answer well-defined.
    word_ids_cache: dict[str, list[int]] = {}
    context_ids: list[int] = []
    counts: Counter[str] = Counter()
    round_idx = 0
    budget_hit = False
    while not budget_hit:
        start = (round_idx * _CWE_BG_PER_ROUND) % len(background)
        chunk = (background + background)[start:start + _CWE_BG_PER_ROUND]
        block = targets + chunk
        rng.shuffle(block)
        for word in block:
            if word not in word_ids_cache:
                word_ids_cache[word] = _encode(tokenizer, " " + word)
            wids = word_ids_cache[word]
            if len(context_ids) + len(wids) > context_budget:
                budget_hit = True
                break
            context_ids.extend(wids)
            counts[word] += 1
        round_idx += 1

    min_target = min(counts[t] for t in targets)
    max_background = max(
        (c for w, c in counts.items() if w not in targets), default=0
    )
    if min_target <= max_background:
        raise ValueError(
            f"cwe targets are not strictly the most frequent words "
            f"(min target count {min_target} <= max background count "
            f"{max_background}); context_length={context_length} is too short"
        )

    ids = preamble_ids + context_ids + query_ids
    return {
        "ids": ids,
        "answers": list(targets),
        "meta": {
            "targets": list(targets),
            "rounds": round_idx,
            "min_target_count": min_target,
            "max_background_count": max_background,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_example(
    task: str,
    tokenizer: TokenizerLike,
    *,
    context_length: int,
    depth_percent: int | None,
    seed: int,
    index: int,
) -> dict[str, Any]:
    """Generate one deterministic example for ``task``.

    Returns a dict with ``ids`` (full prompt token ids), ``answers`` (gold
    strings for :func:`score_example`), and ``meta`` (task-specific
    diagnostics including realised needle depth fractions).
    """
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {RULER_TASKS}")
    if task in DEPTH_SWEPT_TASKS:
        if depth_percent is None:
            raise ValueError(f"task {task!r} requires depth_percent")
        if not 0 <= depth_percent <= 100:
            raise ValueError(f"depth_percent={depth_percent} outside [0, 100]")
    elif depth_percent is not None:
        raise ValueError(
            f"task {task!r} has no needle depth; pass depth_percent=None"
        )

    # String seeding is deterministic across processes/platforms (CPython
    # hashes str seeds with sha512), so each cell is reproducible in isolation
    # regardless of generation order.
    rng = random.Random(f"ruler:{seed}:{task}:{context_length}:{depth_percent}:{index}")

    if task == "cwe":
        example = _gen_cwe(tokenizer, rng, context_length)
    else:
        depth = depth_percent / 100.0  # type: ignore[operator]
        if task == "niah_single":
            example = _gen_niah_single(tokenizer, rng, context_length, depth)
        elif task == "niah_multikey":
            example = _gen_niah_multikey(tokenizer, rng, context_length, depth)
        elif task == "niah_multivalue":
            example = _gen_niah_multivalue(tokenizer, rng, context_length, depth)
        else:
            example = _gen_vt(tokenizer, rng, context_length, depth)

    example["task"] = task
    example["context_length"] = context_length
    example["depth_percent"] = depth_percent
    example["index"] = index
    return example


def cells_for_task(
    task: str,
    lengths: Sequence[int],
    depths: Sequence[int],
    n_per_cell: int,
) -> list[tuple[int, int | None, int]]:
    """Deterministic cell order ``(length, depth_or_None, index)`` for a task.

    ``cwe`` has a single ``None`` depth per length (its context has no needle
    position, so sweeping depth would duplicate identical cells).
    """
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {RULER_TASKS}")
    if n_per_cell < 1:
        raise ValueError(f"n_per_cell must be >= 1, got {n_per_cell}")
    task_depths: Sequence[int | None]
    task_depths = list(depths) if task in DEPTH_SWEPT_TASKS else [None]
    return [
        (length, depth, i)
        for length in lengths
        for depth in task_depths
        for i in range(n_per_cell)
    ]


def score_example(task: str, pred: str, answers: Sequence[str]) -> float:
    """Score a prediction; see the module docstring for per-task metrics."""
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {RULER_TASKS}")
    if not answers:
        raise ValueError(f"task {task!r}: empty answers")
    if task in ("niah_single", "niah_multikey"):
        if len(answers) != 1:
            raise ValueError(f"task {task!r} expects exactly one answer, got {answers}")
        return 1.0 if answers[0] in pred else 0.0
    if task == "niah_multivalue":
        return sum(1.0 for a in answers if a in pred) / len(answers)
    if task == "vt":
        # Canonical string_match_all over the chain's variable names:
        # case-insensitive substring containment, partial credit = fraction
        # of names matched.
        pred_lower = pred.lower()
        return sum(1.0 for a in answers if a.lower() in pred_lower) / len(answers)
    # cwe: case-insensitive whole-word set match with partial credit.
    pred_lower = pred.lower()
    hits = sum(
        1.0
        for w in answers
        if re.search(rf"\b{re.escape(w.lower())}\b", pred_lower) is not None
    )
    return hits / len(answers)
