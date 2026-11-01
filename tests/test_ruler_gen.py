"""CPU tests for the RULER-style synthetic task generators and scorers.

No model or GPU: a whitespace-token stub tokenizer stands in for the real
one, so these tests cover generator determinism, needle placement/presence,
VT chain semantics, CWE frequency guarantees, token-budget accuracy, and
scorer correctness on hand-built cases.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from lagged_indexer_dsa.ruler_gen import (
    CWE_BACKGROUND_POOL,
    CWE_TARGET_POOL,
    DEFAULT_DEPTHS,
    DEPTH_SWEPT_TASKS,
    GEN_TOKEN_SLACK,
    MAX_NEW_BY_TASK,
    METRIC_BY_TASK,
    RULER_TASKS,
    VT_HOPS,
    cells_for_task,
    generate_example,
    score_example,
)


class WhitespaceStubTokenizer:
    """Whitespace tokenizer: one token per whitespace-separated word."""

    def __init__(self) -> None:
        self._id_by_word: dict[str, int] = {}
        self._word_by_id: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._id_by_word:
                self._id_by_word[word] = len(self._word_by_id)
                self._word_by_id.append(word)
            ids.append(self._id_by_word[word])
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(self._word_by_id[i] for i in ids)


_LENGTH = 4096


def _gen(task: str, *, depth: int | None = 50, seed: int = 0, index: int = 0,
         length: int = _LENGTH, tokenizer: WhitespaceStubTokenizer | None = None):
    tok = tokenizer if tokenizer is not None else WhitespaceStubTokenizer()
    return tok, generate_example(
        task, tok,
        context_length=length,
        depth_percent=depth if task in DEPTH_SWEPT_TASKS else None,
        seed=seed, index=index,
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", RULER_TASKS)
def test_determinism_same_seed(task: str) -> None:
    # Fresh tokenizers per call: vocab ids are assigned in generation order,
    # so identical ids across instances proves identical generation.
    _, ex_a = _gen(task, seed=7)
    _, ex_b = _gen(task, seed=7)
    assert ex_a["ids"] == ex_b["ids"]
    assert ex_a["answers"] == ex_b["answers"]
    assert ex_a["meta"] == ex_b["meta"]


@pytest.mark.parametrize("task", RULER_TASKS)
def test_different_seed_changes_example(task: str) -> None:
    tok_a, ex_a = _gen(task, seed=0)
    tok_b, ex_b = _gen(task, seed=1)
    text_a = tok_a.decode(ex_a["ids"])
    text_b = tok_b.decode(ex_b["ids"])
    assert text_a != text_b


@pytest.mark.parametrize("task", RULER_TASKS)
def test_different_index_changes_example(task: str) -> None:
    tok_a, ex_a = _gen(task, index=0)
    tok_b, ex_b = _gen(task, index=1)
    assert tok_a.decode(ex_a["ids"]) != tok_b.decode(ex_b["ids"])


# ---------------------------------------------------------------------------
# Needle presence
# ---------------------------------------------------------------------------

def test_niah_single_needle_present_exactly_once() -> None:
    tok, ex = _gen("niah_single")
    text = tok.decode(ex["ids"])
    word = ex["meta"]["key_word"]
    number = ex["answers"][0]
    needle = f"The special magic number for {word} is {number}."
    assert text.count(needle) == 1
    assert text.count(number) == 1


def test_niah_multikey_three_needles_each_once() -> None:
    tok, ex = _gen("niah_multikey")
    text = tok.decode(ex["ids"])
    words = [ex["meta"]["key_word"]] + ex["meta"]["distractor_words"]
    numbers = [ex["answers"][0]] + ex["meta"]["distractor_numbers"]
    assert len(set(words)) == 3
    assert len(set(numbers)) == 3
    for word, number in zip(words, numbers):
        needle = f"The special magic number for {word} is {number}."
        assert text.count(needle) == 1
    # The queried key's needle carries the gold answer.
    match = re.search(
        rf"The special magic number for {ex['meta']['key_word']} is (\d+)\.",
        text,
    )
    assert match is not None
    assert match.group(1) == ex["answers"][0]


def test_niah_multivalue_four_values_each_once() -> None:
    tok, ex = _gen("niah_multivalue")
    text = tok.decode(ex["ids"])
    word = ex["meta"]["key_word"]
    assert len(ex["answers"]) == 4
    assert len(set(ex["answers"])) == 4
    for number in ex["answers"]:
        needle = f"The special magic number for {word} is {number}."
        assert text.count(needle) == 1
    assert text.count(f"special magic number for {word} is") == 4


# ---------------------------------------------------------------------------
# Depth placement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", DEFAULT_DEPTHS)
@pytest.mark.parametrize("task", sorted(DEPTH_SWEPT_TASKS))
def test_depth_placement_within_tolerance(task: str, depth: int) -> None:
    _, ex = _gen(task, depth=depth)
    # primary_frac is the realised start of the depth-swept needle as a
    # fraction of the full prompt; preamble/needle offsets shift it by well
    # under 5 points at 4096 tokens.
    assert abs(ex["meta"]["primary_frac"] - depth / 100.0) < 0.05


def test_cwe_rejects_depth_and_swept_tasks_require_it() -> None:
    tok = WhitespaceStubTokenizer()
    with pytest.raises(ValueError, match="no needle depth"):
        generate_example("cwe", tok, context_length=_LENGTH,
                         depth_percent=50, seed=0, index=0)
    with pytest.raises(ValueError, match="requires depth_percent"):
        generate_example("niah_single", tok, context_length=_LENGTH,
                         depth_percent=None, seed=0, index=0)


def test_unknown_task_rejected() -> None:
    tok = WhitespaceStubTokenizer()
    with pytest.raises(ValueError, match="Unknown task"):
        generate_example("niah_bogus", tok, context_length=_LENGTH,
                         depth_percent=50, seed=0, index=0)


def test_too_small_context_rejected() -> None:
    tok = WhitespaceStubTokenizer()
    with pytest.raises(ValueError, match="too small"):
        generate_example("niah_single", tok, context_length=128,
                         depth_percent=50, seed=0, index=0)


# ---------------------------------------------------------------------------
# Variable tracking
# ---------------------------------------------------------------------------

def test_vt_chain_answer_correctness() -> None:
    tok, ex = _gen("vt", depth=25, seed=3)
    text = tok.decode(ex["ids"])
    names = ex["meta"]["chain_names"]
    assert len(names) == VT_HOPS + 1
    assert len(set(names)) == VT_HOPS + 1

    assignments = re.findall(r"The value of variable ([A-Z]{4}) is (\d+)\.", text)
    copies = re.findall(
        r"The value of variable ([A-Z]{4}) equals the value of variable ([A-Z]{4})\.",
        text,
    )
    assert len(assignments) == 1
    assert len(copies) == VT_HOPS

    # Canonical vt direction: the gold answers are ALL chain variable names.
    assert ex["answers"] == names

    # Resolve the chain independently from the parsed statements: the value
    # is assigned to the first name and copied forward hop by hop.
    assert assignments[0] == (names[0], ex["meta"]["value"])
    dest_by_source = {src: dest for dest, src in copies}
    chain = [names[0]]
    while chain[-1] in dest_by_source:
        chain.append(dest_by_source[chain[-1]])
    assert chain == names

    # The query gives the assigned value (not a variable name) and asks for
    # the variable names.
    assert f"Find all variables that are assigned the value {ex['meta']['value']}" in text


# ---------------------------------------------------------------------------
# Common-words extraction
# ---------------------------------------------------------------------------

def test_cwe_targets_are_strictly_most_frequent() -> None:
    tok, ex = _gen("cwe")
    targets = ex["answers"]
    assert len(targets) == 10
    assert len(set(targets)) == 10
    assert set(targets) <= set(CWE_TARGET_POOL)

    words = tok.decode(ex["ids"]).split()
    target_counts = Counter(w for w in words if w in set(targets))
    background_counts = Counter(w for w in words if w in set(CWE_BACKGROUND_POOL))
    assert set(target_counts) == set(targets)
    assert min(target_counts.values()) > max(background_counts.values())
    # The 10 most common list-words really are the targets.
    combined = target_counts + background_counts
    top10 = {w for w, _ in combined.most_common(10)}
    assert top10 == set(targets)


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", [4096, 8192])
@pytest.mark.parametrize("task", RULER_TASKS)
def test_token_length_within_2pct(task: str, length: int) -> None:
    _, ex = _gen(task, length=length)
    assert abs(len(ex["ids"]) - length) / length <= 0.02
    # Prompt plus the task's generation budget never overflows the nominal
    # context length (the invariant the KV-cache sizing relies on).
    assert len(ex["ids"]) + MAX_NEW_BY_TASK[task] <= length
    assert MAX_NEW_BY_TASK[task] <= GEN_TOKEN_SLACK


# ---------------------------------------------------------------------------
# Cell grid
# ---------------------------------------------------------------------------

def test_cells_for_task_grid_shape_and_order() -> None:
    lengths = [4096, 8192]
    depths = [10, 50, 90]
    cells = cells_for_task("niah_single", lengths, depths, 2)
    assert len(cells) == len(lengths) * len(depths) * 2
    assert cells[0] == (4096, 10, 0)
    assert cells[1] == (4096, 10, 1)
    assert cells[-1] == (8192, 90, 1)
    # cwe collapses the depth axis to a single None cell per length.
    cwe_cells = cells_for_task("cwe", lengths, depths, 2)
    assert cwe_cells == [(4096, None, 0), (4096, None, 1),
                         (8192, None, 0), (8192, None, 1)]
    with pytest.raises(ValueError, match="n_per_cell"):
        cells_for_task("cwe", lengths, depths, 0)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def test_scorer_substring_tasks() -> None:
    for task in ("niah_single", "niah_multikey"):
        assert score_example(task, "The answer is 1234567.", ["1234567"]) == 1.0
        assert score_example(task, "The answer is 7654321.", ["1234567"]) == 0.0
        assert score_example(task, "", ["1234567"]) == 0.0
    with pytest.raises(ValueError, match="exactly one answer"):
        score_example("niah_single", "x", ["1", "2"])


def test_scorer_vt_all_names_with_partial_credit() -> None:
    # Canonical string_match_all: all chain names required for full credit;
    # partial credit is the fraction of names contained (case-insensitive).
    answers = ["ABCD", "EFGH", "IJKL", "MNOP", "QRST"]
    assert score_example("vt", "ABCD, EFGH, IJKL, MNOP, QRST", answers) == 1.0
    assert score_example("vt", "they are abcd and efgh", answers) == pytest.approx(0.4)
    assert score_example("vt", "MNOP", answers) == pytest.approx(0.2)
    assert score_example("vt", "no variables here", answers) == 0.0
    assert score_example("vt", "", answers) == 0.0


def test_scorer_multivalue_partial_credit() -> None:
    answers = ["1111111", "2222222", "3333333", "4444444"]
    assert score_example("niah_multivalue", "1111111 and 2222222", answers) == 0.5
    assert score_example(
        "niah_multivalue", "1111111, 2222222, 3333333, 4444444", answers
    ) == 1.0
    assert score_example("niah_multivalue", "nothing here", answers) == 0.0


def test_scorer_cwe_whole_word_set_match() -> None:
    answers = ["cat", "dog", "bird", "fish", "lion",
               "bear", "wolf", "deer", "frog", "crow"]
    # "catalog" must not count as "cat" (whole-word boundary).
    assert score_example("cwe", "catalog and dogma", answers) == 0.0
    # Case-insensitive, punctuation-separated hits count.
    pred = "Cat, dog, bird; fish lion bear wolf... but nothing else"
    assert score_example("cwe", pred, answers) == pytest.approx(0.7)
    assert score_example("cwe", " ".join(answers), answers) == 1.0


def test_scorer_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        score_example("niah_bogus", "x", ["y"])
    with pytest.raises(ValueError, match="empty answers"):
        score_example("cwe", "x", [])


def test_metric_and_max_new_registries_cover_all_tasks() -> None:
    assert set(METRIC_BY_TASK) == set(RULER_TASKS)
    assert set(MAX_NEW_BY_TASK) == set(RULER_TASKS)
