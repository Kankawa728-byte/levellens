"""Independent small examples for the Share B counting and sampling logic."""

import importlib.util
from pathlib import Path

import pytest

from levellens.data import info, make_item

SPEC = importlib.util.spec_from_file_location("data_checks", Path(__file__).parents[1] / "notebooks/data_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def item(id, text="Same?", level=0, split="train"):
    return make_item(id=id, info=info("toy"), question=text, level=level, split=split)


def test_exact_duplicates_count_pairs_and_label_conflicts():
    splits = {"train": [item("a"), item("b"), item("c", "Same? ")],
              "test": [item("d", level=1, split="test"), item("e", split="test")]}
    result = checks.duplicate_summary(splits, "question")
    assert result["within"]["train"] == {"groups": 1, "rows": 2, "extra_rows": 1, "conflicting_level_groups": 0}
    assert result["across"]["train:test"] == {"groups": 1, "rows_left": 2, "rows_right": 2, "pairs": 4, "conflicting_level_groups": 1}
    assert checks.duplicate_summary(splits, "text")["across"]["train:test"]["pairs"] == 6


def test_length_units_and_absent_level():
    result = checks.length_summary([item("a", "one two"), item("b", "x")], 2)
    assert result[0]["question_words"]["mean"] == 1.5
    assert result[0]["question_chars"]["median"] == 4.0
    assert result[1]["n"] == 0
    assert result[1]["question_words"]["mean"] is None


def test_review_sample_is_per_level_and_order_invariant():
    rows = [item(str(i), level=i % 3) for i in range(37)]
    picked = checks.review_sample(rows, 3, 10)
    assert len(picked) == 30
    assert picked == checks.review_sample(rows[::-1], 3, 10)
    assert all(sum(r["level"] == k for r in picked) == 10 for k in range(3))
    assert checks.review_sample(rows[:2], 3, 10) == rows[:2]
    with pytest.raises(ValueError):
        checks.review_sample(rows, 3, 0)


def test_validate_rejects_nulls_bad_input_and_empty_data():
    assert checks.validate([item("a")], "toy", "train")["shape"] == [1, 10]
    for bad in [{**item("a"), "question": None}, {**item("a"), "text": None}, {**item("a"), "text": "changed"}, {**item("a"), "split": "test"}]:
        with pytest.raises(ValueError):
            checks.validate([bad], "toy", "train")
    with pytest.raises(ValueError):
        checks.validate([], "toy", "train")
