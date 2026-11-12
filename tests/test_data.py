"""Item format, deterministic subsets, the toy dataset and the ARC option parser."""

from collections import Counter

import pytest

from levellens.data import info, load_split
from levellens.data.arc import split_options
from levellens.data.schema import ITEM_FIELDS, check_items, format_text, make_item
from levellens.data.splits import capped_per_level, fingerprint, read_jsonl, stratified_subset, write_jsonl


def test_format_text_puts_letters_before_options():
    assert format_text("Which is  a mammal?", ["frog", "whale"]) == "Which is a mammal? A. frog B. whale"
    assert format_text("Solve $x^2 = 4$.") == "Solve $x^2 = 4$."


def test_make_item_has_every_field_and_rejects_bad_level():
    toy = info("toy")
    item = make_item(id="a", info=toy, split="train", question="Q?", options=["x", "y"], level=2)
    assert tuple(item) == ITEM_FIELDS
    assert item["level_name"] == "hard" and item["text"] == "Q? A. x B. y"
    with pytest.raises(ValueError):
        make_item(id="b", info=toy, split="train", question="Q?", level=3)


def test_toy_splits_follow_the_format():
    toy = info("toy")
    sizes = {split: len(load_split("toy", split)) for split in ("train", "dev", "test", "test_1k")}
    assert sizes == {"train": 240, "dev": 60, "test": 90, "test_1k": 90}
    for split in sizes:
        check_items(load_split("toy", split), toy)
    assert load_split("toy", "train") == load_split("toy", "train")  # same items on every call


def test_check_items_finds_duplicates():
    items = load_split("toy", "dev")
    with pytest.raises(ValueError, match="duplicate"):
        check_items(items + items[:1], info("toy"))


def test_stratified_subset_is_proportional_and_ignores_order():
    items = [{"id": f"i{n:03d}", "level": 0 if n < 70 else 1 if n < 95 else 2} for n in range(100)]
    subset = stratified_subset(items, 20, seed=0)
    assert Counter(item["level"] for item in subset) == {0: 14, 1: 5, 2: 1}
    assert subset == stratified_subset(items[::-1], 20, seed=0)
    assert subset != stratified_subset(items, 20, seed=1)
    assert len(stratified_subset(items, 500)) == 100  # asking for more than there is returns everything


def test_capped_per_level_cuts_only_the_listed_levels():
    items = [{"id": f"i{n:03d}", "level": n % 2} for n in range(40)]
    capped = capped_per_level(items, {0: 5})
    assert Counter(item["level"] for item in capped) == {0: 5, 1: 20}


def test_jsonl_round_trip_and_fingerprint(tmp_path):
    items = load_split("toy", "dev")
    write_jsonl(tmp_path / "dev.jsonl", items)
    assert read_jsonl(tmp_path / "dev.jsonl") == items
    first = fingerprint(items, 3)
    assert first["n"] == 60 and first["per_level"] == [20, 20, 20]
    assert fingerprint(items[::-1], 3) == first
    assert fingerprint(items[1:], 3)["sha256"] != first["sha256"]


def test_arc_option_parser():
    assert split_options("What melts? (A) ice (B) rock (C) wood (D) sand") == ("What melts?", ["ice", "rock", "wood", "sand"])
    assert split_options("Pick one. (1) a (2) b (3) c (4) d") == ("Pick one.", ["a", "b", "c", "d"])
    assert split_options("Five options (A) a (B) b (C) c (D) d (E) e") is None
    assert split_options("Three options (A) a (B) b (C) c") is None
