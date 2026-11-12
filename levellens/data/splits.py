"""Deterministic subsets, split files and their fingerprints.

The subsets here do not use a random number generator. An item is ranked by a hash
of the seed and its id, so the same items are picked on every machine and with every
library version. That is how "the splits are saved once" holds for the whole team.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Hashable, Sequence


def _rank(seed: int, item_id: str) -> str:
    """Return the sort key of an item: a hash of the seed and the item id."""
    return hashlib.sha1(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def _by_level(item: dict) -> Hashable:
    return item["level"]


def stratified_subset(
    items: Sequence[dict],
    n: int,
    seed: int = 0,
    key: Callable[[dict], Hashable] = _by_level,
) -> list[dict]:
    """Pick `n` items so that every group keeps its share of the whole.

    The groups are the values of `key`, by default the level. Seats are given to the
    groups by the largest-remainder rule. Inside a group the items with the smallest
    hash rank are taken. The result is sorted by id.
    """
    if n >= len(items):
        return sorted(items, key=lambda item: item["id"])
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)

    exact = {group: n * len(members) / len(items) for group, members in groups.items()}
    seats = {group: int(share) for group, share in exact.items()}
    by_remainder = sorted(groups, key=lambda group: (-(exact[group] - seats[group]), str(group)))
    for group in by_remainder[: n - sum(seats.values())]:
        seats[group] += 1

    picked = []
    for group, members in groups.items():
        members = sorted(members, key=lambda item: _rank(seed, item["id"]))
        picked += members[: seats[group]]
    return sorted(picked, key=lambda item: item["id"])


def capped_per_level(items: Sequence[dict], caps: dict[int, int], seed: int = 0) -> list[dict]:
    """Keep at most `caps[level]` items of each listed level and every item of the other levels."""
    groups = defaultdict(list)
    for item in items:
        groups[item["level"]].append(item)
    picked = []
    for level, members in groups.items():
        members = sorted(members, key=lambda item: _rank(seed, item["id"]))
        picked += members[: caps.get(level, len(members))]
    return sorted(picked, key=lambda item: item["id"])


def write_jsonl(path: Path, items: Sequence[dict]) -> None:
    """Write one JSON object per line, UTF-8, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a file written by `write_jsonl`."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fingerprint(items: Sequence[dict], num_levels: int) -> dict:
    """Return the size, the count per level and a hash of the ids, levels and texts of a split.

    Two machines hold the same split exactly when their fingerprints are equal.
    """
    lines = sorted(f"{item['id']}\t{item['level']}\t{item['text']}" for item in items)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    counts = Counter(item["level"] for item in items)
    return {
        "n": len(items),
        "per_level": [counts.get(level, 0) for level in range(num_levels)],
        "sha256": digest,
    }
