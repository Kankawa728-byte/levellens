"""RACE++ adapter: RACE middle and high, plus RACE-C (college). English reading, 3 levels.

The level of a question is the exam stage of its source: middle school, high school
or college. The published train, dev and test splits are kept, so no passage appears
in two splits. The passage is stored but is not part of the model input.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from levellens.data.schema import DatasetInfo, make_item

INFO = DatasetInfo(
    name="race",
    level_names=("middle school", "high school", "college"),
    trained=True,
    source="ehovy/race (middle, high) + tasksource/race-c",
)

RACE_REPO, RACE_REVISION = "ehovy/race", "2fec9fd81f1dc971569a9b729c43f2f0e6436637"
RACE_C_REPO, RACE_C_REVISION = "tasksource/race-c", "99b95125875f1ef0e47f89801cb435d14911fffa"
HUB_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def _items(rows, split: str, level: int, source: str, prefix: str) -> list[dict]:
    """Turn the rows of one source and split into items. `rows` yields (passage_id, passage, question, options, answer)."""
    items = []
    seen = Counter()  # one passage has several questions; number them in file order
    for passage_id, passage, question, options, answer in rows:
        passage_id = passage_id.removesuffix(".txt")
        number = seen[passage_id]
        seen[passage_id] += 1
        items.append(
            make_item(
                id=f"race-{split}-{prefix}{passage_id}-q{number}",
                info=INFO,
                split=split,
                question=question,
                options=options,
                passage=passage,
                level=level,
                meta={"source": source, "passage_id": passage_id, "answer": answer},
            )
        )
    return items


def build(raw_dir: Path) -> dict[str, list[dict]]:
    """Download RACE and RACE-C if needed and return the train, dev and test items."""
    from datasets import load_dataset

    cache = str(raw_dir / "hf")
    middle = load_dataset(RACE_REPO, "middle", revision=RACE_REVISION, cache_dir=cache)
    high = load_dataset(RACE_REPO, "high", revision=RACE_REVISION, cache_dir=cache)
    college = load_dataset(RACE_C_REPO, revision=RACE_C_REVISION, cache_dir=cache)

    splits = {}
    for split, hub_split in HUB_SPLITS.items():
        items = []
        for level, (data, source) in enumerate([(middle, "RACE-M"), (high, "RACE-H")]):
            rows = (
                (row["example_id"], row["article"], row["question"], row["options"], "ABCD".index(row["answer"]))
                for row in data[hub_split]
            )
            items += _items(rows, split, level, source, prefix="")
        rows = (
            (row["id"], row["article"], row["question"], row["option"], int(row["label"]))
            for row in college[hub_split]
        )
        items += _items(rows, split, 2, "RACE-C", prefix="college")
        splits[split] = items
    return splits
