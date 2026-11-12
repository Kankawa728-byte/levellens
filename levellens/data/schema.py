"""The one item format that every dataset adapter returns.

| Field      | Type        | Meaning                                                        |
|------------|-------------|----------------------------------------------------------------|
| id         | str         | Unique inside the dataset                                      |
| dataset    | str         | race, arc, math, scienceqa, toy                                |
| split      | str         | train, dev or test                                             |
| question   | str         | The question text                                              |
| options    | list of str | The answer options; empty for MATH                             |
| passage    | str or None | The reading passage (RACE++ only); not part of `text`          |
| text       | str         | The model input: the question plus all options                 |
| level      | int         | 0 to K - 1, ordered from the lowest level to the highest       |
| level_name | str         | For example "middle school", "grade 5", "Level 3"              |
| meta       | dict        | Extra fields, for example the MATH problem type                |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

ITEM_FIELDS = (
    "id", "dataset", "split", "question", "options", "passage", "text", "level", "level_name", "meta",
)
OPTION_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class DatasetInfo:
    """What the rest of the code needs to know about one dataset."""

    name: str
    level_names: tuple[str, ...]
    trained: bool  # False for the extrapolation sets, which no model is trained on
    source: str

    @property
    def num_levels(self) -> int:
        """K, the number of ordered levels."""
        return len(self.level_names)


def format_text(question: str, options: Sequence[str] = ()) -> str:
    """Return the model input: the question, then every option with its letter.

    >>> format_text("Which is a mammal?", ["frog", "whale"])
    'Which is a mammal? A. frog B. whale'
    """
    parts = [" ".join(question.split())]
    parts += [f"{OPTION_LETTERS[i]}. {' '.join(option.split())}" for i, option in enumerate(options)]
    return " ".join(parts)


def make_item(
    *,
    id: str,
    info: DatasetInfo,
    split: str,
    question: str,
    level: int,
    options: Sequence[str] = (),
    passage: str | None = None,
    meta: dict | None = None,
) -> dict:
    """Build one item in the shared format and check its level."""
    if not 0 <= level < info.num_levels:
        raise ValueError(f"{info.name} item {id}: level {level} is outside 0..{info.num_levels - 1}")
    options = [str(option) for option in options]
    return {
        "id": str(id),
        "dataset": info.name,
        "split": split,
        "question": question,
        "options": options,
        "passage": passage,
        "text": format_text(question, options),
        "level": int(level),
        "level_name": info.level_names[level],
        "meta": dict(meta or {}),
    }


def check_items(items: Iterable[dict], info: DatasetInfo) -> None:
    """Raise ValueError if any item breaks the shared format: fields, empty text, level range, duplicate ids."""
    seen = set()
    for item in items:
        missing = [field for field in ITEM_FIELDS if field not in item]
        if missing:
            raise ValueError(f"{info.name} item {item.get('id')}: missing fields {missing}")
        if not item["text"].strip():
            raise ValueError(f"{info.name} item {item['id']}: empty text")
        if not 0 <= item["level"] < info.num_levels:
            raise ValueError(f"{info.name} item {item['id']}: level {item['level']} out of range")
        if item["id"] in seen:
            raise ValueError(f"{info.name}: duplicate id {item['id']}")
        seen.add(item["id"])
