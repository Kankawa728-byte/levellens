"""A tiny made-up dataset for tests and smoke runs. It needs no download.

    python -m levellens.run data=toy model=majority

Three levels. Harder questions are longer and use rarer words, with some noise,
so a real model can learn something but cannot be perfect.
"""

from __future__ import annotations

import random
from functools import lru_cache

from levellens.data.schema import DatasetInfo, make_item
from levellens.data.splits import stratified_subset

INFO = DatasetInfo(name="toy", level_names=("easy", "medium", "hard"), trained=True, source="generated in levellens/data/toy.py")

SIZES = {"train": 240, "dev": 60, "test": 90}
WORDS = (
    ("cat", "dog", "sun", "red", "big", "run", "eat", "day", "cup", "hat"),
    ("planet", "energy", "method", "volume", "history", "compare", "measure", "pattern", "surface", "balance"),
    ("equilibrium", "hypothesis", "derivative", "photosynthesis", "inference", "asymptote", "catalyst", "paradigm", "entropy", "eigenvalue"),
)
NOISE = 0.25  # chance that a word is drawn from a neighbouring level
SEED = 0


def _question(rng: random.Random, level: int) -> tuple[str, list[str]]:
    """Make one question and its four options for a level."""
    def word() -> str:
        """Draw one word of the level or, with probability NOISE, of a neighbouring level."""
        source = level
        if rng.random() < NOISE:
            source = min(2, max(0, level + rng.choice((-1, 1))))
        return rng.choice(WORDS[source])

    stem = " ".join(word() for _ in range(4 + 3 * level + rng.randrange(3)))
    return f"Which answer fits: {stem}?", [word() for _ in range(4)]


@lru_cache(maxsize=1)
def generate() -> dict[str, list[dict]]:
    """Return the train, dev, test and test_1k items. The same items on every machine."""
    rng = random.Random(SEED)
    splits = {}
    for split, size in SIZES.items():
        items = []
        for index in range(size):
            level = index % 3
            question, options = _question(rng, level)
            items.append(
                make_item(
                    id=f"toy-{split}-{index:03d}", info=INFO, split=split,
                    question=question, options=options, level=level, meta={"answer": 0},
                )
            )
        splits[split] = items
    splits["test_1k"] = stratified_subset(splits["test"], 1000)
    return splits
