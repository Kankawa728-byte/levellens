"""MATH adapter: competition maths problems, 5 levels (Level 1 to Level 5).

The model input is the problem text only. The worked solution is kept in `meta` for
interpretation and is never part of the input. The dataset has no dev split, so 10% of
the training problems are held out as dev, with the same mix of levels and problem types.
The file is called `math_` so that it never hides Python's own `math` module.
"""

from __future__ import annotations

from pathlib import Path

from levellens.data.schema import DatasetInfo, make_item
from levellens.data.splits import stratified_subset

INFO = DatasetInfo(
    name="math",
    level_names=("Level 1", "Level 2", "Level 3", "Level 4", "Level 5"),
    trained=True,
    source="EleutherAI/hendrycks_math",
)

REPO, REVISION = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
DEV_FRACTION = 0.10
DEV_SEED = 0


def build(raw_dir: Path) -> dict[str, list[dict]]:
    """Download MATH if needed and return the train, dev and test items.

    Problems labelled "Level ?" have no usable level and are dropped; the count is printed.
    """
    from datasets import load_dataset

    cache = str(raw_dir / "hf")
    pools = {"train": [], "test": []}
    dropped = 0
    for subject in SUBJECTS:
        data = load_dataset(REPO, subject, revision=REVISION, cache_dir=cache)
        for split in pools:
            for index, row in enumerate(data[split]):
                if row["level"] not in INFO.level_names:
                    dropped += 1
                    continue
                pools[split].append(
                    make_item(
                        id=f"math-{split}-{subject}-{index:04d}",
                        info=INFO,
                        split=split,
                        question=row["problem"],
                        level=INFO.level_names.index(row["level"]),
                        meta={"type": row["type"], "solution": row["solution"]},
                    )
                )
    print(f"math: dropped {dropped} problems without a level")

    dev = stratified_subset(
        pools["train"],
        n=round(DEV_FRACTION * len(pools["train"])),
        seed=DEV_SEED,
        key=lambda item: (item["level"], item["meta"]["type"]),
    )
    dev_ids = {item["id"] for item in dev}
    train = [item for item in pools["train"] if item["id"] not in dev_ids]
    for item in dev:
        item["split"] = "dev"
    return {"train": train, "dev": dev, "test": pools["test"]}
