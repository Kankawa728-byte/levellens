"""ScienceQA adapter, text-only questions: science, social and language questions for grades 1 to 12.

No model is trained on this dataset. It tests how far a model trained on ARC (grades 3 to 9)
extrapolates. Questions with an image are dropped. The level is the school grade minus 1.
"""

from __future__ import annotations

from pathlib import Path

from levellens.data.schema import DatasetInfo, make_item

INFO = DatasetInfo(
    name="scienceqa",
    level_names=tuple(f"grade {grade}" for grade in range(1, 13)),
    trained=False,
    source="derek-thomas/ScienceQA",
)

REPO, REVISION = "derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4"
HUB_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def build(raw_dir: Path) -> dict[str, list[dict]]:
    """Download ScienceQA if needed and return the text-only train, dev and test items."""
    from datasets import Image, load_dataset

    data = load_dataset(REPO, revision=REVISION, cache_dir=str(raw_dir / "hf"))
    splits = {}
    for split, hub_split in HUB_SPLITS.items():
        # read the image column without decoding it, then drop it: we only need to know if it is empty
        rows = data[hub_split].cast_column("image", Image(decode=False))
        has_image = [image is not None for image in rows["image"]]
        rows = rows.remove_columns("image")
        items = []
        for index, row in enumerate(rows):
            if has_image[index]:
                continue
            grade = int(row["grade"].removeprefix("grade"))
            items.append(
                make_item(
                    id=f"scienceqa-{split}-{index:05d}",
                    info=INFO,
                    split=split,
                    question=row["question"],
                    options=row["choices"],
                    level=grade - 1,
                    meta={
                        "grade": grade,
                        "subject": row["subject"],
                        "topic": row["topic"],
                        "category": row["category"],
                        "skill": row["skill"],
                        "hint": row["hint"],
                        "answer": int(row["answer"]),
                    },
                )
            )
        print(f"scienceqa {split}: kept {len(items)} text-only questions of {len(has_image)}")
        splits[split] = items
    return splits
