"""Load the saved splits. Every model reads its data through `load_split`.

    from levellens.data import info, load_split

    train = load_split("math", "train")     # list of items, see levellens/data/schema.py
    info("math").level_names                # ('Level 1', ..., 'Level 5')

The split files are made once by `python -m levellens.data.prepare`. Nobody re-splits.
Splits of every dataset: train, dev, test, test_1k. ARC also has train_balanced.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from levellens.data.schema import DatasetInfo, check_items, format_text, make_item
from levellens.data.splits import read_jsonl, stratified_subset
from levellens.paths import data_dir

__all__ = [
    "ADAPTERS", "DatasetInfo", "check_items", "format_text", "info", "load_frame", "load_split",
    "make_item", "split_path", "stratified_subset",
]

# dataset name -> module with INFO and build(), or INFO and generate() for the built-in toy set
ADAPTERS = {
    "race": "levellens.data.race",
    "arc": "levellens.data.arc",
    "math": "levellens.data.math_",
    "scienceqa": "levellens.data.scienceqa",
    "toy": "levellens.data.toy",
}


def adapter(name: str):
    """Return the adapter module of a dataset."""
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(ADAPTERS)}")
    return import_module(ADAPTERS[name])


def info(name: str) -> DatasetInfo:
    """Return the level names and the role of a dataset."""
    return adapter(name).INFO


def split_path(name: str, split: str) -> Path:
    """Return the file of one saved split."""
    return data_dir() / "processed" / name / f"{split}.jsonl"


def load_split(name: str, split: str) -> list[dict]:
    """Return one saved split as a list of items.

    The toy dataset is generated in memory. Every other dataset must be prepared first.
    """
    module = adapter(name)
    if hasattr(module, "generate"):
        splits = module.generate()
        if split not in splits:
            raise KeyError(f"{name} has no split {split!r}. Available: {sorted(splits)}")
        return splits[split]
    path = split_path(name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Prepare the dataset first: python -m levellens.data.prepare {name}"
        )
    return read_jsonl(path)


def load_frame(name: str, split: str):
    """Return one saved split as a pandas DataFrame, one row per item."""
    import pandas as pd

    return pd.DataFrame(load_split(name, split))
