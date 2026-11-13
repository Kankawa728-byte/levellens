"""Where data, caches, configs and results live.

No other module hard-codes a path. Three environment variables move the folders,
so the same code runs on a laptop and on a GPU machine:

    LEVELLENS_DATA     downloaded datasets and saved splits     default <repo>/data
    LEVELLENS_CACHE    model weights, features, checkpoints     default <repo>/cache
    LEVELLENS_RESULTS  runs.jsonl, predictions, tables, figures default <repo>/results
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _folder(variable: str, default: str) -> Path:
    """Return the folder named by an environment variable, or the default inside the repository."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else REPO_ROOT / default


def data_dir() -> Path:
    """Folder with the downloaded datasets (`raw/`) and the saved splits (`processed/`)."""
    return _folder("LEVELLENS_DATA", "data")


def cache_dir() -> Path:
    """Folder for model weights, tokenised data, feature caches and checkpoints."""
    return _folder("LEVELLENS_CACHE", "cache")


def results_dir() -> Path:
    """Folder with `runs.jsonl`, the saved predictions, the tables and the figures."""
    return _folder("LEVELLENS_RESULTS", "results")


def scratch_dir() -> Path:
    """Folder for try-out runs on the toy dataset. Same layout as the results folder; ignored by git."""
    return results_dir() / "scratch"


def shown(path: Path | str) -> str:
    """Return a path as the commands print it: relative to the repository when it is inside it."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def config_dir() -> Path:
    """Folder with `train.yaml` and the `data/`, `model/` and `head/` config groups."""
    return REPO_ROOT / "configs"
