"""Run one experiment from a config and write one line to `results/runs.jsonl`.

    python -m levellens.run data=toy model=majority
    python -m levellens.run data=arc model=majority seed=1 owner=D
    python -m levellens.run data=race model=<name> head=<name> limit_train=20000 save_model=true

The settings are explained in `configs/train.yaml` and in `levellens/config.py`.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from omegaconf import DictConfig

from levellens.config import compose, to_dict
from levellens.data import info, load_split, stratified_subset
from levellens.evaluate import evaluate
from levellens.models.base import build_model
from levellens.paths import REPO_ROOT, results_dir, scratch_dir, shown
from levellens.predictor import save_for_predictor
from levellens.results import append_run, save_predictions

SINGAPORE = timezone(timedelta(hours=8))
SUBSET = "test_1k"
SCRATCH_DATASETS = ("toy",)  # try-out runs: their rows never go into the shared results store


def resolve_device(requested: str) -> str:
    """Return the device to use: "cpu", "cuda" or "cuda:0".

    The environment variable LEVELLENS_DEVICE wins over the config. "auto" means the GPU
    when PyTorch sees one, and the CPU otherwise.
    """
    requested = os.environ.get("LEVELLENS_DEVICE", requested)
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # the classical models need no PyTorch
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> str:
    """Return the device with the name of the GPU, for the efficiency table."""
    if device.startswith("cuda"):
        import torch

        return f"{device} ({torch.cuda.get_device_name(torch.device(device))})"
    return device


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and, when it is installed, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Return the short hash of the current commit, with "+dirty" if files have changed since."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"  # no git on this machine, or the code was copied without its history
    return commit + ("+dirty" if dirty else "")


def variant_label(config: DictConfig) -> str:
    """Return the label that tells apart runs of the same dataset, model, head and seed."""
    parts = [str(config.variant)] if config.variant else []
    if config.data.train_split != "train":
        parts.append(str(config.data.train_split).removeprefix("train_"))
    if config.limit_train:
        parts.append(f"n{config.limit_train}")
    return "-".join(parts)


def run(config: DictConfig) -> dict:
    """Train and evaluate one model. Append the result row, save the predictions, return the row."""
    dataset = info(config.data.name)
    config.device = resolve_device(config.device)
    set_seed(config.seed)

    train = load_split(dataset.name, config.data.train_split)
    dev = load_split(dataset.name, "dev")
    if config.limit_train:
        train = stratified_subset(train, config.limit_train, seed=config.seed)

    variant = variant_label(config)
    run_id = config.run_id or "-".join(
        str(part) for part in (dataset.name, config.model.name, config.head.name, variant, f"s{config.seed}") if part != ""
    )
    print(f"{run_id}: {len(train)} training items, {dataset.num_levels} levels, device {config.device}")

    model = build_model(config, dataset.num_levels)
    started = datetime.now(SINGAPORE).isoformat(timespec="seconds")
    clock = time.perf_counter()
    model.fit(train, dev, config)
    train_seconds = time.perf_counter() - clock

    metrics, predictions, predict_seconds = {}, {}, {}
    for split in config.eval_splits:
        items = load_split(dataset.name, split)
        if split == SUBSET and "test" in predictions:
            # the subset is part of test: reuse the test predictions instead of running the model again
            row_of = {item_id: row for row, item_id in enumerate(predictions["test"]["ids"])}
            rows = [row_of[item["id"]] for item in items]
            y, output = predictions["test"]["y"][rows], predictions["test"]["output"][rows]
        else:
            clock = time.perf_counter()
            output = np.asarray(model.predict_proba(items))
            predict_seconds[split] = round(time.perf_counter() - clock, 3)
            y = np.array([item["level"] for item in items])
            predictions[split] = {"ids": [item["id"] for item in items], "y": y, "output": output}
        metrics[split] = evaluate(y, output, dataset.num_levels)

    row = {
        "run_id": run_id,
        "started": started,
        "owner": str(config.owner),
        "tier": int(config.tier),
        "dataset": dataset.name,
        "model": str(config.model.name),
        "head": str(config.head.name),
        "variant": variant,
        "seed": int(config.seed),
        "config": to_dict(config),
        "metrics": metrics,
        "train_seconds": round(train_seconds, 3),
        "predict_seconds": predict_seconds,
        "device": describe_device(config.device),
        "n_params": int(model.n_params),
        "git_commit": git_commit(),
    }
    store = scratch_dir() if dataset.name in SCRATCH_DATASETS else results_dir()
    append_run(row, store)
    if config.save_predictions:
        save_predictions(run_id, predictions, store)
    if config.save_model:
        print(f"{run_id}: model saved to {shown(save_for_predictor(model, run_id, dataset.name))}")

    headline = next((split for split in ("test", SUBSET, "dev") if split in metrics), None)
    if headline:
        summary = ", ".join(
            f"{name} {metrics[headline][name]:.3f}" for name in ("accuracy", "qwk", "rmse", "balanced_drps")
            if metrics[headline][name] is not None
        )
        print(f"{run_id}: {headline} {summary}; trained in {train_seconds:.1f} s")
    print(f"{run_id}: row written to {shown(store / 'runs.jsonl')}")
    return row


def main(argv: list[str] | None = None) -> None:
    """Run the command line: every argument is a `key=value` override."""
    run(compose(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
