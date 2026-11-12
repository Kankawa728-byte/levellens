"""The results store: one line per run in `results/runs.jsonl`, and the tables built from it.

    python -m levellens.results        # rebuild every table in results/tables/

Write rows through `append_run` only. Never edit `runs.jsonl` by hand.
If a run is repeated with the same `run_id`, the newest row counts.
Predictions of each run are saved next to it in `results/preds/<run_id>.npz`, so figures,
bootstrap intervals and the error audit do not need the model again.

Runs on the toy dataset are try-outs. `levellens.run` keeps them in a store of their own,
`results/scratch/`, which git ignores. Every function here takes that folder as `folder=`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from levellens.paths import results_dir, scratch_dir

ROW_FIELDS = (
    "run_id", "started", "owner", "tier", "dataset", "model", "head", "variant", "seed", "config",
    "metrics", "train_seconds", "predict_seconds", "device", "n_params", "git_commit",
)
GROUP = ["model", "head", "variant"]  # runs that differ only in the seed are averaged
# column in a table -> (metric key, higher is better)
TABLE_METRICS = {
    "Acc": "accuracy", "F1": "macro_f1", "Adj": "adjacent_accuracy", "QWK": "qwk", "MAE": "mae",
    "RMSE": "rmse", "Spearman": "spearman", "Bal. DRPS": "balanced_drps", "ECE": "ece",
}
TABLE_SPLITS = ("test", "test_1k", "dev")
SORT_BY = "qwk"


def _store(folder: Path | None) -> Path:
    """Return the folder of a results store: the given one, or the shared results folder."""
    return Path(folder) if folder else results_dir()


def runs_file(folder: Path | None = None) -> Path:
    """Return the path of `runs.jsonl`."""
    return _store(folder) / "runs.jsonl"


def append_run(row: dict, folder: Path | None = None) -> None:
    """Check a result row and append it as one line to `runs.jsonl`."""
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Result row {row.get('run_id')!r} lacks the fields {missing}")
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)  # NaN is not JSON: store None instead
    path = runs_file(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def load_rows(folder: Path | None = None) -> list[dict]:
    """Return the rows of `runs.jsonl`, the newest row of each `run_id` only, in file order."""
    path = runs_file(folder)
    if not path.exists():
        return []
    latest = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {number} is not valid JSON: {error}") from error
            latest.pop(row["run_id"], None)
            latest[row["run_id"]] = row
    return list(latest.values())


def load_runs(folder: Path | None = None) -> pd.DataFrame:
    """Return the runs as a table: one row per run, metrics as columns such as `metrics.test.qwk`."""
    rows = [{key: value for key, value in row.items() if key != "config"} for row in load_rows(folder)]
    return pd.json_normalize(rows, sep=".") if rows else pd.DataFrame(columns=[f for f in ROW_FIELDS if f != "config"])


def predictions_file(run_id: str, folder: Path | None = None) -> Path:
    """Return the path of the saved predictions of one run."""
    return _store(folder) / "preds" / f"{run_id}.npz"


def save_predictions(run_id: str, predictions: dict[str, dict], folder: Path | None = None) -> Path:
    """Save the predictions of one run.

    `predictions` maps a split to {"ids": [n] str, "y": [n] int, "output": [n, K] or [n] float}.
    """
    arrays = {}
    for split, parts in predictions.items():
        arrays[f"{split}__ids"] = np.asarray(parts["ids"], dtype=str)
        arrays[f"{split}__y"] = np.asarray(parts["y"], dtype=np.int16)
        arrays[f"{split}__output"] = np.asarray(parts["output"], dtype=np.float32)
    path = predictions_file(run_id, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_predictions(run_id: str, folder: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Return what `save_predictions` stored: split -> {"ids", "y", "output"}."""
    predictions = {}
    with np.load(predictions_file(run_id, folder)) as arrays:
        for key in arrays.files:
            split, part = key.split("__")
            predictions.setdefault(split, {})[part] = arrays[key]
    return predictions


def _mean_std(values: pd.Series) -> str:
    """Format the values of several seeds as "0.861 ± 0.003", or as one number for one seed."""
    values = values.dropna().astype(float)
    if values.empty:
        return "n/a"
    if len(values) == 1:
        return f"{values.iloc[0]:.3f}"
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def leaderboard(runs: pd.DataFrame, dataset: str, split: str = "test") -> pd.DataFrame:
    """Return the table of one dataset and split: one line per model, head and variant, best QWK first."""
    runs = runs[runs["dataset"] == dataset]
    prefix = f"metrics.{split}."  # load_runs names the columns metrics.<split>.<metric>
    sort_column = prefix + SORT_BY
    if runs.empty or prefix + "n" not in runs.columns:
        return pd.DataFrame()
    runs = runs[runs[prefix + "n"].notna()]
    lines = []
    for (model, head, variant), group in runs.groupby(GROUP, dropna=False, sort=False):
        line = {"Model": model, "Head": head, "Variant": variant, "Seeds": len(group)}
        for title, metric in TABLE_METRICS.items():
            line[title] = _mean_std(group.get(prefix + metric, pd.Series(dtype=float)))
        line["Train s"] = f"{group['train_seconds'].mean():.0f}"
        line["Params"] = f"{int(group['n_params'].max()):,}"
        line["_sort"] = group[sort_column].astype(float).mean() if sort_column in group else np.nan
        lines.append(line)
    table = pd.DataFrame(lines)
    if table.empty:
        return table
    return table.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort").reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    """Return a table as GitHub Markdown."""
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def build_tables(folder: Path | None = None) -> list[Path]:
    """Rebuild every table from `runs.jsonl`. Return the files written.

    The tables go to `tables/` in the store. For each dataset and split: `<dataset>_<split>.csv`.
    All of them together: `leaderboards.md`.
    """
    runs = load_runs(folder)
    folder = _store(folder) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    note = f"Built from `runs.jsonl`: {len(runs)} runs. Do not edit; run `python -m levellens.results`."
    written, sections = [], ["# Leaderboards", "", note]
    for dataset in sorted(runs["dataset"].unique()):
        for split in TABLE_SPLITS:
            table = leaderboard(runs, dataset, split)
            if table.empty:
                continue
            path = folder / f"{dataset}_{split}.csv"
            table.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
            written.append(path)
            sections += ["", f"## {dataset}, {split}", "", to_markdown(table)]
    path = folder / "leaderboards.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8", newline="\n")
    return written + [path]


def main(argv: list[str] | None = None) -> None:
    """Rebuild every table and print where they are."""
    parser = argparse.ArgumentParser(description="Rebuild every table from runs.jsonl")
    parser.add_argument("--scratch", action="store_true", help="use the try-out store of the toy dataset")
    args = parser.parse_args(argv)
    for path in build_tables(scratch_dir() if args.scratch else None):
        print(path)


if __name__ == "__main__":
    main()
