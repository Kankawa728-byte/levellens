"""Where a model is wrong: the confusion matrix, and the worst errors for the human audit (task F2).

    python -m levellens.explain.errors                        # the best run of every dataset, by dev QWK
    python -m levellens.explain.errors race-distilbert-corn-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0

Everything here reads the saved predictions in `results/preds/<run_id>.npz`. No model is loaded.

The worst errors of a run are its test questions ordered by
1. the gap between the predicted and the labelled level, largest first,
2. the confidence of the model (the probability of the level it predicted), highest first,
3. the question id, so that the choice is reproducible.

The command writes two files, as docs/audit.md asks:
    results/audit/worst_errors.csv          ids, levels, probabilities and the hash of the prediction file
    <LEVELLENS_CACHE>/audit/worst_errors.md the full questions for the reviewer. The cache folder is not in
                                            git: RACE and ScienceQA texts must not be republished.

Picking questions from the test set is error analysis after the fact. Nothing is tuned on them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from levellens.data import info, load_split
from levellens.evaluate import interpret
from levellens.paths import cache_dir, results_dir
from levellens.results import load_predictions, load_rows, predictions_file

ERRORS_PER_DATASET = 5  # 15 in all, cut from more on Thu 12 Nov
SPLITS = ("test", "test_1k")  # use the full test set; a smoke run may only have the 1k subset


def run_row(run_id: str) -> dict:
    """Return the result row of a run."""
    for row in load_rows():
        if row["run_id"] == run_id:
            return row
    raise KeyError(f"No run {run_id!r} in runs.jsonl")


def scored_split(run_id: str, split: str | None = None) -> dict:
    """Return the saved predictions of a run on one split, with the predicted level and the confidence.

    Keys: dataset, split, level_names, ids, y, level, proba, confidence.
    """
    dataset = run_row(run_id)["dataset"]
    stored = load_predictions(run_id)
    split = split or next((name for name in SPLITS if name in stored), None)
    if split not in stored:
        raise KeyError(f"Run {run_id!r} has no saved predictions for {split or SPLITS}")
    level_names = list(info(dataset).level_names)
    level, proba, _ = interpret(stored[split]["output"], len(level_names))
    return {
        "dataset": dataset, "split": split, "level_names": level_names,
        "ids": stored[split]["ids"], "y": stored[split]["y"].astype(int), "level": level,
        "proba": proba, "confidence": proba[np.arange(len(level)), level],
    }


def confusion(y, level, num_levels: int) -> np.ndarray:
    """Return the confusion matrix as counts: row = labelled level, column = predicted level."""
    counts = np.zeros((num_levels, num_levels), dtype=int)
    np.add.at(counts, (np.asarray(y), np.asarray(level)), 1)
    return counts


def recall_per_level(counts: np.ndarray) -> np.ndarray:
    """Return the share of each labelled level that the model got right. NaN for a level with no question."""
    totals = counts.sum(axis=1)
    return np.divide(np.diag(counts), totals, out=np.full(len(counts), np.nan), where=totals > 0)


def worst_errors(run_id: str, n: int = ERRORS_PER_DATASET, split: str | None = None) -> pd.DataFrame:
    """Return the `n` worst errors of a run, ordered as described at the top of this file."""
    s = scored_split(run_id, split)
    table = pd.DataFrame({
        "dataset": s["dataset"], "item_id": s["ids"], "run_id": run_id, "split": s["split"],
        "true_level": s["y"], "predicted_level": s["level"],
        "abs_error": np.abs(s["level"] - s["y"]), "confidence": s["confidence"].round(4),
        "proba": [json.dumps([round(float(p), 4) for p in row]) for row in s["proba"]],
    })
    table = table[table["abs_error"] > 0]
    table = table.sort_values(["abs_error", "confidence", "item_id"], ascending=[False, False, True]).head(n)
    table.insert(5, "true_name", [s["level_names"][k] for k in table["true_level"]])
    table.insert(7, "predicted_name", [s["level_names"][k] for k in table["predicted_level"]])
    return table.reset_index(drop=True)


def best_runs() -> list[str]:
    """Return, for every dataset, the run with the best dev QWK among the runs with saved predictions."""
    best: dict[str, dict] = {}
    for row in load_rows():
        score = (row["metrics"].get("dev") or {}).get("qwk")
        if score is None or not predictions_file(row["run_id"]).exists():
            continue
        if row["dataset"] not in best or score > best[row["dataset"]]["metrics"]["dev"]["qwk"]:
            best[row["dataset"]] = row
    return [best[dataset]["run_id"] for dataset in sorted(best)]


def file_hash(path: Path) -> str:
    """Return the SHA-256 of a file, so the audit can name the exact predictions it was drawn from."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_text(case: pd.Series, item: dict) -> str:
    """Return one case for the reviewer: what the model saw, and the context it did not see."""
    lines = [
        f"## {case['dataset']} · {case['item_id']}",
        "",
        f"- Run: `{case['run_id']}` ({case['split']})",
        f"- Labelled: **{case['true_name']}** · predicted: **{case['predicted_name']}** "
        f"(gap {case['abs_error']}, confidence {case['confidence']:.2f})",
        f"- Probabilities: {case['proba']}",
        "",
        "**What the model saw**",
        "",
        item["text"],
    ]
    context = {"Passage": item.get("passage"), "Solution": item["meta"].get("solution"), "Hint": item["meta"].get("hint")}
    for title, text in context.items():
        if text:
            lines += ["", f"**{title} (not shown to the model)**", "", str(text)]
    return "\n".join(lines)


def audit_list(run_ids: list[str], n: int = ERRORS_PER_DATASET) -> tuple[pd.DataFrame, list[Path]]:
    """Write the worst errors of the given runs for the human audit. Return the table and the two files."""
    tables, cases = [], []
    for run_id in run_ids:
        table = worst_errors(run_id, n)
        table["predictions_sha256"] = file_hash(predictions_file(run_id))
        tables.append(table)
        if table.empty:
            continue
        items = {item["id"]: item for item in load_split(table["dataset"].iloc[0], "test")}
        cases += [_case_text(case, items[case["item_id"]]) for _, case in table.iterrows()]
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    public = results_dir() / "audit" / "worst_errors.csv"
    public.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(public, index=False, encoding="utf-8", lineterminator="\n")
    private = cache_dir() / "audit" / "worst_errors.md"
    private.parent.mkdir(parents=True, exist_ok=True)
    header = "# Worst errors for the human audit\n\nPrivate file: contains dataset text. Do not commit it.\n\n"
    private.write_text(header + "\n\n".join(cases) + "\n", encoding="utf-8")
    return table, [public, private]


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    run_ids = (sys.argv[1:] if argv is None else argv) or best_runs()
    if not run_ids:
        print("No runs with saved predictions.")
        return
    table, files = audit_list(run_ids)
    print(table.drop(columns=["proba", "predictions_sha256"]).to_string(index=False))
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
