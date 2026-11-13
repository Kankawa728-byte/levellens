"""Does the difficulty signal carry over? The transfer matrix (F3) and the ScienceQA extrapolation (F4).

    python -m levellens.explain.transfer                               # the best saved model of every dataset
    python -m levellens.explain.transfer race=race-distilbert-corn-s0 arc=arc-distilbert-corn-s0 math=...

Transfer: the model of each subject scores the test questions of every subject. The levels of two
datasets are not the same scale (3 school stages, 7 grades, 5 competition levels), so the only fair
number is a rank correlation: Spearman between the model's expected level and the labelled level.
The diagonal is the usual in-subject result; the other cells are transfer.

Extrapolation: the ARC model, trained on grades 3 to 9, scores ScienceQA, which has grades 1 to 12.
Its predicted grade is 3 + expected level, so it can never leave 3 to 9. We report Spearman inside
grades 3 to 9, outside them, and over all grades, for all questions and for natural science only
(the subject closest to ARC), and the mean predicted grade at every true grade.

Files written, in `results/transfer/`:
    transfer.csv                  one line per model and test set
    extrapolation_summary.csv     Spearman per grade range and subject
    extrapolation_by_grade.csv    mean and spread of the predicted grade at every true grade
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from levellens.data import info, load_split
from levellens.export import best_saved_runs
from levellens.paths import results_dir
from levellens.predictor import Predictor

SUBJECTS = ("race", "arc", "math")
TRANSFER_SPLIT = "test_1k"   # the fixed 1,000 questions every system is scored on
EXTRAPOLATION_SPLIT = "test"
CLOSEST_SUBJECT = "natural science"


def rank_correlation(y, score) -> float | None:
    """Return Spearman between labels and scores, or None when one of them is constant or too short."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(score) == 0:
        return None
    return float(spearmanr(y, score).statistic)


def folder() -> Path:
    """Return `results/transfer/`, created if needed."""
    path = results_dir() / "transfer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def transfer_matrix(runs: dict[str, str], split: str = TRANSFER_SPLIT, predictors: dict | None = None) -> pd.DataFrame:
    """Score every subject's test questions with every given model.

    `runs` maps the dataset a model was trained on to its run id. Returns one line per model and
    test set: trained_on, run_id, tested_on, spearman, n. Subjects without a model have no line.
    """
    predictors = predictors or {}
    lines = []
    for trained_on in (subject for subject in SUBJECTS if subject in runs):
        predictor = predictors.get(trained_on) or Predictor.load(runs[trained_on])
        for tested_on in SUBJECTS:
            items = load_split(tested_on, split)
            score = predictor.predict(items)["expected_level"]
            lines.append({
                "trained_on": trained_on, "run_id": runs[trained_on], "tested_on": tested_on, "split": split,
                "spearman": rank_correlation([item["level"] for item in items], score), "n": len(items),
            })
    return pd.DataFrame(lines, columns=["trained_on", "run_id", "tested_on", "split", "spearman", "n"])


def first_grade(dataset: str) -> int:
    """Return the school grade of level 0, read from the level names ("grade 3" -> 3)."""
    return int(info(dataset).level_names[0].removeprefix("grade "))


def extrapolate(
    run_id: str, split: str = EXTRAPOLATION_SPLIT, predictor: Predictor | None = None, items: list[dict] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score ScienceQA with the ARC model. Return (summary, by_grade).

    summary   one line per subject group and grade range: n and Spearman
    by_grade  one line per subject group and true grade: n, mean and standard deviation of the predicted grade
    """
    predictor = predictor or Predictor.load(run_id)
    items = items if items is not None else load_split("scienceqa", split)
    low = first_grade("arc")
    high = low + len(predictor.level_names) - 1
    table = pd.DataFrame({
        "grade": [item["meta"]["grade"] for item in items],
        "subject": [item["meta"]["subject"] for item in items],
        "predicted_grade": low + predictor.predict(items)["expected_level"],
    })
    inside = table["grade"].between(low, high)
    ranges = {
        f"inside {low}-{high}": inside, f"outside {low}-{high}": ~inside,
        f"below {low}": table["grade"] < low, f"above {high}": table["grade"] > high,
        "all grades": pd.Series(True, index=table.index),
    }
    groups = {"all subjects": pd.Series(True, index=table.index), CLOSEST_SUBJECT: table["subject"] == CLOSEST_SUBJECT}

    summary, by_grade = [], []
    for group, in_group in groups.items():
        for name, in_range in ranges.items():
            part = table[in_group & in_range]
            summary.append({
                "run_id": run_id, "subjects": group, "grades": name, "n": len(part),
                "spearman": rank_correlation(part["grade"], part["predicted_grade"]),
                "mean_predicted_grade": float(part["predicted_grade"].mean()) if len(part) else None,
            })
        stats = table[in_group].groupby("grade")["predicted_grade"].agg(["size", "mean", "std"]).reset_index()
        stats.columns = ["grade", "n", "mean_predicted_grade", "sd_predicted_grade"]
        stats.insert(0, "subjects", group)
        stats.insert(0, "run_id", run_id)
        stats["inside"] = stats["grade"].between(low, high)
        by_grade.append(stats)
    return pd.DataFrame(summary), pd.concat(by_grade, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    arguments = sys.argv[1:] if argv is None else argv
    runs = dict(argument.split("=", 1) for argument in arguments)
    if not runs:
        from levellens.explain.local import run_dataset

        runs = {run_dataset(run_id): run_id for run_id in best_saved_runs()}
    unknown = sorted(set(runs) - set(SUBJECTS))
    if unknown:
        raise SystemExit(f"Unknown datasets {unknown}. Use dataset=run_id with a dataset from {SUBJECTS}.")

    from levellens.figures import extrapolation_figure, transfer_figure

    written = []
    if runs:
        matrix = transfer_matrix(runs)
        written.append(folder() / "transfer.csv")
        matrix.to_csv(written[-1], index=False, encoding="utf-8", lineterminator="\n")
        print(matrix.pivot(index="trained_on", columns="tested_on", values="spearman").round(3).to_string())
        written.append(transfer_figure())
    if "arc" in runs:
        summary, by_grade = extrapolate(runs["arc"])
        for name, table in (("extrapolation_summary.csv", summary), ("extrapolation_by_grade.csv", by_grade)):
            written.append(folder() / name)
            table.to_csv(written[-1], index=False, encoding="utf-8", lineterminator="\n")
        print(summary.round(3).to_string(index=False))
        written.append(extrapolation_figure())
    for path in written:
        print(path)
    if not written:
        print("No saved models. Train them with save_model=true.")


if __name__ == "__main__":
    main()
