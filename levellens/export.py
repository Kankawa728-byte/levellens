"""Check the saved models behind the Predictor API (task E8).

    python -m levellens.export                              # best saved model per dataset, plus ARC DistilBERT for the demo
    python -m levellens.export arc-distilbert-ce-s0 ...     # these runs

A run with `save_model=true` writes its model to `<LEVELLENS_CACHE>/models/<run_id>/`, and
`Predictor.load("<run_id>")` loads it. For each run this command checks two things and writes them
to `results/export/<run_id>.json`:

1. Same predictions: the Predictor, loaded from the folder alone, gives the probabilities that the
   training run stored in `results/preds/<run_id>.npz` (largest difference below TOLERANCE).
2. Latency: seconds per question on the CPU, one question at a time, mean over 100 test questions
   (specification R7: at most 1 second for DistilBERT on a laptop CPU).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from levellens.data import load_split
from levellens.paths import results_dir
from levellens.predictor import CARD, Predictor, model_folder
from levellens.results import load_predictions, load_rows

TOLERANCE = 1e-3   # bf16 on the GPU against fp32 on the CPU differ in the fourth decimal
TIMED_ITEMS = 100
R7_SECONDS = 1.0


def best_saved_runs() -> list[str]:
    """Return the best saved run per dataset and the best saved ARC DistilBERT for the demo."""
    best: dict[str, dict] = {}
    demo: dict | None = None
    for row in load_rows():
        score = (row["metrics"].get("dev") or {}).get("qwk")
        if score is None or not (model_folder(row["run_id"]) / CARD).exists():
            continue
        if row["dataset"] not in best or score > best[row["dataset"]]["metrics"]["dev"]["qwk"]:
            best[row["dataset"]] = row
        if row["dataset"] == "arc" and row["model"] == "distilbert":
            if demo is None or score > demo["metrics"]["dev"]["qwk"]:
                demo = row
    run_ids = [row["run_id"] for row in best.values()]
    if demo is not None and demo["run_id"] not in run_ids:
        run_ids.append(demo["run_id"])
    return run_ids


def check(run_id: str) -> dict:
    """Compare the Predictor with the stored predictions of a run and time it on the CPU."""
    previous_device = os.environ.get("LEVELLENS_DEVICE")
    os.environ["LEVELLENS_DEVICE"] = "cpu"
    try:
        predictor = Predictor.load(run_id)
    finally:
        if previous_device is None:
            os.environ.pop("LEVELLENS_DEVICE", None)
        else:
            os.environ["LEVELLENS_DEVICE"] = previous_device
    dataset = json.loads((model_folder(run_id) / CARD).read_text(encoding="utf-8"))["dataset"]
    stored = load_predictions(run_id)
    split = "test" if "test" in stored else next(iter(stored))
    items = {item["id"]: item for item in load_split(dataset, split)}
    ordered = [items[item_id] for item_id in stored[split]["ids"]]
    difference = float(np.abs(predictor.predict(ordered)["proba"] - stored[split]["output"]).max())

    clock = time.perf_counter()
    for item in ordered[:TIMED_ITEMS]:
        predictor.predict([item["text"]])
    seconds = (time.perf_counter() - clock) / min(TIMED_ITEMS, len(ordered))

    report = {
        "run_id": run_id, "dataset": dataset, "split": split, "n": len(ordered),
        "max_abs_difference": difference, "same_predictions": difference < TOLERANCE,
        "seconds_per_question": round(seconds, 4), "timed_items": min(TIMED_ITEMS, len(ordered)),
        "device": str(getattr(predictor.model, "device", "cpu")), "r7_pass": seconds <= R7_SECONDS,
    }
    folder = results_dir() / "export"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run_id}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    run_ids = (sys.argv[1:] if argv is None else argv) or best_saved_runs()
    if not run_ids:
        print("No saved models. Train one with save_model=true.")
    for run_id in run_ids:
        report = check(run_id)
        print(f"{run_id}: same predictions {report['same_predictions']} (max difference {report['max_abs_difference']:.1e} "
              f"on {report['n']} {report['split']} items); {report['seconds_per_question']:.3f} s per question on {report['device']}")


if __name__ == "__main__":
    main()
