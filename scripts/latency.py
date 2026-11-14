"""C6: how long one question takes, on the device you name.

    python scripts/latency.py                                  # the best saved model of every dataset, on the CPU
    python scripts/latency.py --device cuda                     # the same models on the GPU
    python scripts/latency.py arc-deberta-v3-ce-s0 --device cuda
    python scripts/latency.py --device cpu --items 100          # R7: DistilBERT on a laptop CPU

One report per run and device goes to `latency/<run_id>__<device>.json` in the results store, and
`python scripts/efficiency.py` puts the numbers in the efficiency table.

`python -m levellens.export` already times the saved models on the CPU, which is what specification
R7 asks for. This command adds the other half of C6, the GPU number, and lets the device, the number
of questions and the batch size be chosen.

Two numbers are measured, after a few questions that are thrown away so that the first batch, the
one that moves the weights and builds the kernels, is not counted:

    seconds_per_question           one question at a time, as the app scores a pasted question
    batched_seconds_per_question   all questions in batches, as a run scores a split
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ITEMS = 100
WARMUP = 3
BATCH = 32


def main(argv: list[str] | None = None) -> int:
    """Run the command line. The exit code is the number of runs that could not be timed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_ids", nargs="*", metavar="RUN_ID", help="default: the best saved model of every dataset")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:0; default cpu")
    parser.add_argument("--items", type=int, default=ITEMS, help=f"questions to time; default {ITEMS}")
    parser.add_argument("--batch", type=int, default=BATCH, help=f"batch size of the batched number; default {BATCH}")
    parser.add_argument("--split", default="test", help="which split the questions come from; default test")
    args = parser.parse_args(argv)

    os.environ["LEVELLENS_DEVICE"] = args.device  # NeuralModel.load reads this

    from levellens.data import load_split
    from levellens.export import best_saved_runs
    from levellens.paths import results_dir
    from levellens.predictor import CARD, Predictor, model_folder

    run_ids = args.run_ids or best_saved_runs()
    if not run_ids:
        print("No saved models. Train one with save_model=true.")
        return 0

    failures = 0
    for run_id in run_ids:
        card_path = model_folder(run_id) / CARD
        if not card_path.exists():
            print(f"{run_id}: no saved model at {model_folder(run_id)}")
            failures += 1
            continue
        dataset = json.loads(card_path.read_text(encoding="utf-8"))["dataset"]
        items = load_split(dataset, args.split)[: args.items + WARMUP]
        if len(items) <= WARMUP:
            print(f"{run_id}: {dataset}/{args.split} has only {len(items)} questions")
            failures += 1
            continue
        predictor = Predictor.load(run_id)
        timed = items[WARMUP:]

        for item in items[:WARMUP]:
            predictor.predict([item["text"]])
        clock = time.perf_counter()
        for item in timed:
            predictor.predict([item["text"]])
        one_by_one = (time.perf_counter() - clock) / len(timed)

        predictor.predict([item["text"] for item in items[:WARMUP]])
        clock = time.perf_counter()
        for start in range(0, len(timed), args.batch):
            predictor.predict([item["text"] for item in timed[start : start + args.batch]])
        batched = (time.perf_counter() - clock) / len(timed)

        device = str(getattr(predictor.model, "device", args.device))
        report = {
            "run_id": run_id, "dataset": dataset, "split": args.split, "device": device,
            "timed_items": len(timed), "warmup_items": WARMUP, "batch_size": args.batch,
            "seconds_per_question": round(one_by_one, 4),
            "batched_seconds_per_question": round(batched, 5),
            "questions_per_second_batched": round(1 / batched, 1) if batched else None,
        }
        folder = results_dir() / "latency"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}__{device.replace(':', '-')}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{run_id} on {device}: {one_by_one:.4f} s per question one by one, "
              f"{batched:.5f} s per question in batches of {args.batch} ({len(timed)} questions) -> {path}")
    return min(failures, 1)


if __name__ == "__main__":
    raise SystemExit(main())
