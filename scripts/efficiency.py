"""C6: the efficiency table. What every model cost to train and how fast it answers one question.

    python scripts/efficiency.py                     # build the tables and print the verdicts
    python scripts/efficiency.py --print             # print them, write nothing
    python scripts/efficiency.py --strict            # exit 1 when R6 or R7 fails

Three files are written into the tables folder of the results store:

    efficiency_runs.csv   one line per run: the compute hours of that run, as R6 asks
    efficiency.csv        one line per model: runs, parameters, training time, GPU-hours
    efficiency.md         both tables, with the verdict on R6 and R7

What goes in:

- `runs.jsonl` gives `train_seconds`, `predict_seconds`, `n_params` and `device` of every run.
  The compute time of a run is the training plus the scoring of every split. The newest row of a
  `run_id` counts, as in `levellens.results`.
- The device of the row decides what those hours are: a `cuda` run spends GPU-hours, a `cpu` run
  CPU-hours, and `mps` (an Apple laptop) is counted apart because it is neither. The per-run limit
  of R6 is checked for every fine-tune, whichever device it ran on; the project limit counts the
  GPU-hours only.
- `export/<run_id>.json`, written by `python -m levellens.export`, gives the seconds per question
  on the CPU of the saved models. That is the measurement R7 asks for.
- `latency/<run_id>__<device>.json`, written by `python scripts/latency.py`, gives the seconds per
  question on any other device, for example DeBERTa-v3 on the GPU.

The two specifications, frozen in docs/design.md:

- **R6**: every fine-tune at most 1 GPU-hour, the whole project at most 20 GPU-hours.
- **R7**: at most 1 second per question on a laptop CPU, DistilBERT, over 100 items.

Only the standard library is needed, so the table can be rebuilt in a fresh environment before
anything is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# model name -> the family it belongs to in docs/design.md. Anything unknown is called "other".
FAMILIES = {
    "majority": "floor", "stratified_random": "floor", "subject_prior": "floor",
    "tfidf_logreg": "classical", "tfidf_svm": "classical", "tfidf_cnb": "classical",
    "r2de": "classical", "linguistic_lightgbm": "feature",
    "bilstm": "recurrent", "distilbert": "transformer", "deberta-v3": "transformer",
    "qwen_zero_shot": "open LLM",
}
FINE_TUNED = ("recurrent", "transformer")  # the families that train weights, so R6 applies to them
R6_RUN_HOURS = 1.0
R6_PROJECT_HOURS = 20.0
R7_SECONDS = 1.0
R7_MODEL = "distilbert"
RUN_COLUMNS = ("run_id", "dataset", "model", "head", "family", "seed", "device", "device_kind",
               "n_params", "train_seconds", "predict_seconds", "compute_hours", "r6_run_pass")
MODEL_COLUMNS = ("model", "family", "runs", "n_params", "device_kind", "mean_train_seconds",
                 "longest_train_seconds", "max_run_hours", "total_hours",
                 "cpu_seconds_per_question", "gpu_seconds_per_question")


def results_folder(given: Path | None = None) -> Path:
    """Return the results store: the folder given, LEVELLENS_RESULTS, or `results/` in the repository."""
    if given:
        return Path(given)
    if value := os.environ.get("LEVELLENS_RESULTS"):
        return Path(value).expanduser()
    return REPO_ROOT / "results"


def load_rows(folder: Path) -> list[dict]:
    """Return the newest row of every `run_id` in `runs.jsonl`, in file order."""
    path = folder / "runs.jsonl"
    if not path.exists():
        raise SystemExit(f"No result rows: {path} does not exist.")
    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path} line {number} is not valid JSON: {error}") from error
            latest.pop(row["run_id"], None)
            latest[row["run_id"]] = row
    return list(latest.values())


def device_kind(device: str) -> str:
    """Return what kind of hours a run on this device spends: GPU, CPU or Apple GPU."""
    device = (device or "").lower()
    if device.startswith("cuda"):
        return "GPU"
    if device.startswith("mps"):
        return "Apple GPU"
    return "CPU"


def latencies(folder: Path) -> dict[tuple[str, str], float]:
    """Return the measured seconds per question, keyed by run id and device kind.

    Two sources: the reports of `python -m levellens.export` (always on the CPU) and the reports of
    `python scripts/latency.py` (any device). The newest file wins when both measured the same pair.
    """
    found: dict[tuple[str, str], tuple[float, float]] = {}
    for path in sorted((folder / "export").glob("*.json")) + sorted((folder / "latency").glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id, seconds = report.get("run_id"), report.get("seconds_per_question")
        if not run_id or seconds is None:
            continue
        key = (run_id, device_kind(report.get("device", "cpu")))
        stamp = path.stat().st_mtime
        if key not in found or stamp >= found[key][0]:
            found[key] = (stamp, float(seconds))
    return {key: seconds for key, (_, seconds) in found.items()}


def run_table(rows: list[dict]) -> list[dict]:
    """Return one line per run: its family, its device, and the compute hours it spent."""
    table = []
    for row in rows:
        predict_seconds = sum(float(value) for value in (row.get("predict_seconds") or {}).values())
        train_seconds = float(row.get("train_seconds") or 0.0)
        hours = (train_seconds + predict_seconds) / 3600
        family = FAMILIES.get(row["model"], "other")
        kind = device_kind(row.get("device", ""))
        applies = family in FINE_TUNED  # R6 limits the length of a fine-tune, on whatever device it ran
        table.append({
            "run_id": row["run_id"], "dataset": row["dataset"], "model": row["model"],
            "head": row["head"], "family": family, "seed": row["seed"],
            "device": row.get("device", ""), "device_kind": kind,
            "n_params": int(row.get("n_params") or 0),
            "train_seconds": round(train_seconds, 1),
            "predict_seconds": round(predict_seconds, 1),
            "compute_hours": round(hours, 4),
            "r6_run_pass": "" if not applies else ("pass" if hours <= R6_RUN_HOURS else "FAIL"),
        })
    return table


def model_table(runs: list[dict], measured: dict[tuple[str, str], float]) -> list[dict]:
    """Return one line per model: how many runs, how long they trained and what they cost."""
    table = []
    for model in sorted({run["model"] for run in runs}, key=lambda name: (FAMILIES.get(name, "other"), name)):
        group = [run for run in runs if run["model"] == model]
        trains = [run["train_seconds"] for run in group]
        cpu = [measured[key] for key in measured if key[0] in {run["run_id"] for run in group} and key[1] == "CPU"]
        gpu = [measured[key] for key in measured if key[0] in {run["run_id"] for run in group} and key[1] == "GPU"]
        table.append({
            "model": model,
            "family": group[0]["family"],
            "runs": len(group),
            "n_params": max(run["n_params"] for run in group),
            "device_kind": ", ".join(sorted({run["device_kind"] for run in group})),
            "mean_train_seconds": round(sum(trains) / len(trains), 1),
            "longest_train_seconds": round(max(trains), 1),
            "max_run_hours": round(max(run["compute_hours"] for run in group), 3),
            "total_hours": round(sum(run["compute_hours"] for run in group), 3),
            "cpu_seconds_per_question": round(min(cpu), 4) if cpu else "",
            "gpu_seconds_per_question": round(min(gpu), 4) if gpu else "",
        })
    return table


def verdicts(runs: list[dict], measured: dict[tuple[str, str], float]) -> dict:
    """Return the pass or fail of R6 and R7, with the numbers behind each one."""
    fine_tunes = [run for run in runs if run["r6_run_pass"]]
    over = [run for run in fine_tunes if run["r6_run_pass"] == "FAIL"]
    gpu_hours = sum(run["compute_hours"] for run in runs if run["device_kind"] == "GPU")
    other_hours = sum(run["compute_hours"] for run in runs if run["device_kind"] == "Apple GPU")
    r7 = {run_id: seconds for (run_id, kind), seconds in measured.items()
          if kind == "CPU" and R7_MODEL in run_id}
    longest = max(fine_tunes, key=lambda run: run["compute_hours"], default=None)
    return {
        "R6": {
            "fine_tunes": len(fine_tunes),
            "longest_fine_tune": {"run_id": longest["run_id"], "hours": longest["compute_hours"]} if longest else None,
            "over_one_hour": [run["run_id"] for run in over],
            "gpu_hours": round(gpu_hours, 3),
            "other_accelerator_hours": round(other_hours, 3),
            "cpu_hours": round(sum(run["compute_hours"] for run in runs if run["device_kind"] == "CPU"), 3),
            "pass": not over and gpu_hours <= R6_PROJECT_HOURS,
            "measured": bool(fine_tunes),
        },
        "R7": {
            "model": R7_MODEL,
            "seconds_per_question": {run_id: seconds for run_id, seconds in sorted(r7.items())},
            "best": round(min(r7.values()), 4) if r7 else None,
            "pass": bool(r7) and min(r7.values()) <= R7_SECONDS,
            "measured": bool(r7),
        },
    }


def _markdown(table: list[dict], columns: tuple[str, ...], titles: dict[str, str]) -> str:
    """Return a table as GitHub Markdown, with the given columns in the given order."""
    header = [titles.get(column, column) for column in columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in table:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:,}" if isinstance(value, int) and column == "n_params" else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


MODEL_TITLES = {
    "model": "Model", "family": "Family", "runs": "Runs", "n_params": "Parameters",
    "device_kind": "Ran on", "mean_train_seconds": "Mean train s", "longest_train_seconds": "Longest train s",
    "max_run_hours": "Hours, longest run", "total_hours": "Hours, all runs",
    "cpu_seconds_per_question": "s/question CPU", "gpu_seconds_per_question": "s/question GPU",
}
RUN_TITLES = {
    "run_id": "Run", "dataset": "Data", "model": "Model", "head": "Head", "family": "Family",
    "seed": "Seed", "device": "Device", "device_kind": "Kind", "n_params": "Parameters",
    "train_seconds": "Train s", "predict_seconds": "Score s", "compute_hours": "Hours", "r6_run_pass": "R6",
}


def report(models: list[dict], runs: list[dict], checks: dict) -> str:
    """Return the whole `efficiency.md` page: the verdicts, the model table and the run table."""
    r6, r7 = checks["R6"], checks["R7"]
    note = ("Built from `runs.jsonl`, `export/` and `latency/` by `python scripts/efficiency.py`. "
            "Do not edit by hand.")
    lines = ["# Efficiency", "", note, "", "## Specifications", ""]

    if not r6["measured"]:
        lines.append("- **R6**: not measured yet. No fine-tune has a result row.")
    else:
        longest = r6["longest_fine_tune"]
        state = "pass" if r6["pass"] else "FAIL"
        lines += [
            f"- **R6 {state}**: {r6['fine_tunes']} fine-tunes. The longest is `{longest['run_id']}` at "
            f"{longest['hours']:.2f} hours, against a limit of {R6_RUN_HOURS:.0f} hour per run."
            + (f" Over the limit: {', '.join(r6['over_one_hour'])}." if r6["over_one_hour"] else ""),
            f"- **R6, whole project**: {r6['gpu_hours']:.2f} GPU-hours, against a limit of "
            f"{R6_PROJECT_HOURS:.0f}. Plus {r6['cpu_hours']:.2f} CPU-hours"
            + (f" and {r6['other_accelerator_hours']:.2f} hours on another accelerator" if r6["other_accelerator_hours"] else "")
            + ", which the limit does not count.",
        ]
    if not r7["measured"]:
        lines.append(f"- **R7**: not measured yet. Run `python -m levellens.export` on a saved {R7_MODEL} model.")
    else:
        lines.append(
            f"- **R7 {'pass' if r7['pass'] else 'FAIL'}**: {r7['best']:.3f} seconds per question on the CPU "
            f"({R7_MODEL}, one question at a time), against a limit of {R7_SECONDS:.0f} second."
        )
    lines += ["", "## One line per model", "", _markdown(models, MODEL_COLUMNS, MODEL_TITLES),
              "", "## One line per run", "", "R6 is checked per run, so every run is listed. The column R6 is "
              "empty for a model that trains no weights.", "", _markdown(runs, RUN_COLUMNS, RUN_TITLES), ""]
    return "\n".join(lines)


def write_csv(path: Path, table: list[dict], columns: tuple[str, ...]) -> Path:
    """Write a table as CSV with the given columns and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return path


def build(folder: Path, write: bool = True) -> tuple[str, dict]:
    """Build the tables of one results store. Return the Markdown page and the verdicts."""
    rows = load_rows(folder)
    measured = latencies(folder)
    runs = run_table(rows)
    models = model_table(runs, measured)
    page = report(models, runs, verdicts(runs, measured))
    if write:
        tables = folder / "tables"
        for path in (write_csv(tables / "efficiency_runs.csv", runs, RUN_COLUMNS),
                     write_csv(tables / "efficiency.csv", models, MODEL_COLUMNS)):
            print(path)
        (tables / "efficiency.md").write_text(page + "\n", encoding="utf-8", newline="\n")
        print(tables / "efficiency.md")
    return page, verdicts(runs, measured)


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, metavar="FOLDER", help="the results store; default LEVELLENS_RESULTS")
    parser.add_argument("--print", dest="show", action="store_true", help="print the page, write no file")
    parser.add_argument("--strict", action="store_true", help="exit 1 when R6 or R7 fails or was not measured")
    args = parser.parse_args(argv)

    folder = results_folder(args.results)
    page, checks = build(folder, write=not args.show)
    if args.show:
        print(page)
    else:
        print("\n".join(line for line in page.splitlines() if line.startswith("- **")))
    if args.strict and not all(check["pass"] for check in checks.values()):
        print("\nFAIL: a specification is not met or not measured yet.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
