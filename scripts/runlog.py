"""C4: the run log. What was submitted, when, with which job id, and how it ended.

    python scripts/runlog.py                 # the Markdown table that goes into docs/compute.md
    python scripts/runlog.py --csv log.csv   # the same rows as CSV
    python scripts/runlog.py --watch         # only the jobs that are still queued or running

`scripts/submit.sh` appends one line per submission to `logs/submissions.tsv`. This command reads
that file and asks the scheduler, through `sacct`, how each job ended. Without `sacct` it prints
the submissions with an empty state column, so the log still works on a machine with no scheduler.

Only the standard library is needed, so the command also runs before the environment is built.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "logs" / "submissions.tsv"
SACCT_FIELDS = ("JobID", "State", "Elapsed", "ExitCode")
UNFINISHED = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")
COLUMNS = ("Submitted", "Kind", "Job", "What", "State", "Elapsed")


def submissions(path: Path = LOG) -> list[dict[str, str]]:
    """Return the lines of the submission log, oldest first."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row.get("job_id")]


def sacct_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return what the scheduler says about these jobs: state, elapsed time and exit code."""
    if not job_ids or not shutil.which("sacct"):
        return {}
    try:
        output = subprocess.run(
            ["sacct", "--jobs=" + ",".join(job_ids), "--format=" + ",".join(SACCT_FIELDS),
             "--parsable2", "--noheader"],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    states = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < len(SACCT_FIELDS):
            continue
        job_id = fields[0].split(".")[0]  # the steps of a job are JobID.batch, JobID.0 and so on
        states.setdefault(job_id, dict(zip(SACCT_FIELDS, fields)))
    return states


def rows(path: Path = LOG, unfinished_only: bool = False) -> list[dict[str, str]]:
    """Return one row per submission, with the state the scheduler reports."""
    entries = submissions(path)
    states = sacct_states([entry["job_id"] for entry in entries])
    table = []
    for entry in entries:
        state = states.get(entry["job_id"], {})
        name = state.get("State", "")
        if state.get("ExitCode", "0:0") not in ("0:0", "") and name.startswith("FAILED"):
            name = f"{name} ({state['ExitCode']})"
        if unfinished_only and not any(name.startswith(open_state) for open_state in UNFINISHED):
            continue
        table.append({
            "Submitted": entry["submitted"],
            "Kind": entry["kind"],
            "Job": entry["job_id"],
            "What": entry["arguments"] or entry["name"],
            "State": name or "unknown",
            "Elapsed": state.get("Elapsed", ""),
        })
    return table


def as_markdown(table: list[dict[str, str]]) -> str:
    """Return the rows as the Markdown table of docs/compute.md."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for row in table:
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=LOG, help=f"the submission log; default {LOG}")
    parser.add_argument("--csv", type=Path, metavar="FILE", help="also write the rows as CSV")
    parser.add_argument("--watch", action="store_true", help="only the jobs that have not finished")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No submissions yet: {args.log} does not exist. Submit a job with scripts/submit.sh.")
        return 0
    table = rows(args.log, unfinished_only=args.watch)
    if not table:
        print("No jobs are queued or running." if args.watch else f"{args.log} has no submissions.")
        return 0
    print(as_markdown(table))
    if not shutil.which("sacct"):
        print("\nNo `sacct` on this machine, so the State column is empty.", file=sys.stderr)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
        print(f"\nWritten to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
