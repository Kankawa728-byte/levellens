"""C5: merge the result rows of the GPU jobs into the shared results store.

    python scripts/collect.py results/jobs/*          # every per-job store into results/
    python scripts/collect.py --dry-run results/jobs/t1-arc
    python scripts/collect.py --into results/mine results/jobs/t1-arc

Each GPU job writes to a results store of its own, so that two jobs never append to the same
`runs.jsonl` at the same time. This command brings them together: it appends the rows through
`levellens.results.append_run`, which is the only sanctioned way to write a row, and copies the
saved predictions next to them.

A row that is already in the target with the same `run_id` and the same `started` time is skipped,
so running the command twice changes nothing. A row of a `run_id` that the target already has with
another `started` time is appended, and then it is the newest row of that run: a repeated run wins
over the one it repeats, which is the rule in `levellens.results`.

Copied along with the rows, when the source store has them: `preds/`, `export/`, `latency/` and
`optuna/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from levellens.paths import results_dir  # noqa: E402
from levellens.results import append_run, load_rows, runs_file  # noqa: E402

COPIED = ("preds", "export", "latency", "optuna")


def collect(source: Path, target: Path, dry_run: bool = False) -> tuple[int, int]:
    """Merge one store into another. Return how many rows were appended and how many were skipped."""
    if not runs_file(source).exists():
        print(f"{source}: no runs.jsonl, nothing to merge")
        return 0, 0
    known = {(row["run_id"], row.get("started")) for row in load_rows(target)}
    appended = skipped = 0
    for row in load_rows(source):
        if (row["run_id"], row.get("started")) in known:
            skipped += 1
            continue
        print(f"  {'would append' if dry_run else 'append'} {row['run_id']}"
              f" ({row['dataset']}/{row['model']}/{row['head']}, seed {row['seed']},"
              f" {row['train_seconds']:.0f} s on {row['device']})")
        if not dry_run:
            append_run(row, target)
        appended += 1
        for folder in COPIED:
            for path in sorted((source / folder).glob(f"{row['run_id']}*")):
                destination = target / folder / path.name
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                print(f"    {'would copy' if dry_run else 'copy'} {folder}/{path.name}")
    print(f"{source}: {appended} rows appended, {skipped} already there")
    return appended, skipped


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", type=Path, metavar="STORE", help="the per-job results stores")
    parser.add_argument("--into", type=Path, metavar="FOLDER", help="the target store; default LEVELLENS_RESULTS")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, change nothing")
    args = parser.parse_args(argv)

    target = args.into or results_dir()
    print(f"Target: {target}\n")
    total = 0
    for source in args.sources:
        if source.resolve() == Path(target).resolve():
            print(f"{source}: this is the target itself, skipped")
            continue
        total += collect(source, target, args.dry_run)[0]
    print(f"\n{total} rows {'would be' if args.dry_run else ''} appended to {runs_file(target)}")
    if total and not args.dry_run:
        print("Now rebuild the tables:  python -m levellens.results && python scripts/efficiency.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
