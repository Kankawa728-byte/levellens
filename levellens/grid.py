"""Run the lines of a grid file. A grid file lists one run per line, as the arguments of `levellens.run`.

    python -m levellens.grid configs/grid/tier1_transformers.txt --list       # print the commands, numbered
    python -m levellens.grid configs/grid/tier1_transformers.txt              # run all, one after the other
    python -m levellens.grid configs/grid/tier1_transformers.txt --only 3,7   # run lines 3 and 7

Empty lines and lines that start with `#` are skipped. A run that fails does not stop the others;
the failed lines are listed at the end and the exit code is 1.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from levellens.config import compose
from levellens.run import run


def read_grid(path: Path) -> list[list[str]]:
    """Return the runs of a grid file: one list of `key=value` overrides per run."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(description="Run the lines of a grid file")
    parser.add_argument("grid", type=Path, help="for example configs/grid/tier1_transformers.txt")
    parser.add_argument("--list", action="store_true", help="print the numbered commands and stop")
    parser.add_argument("--only", metavar="N,N", help="run these lines only, counted from 1, for example 3 or 3,7")
    parser.add_argument("extra", nargs="*", help="overrides added to every run, for example device=cuda:1")
    args = parser.parse_args(argv)

    runs = read_grid(args.grid)
    chosen = [int(number) for number in args.only.split(",")] if args.only else range(1, len(runs) + 1)
    failed = []
    for number in chosen:
        overrides = runs[number - 1] + args.extra
        if args.list:
            print(f"{number:2d}  python -m levellens.run {' '.join(overrides)}")
            continue
        print(f"[{number}/{len(runs)}] {' '.join(overrides)}", flush=True)
        try:
            run(compose(overrides))
        except Exception:  # one failed run must not stop the queue
            traceback.print_exc()
            failed.append(number)
    if failed:
        print(f"Failed lines: {failed}. Run them again with --only {','.join(map(str, failed))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
