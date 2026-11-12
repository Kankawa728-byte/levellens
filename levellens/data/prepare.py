"""Download the datasets and write the split files that every model loads.

    python -m levellens.data.prepare              # all four datasets
    python -m levellens.data.prepare math arc     # only these

Files go to `<LEVELLENS_DATA>/processed/<dataset>/<split>.jsonl`. After writing, each split
is compared with `levellens/data/manifest.json`. A mismatch means this machine does not
hold the team's splits, and the command fails. Only the lead runs `--update-manifest`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from levellens.data import ADAPTERS, adapter, split_path
from levellens.data.schema import check_items
from levellens.data.splits import fingerprint, stratified_subset, write_jsonl
from levellens.paths import data_dir

MANIFEST = Path(__file__).with_name("manifest.json")
SUBSET_SIZE = 1000  # the fixed test subset on which every system, the LLM included, is scored
SUBSET_SEED = 0
PREPARED = [name for name in ADAPTERS if name != "toy"]


def read_manifest() -> dict:
    """Return the saved fingerprints, or an empty dict before the first run."""
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def prepare(name: str) -> dict[str, dict]:
    """Build one dataset, check it, write its split files and return the fingerprint of each split."""
    module = adapter(name)
    splits = module.build(data_dir() / "raw")
    splits["test_1k"] = stratified_subset(splits["test"], SUBSET_SIZE, seed=SUBSET_SEED)

    for split, items in splits.items():
        check_items(items, module.INFO)
    ids = {split: {item["id"] for item in splits[split]} for split in ("train", "dev", "test")}
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = ids[first] & ids[second]
        if shared:
            raise ValueError(f"{name}: {len(shared)} ids are in both {first} and {second}, e.g. {sorted(shared)[:3]}")

    prints = {}
    for split, items in splits.items():
        write_jsonl(split_path(name, split), items)
        prints[split] = fingerprint(items, module.INFO.num_levels)
        print(f"{name:10s} {split:15s} {prints[split]['n']:7d}  per level {prints[split]['per_level']}")
    return prints


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="*", default=PREPARED, help=f"any of {PREPARED}; default: all")
    parser.add_argument("--update-manifest", action="store_true", help="lead only: save the fingerprints of this build")
    args = parser.parse_args(argv)
    unknown = [name for name in args.datasets if name not in PREPARED]
    if unknown:
        parser.error(f"unknown datasets {unknown}; choose from {PREPARED}")

    manifest = read_manifest()
    problems = []
    for name in args.datasets:
        prints = prepare(name)
        if args.update_manifest:
            manifest[name] = prints
        elif name not in manifest:
            problems.append(f"{name}: no fingerprint in {MANIFEST.name} yet")
        elif manifest[name] != prints:
            changed = [split for split in prints if manifest[name].get(split) != prints[split]]
            problems.append(f"{name}: splits {changed} differ from {MANIFEST.name}")

    if args.update_manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"Saved the fingerprints to {MANIFEST}")
    elif problems:
        raise SystemExit("These splits are not the team's splits:\n  " + "\n  ".join(problems))
    else:
        print("All splits match the saved fingerprints.")


if __name__ == "__main__":
    main()
