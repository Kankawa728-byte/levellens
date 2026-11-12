"""Share B data checks. Read the team's saved splits without modifying them.

Run from the repository root: python notebooks/data_checks.py
Aggregate evidence is public; question text stays under the ignored data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from levellens.data import check_items, info, load_split
from levellens.data.prepare import read_manifest
from levellens.data.schema import ITEM_FIELDS, format_text
from levellens.data.splits import fingerprint, write_jsonl
from levellens.paths import REPO_ROOT, data_dir, results_dir

DATASETS = ("race", "arc", "math", "scienceqa")
SPLITS = ("train", "dev", "test")
EXPECTED = {  # docs/design.md, frozen on 12 November 2026
    "race": (100568, 5599, 5642), "arc": (3358, 862, 3530),
    "math": (6748, 750, 5000), "scienceqa": (6508, 2144, 2224),
}
SAMPLE_PER_LEVEL = 10
SAMPLE_SEED = 38
COLORS = ("#176B70", "#DC9948", "#7788A3")


def validate(items: list[dict], dataset: str, split: str) -> dict:
    """Check shape, required fields, nulls and input integrity before analysis."""
    if not items:
        raise ValueError(f"{dataset}/{split}: empty split")
    descriptor = info(dataset)
    missing = Counter()
    for row in items:
        absent = set(ITEM_FIELDS) - row.keys()
        if absent:
            raise ValueError(f"{dataset}/{split}: missing fields {sorted(absent)}")
        for field in ITEM_FIELDS:
            if row[field] is None:
                missing[field] += 1
        if any(row[f] is None for f in ITEM_FIELDS if f != "passage"):
            raise ValueError(f"{row['id']}: null required field")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{row['id']}: blank or invalid model input")
        if not isinstance(row["meta"], dict):
            raise ValueError(f"{row['id']}: invalid metadata")
        if row["dataset"] != dataset or row["split"] != split:
            raise ValueError(f"{row['id']}: wrong dataset or split")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{row['id']}: blank question")
        if type(row["level"]) is not int or not 0 <= row["level"] < descriptor.num_levels or row["level_name"] != descriptor.level_names[row["level"]]:
            raise ValueError(f"{row['id']}: invalid level or name")
        if not isinstance(row["options"], list) or any(not isinstance(x, str) or not x.strip() for x in row["options"]):
            raise ValueError(f"{row['id']}: invalid option")
        if row["text"] != format_text(row["question"], row["options"]):
            raise ValueError(f"{row['id']}: input differs from question and options")
        if dataset in ("race", "arc") and len(row["options"]) != 4:
            raise ValueError(f"{row['id']}: expected four options")
        if dataset == "math" and row["options"]:
            raise ValueError(f"{row['id']}: MATH must have no options")
        if dataset == "race" and (not isinstance(row["passage"], str) or not row["passage"].strip()):
            raise ValueError(f"{row['id']}: missing reading passage")
        if dataset in ("arc", "scienceqa"):
            offset = 3 if dataset == "arc" else 1
            if row["meta"].get("grade") != row["level"] + offset:
                raise ValueError(f"{row['id']}: grade mapping mismatch")
        if row["options"]:
            answer = row["meta"].get("answer")
            if type(answer) is not int or not 0 <= answer < len(row["options"]):
                raise ValueError(f"{row['id']}: invalid answer index")
    check_items(items, descriptor)
    return {"shape": [len(items), len(ITEM_FIELDS)], "nulls": dict(missing), "valid": True}


def duplicate_summary(splits: dict[str, list[dict]], field: str) -> dict:
    """Count exact string repeats; do not lowercase, strip, or re-tokenise values.

    A group is one repeated string. Extra rows = sum(group size - 1).
    Cross-split pairs count every matching pair, not just shared strings.
    """
    indices = {}
    for split, items in splits.items():
        index = defaultdict(list)
        for row in items:
            value = row[field]
            if isinstance(value, str) and value:
                index[value].append(row)
        indices[split] = index
    within, across = {}, {}
    for split, index in indices.items():
        groups = [rows for rows in index.values() if len(rows) > 1]
        within[split] = {
            "groups": len(groups), "rows": sum(len(rows) for rows in groups),
            "extra_rows": sum(len(rows) - 1 for rows in groups),
            "conflicting_level_groups": sum(len({r['level'] for r in rows}) > 1 for rows in groups),
        }
    for a, b in combinations(splits, 2):
        shared = indices[a].keys() & indices[b].keys()
        across[f"{a}:{b}"] = {
            "groups": len(shared),
            "rows_left": sum(len(indices[a][v]) for v in shared),
            "rows_right": sum(len(indices[b][v]) for v in shared),
            "pairs": sum(len(indices[a][v]) * len(indices[b][v]) for v in shared),
            "conflicting_level_groups": sum(len({r['level'] for r in indices[a][v] + indices[b][v]}) > 1 for v in shared),
        }
    return {"within": within, "across": across}


def length_summary(items: list[dict], num_levels: int) -> list[dict]:
    """Describe stem and input length per level; words are whitespace tokens."""
    rows = []
    for level in range(num_levels):
        selected = [row for row in items if row["level"] == level]
        result = {"level": level, "n": len(selected)}
        for field in ("question", "text"):
            for unit, measure in (("words", lambda s: len(s.split())), ("chars", len)):
                values = np.asarray([measure(row[field]) for row in selected], dtype=float)
                result[f"{field}_{unit}"] = {
                    "mean": round(float(values.mean()), 2) if values.size else None,
                    "median": float(np.median(values)) if values.size else None,
                    "p95": round(float(np.quantile(values, .95)), 2) if values.size else None,
                    "min": int(values.min()) if values.size else None,
                    "max": int(values.max()) if values.size else None,
                }
        rows.append(result)
    return rows


def review_sample(items: list[dict], num_levels: int, n: int = SAMPLE_PER_LEVEL) -> list[dict]:
    """Select up to n training items per level, deterministic under row reordering."""
    if n < 1:
        raise ValueError("sample size must be positive")
    picked = []
    for level in range(num_levels):
        members = [r for r in items if r["level"] == level]
        members.sort(key=lambda r: hashlib.sha256(f"{SAMPLE_SEED}:{r['id']}".encode()).hexdigest())
        picked.extend(members[:n])
    return picked


def analyse(dataset: str, manifest: dict) -> tuple[dict, list[dict]]:
    """Load and validate saved splits, then produce aggregate checks and a private sample."""
    splits = {s: load_split(dataset, s) for s in SPLITS}
    descriptor = info(dataset)
    report = {"dataset": dataset, "level_names": list(descriptor.level_names), "splits": {}}
    for split, expected_n in zip(SPLITS, EXPECTED[dataset]):
        items = splits[split]
        checks = validate(items, dataset, split)
        fp = fingerprint(items, descriptor.num_levels)
        report["splits"][split] = {
            **checks, **fp, "expected_n": expected_n,
            "count_matches_design": len(items) == expected_n,
            "matches_manifest": fp == manifest[dataset][split],
            "lengths": length_summary(items, descriptor.num_levels),
        }
    ids = {s: {r["id"] for r in rows} for s, rows in splits.items()}
    report["shared_ids"] = {f"{a}:{b}": len(ids[a] & ids[b]) for a, b in combinations(SPLITS, 2)}
    report["duplicates"] = {field: duplicate_summary(splits, field) for field in ("question", "text")}
    report["repeated_options"] = {s: sum(len(r["options"]) != len(set(r["options"])) for r in rows) for s, rows in splits.items()}
    if dataset == "race":
        report["passage_duplicates"] = duplicate_summary(splits, "passage")
    if dataset == "scienceqa":
        report["test_grade_ranges"] = dict(Counter("inside_3_9" if 3 <= r["meta"]["grade"] <= 9 else "outside_3_9" for r in splits["test"]))
        report["test_subjects"] = dict(Counter(r["meta"]["subject"] for r in splits["test"]))
        report["nonempty_hints"] = {s: sum(bool(r["meta"].get("hint", "").strip()) for r in rows) for s, rows in splits.items()}
        report["test_subject_by_grade"] = {
            str(grade): dict(Counter(r["meta"]["subject"] for r in splits["test"] if r["meta"]["grade"] == grade))
            for grade in range(1, 13)
        }
    if dataset == "math":
        report["asy_markup"] = {s: sum("[asy]" in r["question"] for r in rows) for s, rows in splits.items()}
    report["derived_splits"] = {}
    for split in ("test_1k", "train_balanced") if dataset == "arc" else ("test_1k",):
        rows = load_split(dataset, split)
        fp = fingerprint(rows, descriptor.num_levels)
        parent = "train" if split == "train_balanced" else "test"
        originals = {r["id"]: r for r in splits[parent]}
        report["derived_splits"][split] = {
            **fp, "matches_manifest": fp == manifest[dataset][split],
            "exact_subset_of_parent": all(originals.get(r["id"]) == r for r in rows),
        }
    sample = review_sample(splits["train"], descriptor.num_levels)
    report["review_sample_per_level"] = [sum(r["level"] == k for r in sample) for k in range(descriptor.num_levels)]
    return report, sample


def plot_balance(report: dict, output: Path) -> None:
    """Save class proportions per split; each dataset gets its own readable figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    x = np.arange(len(report["level_names"]))
    for i, (split, color) in enumerate(zip(SPLITS, COLORS)):
        s = report["splits"][split]
        ax.bar(x + (i - 1) * .25, np.asarray(s["per_level"]) * 100 / s["n"], width=.24, label=f"{split} (n={s['n']:,})", color=color)
    ax.set_xticks(x, report["level_names"], rotation=25 if len(x) > 7 else 0)
    ax.set_ylabel("Items within split (%)")
    ax.set_title(f"{report['dataset'].upper() if report['dataset'] != 'race' else 'RACE++'}: target-level balance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(70, ax.get_ylim()[1] * 1.15))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    """Write public aggregates and ignored private review samples, failing on split drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args(argv)
    output = results_dir() / "data_checks"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    reports, problems = {}, []
    for dataset in args.datasets:
        report, sample = analyse(dataset, manifest)
        reports[dataset] = report
        write_jsonl(data_dir() / "review" / f"{dataset}_10_per_level.jsonl", sample)
        plot_balance(report, output / f"{dataset}_balance.png")
        for split, result in report["splits"].items():
            if not result["matches_manifest"] or not result["count_matches_design"]:
                problems.append(f"{dataset}/{split}: split drift")
        for split, result in report["derived_splits"].items():
            if not result["matches_manifest"] or not result["exact_subset_of_parent"]:
                problems.append(f"{dataset}/{split}: derived split drift")
        if any(report["shared_ids"].values()):
            problems.append(f"{dataset}: shared IDs across primary splits")
        print(dataset, {s: report["splits"][s]["n"] for s in SPLITS}, "checked")
    summary = {"python": platform.python_version(), "numpy": np.__version__,
               "manifest_sha256": hashlib.sha256((REPO_ROOT / "levellens/data/manifest.json").read_bytes()).hexdigest(),
               "definition": "Exact raw question strings and exact prepared model-input strings; only primary splits compared.",
               "datasets": reports, "problems": problems}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if problems:
        raise ValueError("; ".join(problems))
    print("Aggregate evidence saved. Raw review text remains in the configured data directory.")
    return summary


if __name__ == "__main__":
    main()
