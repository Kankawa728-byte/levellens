"""Result and error figures, all rebuilt from the results store (tasks F2, F3, F4; specification R8).

    python -m levellens.figures                    # every figure that has its data
    python -m levellens.figures arc-bilstm-corn-s0 # also the confusion matrix of these runs

Figures go to `results/figures/`:
    leaderboard.png              QWK of every system, one panel per dataset
    confusion_<run_id>.png       ordered confusion matrix and recall per level
    cue_words_<run_id>.png       LIME cue words per level    (data: python -m levellens.explain.local)
    transfer.png                 transfer matrix, Spearman   (data: python -m levellens.explain.transfer)
    extrapolation.png            ARC model on ScienceQA      (same command)

No function here loads a model. Each one reads `runs.jsonl`, the saved predictions, or a CSV that
another command of share F wrote, so the figures can be redrawn on any machine.

Colour rules: one hue from light to dark for a magnitude, two hues around a neutral grey for a
signed value, and a fixed colour per series. Values are written on the marks, so no figure depends
on colour alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed on a GPU node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from levellens.data import info  # noqa: E402
from levellens.explain.errors import best_runs, confusion, recall_per_level, scored_split  # noqa: E402
from levellens.paths import results_dir  # noqa: E402
from levellens.results import load_runs  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#ecebe7"
SEQUENTIAL = LinearSegmentedColormap.from_list("levellens_blue", [SURFACE, BLUE, "#0f3f7d"])
DIVERGING = LinearSegmentedColormap.from_list("levellens_diverging", [ORANGE, NEUTRAL, BLUE])
DATASET_TITLES = {"race": "RACE++", "arc": "ARC", "math": "MATH", "scienceqa": "ScienceQA", "toy": "toy"}
LEADERBOARD_SPLIT = "test_1k"  # the split every system has, the open LLM too
DPI = 200


def _style() -> None:
    """Set the look shared by every figure: quiet axes, no top and right frame."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def figure_file(name: str) -> Path:
    """Return the path of a figure in `results/figures/`, creating the folder."""
    folder = results_dir() / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.png"


def _save(fig, name: str) -> Path:
    """Write a figure and close it."""
    path = figure_file(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _cell_ink(share: float) -> str:
    """Return a text colour that stays readable on a heat-map cell of this strength (0 to 1)."""
    return "white" if share > 0.55 else INK


def confusion_figure(run_id: str, split: str | None = None) -> Path:
    """Draw the ordered confusion matrix of a run next to its recall per level.

    Each row of the matrix is one labelled level and sums to 100%. The number in a cell is the count.
    Levels are in order, so errors close to the diagonal are near misses.
    """
    _style()
    s = scored_split(run_id, split)
    names = s["level_names"]
    counts = confusion(s["y"], s["level"], len(names))
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    recall = recall_per_level(counts)

    size = max(3.2, 0.62 * len(names) + 1.2)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(size + 2.6, size), gridspec_kw={"width_ratios": [size, 1.9], "wspace": 0.08}
    )
    left.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for row in range(len(names)):
        for column in range(len(names)):
            left.text(column, row, f"{counts[row, column]:,}", ha="center", va="center",
                      fontsize=8, color=_cell_ink(shares[row, column]))
    left.set_xticks(range(len(names)), names, rotation=45 if len(names) > 3 else 0,
                    ha="right" if len(names) > 3 else "center")
    left.set_yticks(range(len(names)), names)
    left.set_xlabel("Predicted level")
    left.set_ylabel("Labelled level")
    left.tick_params(length=0)
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title(f"{DATASET_TITLES.get(s['dataset'], s['dataset'])} · {run_id}\n"
                   f"{s['split']}, n = {len(s['y']):,}; colour = share of the labelled level", fontsize=9)

    positions = np.arange(len(names))
    right.barh(positions, np.nan_to_num(recall), height=0.62, color=BLUE)
    for position, value in zip(positions, recall):
        label = "no items" if np.isnan(value) else f"{value:.0%}"
        right.text(np.nan_to_num(value) + 0.03, position, label, va="center", fontsize=8, color=INK)
    right.set_xlim(0, 1.25)
    right.set_ylim(len(names) - 0.5, -0.5)
    right.set_yticks([])
    right.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    right.set_title("Recall per level", fontsize=9)
    right.spines["left"].set_visible(False)
    return _save(fig, f"confusion_{run_id}")


def leaderboard_figure(split: str = LEADERBOARD_SPLIT, metric: str = "qwk") -> Path | None:
    """Draw the metric of every system, one panel per dataset, best at the top.

    Runs that differ only in the seed are one bar: its length is the mean, the whisker is the standard deviation.
    Returns None when the results store has no run with this split.
    """
    _style()
    runs = load_runs()
    column = f"metrics.{split}.{metric}"
    if column not in runs.columns:
        return None
    runs = runs[runs[column].notna()]
    datasets = [name for name in ("race", "arc", "math") if name in set(runs["dataset"])]
    datasets += sorted(set(runs["dataset"]) - set(datasets))
    if not datasets:
        return None

    panels = []
    for dataset in datasets:
        part = runs[runs["dataset"] == dataset].fillna({"variant": "", "head": ""})
        table = part.groupby(["model", "head", "variant"])[column].agg(["mean", "std", "size"]).reset_index()
        table["label"] = [
            " · ".join(part for part in (row.model, "" if row.head == "none" else row.head, row.variant) if part)
            for row in table.itertuples()
        ]
        panels.append(table.sort_values("mean", ascending=True))

    rows = max(len(table) for table in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 1.1 + 0.36 * rows), squeeze=False)
    for ax, dataset, table in zip(axes[0], datasets, panels):
        positions = np.arange(len(table))
        ax.barh(positions, table["mean"], height=0.6, color=BLUE,
                xerr=table["std"].fillna(0) if table["std"].notna().any() else None, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        for position, row in zip(positions, table.itertuples()):
            seeds = f"  ({row.size} seeds)" if row.size > 1 else ""
            ax.text(max(row.mean, 0) + (0 if np.isnan(row.std) else row.std) + 0.015, position,
                    f"{row.mean:.3f}{seeds}", va="center", fontsize=8, color=INK)
        ax.set_yticks(positions, table["label"])
        ax.set_xlim(min(0, table["mean"].min() - 0.05), 1.0)
        ax.set_ylim(-0.6, rows - 0.4)
        ax.set_xlabel(f"{metric.upper()} on {split}")
        ax.set_title(DATASET_TITLES.get(dataset, dataset))
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.tight_layout()
    return _save(fig, "leaderboard")


def cue_words_figure(run_id: str) -> Path | None:
    """Draw the LIME cue words of each level, one small panel per level. None if the table is missing."""
    _style()
    source = results_dir() / "explain" / f"{run_id}_cue_words.csv"
    if not source.exists():
        return None
    table = pd.read_csv(source, keep_default_na=False)  # "null" and "nan" can be words of a question
    names = table.drop_duplicates("level").sort_values("level")[["level", "level_name"]].values.tolist()
    if not names:
        return None
    top = int(table["rank"].max())
    columns = min(len(names), 4)
    rows = -(-len(names) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.9 * columns, 0.6 + 0.27 * top * rows + 0.5 * rows), squeeze=False)
    limit = float(table["weight"].max()) * 1.35
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, (level, name) in zip(axes.flat, names):
        part = table[table["level"] == level].sort_values("rank")
        positions = np.arange(len(part))
        ax.barh(positions, part["weight"], height=0.62, color=BLUE)
        for position, row in zip(positions, part.itertuples()):
            ax.text(row.weight + limit * 0.02, position, f"{row.weight:.2f}", va="center", fontsize=7, color=MUTED)
        ax.set_yticks(positions, part["word"])
        ax.set_ylim(top - 0.5, -0.5)
        ax.set_xlim(0, limit)
        ax.set_xticks([])
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle(f"Words that push a question towards each level · {run_id}\n"
                 "LIME weight, summed over the sampled test questions", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"cue_words_{run_id}")


def transfer_figure() -> Path | None:
    """Draw the transfer matrix: row = the subject a model was trained on, column = the test set. None if missing."""
    _style()
    source = results_dir() / "transfer" / "transfer.csv"
    if not source.exists():
        return None
    table = pd.read_csv(source)
    order = [name for name in ("race", "arc", "math")]
    matrix = table.pivot(index="trained_on", columns="tested_on", values="spearman")
    matrix = matrix.reindex(index=[name for name in order if name in matrix.index], columns=order)
    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * values.shape[1], 1.3 + 0.8 * values.shape[0]))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "n/a" if np.isnan(value) else f"{value:+.2f}"
            weight = "bold" if matrix.index[row] == matrix.columns[column] else "normal"
            ax.text(column, row, text, ha="center", va="center", fontsize=10, fontweight=weight,
                    color="white" if abs(np.nan_to_num(value)) > 0.5 else INK)
    ax.set_xticks(range(values.shape[1]), [DATASET_TITLES[name] for name in matrix.columns])
    ax.set_yticks(range(values.shape[0]), [DATASET_TITLES[name] for name in matrix.index])
    ax.set_xlabel("Test questions from")
    ax.set_ylabel("Model trained on")
    ax.tick_params(length=0, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Transfer between subjects\nSpearman, expected level against label, {table['split'].iloc[0]}", fontsize=9)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, ticks=[-0.8, 0, 0.8])
    bar.outline.set_visible(False)
    return _save(fig, "transfer")


def extrapolation_figure() -> Path | None:
    """Draw the ARC model's predicted grade against the true ScienceQA grade. None if the tables are missing.

    The shaded band is the range of grades the model was trained on. The dashed line is a perfect prediction.
    """
    _style()
    folder = results_dir() / "transfer"
    if not (folder / "extrapolation_by_grade.csv").exists():
        return None
    by_grade = pd.read_csv(folder / "extrapolation_by_grade.csv")
    summary = pd.read_csv(folder / "extrapolation_summary.csv")
    trained = by_grade.loc[by_grade["inside"], "grade"]
    low, high = int(trained.min()), int(trained.max())

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axvspan(low - 0.5, high + 0.5, color=NEUTRAL, zorder=0)
    ax.text((low + high) / 2, 12.3, f"trained on grades {low}–{high}", ha="center", va="top", fontsize=8, color=MUTED)
    ax.plot([1, 12], [1, 12], linestyle=(0, (4, 3)), linewidth=1, color=MUTED, zorder=1)
    ax.text(11.9, 11.3, "perfect", ha="right", fontsize=8, color=MUTED, rotation=33)
    for (group, part), colour, shift in zip(by_grade.groupby("subjects", sort=False), (BLUE, ORANGE), (-0.08, 0.08)):
        ax.errorbar(part["grade"] + shift, part["mean_predicted_grade"], yerr=part["sd_predicted_grade"].fillna(0),
                    color=colour, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                    elinewidth=1, capsize=2, label=group, zorder=3)
    ax.set_xticks(range(1, 13))
    ax.set_yticks(range(1, 13))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.5, 12.5)
    ax.grid(axis="y")
    ax.set_xlabel("True grade of the ScienceQA question")
    ax.set_ylabel("Predicted grade (mean ± 1 SD)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    def rho(grades: str) -> str:
        """Format the Spearman correlation of one grade range, all subjects."""
        row = summary[(summary["subjects"] == "all subjects") & summary["grades"].str.startswith(grades)]
        value = row["spearman"].iloc[0] if len(row) else np.nan
        return "n/a" if pd.isna(value) else f"{value:+.2f}"

    ax.set_title(f"ARC model on ScienceQA · {by_grade['run_id'].iloc[0]}\n"
                 f"Spearman, all subjects: inside {rho('inside')}, outside {rho('outside')}, all grades {rho('all')}",
                 fontsize=9)
    return _save(fig, "extrapolation")


def lime_runs() -> list[str]:
    """Return the runs that have a cue-word table."""
    folder = results_dir() / "explain"
    return sorted(path.name.removesuffix("_cue_words.csv") for path in folder.glob("*_cue_words.csv"))


def build_figures(run_ids: list[str] | None = None) -> list[Path]:
    """Rebuild every figure that has its data. Return the files written.

    Confusion matrices are drawn for the best run of every dataset (by dev QWK) and for `run_ids`.
    """
    confusion_runs = list(dict.fromkeys([*best_runs(), *(run_ids or [])]))
    files = [leaderboard_figure()]
    files += [confusion_figure(run_id) for run_id in confusion_runs]
    files += [cue_words_figure(run_id) for run_id in lime_runs()]
    files += [transfer_figure(), extrapolation_figure()]
    return [path for path in files if path is not None]


def main(argv: list[str] | None = None) -> None:
    """Rebuild the figures and print where they are."""
    for path in build_figures(sys.argv[1:] if argv is None else argv):
        print(path)


if __name__ == "__main__":
    main()
