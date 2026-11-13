"""Share D global TreeSHAP for the saved surface-feature LightGBM models.

From the repository root, after D4 runs with `save_model=true`:

    python -m pip install "matplotlib>=3.8,<4"
    python -m levellens.explain.global_shap
    python -m pytest tests/test_global_shap.py -q

Figures and three-sentence captions are written to
`<LEVELLENS_RESULTS>/figures/share_d_shap/`. Each subject uses its fixed
`test_1k` split. Shares are sums of mean absolute SHAP across features;
the mean weights each question and each model output class equally.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np

from levellens.data import load_split
from levellens.features.linguistic import FEATURE_GROUPS, FEATURE_NAMES
from levellens.models.classic import LinguisticLightGBM
from levellens.paths import results_dir
from levellens.predictor import Predictor


SUBJECTS = {
    "race": ("RACE++", "Reading comprehension", "阅读理解"),
    "arc": ("ARC", "Science", "科学"),
    "math": ("MATH", "Competition math", "竞赛数学"),
}
LABELS = {
    "flesch_reading_ease": ("Flesch reading ease", "Flesch 可读性"),
    "mean_word_rarity": ("Mean word rarity", "平均词语罕见度"),
    "word_count": ("Word count", "词数"),
    "character_count": ("Character count", "字符数"),
    "sentence_count": ("Sentence count", "句子数"),
    "number_count": ("Number count", "数字出现次数"),
    "negation_count": ("Negation count", "否定词出现次数"),
    "latex_operator_count": ("LaTeX operator count", "LaTeX 运算符计数"),
}
COLORS = {"language": "#126B74", "reasoning": "#B75124"}


def tree_shap(booster, matrix, *, num_threads=4):
    """Return SHAP as [questions, output classes, features] and the max reconstruction error."""
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != booster.num_feature():
        raise ValueError("Expected a non-empty feature matrix matching the booster")
    n_rows, n_features = matrix.shape
    n_outputs = booster.num_model_per_iteration()
    output = np.asarray(booster.predict(matrix, pred_contrib=True, num_threads=num_threads))
    if output.shape != (n_rows, n_outputs * (n_features + 1)):
        raise ValueError(f"Unexpected LightGBM contribution shape: {output.shape}")
    blocks = output.reshape(n_rows, n_outputs, n_features + 1)
    raw = np.asarray(booster.predict(matrix, raw_score=True, num_threads=num_threads))
    raw = raw.reshape(n_rows, n_outputs)
    reconstructed = blocks.sum(axis=-1)
    if not np.isfinite(blocks).all() or not np.allclose(reconstructed, raw, rtol=1e-7, atol=1e-8):
        raise ValueError("TreeSHAP contributions plus expected values do not recover LightGBM's raw scores")
    return blocks[:, :, :-1], float(np.max(np.abs(reconstructed - raw)))


def summarize(contributions):
    """Aggregate magnitudes before grouping so effects with opposite signs do not cancel."""
    if contributions.ndim != 3 or contributions.shape[-1] != len(FEATURE_NAMES) or not contributions.size:
        raise ValueError("Expected SHAP with axes for questions, outputs, and surface features")
    magnitudes = dict(zip(FEATURE_NAMES, np.abs(contributions).mean(axis=(0, 1)).tolist()))
    total = sum(magnitudes.values())
    shares = {name: value / total if total else 0.0 for name, value in magnitudes.items()}
    groups = {group: sum(shares[name] for name in names) for group, names in FEATURE_GROUPS.items()}
    return {"mean_abs_shap": magnitudes, "feature_share": shares, "group_share": groups, "total_mean_abs_shap": total}


def explain_subject(dataset, seed):
    run_id = f"{dataset}-linguistic_lightgbm-none-s{seed}"
    model = Predictor.load(run_id).model
    if not isinstance(model, LinguisticLightGBM):
        raise ValueError(f"{run_id} is not a D4 surface-feature model")
    items = load_split(dataset, "test_1k")
    if len(items) != 1000:
        raise ValueError(f"{dataset}: expected 1,000 questions in the fixed test subset, got {len(items)}")
    matrix = model.feature_matrix(items)
    contributions, error = tree_shap(model.booster_, matrix)
    return {
        "dataset": dataset, "run_id": run_id, "split": "test_1k", "n_questions": len(items),
        "n_outputs": contributions.shape[1], "max_additivity_error": error,
        **summarize(contributions),
    }


def captions(summary):
    """Three Chinese sentences derived entirely from one subject's measured shares."""
    group = summary["group_share"]
    shares = summary["feature_share"]
    if not summary["total_mean_abs_shap"]:
        return (
            "在固定的 1,000 道测试题上，所有特征的 SHAP 值均为零。",
            "这批题的模型原始分数都等于各等级的基准分数。",
            "绝对贡献总量为零，因此无法计算有意义的组别占比。",
        )
    ranked = sorted(shares, key=shares.get, reverse=True)
    top, second = ranked[:2]
    reasoning = max(FEATURE_GROUPS["reasoning"], key=shares.get)
    reasoning_detail = (
        f"推理线索中，{LABELS[reasoning][1]}贡献最大，占该组绝对贡献的 "
        f"{shares[reasoning] / group['reasoning']:.1%}。"
        if group["reasoning"] else "推理组的三个特征在这批题上的 SHAP 贡献均为零。"
    )
    return (
        f"在固定的 1,000 道测试题上，语言特征占绝对 SHAP 总量的 {group['language']:.1%}，"
        f"推理线索占 {group['reasoning']:.1%}。",
        f"贡献最大的两个特征分别为 {LABELS[top][1]}（{shares[top]:.1%}）"
        f"和 {LABELS[second][1]}（{shares[second]:.1%}）。",
        reasoning_detail,
    )


def draw_figure(summary, path: Path, feature_axis_limit):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    ink, muted = "#172B36", "#4D626C"
    name, subject, _ = SUBJECTS[summary["dataset"]]
    fig = Figure(figsize=(8.5, 5.4), facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(.055, .94, f"{name} | {subject}", color=ink, fontsize=21, fontweight="bold")
    fig.text(.055, .887, "Global TreeSHAP  /  fixed test subset  /  1,000 questions",
             color=muted, fontsize=10.5)

    group_ax = fig.add_axes([.055, .728, .89, .079])
    group_ax.set_axis_off()
    group_ax.set(xlim=(0, 1), ylim=(-.6, .6))
    start = 0.0
    for group, label, x, align in (
        ("language", "Language", .055, "left"), ("reasoning", "Reasoning cues", .945, "right"),
    ):
        value = summary["group_share"][group]
        group_ax.barh(0, value, left=start, height=.7, color=COLORS[group])
        fig.text(x, .811, f"{label}  {value:.1%}", ha=align, color=COLORS[group],
                 fontsize=11, fontweight="bold")
        start += value

    shares = summary["feature_share"]
    ranked = sorted(shares, key=shares.get, reverse=True)
    groups = {feature: group for group, names in FEATURE_GROUPS.items() for feature in names}
    ax = fig.add_axes([.29, .162, .64, .485])
    y = np.arange(len(ranked))
    values = [shares[name] for name in ranked]
    ax.barh(y, values, height=.62, color=[COLORS[groups[name]] for name in ranked])
    ax.set_yticks(y, [LABELS[name][0] for name in ranked], fontsize=10, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, feature_axis_limit)
    ax.set_xticks(np.arange(0, feature_axis_limit + .001, .1))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", color="#DEE5E7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7, labelcolor=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, value in enumerate(values):
        ax.text(value + .009, row, f"{value:.1%}", color=ink, va="center", fontsize=9)
    ax.set_xlabel("Share of total mean absolute SHAP", color=muted, labelpad=8)
    fig.text(.055, .045, "Raw class scores averaged equally across questions and output classes.",
             color=muted, fontsize=9)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="seed in the saved D4 run IDs")
    args = parser.parse_args(argv)
    summaries = [explain_subject(dataset, args.seed) for dataset in SUBJECTS]
    largest = max(max(summary["feature_share"].values()) for summary in summaries)
    axis_limit = (ceil(largest * 10) + 1) / 10
    folder = results_dir() / "figures" / "share_d_shap"
    folder.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# D5 · 三个学科的全局 TreeSHAP",
        "",
        "English: [summary.md](summary.md)",  # the English page is written by hand; this file is the Chinese twin
        "",
        "每个模型使用团队固定的测试子集。对每题、每个等级的原始分数求 TreeSHAP，"
        "再对题目和等级等权平均绝对值。占比描述模型的打分依据。",
    ]
    for summary in summaries:
        dataset = summary["dataset"]
        name, _, subject = SUBJECTS[dataset]
        path = folder / f"{dataset}_global_shap.png"
        draw_figure(summary, path, axis_limit)
        markdown += ["", f"## {name} · {subject}", "", f"[图]({path.name})", "", *captions(summary)]
        print(f"{name}: {path}")
    (folder / "summary.zh-CN.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
