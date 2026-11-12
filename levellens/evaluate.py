"""All metrics and the bootstrap intervals. Every number on a slide is computed here.

    from levellens.evaluate import evaluate
    metrics = evaluate(y_true, model.predict_proba(items), num_levels=5)

A model output is either
- an array of shape [n, K] with one probability row per question, or
- an array of shape [n] with one number per question (regression, R2DE). The number is
  rounded to the nearest level and clipped to 0..K-1, and the probability row is one-hot.

Metrics (levels are the integers 0..K-1):
    accuracy            share of questions with the exact level
    macro_f1            F1 per level, then the plain mean over levels
    adjacent_accuracy   share of questions predicted within one level
    qwk                 quadratic weighted kappa: agreement that punishes far misses more
    mae, rmse           error on the level index
    spearman            rank correlation between the true level and the predicted level
    spearman_score      the same with the continuous score (expected level, or the regression output)
    drps                discrete ranked probability score: squared error between the two cumulative
                        distributions, summed over the first K-1 levels, mean over questions
    balanced_drps       the same, averaged per true level first, so every level counts equally.
                        Definition and code follow Thuy et al. (2025), github.com/arthur-thuy/qde-ordinality (MIT).
    ece                 expected calibration error of the top level, 15 equal-width bins
A metric that is undefined, such as a correlation with a constant prediction, is None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

ECE_BINS = 15
PROBA_TOLERANCE = 1e-4
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


@dataclass
class Scored:
    """True levels next to the three views of a model output."""

    y: np.ndarray  # [n] true level
    level: np.ndarray  # [n] predicted level
    proba: np.ndarray  # [n, K] probability of each level
    score: np.ndarray  # [n] continuous score
    num_levels: int

    def take(self, index: np.ndarray) -> "Scored":
        """Return the rows listed in `index` (repeats allowed, as in a bootstrap sample)."""
        return Scored(self.y[index], self.level[index], self.proba[index], self.score[index], self.num_levels)


def interpret(output, num_levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a model output into the predicted level [n], the probabilities [n, K] and the score [n]."""
    output = np.asarray(output, dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("The model output contains NaN or infinite values")
    if output.ndim == 1:
        level = np.clip(np.floor(output + 0.5), 0, num_levels - 1).astype(int)
        return level, np.eye(num_levels)[level], output
    if output.ndim != 2 or output.shape[1] != num_levels:
        raise ValueError(f"Expected shape [n, {num_levels}] or [n], got {output.shape}")
    if (output < -PROBA_TOLERANCE).any() or not np.allclose(output.sum(axis=1), 1.0, atol=PROBA_TOLERANCE):
        raise ValueError("Each probability row must be non-negative and sum to 1")
    return output.argmax(axis=1), output, output @ np.arange(num_levels)


def score_output(y_true, output, num_levels: int) -> Scored:
    """Check the true levels and the model output and bundle them for the metric functions."""
    y = np.asarray(y_true)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= num_levels:
        raise ValueError(f"True levels must be integers in 0..{num_levels - 1}")
    level, proba, score = interpret(output, num_levels)
    if len(level) != len(y):
        raise ValueError(f"{len(y)} true levels but {len(level)} predictions")
    return Scored(y, level, proba, score, num_levels)


def _or_none(value: float) -> float | None:
    """Return a plain float, or None when the value is not a number."""
    return None if np.isnan(value) else float(value)


def accuracy(s: Scored) -> float:
    return float(np.mean(s.level == s.y))


def macro_f1(s: Scored) -> float:
    return float(f1_score(s.y, s.level, average="macro", zero_division=0))


def adjacent_accuracy(s: Scored) -> float:
    return float(np.mean(np.abs(s.level - s.y) <= 1))


def qwk(s: Scored) -> float | None:
    if np.ptp(s.y) == 0 and np.ptp(s.level) == 0:
        return None  # one level only on both sides: kappa is 0/0
    return _or_none(cohen_kappa_score(s.y, s.level, weights="quadratic", labels=np.arange(s.num_levels)))


def mae(s: Scored) -> float:
    return float(np.mean(np.abs(s.level - s.y)))


def rmse(s: Scored) -> float:
    return float(np.sqrt(np.mean((s.level - s.y) ** 2)))


def _spearman(y: np.ndarray, x: np.ndarray) -> float | None:
    if np.ptp(y) == 0 or np.ptp(x) == 0:
        return None  # a constant has no ranks to correlate
    return _or_none(spearmanr(y, x)[0])


def spearman(s: Scored) -> float | None:
    return _spearman(s.y, s.level)


def spearman_score(s: Scored) -> float | None:
    return _spearman(s.y, s.score)


def _rps_per_item(s: Scored) -> np.ndarray:
    """Ranked probability score of each question: squared gap between the cumulative distributions."""
    cumulative_pred = np.cumsum(s.proba, axis=1)[:, :-1]
    cumulative_true = (np.arange(s.num_levels - 1)[None, :] >= s.y[:, None]).astype(float)
    return ((cumulative_pred - cumulative_true) ** 2).sum(axis=1)


def drps(s: Scored) -> float:
    return float(_rps_per_item(s).mean())


def balanced_drps(s: Scored) -> float:
    per_item = _rps_per_item(s)
    return float(np.mean([per_item[s.y == level].mean() for level in np.unique(s.y)]))


def ece(s: Scored) -> float:
    confidence = s.proba.max(axis=1)
    correct = (s.level == s.y).astype(float)
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    total = 0.0
    for b in np.unique(bins):
        inside = bins == b
        total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


METRICS: dict[str, Callable[[Scored], float | None]] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "adjacent_accuracy": adjacent_accuracy,
    "qwk": qwk,
    "mae": mae,
    "rmse": rmse,
    "spearman": spearman,
    "spearman_score": spearman_score,
    "drps": drps,
    "balanced_drps": balanced_drps,
    "ece": ece,
}


def evaluate(y_true, output, num_levels: int) -> dict:
    """Return every metric for one model output, plus precision, recall and support per level."""
    s = score_output(y_true, output, num_levels)
    metrics = {"n": int(len(s.y))}
    metrics.update({name: function(s) for name, function in METRICS.items()})
    precision, recall, _, support = precision_recall_fscore_support(
        s.y, s.level, labels=np.arange(num_levels), zero_division=0
    )
    metrics["per_level"] = {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "support": [int(v) for v in support],
    }
    return metrics


def _resample(s: Scored, metric: str, samples: int, seed: int):
    """Yield `samples` bootstrap resamples of the question indices."""
    if metric not in METRICS:
        raise KeyError(f"Unknown metric {metric!r}. Known: {sorted(METRICS)}")
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        yield rng.integers(0, len(s.y), size=len(s.y))


def _interval(values: list, confidence: float) -> tuple[float | None, float | None]:
    """Return the percentile interval of the values, ignoring undefined ones."""
    values = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.isnan(values).all():
        return None, None
    tail = 100 * (1 - confidence) / 2
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_interval(
    y_true, output, num_levels: int, metric: str = "qwk",
    samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED, confidence: float = 0.95,
) -> dict:
    """Return a metric with its bootstrap confidence interval: {"value", "low", "high"}."""
    s = score_output(y_true, output, num_levels)
    values = [METRICS[metric](s.take(index)) for index in _resample(s, metric, samples, seed)]
    low, high = _interval(values, confidence)
    return {"metric": metric, "value": METRICS[metric](s), "low": low, "high": high, "samples": samples}


def paired_bootstrap(
    y_true, output_a, output_b, num_levels: int, metric: str = "qwk",
    samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED, confidence: float = 0.95,
) -> dict:
    """Compare two models on the same questions.

    Each resample draws the same questions for both models, so the interval is for the
    difference A minus B. If the interval lies above zero, A is better on a higher-is-better metric.
    """
    a = score_output(y_true, output_a, num_levels)
    b = score_output(y_true, output_b, num_levels)
    differences = []
    for index in _resample(a, metric, samples, seed):
        value_a, value_b = METRICS[metric](a.take(index)), METRICS[metric](b.take(index))
        differences.append(None if value_a is None or value_b is None else value_a - value_b)
    low, high = _interval(differences, confidence)
    value_a, value_b = METRICS[metric](a), METRICS[metric](b)
    difference = None if value_a is None or value_b is None else value_a - value_b
    return {"metric": metric, "a": value_a, "b": value_b, "difference": difference, "low": low, "high": high, "samples": samples}
