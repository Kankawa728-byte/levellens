"""Metrics: shapes, value ranges, hand-computed cases and the published majority rows."""

import numpy as np
import pytest

from levellens.evaluate import bootstrap_interval, evaluate, interpret, paired_bootstrap

# Test-set sizes per level, from levellens/data/manifest.json
RACE_TEST = [1436, 3498, 708]
ARC_TEST = [130, 355, 751, 154, 408, 1444, 288]


def levels(counts):
    """Return a label array with `counts[k]` questions of level k."""
    return np.repeat(np.arange(len(counts)), counts)


def always(level, n, num_levels):
    """Return the output of a model that always predicts one level."""
    output = np.zeros((n, num_levels))
    output[:, level] = 1.0
    return output


def test_perfect_prediction():
    y = levels([5, 7, 3])
    metrics = evaluate(y, np.eye(3)[y], 3)
    assert metrics["n"] == 15
    for name in ("accuracy", "macro_f1", "adjacent_accuracy", "qwk", "spearman", "spearman_score"):
        assert metrics[name] == pytest.approx(1.0)
    for name in ("mae", "rmse", "drps", "balanced_drps", "ece"):
        assert metrics[name] == pytest.approx(0.0)
    assert metrics["per_level"]["support"] == [5, 7, 3]


def test_published_majority_row_race():
    """Thuy et al. (2025), RACE++ majority baseline: accuracy 0.620, RMSE 0.616, Balanced DRPS 0.667."""
    y = levels(RACE_TEST)
    metrics = evaluate(y, always(1, len(y), 3), 3)
    assert round(metrics["accuracy"], 3) == 0.620
    assert round(metrics["rmse"], 3) == 0.616
    assert round(metrics["balanced_drps"], 3) == 0.667
    assert metrics["qwk"] == pytest.approx(0.0) and metrics["spearman"] is None


def test_published_majority_row_arc():
    """Thuy et al. (2025), ARC majority baseline (grade 8): accuracy 0.409, RMSE 2.195, Balanced DRPS 2.286."""
    y = levels(ARC_TEST)
    metrics = evaluate(y, always(5, len(y), 7), 7)
    assert round(metrics["accuracy"], 3) == 0.409
    assert round(metrics["rmse"], 3) == 2.195
    assert round(metrics["balanced_drps"], 3) == 2.286


def test_drps_and_ece_by_hand():
    y = np.array([0, 2])
    output = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    metrics = evaluate(y, output, 3)
    # question 1: cumulative (0.5, 0.8) against (1, 1) -> 0.25 + 0.04; question 2: (0.1, 0.3) against (0, 0) -> 0.01 + 0.09
    assert metrics["drps"] == pytest.approx((0.29 + 0.10) / 2)
    assert metrics["balanced_drps"] == pytest.approx((0.29 + 0.10) / 2)
    # both correct; confidences 0.5 and 0.7 fall in different bins: 0.5 * |1 - 0.5| + 0.5 * |1 - 0.7|
    assert metrics["ece"] == pytest.approx(0.4)
    assert metrics["spearman_score"] == pytest.approx(1.0)


def test_balanced_drps_weights_every_level_equally():
    y = levels([9, 1])
    metrics = evaluate(y, always(0, 10, 2), 2)
    assert metrics["drps"] == pytest.approx(0.1)
    assert metrics["balanced_drps"] == pytest.approx(0.5)


def test_regression_output_is_rounded_and_clipped():
    level, proba, score = interpret(np.array([-0.7, 0.49, 0.5, 1.5, 9.0]), 3)
    assert level.tolist() == [0, 0, 1, 2, 2]
    assert proba.shape == (5, 3) and np.allclose(proba.sum(axis=1), 1.0)
    assert score.tolist() == [-0.7, 0.49, 0.5, 1.5, 9.0]


def test_adjacent_accuracy_and_qwk_order():
    y = np.array([0, 0, 2, 2])
    near = evaluate(y, np.array([1.0, 0.0, 1.0, 2.0]), 3)
    far = evaluate(y, np.array([2.0, 0.0, 0.0, 2.0]), 3)
    assert near["accuracy"] == far["accuracy"] == 0.5
    assert near["adjacent_accuracy"] == 1.0 and far["adjacent_accuracy"] == 0.5
    assert near["qwk"] > far["qwk"]


@pytest.mark.parametrize(
    "output",
    [np.ones((4, 2)), np.full((4, 3), 0.5), np.array([[0.5, 0.5, np.nan]] * 4), np.ones((4, 3, 1))],
)
def test_bad_outputs_are_rejected(output):
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1, 2, 1]), output, 3)


def test_bad_labels_are_rejected():
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1, 3]), np.eye(3), 3)
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.eye(3), 3)


def test_bootstrap_interval_contains_the_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    noisy = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, size=300))
    result = bootstrap_interval(y, np.eye(3)[noisy], 3, metric="qwk", samples=200)
    assert result["low"] < result["value"] < result["high"]
    assert 0.0 < result["high"] - result["low"] < 0.3


def test_paired_bootstrap_separates_a_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=400)
    good = np.where(rng.random(400) < 0.9, y, rng.integers(0, 3, size=400))
    weak = np.where(rng.random(400) < 0.5, y, rng.integers(0, 3, size=400))
    result = paired_bootstrap(y, np.eye(3)[good], np.eye(3)[weak], 3, metric="qwk", samples=200)
    assert result["difference"] > 0 and result["low"] > 0
    same = paired_bootstrap(y, np.eye(3)[good], np.eye(3)[good], 3, metric="accuracy", samples=50)
    assert same["difference"] == 0 and same["low"] == same["high"] == 0
