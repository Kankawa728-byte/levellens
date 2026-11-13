import numpy as np
import pytest

pytest.importorskip("cmudict")
pytest.importorskip("pyphen")
pytest.importorskip("wordfreq")

from levellens.explain.global_shap import summarize, tree_shap
from levellens.features.linguistic import FEATURE_NAMES


def test_group_shares_keep_opposite_signs_and_exclude_zero_cues():
    values = np.zeros((2, 2, len(FEATURE_NAMES)))
    values[:, :, FEATURE_NAMES.index("word_count")] = [[-3, 3], [3, -3]]
    values[:, :, FEATURE_NAMES.index("number_count")] = 1

    summary = summarize(values)

    assert summary["mean_abs_shap"]["word_count"] == 3
    assert summary["feature_share"]["negation_count"] == 0
    assert summary["group_share"] == {"language": .75, "reasoning": .25}
    assert summarize(np.zeros_like(values))["total_mean_abs_shap"] == 0


@pytest.mark.parametrize("classes", [2, 3])
def test_native_tree_shap_reconstructs_lightgbm_raw_scores(classes):
    lightgbm = pytest.importorskip("lightgbm")
    labels = np.tile(np.arange(classes), 12)
    matrix = np.column_stack((labels, np.arange(len(labels)) % 7)).astype(float)
    model = lightgbm.LGBMClassifier(n_estimators=5, min_child_samples=2, n_jobs=1, verbosity=-1)
    model.fit(matrix, labels)

    values, error = tree_shap(model.booster_, matrix[:7], num_threads=1)

    assert values.shape == (7, classes if classes > 2 else 1, 2)
    assert np.abs(values[:, :, 0]).max() > 0
    assert error < 1e-8
