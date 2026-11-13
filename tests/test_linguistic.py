import numpy as np
import pytest

pytest.importorskip("cmudict")
pytest.importorskip("pyphen")
pytest.importorskip("wordfreq")

from levellens.config import compose
from levellens.data import load_split
from levellens.features import linguistic
from levellens.models.base import build_model
from levellens.paths import cache_dir, scratch_dir
from levellens.predictor import Predictor
from levellens.results import load_predictions
from levellens.run import run


def test_features_separate_prose_from_formula_cues():
    row = linguistic.extract_features(r"Do not add $3 + \frac{1}{2}$ to 4.5.")
    assert tuple(row) == linguistic.FEATURE_NAMES
    assert row["word_count"] == 4
    assert row["sentence_count"] == 1
    assert row["number_count"] == 4
    assert row["negation_count"] == 1
    assert row["latex_operator_count"] == 2
    assert np.isfinite(list(row.values())).all()
    assert linguistic.extract_features("It isn’t true.")["negation_count"] == 1
    assert not any(linguistic.extract_features("").values())
    assert (linguistic.extract_features("xylophone")["mean_word_rarity"]
            > linguistic.extract_features("good")["mean_word_rarity"])


def test_disk_cache_preserves_content_and_row_order(monkeypatch):
    inputs = ["Count 4 dots.", "Never divide by zero."]
    first = linguistic.feature_matrix(inputs, use_cache=True)
    files = list((cache_dir() / "share-d" / "linguistic").glob("*.npy"))
    assert len(files) == 1
    with monkeypatch.context() as patch:
        def fail(text):
            pytest.fail(f"Cache hit unexpectedly extracted {text!r}")

        patch.setattr(linguistic, "extract_features", fail)
        np.testing.assert_array_equal(linguistic.feature_matrix(inputs, use_cache=True), first)

    reversed_rows = linguistic.feature_matrix(inputs[::-1], use_cache=True)
    np.testing.assert_array_equal(reversed_rows, first[::-1])
    changed = linguistic.feature_matrix(["Count 4 or 5 dots.", inputs[1]], use_cache=True)
    number = linguistic.FEATURE_NAMES.index("number_count")
    assert changed[0, number] == first[0, number] + 1


def test_lightgbm_roundtrips_through_predictor_and_runner():
    pytest.importorskip("lightgbm")
    config = compose([
        "data=toy", "model=linguistic_lightgbm", "owner=D", "device=cpu", "save_model=true",
    ])
    row = run(config)
    prediction = load_predictions(row["run_id"], folder=scratch_dir())["test"]["output"]
    reloaded = Predictor.load(row["run_id"])
    restored = reloaded.predict(load_split("toy", "test"))["proba"]

    assert set(row["metrics"]) == {"dev", "test", "test_1k"}
    assert prediction.shape == (90, 3)
    assert len(list((cache_dir() / "share-d" / "linguistic").glob("*.npy"))) == 3
    np.testing.assert_allclose(restored, prediction, atol=1e-7)
    np.testing.assert_allclose(restored.sum(axis=1), 1)


def test_lightgbm_binary_fit_aligns_missing_level_columns():
    pytest.importorskip("lightgbm")
    config = compose(["model=linguistic_lightgbm", "model.params.min_child_samples=1"])
    model = build_model(config, num_levels=3)
    train = [{"text": "Simple test.", "level": 0}, {"text": "Never add 3 or 4.", "level": 2}] * 12
    model.fit(train, [], config)
    proba = model.predict_proba(train[:2])

    np.testing.assert_array_equal(proba.argmax(axis=1), [0, 2])
    np.testing.assert_array_equal(proba[:, 1], 0)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
