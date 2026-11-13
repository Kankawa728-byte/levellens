import numpy as np
import pytest

from levellens.config import compose
from levellens.features.tfidf import make_tfidf
from levellens.models.classic import _training_folds
from levellens.paths import scratch_dir
from levellens.results import load_predictions
from levellens.run import run


def test_feature_union_extracts_both_ngram_families():
    vectorizer = make_tfidf(min_df=1)
    matrix = vectorizer.fit_transform(["solve x squared", "find x squared"])
    names = set(vectorizer.get_feature_names_out())
    assert {"word__x", "word__x squared", "char__squ", "char__squar"} <= names
    assert matrix.shape == (2, len(names))


def test_race_passages_stay_in_one_fold():
    train = [
        {"dataset": "race", "meta": {"source": str(level), "passage_id": str(passage)}}
        for level in range(3) for passage in range(15) for _ in range(2)
    ]
    labels = np.repeat(np.arange(3), 30)
    folds = _training_folds(train, labels, seed=0)
    assert len(folds) == 5
    for fit, held_out in folds:
        assert set(fit // 2).isdisjoint(held_out // 2)
        assert set(labels[held_out]) == {0, 1, 2}


@pytest.mark.parametrize(("name", "tune", "parameter"), [
    ("tfidf_logreg", False, "C"),
    ("tfidf_logreg", True, "C"),
    ("tfidf_svm", True, "C"),
    ("tfidf_cnb", True, "alpha"),
])
def test_runner_records_predictions_and_cv_selection(name, tune, parameter):
    config = compose([
        "data=toy", f"model={name}", "owner=D", "device=cpu",
        f"model.params.tune={str(tune).lower()}",
    ])
    row = run(config)
    predictions = load_predictions(row["run_id"], folder=scratch_dir())
    output = predictions["test"]["output"]
    assert output.shape == (90, 3)
    assert np.isfinite(output).all()
    assert (output >= 0).all()
    np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6, rtol=0)
    assert set(row["metrics"]) == {"dev", "test", "test_1k"}
    if tune:
        selection = row["config"]["model"]["cv_selection"]
        assert selection["folds"] == 5
        assert selection["metric"] == "qwk"
        assert selection["params"][parameter] in config.model.params.grid
    else:
        assert "cv_selection" not in row["config"]["model"]
