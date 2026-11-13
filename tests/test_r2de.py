import numpy as np

from levellens.config import compose
from levellens.models.base import build_model
from levellens.paths import scratch_dir
from levellens.results import load_predictions
from levellens.run import run


def test_r2de_returns_continuous_scores_from_training_labels():
    config = compose(["data=toy", "model=r2de"])
    model = build_model(config, num_levels=3)
    train = [{"text": "identical question", "level": level} for level in [0, 1, 1] * 10]
    dev = [{"text": "validation token", "level": 2}] * 10
    model.fit(train, dev, config)

    assert "validation" not in model.model_.named_steps["tfidf"].vocabulary_
    score = model.predict_proba([{"text": "identical question", "level": 2}])
    assert score.shape == (1,)
    assert 0.5 < score[0] < 1.0
    assert model.predict_proba([]).shape == (0,)


def test_r2de_runner_saves_scores_and_evaluates_all_splits():
    row = run(compose(["data=toy", "model=r2de", "owner=D", "device=cpu"]))
    output = load_predictions(row["run_id"], folder=scratch_dir())["test"]["output"]

    assert row["model"] == "r2de"
    assert set(row["metrics"]) == {"dev", "test", "test_1k"}
    assert row["metrics"]["test"]["spearman_score"] is not None
    assert output.shape == (90,)
    assert np.isfinite(output).all()
    assert ((0 <= output) & (output <= 2)).all()
