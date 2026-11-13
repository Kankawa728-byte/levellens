import numpy as np
import pytest

from levellens.config import compose
from levellens.data import load_split
from levellens.models.base import build_model
from levellens.paths import scratch_dir
from levellens.results import load_predictions
from levellens.run import run


def test_random_floor_uses_training_frequencies_and_preserves_item_draw():
    config = compose(["model=stratified_random", "seed=17"])
    model = build_model(config, num_levels=3)
    model.fit([{"level": 0}] * 3 + [{"level": 2}], [{"level": 1}], config)
    items = [{"id": f"q{i}", "dataset": "toy"} for i in range(5000)]

    full = model.predict_proba(items)
    assert np.all(full.sum(axis=1) == 1)
    assert full[:, 1].sum() == 0
    assert abs(full[:, 2].mean() - .25) < .03
    np.testing.assert_array_equal(model.predict_proba(items[30:18:-3]), full[30:18:-3])


def test_subject_prior_ties_and_unseen_topics():
    def item(level, topic):
        return {"dataset": "math", "level": level, "meta": {"type": topic}}

    config = compose(["data=math", "model=subject_prior"])
    model = build_model(config, num_levels=5)
    train = [item(4, "algebra"), item(4, "algebra"), item(0, "geometry"), item(1, "geometry")]
    model.fit(train, [item(3, "geometry")], config)
    result = model.predict_proba([item(None, "geometry"), item(None, "unknown"), item(None, None)])
    assert result.shape == (3, 5)
    np.testing.assert_array_equal(result.argmax(axis=1), [1, 4, 4])
    with pytest.raises(ValueError, match="only to MATH"):
        model.predict_proba([{"dataset": "arc"}])


def test_random_floor_runner_writes_predictions_and_consistent_subset():
    row = run(compose(["data=toy", "model=stratified_random", "owner=D", "device=cpu"]))
    predictions = load_predictions(row["run_id"], folder=scratch_dir())
    assert set(row["metrics"]) == {"dev", "test", "test_1k"}
    assert predictions["test"]["output"].shape == (90, 3)
    config = compose(["model=stratified_random"])
    model = build_model(config, num_levels=3)
    model.fit(load_split("toy", "train"), [], config)
    subset = load_split("toy", "test_1k")
    test = predictions["test"]
    row_by_id = {item_id: i for i, item_id in enumerate(test["ids"])}
    expected = test["output"][[row_by_id[item["id"]] for item in subset]]
    np.testing.assert_array_equal(model.predict_proba(subset), expected)
