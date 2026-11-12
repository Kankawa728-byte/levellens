"""Config, one run from start to finish, the results store, the tables and the Predictor."""

import json

import numpy as np
import pytest

from levellens.config import compose
from levellens.data import load_split, split_path
from levellens.paths import results_dir, scratch_dir
from levellens.predictor import Predictor
from levellens.results import ROW_FIELDS, append_run, build_tables, load_predictions, load_rows, load_runs, runs_file
from levellens.run import run


def test_compose_picks_files_and_applies_overrides():
    config = compose(["data=arc_balanced", "seed=3", "model.params.alpha=0.5", "owner=D"])
    assert config.data.name == "arc" and config.data.train_split == "train_balanced"
    assert config.model.target == "levellens.models.floors.Majority"
    assert config.seed == 3 and config.model.params.alpha == 0.5 and config.owner == "D"
    assert config.head.name == "none"


def test_compose_reports_unknown_names():
    with pytest.raises(FileNotFoundError, match="majority"):
        compose(["model=does-not-exist"])
    with pytest.raises(ValueError):
        compose(["seed"])


def test_run_writes_one_row_and_the_predictions():
    row = run(compose(["data=toy", "model=majority"]))
    assert set(ROW_FIELDS) <= set(row)
    assert row["run_id"] == "toy-majority-none-s0" and row["dataset"] == "toy"
    assert set(row["metrics"]) == {"dev", "test", "test_1k"}
    assert row["metrics"]["test"]["accuracy"] == pytest.approx(1 / 3)

    lines = runs_file(scratch_dir()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["run_id"] == row["run_id"]
    assert not runs_file().exists()  # toy runs are try-outs: they stay out of the shared store

    predictions = load_predictions(row["run_id"], scratch_dir())
    assert set(predictions) == {"dev", "test"}  # test_1k is cut from test, not stored twice
    assert predictions["test"]["output"].shape == (90, 3)
    assert predictions["test"]["ids"].tolist() == [item["id"] for item in load_split("toy", "test")]


def test_variant_and_limit_change_the_run_id():
    row = run(compose(["data=toy", "model=majority", "variant=demo", "limit_train=30", "seed=2"]))
    assert row["run_id"] == "toy-majority-none-demo-n30-s2" and row["variant"] == "demo-n30"


def test_newest_row_of_a_run_id_counts_and_tables_are_built():
    run(compose(["data=toy", "model=majority"]))
    run(compose(["data=toy", "model=majority"]))
    run(compose(["data=toy", "model=majority", "seed=1"]))
    store = scratch_dir()
    assert len(runs_file(store).read_text(encoding="utf-8").splitlines()) == 3
    assert [row["run_id"] for row in load_rows(store)] == ["toy-majority-none-s0", "toy-majority-none-s1"]

    runs = load_runs(store)
    assert len(runs) == 2 and "metrics.test.qwk" in runs.columns
    written = build_tables(store)
    board = (store / "tables" / "leaderboards.md").read_text(encoding="utf-8")
    assert "## toy, test" in board and "| majority | none |" in board and "0.333 ± 0.000" in board
    assert {path.name for path in written} == {"toy_test.csv", "toy_test_1k.csv", "toy_dev.csv", "leaderboards.md"}


def test_append_run_rejects_incomplete_rows_and_nan():
    with pytest.raises(ValueError, match="lacks"):
        append_run({"run_id": "x"})
    row = run(compose(["data=toy", "model=majority"]))
    row["metrics"]["test"]["qwk"] = float("nan")
    with pytest.raises(ValueError):
        append_run(row)


def test_stub_predictor_returns_probabilities():
    predictor = Predictor.load("stub")
    result = predictor.predict(["What is 2 + 2? A. 3 B. 4 C. 5 D. 6", "word " * 80])
    assert result["proba"].shape == (2, 3) and np.allclose(result["proba"].sum(axis=1), 1.0)
    assert result["level_names"] == ["middle school", "high school", "college"]
    assert result["level"].tolist() == [0, 2]  # the stub reads longer questions as harder
    assert (result["expected_level"] >= 0).all() and (result["expected_level"] <= 2).all()
    assert Predictor.load("stub", dataset="arc").predict(["Q?"])["proba"].shape == (1, 7)
    again = predictor.predict(["What is 2 + 2? A. 3 B. 4 C. 5 D. 6"])
    assert np.array_equal(again["proba"][0], result["proba"][0])


def test_saved_model_loads_through_the_predictor():
    row = run(compose(["data=toy", "model=majority", "save_model=true"]))
    predictor = Predictor.load(row["run_id"])
    result = predictor.predict(load_split("toy", "test")[:5])
    assert result["level"].tolist() == [2] * 5 and result["level_names"] == ["easy", "medium", "hard"]
    with pytest.raises(FileNotFoundError):
        Predictor.load("no-such-run")


@pytest.mark.data
@pytest.mark.parametrize(
    "data, expected",
    [
        ("race", {"accuracy": 0.620, "rmse": 0.616, "balanced_drps": 0.667}),
        ("arc_balanced", {"accuracy": 0.409, "rmse": 2.195, "balanced_drps": 2.286}),
    ],
)
def test_majority_reproduces_the_published_rows_on_our_splits(data, expected):
    """Our splits and our metric code give the majority rows of Thuy et al. (2025). Needs the prepared data."""
    name = compose([f"data={data}"]).data.name
    if not split_path(name, "test").exists():
        pytest.skip(f"{name} is not prepared: python -m levellens.data.prepare {name}")
    row = run(compose([f"data={data}", "model=majority"]))
    for metric, value in expected.items():
        assert round(row["metrics"]["test"][metric], 3) == value
    assert [stored["run_id"] for stored in load_rows()] == [row["run_id"]]  # real datasets use the shared store
    assert runs_file() == results_dir() / "runs.jsonl"
