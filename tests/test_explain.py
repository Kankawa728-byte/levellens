"""Share F: LIME, the error list, transfer and extrapolation, the figures and the app.

Everything runs on the stub predictor and the toy dataset: no download, no trained model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lime")
pytest.importorskip("matplotlib")

from levellens import figures  # noqa: E402
from levellens.data import load_split  # noqa: E402
from levellens.explain import errors, local, transfer  # noqa: E402
from levellens.models.base import Model  # noqa: E402
from levellens.predictor import Predictor  # noqa: E402
from levellens.results import ROW_FIELDS, append_run, save_predictions  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"


class KeywordModel(Model):
    """Says the top level when the text contains "photosynthesis", the bottom level otherwise."""

    def predict_proba(self, items) -> np.ndarray:
        hard = np.array(["photosynthesis" in item["text"] for item in items])
        proba = np.full((len(items), self.num_levels), 0.1 / (self.num_levels - 1))
        proba[np.arange(len(items)), np.where(hard, self.num_levels - 1, 0)] = 0.9
        return proba


def store_run(run_id: str, y, output, dataset: str = "toy", dev_qwk: float = 0.5) -> None:
    """Put a made-up run with test predictions into the temporary results store."""
    row = dict.fromkeys(ROW_FIELDS, 0) | {
        "run_id": run_id, "dataset": dataset, "model": "made-up", "head": "ce", "variant": "", "owner": "F",
        "config": {}, "predict_seconds": {}, "started": "", "device": "cpu", "git_commit": "",
        "metrics": {"dev": {"n": 3, "qwk": dev_qwk}, "test_1k": {"n": len(y), "qwk": dev_qwk}},
    }
    append_run(row)
    ids = [item["id"] for item in load_split(dataset, "test")[: len(y)]]
    save_predictions(run_id, {"test": {"ids": ids, "y": y, "output": output}})


# ---- F1: LIME -------------------------------------------------------------------------------

def test_lime_finds_the_word_the_model_listens_to():
    predictor = Predictor(KeywordModel(num_levels=3), ["low", "middle", "high"])
    result = local.explain(predictor, "Which gas does photosynthesis release in a leaf", num_samples=300)
    assert result["weights"].shape == (len(result["words"]), 3) and result["expected"].shape == (len(result["words"]),)
    assert result["words"][int(result["expected"].argmax())] == "photosynthesis"
    weights = dict(zip(result["words"], result["weights"]))
    assert weights["photosynthesis"][2] > 0.5 and weights["photosynthesis"][0] < -0.5  # 0.9 - 0.05 = 0.85 in theory
    assert abs(weights["leaf"][2]) < 0.1


def test_lime_handles_a_text_without_words_and_repeats_itself():
    predictor = Predictor.load("stub", dataset="toy")
    assert local.explain(predictor, "?!")["words"] == []
    first = local.explain(predictor, "What is two plus two", num_samples=100, seed=1)
    again = local.explain(predictor, "What is two plus two", num_samples=100, seed=1)
    assert np.allclose(first["weights"], again["weights"])


def test_cue_word_summary_drops_stop_words_and_option_letters():
    predictor = Predictor(KeywordModel(num_levels=3), ["low", "middle", "high"])
    items = [{"id": "q1", "level": 2, "text": "Where does photosynthesis happen? A. the leaf B. the root"},
             {"id": "q2", "level": 0, "text": "Which is a leaf? A. this B. that"}]
    weights = local.explain_items(predictor, items, num_samples=200)
    summary = local.summarise(weights, predictor.level_names)
    assert not {"the", "a", "b", "is"} & set(summary["word"])
    top = summary[(summary["level"] == 2) & (summary["rank"] == 1)].iloc[0]
    assert top["word"] == "photosynthesis" and top["level_name"] == "high" and top["items"] == 1
    assert (summary["weight"] > 0).all()


# ---- F2: errors -----------------------------------------------------------------------------

def test_confusion_and_recall():
    counts = errors.confusion([0, 0, 1, 2, 2, 2], [0, 1, 1, 0, 2, 2], num_levels=4)
    assert counts.tolist() == [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]]
    recall = errors.recall_per_level(counts)
    assert np.allclose(recall[:3], [0.5, 1.0, 2 / 3]) and np.isnan(recall[3])


def test_worst_errors_are_ordered_by_gap_then_confidence_then_id(isolated_folders):
    y = [2, 2, 0, 1, 2, 0]
    output = [[0.9, 0.1, 0.0],   # gap 2, confidence 0.9
              [0.6, 0.3, 0.1],   # gap 2, confidence 0.6
              [0.1, 0.2, 0.7],   # gap 2, confidence 0.7
              [0.8, 0.1, 0.1],   # gap 1
              [0.0, 0.1, 0.9],   # right
              [0.9, 0.1, 0.0]]   # right
    store_run("toy-made-up-ce-s0", y, output)
    table = errors.worst_errors("toy-made-up-ce-s0", n=4)
    assert table["abs_error"].tolist() == [2, 2, 2, 1]
    assert table["confidence"].tolist() == pytest.approx([0.9, 0.7, 0.6, 0.8])
    assert table["item_id"].tolist() == [item["id"] for item in load_split("toy", "test")[:4]][0:1] + \
        [load_split("toy", "test")[2]["id"], load_split("toy", "test")[1]["id"], load_split("toy", "test")[3]["id"]]

    listed, (public, private) = errors.audit_list(["toy-made-up-ce-s0"], n=2)
    assert len(listed) == 2 and len(listed["predictions_sha256"].iloc[0]) == 64
    assert "text" not in pd.read_csv(public).columns          # the public file holds no dataset text
    assert listed["item_id"].iloc[0] in private.read_text(encoding="utf-8")
    assert isolated_folders / "cache" in private.parents      # the full questions stay out of the repository


def test_best_runs_picks_the_best_dev_qwk_per_dataset():
    store_run("toy-a-ce-s0", [0, 1], [[1, 0, 0], [0, 1, 0]], dev_qwk=0.2)
    store_run("toy-b-ce-s0", [0, 1], [[1, 0, 0], [0, 1, 0]], dev_qwk=0.7)
    assert errors.best_runs() == ["toy-b-ce-s0"]


# ---- F3, F4: transfer and extrapolation ------------------------------------------------------

def test_rank_correlation():
    assert transfer.rank_correlation([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.9]) == pytest.approx(1.0)
    assert transfer.rank_correlation([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)
    assert transfer.rank_correlation([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_transfer_matrix_has_a_line_per_model_and_test_set(monkeypatch):
    monkeypatch.setattr(transfer, "SUBJECTS", ("toy",))
    predictor = Predictor.load("stub", dataset="toy")
    matrix = transfer.transfer_matrix({"toy": "stub"}, split="test", predictors={"toy": predictor})
    assert matrix[["trained_on", "tested_on", "n"]].values.tolist() == [["toy", "toy", 90]]
    assert -1 <= matrix["spearman"].iloc[0] <= 1


def test_extrapolation_splits_the_grades_inside_and_outside_training():
    # an "ARC model" whose expected level grows with the length of the question: grade = 3 + level
    predictor = Predictor.load("stub", dataset="arc")
    items = [{"id": f"s{grade}-{copy}", "text": "word " * (5 * grade + copy), "meta": {"grade": grade, "subject": subject}}
             for grade in range(1, 13) for copy, subject in enumerate(["natural science", "language science"])]
    summary, by_grade = transfer.extrapolate("stub", predictor=predictor, items=items)
    rows = summary[summary["subjects"] == "all subjects"].set_index("grades")
    assert rows["n"].to_dict() == {"inside 3-9": 14, "outside 3-9": 10, "below 3": 4, "above 9": 6, "all grades": 24}
    assert rows.loc["inside 3-9", "spearman"] > 0.8
    assert by_grade["mean_predicted_grade"].between(3, 9).all()   # the model cannot leave its own grades
    assert by_grade.loc[by_grade["subjects"] == "all subjects", "inside"].sum() == 7


# ---- figures ---------------------------------------------------------------------------------

def test_every_figure_is_rebuilt_from_the_store(isolated_folders):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=60)
    store_run("toy-made-up-ce-s0", y, rng.dirichlet(np.ones(3), size=60))
    predictor = Predictor(KeywordModel(num_levels=3), ["easy", "medium", "hard"])
    items = load_split("toy", "test")[:3]
    summary = local.summarise(local.explain_items(predictor, items, num_samples=50), predictor.level_names)
    folder = isolated_folders / "results" / "explain"
    folder.mkdir(parents=True)
    summary.to_csv(folder / "toy-made-up-ce-s0_cue_words.csv", index=False)
    pd.DataFrame({"trained_on": ["arc", "arc", "arc"], "run_id": "r", "tested_on": ["race", "arc", "math"],
                  "split": "test_1k", "spearman": [0.2, 0.6, None], "n": 3}).to_csv(
        transfer.folder() / "transfer.csv", index=False)
    stub = Predictor.load("stub", dataset="arc")
    sqa = [{"id": str(g), "text": "word " * 4 * g, "meta": {"grade": g, "subject": "natural science"}} for g in range(1, 13)]
    for name, table in zip(("extrapolation_summary.csv", "extrapolation_by_grade.csv"),
                           transfer.extrapolate("stub", predictor=stub, items=sqa)):
        table.to_csv(transfer.folder() / name, index=False)

    files = figures.build_figures()
    names = sorted(path.name for path in files)
    assert names == ["confusion_toy-made-up-ce-s0.png", "cue_words_toy-made-up-ce-s0.png",
                     "extrapolation.png", "leaderboard.png", "transfer.png"]
    assert all(path.stat().st_size > 5000 for path in files)


def test_figures_without_data_are_skipped():
    assert figures.build_figures() == []


# ---- F6, F7: the app --------------------------------------------------------------------------

def test_calibrate_page_scores_explains_and_rescores_an_edit():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].value == "stub · race"          # no saved model in the temporary cache
    assert [metric.label for metric in app.metric] == ["Most likely level", "Expected level", "Original", "Edited"]
    assert not app.metric[3].delta

    app.text_area(key="edited_race_question").set_value("Why? " + "very long question " * 30).run()
    assert not app.exception
    assert app.metric[3].delta.endswith("levels") and float(app.metric[3].value) > float(app.metric[2].value)


def test_highlight_escapes_html_and_shades_weighted_words():
    import importlib.util

    testing = pytest.importorskip("streamlit.testing.v1")
    del testing
    spec = importlib.util.spec_from_file_location("calibrate_page", APP / "streamlit_app.py")
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)   # outside `streamlit run` the widgets are no-ops
    shaded = page.highlight("Is <b>mitosis</b> fast?", {"mitosis": 0.4, "fast": -0.1})
    assert "&lt;b&gt;" in shaded and "<b>" not in shaded
    assert f"rgba({page.HARDER}, 0.60)" in shaded and f"rgba({page.EASIER}, 0.15)" in shaded


def test_results_page_shows_the_leaderboard_and_a_confusion_matrix():
    testing = pytest.importorskip("streamlit.testing.v1")
    rng = np.random.default_rng(0)
    store_run("toy-made-up-ce-s0", rng.integers(0, 3, size=30), rng.dirichlet(np.ones(3), size=30))
    app = testing.AppTest.from_file(str(APP / "pages" / "results.py"), default_timeout=60).run()
    assert not app.exception
    assert app.dataframe[0].value["Model"].tolist() == ["made-up"]
    assert app.selectbox[0].value == "toy-made-up-ce-s0"
