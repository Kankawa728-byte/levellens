"""Share E: select the demo model and measure export latency on the CPU."""

import json

import numpy as np

from levellens import export


ROWS = [
    {"run_id": "race-best", "dataset": "race", "model": "bilstm", "metrics": {"dev": {"qwk": 0.5}}},
    {"run_id": "arc-best", "dataset": "arc", "model": "bilstm", "metrics": {"dev": {"qwk": 0.7}}},
    {"run_id": "arc-demo", "dataset": "arc", "model": "distilbert", "metrics": {"dev": {"qwk": 0.6}}},
    {"run_id": "arc-unsaved", "dataset": "arc", "model": "distilbert", "metrics": {"dev": {"qwk": 0.9}}},
]


def saved_store(tmp_path, monkeypatch):
    """Stand in for the result store and the model folders: every run of ROWS is saved, but arc-unsaved."""
    from levellens.explain import local

    for row in ROWS:
        if row["run_id"] != "arc-unsaved":
            folder = tmp_path / row["run_id"]
            folder.mkdir()
            (folder / export.CARD).write_text(json.dumps({"dataset": row["dataset"]}), encoding="utf-8")
    monkeypatch.setattr(export, "load_rows", lambda: ROWS)
    monkeypatch.setattr(export, "model_folder", lambda run_id: tmp_path / run_id)
    monkeypatch.setattr(local, "model_folder", lambda run_id: tmp_path / run_id)


def test_runs_to_check_includes_arc_distilbert_for_demo(tmp_path, monkeypatch):
    """Check the best model of each dataset and also the best saved ARC DistilBERT."""
    saved_store(tmp_path, monkeypatch)

    assert export.runs_to_check() == ["race-best", "arc-best", "arc-demo"]
    assert export.best_saved_runs() == ["race-best", "arc-best"]
    assert export.demo_run() == "arc-demo"


def test_transfer_default_keeps_best_model_of_every_dataset(tmp_path, monkeypatch):
    """The demo DistilBERT must not replace a better ARC model in the transfer matrix (F3, F4)."""
    import pytest

    from levellens.explain import transfer

    saved_store(tmp_path, monkeypatch)
    seen = {}

    class Stop(Exception):
        """Raised by the stand-in for transfer_matrix, so that no model is loaded."""

    def stand_in(runs, *args, **kwargs):
        seen.update(runs)
        raise Stop

    monkeypatch.setattr(transfer, "transfer_matrix", stand_in)
    with pytest.raises(Stop):
        transfer.main([])

    assert seen == {"race": "race-best", "arc": "arc-best"}


def test_check_forces_cpu_and_restores_device_setting(tmp_path, monkeypatch):
    """A GPU setting must not turn the R7 laptop CPU timing into a GPU timing."""
    run_id = "arc-demo"
    folder = tmp_path / "models" / run_id
    folder.mkdir(parents=True)
    (folder / export.CARD).write_text(json.dumps({"dataset": "arc"}), encoding="utf-8")
    monkeypatch.setattr(export, "model_folder", lambda _: folder)
    monkeypatch.setattr(export, "results_dir", lambda: tmp_path / "results")
    monkeypatch.setattr(export, "load_split", lambda dataset, split: [{"id": "q1", "text": "One"}, {"id": "q2", "text": "Two"}])
    monkeypatch.setattr(export, "load_predictions", lambda _: {
        "test": {"ids": ["q1", "q2"], "output": np.array([[0.3, 0.7], [0.6, 0.4]])}
    })
    monkeypatch.setenv("LEVELLENS_DEVICE", "cuda")

    class FakePredictor:
        """Return fixed probabilities while recording the device at load time."""

        model = type("Model", (), {"device": "cpu"})()

        @classmethod
        def load(cls, name):
            assert name == run_id
            assert export.os.environ["LEVELLENS_DEVICE"] == "cpu"
            return cls()

        def predict(self, items):
            return {"proba": np.array([[0.3, 0.7], [0.6, 0.4]])[:len(items)]}

    monkeypatch.setattr(export, "Predictor", FakePredictor)
    report = export.check(run_id)

    assert report["same_predictions"] is True
    assert report["device"] == "cpu"
    assert export.os.environ["LEVELLENS_DEVICE"] == "cuda"
