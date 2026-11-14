"""The efficiency table of share C: hours per run, the model table, and the verdicts on R6 and R7.

`scripts/efficiency.py` is a script, not part of the package, so it is loaded by its path.
It uses the standard library only, so these tests need nothing installed beyond pytest.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    """Import one of the scripts in `scripts/` by its path."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


efficiency = load_script("efficiency")


def row(run_id, model, device="cuda (Test GPU)", train_seconds=60.0, predict=None, n_params=1000, **extra):
    """Return a result row with the fields the efficiency table reads."""
    return {
        "run_id": run_id, "dataset": extra.get("dataset", "arc"), "model": model,
        "head": extra.get("head", "ce"), "seed": extra.get("seed", 0), "device": device,
        "train_seconds": train_seconds, "predict_seconds": predict or {"dev": 1.0, "test": 2.0},
        "n_params": n_params,
    }


def write_store(folder, rows, export=(), latency=()):
    """Write a results store: `runs.jsonl` and the latency reports. Return the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "runs.jsonl", "w", encoding="utf-8") as handle:
        for one in rows:
            handle.write(json.dumps(one) + "\n")
    for name, report in (("export", export), ("latency", latency)):
        if report:
            (folder / name).mkdir(exist_ok=True)
            for number, one in enumerate(report):
                (folder / name / f"{one['run_id']}__{number}.json").write_text(json.dumps(one), encoding="utf-8")
    return folder


def test_compute_hours_add_training_and_scoring(tmp_path):
    store = write_store(tmp_path / "results", [row("a", "distilbert", train_seconds=3597.0)])
    table = efficiency.run_table(efficiency.load_rows(store))
    assert table[0]["compute_hours"] == pytest.approx(3600 / 3600, abs=1e-3)  # 3597 s + 1 s + 2 s
    assert table[0]["family"] == "transformer"
    assert table[0]["device_kind"] == "GPU"


@pytest.mark.parametrize("device, kind", [
    ("cuda", "GPU"), ("cuda:1 (NVIDIA Test)", "GPU"), ("cpu", "CPU"), ("mps", "Apple GPU"), ("", "CPU"),
])
def test_device_kind(device, kind):
    assert efficiency.device_kind(device) == kind


def test_newest_row_of_a_run_id_counts(tmp_path):
    store = write_store(tmp_path / "results", [
        row("same", "distilbert", train_seconds=100.0),
        row("same", "distilbert", train_seconds=200.0),
    ])
    table = efficiency.run_table(efficiency.load_rows(store))
    assert len(table) == 1
    assert table[0]["train_seconds"] == 200.0


def test_r6_fails_when_one_fine_tune_is_over_an_hour(tmp_path):
    store = write_store(tmp_path / "results", [
        row("short", "distilbert", train_seconds=600.0),
        row("long", "deberta-v3", train_seconds=4000.0),
    ])
    runs = efficiency.run_table(efficiency.load_rows(store))
    checks = efficiency.verdicts(runs, {})
    assert checks["R6"]["pass"] is False
    assert checks["R6"]["over_one_hour"] == ["long"]
    assert checks["R6"]["longest_fine_tune"]["run_id"] == "long"


def test_r6_fails_when_the_project_is_over_twenty_gpu_hours(tmp_path):
    rows = [row(f"run{n}", "distilbert", train_seconds=3000.0) for n in range(25)]
    store = write_store(tmp_path / "results", rows)
    checks = efficiency.verdicts(efficiency.run_table(efficiency.load_rows(store)), {})
    assert checks["R6"]["gpu_hours"] > efficiency.R6_PROJECT_HOURS
    assert checks["R6"]["pass"] is False


def test_cpu_hours_do_not_count_towards_the_project_limit(tmp_path):
    store = write_store(tmp_path / "results", [
        row("gpu", "distilbert", train_seconds=1800.0),
        row("cpu", "tfidf_logreg", device="cpu", train_seconds=90000.0),
    ])
    checks = efficiency.verdicts(efficiency.run_table(efficiency.load_rows(store)), {})
    assert checks["R6"]["gpu_hours"] == pytest.approx(0.5008, abs=1e-3)
    assert checks["R6"]["cpu_hours"] > 20
    assert checks["R6"]["pass"] is True


def test_floors_and_the_open_llm_are_not_fine_tunes(tmp_path):
    store = write_store(tmp_path / "results", [
        row("floor", "majority", device="cpu", train_seconds=0.0),
        row("llm", "qwen_zero_shot", train_seconds=5000.0),
    ])
    runs = efficiency.run_table(efficiency.load_rows(store))
    assert [one["r6_run_pass"] for one in runs] == ["", ""]
    checks = efficiency.verdicts(runs, {})
    assert checks["R6"]["fine_tunes"] == 0
    assert checks["R6"]["measured"] is False
    assert checks["R6"]["gpu_hours"] > 1  # the open LLM still spends GPU-hours


def test_r7_uses_the_cpu_measurement_of_distilbert(tmp_path):
    store = write_store(
        tmp_path / "results",
        [row("arc-distilbert-ce-s0", "distilbert"), row("arc-deberta-v3-ce-s0", "deberta-v3")],
        export=[{"run_id": "arc-distilbert-ce-s0", "device": "cpu", "seconds_per_question": 0.4}],
        latency=[{"run_id": "arc-deberta-v3-ce-s0", "device": "cuda", "seconds_per_question": 0.02}],
    )
    runs = efficiency.run_table(efficiency.load_rows(store))
    measured = efficiency.latencies(store)
    checks = efficiency.verdicts(runs, measured)
    assert checks["R7"]["pass"] is True
    assert checks["R7"]["best"] == 0.4

    models = efficiency.model_table(runs, measured)
    by_name = {one["model"]: one for one in models}
    assert by_name["distilbert"]["cpu_seconds_per_question"] == 0.4
    assert by_name["distilbert"]["gpu_seconds_per_question"] == ""
    assert by_name["deberta-v3"]["gpu_seconds_per_question"] == 0.02


def test_r7_fails_over_one_second(tmp_path):
    store = write_store(
        tmp_path / "results", [row("arc-distilbert-ce-s0", "distilbert")],
        export=[{"run_id": "arc-distilbert-ce-s0", "device": "cpu", "seconds_per_question": 1.7}],
    )
    checks = efficiency.verdicts(efficiency.run_table(efficiency.load_rows(store)), efficiency.latencies(store))
    assert checks["R7"]["pass"] is False


def test_r7_is_not_measured_without_a_report(tmp_path):
    store = write_store(tmp_path / "results", [row("arc-distilbert-ce-s0", "distilbert")])
    checks = efficiency.verdicts(efficiency.run_table(efficiency.load_rows(store)), {})
    assert checks["R7"]["measured"] is False
    assert checks["R7"]["pass"] is False


def test_model_table_groups_the_seeds_of_one_model(tmp_path):
    store = write_store(tmp_path / "results", [
        row("s0", "deberta-v3", seed=0, train_seconds=100.0, n_params=184_000_000),
        row("s1", "deberta-v3", seed=1, train_seconds=300.0, n_params=184_000_000),
    ])
    models = efficiency.model_table(efficiency.run_table(efficiency.load_rows(store)), {})
    assert len(models) == 1
    assert models[0]["runs"] == 2
    assert models[0]["mean_train_seconds"] == 200.0
    assert models[0]["longest_train_seconds"] == 300.0
    assert models[0]["n_params"] == 184_000_000


def test_build_writes_three_files(tmp_path):
    store = write_store(tmp_path / "results", [row("a", "distilbert")])
    page, checks = efficiency.build(store, write=True)
    tables = store / "tables"
    assert (tables / "efficiency.csv").exists()
    assert (tables / "efficiency_runs.csv").exists()
    assert (tables / "efficiency.md").read_text(encoding="utf-8") == page + "\n"
    assert "R6" in page and "R7" in page
    assert checks["R6"]["pass"] is True


def test_missing_runs_file_says_so(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        efficiency.load_rows(tmp_path / "nothing")


def test_broken_json_line_names_the_line(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "runs.jsonl").write_text('{"run_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="line 2"):
        efficiency.load_rows(folder)


def test_results_folder_follows_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("LEVELLENS_RESULTS", str(tmp_path / "elsewhere"))
    assert efficiency.results_folder() == tmp_path / "elsewhere"
    assert efficiency.results_folder(tmp_path / "given") == tmp_path / "given"
