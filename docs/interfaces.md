# Interfaces (v1)

中文版：[interfaces.zh-CN.md](interfaces.zh-CN.md)

This page describes the code skeleton. The code is the reference, and the lead keeps this page in step with it.

## First run

```bash
pip install -r requirements.txt
python -m pytest
python -m levellens.run data=toy model=majority
python -m levellens.data.prepare
python -m levellens.run data=math model=majority
python -m levellens.results
```

- `data=toy` is a small made-up dataset that needs no download. Use it to try your model code. Its rows go to `results/scratch/`, which git ignores, and never into the shared `results/runs.jsonl`.
- `python -m levellens.data.prepare` downloads the four datasets and writes the split files. It needs about 1.2 GB of disk. Name datasets to prepare only some of them: `python -m levellens.data.prepare math arc`.
- `python -m levellens.results` rebuilds every table in `results/tables/` from `results/runs.jsonl`.

## One item

Every dataset adapter returns items with the same fields.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Unique inside the dataset |
| `dataset` | str | `race`, `arc`, `math`, `scienceqa` or `toy` |
| `split` | str | `train`, `dev` or `test` |
| `question` | str | The question text |
| `options` | list of str | The answer options; empty for MATH |
| `passage` | str or None | The reading passage (RACE++ only). Not part of the model input in Tier 1 |
| `text` | str | The model input: the question, then every option with its letter, for example `Which is a mammal? A. frog B. whale` |
| `level` | int | 0 to K − 1, ordered from the lowest level to the highest |
| `level_name` | str | For example `middle school`, `grade 5`, `Level 3` |
| `meta` | dict | Extra fields: the MATH problem type and solution, the ARC grade, set (Easy or Challenge) and exam, the ScienceQA grade and subject, the index of the correct option |

K is 3 for RACE++, 7 for ARC, 5 for MATH, 12 for ScienceQA and 3 for the toy dataset.

```python
from levellens.data import info, load_split

train = load_split("math", "train")   # a list of items
info("math").level_names              # ('Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5')
```

## The splits

| Dataset | `train` | `dev` | `test` | `test_1k` | Other |
|---|---|---|---|---|---|
| `race` | 100,568 | 5,599 | 5,642 | 1,000 | |
| `arc` | 3,358 | 862 | 3,530 | 1,000 | `train_balanced`: 2,293 |
| `math` | 6,748 | 750 | 5,000 | 1,000 | |
| `scienceqa` | 6,508 | 2,144 | 2,224 | 1,000 | |
| `toy` | 240 | 60 | 90 | 90 | |

- `test_1k` is the fixed stratified subset of `test` on which every system, the open LLM included, is scored.
- `train_balanced` is the ARC training subset of the published papers: grades 5 and 8 cut to 500 questions each. Use it with `data=arc_balanced`, for replication runs only.
- MATH has no published dev split. 10% of the training problems are held out, with the same mix of levels and problem types. 2 training problems labelled "Level ?" are dropped.
- The subsets are picked by a hash of the item id, not by a random number generator. Every machine gets the same items.
- `python -m levellens.data.prepare` compares each split with the fingerprints in `levellens/data/manifest.json` and fails if they differ. Nobody re-splits, and only the lead updates the fingerprints.
- `scripts/download_data.py` (share C) can call this command and then fetch the model weights.

## One model

```python
from levellens.models.base import Model, texts

class MyModel(Model):
    def fit(self, train, dev, config):
        """Train on `train`. Use `dev` for early stopping and for choosing settings."""

    def predict_proba(self, items):
        """Return an array of shape [n, K]. Each row sums to 1."""
```

- The constructor receives `num_levels` and the settings under `params:` in the model config. They are stored as `self.num_levels` and `self.params`.
- `config` is the whole run config: `config.seed`, `config.device`, `config.head.name`, `config.model.params`.
- `texts(items)` returns the model input of each item.
- A model that predicts one number per question (regression, R2DE) returns an array of shape [n]. The metrics module rounds it to a level, clips it to the valid range and builds the one-hot row for the probability metrics.
- The open LLM returns the probabilities of the level labels, renormalised to sum to 1.
- Optional: `n_params` for the efficiency table, and `save(folder)` with `load(folder)` so that the Predictor can load the trained model.

Each model has a config file in `configs/model/`. `target` is the class, and `params` are its settings:

```yaml
# configs/model/majority.yaml
name: majority
target: levellens.models.floors.Majority
params: {}
```

`levellens/models/floors.py` is the worked example. Output heads get a file in `configs/head/` in the same way; a model reads its head from `config.head`.

## One run

A run starts from a config and writes one line to `results/runs.jsonl`:

```bash
python -m levellens.run data=arc model=majority seed=1 owner=D
python -m levellens.run data=race model=majority limit_train=20000 variant=demo
```

- `data=`, `model=` and `head=` each pick one file from `configs/data/`, `configs/model/` and `configs/head/`.
- Every other `key=value` overrides one setting. Dots go into nested settings: `model.params.C=10`.
- The shared settings and their defaults are in `configs/train.yaml`: `seed`, `owner`, `tier`, `variant`, `run_id`, `device`, `limit_train`, `eval_splits`, `save_predictions`, `save_model`.

```json
{
  "run_id": "arc-deberta-v3-corn-s0",
  "started": "2026-11-12T14:03:00+08:00",
  "owner": "E",
  "tier": 1,
  "dataset": "arc",
  "model": "deberta-v3",
  "head": "corn",
  "variant": "",
  "seed": 0,
  "config": {},
  "metrics": {"dev": {}, "test": {}, "test_1k": {}},
  "train_seconds": 0,
  "predict_seconds": {"dev": 0, "test": 0},
  "device": "",
  "n_params": 0,
  "git_commit": ""
}
```

- The default `run_id` is `<dataset>-<model>-<head>[-<variant>]-s<seed>`. Give runs that differ only in their settings a different `variant`. If a `run_id` is run again, the newest row counts.
- Each split under `metrics` holds `n`, `accuracy`, `macro_f1`, `adjacent_accuracy`, `qwk`, `mae`, `rmse`, `spearman`, `spearman_score`, `drps`, `balanced_drps`, `ece` and `per_level` (precision, recall and support of each level). A metric that is undefined is `null`.
- `spearman` uses the predicted level. `spearman_score` uses the continuous score: the expected level, or the regression output.
- Write rows through `levellens.results` only. Do not edit the file by hand.
- Each run also saves its predictions (item ids, labels, model outputs for dev and test) in `results/preds/<run_id>.npz`, so that figures, bootstrap intervals and the error audit do not need the model again.
- The test split is for final numbers. Choose settings on dev, or by cross-validation on train.

```python
from levellens.evaluate import evaluate, paired_bootstrap
from levellens.results import load_predictions, load_runs

runs = load_runs()                                   # one row per run; columns such as metrics.test.qwk
test = load_predictions("arc-majority-none-s0")["test"]   # {"ids", "y", "output"}
evaluate(test["y"], test["output"], num_levels=7)
```

## The Predictor

The experiments and the app use the same class.

```python
from levellens.predictor import Predictor

predictor = Predictor.load("stub")
result = predictor.predict(["What is the main idea of the passage? A. ... B. ... C. ... D. ..."])
# result["proba"]           shape [n, K]
# result["level"]           most likely level for each question
# result["expected_level"]  probability-weighted level for each question
# result["level_names"]     names of the K levels
```

`"stub"` needs no trained model. It returns made-up probabilities, so the app can be built from day one: longer questions get a higher level. `Predictor.load("stub", dataset="arc")` gives the 7 ARC levels. Real models replace it on Fri 13 Nov (task E8): a run with `save_model=true` saves the model, and `Predictor.load("<run_id>")` loads it.

## Paths and devices

No hard-coded paths. Four environment variables:

| Variable | Holds | Default |
|---|---|---|
| `LEVELLENS_DATA` | Downloaded datasets and saved splits | `data/` in the repository |
| `LEVELLENS_CACHE` | Model weights, tokenised data, feature caches, checkpoints, saved models | `cache/` in the repository |
| `LEVELLENS_RESULTS` | `runs.jsonl`, predictions, tables, figures | `results/` in the repository |
| `LEVELLENS_DEVICE` | `cpu`, `cuda` or `cuda:0`. It wins over `device=` in the config | not set: the GPU if PyTorch sees one, else the CPU |

`data/` and `cache/` are ignored by git. Use `levellens.paths` to get the folders: `data_dir()`, `cache_dir()`, `results_dir()`.
