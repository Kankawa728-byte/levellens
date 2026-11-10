# Interfaces (draft v0)

This page lets everyone start before the skeleton is merged. When the skeleton is in `main` (Wed 11 Nov 12:00), the code is the reference, and the lead updates this page to match it.

## One item

Every dataset adapter returns items with the same fields.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Unique inside the dataset |
| `dataset` | str | `race`, `arc`, `math`, `scienceqa` or `gsm8k` |
| `split` | str | `train`, `dev` or `test` |
| `question` | str | The question text |
| `options` | list of str | The answer options; empty for MATH and GSM8K |
| `passage` | str or None | The reading passage (RACE++ only). Not part of the model input in Tier 1 |
| `text` | str | The model input: the question plus all options, in one fixed format |
| `level` | int | 0 to K − 1, ordered from the lowest level to the highest |
| `level_name` | str | For example `middle school`, `grade 5`, `Level 3` |
| `meta` | dict | Extra fields, for example the MATH problem type, ARC Easy or Challenge, the ScienceQA grade |

K is 3 for RACE++, 7 for ARC and 5 for MATH.

The splits are saved once by the lead. Every model loads the same files. Nobody re-splits.

## One model

```python
class Model:
    def fit(self, train, dev, config):
        """Train on `train`. Use `dev` for early stopping and for choosing settings."""

    def predict_proba(self, items):
        """Return an array of shape [n, K]. Each row sums to 1."""
```

- A model that predicts one number per question (regression, R2DE) returns that number. The metrics module rounds it to a level, clips it to the valid range and builds the one-hot row for the probability metrics.
- The open LLM returns the probabilities of the level labels, renormalised to sum to 1.

## One run

A run starts from a config and writes one line to `results/runs.jsonl`:

```json
{
  "run_id": "arc-deberta-v3-corn-s0",
  "started": "2026-11-12T14:03:00+08:00",
  "owner": "E",
  "tier": 1,
  "dataset": "arc",
  "model": "deberta-v3",
  "head": "corn",
  "seed": 0,
  "config": {},
  "metrics": {"dev": {}, "test": {}, "test_1k": {}},
  "train_seconds": 0,
  "device": "",
  "n_params": 0,
  "git_commit": ""
}
```

- Write rows through `levellens.results` only. Do not edit the file by hand.
- Each run also saves its predictions (item ids, labels, probabilities for dev and test) next to the results, so that figures, bootstrap intervals and the error audit do not need the model again.
- The test split is for final numbers. Choose settings on dev, or by cross-validation on train.

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

`"stub"` needs no trained model. It returns made-up probabilities, so the app can be built from day one. Real models replace it on Fri 13 Nov (task E8).

## Paths

No hard-coded paths. Two environment variables:

| Variable | Holds | Default |
|---|---|---|
| `LEVELLENS_DATA` | Downloaded datasets and saved splits | `./data` |
| `LEVELLENS_CACHE` | Model weights, tokenised data, feature caches, checkpoints | `./cache` |

Both folders are ignored by git.
