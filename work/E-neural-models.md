# Share E · Neural models: BiLSTM and transformers

中文版：[E-neural-models.zh-CN.md](E-neural-models.zh-CN.md)

**Owner:** open · **Buddy:** C · **Size:** 24 h Tier 1 + 4 h Tier 2

The trained neural models and the improvements on top of them: ordinal output heads, Bayesian tuning with Optuna, and stable training on the small science set.

This share is on the critical path. The interpretation work and the demo wait for its checkpoints. It suits someone who has fine-tuned a transformer before.

## Files you own

- `levellens/heads/`
- `levellens/models/recurrent.py`, `levellens/models/transformer.py`
- `levellens/features/glove.py`, `levellens/features/latex.py`
- Your configs in `configs/model/` and `configs/head/`, and the tests for your modules

## Tier 1

Re-plan of Thu 12 Nov: the dates are new, and a task marked "Cut" is not done. The hours are the first estimates. The whole re-plan is in [docs/schedule.md](../docs/schedule.md).

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| E1 | Output heads: cross-entropy and CORN (conditional ordinal regression). Each head returns a loss and the probability of every level. | A test shows that the CORN probabilities are between 0 and 1 and sum to 1. | 1.5 | Thu 12 Nov 23:59 |
| E2 | Two-layer BiLSTM with GloVe 300d and attention pooling, with both heads, one configuration | It trains on MATH on Fri 13 Nov at 12:00. Rows for all three datasets at 20:00. | 5.5 | MATH: Fri 13 Nov 12:00. All: Fri 13 Nov 20:00 |
| E3 | Cut: LaTeX-aware tokeniser for MATH | Commands such as `\frac`, the symbols `^` and `_`, and numbers stay whole tokens. A test covers five problems. | 1.5 | Cut on Thu 12 Nov |
| E4 | Cut: BiLSTM grid: 4 configurations on all three datasets | Rows in `results/runs.jsonl`, and the best configuration is named. | 1 | Cut on Thu 12 Nov |
| E5 | Transformer trainer wrapper: any backbone with any head, class-weighted loss, early stopping on dev QWK, cached tokenisation, dynamic padding, each run 1 GPU-hour or less | A DistilBERT run with cross-entropy on a small subset writes a row on Thu 12 Nov. The wrapper is complete on Fri 13 Nov. | 5.5 | Smoke run: Thu 12 Nov 23:59. Complete: Fri 13 Nov 12:00 |
| E6 | Tier 1 grid configs for C: `microsoft/deberta-v3-base` and `distilbert/distilbert-base-uncased`, two heads, three datasets, seed 0. On ARC also seeds 1 and 2, with all 3,358 training items. | C has the 20 configs and the command to run them. | 1.5 | Fri 13 Nov 12:00 |
| E7 | Optuna on ARC with DistilBERT: TPE sampler, median pruner, 12 trials. The best setting is re-trained on the full training split. | The search is launched on Fri 13 Nov. The re-trained row is in at the results freeze. | 3 | Launch: Fri 13 Nov 15:00. Row: Fri 13 Nov 23:59 |
| E8 | Export of the final models to the Predictor API: the best model for each dataset, and DistilBERT for the demo | `Predictor.load(...)` returns the same test predictions as the training run. DistilBERT answers in 1 second or less per question on a laptop CPU. | 1.5 | Sat 14 Nov 10:00 |

Known trap: DeBERTa-v3 can give NaN losses in fp16. Use bf16 or fp32, or tell the lead if RoBERTa should become the main backbone.

Suggested libraries: PyTorch, transformers, datasets, optuna, coral-pytorch for the CORN loss. You decide.

## Tier 2

Cancelled on Thu 12 Nov. None of these tasks is done.

| Id | Task | Done when | Hours |
|---|---|---|---|
| E9 | Two more heads: regression with rounding, and an ordered-logit head with learned cut-points. Both added to the grid. | Rows for both heads on both backbones and all three datasets. | 2 |
| E10 | Configs for the BERT replication (`google-bert/bert-base-uncased`, same splits as the papers, balanced ARC training subset) and for `FacebookAI/roberta-base` with the best head | C has the configs. The replication rows sit next to the published numbers in the table. | 1 |
| E11 | BiLSTM Optuna sweep instead of the 4-configuration grid | Rows for the best trial on all three datasets. | 1 |

## Others wait for you

- C: the Tier 1 grid configs on Fri 13 Nov at 12:00.
- F: the DistilBERT smoke checkpoint on Fri 13 Nov at 12:00, and the final models behind the Predictor API on Sat 14 Nov at 10:00.
- A: your rows for the tables, at the results freeze.

## You wait for

- A: the skeleton and the adapters. They are in `main` since Thu 12 Nov. Start with "First run" in [docs/interfaces.md](../docs/interfaces.md).
- C: the smoke job on Fri 13 Nov at 12:00, then the result rows and checkpoints.

## Video part

3:40–5:30: the BiLSTM and the CORN head with a training cell running, the transformers, the Optuna search, stable training on ARC, and our numbers next to the published ones.
