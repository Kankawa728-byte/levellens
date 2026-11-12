# The six shares

中文版：[README.zh-CN.md](README.zh-CN.md)

The project is cut into six shares of the same size. This page shows the count, so that everyone can check that it is fair.

- **Tier 1** is what we must ship. Every share has 24 hours of it.
- **Tier 2** was what we would add if Tier 1 was on time. Every share had 4 hours of it. The lead cancelled Tier 2 on Thu 12 Nov.
- **Re-plan of Thu 12 Nov.** We started late, so the lead also cut some Tier 1 tasks and made others smaller. The share pages mark every cut task, and [docs/schedule.md](../docs/schedule.md) lists them all. The hours on this page are the first estimates and were not counted again.
- The hours are estimates. If your share is running more than 3 hours over its estimate, tell the lead. The lead moves a task to a share that is under.

All times are Singapore time.

## Owners

| Share | Role | Owner | Tier 1 | Tier 2 | Video part | Buddy |
|---|---|---|---|---|---|---|
| [A](A-framework-integration.md) | Framework and integration | Xingyang (lead) | 24 h | 4 h | Opening and closing | F |
| [B](B-literature-data-audit.md) | Literature, data checks and audit | Wenyue | 24 h | 4 h | Datasets, published baselines, audit | D |
| [C](C-compute-runs-efficiency.md) | Compute, runs and efficiency | Qiutian | 24 h | 4 h | Runs and efficiency | E |
| [D](D-baselines.md) | Baselines: classical, feature models, open LLM | Hailing | 24 h | 4 h | Baselines | B |
| [E](E-neural-models.md) | Neural models: BiLSTM and transformers | open | 24 h | 4 h | Neural models and tuning | C |
| [F](F-interpretation-demo.md) | Interpretation, extrapolation and demo app | open | 24 h | 4 h | Interpretation and live demo | A |

Your buddy reviews your pull requests and covers for you if you are away for a day.

## Common tasks: 3 hours for everyone

| Task | Hours | Due |
|---|---|---|
| Your own slides (draft Sat 14 Nov 10:00, final Sat 14 Nov 16:00) | 1 | Sat 14 Nov 16:00 |
| Your own recording, voice and screen | 1 | Sat 14 Nov 23:59 |
| Timed dry run, full-video review, your buddy's pull requests | 0.5 | Sun 15 Nov 12:00 |
| Your lines for the reflection document, your signature, the peer evaluation | 0.5 | Sun 15 Nov 10:00 |

## Tier 1, task by task

Each share has 21 hours of its own tasks and the 3 common hours.

### A · Framework and integration

| Id | Task | Hours |
|---|---|---|
| A1 | Package skeleton, configs, command line, Predictor API with a stub model | 4 |
| A2 | Unified schema, dataset adapters, saved splits, fixed 1,000-item test subsets, ARC grade-label check | 5 |
| A3 | Metrics module, bootstrap intervals, majority-row test | 3.5 |
| A4 | Results store and table builder | 2 |
| A5 | Specification freeze, reviews and merges, daily integration, go or no-go calls | 2 |
| A6 | README, narrated driver notebook, docstring pass | 2 |
| A7 | Video assembly from the six recordings | 2 |
| A8 | Final package and upload | 0.5 |
| | Common tasks | 3 |
| | **Total** | **24** |

### B · Literature, data checks and audit

| Id | Task | Hours |
|---|---|---|
| B1 | Data checks and dataset cards for four datasets | 5.5 |
| B2 | Published baselines and related work, reference slide | 4 |
| B3 | Human audit of the 15 worst errors | 3.5 |
| B4 | Slide template, storyboard, recording guide and schedule | 3.5 |
| B5 | Proofreading of all slides, limitations slide | 2 |
| B6 | Reflection document | 1.5 |
| B7 | Submission checklist, peer-evaluation reminders | 1 |
| | Common tasks | 3 |
| | **Total** | **24** |

### C · Compute, runs and efficiency

| Id | Task | Hours |
|---|---|---|
| C1 | GPU environment, hardware and software specifications | 2 |
| C2 | Offline cache of datasets and model weights, download scripts | 3 |
| C3 | Job templates, smoke job | 3 |
| C4 | Running the Tier 1 grid: submit, monitor, retry | 5 |
| C5 | Returning results: rows, logs, prediction files, final checkpoints | 2 |
| C6 | Efficiency table: time, GPU-hours, parameters, latency | 4 |
| C7 | Clean-environment run, reproducibility notes | 2 |
| | Common tasks | 3 |
| | **Total** | **24** |

### D · Baselines: classical, feature models, open LLM

| Id | Task | Hours |
|---|---|---|
| D1 | Floors: majority, stratified random, subject prior | 1 |
| D2 | TF-IDF models with 5-fold search | 5 |
| D3 | R2DE | 2 |
| D4 | Surface feature extractor and LightGBM | 6.5 |
| D5 | Global SHAP per subject | 2 |
| D6 | Open LLM, zero-shot | 4.5 |
| | Common tasks | 3 |
| | **Total** | **24** |

### E · Neural models: BiLSTM and transformers

| Id | Task | Hours |
|---|---|---|
| E1 | Output heads: cross-entropy and CORN | 1.5 |
| E2 | BiLSTM with GloVe and attention pooling | 5.5 |
| E3 | Cut: LaTeX-aware tokeniser for MATH | 1.5 |
| E4 | Cut: BiLSTM 4-configuration grid | 1 |
| E5 | Transformer trainer wrapper | 5.5 |
| E6 | Tier 1 grid configs | 1.5 |
| E7 | Optuna search on ARC with DistilBERT | 3 |
| E8 | Export of the final models to the Predictor API | 1.5 |
| | Common tasks | 3 |
| | **Total** | **24** |

### F · Interpretation, extrapolation and demo app

| Id | Task | Hours |
|---|---|---|
| F1 | LIME on the final DistilBERT | 4.5 |
| F2 | Error and results figures, worst-error list | 3 |
| F3 | Transfer matrix | 1.5 |
| F4 | Extrapolation to ScienceQA | 2 |
| F5 | Cut: physical-meaning plot | 1.5 |
| F6 | Streamlit Calibrate page | 6.5 |
| F7 | Streamlit Results page | 2 |
| | Common tasks | 3 |
| | **Total** | **24** |

## Tier 2, task by task

Cancelled on Thu 12 Nov. None of these tasks is done.

| Share | Id | Task | Hours |
|---|---|---|---|
| A | A9 | GSM8K adapter | 1 |
| A | A10 | Automated tests on every push | 1.5 |
| A | A11 | Guide "add your own dataset or model" with a worked example | 1.5 |
| B | B8 | GSM8K dataset card | 0.5 |
| B | B9 | Audit extended to 60 errors | 2 |
| B | B10 | Background notes and one slide: how difficulty is set in practice | 1.5 |
| C | C8 | Running the Tier 2 grid | 3 |
| C | C9 | Latency for every model family | 1 |
| D | D7 | Open LLM, few-shot | 1.5 |
| D | D8 | Hybrid Random Forest | 1 |
| D | D9 | Phrase-level TF-IDF, GloVe average with Logistic Regression | 1.5 |
| E | E9 | Regression and ordered-logit heads, added to the grid | 2 |
| E | E10 | BERT replication and RoBERTa configs | 1 |
| E | E11 | BiLSTM Optuna sweep | 1 |
| F | F8 | GSM8K extrapolation | 1.5 |
| F | F9 | Streamlit "Audit a quiz" page | 2.5 |

Every share: 4 hours.

## Who waits for whom

| From | To | What | When |
|---|---|---|---|
| A | everyone | Skeleton in `main` with all four adapters, saved splits and 1,000-item subsets. Start with [docs/interfaces.md](../docs/interfaces.md) | Done on Thu 12 Nov |
| E | C | Tier 1 grid configs | Fri 13 Nov 12:00 |
| D | C | Open LLM scoring script | Fri 13 Nov 12:00 |
| C | A, E, F | Result rows and checkpoints as runs finish; everything by | Sat 14 Nov 10:00 |
| E | F | Final models behind the Predictor API | Sat 14 Nov 10:00 |
| F | B | List of the 15 worst errors | Sat 14 Nov 10:00 |
| everyone | B | Final slides for proofreading | Sat 14 Nov 16:00 |
| everyone | A | Own recording | Sat 14 Nov 23:59 |
| everyone | B | Lines for the reflection document | Sun 15 Nov 10:00 |

## What each grading row needs and who delivers it

| Grading row | Evidence | Shares |
|---|---|---|
| Requirements | Specifications R1–R8 with pass or fail | A, with numbers from all |
| Application and datasets | Three trained datasets, an extrapolation set, dataset cards | A, B |
| Framework | Model choice and the improvements: ordinal heads, Optuna, ARC stability | E |
| Experiment design | Nine metrics, published baselines, our own baselines | A, B, D |
| Interpretation | Global SHAP, LIME, transfer, extrapolation, error audit | D, F, B |
| Readability | One package, docstrings, driver notebook, README, reviewed pull requests | everyone, A |
| Efficiency | Cached tokenisation, pruning, capped sets, efficiency table | E, C |
| Reusability | Shared schema, Predictor API, configs | A |
| Presentation | Streamlit app | F |
| Video | All six present; storyboard; assembly | everyone, B, A |
