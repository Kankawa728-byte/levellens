# LevelLens

LevelLens estimates the target difficulty level of an exam question from its text. A teacher pastes a new question and sees which level it is aimed at, and why.

Group project of Table 38, six members. Hard deadline: **Sun 15 Nov 2026, 23:59**. All times on these pages are Singapore time.

中文版：[README.zh-CN.md](README.zh-CN.md). Every page has a Chinese version; the link is at the top of the page.

## Start here

1. Read this page to the end. It takes five minutes.
2. Open the page of your share from the table below.
3. No share yet? Shares E and F are still open. The lead assigns them on Thu 12 Nov.
4. Send your GitHub username to the lead in the group chat. You need it to push code.

## The six shares

The work is cut into six shares of the same size. Each share has 24 hours of Tier 1 work and 4 hours of Tier 2 work. The hours are estimates. They include your own slides and your own recording. The count, task by task, is in [work/README.md](work/README.md).

| Share | Role | Owner | Tier 1 | Tier 2 | Status |
|---|---|---|---|---|---|
| [A](work/A-framework-integration.md) | Framework and integration | Xingyang (lead) | 24 h | 4 h | Assigned |
| [B](work/B-literature-data-audit.md) | Literature, data checks and audit | Wenyue | 24 h | 4 h | Assigned |
| [C](work/C-compute-runs-efficiency.md) | Compute, runs and efficiency | Qiutian | 24 h | 4 h | Assigned |
| [D](work/D-baselines.md) | Baselines: classical, feature models, open LLM | Hailing | 24 h | 4 h | Assigned |
| [E](work/E-neural-models.md) | Neural models: BiLSTM and transformers | **open** | 24 h | 4 h | [Claim in issue 5](https://github.com/Kankawa728-byte/levellens/issues/5) |
| [F](work/F-interpretation-demo.md) | Interpretation, extrapolation and demo app | Zhifei | 24 h | 4 h | Assigned |

- **Tier 1** is what we must ship.
- **Tier 2** was what we would add if Tier 1 was on time. The lead cancelled it on Thu 12 Nov and also cut part of Tier 1. The re-plan is in [docs/schedule.md](docs/schedule.md). The hours in the table are the first estimates and were not counted again.

Which open share suits you:

- **E** is on the critical path. It suits someone who has fine-tuned a transformer before.
- **F** suits someone who likes plots and front ends. The app uses Streamlit.

## How to claim a share

1. Read the pages of shares E and F.
2. Open the issue of the share you want and write the comment "I take this", with your name. The first comment gets the share. If you have no GitHub account, tell the lead in the group chat instead.
3. The lead writes your name in the table above.

The claim deadline was Tue 10 Nov 23:59. Shares E and F were still open on Thu 12 Nov, so the lead assigns them that day.

## Key dates

| When | What |
|---|---|
| Thu 12 Nov 16:00 | Code skeleton in `main`; specifications frozen; Tier 2 cancelled and Tier 1 cut |
| Thu 12 Nov 23:59 | Everyone has done the first run; first result rows from D, E and F |
| Fri 13 Nov 12:00 | Transformer trainer complete; grid handed to the GPU |
| Fri 13 Nov 23:59 | **Results freeze**: no new experiments |
| Sat 14 Nov 12:00 | All tables and figures final |
| Sat 14 Nov 16:00 | Final slides; dry run at 16:30 |
| Sat 14 Nov 23:59 | Everyone's recording is with the lead |
| Sun 15 Nov 14:00 | Sign-off: video, reflection, code |
| Sun 15 Nov 15:00 | Upload, latest 18:00 |
| Sun 15 Nov 23:59 | Hard deadline |

These are the dates of the re-plan of Thu 12 Nov. The full schedule, day by day and share by share, is in [docs/schedule.md](docs/schedule.md).

## What we build

- **Task.** Predict the target level of a question: exam stage, school grade or competition level. The levels are ordered.
- **Data.** Three trained datasets: RACE++ (English reading, 3 levels), ARC (school science, 7 grades), MATH (competition maths, 5 levels). ScienceQA tests how far the science model extrapolates.
- **Models.** Floors, classical and feature models, a BiLSTM, DeBERTa-v3 and DistilBERT with ordinal output heads, and an open LLM without training.
- **Evaluation.** Nine metrics, published baselines on the same test splits, and eight measurable specifications with pass or fail.
- **Interpretation.** SHAP and LIME, an audit of the worst errors, and transfer between subjects.
- **Demo.** A Streamlit app in which a teacher scores a question, sees why it is hard, edits it and scores it again.

The details are in [docs/design.md](docs/design.md).

## Repository layout

The letter is the share that owns the file. Each file has one owner.

```text
levellens/                    the Python package
  data/                 A     race.py  arc.py  math_.py  scienceqa.py  toy.py   -> one schema
                              prepare.py writes the split files that every model loads
  features/             D     tfidf.py  linguistic.py
                        E     glove.py  latex.py
  models/               A     base.py (the model contract)  floors.py (majority, the worked example)
                        D     classic.py  llm.py
                        E     recurrent.py  transformer.py
  heads/                E     ce.py  corn.py
  explain/              D     global_shap.py
                        F     local.py  errors.py
  evaluate.py           A     all metrics, bootstrap intervals
  figures.py            F     result and error figures
  predictor.py          A     one Predictor API for the experiments and the app
  results.py            A     runs.jsonl -> tables
  run.py                A     python -m levellens.run data=arc model=deberta-v3 head=corn seed=0
  config.py  paths.py   A     the config of a run; the data, cache and results folders
configs/                A     layout; every owner adds the configs of their own models
scripts/                C     download_data.py, stage_cache, job templates
notebooks/main.ipynb    A     small end-to-end run, then rebuilds every table
notebooks/data_checks.* B     data checks behind docs/datasets.md
app/                    F     streamlit_app.py, pages/
tests/                  A     shared tests; every owner adds tests for their own modules
results/                      runs.jsonl, tables, figures
docs/                   A, B  design, interfaces, schedule, workflow, video; B adds datasets, related work, audit
work/                   A     the six shares
```

The code skeleton is in `main`. Start with "First run" in [docs/interfaces.md](docs/interfaces.md).

## Rules in short

- One branch for each piece of work. The lead merges into `main`.
- Every run writes one line to `results/runs.jsonl`. Every number on a slide comes from there.
- The test split is for final numbers only.
- This repository is public. Do not commit data, model weights, recordings, matriculation numbers, emails, signatures or access tokens.
- Blocked for more than an hour? Say so in the group chat.

The full rules are in [docs/workflow.md](docs/workflow.md).

## Pages

| Page | What it holds |
|---|---|
| [work/README.md](work/README.md) | The six shares, hour by hour; who waits for whom |
| [docs/design.md](docs/design.md) | Datasets, models, metrics, specifications, scope, references |
| [docs/interfaces.md](docs/interfaces.md) | Item fields, model contract, result rows, Predictor |
| [docs/schedule.md](docs/schedule.md) | Milestones and the day-by-day plan |
| [docs/workflow.md](docs/workflow.md) | Branches, pull requests, files, runs |
| [docs/video.md](docs/video.md) | Video parts, slides, recording, reflection document |
| [docs/datasets.md](docs/datasets.md) | Dataset cards: counts per level, lengths, exact repeats, licences |
| [docs/related_work.md](docs/related_work.md) | Published baselines, checked against the papers and their code |
| [docs/audit.md](docs/audit.md) | Human audit of the 15 worst errors: rules and worksheet |
