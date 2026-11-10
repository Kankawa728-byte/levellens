# LevelLens

LevelLens estimates the target difficulty level of an exam question from its text. A teacher pastes a new question and sees which level it is aimed at, and why.

Group project of Table 38, six members. Hard deadline: **Sun 15 Nov 2026, 23:59**. All times on these pages are Singapore time.

中文版：[README.zh-CN.md](README.zh-CN.md)

## Start here

1. Read this page to the end. It takes five minutes.
2. Open the page of your share from the table below.
3. No share yet? Claim one of the three open shares today. The steps are under "How to claim a share".
4. Send your GitHub username to the lead in the group chat. You need it to push code.

## The six shares

The work is cut into six shares of the same size. Each share has 24 hours of Tier 1 work and 4 hours of Tier 2 work. The hours are estimates. They include your own slides and your own recording. The count, task by task, is in [work/README.md](work/README.md).

| Share | Role | Owner | Tier 1 | Tier 2 | Status |
|---|---|---|---|---|---|
| [A](work/A-framework-integration.md) | Framework and integration | Xingyang (lead) | 24 h | 4 h | Assigned |
| [B](work/B-literature-data-audit.md) | Literature, data checks and audit | Wenyue | 24 h | 4 h | Assigned |
| [C](work/C-compute-runs-efficiency.md) | Compute, runs and efficiency | Qiutian | 24 h | 4 h | Assigned |
| [D](work/D-baselines.md) | Baselines: classical, feature models, open LLM | **open** | 24 h | 4 h | [Claim in issue 4](https://github.com/Kankawa728-byte/levellens/issues/4) |
| [E](work/E-neural-models.md) | Neural models: BiLSTM and transformers | **open** | 24 h | 4 h | [Claim in issue 5](https://github.com/Kankawa728-byte/levellens/issues/5) |
| [F](work/F-interpretation-demo.md) | Interpretation, extrapolation and demo app | **open** | 24 h | 4 h | [Claim in issue 6](https://github.com/Kankawa728-byte/levellens/issues/6) |

- **Tier 1** is what we must ship.
- **Tier 2** is what we add if Tier 1 is on time. The lead decides on Thu 12 Nov at 21:00.

Which open share suits you:

- **D** suits someone who is comfortable with scikit-learn. Most of it runs on a laptop CPU.
- **E** is on the critical path. It suits someone who has fine-tuned a transformer before.
- **F** suits someone who likes plots and front ends. The app uses Streamlit.

## How to claim a share

1. Read the pages of shares D, E and F.
2. Open the issue of the share you want and write the comment "I take this", with your name. The first comment gets the share. If you have no GitHub account, tell the lead in the group chat instead.
3. The lead writes your name in the table above.

Claim by **Tue 10 Nov 23:59**. The lead assigns any share that is still open on Wed 11 Nov at 09:00.

## Key dates

| When | What |
|---|---|
| Tue 10 Nov 23:59 | Open shares claimed |
| Wed 11 Nov 12:00 | Code skeleton in `main` |
| Wed 11 Nov 23:59 | Every model family has written one result row |
| Thu 12 Nov 12:00 | Specifications frozen; Tier 1 grid handed to the GPU |
| Thu 12 Nov 21:00 | Go or no-go for Tier 2 |
| Fri 13 Nov 12:00 | **Results freeze**: no new experiments |
| Fri 13 Nov 23:59 | All tables and figures final |
| Sat 14 Nov 10:00 | Final slides; dry run at 10:30 |
| Sat 14 Nov 16:00 | Everyone's recording is with the lead |
| Sat 14 Nov 22:00 | Sign-off: video, reflection, code |
| Sun 15 Nov 12:00 | Upload, latest 18:00 |
| Sun 15 Nov 23:59 | Hard deadline |

The full schedule, day by day and share by share, is in [docs/schedule.md](docs/schedule.md).

## What we build

- **Task.** Predict the target level of a question: exam stage, school grade or competition level. The levels are ordered.
- **Data.** Three trained datasets: RACE++ (English reading, 3 levels), ARC (school science, 7 grades), MATH (competition maths, 5 levels). ScienceQA tests how far the science model extrapolates.
- **Models.** Floors, classical and feature models, a BiLSTM, DeBERTa-v3 and DistilBERT with ordinal output heads, and an open LLM without training.
- **Evaluation.** Nine metrics, published baselines on the same test splits, and eight measurable specifications with pass or fail.
- **Interpretation.** SHAP and LIME, an audit of the worst errors, transfer between subjects, and the reading of the score as an item difficulty.
- **Demo.** A Streamlit app in which a teacher scores a question, sees why it is hard, edits it and scores it again.

The details are in [docs/design.md](docs/design.md).

## Repository layout

The letter is the share that owns the file. Each file has one owner.

```text
levellens/                    the Python package
  data/                 A     race.py  arc.py  math.py  scienceqa.py   -> one schema
  features/             D     tfidf.py  linguistic.py
                        E     glove.py  latex.py
  models/               D     classic.py  llm.py
                        E     recurrent.py  transformer.py
  heads/                E     ce.py  corn.py
  explain/              D     global_shap.py
                        F     local.py  errors.py
  evaluate.py           A     all metrics, bootstrap intervals
  figures.py            F     result and error figures
  predictor.py          A     one Predictor API for the experiments and the app
  results.py            A     runs.jsonl -> tables
  run.py                A     python -m levellens.run data=arc model=deberta-v3 head=corn seed=0
configs/                A     layout; every owner adds the configs of their own models
scripts/                C     download_data.py, stage_cache, job templates
notebooks/main.ipynb    A     small end-to-end run, then rebuilds every table
app/                    F     streamlit_app.py, pages/
tests/                  A     shared tests; every owner adds tests for their own modules
results/                      runs.jsonl, tables, figures
docs/                   A, B  design, interfaces, schedule, workflow, video; B adds datasets, related work, audit
work/                   A     the six shares
```

The code skeleton arrives on Wed 11 Nov at 12:00. Until then, code against [docs/interfaces.md](docs/interfaces.md).

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
