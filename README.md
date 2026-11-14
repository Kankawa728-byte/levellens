# LevelLens

LevelLens estimates the target difficulty level of an exam question from its text. A teacher pastes a new question and sees which level it is aimed at, and why.

Group project of Table 38, six members. Hard deadline: **Sun 15 Nov 2026, 23:59**. All times on these pages are Singapore time.

中文版：[README.zh-CN.md](README.zh-CN.md). Every page has a Chinese version; the link is at the top of the page.

<!-- board:start -->
## Live status: who waits for whom

A program rebuilds this table from the files in the repository and from the reports in the six share issues. Last change: Sat 14 Nov 22:58.

**Most people wait for:** Qiutian · result rows of the Tier 1 grid (0 / 20 runs) · by Sat 14 Nov 10:00. Waiting, directly or through someone else: everyone.

| Who | Can do now | Waits for | Who waits for this share |
|---|---|---|---|
| **Xingyang** · [A](work/A-framework-integration.md)<br>🟡 Partly waiting<br>4 / 9 own tasks done<br>⏰ 2 past the time | A6 README, narrated driver notebook, docstring pass · Sat 14 Nov 22:00 ⏰<br>A5 Specification freeze, reviews and merges, daily integration, go or no-go calls · Sun 15 Nov 14:00 | **Qiutian**: result rows of the Tier 1 grid (0 / 20 runs) · by Sat 14 Nov 10:00 ⏰<br>**everyone**: own recording · by Sat 14 Nov 23:59 · for A7, A8 | **everyone**: final tables and figures · by Sat 14 Nov 12:00 ⏰<br>**Qiutian**: README and driver notebook · by Sat 14 Nov 22:00 ⏰<br>**everyone**: joined video for the review · by Sun 15 Nov 11:00 |
| **Wenyue** · [B](work/B-literature-data-audit.md)<br>🟢 Can work<br>2 / 7 own tasks done<br>⏰ 3 past the time | B4 Slide template, storyboard, recording guide and schedule · Sat 14 Nov 10:00 ⏰<br>B7 Submission checklist, peer-evaluation reminders · Sun 15 Nov 15:00 | **Zhifei**: list of the 15 worst errors · by Sat 14 Nov 10:00 ⏰ · for B3<br>**everyone**: final slides for proofreading · by Sat 14 Nov 16:00 ⏰ · for B5<br>**everyone**: lines for the reflection document · by Sun 15 Nov 10:00 · for B6 | nobody |
| **Qiutian** · [C](work/C-compute-runs-efficiency.md)<br>🟢 Can work<br>0 / 7 own tasks done<br>⏰ 6 past the time | C1 GPU environment, hardware and software specifications · Thu 12 Nov 23:59 ⏰<br>C2 Offline cache of datasets and model weights, download scripts · Fri 13 Nov 10:00 ⏰<br>C3 Job templates, smoke job · Fri 13 Nov 12:00 ⏰<br>+ 3 more | **Xingyang**: README and driver notebook · by Sat 14 Nov 22:00 ⏰ · for C7 | **Xingyang, Jikai, Zhifei**: result rows of the Tier 1 grid (0 / 20 runs) · by Sat 14 Nov 10:00 ⏰ |
| **Hailing** · [D](work/D-baselines.md)<br>✅ Finished<br>7 / 7 own tasks done | — | nobody | nobody |
| **Jikai** · [E](work/E-neural-models.md)<br>🟢 Can work<br>3 / 6 own tasks done<br>⏰ 3 past the time | E2 BiLSTM with GloVe and attention pooling (4 / 6 runs) · Fri 13 Nov 20:00 ⏰<br>E7 Optuna search on ARC with DistilBERT · Fri 13 Nov 23:59 ⏰ | **Qiutian**: result rows of the Tier 1 grid (0 / 20 runs) · by Sat 14 Nov 10:00 ⏰ · for E8 | **Zhifei**: final models behind the Predictor API · by Sat 14 Nov 10:00 ⏰ |
| **Zhifei** · [F](work/F-interpretation-demo.md)<br>🔴 Waiting<br>2 / 6 own tasks done<br>⏰ 4 past the time | — | **Jikai**: final models behind the Predictor API · by Sat 14 Nov 10:00 ⏰ · for F1, F3, F4<br>**Qiutian**: result rows of the Tier 1 grid (0 / 20 runs) · by Sat 14 Nov 10:00 ⏰ · for F1, F2, F3, F4 | **Wenyue**: list of the 15 worst errors · by Sat 14 Nov 10:00 ⏰ |

| Everyone hands in | To | By | Still missing from |
|---|---|---|---|
| Final slides for proofreading | Wenyue | Sat 14 Nov 16:00 ⏰ | everyone |
| Own recording | Xingyang | Sat 14 Nov 23:59 | everyone |
| Lines for the reflection document | Wenyue | Sun 15 Nov 10:00 | everyone |

Pull requests waiting for the lead's review: none.

- **Finished something?** Comment `done C4` on the issue of your share: [A](https://github.com/Kankawa728-byte/levellens/issues/1), [B](https://github.com/Kankawa728-byte/levellens/issues/2), [C](https://github.com/Kankawa728-byte/levellens/issues/3), [D](https://github.com/Kankawa728-byte/levellens/issues/4), [E](https://github.com/Kankawa728-byte/levellens/issues/5), [F](https://github.com/Kankawa728-byte/levellens/issues/6). Several at once: `done C1 C2`. What everyone hands in: `done slides`, `done recording`, `done reflection`. To take one back: `undo C4`.
- 🟢 the next task can be done now. 🟡 the next task waits for someone, a later one can be done. 🔴 every open task waits for someone.
- ⏰ marks what is past its planned time and is neither found in `main` nor reported.
- Files and result rows in `main` are found without a report. The table is rebuilt after every push and every comment, and every 15 minutes.
<!-- board:end -->

## Start here

1. Read this page to the end. It takes five minutes.
2. Open the page of your share from the table below.
3. Look at the live status table above. It shows what you can do now, whom you wait for and who waits for you.
4. Send your GitHub username to the lead in the group chat. You need it to push code.

## The six shares

The work is cut into six shares of the same size. Each share has 24 hours of Tier 1 work and 4 hours of Tier 2 work. The hours are estimates. They include your own slides and your own recording. The count, task by task, is in [work/README.md](work/README.md).

| Share | Role | Owner | Tier 1 | Tier 2 | Status |
|---|---|---|---|---|---|
| [A](work/A-framework-integration.md) | Framework and integration | Xingyang (lead) | 24 h | 4 h | Assigned |
| [B](work/B-literature-data-audit.md) | Literature, data checks and audit | Wenyue | 24 h | 4 h | Assigned |
| [C](work/C-compute-runs-efficiency.md) | Compute, runs and efficiency | Qiutian | 24 h | 4 h | Assigned |
| [D](work/D-baselines.md) | Baselines: classical, feature models, open LLM | Hailing | 24 h | 4 h | Assigned |
| [E](work/E-neural-models.md) | Neural models: BiLSTM and transformers | Jikai | 24 h | 4 h | Assigned |
| [F](work/F-interpretation-demo.md) | Interpretation, extrapolation and demo app | Zhifei | 24 h | 4 h | Assigned |

- **Tier 1** is what we must ship.
- **Tier 2** was what we would add if Tier 1 was on time. The lead cancelled it on Thu 12 Nov and also cut part of Tier 1. The re-plan is in [docs/schedule.md](docs/schedule.md). The hours in the table are the first estimates and were not counted again.

All six shares have had an owner since Fri 13 Nov.

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

## Install and run

You need git and Python 3.10 or newer. These steps were tested with Python 3.12 on Windows. Steps 1 to 3 need no GPU and no dataset.

### 1. Install

```bash
git clone https://github.com/Kankawa728-byte/levellens.git
cd levellens
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

- On Windows, type `.venv\Scripts\activate` in place of the fourth line.
- The environment takes about 1.6 GB of disk with the CPU build of PyTorch on Windows.
- `python -m pytest` takes less than a minute. Two tests need the downloaded datasets and are skipped until step 4.
- On Windows, keep the repository in a short path such as `C:\work\levellens`. PyTorch has files with very long names, and the install stops with an error when a full path passes the Windows limit of 260 characters.
- For runs on a GPU you need a CUDA build of PyTorch. The [PyTorch installation guide](https://pytorch.org/get-started/locally/) gives the command for your system.

### 2. Run the small example

```bash
python -m levellens.run data=toy model=majority
python -m levellens.run data=toy model=tfidf_logreg
python -m levellens.run data=toy model=bilstm head=corn model.params.embeddings=random save_model=true
python -m levellens.results --scratch
```

- The toy dataset is generated in memory, so nothing is downloaded. Each command takes a few seconds on a CPU.
- Each run trains one model, prints its metrics and appends one row to `results/scratch/runs.jsonl`. The last command builds the tables in `results/scratch/tables/`. Open `leaderboards.md` in that folder: it has one line per model.
- git ignores `results/scratch/`, so these try-outs never mix with the results of the project.

The driver notebook [notebooks/main.ipynb](notebooks/main.ipynb) makes the same runs and explains each step. It then scores a new question, explains the prediction, and rebuilds the tables and figures of step 3. Open it in Jupyter or VS Code and run all cells, or run it from the command line. It takes about one minute.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb
```

### 3. Rebuild the tables and figures

```bash
python -m levellens.results
python -m levellens.figures
```

- Both commands read `results/runs.jsonl`, which holds one row per run, and the saved predictions in `results/preds/`. Both are in the repository, so no dataset and no model are needed.
- The first command writes `results/tables/`: one CSV file per dataset and split, and `leaderboards.md` with all of them.
- The second command writes `results/figures/`: the leaderboard, the confusion matrices, the cue words, the transfer matrix and the extrapolation figure.

### 4. Train on the real datasets

```bash
python -m levellens.data.prepare
python -m levellens.run data=math model=tfidf_logreg
```

- The first command downloads the four datasets and writes the split files. It needs about 1.2 GB of disk. It then compares every split with the fingerprints in `levellens/data/manifest.json` and stops if they differ, so every machine trains and tests on the same questions.
- A run on a real dataset appends its row to `results/runs.jsonl`. To keep your runs apart from the rows in the repository, set the environment variable `LEVELLENS_RESULTS` to another folder first.

| Models | How to train them | Runs on |
|---|---|---|
| Floors, classical models, feature model, open LLM | The commands in [work/D-reproduce.md](work/D-reproduce.md) | CPU; the open LLM needs a GPU |
| BiLSTM | `python -m levellens.grid configs/grid/tier1_bilstm.txt` | GPU |
| DeBERTa-v3 and DistilBERT | `python -m levellens.grid configs/grid/tier1_transformers.txt` | GPU |
| Optuna search | `python -m levellens.tune data=arc model=distilbert head=ce owner=E n_trials=12` | GPU |

- The first BiLSTM run downloads the GloVe word vectors (480 MB). The transformers download their weights on the first run.
- Add `--list` to a grid command to print its runs without starting them.

### 5. Explanations, checks and the demo app

```bash
python -m levellens.explain.errors
python -m levellens.explain.local arc-distilbert-ce-s0
python -m levellens.explain.transfer
python -m levellens.export
streamlit run app/streamlit_app.py
```

- The first four commands need the datasets of step 4. The second, third and fourth also need a saved model: a run with `save_model=true` saves its model in `cache/models/<run id>/`. `arc-distilbert-ce-s0` is an example of a run id.
- `levellens.explain.errors` lists the worst errors of the best run of every dataset, for the human audit in [docs/audit.md](docs/audit.md).
- `levellens.explain.local` computes the LIME cue words of one run.
- `levellens.explain.transfer` computes the transfer matrix and the ScienceQA extrapolation.
- `levellens.export` checks that every saved model gives the same predictions after it is loaded, and measures the time per question on the CPU.
- The app needs no trained model. Its model list always holds a stand-in with made-up probabilities, and the saved models appear next to it.

## Repository layout

The letter is the share that owns the file. Each file has one owner.

```text
levellens/                    the Python package
  data/                 A     race.py  arc.py  math_.py  scienceqa.py  toy.py   -> one schema
                              prepare.py writes the split files that every model loads
  features/             D     tfidf.py  linguistic.py
                        E     glove.py
  models/               A     base.py (the model contract)  floors.py (majority, the worked example)
                        D     classic.py  llm.py
                        E     neural.py (the training loop)  recurrent.py  transformer.py
  heads/                E     base.py  ce.py  corn.py
  explain/              D     global_shap.py
                        F     local.py  errors.py  transfer.py
  evaluate.py           A     all metrics, bootstrap intervals
  figures.py            F     result and error figures
  predictor.py          A     one Predictor API for the experiments and the app
  results.py            A     runs.jsonl -> tables
  run.py                A     python -m levellens.run data=arc model=deberta-v3 head=corn seed=0
  grid.py  tune.py      E     run the lines of a grid file; Optuna search
  export.py             E     check the saved models behind the Predictor
  config.py  paths.py   A     the config of a run; the data, cache and results folders
configs/                A     layout; every owner adds the configs of their own models
scripts/                C     job templates and download scripts (not in main yet)
notebooks/main.ipynb    A     small end-to-end run, then rebuilds every table and figure
notebooks/data_checks.* B     data checks behind docs/datasets.md
app/                    F     streamlit_app.py, pages/
tests/                  A     shared tests; every owner adds tests for their own modules
results/                      runs.jsonl, saved predictions, tables, figures
docs/                   A, B  design, interfaces, schedule, workflow, video; B adds datasets, related work, audit
work/                   A     the six shares
```

The contracts between the modules are in [docs/interfaces.md](docs/interfaces.md): the fields of an item, the model contract, the result row and the Predictor.

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
