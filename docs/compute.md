# Compute, runs and efficiency

中文版：[compute.zh-CN.md](compute.zh-CN.md)

Owner: share C, [work/C-compute-runs-efficiency.md](../work/C-compute-runs-efficiency.md).

This page holds three things: the machine the GPU runs were made on, the log of what was submitted,
and the notes that let someone else get the same numbers. The scripts behind it are in
[scripts/README.md](../scripts/README.md); the efficiency table itself is
[results/tables/efficiency.md](../results/tables/efficiency.md), rebuilt by one command.

This repository is public, so no machine name, account name or user folder of the cluster appears
here or in the scripts. They live in `scripts/site.env`, which git ignores.

## 1. The machine

GPU runs: a shared university GPU cluster with a Slurm scheduler. Login nodes are for editing and
submitting only, so the environment is built and the datasets are downloaded by CPU jobs, and every
run is a batch job. Compute nodes have no internet, which is why task C2 exists.

CPU runs of the baselines (share D) and the BiLSTM rows marked `mps` (share E) were made on the
members' own laptops. The `device` field of every row in `results/runs.jsonl` says which machine
kind a run used, and the efficiency table counts the hours per kind.

<!-- machine:start -->
The versions below are what `python scripts/env_report.py` printed at the start of the smoke job on
14 Nov 2026. Every job log starts with this table, so each run can be traced to the machine it ran on.

| What | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| Driver | 595.71.05 |
| Operating system | Linux 6.8.0, glibc 2.39 |
| Python | 3.12.9, in a virtual environment on the project directory |
| PyTorch | 2.11.0+cu128, built for CUDA 12.8. Sees the GPU, bfloat16 supported |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| datasets | 5.2.0 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| optuna | 5.0.0 |
| lightgbm | 4.7.0 |

Two notes on the versions:

- **The card is not Ampere.** R6 asks for 20 GPU-hours on an Ampere-class card and allows another
  card if the actual hours are reported, which is what the efficiency table does. The Ampere cards
  of the cluster were fully booked. This card is newer and faster, so our hours are **lower** than
  the same runs would cost on an Ampere card: passing the 20-hour budget here does not prove that
  it would be passed on Ampere, and how much lower is not measured. Read the number as what the
  project actually cost on this card, which is what R6 allows, and not as an Ampere figure.
- **transformers is pinned to 4.57.6**, not the newest release. `requirements.txt` gives lower
  bounds, so pip installs 5.23.0, while `levellens/models/llm.py` is written for `>=4.57.1,<5` and
  share D's rows were produced with 4.57.6. The pin is `LEVELLENS_PIP_EXTRA` in `scripts/site.env`,
  so every share's numbers come from the same version and `requirements.txt` stays as the lead
  wrote it. A Blackwell card needs a cu128 build of PyTorch, which is `LEVELLENS_TORCH_INDEX`.
<!-- machine:end -->

## 2. How a run gets to the GPU

```bash
cp scripts/site.env.example scripts/site.env          # once: folders, modules, GPU request
scripts/submit.sh setup                               # CPU job: environment from requirements.txt
scripts/submit.sh fetch --all                         # CPU job: splits and weights into the cache
scripts/submit.sh -r results/smoke grid configs/grid/smoke_gpu.txt     # GPU job: one tiny run per family
scripts/submit.sh -r results/jobs/t1a grid configs/grid/tier1_transformers.txt 1-5   # GPU job: five runs
scripts/submit.sh tune data=arc model=distilbert head=ce owner=E n_trials=12            # GPU job: Optuna
python scripts/runlog.py                              # what was submitted and how it ended
python scripts/collect.py results/jobs/*              # the rows of every job into results/
python -m levellens.results                           # rebuild the leaderboards
python scripts/efficiency.py                          # the efficiency table, with R6 and R7
```

Each GPU job writes to a results store of its own, named by `-r`. Two jobs then never append to the
same `runs.jsonl` at the same time, and `scripts/collect.py` merges them through
`levellens.results.append_run` afterwards. That merge is task C5.

One config in, one row out: `scripts/job.sbatch` passes everything after its name to
`python -m levellens.run`, and nothing else decides what a run does. That is specification R8 on the
compute side.

The large folders are kept off the home folder, because the quota there is small:
`LEVELLENS_PROJECT_DIR` in `site.env` places the splits, the Hugging Face cache, the checkpoints and
the results under one folder on shared storage. The four variables are explained in
[interfaces.md](interfaces.md).

### Offline mode

The GPU jobs export `HF_HUB_OFFLINE=1`. Everything they need is fetched beforehand by
`scripts/download_data.py`: the four datasets through `python -m levellens.data.prepare`, which also
checks every split against `levellens/data/manifest.json`, and the weights of DeBERTa-v3-base,
DistilBERT, GloVe 6B 300d and Qwen3-4B-Instruct-2507 at the revision pinned in
`levellens/models/llm.py`. `python scripts/download_data.py --check` repeats the load with the hub
switched off, so a missing file shows up in the CPU job and not in a GPU job.

`scripts/stage_cache` then copies the splits and the weights to the local disk of the compute node
at the start of each job. Only read-only things are copied; the checkpoints and the result rows are
written straight to shared storage, so they outlive the job.

## 3. Decisions and deviations from the plan

| What the plan says | What was done | Why |
|---|---|---|
| C3: each job under one hour | One run per job is the template (`job.sbatch`, one hour). The Tier 1 grid was submitted in slices of five runs (`grid.sbatch`) | The scheduler lets one user run only a few jobs at a time. A slice of five waits in the queue once instead of five times, which is faster end to end than 20 one-run jobs |
| R6: 20 GPU-hours on an Ampere-class card | The hours on the card the scheduler actually gave are reported, per run and in total | R6 allows this: "On another card we report the actual hours." The Ampere cards of the cluster were fully booked |
| C2: cache the datasets by hand | `python -m levellens.data.prepare` writes them and the manifest check guards them | The re-plan of Thu 12 Nov cut this: only the weights are cached by hand |
| C4: Optuna, 12 trials | Run as its own job, because `levellens.tune` keeps one study file and must not run twice at the same time | The study is a single SQLite file in the cache |

The fallback in the share description, the lead's own workstation, was not needed.

### Three ways a job quietly writes into the wrong place

All three cost us a failed job before they were fixed, and all three are silent: the job does not
say it is writing somewhere else, it only fails later with a disk quota.

1. **`import levellens` succeeds with any Python when the working directory is the repository**,
   because `levellens/` is a folder right there. So a check of "is the environment built" passes
   before anything is installed. `scripts/job_setup.sh` asks from another directory instead, which
   is what really tests `pip install -e .`.
2. **pip falls back to a user install** into `~/.local` when it cannot write to the site-packages of
   the Python it was run with. A half-downloaded PyTorch is 2.5 GB in a home folder with a quota.
   `job_setup.sh` sets `PIP_USER=0` and `PYTHONNOUSERSITE=1`, and `setup.sbatch` refuses to install
   unless the target folder can be written to.
3. **A job inherits `TMPDIR`**, usually `/tmp`, which can be a small per-user volume. pip downloads
   wheels there. `job_setup.sh` now sets `TMPDIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`,
   `TRITON_CACHE_DIR` and `CONDA_PKGS_DIRS` under `LEVELLENS_PROJECT_DIR` instead of defaulting to
   them, because moving them is the point of naming a project directory.

## 4. Run log

What was submitted, when, with which job id, and how it ended. `scripts/submit.sh` writes every
submission to `logs/submissions.tsv`, and `python scripts/runlog.py` asks the scheduler for the
state and prints this table.

<!-- runlog:start -->
| Submitted | Kind | Job | What | State | Elapsed |
|---|---|---|---|---|---|
| 2026-11-14 08:07:24 | setup | 1738348 | levellens-setup | FAILED (1:0) | 00:00:14 |
| 2026-11-14 08:07:30 | fetch | 1738349 | --all | CANCELLED | 00:00:00 |
| 2026-11-14 08:10:57 | setup | 1738361 | levellens-setup | COMPLETED | 00:06:37 |
| 2026-11-14 08:10:57 | fetch | 1738362 | --all | COMPLETED | 00:03:18 |
| 2026-11-14 08:22:25 | setup | 1738385 | levellens-setup | COMPLETED | 00:01:44 |
| 2026-11-14 08:22:25 | fetch | 1738386 | --check | COMPLETED | 00:00:12 |
| 2026-11-14 08:22:25 | grid | 1738387 | configs/grid/smoke_gpu.txt | COMPLETED | 00:04:14 |
| 2026-11-14 19:07:01 | grid | 1739442 | configs/grid/tier1_transformers.txt 1-10 | COMPLETED | 00:38:47 |
| 2026-11-14 19:07:01 | grid | 1739443 | configs/grid/tier1_transformers.txt 11-20 | COMPLETED | 00:11:52 |
| 2026-11-14 19:07:01 | grid | 1739444 | configs/grid/tier1_bilstm.txt 5-6 | COMPLETED | 00:02:30 |
| 2026-11-14 19:07:01 | tune | 1739445 | data=arc model=distilbert head=ce owner=E n_trials=12 | COMPLETED | 00:03:23 |
| 2026-11-14 19:49:51 | measure | 1739497 | gpu race-deberta-v3-ce-s0 arc-deberta-v3-ce-s0 math-deberta-v3-ce-s0 arc-distilbert-corn-s0 | COMPLETED | 00:00:54 |
| 2026-11-14 19:49:51 | measure | 1739498 | cpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 | COMPLETED | 00:13:57 |
| 2026-11-14 20:06:36 | measure | 1739513 | gpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 race-deberta-v3-ce-s0 arc-deberta-v3-ce-s0 math-deberta-v3-ce-s0 | COMPLETED | 00:01:04 |
| 2026-11-14 20:06:36 | measure | 1739514 | cpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 | COMPLETED | 00:13:40 |
| 2026-11-14 20:24:26 | clean | 1739543 | levellens-clean | COMPLETED | 00:12:57 |

How to read it:

- The first setup job failed after installing PyTorch into the wrong place; the three ways that can
  happen are in section 3 and are fixed. The fetch job after it never ran, because it was to start
  only if the setup succeeded.
- The second setup job built the environment and ran the tests. The fetch job downloaded the four
  datasets, the two encoders, GloVe and the 8 GB open LLM, and its offline check loaded every split
  and every model with the hub switched off.
- The third setup job pinned transformers to 4.57.6, and the fetch after it repeated the offline
  check under that version.
- The smoke job trained, scored and wrote a row for each of the three families that need a GPU, in
  four minutes on one card.
- `t1-tr-a` and `t1-tr-b` are the 20 transformer runs, ten each; `t1-bilstm` the two RACE++ BiLSTM
  runs that were still missing; `levellens-tune` the Optuna search with the best setting retrained.
  All four completed with no failed run, two at a time, in 56 minutes of wall-clock time.
- The `measure` jobs are task C6. The gpu one checks the saved models at the precision they trained
  in and times them on the GPU; the cpu one times them on a CPU, which is specification R7. They
  were run twice: the first pair measured, the second pair added the reload check on the GPU after
  the CPU check had failed on a precision difference, see section 6.
- `levellens-clean` is task C7, section 7.

No Tier 1 run failed, so there is no documented failure to report for task C4.
<!-- runlog:end -->

## 5. What goes back to the other shares

Task C5, done on 14 Nov. These are in the branch of this share:

| What | Where | For |
|---|---|---|
| 23 new result rows | `results/runs.jsonl`, 56 rows in all | A: tables; F: figures |
| 23 prediction files | `results/preds/<run_id>.npz` | F: confusion matrices, bootstrap intervals, error audit |
| Optuna trial table | `results/optuna/arc-distilbert.csv` and `.json` | E: task E7 |
| Latency reports | `results/export/`, `results/latency/` | A: pass or fail for R7 |
| Efficiency table | `results/tables/efficiency.{md,csv}`, `efficiency_runs.csv` | A: pass or fail for R6 |
| Rebuilt tables and figures | `results/tables/`, `results/figures/` | A, B, F: slides |
| 23 saved models, 9.8 GB | the project directory, not committed | E: `levellens.export`; F: the app |
| Job logs | `logs/`, not committed | whoever has to read why a run failed |

Every row went in through `levellens.results.append_run`, by way of `scripts/collect.py`: each job
wrote to a results store of its own, so no two jobs ever appended to the same `runs.jsonl`.

Model weights are not committed, as the rules in [workflow.md](workflow.md) say. The DistilBERT
checkpoint for the demo, `arc-distilbert-corn-s0`, is the fastest of the saved models at 0.003
seconds per question on the GPU and 0.056 on a CPU; it is handed to share F outside the repository.

## 6. Efficiency and latency

Task C6. The table is [results/tables/efficiency.md](../results/tables/efficiency.md), rebuilt by
`python scripts/efficiency.py` from `runs.jsonl` and the latency reports. One line per model, one
line per run, and the verdict on the two specifications this share owns.

| | Measured | Limit | |
|---|---|---|---|
| **R6**, per fine-tune | 0.19 hours, the longest of 28 | 1 GPU-hour | pass |
| **R6**, whole project | 0.89 GPU-hours | 20 GPU-hours | pass |
| **R7**, one question on a CPU | 0.028 to 0.058 seconds over 4 saved DistilBERT models | 1 second | pass |

The project limit counts GPU-hours only. On top of them the project spent 0.45 CPU-hours for the
classical baselines and the feature model, and 0.41 hours on a laptop GPU for the first BiLSTM rows;
the efficiency table lists the three kinds apart, because an hour of each is not the same thing.
Read the GPU figure as what the project cost on this card, not as an Ampere figure: see section 1.

The 20 transformer runs took 51 minutes of GPU time altogether, which is why 0.89 is so far inside
the budget. The longest fine-tune of the whole project is not a transformer at all but the first
BiLSTM run on MATH, on a laptop GPU.

Latency, one question at a time, as the app scores a pasted question:

| Model | On a CPU | On the GPU | In batches of 32 on the GPU |
|---|---|---|---|
| DistilBERT | 0.046 to 0.058 s | 0.003 s | 0.0003 s |
| DeBERTa-v3 | not measured | 0.014 to 0.015 s | 0.0006 to 0.0011 s |

- The CPU numbers come from `python -m levellens.export`, which is also the check of task E8. R7 asks
  for a laptop CPU; these were measured on a compute node, so they are a server CPU. The one laptop
  measurement in the repository, 0.028 s for the DistilBERT smoke model, was made by another share
  and is in the same table. Both are more than an order of magnitude inside the limit, so the
  verdict does not depend on which machine is meant.
- The GPU numbers are `python scripts/latency.py --device cuda`. DeBERTa-v3 answers one question in
  15 milliseconds, so the demo would feel instant on a GPU and still well under a second on a CPU.

### One thing for share E, task E8

`levellens.export` reports **same predictions False** for every saved model: the largest difference
between the Predictor's probabilities and the ones the training run stored is 6.2e-03 to 2.4e-02,
against the tolerance of 1e-03 in `levellens/export.py`.

Saving and loading are not at fault. The same check with the model on the GPU, the precision the
runs trained in, gives a largest difference of **3.0e-08** on all six models tested. The difference
is bfloat16 against fp32: bfloat16 has an 8-bit mantissa, so the two agree to about the second or
third decimal, not the fourth as the comment on `TOLERANCE` says. `scripts/measure.sbatch` runs both
checks, the GPU one first.

Whether to compare at the training precision or to widen the tolerance is share E's call;
share C changed neither.

## 7. Reproducibility notes

Task C7. What someone needs to get the same numbers.

**Commands.** Install and the small end-to-end example are in [the README](../README.md). The runs
behind the committed rows: the baselines in [work/D-reproduce.md](../work/D-reproduce.md), the
neural grids in `configs/grid/`, and the grid files are the record of what was run — one line per
run, with every setting that differs from the config file.

**Seeds.** `seed=` seeds Python, NumPy and PyTorch, see `levellens.run.set_seed`. The splits do not
depend on the seed: `levellens.data.prepare` picks `test_1k` by a hash of the item id, so every
machine gets the same 1,000 questions. The Optuna study has its own seed in `levellens.tune`.

**Versions.** The table in section 1, and the exact pins of the open LLM in
[work/D-reproduce.md](../work/D-reproduce.md). `n_params` and `device` in every result row say what
the run actually used.

**What is not bit-for-bit repeatable.**

- A different GPU model, or a different PyTorch or CUDA build, changes the last decimals of a
  fine-tune: the kernels sum in a different order. The metrics move in the third decimal, not more.
- `precision: auto` trains in bfloat16 on a GPU that supports it and in fp32 elsewhere, so a CPU
  rerun of a GPU run does not give the same weights. `model.params.precision=fp32` makes it equal.
- Early stopping reads dev QWK, so a run that stops one evaluation earlier or later gives a
  different model. `max_minutes` can also end a run on a slower card after fewer epochs; the result
  row keeps `train_seconds`, so this is visible.
- Everything else is seeded: the batches of an epoch come from `length_buckets` with a generator
  seeded by the run seed, and the class weights are computed from the training labels.

**Clean-environment run.**

<!-- clean:start -->
Run on 14 Nov as `scripts/submit.sh clean`, job 1739543, on the branch at commit `e23164f`.
`scripts/clean_check.sbatch` clones the branch into an empty folder, builds a new virtual
environment from `requirements.txt` and then runs the commands of the README, steps 1 to 3, with
every cache and temporary folder inside the throw-away directory and with all `LEVELLENS_*`
variables unset, so it runs as a new reader of the README would.

It took 13 minutes, and everything in the README worked:

| Step | Result |
|---|---|
| 1. Install and `python -m pytest` | 112 passed, 2 skipped. The two skipped are the ones marked `data`, which need the downloaded datasets |
| 2. The small end-to-end example | The four commands ran; three rows in `results/scratch/runs.jsonl` and the leaderboard built |
| 3. `levellens.results` and `levellens.figures` | **Every table and every figure came out identical to the committed ones, byte for byte** |

The third line is specification R8 on the compute side, and it holds more strongly than expected:
this new environment installed the **newest** release of everything, PyTorch 2.15.0+cu132 and
transformers 5.23.0, while the runs were made with PyTorch 2.11.0+cu128 and transformers 4.57.6.
The tables still matched, because rebuilding them re-runs no model: `levellens.results` and
`levellens.figures` read `runs.jsonl` and the saved predictions, both of which are in the
repository. So anyone can rebuild every number of the report from a fresh clone, on any machine,
with no GPU, no dataset and no particular version of PyTorch.

Two things a new reader should know, neither of them a failure of the README:

- The README says to clone from GitHub. This job clones the local branch instead, because the branch
  was not pushed yet when it ran. Everything else is the README as written.
- On this cluster `python` comes from a module, so the job loads it before step 1. A laptop needs
  nothing of the kind.
<!-- clean:end -->
