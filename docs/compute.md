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
Not filled in yet. `python scripts/env_report.py` prints this table on the machine that runs the
jobs; every job log starts with it.
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
Not filled in yet.
<!-- runlog:end -->

## 5. What goes back to the other shares

Task C5. After the grid, these are in `main`:

| What | Where | For |
|---|---|---|
| Result rows | `results/runs.jsonl`, one line per run | A: tables; F: figures |
| Predictions | `results/preds/<run_id>.npz` | F: confusion matrices, bootstrap intervals, error audit |
| Saved models | `<LEVELLENS_CACHE>/models/<run_id>/`, not committed | E: `levellens.export`; F: the app |
| Efficiency table | `results/tables/efficiency.{md,csv}`, `efficiency_runs.csv` | A: pass or fail for R6 and R7 |
| Job logs | `logs/`, not committed | whoever has to read why a run failed |

Model weights are not committed, as the rules in [workflow.md](workflow.md) say. The DistilBERT
checkpoint for the demo (share F) is handed over outside the repository.

## 6. Reproducibility notes

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
Not filled in yet.
<!-- clean:end -->
