# scripts/ — compute, runs and efficiency (share C)

Everything needed to take a config from this repository to a result row on a GPU machine, and to
say afterwards what it cost. The share is described in [work/C-compute-runs-efficiency.md](../work/C-compute-runs-efficiency.md).

Nothing here names a machine, an account or a folder of any cluster. The machine is described in
`scripts/site.env`, which git ignores; copy `scripts/site.env.example` and fill it in.

## In order

```bash
cp scripts/site.env.example scripts/site.env   # and fill it in
scripts/submit.sh setup                        # 1. build the environment      (CPU, needs internet)
scripts/submit.sh fetch --all                  # 2. splits and model weights   (CPU, needs internet)
scripts/submit.sh -r results/smoke grid configs/grid/smoke_gpu.txt    # 3. one tiny run per family (GPU)
scripts/submit.sh -r results/jobs/t1a grid configs/grid/tier1_transformers.txt 1-5   # 4. the grid (GPU)
python scripts/runlog.py                       # 5. what was submitted and how it ended
python scripts/collect.py results/jobs/*       # 6. the rows of every job into results/
python -m levellens.results                    # 7. rebuild the leaderboards
python scripts/efficiency.py                   # 8. the efficiency table, with R6 and R7
```

Give each GPU job a results store of its own with `-r`, so that two jobs never append to the same
`runs.jsonl` at the same time, and merge them afterwards with `collect.py`.

Steps 1 and 2 need internet; the GPU jobs do not, because they read the splits and the weights from
the cache that step 2 filled. On a machine with no scheduler, run the job scripts with `bash`
instead of submitting them: `bash scripts/job.sbatch data=arc model=distilbert head=ce`.

## The files

| File | What it is |
|---|---|
| `site.env.example` | the settings of one machine: folders, modules, how to activate the environment, what to ask the scheduler for |
| `job_setup.sh` | the prologue of every job: find the repository, read `site.env`, place the large folders, load the modules, activate the environment |
| `stage_cache` | copy the splits and the weights to the local disk of the compute node, so a run does not read them over the network |
| `setup.sbatch` | C1, CPU job: build the environment from `requirements.txt`, run the tests, print the versions |
| `fetch.sbatch` | C2, CPU job: `download_data.py`, then the offline check |
| `job.sbatch` | C3, GPU job: one run. Everything after the script name goes to `levellens.run` |
| `grid.sbatch` | C3, GPU job: a slice of a grid file, run after run, in one job |
| `tune.sbatch` | C4, GPU job: the Optuna search. One at a time, because the study is a single file |
| `submit.sh` | C3 and C4: submit one of the four job scripts and write the submission to `logs/submissions.tsv` |
| `download_data.py` | C2: the four datasets, DeBERTa-v3, DistilBERT, GloVe, the open LLM, and `--check` to prove a job can run offline |
| `env_report.py` | C1: the hardware and software table of `docs/compute.md`; `--require-gpu` fails a job that has no GPU |
| `runlog.py` | C4: the run log. Reads the submissions and asks the scheduler how each job ended |
| `collect.py` | C5: merge the result rows and predictions of the per-job stores into `results/` |
| `latency.py` | C6: seconds per question of a saved model, on the device you name |
| `efficiency.py` | C6: the efficiency table and the verdict on R6 and R7 |

`env_report.py`, `runlog.py` and `efficiency.py` use the standard library only, so they also work in
a half-built environment and in the clean-environment check of task C7.

## Where things are written

| | |
|---|---|
| `$LEVELLENS_DATA` | the split files, read-only during a run |
| `$HF_HOME` | the model weights, read-only during a run |
| `$LEVELLENS_CACHE` | tokenised splits, saved models, the Optuna study. A run writes here |
| `$LEVELLENS_RESULTS` | `runs.jsonl`, the predictions, the tables. A run writes here |
| `logs/` | job output, `submissions.tsv`, the version reports. git ignores it |

The four variables are explained in [docs/interfaces.md](../docs/interfaces.md). `site.env` can set
them one by one, or set `LEVELLENS_PROJECT_DIR` and let `job_setup.sh` place all four under it.

The hardware, the run log and the reproducibility notes of our own runs are in
[docs/compute.md](../docs/compute.md).
