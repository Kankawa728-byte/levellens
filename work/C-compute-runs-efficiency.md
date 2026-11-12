# Share C · Compute, runs and efficiency

中文版：[C-compute-runs-efficiency.zh-CN.md](C-compute-runs-efficiency.zh-CN.md)

**Owner:** Qiutian · **Buddy:** E · **Size:** 24 h Tier 1 + 4 h Tier 2

Every GPU run of the project goes through this share: the environment, the job scripts, the queue of runs, the returned results and the efficiency numbers.

## Files you own

- `scripts/` : `download_data.py`, `stage_cache`, job templates
- `docs/compute.md` : hardware and software specifications, run log, reproducibility notes
- The efficiency table in `results/`

## Tier 1

Re-plan of Thu 12 Nov: the dates are new, and a task marked "Cut" is not done. The hours are the first estimates. The whole re-plan is in [docs/schedule.md](../docs/schedule.md).

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| C1 | GPU environment built from `requirements.txt`. Record the GPU model, driver, CUDA, Python, PyTorch and Transformers versions. | `docs/compute.md` lists the versions. A job prints that the GPU is available. | 2 | Thu 12 Nov 23:59 |
| C2 | Offline cache. Datasets: run `python -m levellens.data.prepare` on a machine with internet, then copy the `data/` folder. Weights: DeBERTa-v3-base, DistilBERT, Qwen3-4B, GloVe 300d. `scripts/download_data.py` fetches the weights on any machine. | A job in offline mode loads every dataset and every model. | 3 | Fri 13 Nov 10:00 |
| C3 | Job templates for a single run and for a batch of runs, each job under 1 hour. A smoke job with one tiny run of each GPU model family that exists by then: DistilBERT first, then the BiLSTM and open LLM scoring on 20 items. | The smoke job finishes and its rows are in `results/runs.jsonl`. | 3 | Fri 13 Nov 12:00 |
| C4 | Running the Tier 1 grid. The configs come from E and the LLM script from D on Fri 13 Nov at 12:00. Launch, monitor, retry failed runs. Keep a run log: what was submitted, when, job id, status. | Every Tier 1 run has a row or a documented failure at the results freeze. | 5 | Launch: Fri 13 Nov 13:00. Finish: Fri 13 Nov 23:59 |
| C5 | Returning results: merged `runs.jsonl`, logs, prediction files, final checkpoints. The DistilBERT demo checkpoint goes to F. | A, E and F have everything they need for tables, export and figures. | 2 | Sat 14 Nov 10:00 |
| C6 | Efficiency table. For every model: training time, GPU-hours per run and in total, parameter count. For the final models: latency per question, DistilBERT on a laptop CPU over 100 items and DeBERTa-v3 on the GPU. | The table is in `results/` and gives pass or fail for R6 (every fine-tune 1 GPU-hour or less, the whole project 20 GPU-hours or less) and R7 (1 second or less per question on a laptop CPU). | 4 | Sat 14 Nov 12:00 |
| C7 | Clean-environment run: new environment, follow the README only, run the small end-to-end example, rebuild the tables. Reproducibility notes: seeds, versions, commands. | Everything in the README works, or each failure is reported to the lead. The notes are in `docs/compute.md`. | 2 | Sun 15 Nov 10:00 |

The Tier 1 grid, so that you can plan the queue:

- Transformers: DeBERTa-v3-base and DistilBERT, two heads, three datasets, seed 0. On ARC also seeds 1 and 2. That is 20 runs.
- Optuna on ARC with DistilBERT: 12 short trials, with pruning. Then the best setting re-trained.
- BiLSTM: one configuration on three datasets, two heads. Short runs.
- Open LLM: zero-shot scoring of three 1,000-item subsets.

The GPU cluster can be used for this project; this was confirmed on Thu 12 Nov. If that changes, tell the lead at once. The fallback is the lead's workstation with an RTX 5090 (32 GB). You still prepare the scripts, the run list and the run log; the lead starts the jobs.

Do not write node names, account names, user names or internal addresses of the cluster in this public repository.

## Tier 2

Cancelled on Thu 12 Nov. None of these tasks is done.

| Id | Task | Done when | Hours |
|---|---|---|---|
| C8 | Running the Tier 2 grid: two more heads, BERT replication, RoBERTa, 3 seeds for the headline settings, few-shot LLM, BiLSTM Optuna | Every Tier 2 run has a row or a documented failure at the results freeze. | 3 |
| C9 | Latency for every model family | The efficiency table has a latency column for all families. | 1 |

## Others wait for you

- E, F and A: result rows and checkpoints as the runs finish, and everything by Sat 14 Nov 10:00.
- A: the clean-environment report on Sun 15 Nov at 10:00.

## You wait for

- A: `requirements.txt` and the run command. They are in `main` since Thu 12 Nov.
- E: Tier 1 grid configs on Fri 13 Nov at 12:00.
- D: the open LLM scoring script on Fri 13 Nov at 12:00.

## Video part

5:30–6:30: how one run starts from one config, the queue of runs on the GPU, the GPU-hours, and the efficiency and latency table with pass or fail for R6 and R7.
