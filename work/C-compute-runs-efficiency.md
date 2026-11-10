# Share C · Compute, runs and efficiency

**Owner:** Qiutian · **Buddy:** E · **Size:** 24 h Tier 1 + 4 h Tier 2

Every GPU run of the project goes through this share: the environment, the job scripts, the queue of runs, the returned results and the efficiency numbers.

## Files you own

- `scripts/` : `download_data.py`, `stage_cache`, job templates
- `docs/compute.md` : hardware and software specifications, run log, reproducibility notes
- The efficiency table in `results/`

## Tier 1

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| C1 | GPU environment built from `requirements.txt`. Record the GPU model, driver, CUDA, Python, PyTorch and Transformers versions. | `docs/compute.md` lists the versions. A job prints that the GPU is available. | 2 | Wed 11 Nov 14:00 |
| C2 | Offline cache. Datasets: RACE, RACE-C, MATH and ScienceQA from the Hugging Face Hub, and the ARC zip from AI2. Weights: DeBERTa-v3-base, DistilBERT, Qwen3-4B, GloVe 300d. `scripts/download_data.py` works on any machine. | A job in offline mode loads every dataset and every model. | 3 | Wed 11 Nov 20:00 |
| C3 | Job templates for a single run and for a batch of runs, with checkpoint and resume, each job under 1 hour. A smoke job with one tiny run of each GPU model family: BiLSTM, DistilBERT, open LLM scoring on 20 items. | The smoke job finishes and its rows are in `results/runs.jsonl`. | 3 | Wed 11 Nov 20:00 |
| C4 | Running the Tier 1 grid. Configs come from E on Thu 12 Nov at 12:00 and the LLM script from D at 18:00. Launch, monitor, retry failed runs. Keep a run log: what was submitted, when, job id, status. | Every Tier 1 run has a row or a documented failure at the results freeze. | 5 | Launch: Thu 12 Nov 14:00. Finish: Fri 13 Nov 12:00 |
| C5 | Returning results: merged `runs.jsonl`, logs, prediction files, final checkpoints. The DistilBERT demo checkpoint goes to F. | A, E and F have everything they need for tables, export and figures. | 2 | Fri 13 Nov 14:00 |
| C6 | Efficiency table. For every model: training time, GPU-hours per run and in total, parameter count. For the final models: latency per question, DistilBERT on a laptop CPU over 100 items and DeBERTa-v3 on the GPU. | The table is in `results/` and gives pass or fail for R6 (every fine-tune 1 GPU-hour or less, the whole project 20 GPU-hours or less) and R7 (1 second or less per question on a laptop CPU). | 4 | Fri 13 Nov 23:59 |
| C7 | Clean-environment run: new environment, follow the README only, run the small end-to-end example, rebuild the tables. Reproducibility notes: seeds, versions, commands. | Everything in the README works, or each failure is reported to the lead. The notes are in `docs/compute.md`. | 2 | Sat 14 Nov 18:00 |

The Tier 1 grid, so that you can plan the queue:

- Transformers: DeBERTa-v3-base and DistilBERT, two heads, three datasets, seed 0. On ARC also seeds 1 and 2. That is 20 runs.
- Optuna on DeBERTa-v3: 12 short trials per dataset, with pruning. Then the best setting re-trained.
- BiLSTM: 4 configurations on three datasets, two heads. Short runs.
- Open LLM: zero-shot scoring of three 1,000-item subsets.

If the GPU cluster cannot be used for this project, tell the lead at once. The fallback is the lead's workstation with an RTX 5090 (32 GB). You still prepare the scripts, the run list and the run log; the lead starts the jobs.

Do not write node names, account names, user names or internal addresses of the cluster in this public repository.

## Tier 2

| Id | Task | Done when | Hours |
|---|---|---|---|
| C8 | Running the Tier 2 grid: two more heads, BERT replication, RoBERTa, 3 seeds for the headline settings, few-shot LLM, BiLSTM Optuna | Every Tier 2 run has a row or a documented failure at the results freeze. | 3 |
| C9 | Latency for every model family | The efficiency table has a latency column for all families. | 1 |

## Others wait for you

- E, F and A: result rows and checkpoints as the runs finish, and everything by Fri 13 Nov 14:00.
- A: the clean-environment report on Sat 14 Nov at 18:00.

## You wait for

- A: `requirements.txt` and the run command (skeleton v0, Wed 11 Nov 12:00).
- E: Tier 1 grid configs on Thu 12 Nov at 12:00.
- D: the open LLM scoring script on Thu 12 Nov at 18:00.

## Video part

5:30–6:30: how one run starts from one config, the queue of runs on the GPU, the GPU-hours, and the efficiency and latency table with pass or fail for R6 and R7.
