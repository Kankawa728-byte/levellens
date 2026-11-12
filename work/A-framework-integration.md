# Share A · Framework and integration

中文版：[A-framework-integration.zh-CN.md](A-framework-integration.zh-CN.md)

**Owner:** Xingyang (lead) · **Buddy:** F · **Size:** 24 h Tier 1 + 4 h Tier 2

The shared code that every other share plugs into, and putting the pieces together at the end: tables, package, video and upload.

## Files you own

- `levellens/data/`, `levellens/evaluate.py`, `levellens/results.py`, `levellens/predictor.py`, `levellens/run.py`
- `configs/data/`, `configs/train.yaml`, `requirements.txt`
- `tests/` (layout and the shared tests), `notebooks/main.ipynb`
- `README.md`, `docs/`, `work/`

## Tier 1

Re-plan of Thu 12 Nov: the dates are new, and a task marked "Cut" is not done. The hours are the first estimates. The whole re-plan is in [docs/schedule.md](../docs/schedule.md).

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| A1 | Package skeleton, configs, command line, device and paths from environment variables, Predictor API with a stub model | `python -m levellens.run data=math model=majority` writes one row to `results/runs.jsonl` on a laptop and on the GPU machine with no code change. `Predictor.load("stub")` returns probabilities. | 4 | Done on Thu 12 Nov |
| A2 | Unified schema and adapters for RACE++, ARC, MATH and ScienceQA (text only). Saved splits. One fixed 1,000-item stratified test subset per dataset. ARC grade-label check. | The counts match [docs/design.md](../docs/design.md), or the difference is explained in the issue. The splits are written once and every model loads the same files. | 5 | Done on Thu 12 Nov |
| A3 | Metrics module: accuracy, macro-F1, adjacent accuracy, QWK, MAE, RMSE, Spearman, Balanced DRPS, ECE. Paired bootstrap intervals. The majority-row test. | The majority baseline on our splits reproduces the published rows in [docs/design.md](../docs/design.md) for RACE++ and for ARC with the balanced training subset. | 3.5 | Done on Thu 12 Nov |
| A4 | Results store and table builder | One command rebuilds every table from `results/runs.jsonl`. | 2 | Done on Thu 12 Nov |
| A5 | Specification freeze (R1–R8), reviews and merges, daily integration, go or no-go calls | Specifications frozen on Thu 12 Nov. Pull requests reviewed within 12 hours. Tier 2 cancelled and Tier 1 cut on Thu 12 Nov. Late check on Fri 13 Nov at 15:00. Results freeze called on Fri 13 Nov at 23:59. | 2 | every day |
| A6 | README, narrated driver notebook, docstring pass | A new reader can install the package, run the small end-to-end example and rebuild the tables by following the README only. | 2 | Sat 14 Nov 22:00 |
| A7 | Video assembly | The six recordings are joined in storyboard order, the video is 9:40 or shorter, and it is shared for the 12:00 review. | 2 | Sun 15 Nov 11:00 |
| A8 | Final package and upload | The submission shows every file. Uploaded by Sun 15 Nov 15:00, latest 18:00. | 0.5 | Sun 15 Nov |

## Tier 2

Cancelled on Thu 12 Nov. None of these tasks is done.

| Id | Task | Done when | Hours |
|---|---|---|---|
| A9 | GSM8K adapter | F can run the MATH model on GSM8K (task F8). | 1 |
| A10 | Automated tests on every push | The tests run on GitHub for every pull request and the result shows on the pull request. | 1.5 |
| A11 | Guide "add your own dataset or model" with a worked example | A reader can add a new dataset by following the guide, without reading the rest of the code. | 1.5 |

## Others wait for you

- Everyone: the skeleton, the four adapters, the saved splits and the 1,000-item subsets. Done on Thu 12 Nov. Keep [docs/interfaces.md](../docs/interfaces.md) and the code in step.
- B: the loading code for the data checks (task B1).

## You wait for

- Result rows from C, D and E for the tables.
- Recordings from everyone by Sat 14 Nov 23:59.

## Video part

- Opening, 0:00–1:00: the teacher's problem, the "target level" wording, the research questions, our specifications.
- Closing, 8:50–9:40: pass or fail for each specification, limitations, future work, references.
