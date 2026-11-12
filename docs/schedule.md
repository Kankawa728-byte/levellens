# Schedule

中文版：[schedule.zh-CN.md](schedule.zh-CN.md)

All times are Singapore time. The hard deadline is **Sun 15 Nov 2026, 23:59**.

## Re-plan of Thu 12 Nov

The first plan counted on a start on Tue 10 Nov. The code skeleton reached `main` on the afternoon of Thu 12 Nov, more than a day late, and shares E and F had no owner yet. So the lead changed the plan on Thu 12 Nov:

- **Tier 2 is cancelled.** None of its tasks is done.
- **Tier 1 is cut.** The list is under "What was cut" below. Each share page marks its own cut tasks.
- **The results freeze moves** from Fri 13 Nov 12:00 to Fri 13 Nov 23:59. Slides, recordings and the video move to Saturday afternoon and Sunday morning. The upload stays on Sunday, latest 18:00.
- **The specifications R1–R8 are frozen** as written in [design.md](design.md).

Every date on this page and on the share pages is a date of this re-plan.

## Milestones

| When | What | Who |
|---|---|---|
| Thu 12 Nov 16:00 | Skeleton in `main`: schema, four adapters, saved splits, metrics, results store, stub Predictor. Specifications R1–R8 frozen. | A |
| Thu 12 Nov 23:59 | Shares E and F have owners. Everyone has done the first run in [interfaces.md](interfaces.md). First result rows: TF-IDF with Logistic Regression, a DistilBERT smoke run. The Calibrate page runs on the stub. | everyone |
| Fri 13 Nov 12:00 | Transformer trainer complete. Tier 1 grid configs and the LLM script handed to C. Slide template shared. | E, D, B |
| Fri 13 Nov 13:00 | The Tier 1 grid is launched on the GPU | C |
| Fri 13 Nov 15:00 | Late check: the lead cuts more if a share is still behind | A |
| Fri 13 Nov 23:59 | **Results freeze.** No new experiments. | everyone |
| Sat 14 Nov 10:00 | Everything returned from the GPU. Final models behind the Predictor API. Worst-error list with B. Slide drafts exist. | C, E, F, everyone |
| Sat 14 Nov 12:00 | All tables and figures final. Efficiency table done. Pass or fail for R1–R8. | A, C, F |
| Sat 14 Nov 16:00 | Final slides. Timed dry run online at 16:30. | everyone |
| Sat 14 Nov 23:59 | Recordings with the lead | everyone |
| Sun 15 Nov 11:00 | The lead shares the joined video | A |
| Sun 15 Nov 12:00 | Full-video review | everyone |
| Sun 15 Nov 14:00 | Sign-off: video final, reflection signed, code frozen | everyone |
| Sun 15 Nov 15:00 | Upload, latest 18:00. Only the lead submits. | A |
| Sun 15 Nov 23:59 | Hard deadline. Nobody creates a new attempt after this. | |

Friday evening is class time. Plan your Friday work for the morning and the afternoon. GPU jobs keep running during class.

## Day by day

| Day | A · Framework | B · Literature, data, audit | C · Compute | D · Baselines | E · Neural models | F · Interpretation, demo |
|---|---|---|---|---|---|---|
| **Thu 12** evening | Skeleton in `main`. Freeze R1–R8. Assign E and F. Review pull requests. | Count the items with `load_split`. Read the protocol sections of the two baseline papers. | Build the environment from `requirements.txt`. Prepare the data on the GPU machine. Run the majority baseline there. | First run. The two other floors. TF-IDF with Logistic Regression on all three datasets. | First run. Output heads. DistilBERT smoke run on a small subset. | First run. Calibrate page on the stub Predictor. |
| **Fri 13** | Reviews and merges through the day. Table builder on the first rows. 15:00 late check. 23:59 results freeze. | 12:00 template shared. Dataset cards. Baseline table and references. | 10:00 model weights cached. 12:00 smoke job. 13:00 launch the grid. Monitor, retry, return rows. | 12:00 LLM script to C. 15:00 tuned classical table and R2DE. 20:00 feature model. | 12:00 trainer complete, grid configs to C. 15:00 launch Optuna on ARC. 20:00 BiLSTM rows. | 18:00 LIME pipeline and figure functions on the first checkpoints and rows. 20:00 Calibrate page v0. Transfer and extrapolation scripts. |
| **Sat 14** | 12:00 all tables rebuilt, pass or fail for R1–R8. 22:00 README and driver notebook. | 10:00 storyboard v1. 15:00 audit done. 18:00 slides proofread. | 10:00 all results returned. 12:00 efficiency table. | 10:00 global SHAP. Slides, recording. | 10:00 final models behind the Predictor API. Slides, recording. | 10:00 worst-error list to B. 12:00 final figures. 14:00 app on the real model. Slides, recording of the demo. |
| **Sun 15** | 11:00 joined video. 14:00 code freeze. Upload by 15:00, latest 18:00. Check that the attempt shows every file. | Reflection document signed by 14:00. Submission checklist with the lead. Peer-evaluation reminder. | 10:00 clean-environment run and reproducibility notes. Peer evaluation. | Peer evaluation. | Peer evaluation. | Peer evaluation. |

Saturday for everyone: slide draft at 10:00, final slides at 16:00, dry run at 16:30, own recording by 23:59. Sunday for everyone: reflection lines by 10:00, full-video review at 12:00, sign-off by 14:00.

## What was cut

| Share | Cut | Made smaller |
|---|---|---|
| All | Every Tier 2 task | |
| B | | B1: no search for near-duplicates, and 10 items read per level. B3: 15 worst errors, 5 per dataset. |
| C | Checkpoint and resume in the job templates | C2: the datasets come from `python -m levellens.data.prepare`, so only the model weights are cached by hand. C4: a smaller grid. |
| D | Ordinal logistic regression. The features that need a parser: part-of-speech ratios, dependency depth, clause count. | D1: the majority floor is already in `levellens/models/floors.py`. |
| E | E3, the LaTeX-aware tokeniser. E4, the BiLSTM grid. The BiGRU flag. Checkpoint and resume. | E2: one BiLSTM configuration. E7: Optuna on ARC only, with DistilBERT. |
| F | F5, the physical-meaning plot. SHAP on the transformer. ARC Easy against Challenge. The feature gauges in the app. | F1: LIME on 20 items per dataset. F2: confusion matrices, recall per level and one leaderboard plot. F7: leaderboards and confusion matrices only. |

If a share is still late at the check on Fri 13 Nov at 15:00, the lead cuts in this order: Optuna, the open LLM, DeBERTa-v3 (the grid then uses DistilBERT only), the BiLSTM.

## Rules for the schedule

- A date in this file changes only when the lead changes it here. The lead also posts the change in the group chat.
- Behind by more than half a day? Tell the lead the same day. Do not wait for the next milestone.
- After the results freeze, we fix mistakes and re-run what is broken. We do not start new experiments.
