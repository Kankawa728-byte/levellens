# Schedule

All times are Singapore time. The hard deadline is **Sun 15 Nov 2026, 23:59**.

## Milestones

| When | What | Who |
|---|---|---|
| Tue 10 Nov 23:59 | The three open shares are claimed. The lead assigns any share still open on Wed 11 Nov at 09:00. | D, E, F |
| Wed 11 Nov 12:00 | Skeleton v0 in `main`: schema, RACE++ and MATH adapters, metrics, results store, stub Predictor | A |
| Wed 11 Nov 18:00 | ARC and ScienceQA adapters, saved splits, 1,000-item subsets. The science dataset is decided. | A |
| Wed 11 Nov 20:00 | The smoke job passes on the GPU | C |
| Wed 11 Nov 23:59 | Every model family has written one result row | D, E, F |
| Thu 12 Nov 12:00 | Specifications R1–R8 frozen. Tier 1 grid configs handed to C. Slide template shared. | A, E, B |
| Thu 12 Nov 21:00 | Go or no-go for Tier 2 | A |
| Fri 13 Nov 12:00 | **Results freeze.** No new experiments. Slide drafts exist. | everyone |
| Fri 13 Nov 23:59 | All tables and figures final. Efficiency table done. | A, C, F |
| Sat 14 Nov 10:00 | Final slides. Timed dry run online at 10:30. | everyone |
| Sat 14 Nov 16:00 | Recordings with the lead | everyone |
| Sat 14 Nov 20:00 | Full-video review | everyone |
| Sat 14 Nov 22:00 | Sign-off: video final, reflection signed, code frozen | everyone |
| Sun 15 Nov 12:00 | Upload, latest 18:00. Only the lead submits. | A |
| Sun 15 Nov 23:59 | Hard deadline. Nobody creates a new attempt after this. | |

Friday evening is class time. Plan your Friday work for the morning and the afternoon.

## Day by day

| Day | A · Framework | B · Literature, data, audit | C · Compute | D · Baselines | E · Neural models | F · Interpretation, demo |
|---|---|---|---|---|---|---|
| **Tue 10** evening | Publish this repository. Start the skeleton. | Read the protocol sections of the two baseline papers. | Check GPU access and free disk. Start the environment and the downloads. | Claim. Read the design and interface pages. Set up Python. | Claim. Read the design and interface pages. Set up Python. | Claim. Read the design and interface pages. Set up Python. |
| **Wed 11** | 12:00 skeleton v0. 18:00 ARC and ScienceQA adapters, splits, subsets. Majority-row test. | Data counts as the adapters arrive. Slide template. | Environment, offline cache, job templates. 20:00 smoke job. | Floors, TF-IDF with Logistic Regression on all three datasets. Start the feature extractor. | Heads. DistilBERT smoke run. BiLSTM trains on MATH. LaTeX tokeniser. | Calibrate page on the stub. LIME and SHAP pipeline on a small model. |
| **Thu 12** | 12:00 freeze R1–R8. Table builder. Reviews. 21:00 Tier 2 decision. | 12:00 template shared. Dataset cards. Baseline table and references. Storyboard v0. | 14:00 launch the Tier 1 grid. Monitor, retry, return rows. | Tuned classical table. R2DE. Feature model. 18:00 LLM script to C. | 12:00 grid configs to C. BiLSTM grid. 20:00 launch Optuna. | Calibrate page v0. Figure functions on the first rows. Transfer and extrapolation scripts. |
| **Fri 13** | 12:00 results freeze. Rebuild all tables. Pass or fail for R1–R8. | Storyboard v1. Audit of the worst errors after 15:00. | 14:00 all results returned. Efficiency table. | Global SHAP. LLM rows. Slide draft. | 15:00 final models behind the Predictor API. Slide draft. | 15:00 worst-error list to B. Final explanations, transfer, extrapolation, figures. App on the real model. |
| **Sat 14** | README, driver notebook. 20:00 assembled video. 22:00 code freeze. | 09:00 audit done. 12:00 slides proofread. Reflection document signed by 22:00. | 18:00 clean-environment run and reproducibility notes. | Slides, recording. | Slides, recording. | Slides, recording of the demo. |
| **Sun 15** | Upload by 12:00, latest 18:00. Check that the attempt shows every file. | Submission checklist with the lead. Peer-evaluation reminder. | Peer evaluation. | Peer evaluation. | Peer evaluation. | Peer evaluation. |

Saturday for everyone: final slides at 10:00, dry run at 10:30, own recording by 16:00, full-video review at 20:00, sign-off at 22:00.

## Rules for the schedule

- A date in this file changes only when the lead changes it here. The lead also posts the change in the group chat.
- Behind by more than half a day? Tell the lead the same day. Do not wait for the next milestone.
- After the results freeze, we fix mistakes and re-run what is broken. We do not start new experiments.
