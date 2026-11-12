# Share F · Interpretation, extrapolation and demo app

中文版：[F-interpretation-demo.zh-CN.md](F-interpretation-demo.zh-CN.md)

**Owner:** open · **Buddy:** A · **Size:** 24 h Tier 1 + 4 h Tier 2

What the models learned and how a teacher would use them: explanations of single predictions, tests beyond the training data, the result figures, and the Streamlit app.

It suits someone who likes plots and front ends. Build every pipeline early on the first checkpoints, so that after the results freeze you only run it again.

## Files you own

- `levellens/explain/local.py`, `levellens/explain/errors.py`
- `levellens/figures.py`
- `app/streamlit_app.py` and `app/pages/`
- The tests for your modules

## Tier 1

Re-plan of Thu 12 Nov: the dates are new, and a task marked "Cut" is not done. The hours are the first estimates. The whole re-plan is in [docs/schedule.md](../docs/schedule.md).

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| F1 | LIME on the fine-tuned DistilBERT, for 20 test items per dataset, summarised as the top cue words for each level | The pipeline runs on the DistilBERT smoke checkpoint on Fri 13 Nov. The final tables and figures use the final models. | 4.5 | Pipeline: Fri 13 Nov 18:00. Final: Sat 14 Nov 12:00 |
| F2 | Error and results figures: ordinal confusion matrices, recall per level and one leaderboard plot. The list of the 15 worst errors for the audit, 5 per dataset: largest gap between predicted and labelled level first, then highest confidence. | The functions run on the first rows on Fri 13 Nov. B has the worst-error list on Sat 14 Nov at 10:00. | 3 | Functions: Fri 13 Nov 18:00. List: Sat 14 Nov 10:00. Final: Sat 14 Nov 12:00 |
| F3 | Transfer matrix: each subject's model scores the test sets of the other subjects (3 × 3, Spearman) | One heatmap and three sentences on what it shows. | 1.5 | Sat 14 Nov 12:00 |
| F4 | Extrapolation: the ARC model on ScienceQA, for grades inside and outside 3–9 | One figure and three sentences on what it shows. | 2 | Sat 14 Nov 12:00 |
| F5 | Cut: physical-meaning plot: the score of the ordinal head and its cut-points on the logit scale, read like the item difficulty in the Rasch model. A true item difficulty needs student responses; the caption says so. | One figure with its caption. | 1.5 | Cut on Thu 12 Nov |
| F6 | Streamlit Calibrate page: a teacher pastes a question, its options and, for reading, the passage. The page shows the probability of each level, the expected level, a "why is it hard" panel with highlighted words, and a what-if editor that scores an edited copy side by side. | The page runs on the stub Predictor on Thu 12 Nov, is complete on Fri 13 Nov, and loads the real DistilBERT model on Sat 14 Nov. | 6.5 | Stub: Thu 12 Nov 23:59. v0: Fri 13 Nov 20:00. Real model: Sat 14 Nov 14:00 |
| F7 | Streamlit Results page: leaderboards and confusion matrices, read from the results store | The page shows the final tables and figures. | 2 | Sat 14 Nov 14:00 |

Suggested libraries: lime, matplotlib, streamlit. You decide.

## Tier 2

Cancelled on Thu 12 Nov. None of these tasks is done.

| Id | Task | Done when | Hours |
|---|---|---|---|
| F8 | GSM8K extrapolation: the score of the MATH model against the number of solution steps | One figure and two sentences on what it shows. | 1.5 |
| F9 | Streamlit "Audit a quiz" page: upload a CSV or paste about ten questions; every question gets a level; the quiz profile is drawn against a blueprint that the teacher sets, for example 30 / 50 / 20; mismatched questions are flagged; the tagged CSV can be exported | The page works on a ten-question example in the demo. | 2.5 |

## Others wait for you

- B: the list of the 15 worst errors on Sat 14 Nov at 10:00.
- A: the final figures for the README and the driver notebook.
- Everyone: figures for their slides. Post each figure in the group chat when it is ready.

## You wait for

- A: the skeleton with the stub Predictor, and the ScienceQA adapter. They are in `main` since Thu 12 Nov.
- E: the DistilBERT smoke checkpoint on Fri 13 Nov at 12:00; the final models behind the Predictor API on Sat 14 Nov at 10:00.
- C: prediction files and checkpoints as the runs finish.

## Video part

6:30–8:50: the cue words from LIME, transfer and extrapolation, then the live demo: score a question, edit it and score it again.
