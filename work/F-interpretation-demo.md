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

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| F1 | LIME and SHAP on the fine-tuned transformer, for 50 test items per dataset, summarised as the top cue words for each level | The pipeline runs on the DistilBERT smoke checkpoint on Thu 12 Nov. The final tables and figures use the final models. | 4.5 | Pipeline: Thu 12 Nov 23:59. Final: Fri 13 Nov 23:59 |
| F2 | Error and results figures: ordinal confusion matrices, recall per level, breakdowns by MATH problem type, ARC grade and question length, leaderboard plots with intervals. The list of the 30 worst errors for the audit, 10 per dataset: largest gap between predicted and labelled level first, then highest confidence. | The functions run on the first rows on Thu 12 Nov. B has the worst-error list on Fri 13 Nov at 15:00. | 3 | Functions: Thu 12 Nov 23:59. List: Fri 13 Nov 15:00. Final: Fri 13 Nov 23:59 |
| F3 | Transfer matrix: each subject's model scores the test sets of the other subjects (3 × 3, Spearman) | One heatmap and three sentences on what it shows. | 1.5 | Fri 13 Nov 23:59 |
| F4 | Extrapolation: the ARC model on ScienceQA, for grades inside and outside 3–9; ARC Easy against Challenge questions of the same grade | Two figures and three sentences on what they show. | 2 | Fri 13 Nov 23:59 |
| F5 | Physical-meaning plot: the score of the ordinal head and its cut-points on the logit scale, read like the item difficulty in the Rasch model. A true item difficulty needs student responses; the caption says so. | One figure with its caption. | 1.5 | Fri 13 Nov 23:59 |
| F6 | Streamlit Calibrate page: a teacher pastes a question, its options and, for reading, the passage. The page shows the probability of each level, the expected level, a "why is it hard" panel with highlighted words and feature gauges, and a what-if editor that scores an edited copy side by side. | The page runs on the stub Predictor on Wed 11 Nov, is complete on Thu 12 Nov, and loads the real DistilBERT model on Fri 13 Nov. | 6.5 | Stub: Wed 11 Nov 23:59. v0: Thu 12 Nov 20:00. Real model: Fri 13 Nov 22:00 |
| F7 | Streamlit Results page: leaderboards, confusion matrices, the transfer heatmap and the extrapolation plots, all read from the results store | The page shows the final tables and figures. | 2 | Fri 13 Nov 23:59 |

Suggested libraries: shap, lime, matplotlib, streamlit. You decide.

## Tier 2

| Id | Task | Done when | Hours |
|---|---|---|---|
| F8 | GSM8K extrapolation: the score of the MATH model against the number of solution steps | One figure and two sentences on what it shows. | 1.5 |
| F9 | Streamlit "Audit a quiz" page: upload a CSV or paste about ten questions; every question gets a level; the quiz profile is drawn against a blueprint that the teacher sets, for example 30 / 50 / 20; mismatched questions are flagged; the tagged CSV can be exported | The page works on a ten-question example in the demo. | 2.5 |

## Others wait for you

- B: the list of the 30 worst errors on Fri 13 Nov at 15:00.
- A: the final figures for the README and the driver notebook.
- Everyone: figures for their slides. Post each figure in the group chat when it is ready.

## You wait for

- A: skeleton v0 with the stub Predictor on Wed 11 Nov at 12:00, the ScienceQA adapter at 18:00.
- E: the DistilBERT smoke checkpoint on Thu 12 Nov; the final models behind the Predictor API on Fri 13 Nov at 15:00.
- C: prediction files and checkpoints as the runs finish.
- D: the feature extractor for the feature gauges.

## Video part

6:30–8:50: the cue words from LIME and SHAP, transfer and extrapolation, the item-difficulty reading of the score, then the live demo: score a question, edit it and score it again.
