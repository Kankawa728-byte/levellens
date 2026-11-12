# Design

中文版：[design.zh-CN.md](design.zh-CN.md)

All times are Singapore time. The tiers (Tier 1, Tier 2) are explained in [work/README.md](../work/README.md).

## Problem and users

**Users.** Secondary and pre-university teachers, tutors and question-bank editors who write new assessment items.

**Problem.** Item difficulty is usually set by pretesting on students, which is slow, costly and exposes the items, or by expert judgement, which is inconsistent. The two most complete recent comparisons on our datasets, Benedetto (2023) and Thuy et al. (2025), cover classical models and BERT-family models only. They include no recurrent models and no competition maths, and neither tests whether a difficulty signal transfers between subjects.

**Wording rule.** We estimate the *target level* of a question (exam stage, school grade, competition level). We do not estimate difficulty measured on students. We say this in the first minute of the video and on the limitations slide.

## Research questions

1. How do classical, recurrent, transformer and open-LLM models compare at estimating question level from text in three subjects?
2. Do ordinal output heads, Bayesian tuning and stability fixes improve the best model?
3. Does the learned difficulty signal transfer across subjects, and to unseen datasets and grades?
4. What makes a question hard in each subject?

## Datasets

Sizes come from the dataset pages, checked on 6 Nov, and for the filtered ARC split from the two published papers. Two counts were taken when the data were first loaded on Thu 12 Nov: ScienceQA has 6,508 / 2,144 / 2,224 text-only items, and 2 MATH training items are labelled "Level ?", which we drop.

| Dataset | Subject | Train / dev / test | Levels | Source | Licence | Role |
|---|---|---|---|---|---|---|
| **RACE++** (RACE middle + high, RACE-C) | English reading | 100,568 / 5,599 / 5,642 | 3: middle school, high school, college | `ehovy/race` + `tasksource/race-c` | RACE: non-commercial research. RACE-C: no licence stated. | trained |
| **ARC with school grade** | School science | 3,358 / 862 / 3,530 | 7: grades 3–9, very imbalanced | AI2 `ARC-V1-Feb2018.zip` (681 MB, of which we read only the six question files, 2.3 MB; the Hugging Face copy has no grade field) | CC BY-SA 4.0 | trained |
| **MATH** | Competition maths | 6,748 / 750 / 5,000 | 5: Level 1–5 | `EleutherAI/hendrycks_math`; 10% of train held out as dev | MIT | trained |
| **ScienceQA, text only** | Science, social, language | 6,508 / 2,144 / 2,224, from 12,726 / 4,241 / 4,241 | Grades 1–12 | `derek-thomas/ScienceQA` | CC BY-NC-SA 4.0 (original release; the mirror is tagged CC BY-SA 4.0) | extrapolation |
| **GSM8K** (cancelled with Tier 2) | Grade-school maths | 7,473 / 1,319 | None; number of solution steps as a proxy | `openai/gsm8k` | MIT | extrapolation |

- RACE++ levels come from the config name, because the `all` config has no level column. The splits are passage-disjoint.
- ARC keeps 4-option, text-only multiple-choice questions, as the published papers do. Main runs use all 3,358 training items with a class-weighted loss. The papers' balanced subset (2,293 items) is used only for replication runs and for the majority-row test.
- MATH input is the problem text only. Solutions are used for interpretation, never as input.
- The model input is the question plus all options. On RACE++ the public code of both published papers also puts the reading passage in front of the question, and our input does not. The check is in [related_work.md](related_work.md).
- We ship download scripts, not data. RACE and ScienceQA are for non-commercial use only.

**If ARC fails the first check.** If the ARC grade labels cannot be read, or the filtered ARC training split has 3,000 items or fewer, we use ScienceQA text-only as the trained science set, provided it has more than 3,000 training items. If the counts only differ from 3,358 / 862 / 3,530, we align our filter with the data preparation of the qde-ordinality repository and keep ARC. The lead decides on Wed 11 Nov after the check. Result of the check on Thu 12 Nov: the grade labels are read, our filter gives exactly 3,358 / 862 / 3,530 items and 2,293 in the balanced subset, and the majority baseline on our splits reproduces the published rows for RACE++ and ARC.

## Models

| Family | Models | Runs on | Share |
|---|---|---|---|
| Floors | Majority class, stratified random, subject prior (most common level per MATH problem type) | CPU | D |
| Classical | TF-IDF word 1–2-grams and character 3–5-grams with Logistic Regression, Linear SVM, Complement Naive Bayes. R2DE, the TF-IDF + Random Forest difficulty regressor used as a baseline by Benedetto (2023). Readability, word rarity, length, numbers, negations and LaTeX operators fed to LightGBM. | CPU | D |
| Recurrent | Two-layer BiLSTM with GloVe 300d and attention pooling | GPU | E |
| Transformers | `microsoft/deberta-v3-base` (main), `distilbert/distilbert-base-uncased` (fast, used in the demo). | GPU | E |
| Open LLM | `Qwen/Qwen3-4B-Instruct-2507`, zero-shot, scored from the probabilities of the level labels. No paid API. | GPU | D |

**Output heads.** The levels are ordered, so predicting "high school" for a college question is a smaller error than predicting "middle school". We compare two heads on the same backbone: plain cross-entropy, and CORN (conditional ordinal regression).

## Published baselines

Both papers use the same RACE++ splits and the same filtered ARC test split (3,530 questions) as we do. Their RACE++ input includes the reading passage and ours does not, so the RACE++ column is a published reference and not a comparison on the same input. For ARC they trained on a partially balanced subset (the two most common grades cut to 500 each, 2,293 items), so the ARC column compares directly only with our balanced replication runs. Balanced DRPS is explained under Metrics. The numbers were checked against the arXiv versions on 6 Nov.

| Paper | RACE++ | ARC (balanced training set) |
|---|---|---|
| Benedetto (2023), arXiv:2305.10236 | BERT: RMSE 0.372 ± 0.012, Spearman 0.789 ± 0.014. DistilBERT: Spearman 0.790 ± 0.007. Best classical hybrid: Spearman 0.668 ± 0.004. | BERT: RMSE 1.535 ± 0.148, Spearman 0.502 ± 0.139. Best classical hybrid: Spearman 0.400 ± 0.006. |
| Thuy, Loginova and Benoit (2025), EvalLAC'25, arXiv:2507.00736 | OrderedLogitNN: accuracy 0.861 ± 0.003, RMSE 0.384 ± 0.005, Balanced DRPS 0.130 ± 0.001. CORN: Balanced DRPS 0.127 ± 0.003. Plain classification: accuracy 0.847 ± 0.004. | Classification accuracy 0.421 ± 0.005. OrderedLogitNN Balanced DRPS 0.674 ± 0.004. Regression RMSE 1.412 ± 0.013. |
| Majority baseline (same paper) | Accuracy 0.620, RMSE 0.616, Balanced DRPS 0.667. | Accuracy 0.409, RMSE 2.195, Balanced DRPS 2.286. |

The majority rows double as a unit test: our splits and metric code must reproduce them (for ARC, with the balanced training subset). We know of no published question-difficulty numbers for the MATH levels, so our classical models are the reference there.

## Metrics

- Accuracy, macro-F1 with per-class precision and recall, and adjacent accuracy (within one level).
- Quadratic weighted kappa (QWK): agreement that punishes far misses more than near misses.
- MAE and RMSE on the level index, and the Spearman rank correlation between the true level and the predicted level. The published numbers are computed the same way: the code of Benedetto (2023) maps each prediction to a level before the correlation. We also store the correlation with the continuous score as `spearman_score`.
- Balanced DRPS (discrete ranked probability score): an ordinal probability error, balanced across levels.
- Expected calibration error (15 bins), training time, CPU and GPU latency per question, parameter count.
- Mean ± standard deviation over 3 seeds where we run 3 seeds, and paired bootstrap 95% confidence intervals for the best model against the runner-up.
- Every system, the LLM included, is also scored on one fixed 1,000-item stratified test subset per dataset.

## Our specifications

Frozen by the lead on Thu 12 Nov. The video reports pass or fail for each one. Spearman is the `spearman` metric: the true level against the predicted level.

| Id | Specification |
|---|---|
| R1 | Three trained datasets with more than 3,000 labelled training questions each, plus at least one extrapolation set. |
| R2 | At least four model families (classical, recurrent, transformer, open LLM; the floors come on top) and three output types (classification, regression, ordinal), all on identical splits. |
| R3 | RACE++ Spearman ≥ 0.79 on the full training set (≥ 0.77 if the GPU forces a 20k–30k cap, and we say so). ARC: mean Spearman ≥ 0.50 over 3 seeds, for the setting with the best mean QWK on dev. MATH: the best transformer beats the best classical model on QWK, with the 95% bootstrap interval of the difference above zero. |
| R4 | An ordinal head has lower Balanced DRPS than cross-entropy on at least two of the three datasets (same backbone, same split). |
| R5 | ARC stability: for the same setting as in R3, the standard deviation of Spearman over 3 seeds ≤ 0.05 (published BERT, balanced training set: 0.139). |
| R6 | Every fine-tune ≤ 1 GPU-hour; the whole project ≤ 20 GPU-hours on an Ampere-class card, logged per run. On another card we report the actual hours. |
| R7 | Demo: ≤ 1 second per question on a laptop CPU (DistilBERT, timed over 100 items). |
| R8 | Every run starts from one config, and every table and figure is rebuilt from the results store with one command. |

Note on R3, added on Fri 13 Nov: the RACE++ threshold of 0.79 is the published Spearman, and the published input includes the reading passage. Our input does not. The threshold stays as frozen, and the video states this difference next to the result.

## Scope

The lead cut the scope on Thu 12 Nov, because the work started late. The dates are in [schedule.md](schedule.md).

**Tier 1, must ship**

- Floors; TF-IDF models tuned by 5-fold search; R2DE; the feature model with LightGBM on surface features; global SHAP.
- BiLSTM with cross-entropy and CORN heads, one configuration.
- DeBERTa-v3 and DistilBERT with both heads on all three datasets, one seed.
- ARC stability: full training set, class weights, early stopping on dev QWK, 3 seeds.
- Optuna (TPE with a median pruner), 12 trials, on ARC with DistilBERT.
- Open LLM, zero-shot, on the 1,000-item subsets.
- LIME on the final DistilBERT, 20 test items per dataset; error figures; human audit of the 15 worst errors; transfer matrix; ScienceQA extrapolation.
- Streamlit app: the Calibrate page, and a Results page with the leaderboards and the confusion matrices.
- Efficiency table.

**Cut from Tier 1 on Thu 12 Nov**

- Ordinal logistic regression, and the features that need a parser: part-of-speech ratios, dependency depth, clause count.
- The BiLSTM grid, the BiGRU flag and the LaTeX-aware tokeniser.
- Optuna on RACE++, on MATH and on DeBERTa-v3.
- SHAP on the transformer; ARC Easy against Challenge; the physical-meaning plot; the feature gauges in the app.
- Checkpoint and resume: every run is short enough to start again.

**Tier 2, cancelled on Thu 12 Nov**

- Regression and ordered-logit heads in the grid, so that four heads are compared.
- BERT replication runs next to the published numbers; RoBERTa with the best head.
- 3 seeds for the headline settings on RACE++ and MATH.
- BiLSTM Optuna sweep; open LLM few-shot.
- Hybrid Random Forest; phrase-level TF-IDF; GloVe average with Logistic Regression.
- GSM8K extrapolation; the "Audit a quiz" page.
- Automated tests on every push; the guide "add your own dataset or model".

**Not planned.** Only if everything above is done: input-format ablation, learning curves, layer-wise learning-rate decay on ARC, fusing the DeBERTa sentence vector with the features, one joint model across subjects, ModernBERT with the RACE++ passage, a graph model over dependency parses, a page that suggests rewrites.

## Interpretation and extrapolation

- **Global.** SHAP on the feature LightGBM, per subject: how much difficulty comes from language load (readability, rare words, length) and how much from reasoning load (numbers, negation, LaTeX operators). Share D.
- **Local.** LIME (explains one prediction by removing words and watching the output change) on the fine-tuned DistilBERT for 20 test items per dataset, summarised as the top cue words per level. The app shows the same view. Share F.
- **Errors.** Ordinal confusion matrices and recall per level (share F), and a human audit of the 15 worst errors to separate model errors from label noise (share B).
- **Transfer.** Each subject's model scores the other subjects' test sets (a 3 × 3 matrix, Spearman). Share F.
- **Extrapolation.** The ARC model on ScienceQA grades inside and outside 3–9. Share F.
- **Physical meaning.** The ordinal head's score is on a logit scale, like the item difficulty *b* in the Rasch model, where the chance of a correct answer is sigmoid(ability − *b*). The cut-points mark where one level ends and the next begins. A true *b* needs student responses to anchor it; we say so.
- **Beyond this application.** School question banks, MOOC quizzes, certification items, reading levels of passages, ordering items for adaptive practice, and other languages through a multilingual encoder. Each needs the same anchoring caveat.

## Streamlit app

- **Calibrate.** A teacher pastes a question, its options and, for reading, the passage. The page shows the probability of each level, the expected level, a "why is it hard" panel with highlighted words, and a what-if editor that re-scores an edited copy side by side.
- **Results.** Leaderboards and confusion matrices, read from the results store.
- **Audit a quiz** (cancelled with Tier 2). Upload a CSV or paste about ten questions. Every item gets a level, the quiz profile is drawn against a blueprint the teacher sets (for example 30 / 50 / 20), mismatched items are flagged, and the tagged CSV can be exported.

## Compute

- GPU runs go through share C. The fallback is the lead's workstation with an RTX 5090 (32 GB). The code picks the device from the config.
- Compute nodes may have no internet, so data and model weights are staged in a cache first.
- Each job stays under 1 hour.
- Estimate for the whole grid: about 12–14 GPU-hours on an Ampere-class 16 GB+ card.

## Risks

| Risk | What we do |
|---|---|
| ARC grade labels cannot be read, or too few items remain. | See "If ARC fails the first check" above. |
| GPU access is late or slow. | Run on the RTX 5090; cap RACE++ at 20k–30k; stage the cache offline. |
| DeBERTa-v3 gives NaN losses in fp16. | Use bf16 or fp32, or RoBERTa as the main backbone. |
| We run out of time. | Done on Thu 12 Nov: Tier 2 cancelled, Tier 1 cut. If a share is still late on Fri 13 Nov at 15:00, the lead cuts in this order: Optuna, the open LLM, DeBERTa-v3 (the grid then uses DistilBERT only), the BiLSTM. No new experiments after the results freeze. |
| A member is unavailable. | The buddy covers that day; the lead re-plans in the group chat. |
| The video runs over 10 minutes. | Fixed time per part; timed dry run on Sat 14 Nov; cut content, not speakers. |
| Late upload. | Upload on Sun 15 Nov by 12:00, latest 18:00; only the lead submits; nobody creates an attempt after 23:59. |
| The labels are target levels, not difficulty measured on students. | Say "target level" throughout; show label-noise cases from the audit; present anchoring to student responses as future work. |
| LLM contamination on MATH and GSM8K. | State it, and score on the fixed subsets. |
| Licences. | RACE and ScienceQA are for non-commercial use only; RACE-C states no licence. We cite every source and ship download scripts, not data. |

## References

Confirm each venue before it goes on a slide (task B2).

- L. Benedetto. A quantitative study of NLP approaches to question difficulty estimation. 2023. arXiv:2305.10236. Code: github.com/lucabenedetto/qdet-comparison.
- A. Thuy, E. Loginova, D. F. Benoit. Ordinality in discrete-level question difficulty estimation: introducing Balanced DRPS and OrderedLogitNN. EvalLAC'25 workshop, 2025. arXiv:2507.00736. Code: github.com/arthur-thuy/qde-ordinality (MIT).
- G. Lai et al. RACE: large-scale reading comprehension dataset from examinations. EMNLP 2017. Y. Liang et al. A new multi-choice reading comprehension dataset for curriculum learning (RACE-C). ACML 2019.
- P. Clark et al. Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge. 2018.
- D. Hendrycks et al. Measuring mathematical problem solving with the MATH dataset. NeurIPS 2021 Datasets and Benchmarks.
- P. Lu et al. Learn to explain: multimodal reasoning via thought chains for science question answering (ScienceQA). NeurIPS 2022.
- K. Cobbe et al. Training verifiers to solve math word problems (GSM8K). 2021.
- X. Shi, W. Cao, S. Raschka. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN). 2023.
- P. He, J. Gao, W. Chen. DeBERTaV3. ICLR 2023.
- S. Supraja, A. W. H. Khong, S. Tatinati. Regularized phrase-based topic model for automatic question classification with domain-agnostic class labels. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3604–3616, 2021.
- S. M. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions (SHAP). NeurIPS 2017. M. T. Ribeiro, S. Singh, C. Guestrin. "Why should I trust you?" (LIME). KDD 2016.
