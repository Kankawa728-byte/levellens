# Share D · Baselines: classical, feature models, open LLM

中文版：[D-baselines.zh-CN.md](D-baselines.zh-CN.md)

**Owner:** Hailing · **Buddy:** B · **Size:** 24 h Tier 1 + 4 h Tier 2

Everything that is not a fine-tuned neural network: the floors, the classical models, the feature-based models, and the open LLM used without training. These numbers show how much the neural models really add. The feature model also gives the first answer to "what makes a question hard".

Most of this share runs on a laptop CPU. It suits someone who is comfortable with scikit-learn.

## Files you own

- `levellens/models/classic.py`, `levellens/models/llm.py`
- `levellens/features/tfidf.py`, `levellens/features/linguistic.py`
- `levellens/explain/global_shap.py`
- Your model configs in `configs/model/`, and the tests for your modules

## Tier 1

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| D1 | Floors: majority class, stratified random, subject prior (the most common level for each MATH problem type) | Rows for all three datasets are in `results/runs.jsonl`. | 1 | Wed 11 Nov 23:59 |
| D2 | TF-IDF with word 1–2-grams and character 3–5-grams, with Logistic Regression, Linear SVM and Complement Naive Bayes. Settings chosen by 5-fold search on the training split. | First rows (TF-IDF with Logistic Regression, default settings) on all three datasets on Wed 11 Nov. The tuned table for all three models on Thu 12 Nov. | 5 | First rows: Wed 11 Nov 23:59. Tuned: Thu 12 Nov 20:00 |
| D3 | R2DE: the TF-IDF and Random Forest difficulty regressor used as a baseline by Benedetto (2023) | Rows for all three datasets. | 2 | Thu 12 Nov 20:00 |
| D4 | Linguistic feature extractor: readability, word rarity, length, part-of-speech ratios, dependency depth, clause count, numbers, negations, LaTeX operators. The features go to LightGBM and to an ordinal logistic regression. | The features are cached on disk. Rows for both models on all three datasets. | 6.5 | Thu 12 Nov 23:59 |
| D5 | Global SHAP on the LightGBM model, for each subject: how much comes from language load (readability, rare words, length) and how much from reasoning load (dependency depth, clauses, numbers, negation, LaTeX operators) | One figure per subject, and three sentences on what each one shows. | 2 | Fri 13 Nov 12:00 |
| D6 | Open LLM, zero-shot: `Qwen/Qwen3-4B-Instruct-2507` on the fixed 1,000-item test subsets. The level is read from the probabilities of the level labels. The prompt is tuned on dev only. No paid API. | The scoring script is with C on Thu 12 Nov at 18:00. Rows for all three datasets at the results freeze. | 4.5 | Script: Thu 12 Nov 18:00. Rows: Fri 13 Nov 12:00 |

Suggested libraries: scikit-learn, LightGBM, spaCy, textstat, wordfreq, statsmodels or mord for the ordinal regression, shap, transformers. You decide.

## Tier 2

| Id | Task | Done when | Hours |
|---|---|---|---|
| D7 | Open LLM, few-shot, with examples taken from the training split | Rows for all three datasets. | 1.5 |
| D8 | Hybrid Random Forest on TF-IDF and linguistic features together, following the hybrid classical baseline of Benedetto (2023) | Rows for all three datasets. | 1 |
| D9 | Phrase-level TF-IDF; GloVe average with Logistic Regression | Rows for all three datasets. | 1.5 |

## Others wait for you

- C: the open LLM scoring script on Thu 12 Nov at 18:00.
- F: the feature extractor, for the feature gauges on the Calibrate page.
- A: your rows for the tables, at the results freeze.

## You wait for

- A: skeleton v0 on Wed 11 Nov at 12:00, the ARC adapter and the 1,000-item subsets at 18:00. Until then, code against [docs/interfaces.md](../docs/interfaces.md).

## Video part

2:20–3:40: the floors, the classical and feature models with a cross-validation run on screen, what the global SHAP says about each subject, and the open LLM without training.
