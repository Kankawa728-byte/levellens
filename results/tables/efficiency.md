# Efficiency

Built from `runs.jsonl`, `export/` and `latency/` by `python scripts/efficiency.py`. Do not edit by hand.

## Specifications

- **R6 pass**: 28 fine-tunes. The longest is `math-bilstm-ce-s0` at 0.19 hours, against a limit of 1 hour per run.
- **R6, whole project**: 0.89 GPU-hours, against a limit of 20. Plus 0.45 CPU-hours and 0.41 hours on another accelerator, which the limit does not count.
- **R7 pass**: 0.028 to 0.033 seconds per question over 4 saved models on the CPU (distilbert, one question at a time), against a limit of 1 second. The slowest measurement decides.

## One line per model

| Model | Family | Runs | Parameters | Ran on | Mean train s | Longest train s | Hours, longest run | Hours, all runs | s/question CPU | s/question GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| r2de | classical | 3 | 0 | CPU | 1.5 | 4.1 | 0.001 | 0.001 |  |  |
| tfidf_cnb | classical | 3 | 0 | CPU | 38.9 | 72.2 | 0.021 | 0.034 |  |  |
| tfidf_logreg | classical | 6 | 0 | CPU | 152.6 | 408.5 | 0.114 | 0.259 |  |  |
| tfidf_svm | classical | 3 | 0 | CPU | 154.7 | 320.0 | 0.089 | 0.131 |  |  |
| linguistic_lightgbm | feature | 3 | 0 | CPU | 3.1 | 5.9 | 0.002 | 0.003 |  |  |
| majority | floor | 3 | 0 | CPU | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| stratified_random | floor | 3 | 0 | CPU | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| subject_prior | floor | 1 | 0 | CPU | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| qwen_zero_shot | open LLM | 3 | 4,022,468,096 | GPU | 37.7 | 45.6 | 0.034 | 0.085 |  |  |
| bilstm | recurrent | 6 | 14,687,868 | Apple GPU, GPU | 265.3 | 669.1 | 0.187 | 0.446 |  |  |
| deberta-v3 | transformer | 10 | 183,836,935 | GPU | 199.5 | 621.5 | 0.173 | 0.559 | 0.1411 | 0.0157 |
| distilbert | transformer | 12 | 66,368,263 | CPU, GPU | 65.9 | 171.3 | 0.048 | 0.226 | 0.0331 | 0.0034 |

## One line per run

R6 is checked per run, so every run is listed. The column R6 is empty for a model that trains no weights.

| Run | Data | Model | Head | Family | Seed | Device | Kind | Parameters | Train s | Score s | Hours | R6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arc-distilbert-ce-smoke-n400-s0 | arc | distilbert | ce | transformer | 0 | cpu | CPU | 66,368,263 | 51.1 | 15.5 | 0.0185 | pass |
| math-bilstm-ce-s0 | math | bilstm | ce | recurrent | 0 | mps | Apple GPU | 4,762,194 | 669.1 | 4.3 | 0.1871 | pass |
| math-bilstm-corn-s0 | math | bilstm | corn | recurrent | 0 | mps | Apple GPU | 4,761,681 | 492.7 | 4.4 | 0.1381 | pass |
| arc-bilstm-ce-s0 | arc | bilstm | ce | recurrent | 0 | mps | Apple GPU | 5,195,520 | 177.0 | 2.7 | 0.0499 | pass |
| arc-bilstm-corn-s0 | arc | bilstm | corn | recurrent | 0 | mps | Apple GPU | 5,195,007 | 124.8 | 2.5 | 0.0354 | pass |
| arc-tfidf_logreg-none-s0 | arc | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 9.0 | 2.3 | 0.0031 |  |
| math-tfidf_logreg-none-s0 | math | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 16.2 | 2.6 | 0.0052 |  |
| race-tfidf_logreg-none-s0 | race | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 148.4 | 5.3 | 0.0427 |  |
| arc-tfidf_logreg-none-cv5-s0 | arc | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 120.0 | 2.1 | 0.0339 |  |
| arc-tfidf_svm-none-cv5-s0 | arc | tfidf_svm | none | classical | 0 | cpu | CPU | 0 | 84.3 | 2.6 | 0.0241 |  |
| arc-tfidf_cnb-none-cv5-s0 | arc | tfidf_cnb | none | classical | 0 | cpu | CPU | 0 | 32.6 | 2.3 | 0.0097 |  |
| math-tfidf_logreg-none-cv5-s0 | math | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 213.3 | 2.5 | 0.06 |  |
| math-tfidf_svm-none-cv5-s0 | math | tfidf_svm | none | classical | 0 | cpu | CPU | 0 | 59.8 | 2.2 | 0.0172 |  |
| math-tfidf_cnb-none-cv5-s0 | math | tfidf_cnb | none | classical | 0 | cpu | CPU | 0 | 11.8 | 1.1 | 0.0036 |  |
| race-tfidf_logreg-none-cv5-s0 | race | tfidf_logreg | none | classical | 0 | cpu | CPU | 0 | 408.5 | 2.6 | 0.1142 |  |
| race-tfidf_svm-none-cv5-s0 | race | tfidf_svm | none | classical | 0 | cpu | CPU | 0 | 320.0 | 2.2 | 0.0895 |  |
| race-tfidf_cnb-none-cv5-s0 | race | tfidf_cnb | none | classical | 0 | cpu | CPU | 0 | 72.2 | 2.6 | 0.0208 |  |
| arc-r2de-none-s0 | arc | r2de | none | classical | 0 | cpu | CPU | 0 | 0.1 | 0.1 | 0.0001 |  |
| math-r2de-none-s0 | math | r2de | none | classical | 0 | cpu | CPU | 0 | 0.2 | 0.1 | 0.0001 |  |
| race-r2de-none-s0 | race | r2de | none | classical | 0 | cpu | CPU | 0 | 4.1 | 0.4 | 0.0012 |  |
| arc-linguistic_lightgbm-none-s0 | arc | linguistic_lightgbm | none | feature | 0 | cpu | CPU | 0 | 1.8 | 0.3 | 0.0006 |  |
| math-linguistic_lightgbm-none-s0 | math | linguistic_lightgbm | none | feature | 0 | cpu | CPU | 0 | 1.6 | 0.3 | 0.0005 |  |
| race-linguistic_lightgbm-none-s0 | race | linguistic_lightgbm | none | feature | 0 | cpu | CPU | 0 | 5.9 | 0.9 | 0.0019 |  |
| arc-qwen_zero_shot-none-nf4-s0 | arc | qwen_zero_shot | none | open LLM | 0 | cuda (NVIDIA GeForce RTX 4060 Laptop GPU) | GPU | 4,022,468,096 | 28.6 | 67.0 | 0.0265 |  |
| math-qwen_zero_shot-none-nf4-s0 | math | qwen_zero_shot | none | open LLM | 0 | cuda (NVIDIA GeForce RTX 4060 Laptop GPU) | GPU | 4,022,468,096 | 45.6 | 77.4 | 0.0342 |  |
| race-qwen_zero_shot-none-nf4-s0 | race | qwen_zero_shot | none | open LLM | 0 | cuda (NVIDIA GeForce RTX 4060 Laptop GPU) | GPU | 4,022,468,096 | 38.9 | 47.6 | 0.024 |  |
| race-majority-none-s0 | race | majority | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.0 | 0.0 |  |
| race-stratified_random-none-s0 | race | stratified_random | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.2 | 0.0001 |  |
| arc-majority-none-s0 | arc | majority | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.0 | 0.0 |  |
| arc-stratified_random-none-s0 | arc | stratified_random | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.1 | 0.0 |  |
| math-majority-none-s0 | math | majority | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.0 | 0.0 |  |
| math-stratified_random-none-s0 | math | stratified_random | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.1 | 0.0 |  |
| math-subject_prior-none-s0 | math | subject_prior | none | floor | 0 | cpu | CPU | 0 | 0.0 | 0.0 | 0.0 |  |
| race-deberta-v3-ce-s0 | race | deberta-v3 | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,833,859 | 621.5 | 3.0 | 0.1735 | pass |
| race-deberta-v3-corn-s0 | race | deberta-v3 | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,833,090 | 532.8 | 3.0 | 0.1488 | pass |
| race-distilbert-ce-s0 | race | distilbert | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,365,187 | 171.3 | 1.1 | 0.0479 | pass |
| race-distilbert-corn-s0 | race | distilbert | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,364,418 | 166.5 | 1.8 | 0.0467 | pass |
| math-deberta-v3-ce-s0 | math | deberta-v3 | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,835,397 | 191.1 | 2.0 | 0.0536 | pass |
| math-deberta-v3-corn-s0 | math | deberta-v3 | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,834,628 | 187.5 | 2.1 | 0.0526 | pass |
| math-distilbert-ce-s0 | math | distilbert | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,366,725 | 68.4 | 0.9 | 0.0193 | pass |
| math-distilbert-corn-s0 | math | distilbert | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,365,956 | 66.4 | 0.8 | 0.0187 | pass |
| arc-deberta-v3-ce-s0 | arc | deberta-v3 | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,935 | 83.3 | 1.2 | 0.0235 | pass |
| arc-deberta-v3-corn-s0 | arc | deberta-v3 | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,166 | 71.5 | 1.2 | 0.0202 | pass |
| arc-distilbert-ce-s0 | arc | distilbert | ce | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,368,263 | 126.9 | 0.5 | 0.0354 | pass |
| arc-distilbert-corn-s0 | arc | distilbert | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,367,494 | 21.4 | 0.5 | 0.0061 | pass |
| arc-deberta-v3-ce-s1 | arc | deberta-v3 | ce | transformer | 1 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,935 | 103.6 | 1.3 | 0.0291 | pass |
| arc-deberta-v3-corn-s1 | arc | deberta-v3 | corn | transformer | 1 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,166 | 68.2 | 1.2 | 0.0193 | pass |
| arc-distilbert-ce-s1 | arc | distilbert | ce | transformer | 1 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,368,263 | 23.6 | 0.5 | 0.0067 | pass |
| arc-distilbert-corn-s1 | arc | distilbert | corn | transformer | 1 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,367,494 | 27.3 | 0.5 | 0.0077 | pass |
| arc-deberta-v3-ce-s2 | arc | deberta-v3 | ce | transformer | 2 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,935 | 78.0 | 1.3 | 0.022 | pass |
| arc-deberta-v3-corn-s2 | arc | deberta-v3 | corn | transformer | 2 | cuda (NVIDIA GeForce RTX 5090) | GPU | 183,836,166 | 57.8 | 1.2 | 0.0164 | pass |
| arc-distilbert-ce-s2 | arc | distilbert | ce | transformer | 2 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,368,263 | 23.6 | 0.5 | 0.0067 | pass |
| arc-distilbert-corn-s2 | arc | distilbert | corn | transformer | 2 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,367,494 | 34.5 | 0.5 | 0.0097 | pass |
| race-bilstm-ce-s0 | race | bilstm | ce | recurrent | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 14,687,868 | 69.7 | 0.3 | 0.0195 | pass |
| race-bilstm-corn-s0 | race | bilstm | corn | recurrent | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 14,687,355 | 58.5 | 0.3 | 0.0163 | pass |
| arc-distilbert-corn-optuna-s0 | arc | distilbert | corn | transformer | 0 | cuda (NVIDIA GeForce RTX 5090) | GPU | 66,367,494 | 9.4 | 0.5 | 0.0027 | pass |

