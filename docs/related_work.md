# Published baselines and protocol checks

中文导读：[related_work.zh-CN.md](related_work.zh-CN.md)

Prepared for Share B, Tang Wenyue. Checked on 12 November 2026 against the two **arXiv v1** papers and their public code. The numeric entries below match the rows cited. They are published reference results, not LevelLens measurements.

## Comparison status

**The current project is not an exact replication.** Matching split counts and a majority baseline does not establish equivalence of input text, sampled training IDs, model selection, or preprocessing. Two material differences require disclosure:

1. **RACE++ context:** both authors' code prefixes the reading passage to the question before appending options. LevelLens `format_text` currently uses only the question and lettered options. This is an input-information difference, not just token formatting. Treat the published scores as contextual references until a passage-inclusive replication is authorised by A. Keep the frozen current input unchanged in B's work.
2. **ARC training:** the papers cap grades 5 and 8 at 500 each. LevelLens main runs retain 3,358 training items; its `train_balanced` has the matching 2,293-item class profile. The authors sample with pandas; LevelLens ranks IDs by a deterministic hash. Identical sampled IDs have **not** been established. Even `train_balanced` is a protocol-aligned comparison, not a bit-for-bit reproduction.

Other differences are option separators (`[SEP]` in the authors' transformer input, `A.`, `B.`, etc. here), BERT versus the project's DeBERTa-v3/DistilBERT, and five published runs versus the project's one or three runs. A model architecture change is intentional; the input and training-set differences must remain visible in captions.

## Benedetto 2023

Version: [arXiv:2305.10236v1](https://arxiv.org/abs/2305.10236v1), §§3–5, Tables 1–2 on PDF page 6. RACE++ uses 100,568 train and 5,642 test questions. ARC retains four-option questions, then partially balances training; its test split has 3,530 items. The paper compares question-only, question-with-correct-option and question-with-all-options encodings. The headline transformer rows use all options. Models regress a continuous score, map it to a discrete label, and report RMSE, R² and Spearman correlation. Values are mean and standard deviation across five runs. The article also studies smaller RACE++ training sets; those results are not substituted for the full-data rows.

Code confirms seeds 0–4 in `results_get_evaluation_transformers.py`, discretisation before evaluation, and passage-prefixing in `_get_new_row_text_difficulty_df`. The standard RACE++ dev split is carried by the data manager; the arXiv dataset paragraph does not print its 5,599 count. Do not invent a new validation split or infer unreported tuning details.

| Dataset | Exact row in paper | Metric | Published mean ± SD | Check |
|---|---|---|---|---|
| RACE++ | BERT Q_All | RMSE | 0.372 ± 0.012 | Table 1, match |
| RACE++ | BERT Q_All | Spearman | 0.789 ± 0.014 | Table 1, match |
| RACE++ | DistilBERT Q_All | Spearman | 0.790 ± 0.007 | Table 1, match |
| RACE++ | Ling., Read., TF-IDF Q_Correct | Spearman | 0.668 ± 0.004 | Table 1, match |
| ARC | BERT Q_All | RMSE | 1.535 ± 0.148 | Table 2, match |
| ARC | BERT Q_All | Spearman | 0.502 ± 0.139 | Table 2, match |
| ARC | W2V Q_Correct, Ling., TF-IDF Q_All | Spearman | 0.400 ± 0.006 | Table 2, match |

The two “best classical hybrid” entries use different feature combinations. They must not be presented as one common model configuration.

## Thuy, Loginova and Benoit 2025

Version: [arXiv:2507.00736v1](https://arxiv.org/abs/2507.00736v1), §6 and Tables 2–3 on PDF page 9. RACE++ train/dev/test sizes are 100,568/5,599/5,642. ARC sizes before training downsampling are 3,358/862/3,530; the training subset has 2,293 questions. BERT-base-uncased is fine-tuned with regression, classification or ordinal heads. Input contains the question and all options separated by separator tokens; the implementation also includes RACE++ passages. Results average five runs. The reported metrics are accuracy, RMSE, probability-based Balanced DRPS, and its degenerate-distribution variant.

The public configuration starts at seed 42 and the main loop adds the run index, giving 42–46. Balanced DRPS averages the squared cumulative-probability error within each true class, then across classes. Keep the probabilistic and degenerate columns separate. For hard predictions, the latter behaves as class-balanced absolute level error. Do not divide by K−1 when comparing against these tables.

| Dataset | Output head | Metric | Published mean ± SD | Check |
|---|---|---|---|---|
| RACE++ | OrderedLogitNN | Accuracy | 0.861 ± 0.003 | Table 2, match |
| RACE++ | OrderedLogitNN | RMSE | 0.384 ± 0.005 | Table 2, match |
| RACE++ | OrderedLogitNN | Balanced DRPS | 0.130 ± 0.001 | Table 2, match |
| RACE++ | CORN | Balanced DRPS | 0.127 ± 0.003 | Table 2, match |
| RACE++ | Classification | Accuracy | 0.847 ± 0.004 | Table 2, match |
| ARC | Classification | Accuracy | 0.421 ± 0.005 | Table 3, match |
| ARC | OrderedLogitNN | Balanced DRPS | 0.674 ± 0.004 | Table 3, match |
| ARC | Regression | RMSE | 1.412 ± 0.013 | Table 3, match |
| RACE++ | Majority | Accuracy; RMSE; Balanced DRPS | 0.620; 0.616; 0.667 | Table 2, match; SD 0.000 |
| ARC | Majority | Accuracy; RMSE; Balanced DRPS | 0.409; 2.195; 2.286 | Table 3, match; SD 0.000 |

The ARC majority row selects **grade 8** in the balanced training-set tie, matching the authors' explicit highest-label tie rule and LevelLens `Majority`. Grade 8 is also the full-training-set majority, so this check cannot distinguish full from balanced training. Majority checks validate only the associated label distribution, tie rule and metric arithmetic. Local checks reproduced the published values after rounding to three decimals; see `results/data_checks/majority_check.json`.

## Code evidence and traceability

The inspected commits are pinned below so that future changes in external repositories do not silently change this review.

| Evidence | Primary source |
|---|---|
| Benedetto passage prepending | [`_data_manager.py`, lines 181–183](https://github.com/lucabenedetto/qdet-comparison/blob/e4a5090e227239e59ac61c6a95be567570c085f6/qdet_utils/data_manager/_data_manager.py#L181-L183) |
| Benedetto seeds and discrete evaluation | [`results_get_evaluation_transformers.py`](https://github.com/lucabenedetto/qdet-comparison/blob/e4a5090e227239e59ac61c6a95be567570c085f6/results_get_evaluation_transformers.py) |
| Thuy passage prepending | [`_data_manager.py`, lines 219–249](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/tools/data_manager/_data_manager.py#L219-L249) |
| Thuy option concatenation | [`data_loader.py`, lines 144–180](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/data_loader/data_loader.py#L144-L180) |
| Thuy ARC subset | [`_arc_data_manager.py`, lines 214–258](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/tools/data_manager/_arc_data_manager.py#L214-L258) |
| Thuy seed configuration and loop | [`config.py`](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/config/arc_bert_logit/config.py), [`main.py`](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/main.py) |

## References and confirmed venues

These are publication identities, not claims that every method was implemented in LevelLens. References for cancelled GSM8K work are excluded from the current slide list.

| Reference | Confirmed venue and source | Use |
|---|---|---|
| Benedetto. *A Quantitative Study of NLP Approaches to Question Difficulty Estimation*. 2023. | AIED 2023 poster proceedings, CCIS 1831, pp. 428–434. [Springer](https://link.springer.com/chapter/10.1007/978-3-031-36336-8_67) | Cite **arXiv v1** for the numerical baseline above. The shorter proceedings abstract names two datasets, whereas arXiv includes ARC; do not silently substitute versions. |
| Thuy, Loginova and Benoit. *Ordinality in Discrete-level Question Difficulty Estimation: Introducing Balanced DRPS and OrderedLogitNN*. 2025. | EvalLAC 2025, co-located with AIED 2025, CEUR-WS 4006, paper 4. [Proceedings](https://ceur-ws.org/Vol-4006/), [paper](https://ceur-ws.org/Vol-4006/paper4.pdf) | Ordinal heads and probability evaluation. |
| Lai, Xie, Liu, Yang and Hovy. *RACE: Large-scale ReAding Comprehension Dataset From Examinations*. 2017. | EMNLP 2017, pp. 785–794. [ACL Anthology](https://aclanthology.org/D17-1082/) | RACE provenance. |
| Liang, Li and Yin. *A New Multi-choice Reading Comprehension Dataset for Curriculum Learning*. 2019. | ACML 2019, PMLR 101, pp. 742–757. [PMLR](https://proceedings.mlr.press/v101/liang19a.html) | RACE-C provenance. |
| Clark et al. *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. 2018. | arXiv:1803.05457. [arXiv](https://arxiv.org/abs/1803.05457) | ARC provenance. No conference venue is asserted. |
| Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset*. 2021. | NeurIPS 2021 Datasets and Benchmarks. [Proceedings paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf) | MATH provenance, not a target-level-prediction baseline. |
| Lu et al. *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*. 2022. | NeurIPS 2022. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html) | ScienceQA provenance. |
| Shi, Cao and Raschka. *Deep neural networks for rank-consistent ordinal regression based on conditional probabilities*. 2023. | Pattern Analysis and Applications 26, pp. 941–955. [Publisher](https://link.springer.com/article/10.1007/s10044-023-01181-9) | CORN method. |
| He, Gao and Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. 2023. | ICLR 2023. [Accepted paper](https://openreview.net/pdf?id=sE7-XhLxHA) | Backbone. The preprint first appeared in 2021. |
| Supraja, Khong and Tatinati. *Regularized Phrase-Based Topic Model for Automatic Question Classification With Domain-Agnostic Class Labels*. 2021. | IEEE/ACM TASLP 29, pp. 3604–3616. [DOI](https://doi.org/10.1109/TASLP.2021.3126937), [publisher-deposited metadata](https://api.crossref.org/works/10.1109/TASLP.2021.3126937) | Related classification work; not a same-label benchmark. |
| Lundberg and Lee. *A Unified Approach to Interpreting Model Predictions*. 2017. | NeurIPS 2017. [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | SHAP. |
| Ribeiro, Singh and Guestrin. *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier*. 2016. | ACM KDD 2016. [Conference paper](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | LIME. |

## Slide wording

Use: “Published reference results; input and ARC training protocols differ from our main runs.” The current project estimates **target level**, not difficulty calibrated from student responses. No verified published MATH target-level result was identified in the two comparison papers; this is a limited literature check, not proof that none exists.

Research checks and prose were prepared with AI assistance. Tang Wenyue should verify the interpretation before presenting it as a personal explanation.
