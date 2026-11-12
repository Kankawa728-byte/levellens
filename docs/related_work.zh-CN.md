# 已发表的基线结果和实验设置核对

English: [related_work.md](related_work.md)

**负责人：** Wenyue。2026 年 11 月 12 日对照两篇论文的 **arXiv v1** 版本和它们的公开代码核对。下面的数值和所引的表格行一致。它们是已发表的参考结果，不是 LevelLens 测出来的结果。

## 能不能直接比较

**本项目不是严格的复现。** 划分题数一致、多数类基线一致，不能说明输入文本、抽到的训练题 ID、模型选择或预处理是相同的。有两处实质差别必须说明：

1. **RACE++ 的上下文：** 两组作者的代码都先把阅读文章接在题干前面，再接选项。LevelLens 的 `format_text` 目前只用题干和带字母的选项。这是输入信息量上的差别，不只是 token 格式的差别。在 A 批准做带文章的复现之前，已发表的分数只作为背景参考。B 的工作不改动冻结的现有输入。
2. **ARC 的训练集：** 论文把 5 年级和 8 年级各限制在 500 条。LevelLens 的主实验保留 3,358 条训练题，它的 `train_balanced` 有 2,293 条，各类别的数量和论文一致。作者用 pandas 抽样，LevelLens 按确定性的哈希给 ID 排序。抽到的题目 ID 是否相同**没有**核实。即使用 `train_balanced`，也只是实验设置对齐的比较，不是逐位一致的复现。

其他差别有：选项分隔方式（作者的 Transformer 输入用 `[SEP]`，这里用 `A.`、`B.` 等）、BERT 对本项目的 DeBERTa-v3 和 DistilBERT、已发表结果跑五次而本项目跑一次或三次。换模型结构是有意的。输入和训练集的差别必须在图注和表注里写明。

## Benedetto 2023

版本：[arXiv:2305.10236v1](https://arxiv.org/abs/2305.10236v1)，第 3–5 节，表 1–2 在 PDF 第 6 页。RACE++ 用 100,568 条训练题和 5,642 条测试题。ARC 保留四个选项的题，再对训练集做部分平衡，测试划分有 3,530 条。论文比较了三种编码：只用题干、题干加正确选项、题干加全部选项。主要的 Transformer 结果行用全部选项。模型回归出一个连续分数，映射成离散标签，再报告 RMSE、R² 和 Spearman 相关系数。数值是五次运行的均值和标准差。论文还研究了更小的 RACE++ 训练集，那些结果没有拿来代替全量数据的结果行。

代码确认了三点：`results_get_evaluation_transformers.py` 里的种子是 0–4；评估之前先做离散化；`_get_new_row_text_difficulty_df` 把文章接在前面。标准的 RACE++ 验证划分由数据管理器带入，arXiv 版介绍数据集的那一段没有写它的 5,599 这个题数。不要自己另造验证划分，也不要推测论文没有报告的调参细节。

| 数据集 | 论文里的具体行 | 指标 | 已发表的均值 ± 标准差 | 核对 |
|---|---|---|---|---|
| RACE++ | BERT Q_All | RMSE | 0.372 ± 0.012 | 表 1，一致 |
| RACE++ | BERT Q_All | Spearman | 0.789 ± 0.014 | 表 1，一致 |
| RACE++ | DistilBERT Q_All | Spearman | 0.790 ± 0.007 | 表 1，一致 |
| RACE++ | Ling., Read., TF-IDF Q_Correct | Spearman | 0.668 ± 0.004 | 表 1，一致 |
| ARC | BERT Q_All | RMSE | 1.535 ± 0.148 | 表 2，一致 |
| ARC | BERT Q_All | Spearman | 0.502 ± 0.139 | 表 2，一致 |
| ARC | W2V Q_Correct, Ling., TF-IDF Q_All | Spearman | 0.400 ± 0.006 | 表 2，一致 |

两条“最好的传统混合模型”用的特征组合不同，不能当作同一种模型配置来展示。

## Thuy、Loginova 和 Benoit 2025

版本：[arXiv:2507.00736v1](https://arxiv.org/abs/2507.00736v1)，第 6 节，表 2–3 在 PDF 第 9 页。RACE++ 的训练、验证、测试题数是 100,568/5,599/5,642。ARC 在训练集下采样之前的题数是 3,358/862/3,530，训练子集有 2,293 条。BERT-base-uncased 分别配回归、分类和有序输出头做微调。输入是题干和全部选项，中间用分隔 token 隔开，实现里还加入了 RACE++ 的文章。结果是五次运行的平均。报告的指标有准确率、RMSE、基于概率的平衡 DRPS，以及它的退化分布版本。

公开配置把基础种子设为 42，主循环再加上从 1 开始的运行序号，所以种子是 43–47。平衡 DRPS 先在每个真实类别内对累积概率误差的平方取平均，再在类别之间取平均。概率版和退化版两列要分开。对硬预测来说，退化版相当于按类别平衡的等级绝对误差。和这些表比较时不要除以 K−1。

| 数据集 | 输出头 | 指标 | 已发表的均值 ± 标准差 | 核对 |
|---|---|---|---|---|
| RACE++ | OrderedLogitNN | 准确率 | 0.861 ± 0.003 | 表 2，一致 |
| RACE++ | OrderedLogitNN | RMSE | 0.384 ± 0.005 | 表 2，一致 |
| RACE++ | OrderedLogitNN | 平衡 DRPS | 0.130 ± 0.001 | 表 2，一致 |
| RACE++ | CORN | 平衡 DRPS | 0.127 ± 0.003 | 表 2，一致 |
| RACE++ | 分类 | 准确率 | 0.847 ± 0.004 | 表 2，一致 |
| ARC | 分类 | 准确率 | 0.421 ± 0.005 | 表 3，一致 |
| ARC | OrderedLogitNN | 平衡 DRPS | 0.674 ± 0.004 | 表 3，一致 |
| ARC | 回归 | RMSE | 1.412 ± 0.013 | 表 3，一致 |
| RACE++ | 多数类 | 准确率；RMSE；平衡 DRPS | 0.620; 0.616; 0.667 | 表 2，一致；标准差 0.000 |
| ARC | 多数类 | 准确率；RMSE；平衡 DRPS | 0.409; 2.195; 2.286 | 表 3，一致；标准差 0.000 |

ARC 的多数类这一行在平衡训练集出现并列时选 **8 年级**，这和作者明确写的“并列时取最高标签”规则一致，也和 LevelLens 的 `Majority` 一致。8 年级同时是完整训练集的多数类，所以这项检查分不出用的是完整训练集还是平衡训练集。多数类检查只能验证对应的标签分布、并列规则和指标的算术。本机检查在保留三位小数后复现了已发表的数值，见 `results/data_checks/majority_check.json`。

## 代码证据和可追溯性

下面固定了所查看的提交，这样外部仓库以后的改动不会悄悄改变这次核对的依据。

| 证据 | 原始出处 |
|---|---|
| Benedetto 把文章接在前面 | [`_data_manager.py`，第 181–183 行](https://github.com/lucabenedetto/qdet-comparison/blob/e4a5090e227239e59ac61c6a95be567570c085f6/qdet_utils/data_manager/_data_manager.py#L181-L183) |
| Benedetto 的种子和离散化评估 | [`results_get_evaluation_transformers.py`](https://github.com/lucabenedetto/qdet-comparison/blob/e4a5090e227239e59ac61c6a95be567570c085f6/results_get_evaluation_transformers.py) |
| Thuy 把文章接在前面 | [`_data_manager.py`，第 219–249 行](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/tools/data_manager/_data_manager.py#L219-L249) |
| Thuy 拼接选项 | [`data_loader.py`，第 144–180 行](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/data_loader/data_loader.py#L144-L180) |
| Thuy 的 ARC 子集 | [`_arc_data_manager.py`，第 214–258 行](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/tools/data_manager/_arc_data_manager.py#L214-L258) |
| Thuy 的种子配置和循环 | [`config.py`](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/config/arc_bert_logit/config.py)、[`main.py`](https://github.com/arthur-thuy/qde-ordinality/blob/759903c7bc1495e01b54121d49fe18982933bbfe/src/main.py) |

## 参考文献和已确认的发表出处

这里列的是文献的发表信息，不表示每种方法都在 LevelLens 里实现了。已取消的 GSM8K 工作的参考文献不在目前的 slides 文献列表里。

| 文献 | 已确认的出处和来源 | 用途 |
|---|---|---|
| Benedetto. *A Quantitative Study of NLP Approaches to Question Difficulty Estimation*. 2023. | AIED 2023 海报论文集，CCIS 1831，第 428–434 页。[Springer](https://link.springer.com/chapter/10.1007/978-3-031-36336-8_67) | 上面的数值基线引用 **arXiv v1**。论文集里较短的摘要只提到两个数据集，arXiv 版包含 ARC。不要不加说明地换用版本。 |
| Thuy, Loginova and Benoit. *Ordinality in Discrete-level Question Difficulty Estimation: Introducing Balanced DRPS and OrderedLogitNN*. 2025. | EvalLAC 2025，与 AIED 2025 同期举办，CEUR-WS 4006，第 4 篇。[论文集](https://ceur-ws.org/Vol-4006/)、[论文](https://ceur-ws.org/Vol-4006/paper4.pdf) | 有序输出头和基于概率的评估。 |
| Lai, Xie, Liu, Yang and Hovy. *RACE: Large-scale ReAding Comprehension Dataset From Examinations*. 2017. | EMNLP 2017，第 785–794 页。[ACL Anthology](https://aclanthology.org/D17-1082/) | RACE 的出处。 |
| Liang, Li and Yin. *A New Multi-choice Reading Comprehension Dataset for Curriculum Learning*. 2019. | ACML 2019，PMLR 101，第 742–757 页。[PMLR](https://proceedings.mlr.press/v101/liang19a.html) | RACE-C 的出处。 |
| Clark et al. *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. 2018. | arXiv:1803.05457。[arXiv](https://arxiv.org/abs/1803.05457) | ARC 的出处。这里不写会议出处。 |
| Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset*. 2021. | NeurIPS 2021 Datasets and Benchmarks。[论文集里的论文](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf) | MATH 的出处，不是预测目标等级的基线。 |
| Lu et al. *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*. 2022. | NeurIPS 2022。[论文集](https://proceedings.neurips.cc/paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html) | ScienceQA 的出处。 |
| Shi, Cao and Raschka. *Deep neural networks for rank-consistent ordinal regression based on conditional probabilities*. 2023. | Pattern Analysis and Applications 26，第 941–955 页。[出版方](https://link.springer.com/article/10.1007/s10044-023-01181-9) | CORN 方法。 |
| He, Gao and Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. 2023. | ICLR 2023。[录用论文](https://openreview.net/pdf?id=sE7-XhLxHA) | 主干模型。预印本最早出现在 2021 年。 |
| Supraja, Khong and Tatinati. *Regularized Phrase-Based Topic Model for Automatic Question Classification With Domain-Agnostic Class Labels*. 2021. | IEEE/ACM TASLP 29，第 3604–3616 页。[DOI](https://doi.org/10.1109/TASLP.2021.3126937)、[出版方登记的元数据](https://api.crossref.org/works/10.1109/TASLP.2021.3126937) | 相关的分类工作，不是同一套标签上的基准。 |
| Lundberg and Lee. *A Unified Approach to Interpreting Model Predictions*. 2017. | NeurIPS 2017。[论文集](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | SHAP。 |
| Ribeiro, Singh and Guestrin. *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier*. 2016. | ACM KDD 2016。[会议论文](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | LIME。 |

## slides 上的写法

写法：“Published reference results; input and ARC training protocols differ from our main runs.” 本项目估计的是**目标等级**，不是根据学生作答校准的难度。在这两篇用来比较的论文里，没有找到经过核实的 MATH 目标等级的已发表结果。这只是范围有限的文献检查，不能证明这样的结果不存在。
