# 设计

English: [design.md](design.md)

所有时间都是新加坡时间。Tier 1 和 Tier 2 的含义见 [work/README.zh-CN.md](../work/README.zh-CN.md)。

## 问题和用户

**用户。** 中学和大学预科阶段的老师、辅导老师和题库编辑，也就是需要编写新测验题的人。

**问题。** 题目难度通常有两种定法：一是先让学生试做，又慢又贵，还会让题目外泄；二是靠专家判断，结果不一致。在我们用的数据集上，最完整的两项近期比较是 Benedetto (2023) 和 Thuy 等人 (2025)。它们只覆盖传统模型和 BERT 系列模型，没有循环神经网络，没有竞赛数学，也都没有测试难度信号能否在学科之间迁移。

**用词规则。** 我们估计的是一道题的*目标等级*（考试阶段、学校年级、竞赛级别）。我们估计的不是在学生身上测出的难度。视频第一分钟和局限那一页 slides 都要说明这一点。

## 研究问题

1. 在三个学科上，传统模型、循环神经网络、Transformer 和开源大模型从题目文字估计等级，效果如何比较？
2. 有序输出头、贝叶斯调参和稳定性改进，能否提升最好的模型？
3. 学到的难度信号能否跨学科迁移，能否推广到没见过的数据集和年级？
4. 在每个学科里，是什么让一道题变难？

## 数据集

数据集大小来自各数据集页面，11 月 6 日核对；ARC 过滤后的划分大小来自两篇已发表的论文。有两个数字要在第一次加载数据时再数：ScienceQA 的纯文本题数，以及 MATH 里少量标为 “Level ?” 的题，这些题会被去掉。

| 数据集 | 学科 | 训练 / 验证 / 测试 | 等级 | 来源 | 许可证 | 用途 |
|---|---|---|---|---|---|---|
| **RACE++**（RACE middle + high、RACE-C） | 英语阅读 | 100,568 / 5,599 / 5,642 | 3 级：初中、高中、大学 | `ehovy/race` + `tasksource/race-c` | RACE：仅限非商业研究。RACE-C：未注明许可证。 | 训练 |
| **ARC（带学校年级）** | 中小学科学 | 3,358 / 862 / 3,530 | 7 级：3 到 9 年级，分布很不均衡 | AI2 的 `ARC-V1-Feb2018.zip`（681 MB；Hugging Face 上的版本没有年级字段） | CC BY-SA 4.0 | 训练 |
| **MATH** | 竞赛数学 | 6,750 / 750 / 5,000 | 5 级：Level 1 到 5 | `EleutherAI/hendrycks_math`；从训练集留出 10% 作验证集 | MIT | 训练 |
| **ScienceQA（纯文本）** | 科学、社会、语言 | 原始划分 12,726 / 4,241 / 4,241 | 1 到 12 年级 | `derek-thomas/ScienceQA` | CC BY-NC-SA 4.0（原始发布；这个镜像标的是 CC BY-SA 4.0） | 外推测试 |
| **GSM8K**（Tier 2） | 小学数学 | 7,473 / 1,319 | 没有等级；用解题步数作代理 | `openai/gsm8k` | MIT | 外推测试 |

- RACE++ 的等级从配置名得到，因为 `all` 配置里没有等级列。各划分之间的文章互不重叠。
- ARC 和已发表的论文一样，只保留 4 个选项、纯文字的选择题。主实验用全部 3,358 条训练题，配合按类别加权的损失。论文用的平衡子集（2,293 条）只用于复现实验和多数类基线对照测试。
- MATH 只用题目文字作输入。解答只用于解释分析，从不作为输入。
- 输入格式沿用已发表的做法：默认是题干加全部选项。
- 我们提供下载脚本，不提供数据。RACE 和 ScienceQA 仅限非商业用途。

**如果 ARC 没通过第一次检查。** 如果读不到 ARC 的年级标签，或者过滤后的 ARC 训练集只有 3,000 条或更少，我们改用 ScienceQA 纯文本子集作为训练用的科学数据集，前提是它的训练题超过 3,000 条。如果只是题数和 3,358 / 862 / 3,530 不一致，我们把过滤方法调整成和 qde-ordinality 仓库的数据准备一致，继续用 ARC。组长在 11 月 11 日（周三）检查之后决定。

## 模型

| 类别 | 模型 | 运行设备 | 分工 |
|---|---|---|---|
| 最低基线 | 多数类、分层随机、学科先验（每种 MATH 题型最常见的等级） | CPU | D |
| 传统模型 | TF-IDF（词的 1–2 元组和字符的 3–5 元组）配 Logistic Regression、Linear SVM、Complement Naive Bayes。R2DE，即 Benedetto (2023) 用作基线的 TF-IDF + Random Forest 难度回归模型。把可读性、词的罕见度、长度、词性比例、依存树深度、从句数、数字、否定词和 LaTeX 运算符这些特征送入 LightGBM 和有序逻辑回归。 | CPU | D |
| 循环神经网络 | 两层 BiLSTM（可切换成 BiGRU），用 GloVe 300d 词向量和注意力池化；为 MATH 写一个能识别 LaTeX 的分词器 | GPU | E |
| Transformer | `microsoft/deberta-v3-base`（主模型）、`distilbert/distilbert-base-uncased`（速度快，用于演示）。Tier 2：`google-bert/bert-base-uncased`（复现）、`FacebookAI/roberta-base`（只跑最好的输出头）。 | GPU | E |
| 开源大模型 | `Qwen/Qwen3-4B-Instruct-2507`，零样本（Tier 2：少样本），按等级标签的概率打分。不用付费 API。 | GPU | D |

**输出头。** 等级是有顺序的：把一道大学题预测成“高中”，比预测成“初中”错得少。Tier 1 在同一个主干上比较两种输出头：普通交叉熵，以及 CORN（条件有序回归）。Tier 2 再加两种：回归后取整，以及在等级之间学习分界点的有序 logit 头。

## 已发表的基线结果

两篇论文和我们用的是相同的 RACE++ 划分，以及相同的 ARC 过滤后测试集（3,530 题）。在 ARC 上，它们的训练集是部分平衡过的子集：最常见的两个年级各削减到 500 条，共 2,293 条。所以 ARC 这一列只能和我们的平衡复现实验直接比较。平衡 DRPS 的含义见下面的“指标”。这些数字已在 11 月 6 日对照 arXiv 版本核对。

| 论文 | RACE++ | ARC（平衡训练集） |
|---|---|---|
| Benedetto (2023)，arXiv:2305.10236 | BERT：RMSE 0.372 ± 0.012，Spearman 0.789 ± 0.014。DistilBERT：Spearman 0.790 ± 0.007。最好的传统混合模型：Spearman 0.668 ± 0.004。 | BERT：RMSE 1.535 ± 0.148，Spearman 0.502 ± 0.139。最好的传统混合模型：Spearman 0.400 ± 0.006。 |
| Thuy、Loginova 和 Benoit (2025)，EvalLAC'25，arXiv:2507.00736 | OrderedLogitNN：准确率 0.861 ± 0.003，RMSE 0.384 ± 0.005，平衡 DRPS 0.130 ± 0.001。CORN：平衡 DRPS 0.127 ± 0.003。普通分类：准确率 0.847 ± 0.004。 | 分类准确率 0.421 ± 0.005。OrderedLogitNN 平衡 DRPS 0.674 ± 0.004。回归 RMSE 1.412 ± 0.013。 |
| 多数类基线（同一篇论文） | 准确率 0.620，RMSE 0.616，平衡 DRPS 0.667。 | 准确率 0.409，RMSE 2.195，平衡 DRPS 2.286。 |

多数类这一行同时是单元测试：我们的数据划分和指标代码必须复现出这些数字（ARC 用平衡训练子集）。我们没有找到针对 MATH 等级的已发表难度估计结果，所以在 MATH 上以我们自己的传统模型为参照。

## 指标

- 准确率、宏平均 F1（附每个等级的精确率和召回率），以及相邻准确率（差一级以内算对）。
- 二次加权 kappa（QWK）：衡量一致程度，离得越远的错误罚得越重。
- 等级序号上的 MAE 和 RMSE，以及 Spearman 等级相关系数。
- 平衡 DRPS（Balanced DRPS，离散排序概率得分）：一种考虑等级顺序的概率误差，在各等级之间做了平衡。
- 期望校准误差（ECE，15 个分箱）、训练时间、每道题在 CPU 和 GPU 上的延迟、参数量。
- 跑了 3 个随机种子的设置，报告均值 ± 标准差；最好的模型和第二名之间计算配对 bootstrap 95% 置信区间。
- 所有系统（包括大模型）还要在每个数据集上一个固定的 1,000 题分层测试子集上打分。

## 我们自定的规格

草稿。组长在 11 月 12 日（周四）12:00 冻结。视频里逐条报告是否达成。

| 编号 | 规格 |
|---|---|
| R1 | 3 个训练数据集，每个的带标签训练题都超过 3,000 条；另加至少 1 个外推测试集。 |
| R2 | 至少 4 类模型（传统模型、循环神经网络、Transformer、开源大模型，最低基线另算），3 种输出类型（分类、回归、有序），全部使用相同的数据划分。 |
| R3 | RACE++：用完整训练集时 Spearman ≥ 0.79（如果 GPU 条件迫使训练集截到 2 万到 3 万条，则 ≥ 0.77，并说明这一点）。ARC：3 个随机种子的平均 Spearman ≥ 0.50。MATH：最好的 Transformer 在 QWK 上胜过最好的传统模型，且差值的 95% bootstrap 区间高于零。 |
| R4 | 在 3 个数据集中至少 2 个上，某种有序输出头的平衡 DRPS 低于交叉熵输出头（同一主干、同一划分）。 |
| R5 | ARC 稳定性：3 个随机种子的 Spearman 标准差 ≤ 0.05（已发表的 BERT 在平衡训练集上是 0.139）。 |
| R6 | 每次微调 ≤ 1 GPU 小时；在 Ampere 架构的显卡上，整个项目 ≤ 20 GPU 小时，每次运行都记录。用别的显卡时，报告实际用时。 |
| R7 | 演示：在笔记本电脑的 CPU 上每道题 ≤ 1 秒（DistilBERT，取 100 道题计时）。 |
| R8 | 每次运行都从一个配置启动；所有表格和图都能用一条命令从结果库重建。 |

## 范围

**Tier 1，必须完成**

- 最低基线；用 5 折搜索调参的 TF-IDF 模型；R2DE；带 LightGBM 和有序逻辑回归的特征模型；全局 SHAP。
- 带交叉熵和 CORN 两种输出头的 BiLSTM，4 组配置的网格。
- DeBERTa-v3 和 DistilBERT 配两种输出头，覆盖全部三个数据集，1 个随机种子。
- ARC 稳定性：完整训练集、类别权重、按验证集 QWK 早停、3 个随机种子。
- Optuna（TPE 采样加中位数剪枝），在 DeBERTa-v3 上每个数据集 12 次试验。RACE++ 的试验用 1 万条子集。
- 开源大模型，零样本，在 1,000 题子集上。
- 最终 Transformer 上的 LIME 和 SHAP；错误图；人工核对最严重的 30 个错误；迁移矩阵；ScienceQA 外推；ARC Easy 和 Challenge 的比较；物理意义图。
- Streamlit 应用：校准页面和结果页面。
- 效率表。

**Tier 2，11 月 12 日（周四）21:00 决定要做之后**

- 把回归和有序 logit 两种输出头加进实验网格，这样共比较 4 种输出头。
- BERT 复现实验，和已发表的数字并排；RoBERTa 配最好的输出头。
- RACE++ 和 MATH 的主要设置跑 3 个随机种子。
- BiLSTM 的 Optuna 搜索；开源大模型少样本。
- 混合 Random Forest；短语级 TF-IDF；GloVe 平均向量配 Logistic Regression。
- GSM8K 外推；“审核测验”页面。
- 每次推送自动运行测试；指南“添加你自己的数据集或模型”。

**不在计划内。** 只有上面全部做完才考虑：输入格式消融、学习曲线、在 ARC 上使用逐层学习率衰减、把 DeBERTa 的句向量和特征融合、一个跨学科的联合模型、让 ModernBERT 读取 RACE++ 的阅读文章、基于依存句法图的图模型、一个建议怎样改写题目的页面。

## 解释与外推

- **全局。** 在特征 LightGBM 上按学科做 SHAP：难度有多少来自语言负担（可读性、罕见词、长度），有多少来自推理负担（依存树深度、从句、数字、否定、LaTeX 运算符）。分工 D。
- **局部。** 在微调后的 Transformer 上，对每个数据集的 50 道测试题做 LIME 和 SHAP，汇总成每个等级最主要的线索词。LIME 的做法是删掉一些词、观察输出怎么变，以此解释单个预测。应用里展示同样的视图。分工 F。
- **错误分析。** 有序混淆矩阵、每个等级的召回率，按 MATH 题型、ARC 年级和长度分组统计（分工 F）；人工核对最严重的 30 个错误，区分模型错误和标签噪声（分工 B）。
- **迁移。** 每个学科的模型给其他学科的测试集打分（3 × 3 矩阵，用 Spearman 衡量）。分工 F。
- **外推。** ARC 模型在 ScienceQA 上分别测试 3 到 9 年级以内和以外的题；比较同一年级里 ARC Easy 和 Challenge 的题。Tier 2：MATH 模型在 GSM8K 上看分数和解题步数的关系。分工 F。
- **物理意义。** 有序输出头的分数在 logit 尺度上，类似 Rasch 模型里的题目难度 *b*：答对的概率是 sigmoid(能力 − *b*)。分界点标出一个等级在哪里结束、下一个等级从哪里开始。真正的 *b* 需要用学生的作答数据来锚定，我们会说明这一点。
- **超出本应用。** 学校题库、MOOC 测验、资格考试题、文章的阅读等级、自适应练习里的题目排序，以及借助多语言编码器用于其他语言。每一种都需要附上同样的锚定说明。

## Streamlit 应用

- **校准（Calibrate）。** 老师粘贴一道题和它的选项，阅读题还要粘贴文章。页面显示每个等级的概率和期望等级；一个“为什么难”面板，高亮关键词并用仪表显示各项特征；以及一个“如果这样改”编辑器，把修改后的版本并排重新打分。
- **结果（Results）。** 带置信区间的排行榜、混淆矩阵、迁移热力图和外推图，全部从结果库读取。
- **审核测验（Audit a quiz）**（Tier 2）。上传 CSV 或粘贴约 10 道题。每道题都得到一个等级；画出测验的等级分布，和老师设定的蓝图（例如 30 / 50 / 20）对照，标出不匹配的题，并可导出带标签的 CSV。

## 算力

- GPU 运行都经过分工 C。备用方案是组长的工作站，RTX 5090（32 GB）。代码从配置里选择设备。
- 计算节点可能不能上网，所以先把数据和模型权重放进缓存。
- 每个作业控制在 1 小时以内，支持断点续跑。
- 整个实验网格的估算：在 16 GB 以上的 Ampere 架构显卡上约 12 到 14 GPU 小时。

## 风险

| 风险 | 应对 |
|---|---|
| 读不到 ARC 的年级标签，或者剩下的题太少。 | 见上面的“如果 ARC 没通过第一次检查”。 |
| GPU 权限来得晚，或者速度慢。 | 在 RTX 5090 上跑；RACE++ 截到 2 万到 3 万条；提前把缓存离线准备好。 |
| DeBERTa-v3 在 fp16 下出现 NaN 损失。 | 改用 bf16 或 fp32，或者改用 RoBERTa 作主干。 |
| 时间不够。 | 先做 Tier 1。Tier 2 要等 11 月 12 日（周四）21:00 决定之后才做。结果冻结之后不做新实验。 |
| 有组员没空。 | 当天由搭档顶上；组长在群里重新安排。 |
| 视频超过 10 分钟。 | 每部分固定时长；11 月 14 日（周六）计时彩排；删内容，不删讲者。 |
| 上传晚了。 | 11 月 15 日（周日）12:00 前上传，最晚 18:00；只由组长提交；23:59 之后谁都不再提交。 |
| 标签是目标等级，不是在学生身上测出的难度。 | 全程说“目标等级”；展示人工核对里的标签噪声例子；把“用学生作答数据锚定”作为未来工作。 |
| 大模型可能见过 MATH 和 GSM8K 的题。 | 说明这一点，并在固定子集上打分。 |
| 许可证。 | RACE 和 ScienceQA 仅限非商业用途；RACE-C 未注明许可证。我们注明每个来源，只提供下载脚本，不提供数据。 |

## 参考文献

每一条的发表出处要先确认，再放上 slides（任务 B2）。条目保留英文原文。

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
