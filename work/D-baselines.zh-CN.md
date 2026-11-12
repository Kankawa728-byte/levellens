# 分工 D · 基线：传统模型、特征模型、开源大模型

English: [D-baselines.md](D-baselines.md)

**负责人：** Hailing · **搭档：** B · **大小：** Tier 1 24 小时，Tier 2 4 小时

所有不是微调神经网络的东西：最低基线、传统模型、基于特征的模型，以及不经训练直接使用的开源大模型。这些数字说明神经网络模型到底多带来了多少。特征模型还给出“是什么让一道题变难”的第一个答案。

这一份的大部分在笔记本电脑的 CPU 上就能跑。它适合熟悉 scikit-learn 的同学。

## 归你管的文件

- `levellens/models/classic.py`、`levellens/models/llm.py`
- `levellens/features/tfidf.py`、`levellens/features/linguistic.py`
- `levellens/explain/global_shap.py`
- `configs/model/` 里你自己模型的配置，以及你自己模块的测试

## Tier 1

11 月 12 日（周四）的重排：日期是新的，标有“已砍掉”的任务不做。小时数是最初的估计。完整的重排见 [docs/schedule.zh-CN.md](../docs/schedule.zh-CN.md)。

| 编号 | 任务 | 完成标准 | 小时 | 截止 |
|---|---|---|---|---|
| D1 | 最低基线：多数类、分层随机、学科先验（每种 MATH 题型最常见的等级）。多数类已经在 `levellens/models/floors.py` 里；另外两个写在 `levellens/models/classic.py`。 | 三个数据集的结果行都已写进 `results/runs.jsonl`。 | 1 | 11 月 12 日（周四）23:59 |
| D2 | TF-IDF（词的 1–2 元组和字符的 3–5 元组），配 Logistic Regression、Linear SVM 和 Complement Naive Bayes。在训练集上用 5 折搜索选设置。 | 11 月 12 日（周四）：三个数据集上都有第一批结果行（TF-IDF 配 Logistic Regression，默认设置）。11 月 13 日（周五）：三个模型调参后的结果表。 | 5 | 第一批结果行：11 月 12 日（周四）23:59。调参后：11 月 13 日（周五）15:00 |
| D3 | R2DE：Benedetto (2023) 用作基线的 TF-IDF 加 Random Forest 难度回归模型 | 三个数据集的结果行。 | 2 | 11 月 13 日（周五）15:00 |
| D4 | 表层特征提取器：可读性、词的罕见度、长度、数字、否定词、LaTeX 运算符。不用句法分析器。这些特征送入 LightGBM。 | 特征已缓存到磁盘。三个数据集上都有结果行。 | 6.5 | 11 月 13 日（周五）20:00 |
| D5 | 在 LightGBM 模型上按学科做全局 SHAP：有多少来自语言负担（可读性、罕见词、长度），有多少来自推理负担（数字、否定、LaTeX 运算符） | 每个学科一张图，各配三句话说明它显示了什么。 | 2 | 11 月 14 日（周六）10:00 |
| D6 | 开源大模型，零样本：`Qwen/Qwen3-4B-Instruct-2507`，在固定的 1,000 题测试子集上。等级从各等级标签的概率读出。提示词只在验证集上调。不用付费 API。 | 打分脚本在 11 月 13 日（周五）12:00 交给 C。到结果冻结时，三个数据集都有结果行。 | 4.5 | 脚本：11 月 13 日（周五）12:00。结果行：11 月 13 日（周五）23:59 |

建议用的库：scikit-learn、LightGBM、textstat、wordfreq、shap、transformers。用什么由你定。

## Tier 2

11 月 12 日（周四）已取消。这些任务都不做。

| 编号 | 任务 | 完成标准 | 小时 |
|---|---|---|---|
| D7 | 开源大模型，少样本，例子取自训练集 | 三个数据集的结果行。 | 1.5 |
| D8 | 混合 Random Forest：把 TF-IDF 和语言特征放在一起，做法参照 Benedetto (2023) 的传统混合基线 | 三个数据集的结果行。 | 1 |
| D9 | 短语级 TF-IDF；GloVe 平均向量配 Logistic Regression | 三个数据集的结果行。 | 1.5 |

## 别人在等你

- C：开源大模型的打分脚本，11 月 13 日（周五）12:00。
- A：你的结果行，结果冻结时用来生成表格。

## 你在等

- A：代码骨架、适配器和 1,000 题子集。它们 11 月 12 日（周四）起已经在 `main` 里。从 [docs/interfaces.zh-CN.md](../docs/interfaces.zh-CN.md) 的“第一次运行”开始。

## 视频部分

2:20–3:40：最低基线、传统模型和特征模型（屏幕上跑一次交叉验证）、全局 SHAP 对每个学科说明了什么，以及不经训练的开源大模型。
