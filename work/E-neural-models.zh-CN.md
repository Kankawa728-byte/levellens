# 分工 E · 神经网络模型：BiLSTM 和 Transformer

English: [E-neural-models.md](E-neural-models.md)

**负责人：** Jikai · **搭档：** C · **大小：** Tier 1 24 小时，Tier 2 4 小时

训练出来的神经网络模型，以及在它们之上的改进：有序输出头、用 Optuna 做贝叶斯调参，以及在小的科学数据集上稳定训练。

这一份在关键路径上。模型解释和演示都要等它的检查点。它适合微调过 Transformer 的同学。

## 归你管的文件

- `levellens/heads/`
- `levellens/models/recurrent.py`、`levellens/models/transformer.py`
- `levellens/features/glove.py`、`levellens/features/latex.py`
- `configs/model/` 和 `configs/head/` 里你自己的配置，以及你自己模块的测试

## Tier 1

11 月 12 日（周四）的重排：日期是新的，标有“已砍掉”的任务不做。小时数是最初的估计。完整的重排见 [docs/schedule.zh-CN.md](../docs/schedule.zh-CN.md)。

| 编号 | 任务 | 完成标准 | 小时 | 截止 |
|---|---|---|---|---|
| E1 | 输出头：交叉熵和 CORN（条件有序回归）。每个输出头返回损失和每个等级的概率。 | 有一个测试表明 CORN 的概率都在 0 和 1 之间，并且加起来等于 1。 | 1.5 | 11 月 12 日（周四）23:59 |
| E2 | 两层 BiLSTM，用 GloVe 300d 词向量和注意力池化，两种输出头都支持，1 组配置 | 11 月 13 日（周五）12:00：能在 MATH 上训练。20:00：三个数据集都有结果行。 | 5.5 | MATH：11 月 13 日（周五）12:00。全部：11 月 13 日（周五）20:00 |
| E3 | 已砍掉：为 MATH 写的能识别 LaTeX 的分词器 | `\frac` 这样的命令、符号 `^` 和 `_`、数字都保持为完整的 token。有一个覆盖 5 道题的测试。 | 1.5 | 11 月 12 日（周四）已砍掉 |
| E4 | 已砍掉：BiLSTM 网格：4 组配置，三个数据集都跑 | 结果行已写进 `results/runs.jsonl`，并指出最好的那组配置。 | 1 | 11 月 12 日（周四）已砍掉 |
| E5 | Transformer 训练封装：任意主干配任意输出头、按类别加权的损失、按验证集 QWK 早停、分词结果缓存、动态填充、每次运行不超过 1 GPU 小时 | 11 月 12 日（周四）：DistilBERT 配交叉熵在一个小子集上跑出一行结果。11 月 13 日（周五）：封装全部完成。 | 5.5 | 冒烟运行：11 月 12 日（周四）23:59。全部完成：11 月 13 日（周五）12:00 |
| E6 | 交给 C 的 Tier 1 实验网格配置：`microsoft/deberta-v3-base` 和 `distilbert/distilbert-base-uncased`，两种输出头，三个数据集，随机种子 0。ARC 另加随机种子 1 和 2，用全部 3,358 条训练题。 | C 拿到 20 个配置和运行它们的命令。 | 1.5 | 11 月 13 日（周五）12:00 |
| E7 | 在 ARC 上用 DistilBERT 做 Optuna：TPE 采样器、中位数剪枝器，12 次试验。最好的设置在完整训练集上重新训练。 | 11 月 13 日（周五）：搜索已启动。到结果冻结时，重新训练的结果行已到位。 | 3 | 启动：11 月 13 日（周五）15:00。结果行：11 月 13 日（周五）23:59 |
| E8 | 把最终模型导出到 Predictor API：每个数据集最好的模型，以及演示用的 DistilBERT | `Predictor.load(...)` 给出的测试集预测和训练那次运行的一样。DistilBERT 在笔记本电脑 CPU 上每道题不超过 1 秒。 | 1.5 | 11 月 14 日（周六）10:00 |

已知的坑：DeBERTa-v3 在 fp16 下可能出现 NaN 损失。改用 bf16 或 fp32；如果需要改用 RoBERTa 作主干，告诉组长。

建议用的库：PyTorch、transformers、datasets、optuna、提供 CORN 损失的 coral-pytorch。用什么由你定。

## Tier 2

11 月 12 日（周四）已取消。这些任务都不做。

| 编号 | 任务 | 完成标准 | 小时 |
|---|---|---|---|
| E9 | 另外两种输出头：回归后取整，以及在等级之间学习分界点的有序 logit 头。两种都加进实验网格。 | 两种输出头在两个主干、三个数据集上都有结果行。 | 2 |
| E10 | BERT 复现的配置（`google-bert/bert-base-uncased`，和论文相同的数据划分，ARC 用平衡训练子集），以及 `FacebookAI/roberta-base` 配最好的输出头的配置 | C 拿到这些配置。复现结果在表里和已发表的数字并排。 | 1 |
| E11 | 用 BiLSTM 的 Optuna 搜索代替 4 组配置的网格 | 三个数据集上最好那次试验的结果行。 | 1 |

## 别人在等你

- C：Tier 1 的实验网格配置，11 月 13 日（周五）12:00。
- F：DistilBERT 的冒烟检查点，11 月 13 日（周五）12:00；Predictor API 后面的最终模型，11 月 14 日（周六）10:00。
- A：你的结果行，结果冻结时用来生成表格。

## 你在等

- A：代码骨架和适配器。它们 11 月 12 日（周四）起已经在 `main` 里。从 [docs/interfaces.zh-CN.md](../docs/interfaces.zh-CN.md) 的“第一次运行”开始。
- C：冒烟作业在 11 月 13 日（周五）12:00，之后是结果行和检查点。

## 视频部分

3:40–5:30：BiLSTM 和 CORN 输出头（屏幕上跑一个训练单元格）、Transformer、Optuna 搜索、在 ARC 上的稳定训练，以及我们的数字和已发表数字的对比。
