# 分工 A · 代码框架与整合

English: [A-framework-integration.md](A-framework-integration.md)

**负责人：** Xingyang（组长） · **搭档：** F · **大小：** Tier 1 24 小时，Tier 2 4 小时

其他每一份工作都要接入的公共代码，以及最后把各部分合到一起：表格、代码包、视频和上传。

## 归你管的文件

- `levellens/data/`、`levellens/evaluate.py`、`levellens/results.py`、`levellens/predictor.py`、`levellens/run.py`
- `configs/data/`、`configs/train.yaml`、`requirements.txt`
- `tests/`（目录结构和公共测试）、`notebooks/main.ipynb`
- `README.md`、`docs/`、`work/`

## Tier 1

| 编号 | 任务 | 完成标准 | 小时 | 截止 |
|---|---|---|---|---|
| A1 | 代码包骨架、配置、命令行、从环境变量读取设备和路径、带占位模型的 Predictor API | `python -m levellens.run data=math model=majority` 在笔记本电脑和 GPU 机器上都能向 `results/runs.jsonl` 写入一行，代码不用改。`Predictor.load("stub")` 能返回概率。 | 4 | 11 月 11 日（周三）12:00 |
| A2 | 统一格式，以及 RACE++、ARC、MATH、ScienceQA（纯文本）的适配器。保存好的数据划分。每个数据集一个固定的 1,000 题分层测试子集。检查 ARC 的年级标签。 | 题数和 [docs/design.zh-CN.md](../docs/design.zh-CN.md) 一致，或者在 issue 里说明差异。数据划分只写一次，所有模型读取同一批文件。 | 5 | RACE++ 和 MATH：11 月 11 日（周三）12:00。ARC 和 ScienceQA：11 月 11 日（周三）18:00 |
| A3 | 指标模块：准确率、宏平均 F1、相邻准确率、QWK、MAE、RMSE、Spearman、平衡 DRPS、ECE。配对 bootstrap 区间。多数类测试。 | 在我们的数据划分上，多数类基线能复现 [docs/design.zh-CN.md](../docs/design.zh-CN.md) 里已发表的那一行数字：RACE++ 的，以及 ARC 用平衡训练子集时的。 | 3.5 | 11 月 11 日（周三）23:59 |
| A4 | 结果库和表格生成器 | 一条命令就能从 `results/runs.jsonl` 重建所有表格。 | 2 | 11 月 12 日（周四）12:00 |
| A5 | 冻结规格（R1–R8）、审阅和合并、每天整合、决定做不做 | 规格在 11 月 12 日（周四）12:00 冻结。pull request 在 12 小时内审阅。Tier 2 的决定在 11 月 12 日（周四）21:00 发出。11 月 13 日（周五）12:00 宣布结果冻结。 | 2 | 每天 |
| A6 | README、带讲解的主 notebook、检查文档字符串 | 新读者只看 README，就能安装代码包、跑通小的端到端示例并重建表格。 | 2 | 11 月 14 日（周六）18:00 |
| A7 | 拼接视频 | 六段录制按分镜顺序接好，视频不超过 9:40，并发给大家用于 20:00 的审看。 | 2 | 11 月 14 日（周六）20:00 |
| A8 | 最终打包和上传 | 提交记录里每个文件都在。11 月 15 日（周日）12:00 前上传，最晚 18:00。 | 0.5 | 11 月 15 日（周日） |

## Tier 2

| 编号 | 任务 | 完成标准 | 小时 |
|---|---|---|---|
| A9 | GSM8K 适配器 | F 能在 GSM8K 上运行 MATH 模型（任务 F8）。 | 1 |
| A10 | 每次推送自动运行测试 | 每个 pull request 的测试都在 GitHub 上运行，结果显示在 pull request 页面上。 | 1.5 |
| A11 | 指南“添加你自己的数据集或模型”，附一个完整例子 | 读者照着指南就能添加一个新数据集，不用读其余代码。 | 1.5 |

## 别人在等你

- 所有人：代码骨架 v0 在 11 月 11 日（周三）12:00 前进入 `main`。在那之前大家按 [docs/interfaces.zh-CN.md](../docs/interfaces.zh-CN.md) 写代码，所以这一页和代码要保持一致。
- 所有人：ARC 和 ScienceQA 的适配器、保存好的数据划分和 1,000 题子集，11 月 11 日（周三）18:00 前。
- B：数据检查要用的读取代码（任务 B1）。

## 你在等

- C、D、E 的结果行，用来生成表格。
- 所有人的录制文件，11 月 14 日（周六）16:00 前。

## 视频部分

- 开场，0:00–1:00：老师面临的问题、“目标等级”这个说法、研究问题、我们的规格。
- 结尾，8:50–9:40：每条规格是否达成、局限、未来工作、参考文献。
