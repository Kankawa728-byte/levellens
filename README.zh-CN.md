# LevelLens

LevelLens 根据考题的文字，估计这道题面向的目标难度等级。老师贴进一道新题，就能看到它对应哪个等级，以及原因。

第 38 桌的小组项目，共六人。硬截止时间：**2026 年 11 月 15 日（周日）23:59**。本仓库里的时间都是新加坡时间。

English: [README.md](README.md)。每个页面都有中文版，页面顶部有切换链接。中英文有出入时，以英文版为准。

## 从这里开始

1. 把本页读完，大约五分钟。
2. 在下面的表里找到自己的那一份，点开它的任务页。
3. 还没有分工的同学：E 和 F 还没有人认领。组长在 11 月 12 日（周四）指定。
4. 在群里把你的 GitHub 用户名发给组长。推送代码要用到它。

## 六份工作

全部工作分成大小相同的六份。每一份都是 24 小时的 Tier 1 工作加 4 小时的 Tier 2 工作。小时数是估算，已经包含各自的 slides 和录制。逐项的小时数见 [work/README.zh-CN.md](work/README.zh-CN.md)。

| 分工 | 内容 | 负责人 | Tier 1 | Tier 2 | 状态 |
|---|---|---|---|---|---|
| [A](work/A-framework-integration.zh-CN.md) | 代码框架与整合 | Xingyang（组长） | 24 小时 | 4 小时 | 已分配 |
| [B](work/B-literature-data-audit.zh-CN.md) | 文献、数据检查与人工核对 | Wenyue | 24 小时 | 4 小时 | 已分配 |
| [C](work/C-compute-runs-efficiency.zh-CN.md) | 算力、跑实验与效率 | Qiutian | 24 小时 | 4 小时 | 已分配 |
| [D](work/D-baselines.zh-CN.md) | 基线：传统模型、特征模型、开源大模型 | Hailing | 24 小时 | 4 小时 | 已分配 |
| [E](work/E-neural-models.zh-CN.md) | 神经网络模型：BiLSTM 和 Transformer | **待认领** | 24 小时 | 4 小时 | [在 issue 5 认领](https://github.com/Kankawa728-byte/levellens/issues/5) |
| [F](work/F-interpretation-demo.zh-CN.md) | 模型解释、外推与演示应用 | Zhifei | 24 小时 | 4 小时 | 已分配 |

- **Tier 1** 是必须完成的部分。
- **Tier 2** 原本是 Tier 1 按时完成后再加的部分。组长在 11 月 12 日（周四）已经取消它，并且砍掉了一部分 Tier 1。重排见 [docs/schedule.zh-CN.md](docs/schedule.zh-CN.md)。表里的小时数是最初的估计，没有重新计算。

待认领的工作各适合谁：

- **E** 在关键路径上。它适合微调过 Transformer 的同学。
- **F** 适合喜欢画图和做前端的同学。应用用 Streamlit 写。

## 怎样认领

1. 读 E、F 两份工作的任务页。
2. 打开你想要的那一份的 issue，评论 “I take this”，并写上你的名字。先评论的人得到这一份。没有 GitHub 账号的同学，直接在群里告诉组长。
3. 组长把你的名字写进上面的表。

认领截止时间原是 11 月 10 日（周二）23:59。E 和 F 在 11 月 12 日（周四）还没有人认领，所以组长在这一天指定。

## 关键日期

| 时间 | 事项 |
|---|---|
| 11 月 12 日（周四）16:00 | 代码骨架进入 `main`；规格冻结；Tier 2 取消，Tier 1 已砍掉一部分 |
| 11 月 12 日（周四）23:59 | 每个人都跑通了第一次运行；D、E、F 写出第一批结果行 |
| 11 月 13 日（周五）12:00 | Transformer 训练封装完成；实验网格交给 GPU 跑 |
| 11 月 13 日（周五）23:59 | **结果冻结**：不做新实验 |
| 11 月 14 日（周六）12:00 | 所有表格和图定稿 |
| 11 月 14 日（周六）16:00 | slides 定稿；16:30 计时彩排 |
| 11 月 14 日（周六）23:59 | 每个人的录制文件都交给组长 |
| 11 月 15 日（周日）14:00 | 定稿签字：视频、反思文档、代码 |
| 11 月 15 日（周日）15:00 | 上传，最晚 18:00 |
| 11 月 15 日（周日）23:59 | 硬截止 |

这些是 11 月 12 日（周四）的重排日期。按天、按分工排的完整日程见 [docs/schedule.zh-CN.md](docs/schedule.zh-CN.md)。

## 我们要做什么

- **任务。** 预测一道题的目标等级：考试阶段、学校年级或竞赛级别。等级有先后顺序。
- **数据。** 三个训练数据集：RACE++（英语阅读，3 级）、ARC（中小学科学，7 个年级）、MATH（竞赛数学，5 级）。ScienceQA 用来检验科学模型能外推多远。
- **模型。** 最低基线、传统模型和特征模型、BiLSTM、带有序输出头的 DeBERTa-v3 和 DistilBERT，以及不经训练的开源大模型。
- **评估。** 九项指标、同一测试集上已发表的基线，以及八条可度量的规格，逐条给出是否达成。
- **解释。** SHAP 和 LIME、对最严重错误的人工核对，以及学科之间的迁移。
- **演示。** 一个 Streamlit 应用。老师给一道题打分，看它为什么难，修改后重新打分。

细节见 [docs/design.zh-CN.md](docs/design.zh-CN.md)。

## 仓库结构

字母表示这个文件归哪一份工作。每个文件只有一个负责人。

```text
levellens/                    Python 代码包
  data/                 A     race.py  arc.py  math_.py  scienceqa.py  toy.py   -> 统一格式
                              prepare.py 写出所有模型读取的数据划分文件
  features/             D     tfidf.py  linguistic.py
                        E     glove.py  latex.py
  models/               A     base.py（模型约定）  floors.py（多数类，完整的例子）
                        D     classic.py  llm.py
                        E     recurrent.py  transformer.py
  heads/                E     ce.py  corn.py
  explain/              D     global_shap.py
                        F     local.py  errors.py
  evaluate.py           A     全部指标、bootstrap 区间
  figures.py            F     结果图和错误图
  predictor.py          A     实验和应用共用的 Predictor API
  results.py            A     runs.jsonl -> 表格
  run.py                A     python -m levellens.run data=arc model=deberta-v3 head=corn seed=0
  config.py  paths.py   A     一次运行的配置；数据、缓存和结果文件夹
configs/                A     目录结构；每个负责人添加自己模型的配置
scripts/                C     download_data.py、stage_cache、作业模板
notebooks/main.ipynb    A     先跑一个小的端到端示例，再重建所有表格
notebooks/data_checks.* B     docs/datasets.md 背后的数据检查
app/                    F     streamlit_app.py、pages/
tests/                  A     公共测试；每个负责人为自己的模块加测试
results/                      runs.jsonl、表格、图
docs/                   A、B  设计、接口、日程、协作规则、视频；B 添加数据集卡片、相关工作、人工核对
work/                   A     六份工作
```

代码骨架已经在 `main` 里。从 [docs/interfaces.zh-CN.md](docs/interfaces.zh-CN.md) 的“第一次运行”开始。

## 规则摘要

- 每项工作开一个分支。由组长合并进 `main`。
- 每次运行向 `results/runs.jsonl` 写入一行。slides 上的每个数字都来自这里。
- 测试集只用来出最终数字。
- 这个仓库是公开的。不要提交数据、模型权重、录制文件、学号、邮箱、签名和访问令牌。
- 卡住超过一小时？在群里说。

完整规则见 [docs/workflow.zh-CN.md](docs/workflow.zh-CN.md)。

## 页面

| 页面 | 内容 |
|---|---|
| [work/README.zh-CN.md](work/README.zh-CN.md) | 六份工作逐项的小时数；谁在等谁 |
| [docs/design.zh-CN.md](docs/design.zh-CN.md) | 数据集、模型、指标、规格、范围、参考文献 |
| [docs/interfaces.zh-CN.md](docs/interfaces.zh-CN.md) | 题目字段、模型接口、结果行、Predictor |
| [docs/schedule.zh-CN.md](docs/schedule.zh-CN.md) | 里程碑和逐日安排 |
| [docs/workflow.zh-CN.md](docs/workflow.zh-CN.md) | 分支、pull request、文件、运行 |
| [docs/video.zh-CN.md](docs/video.zh-CN.md) | 视频分段、slides、录制、反思文档 |
| [docs/datasets.zh-CN.md](docs/datasets.zh-CN.md) | 数据集卡片：各等级题数、长度、完全重复、许可证 |
| [docs/related_work.zh-CN.md](docs/related_work.zh-CN.md) | 已发表的基线结果，已对照论文和公开代码核对 |
| [docs/audit.zh-CN.md](docs/audit.zh-CN.md) | 15 个最严重错误的人工核对：规则和记录表 |
