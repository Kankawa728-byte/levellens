# 接口（草稿 v0）

English: [interfaces.md](interfaces.md)

有了这一页，大家不用等代码骨架合并就能开始写。代码骨架进入 `main`（11 月 11 日（周三）12:00）之后，以代码为准，组长会把这一页改成和代码一致。

## 一道题

每个数据集适配器返回的题目都有相同的字段。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 在数据集内唯一 |
| `dataset` | str | `race`、`arc`、`math`、`scienceqa` 或 `gsm8k` |
| `split` | str | `train`、`dev` 或 `test` |
| `question` | str | 题干文字 |
| `options` | list of str | 选项；MATH 和 GSM8K 为空 |
| `passage` | str or None | 阅读文章（只有 RACE++ 有）。Tier 1 里不作为模型输入 |
| `text` | str | 模型输入：题干加全部选项，按一种固定格式拼接 |
| `level` | int | 0 到 K − 1，从最低等级排到最高等级 |
| `level_name` | str | 例如 `middle school`、`grade 5`、`Level 3` |
| `meta` | dict | 额外字段，例如 MATH 的题型、ARC 的 Easy 或 Challenge、ScienceQA 的年级 |

RACE++ 的 K 是 3，ARC 是 7，MATH 是 5。

数据划分由组长保存一次。所有模型读取同一批文件。谁都不重新划分。

## 一个模型

```python
class Model:
    def fit(self, train, dev, config):
        """Train on `train`. Use `dev` for early stopping and for choosing settings."""

    def predict_proba(self, items):
        """Return an array of shape [n, K]. Each row sums to 1."""
```

- 每道题只预测一个数的模型（回归、R2DE）就返回这个数。指标模块把它四舍五入成等级，截到合法范围内，再为概率类指标生成 one-hot 行。
- 开源大模型返回各等级标签的概率，并重新归一化到加起来等于 1。

## 一次运行

一次运行从一个配置启动，并向 `results/runs.jsonl` 写入一行：

```json
{
  "run_id": "arc-deberta-v3-corn-s0",
  "started": "2026-11-12T14:03:00+08:00",
  "owner": "E",
  "tier": 1,
  "dataset": "arc",
  "model": "deberta-v3",
  "head": "corn",
  "seed": 0,
  "config": {},
  "metrics": {"dev": {}, "test": {}, "test_1k": {}},
  "train_seconds": 0,
  "device": "",
  "n_params": 0,
  "git_commit": ""
}
```

- 结果行只通过 `levellens.results` 写入。不要手工编辑这个文件。
- 每次运行还要把它的预测（题目 id、标签、验证集和测试集上的概率）保存在结果旁边，这样画图、算 bootstrap 区间和做人工核对时都不用再加载模型。
- 测试集只用来出最终数字。设置在验证集上选，或者在训练集上用交叉验证选。

## Predictor

实验和应用用的是同一个类。

```python
from levellens.predictor import Predictor

predictor = Predictor.load("stub")
result = predictor.predict(["What is the main idea of the passage? A. ... B. ... C. ... D. ..."])
# result["proba"]           shape [n, K]
# result["level"]           most likely level for each question
# result["expected_level"]  probability-weighted level for each question
# result["level_names"]     names of the K levels
```

`"stub"` 不需要训练好的模型。它返回随意生成的概率，所以应用从第一天就能开始做。真实模型在 11 月 13 日（周五）替换它（任务 E8）。

## 路径

不写死路径。用两个环境变量：

| 变量 | 存放 | 默认值 |
|---|---|---|
| `LEVELLENS_DATA` | 下载的数据集和保存好的数据划分 | `./data` |
| `LEVELLENS_CACHE` | 模型权重、分词后的数据、特征缓存、检查点 | `./cache` |

这两个文件夹都被 git 忽略。
