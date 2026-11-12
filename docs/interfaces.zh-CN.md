# 接口（v1）

English: [interfaces.md](interfaces.md)

这一页说明代码骨架。以代码为准，组长负责让这一页和代码保持一致。

## 第一次运行

```bash
pip install -r requirements.txt
python -m pytest
python -m levellens.run data=toy model=majority
python -m levellens.data.prepare
python -m levellens.run data=math model=majority
python -m levellens.results
```

- `data=toy` 是一个随意生成的小数据集，不用下载。用它来试你的模型代码。它的结果行写到 `results/scratch/`，这个文件夹被 git 忽略，不会进入共用的 `results/runs.jsonl`。
- `python -m levellens.data.prepare` 下载四个数据集并写出数据划分文件。大约需要 1.2 GB 磁盘空间。只准备其中几个时，写上数据集的名字：`python -m levellens.data.prepare math arc`。
- `python -m levellens.results` 从 `results/runs.jsonl` 重建 `results/tables/` 里的所有表格。

## 一道题

每个数据集适配器返回的题目都有相同的字段。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 在数据集内唯一 |
| `dataset` | str | `race`、`arc`、`math`、`scienceqa` 或 `toy` |
| `split` | str | `train`、`dev` 或 `test` |
| `question` | str | 题干文字 |
| `options` | list of str | 选项；MATH 为空 |
| `passage` | str or None | 阅读文章（只有 RACE++ 有）。Tier 1 里不作为模型输入 |
| `text` | str | 模型输入：题干，后面是带字母的每个选项，例如 `Which is a mammal? A. frog B. whale` |
| `level` | int | 0 到 K − 1，从最低等级排到最高等级 |
| `level_name` | str | 例如 `middle school`、`grade 5`、`Level 3` |
| `meta` | dict | 额外字段：MATH 的题型和解答，ARC 的年级、子集（Easy 或 Challenge）和考试名，ScienceQA 的年级和学科，正确选项的序号 |

RACE++ 的 K 是 3，ARC 是 7，MATH 是 5，ScienceQA 是 12，toy 数据集是 3。

```python
from levellens.data import info, load_split

train = load_split("math", "train")   # a list of items
info("math").level_names              # ('Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5')
```

## 数据划分

| 数据集 | `train` | `dev` | `test` | `test_1k` | 其他 |
|---|---|---|---|---|---|
| `race` | 100,568 | 5,599 | 5,642 | 1,000 | |
| `arc` | 3,358 | 862 | 3,530 | 1,000 | `train_balanced`：2,293 |
| `math` | 6,748 | 750 | 5,000 | 1,000 | |
| `scienceqa` | 6,508 | 2,144 | 2,224 | 1,000 | |
| `toy` | 240 | 60 | 90 | 90 | |

- `test_1k` 是从 `test` 里固定抽出的分层子集。每个系统都在它上面打分，开源大模型也一样。
- `train_balanced` 是已发表论文用的 ARC 训练子集：5 年级和 8 年级各截到 500 题。用 `data=arc_balanced` 选它，只用于复现运行。
- MATH 没有公开的验证集。我们从训练题里留出 10%，等级和题型的比例保持不变。训练题里有 2 道标着 “Level ?”，已丢掉。
- 子集按题目 id 的哈希值挑选，不用随机数生成器。每台机器得到的题目都相同。
- `python -m levellens.data.prepare` 会把每个数据划分和 `levellens/data/manifest.json` 里的指纹比对，不一致就报错。谁都不重新划分，只有组长更新指纹。
- `scripts/download_data.py`（分工 C）可以调用这条命令，再去下载模型权重。

## 一个模型

```python
from levellens.models.base import Model, texts

class MyModel(Model):
    def fit(self, train, dev, config):
        """Train on `train`. Use `dev` for early stopping and for choosing settings."""

    def predict_proba(self, items):
        """Return an array of shape [n, K]. Each row sums to 1."""
```

- 构造函数收到 `num_levels` 和模型配置里 `params:` 下面的设置。它们存成 `self.num_levels` 和 `self.params`。
- `config` 是整次运行的配置：`config.seed`、`config.device`、`config.head.name`、`config.model.params`。
- `texts(items)` 返回每道题的模型输入。
- 每道题只预测一个数的模型（回归、R2DE）返回形状为 [n] 的数组。指标模块把它四舍五入成等级，截到合法范围内，再为概率类指标生成 one-hot 行。
- 开源大模型返回各等级标签的概率，并重新归一化到加起来等于 1。
- 可选：给效率表用的 `n_params`，以及 `save(folder)` 和 `load(folder)`，有了它们 Predictor 才能加载训练好的模型。

每个模型在 `configs/model/` 里有一个配置文件。`target` 是类，`params` 是它的设置：

```yaml
# configs/model/majority.yaml
name: majority
target: levellens.models.floors.Majority
params: {}
```

`levellens/models/floors.py` 是完整的例子。输出头按同样的方式在 `configs/head/` 里放一个文件；模型从 `config.head` 读取自己的输出头。

## 一次运行

一次运行从一个配置启动，并向 `results/runs.jsonl` 写入一行：

```bash
python -m levellens.run data=arc model=majority seed=1 owner=D
python -m levellens.run data=race model=majority limit_train=20000 variant=demo
```

- `data=`、`model=` 和 `head=` 各自从 `configs/data/`、`configs/model/` 和 `configs/head/` 里选一个文件。
- 其他的 `key=value` 都是覆盖一项设置。用点号进入嵌套的设置：`model.params.C=10`。
- 公共设置和它们的默认值在 `configs/train.yaml` 里：`seed`、`owner`、`tier`、`variant`、`run_id`、`device`、`limit_train`、`eval_splits`、`save_predictions`、`save_model`。

```json
{
  "run_id": "arc-deberta-v3-corn-s0",
  "started": "2026-11-12T14:03:00+08:00",
  "owner": "E",
  "tier": 1,
  "dataset": "arc",
  "model": "deberta-v3",
  "head": "corn",
  "variant": "",
  "seed": 0,
  "config": {},
  "metrics": {"dev": {}, "test": {}, "test_1k": {}},
  "train_seconds": 0,
  "predict_seconds": {"dev": 0, "test": 0},
  "device": "",
  "n_params": 0,
  "git_commit": ""
}
```

- 默认的 `run_id` 是 `<dataset>-<model>-<head>[-<variant>]-s<seed>`。只有设置不同的几次运行，要给它们不同的 `variant`。同一个 `run_id` 再跑一次时，以最新的一行为准。
- `metrics` 下面每个数据划分都有 `n`、`accuracy`、`macro_f1`、`adjacent_accuracy`、`qwk`、`mae`、`rmse`、`spearman`、`spearman_score`、`drps`、`balanced_drps`、`ece` 和 `per_level`（每个等级的精确率、召回率和题数）。没有定义的指标是 `null`。
- `spearman` 用预测的等级算。`spearman_score` 用连续分数算：期望等级，或者回归输出。
- 结果行只通过 `levellens.results` 写入。不要手工编辑这个文件。
- 每次运行还把它的预测（题目 id、标签、验证集和测试集上的模型输出）保存在 `results/preds/<run_id>.npz`，这样画图、算 bootstrap 区间和做人工核对时都不用再加载模型。
- 测试集只用来出最终数字。设置在验证集上选，或者在训练集上用交叉验证选。

```python
from levellens.evaluate import evaluate, paired_bootstrap
from levellens.results import load_predictions, load_runs

runs = load_runs()                                   # one row per run; columns such as metrics.test.qwk
test = load_predictions("arc-majority-none-s0")["test"]   # {"ids", "y", "output"}
evaluate(test["y"], test["output"], num_levels=7)
```

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

`"stub"` 不需要训练好的模型。它返回随意生成的概率，所以应用从第一天就能开始做：题目越长，等级越高。`Predictor.load("stub", dataset="arc")` 给出 ARC 的 7 个等级。真实模型在 11 月 13 日（周五）替换它（任务 E8）：带 `save_model=true` 的运行会保存模型，`Predictor.load("<run_id>")` 加载它。

## 路径和设备

不写死路径。用四个环境变量：

| 变量 | 存放 | 默认值 |
|---|---|---|
| `LEVELLENS_DATA` | 下载的数据集和保存好的数据划分 | 仓库里的 `data/` |
| `LEVELLENS_CACHE` | 模型权重、分词后的数据、特征缓存、检查点、保存的模型 | 仓库里的 `cache/` |
| `LEVELLENS_RESULTS` | `runs.jsonl`、预测、表格、图 | 仓库里的 `results/` |
| `LEVELLENS_DEVICE` | `cpu`、`cuda` 或 `cuda:0`。它优先于配置里的 `device=` | 不设置：PyTorch 找到 GPU 就用 GPU，否则用 CPU |

`data/` 和 `cache/` 被 git 忽略。用 `levellens.paths` 取这些文件夹：`data_dir()`、`cache_dir()`、`results_dir()`。
