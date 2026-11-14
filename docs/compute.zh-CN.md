# 算力、运行与效率

English: [compute.md](compute.md)

负责人：分工 C，[work/C-compute-runs-efficiency.zh-CN.md](../work/C-compute-runs-efficiency.zh-CN.md)。

这个页面写三件事：GPU 运行所在的机器、提交了什么作业、以及别人要复现这些数字需要知道什么。
背后的脚本在 [scripts/README.md](../scripts/README.md)；效率表本身是
[results/tables/efficiency.md](../results/tables/efficiency.md)，由一条命令重建。

这个仓库是公开的，所以这里和脚本里都不出现集群的机器名、账号名和用户目录。
它们放在 `scripts/site.env`，git 会忽略这个文件。

## 1. 机器

GPU 运行：学校的共享 GPU 集群，调度器是 Slurm。登录节点只用来编辑和提交，所以环境的搭建和数据集的
下载都由 CPU 作业完成，每次运行都是一个批处理作业。计算节点没有外网，这就是任务 C2 存在的原因。

基线的 CPU 运行（分工 D）和 `device` 写作 `mps` 的 BiLSTM 行（分工 E）是在组员自己的笔记本上跑的。
`results/runs.jsonl` 里每一行的 `device` 字段说明这次运行用的是哪一类机器，效率表按类分别统计小时数。

<!-- machine:start -->
还没填。在跑作业的机器上执行 `python scripts/env_report.py` 就会打印这张表；每个作业日志的开头都是它。
<!-- machine:end -->

## 2. 一次运行是怎么到 GPU 上的

```bash
cp scripts/site.env.example scripts/site.env          # 一次性：目录、模块、GPU 申请方式
scripts/submit.sh setup                               # CPU 作业：按 requirements.txt 建环境
scripts/submit.sh fetch --all                         # CPU 作业：数据划分和权重进缓存
scripts/submit.sh -r results/smoke grid configs/grid/smoke_gpu.txt     # GPU 作业：每个模型族跑一个小运行
scripts/submit.sh -r results/jobs/t1a grid configs/grid/tier1_transformers.txt 1-5   # GPU 作业：五个运行
scripts/submit.sh tune data=arc model=distilbert head=ce owner=E n_trials=12            # GPU 作业：Optuna
python scripts/runlog.py                              # 提交了什么、结果如何
python scripts/collect.py results/jobs/*              # 把各作业的结果行合并进 results/
python -m levellens.results                           # 重建排行榜
python scripts/efficiency.py                          # 效率表，含 R6 和 R7 的通过与否
```

每个 GPU 作业都写到自己的结果目录，由 `-r` 指定。这样两个作业不会同时往同一个 `runs.jsonl` 追加，
之后由 `scripts/collect.py` 通过 `levellens.results.append_run` 合并。这个合并就是任务 C5。

一个配置进去，一行结果出来：`scripts/job.sbatch` 把脚本名之后的所有参数交给
`python -m levellens.run`，除此之外没有任何东西决定一次运行做什么。这就是算力这一侧的规格 R8。

大目录不放在 home 下，因为那里的配额很小：`site.env` 里的 `LEVELLENS_PROJECT_DIR` 把数据划分、
Hugging Face 缓存、检查点和结果都放到共享存储上的同一个目录下。这四个环境变量在
[interfaces.zh-CN.md](interfaces.zh-CN.md) 里有说明。

### 离线模式

GPU 作业会导出 `HF_HUB_OFFLINE=1`。它们需要的东西都由 `scripts/download_data.py` 事先取好：
四个数据集通过 `python -m levellens.data.prepare` 获取，这条命令同时会把每个划分与
`levellens/data/manifest.json` 里的指纹比对；权重包括 DeBERTa-v3-base、DistilBERT、
GloVe 6B 300d，以及 `levellens/models/llm.py` 里钉住版本号的 Qwen3-4B-Instruct-2507。
`python scripts/download_data.py --check` 会在关掉 hub 的情况下再加载一遍，这样缺文件会在 CPU 作业里
暴露，而不是在 GPU 作业里。

随后每个作业开始时，`scripts/stage_cache` 把数据划分和权重拷到计算节点的本地盘。只拷只读的东西；
检查点和结果行直接写到共享存储，所以它们比作业活得更久。

## 3. 相对计划的取舍

| 计划怎么写 | 实际怎么做 | 为什么 |
|---|---|---|
| C3：每个作业不超过一小时 | 一个运行一个作业是首选模板（`job.sbatch`，一小时）。Tier 1 网格按每五个运行一个作业提交（`grid.sbatch`） | 调度器限制一个用户同时只能跑几个作业。五个一组只排一次队，总体比 20 个单运行作业更快 |
| R6：在 Ampere 级别的卡上 20 GPU 小时 | 按调度器实际给到的卡，逐个运行和总计都如实报告小时数 | R6 本来就允许：“换成别的卡，我们报告实际小时数。” 集群的 Ampere 卡当时已被占满 |
| C2：手工缓存数据集 | 由 `python -m levellens.data.prepare` 写出，并由指纹校验把关 | 11 月 12 日的重新排期砍掉了这一条：只有权重需要手工缓存 |
| C4：Optuna 12 次试验 | 单独一个作业，因为 `levellens.tune` 只有一个 study 文件，不能同时跑两份 | study 是缓存里的一个 SQLite 文件 |

分工说明里的备选方案，也就是组长自己的工作站，没有用上。

### 作业会悄悄写错地方的三种情况

这三种情况各让我们白费了一次作业，而且都是静默的：作业不会说它把东西写到了别处，只会在之后
以"磁盘配额超限"失败。

1. **工作目录是仓库根目录时，任何 Python 都能 `import levellens` 成功**，因为 `levellens/` 就在那儿。
   于是"环境建好了吗"这个检查在什么都没装之前就通过了。`scripts/job_setup.sh` 改成从别的目录去问，
   这才真正测的是 `pip install -e .`。
2. **pip 会退化成用户安装**，装进 `~/.local`：当它无法写入所用 Python 的 site-packages 时就会这样。
   一个下载到一半的 PyTorch 就是 2.5 GB，而 home 是有配额的。`job_setup.sh` 现在设
   `PIP_USER=0` 和 `PYTHONNOUSERSITE=1`，`setup.sbatch` 在目标目录不可写时直接拒绝安装。
3. **作业会继承 `TMPDIR`**，通常是 `/tmp`，而它可能是一个很小的按用户计的卷。pip 的 wheel 就下载在
   那里。`job_setup.sh` 现在是直接设定 `TMPDIR`、`PIP_CACHE_DIR`、`XDG_CACHE_HOME`、
   `TRITON_CACHE_DIR` 和 `CONDA_PKGS_DIRS` 到 `LEVELLENS_PROJECT_DIR` 之下，而不是"没设才用默认值"，
   因为把它们挪走正是指定项目目录的意义所在。

## 4. 运行记录

提交了什么、什么时候、作业号是多少、最后怎么结束。`scripts/submit.sh` 把每次提交写入
`logs/submissions.tsv`，`python scripts/runlog.py` 向调度器查询状态并打印这张表。

<!-- runlog:start -->
还没填。
<!-- runlog:end -->

## 5. 交回给其他分工的东西

任务 C5。网格跑完后，以下内容都在 `main` 里：

| 什么 | 在哪 | 给谁 |
|---|---|---|
| 结果行 | `results/runs.jsonl`，每次运行一行 | A：表格；F：图 |
| 预测 | `results/preds/<run_id>.npz` | F：混淆矩阵、bootstrap 区间、错误审查 |
| 保存的模型 | `<LEVELLENS_CACHE>/models/<run_id>/`，不入库 | E：`levellens.export`；F：演示应用 |
| 效率表 | `results/tables/efficiency.{md,csv}`、`efficiency_runs.csv` | A：R6 和 R7 的通过与否 |
| 作业日志 | `logs/`，不入库 | 需要查某次运行为什么失败的人 |

按 [workflow.zh-CN.md](workflow.zh-CN.md) 的规则，模型权重不入库。演示用的 DistilBERT 检查点
（分工 F）在仓库之外交接。

## 6. 复现性说明

任务 C7。别人要拿到同样的数字需要知道什么。

**命令。** 安装和那个小的端到端例子在 [README](../README.zh-CN.md) 里。已入库结果行背后的运行：
基线在 [work/D-reproduce.zh-CN.md](../work/D-reproduce.zh-CN.md)，神经网络的网格在 `configs/grid/` 下，
网格文件本身就是运行记录——每个运行一行，与配置文件不同的设置都写在那一行里。

**随机种子。** `seed=` 会给 Python、NumPy 和 PyTorch 播种，见 `levellens.run.set_seed`。
数据划分不依赖种子：`levellens.data.prepare` 按题目 id 的哈希挑选 `test_1k`，所以每台机器拿到的是
同样的 1000 道题。Optuna 的 study 在 `levellens.tune` 里有自己的种子。

**版本。** 见第 1 节的表格，以及 [work/D-reproduce.zh-CN.md](../work/D-reproduce.zh-CN.md) 里开放 LLM 的
精确版本号。每一行结果里的 `n_params` 和 `device` 说明这次运行实际用了什么。

**哪些地方不能逐位复现。**

- 换一张 GPU、换一个 PyTorch 或 CUDA 构建，微调结果的末位小数会变：核函数的求和顺序不同。
  指标会在第三位小数上动，不会更多。
- `precision: auto` 在支持 bfloat16 的 GPU 上用 bfloat16，其他地方用 fp32，所以在 CPU 上重跑一次 GPU
  运行不会得到同样的权重。加 `model.params.precision=fp32` 可以让两者一致。
- 早停看的是 dev 上的 QWK，所以一次运行早一次或晚一次评估停下来，得到的模型就不同。在较慢的卡上
  `max_minutes` 也可能让运行在更少的轮数后结束；结果行里保留了 `train_seconds`，所以这是看得见的。
- 其余部分都有种子：一个 epoch 的批次来自 `length_buckets`，用的是按运行种子播种的生成器；
  类别权重由训练集标签算出。

**干净环境的复跑。**

<!-- clean:start -->
还没填。
<!-- clean:end -->
