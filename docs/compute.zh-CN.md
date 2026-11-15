# 算力、运行与效率

English: [compute.md](compute.md)

负责人：分工 C，[work/C-compute-runs-efficiency.zh-CN.md](../work/C-compute-runs-efficiency.zh-CN.md)。

这个页面写三件事：GPU 运行所在的机器、提交了什么作业、以及别人要复现这些数字需要知道什么。
背后的脚本在 [scripts/README.zh-CN.md](../scripts/README.zh-CN.md)；效率表本身是
[results/tables/efficiency.md](../results/tables/efficiency.md)，由一条命令重建。

这个仓库是公开的，所以这里和脚本里都不出现集群的机器名、账号名和用户目录。
它们放在 `scripts/site.env`，git 会忽略这个文件。

## 1. 机器

GPU 运行：学校的共享 GPU 集群，调度器是 Slurm。登录节点只用来编辑和提交，所以环境的搭建和数据集的
下载都由 CPU 作业完成，每次运行都是一个批处理作业。计算节点没有外网，这就是任务 C2 存在的原因。

基线的 CPU 运行（分工 D）和标着 `mps` 的 BiLSTM 行（分工 E）是在组员自己的笔记本上跑的。
`results/runs.jsonl` 里每一行的 `device` 字段说明这次运行用的是哪一类机器，效率表按类分别统计小时数。

<!-- machine:start -->
下面的版本是 2026 年 11 月 14 日 smoke 作业开头 `python scripts/env_report.py` 打印的内容。
每个作业日志都以这张表开头，所以每次运行都能追到它跑在哪台机器上。

| 什么 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB，计算能力 12.0 |
| 驱动 | 595.71.05 |
| 操作系统 | Linux 6.8.0，glibc 2.39 |
| Python | 3.12.9，项目目录上的虚拟环境 |
| PyTorch | 2.11.0+cu128，按 CUDA 12.8 构建。能看到 GPU，支持 bfloat16 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| datasets | 5.2.0 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| optuna | 5.0.0 |
| lightgbm | 4.7.0 |

关于版本的两点说明：

- **这张卡不是 Ampere。** R6 要求的是 Ampere 级别的卡上 20 GPU 小时，并允许换卡但要如实报告实际
  小时数，效率表就是这么做的。集群的 Ampere 卡当时全被占满。这张卡更新更快，所以我们的小时数
  **低于**同样的运行在 Ampere 卡上的开销：在这里通过 20 小时的预算，并不能证明在 Ampere 上也能通过，
  而且低多少没有测。这个数字应读作这个项目在这张卡上的实际开销（这是 R6 允许的），而不是 Ampere 的数字。
- **transformers 钉在 4.57.6**，不是最新版。`requirements.txt` 给的是下界，所以 pip 会装 5.23.0，
  而 `levellens/models/llm.py` 是按 `>=4.57.1,<5` 写的，分工 D 的结果行也是用 4.57.6 产出的。
  这个钉住写在 `scripts/site.env` 的 `LEVELLENS_PIP_EXTRA` 里，这样各分工的数字来自同一个版本，
  `requirements.txt` 也保持组长写的样子。Blackwell 的卡需要 cu128 构建的 PyTorch，
  对应 `LEVELLENS_TORCH_INDEX`。
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
| C2：手工缓存数据集 | 由 `python -m levellens.data.prepare` 写出，并由指纹校验把关 | 11 月 12 日（周四）的重排砍掉了这一条：只有权重需要手工缓存 |
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
| Submitted | Kind | Job | What | State | Elapsed |
|---|---|---|---|---|---|
| 2026-11-14 08:07:24 | setup | 1738348 | levellens-setup | FAILED (1:0) | 00:00:14 |
| 2026-11-14 08:07:30 | fetch | 1738349 | --all | CANCELLED | 00:00:00 |
| 2026-11-14 08:10:57 | setup | 1738361 | levellens-setup | COMPLETED | 00:06:37 |
| 2026-11-14 08:10:57 | fetch | 1738362 | --all | COMPLETED | 00:03:18 |
| 2026-11-14 08:22:25 | setup | 1738385 | levellens-setup | COMPLETED | 00:01:44 |
| 2026-11-14 08:22:25 | fetch | 1738386 | --check | COMPLETED | 00:00:12 |
| 2026-11-14 08:22:25 | grid | 1738387 | configs/grid/smoke_gpu.txt | COMPLETED | 00:04:14 |
| 2026-11-14 19:07:01 | grid | 1739442 | configs/grid/tier1_transformers.txt 1-10 | COMPLETED | 00:38:47 |
| 2026-11-14 19:07:01 | grid | 1739443 | configs/grid/tier1_transformers.txt 11-20 | COMPLETED | 00:11:52 |
| 2026-11-14 19:07:01 | grid | 1739444 | configs/grid/tier1_bilstm.txt 5-6 | COMPLETED | 00:02:30 |
| 2026-11-14 19:07:01 | tune | 1739445 | data=arc model=distilbert head=ce owner=E n_trials=12 | COMPLETED | 00:03:23 |
| 2026-11-14 19:49:51 | measure | 1739497 | gpu race-deberta-v3-ce-s0 arc-deberta-v3-ce-s0 math-deberta-v3-ce-s0 arc-distilbert-corn-s0 | COMPLETED | 00:00:54 |
| 2026-11-14 19:49:51 | measure | 1739498 | cpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 | COMPLETED | 00:13:57 |
| 2026-11-14 20:06:36 | measure | 1739513 | gpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 race-deberta-v3-ce-s0 arc-deberta-v3-ce-s0 math-deberta-v3-ce-s0 | COMPLETED | 00:01:04 |
| 2026-11-14 20:06:36 | measure | 1739514 | cpu race-distilbert-ce-s0 arc-distilbert-corn-s0 math-distilbert-corn-s0 | COMPLETED | 00:13:40 |
| 2026-11-14 20:24:26 | clean | 1739543 | levellens-clean | COMPLETED | 00:12:57 |

怎么读这张表：

- 第一个 setup 作业把 PyTorch 装到了错误的位置而失败；会导致这种情况的三条路径写在第 3 节，都已修好。
  它后面那个 fetch 作业没有运行，因为它被设定为只在 setup 成功时才启动。
- 第二个 setup 作业建好了环境并跑了测试。紧随的 fetch 作业下载了四个数据集、两个 encoder、GloVe
  和 8 GB 的开放 LLM，它的离线检查在关掉 hub 的情况下加载了每一个划分和每一个模型。
- 第三个 setup 作业把 transformers 钉到 4.57.6，之后的 fetch 在该版本下重跑了离线检查。
- smoke 作业为三个需要 GPU 的模型族各自完成了训练、打分和写行，在一张卡上用了四分钟。
- `t1-tr-a` 和 `t1-tr-b` 是那 20 个 transformer 运行，各十个；`t1-bilstm` 是还缺的两个 RACE++
  BiLSTM 运行；`levellens-tune` 是 Optuna 搜索加最佳设置的重训。四个作业全部完成、没有失败的运行，
  两两并行，墙上时间 56 分钟。
- `measure` 作业是任务 C6。gpu 那个在模型训练时的精度下做重载检查并在 GPU 上计时；cpu 那个在 CPU 上
  计时，也就是规格 R7。它们跑了两轮：第一轮完成测量，第二轮在 CPU 检查因精度差异失败之后补上了
  GPU 上的重载检查，见第 6 节。
- `levellens-clean` 是任务 C7，见第 7 节。

没有任何 Tier 1 运行失败，所以任务 C4 没有需要记录的失败。
<!-- runlog:end -->

## 5. 交回给其他分工的东西

任务 C5，11 月 14 日完成。以下内容都在这个分工的分支里：

| 什么 | 在哪 | 给谁 |
|---|---|---|
| 23 行新结果 | `results/runs.jsonl`，共 56 行 | A：表格；F：图 |
| 23 个预测文件 | `results/preds/<run_id>.npz` | F：混淆矩阵、bootstrap 区间、错误审查 |
| Optuna 试验表 | `results/optuna/arc-distilbert.csv` 和 `.json` | E：任务 E7 |
| 延迟报告 | `results/export/`、`results/latency/` | A：R7 的通过与否 |
| 效率表 | `results/tables/efficiency.{md,csv}`、`efficiency_runs.csv` | A：R6 的通过与否 |
| 重建的表格和图 | `results/tables/`、`results/figures/` | A、B、F：slides |
| 23 个保存的模型，9.8 GB | 项目目录，不入库 | E：`levellens.export`；F：演示应用 |
| 作业日志 | `logs/`，不入库 | 需要查某次运行为什么失败的人 |

每一行都是通过 `levellens.results.append_run` 写入的，经由 `scripts/collect.py`：每个作业写到自己的
结果目录，所以从来没有两个作业同时往同一个 `runs.jsonl` 追加。

按 [workflow.zh-CN.md](workflow.zh-CN.md) 的规则，模型权重不入库。演示用的 DistilBERT 检查点
`arc-distilbert-corn-s0` 是所有保存模型里最快的，GPU 上每题 0.003 秒、计算节点的 CPU 上 0.056 秒；
它在仓库之外交给分工 F。

## 6. 效率与延迟

任务 C6。表在 [results/tables/efficiency.md](../results/tables/efficiency.md)，由
`python scripts/efficiency.py` 从 `runs.jsonl` 和延迟报告重建。一行一个模型、一行一次运行，
加上这个分工负责的两条规格的判定。

| | 实测 | 上限 | |
|---|---|---|---|
| **R6**，每次微调 | 0.19 小时，28 次里最长的 | 1 GPU 小时 | 通过 |
| **R6**，整个项目 | 0.89 GPU 小时 | 20 GPU 小时 | 通过 |
| **R7**，笔记本 CPU 上一道题 | 4 个已保存 DistilBERT 模型，0.0275 到 0.0331 秒 | 1 秒 | 通过 |

项目上限只计 GPU 小时。除此之外，项目还花了 0.45 CPU 小时（经典基线和特征模型）和 0.41 小时的
笔记本 GPU（最早的几行 BiLSTM）；效率表把这三类分开列，因为三者的一小时不是一回事。
这个 GPU 数字应读作项目在这张卡上的开销，不是 Ampere 的数字，见第 1 节。

20 个 transformer 运行一共只用了 51 分钟 GPU 时间，这就是 0.89 远在预算之内的原因。
整个项目最长的那次微调根本不是 transformer，而是 MATH 上第一次 BiLSTM 运行，跑在笔记本 GPU 上。

延迟，一次一道题，就像应用给粘贴进来的题目打分那样：

| 模型 | CPU 上 | GPU 上 | GPU 上按 32 批 |
|---|---|---|---|
| DistilBERT | Jikai 笔记本上 0.0275 到 0.0331 秒 | 0.003 秒 | 0.0003 秒 |
| DeBERTa-v3 | Jikai 笔记本上 0.1267 到 0.1411 秒 | 0.014 到 0.015 秒 | 0.0006 到 0.0011 秒 |

- CPU 数字来自 `python -m levellens.export`，它同时检查任务 E8。R7 要求笔记本 CPU，因此只计入
  Jikai 的 Intel i5-11400H 笔记本测得的四个 DistilBERT checkpoint：ARC、MATH、RACE++ 的 CORN
  模型及 ARC Optuna 重训模型。每份报告均按正常 CPU fp32 推理逐题计时 100 道题。此前在计算集群上
  测得的 DistilBERT smoke 与 RACE++ 交叉熵模型报告保存在 `results/export/cluster_legacy/`，
  不进入 R7 汇总。两个 DeBERTa-v3 CORN CPU 结果仅供参考，不参与 R7 判定。
- GPU 的数字来自 `python scripts/latency.py --device cuda`。DeBERTa-v3 回答一道题是 15 毫秒，
  所以演示在 GPU 上会是即时的，在 CPU 上也仍远低于一秒。

### 分工 E 的任务 E8：保存模型检查

11 月 14 日，`levellens.export` 对每一个已保存模型都报 **same predictions False**：Predictor 的概率
与训练时存下的概率之间最大差异为 6.2e-03 到 2.4e-02，而 `levellens/export.py` 当时的容差是 1e-03。

问题不在保存和加载。同样的检查把模型放到 GPU 上（也就是这些运行训练时的精度）做，六个被测模型的
最大差异都是 **3.0e-08**。差异来自 bfloat16 与 fp32 之比：bfloat16 的尾数只有 8 位，所以两者大约
在第二到第三位小数一致，而不是 `TOLERANCE` 当时的注释所说的第四位。`scripts/measure.sbatch` 会把两个
检查都跑，GPU 那个先跑。

分工 E 最终在笔记本上以训练打分时的 bf16 精度比较保存模型与原始概率，延迟则继续按正常 CPU fp32
推理计时。不同设备的 bf16 内核仍有细微差异，因此保留 0.03 的 `TOLERANCE`；每份报告记录实际最大差、
比较精度、标签一致率、库版本及 CPU 延迟。六个收到的 checkpoint 均通过概率检查：四个 DistilBERT
报告中的最大差为 0.0216，两个 DeBERTa-v3 报告中的最大差为 0.0191。复现时应使用训练所用的
Transformers 4.57.6 与 tokenizers 0.22.2。较新版本改变了 18 道 RACE++ 测试题的 token ID；
导出检查在比较阶段绕过旧 token 缓存。

## 7. 复现性说明

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
11 月 14 日以 `scripts/submit.sh clean` 运行，作业 1739543，分支位于提交 `e23164f`。
`scripts/clean_check.sbatch` 把分支克隆到一个空文件夹，按 `requirements.txt` 建一个全新的虚拟环境，
然后执行 README 第 1 到 3 步的命令，所有缓存和临时目录都放在这个一次性目录里，并且把全部
`LEVELLENS_*` 变量清空，这样它就像一个刚读到 README 的人那样运行。

一共 13 分钟，README 里的每一步都可用：

| 步骤 | 结果 |
|---|---|
| 1. 安装与 `python -m pytest` | 112 通过，2 跳过。跳过的两个标了 `data`，需要下载好的数据集 |
| 2. 小的端到端例子 | 四条命令都跑通；`results/scratch/runs.jsonl` 里三行，排行榜已生成 |
| 3. `levellens.results` 和 `levellens.figures` | **每一张表和每一张图都与入库的逐字节一致** |

第三行就是算力这一侧的规格 R8，而且它成立得比预期更强：这个新环境装的是**最新版本**的一切，
PyTorch 2.15.0+cu132 和 transformers 5.23.0，而那些运行是用 PyTorch 2.11.0+cu128 和
transformers 4.57.6 做的。表格依然一致，因为重建表格不重跑任何模型：`levellens.results` 和
`levellens.figures` 读的是 `runs.jsonl` 和保存下来的预测，两者都在仓库里。所以任何人都可以从一个
全新克隆重建报告里的每一个数字，在任何机器上，不需要 GPU、不需要数据集、也不需要特定版本的 PyTorch。

有两点新读者需要知道，但都不是 README 的问题：

- README 说从 GitHub 克隆。这个作业克隆的是本地分支，因为它运行时分支还没推上去。其余都与
  README 写的一致。
- 在这个集群上 `python` 来自一个 module，所以作业在第 1 步之前先加载它。笔记本上不需要这一步。
<!-- clean:end -->
