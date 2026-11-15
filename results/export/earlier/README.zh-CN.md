# 较早的 CPU 报告，不进入 R7

English: [README.md](README.md)

这里是最终笔记本检查之前由 `python -m levellens.export` 写出的两份报告，保留下来用于追溯。
`scripts/efficiency.py` 为 R7 读取的是 `results/export/*.json`，不读这个文件夹。R7 要求在笔记本 CPU
上测最终 DistilBERT 模型的延迟；在 Jikai 笔记本上重新检查过的最终 DistilBERT checkpoint 在上一级文件夹里。

| 报告 | 模型 | 在哪里测的 | 为什么不进入 R7 |
|---|---|---|---|
| `arc-distilbert-ce-smoke-n400-s0.json` | DistilBERT smoke 模型：用 400 道 ARC 训练题做的试跑，在 dev 划分上检查 | 11 月 13 日在一位组员自己的笔记本 CPU 上，那时还没有用集群 | 是试跑模型，不是最终模型 |
| `race-distilbert-ce-s0.json` | RACE++ 的 DistilBERT，交叉熵输出头 | 在集群的计算节点上 | 是服务器 CPU，不是笔记本 |

- smoke checkpoint 和 RACE++ 交叉熵 checkpoint 不在交给笔记本复查的那份 checkpoint 压缩包里。
  计算节点上的计时不能当作笔记本上的证据。
- RACE++ 那份报告写的是 `same_predictions: false`，因为它是按当时的容差 1e-03 检查的。它的最大差
  0.006 在现在的容差 0.03 以内。
