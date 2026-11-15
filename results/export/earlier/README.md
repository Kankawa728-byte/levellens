# Earlier CPU reports, outside R7

中文版：[README.zh-CN.md](README.zh-CN.md)

Two reports of `python -m levellens.export` from before the final laptop check. They are retained
for provenance. `scripts/efficiency.py` reads `results/export/*.json` for R7 and does not look in
this folder. R7 requires latency measured on a laptop CPU for a final DistilBERT model; the final
DistilBERT checkpoints rechecked on Jikai's laptop are in the parent folder.

| Report | Model | Where it was measured | Why it is outside R7 |
|---|---|---|---|
| `arc-distilbert-ce-smoke-n400-s0.json` | DistilBERT smoke model: a trial run on 400 ARC training questions, checked on the dev split | on a team member's own laptop CPU, on 13 Nov, before the cluster was used | a trial model, not a final one |
| `race-distilbert-ce-s0.json` | RACE++ DistilBERT with the cross-entropy head | on a compute node of the cluster | a server CPU, not a laptop |

- The smoke checkpoint and the RACE++ cross-entropy checkpoint were not included in the checkpoint
  archive supplied for the laptop recheck. The compute-node timing must not be presented as laptop
  evidence.
- The RACE++ report says `same_predictions: false` because it was checked against the tolerance of
  that time, 1e-03. Its largest difference, 0.006, is inside the present tolerance of 0.03.
