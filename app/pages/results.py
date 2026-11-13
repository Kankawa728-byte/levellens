"""LevelLens demo app, Results page (task F7): leaderboards and confusion matrices from the results store.

Nothing is computed twice: the tables come from `levellens.results.leaderboard`, the confusion
matrix from the saved predictions of a run. No model is loaded on this page.
"""

from __future__ import annotations

import streamlit as st

from levellens.figures import DATASET_TITLES, confusion_figure, leaderboard_figure
from levellens.results import TABLE_SPLITS, leaderboard, load_runs, predictions_file

st.title("Results")

runs = load_runs()
if runs.empty:
    st.info("The results store is empty. Start a run: `python -m levellens.run data=toy model=majority`.")
    st.stop()
st.caption(f"Read from `results/runs.jsonl`: {len(runs)} runs. Runs that differ only in the seed are one line.")

order = [name for name in ("race", "arc", "math") if name in set(runs["dataset"])]
order += sorted(set(runs["dataset"]) - set(order))
dataset = st.radio("Dataset", order, format_func=lambda name: DATASET_TITLES.get(name, name), horizontal=True)

st.subheader("Leaderboard")
splits = [split for split in TABLE_SPLITS if not leaderboard(runs, dataset, split).empty]
split = st.radio("Split", splits, horizontal=True,
                 help="test_1k is the fixed 1,000-question subset every system is scored on.")
st.dataframe(leaderboard(runs, dataset, split), hide_index=True, width="stretch")
st.caption("Sorted by QWK (quadratic weighted kappa: agreement that punishes far misses more than near ones). "
           "Lower is better for MAE, RMSE, Bal. DRPS and ECE.")
figure = leaderboard_figure()
if figure:
    with st.expander("All datasets in one figure"):
        st.image(str(figure))

st.subheader("Confusion matrix")
candidates = [run_id for run_id in runs.loc[runs["dataset"] == dataset, "run_id"] if predictions_file(run_id).exists()]
if not candidates:
    st.info("No run of this dataset has saved predictions.")
    st.stop()
best = runs[runs["run_id"].isin(candidates)].sort_values("metrics.dev.qwk", ascending=False)["run_id"].iloc[0]
run_id = st.selectbox("Run", candidates, index=candidates.index(best), help="Preselected: the best QWK on dev.")
st.image(str(confusion_figure(run_id)), width=760)
st.caption("Rows are the labelled level, columns the predicted level; each row sums to 100%. "
           "The levels are in order, so cells next to the diagonal are near misses.")
