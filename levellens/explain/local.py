"""LIME on one question, and the cue words of each level over a sample of test questions (task F1).

    python -m levellens.explain.local arc-distilbert-ce-s0             # 20 test questions of the run's dataset
    python -m levellens.explain.local arc-distilbert-ce-s0 --items 40 --samples 1000

LIME (Ribeiro et al., 2016) explains one prediction: it removes random subsets of the words of the
question, scores every shortened copy with the model, and fits a weighted linear model from "which
words are present" to the model output. The weight of a word says how much its presence moves the
output. We use the `lime` package (BSD 2-Clause) for the sampling and the fit.

What this file adds:
- One call gives the weights towards every level, and one weight towards the expected level. The linear
  fit is linear in its target, and we keep every word in the fit, so the weight towards the expected
  level is exactly sum_k k * weight_k. The app shows that single number per word.
- `cue_words` runs LIME on a fixed stratified sample of test questions and sums the weights per level.

Files written by the command, in `results/explain/`:
    <run_id>_lime_items.csv   one line per question, word and level
    <run_id>_cue_words.csv    the top words of each level
and the figure `results/figures/cue_words_<run_id>.png`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from levellens.data import load_split, stratified_subset
from levellens.paths import results_dir
from levellens.predictor import CARD, Predictor, model_folder

NUM_SAMPLES = 500   # shortened copies per question; the app uses fewer to stay fast
NUM_ITEMS = 20      # test questions per dataset (cut from more on Thu 12 Nov)
TOP_WORDS = 8       # cue words listed per level
MIN_WORD_LENGTH = 2  # drops the option letters A, B, C, D and single digits from the cue-word summary


def explain(predictor: Predictor, text: str, num_samples: int = NUM_SAMPLES, seed: int = 0) -> dict:
    """Explain the prediction for one question text.

    Returns
        words      the distinct words of the text, in order of first appearance
        weights    [n_words, K]: how much each word moves the probability of each level
        expected   [n_words]: how much each word moves the expected level (positive = harder)
        proba      [K]: the probabilities for the unchanged text
    """
    from lime.lime_text import IndexedString, LimeTextExplainer

    num_levels = len(predictor.level_names)
    proba = predictor.predict([text])["proba"][0]
    num_words = IndexedString(text, bow=True).num_words()
    if num_words == 0:
        return {"words": [], "weights": np.zeros((0, num_levels)), "expected": np.zeros(0), "proba": proba}

    def classifier(copies: list[str]) -> np.ndarray:
        """Score the shortened copies of the text. LIME calls this function."""
        # a copy with every word removed is empty; a full stop keeps every model's tokeniser happy
        return predictor.predict([copy if copy.strip() else "." for copy in copies])["proba"]

    explainer = LimeTextExplainer(class_names=predictor.level_names, bow=True, random_state=seed)
    explanation = explainer.explain_instance(
        text, classifier, labels=list(range(num_levels)), num_features=num_words, num_samples=num_samples
    )
    weights = np.zeros((num_words, num_levels))
    for level, pairs in explanation.as_map().items():
        for word_index, weight in pairs:
            weights[word_index, level] = weight
    words = [explanation.domain_mapper.indexed_string.word(index) for index in range(num_words)]
    return {"words": words, "weights": weights, "expected": weights @ np.arange(num_levels), "proba": proba}


def run_dataset(run_id: str) -> str:
    """Return the dataset a saved model was trained on."""
    return json.loads((model_folder(run_id) / CARD).read_text(encoding="utf-8"))["dataset"]


def explain_items(
    predictor: Predictor, items: list[dict], num_samples: int = NUM_SAMPLES, seed: int = 0
) -> pd.DataFrame:
    """Run LIME on every item. One line per item, word and level."""
    lines = []
    for item in items:
        result = explain(predictor, item["text"], num_samples=num_samples, seed=seed)
        predicted = int(result["proba"].argmax())
        for word, row in zip(result["words"], result["weights"]):
            for level, weight in enumerate(row):
                lines.append({
                    "item_id": item["id"], "true_level": item["level"], "predicted_level": predicted,
                    "word": word.lower(), "level": level, "weight": float(weight),
                })
    return pd.DataFrame(lines, columns=["item_id", "true_level", "predicted_level", "word", "level", "weight"])


def summarise(item_weights: pd.DataFrame, level_names: list[str], top: int = TOP_WORDS) -> pd.DataFrame:
    """Return the cue words of each level: the words with the largest summed weight towards it.

    Stop words (the, of, ...) and one-character tokens are left out. `items` counts the questions
    the word came from, so a reader can tell a word seen once from a word seen in many questions.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = item_weights["word"]
    kept = item_weights[(words.str.len() >= MIN_WORD_LENGTH) & ~words.isin(ENGLISH_STOP_WORDS)]
    totals = (
        kept.groupby(["level", "word"])
        .agg(weight=("weight", "sum"), items=("item_id", "nunique"))
        .reset_index()
    )
    totals = totals[totals["weight"] > 0].sort_values(["level", "weight", "word"], ascending=[True, False, True])
    totals = totals.groupby("level").head(top).copy()
    totals.insert(1, "level_name", [level_names[level] for level in totals["level"]])
    totals["rank"] = totals.groupby("level").cumcount() + 1
    return totals.reset_index(drop=True)


def cue_words(
    run_id: str, num_items: int = NUM_ITEMS, num_samples: int = NUM_SAMPLES, split: str = "test", seed: int = 0
) -> tuple[pd.DataFrame, list[Path]]:
    """Run LIME for one saved model on a stratified sample of its test questions and write both tables.

    The sample is picked by hash of the question id, so it is the same on every machine.
    Returns the cue-word table and the files written.
    """
    predictor = Predictor.load(run_id)
    items = stratified_subset(load_split(run_dataset(run_id), split), num_items, seed=seed)
    item_weights = explain_items(predictor, items, num_samples=num_samples, seed=seed)
    summary = summarise(item_weights, predictor.level_names)

    folder = results_dir() / "explain"
    folder.mkdir(parents=True, exist_ok=True)
    files = [folder / f"{run_id}_lime_items.csv", folder / f"{run_id}_cue_words.csv"]
    item_weights.round({"weight": 5}).to_csv(files[0], index=False, encoding="utf-8", lineterminator="\n")
    summary.round({"weight": 5}).to_csv(files[1], index=False, encoding="utf-8", lineterminator="\n")
    return summary, files


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(description="LIME cue words per level for saved models")
    parser.add_argument("run_ids", nargs="+", help="runs trained with save_model=true")
    parser.add_argument("--items", type=int, default=NUM_ITEMS, help="test questions per run")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="shortened copies per question")
    args = parser.parse_args(argv)

    from levellens.figures import cue_words_figure

    for run_id in args.run_ids:
        _, files = cue_words(run_id, num_items=args.items, num_samples=args.samples)
        for path in [*files, cue_words_figure(run_id)]:
            print(path)


if __name__ == "__main__":
    main()
