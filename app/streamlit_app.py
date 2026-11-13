"""LevelLens demo app, Calibrate page (task F6).

    streamlit run app/streamlit_app.py

A teacher pastes a question and its options. The page shows
1. the probability of each level and the expected level,
2. "Why is it hard": the words that push the level up or down (LIME, see levellens/explain/local.py),
3. "What if": an edited copy of the question, scored side by side with the original.

The page works without a trained model: the model list always holds a stub with made-up
probabilities for each dataset. Saved models (runs with save_model=true) appear next to it.
The Results page is in app/pages/.
"""

from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from levellens.data import format_text, info
from levellens.explain.local import explain
from levellens.paths import cache_dir
from levellens.predictor import CARD, Predictor

STUB_DATASETS = ("race", "arc", "math")
DATASET_TITLES = {"race": "RACE++ (reading)", "arc": "ARC (science)", "math": "MATH (competition maths)"}
LIME_SAMPLES = 300      # fewer than the offline analysis, so the panel answers in a few seconds on a CPU
WORDS_LISTED = 6        # words named under the highlighted text, per direction
HARDER, EASIER = "235, 104, 52", "42, 120, 214"   # orange pushes the level up, blue pulls it down
EXAMPLES = {
    "race": ("What can we learn from the passage about the writer's attitude towards online learning?",
             "It is always better than classroom teaching.\nIt works well when students manage their own time.\n"
             "It should replace examinations.\nIt is too expensive for most families."),
    "arc": ("Which property of a mineral can be determined just by looking at it?",
            "luster\nmass\nweight\nhardness"),
    "math": ("How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?", ""),
}


def saved_models() -> dict[str, dict]:
    """Return the saved models in the cache folder: run id -> its predictor card."""
    folder = cache_dir() / "models"
    cards = {}
    for card in sorted(folder.glob(f"*/{CARD}")) if folder.is_dir() else []:
        cards[card.parent.name] = json.loads(card.read_text(encoding="utf-8"))
    return cards


@st.cache_resource(show_spinner="Loading the model…")
def load_predictor(name: str, dataset: str) -> Predictor:
    """Load a predictor once per session. `dataset` only matters for the stub."""
    return Predictor.load(name, dataset=dataset)


@st.cache_data(show_spinner="Asking the model which words matter…")
def explain_text(name: str, dataset: str, text: str) -> dict:
    """Return the LIME weights of one text. Cached, so the same text is explained once."""
    result = explain(load_predictor(name, dataset), text, num_samples=LIME_SAMPLES)
    return {"words": result["words"], "expected": result["expected"].tolist()}


def highlight(text: str, weights: dict[str, float]) -> str:
    """Return the text as HTML, each word shaded by its weight: orange = harder, blue = easier."""
    largest = max((abs(weight) for weight in weights.values()), default=0) or 1.0
    parts = []
    for piece in re.split(r"(\W+)", text):  # the same word boundaries as LIME
        weight = weights.get(piece)
        if weight is None:
            parts.append(html.escape(piece))
            continue
        colour = HARDER if weight > 0 else EASIER
        strength = min(abs(weight) / largest, 1.0) * 0.6
        parts.append(
            f'<span style="background: rgba({colour}, {strength:.2f}); border-radius: 3px; padding: 0 1px" '
            f'title="{weight:+.3f} levels">{html.escape(piece)}</span>'
        )
    return f'<div style="line-height: 1.9">{"".join(parts)}</div>'


def score(predictor: Predictor, text: str) -> dict:
    """Score one text. Returns proba [K], level, level name and expected level."""
    result = predictor.predict([text])
    level = int(result["level"][0])
    return {
        "proba": result["proba"][0], "level": level, "level_name": predictor.level_names[level],
        "expected": float(result["expected_level"][0]),
    }


def probability_table(level_names: list[str], columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return the probabilities as a table: one row per level, one column per version of the question."""
    table = pd.DataFrame(columns, index=pd.CategoricalIndex(level_names, categories=level_names, ordered=True))
    table.index.name = "Level"
    return table


def expected_label(level_names: list[str], expected: float) -> str:
    """Describe an expected level in words, for example "2.4, between Level 3 and Level 4"."""
    low = int(np.floor(expected))
    high = min(low + 1, len(level_names) - 1)
    if high == low or expected - low < 0.05:
        return f"{expected:.2f} on a 0–{len(level_names) - 1} scale, at {level_names[low]}"
    return f"{expected:.2f} on a 0–{len(level_names) - 1} scale, between {level_names[low]} and {level_names[high]}"


def question_form(key: str, dataset: str, question: str, options: str, passage: str = "") -> str:
    """Draw the inputs of one question and return the model input built from them."""
    question = st.text_area("Question", question, key=f"{key}_question", height=100)
    if dataset != "math":
        options = st.text_area("Options, one per line", options, key=f"{key}_options", height=120)
    if dataset == "race":
        st.text_area(
            "Passage (optional)", passage, key=f"{key}_passage", height=120,
            help="Kept next to the question for your reference. The Tier 1 models read the question and "
                 "the options only, so the passage does not change the score.",
        )
    lines = [line.strip() for line in options.splitlines() if line.strip()] if dataset != "math" else []
    return format_text(question, lines)


def calibrate() -> None:
    """Draw the Calibrate page."""
    st.title("Calibrate a question")
    st.caption("Estimate the level a question is written for, see which words drive it, and try an edit.")

    cards = saved_models()
    # transformers first: the demo model is the fine-tuned DistilBERT
    ordered = sorted(cards, key=lambda run_id: ("Transformer" not in cards[run_id]["target"], run_id))
    choices = {run_id: cards[run_id]["dataset"] for run_id in ordered}
    choices.update({f"stub · {dataset}": dataset for dataset in STUB_DATASETS})
    with st.sidebar:
        st.header("Model")
        choice = st.selectbox("Model", list(choices), label_visibility="collapsed")
        dataset = choices[choice]
        name = "stub" if choice.startswith("stub") else choice
        st.write(f"Trained on **{DATASET_TITLES.get(dataset, dataset)}**")
        st.write("Levels: " + ", ".join(info(dataset).level_names))
        if name == "stub":
            st.warning("The stub is not a trained model. Its numbers are made up from the length of the text.")
        st.caption(
            "A level here is the level the question was written for (school stage, grade or competition "
            "level). It is not a measured difficulty: that needs the answers of real students."
        )

    predictor = load_predictor(name, dataset)
    names = predictor.level_names
    example_question, example_options = EXAMPLES.get(dataset, EXAMPLES["arc"])

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Your question")
        text = question_form(f"original_{dataset}", dataset, example_question, example_options)
    if not text.strip():
        st.info("Type a question to score it.")
        return
    original = score(predictor, text)

    with right:
        st.subheader("Estimated level")
        first, second = st.columns(2)
        first.metric("Most likely level", original["level_name"],
                     help=f"The model gives this level {original['proba'].max():.0%} probability.")
        second.metric("Expected level", f"{original['expected']:.2f}")
        st.caption(expected_label(names, original["expected"]))
        st.bar_chart(probability_table(names, {"Probability": original["proba"]}), y_label="Probability", height=260)

    st.divider()
    st.subheader("Why is it hard?")
    st.caption(
        "Each word is removed in turn in many random copies of the question (LIME). "
        "Orange words raise the expected level, blue words lower it; a stronger colour means a larger effect."
    )
    explanation = explain_text(name, dataset, text)
    weights = dict(zip(explanation["words"], explanation["expected"]))
    st.markdown(highlight(text, weights), unsafe_allow_html=True)
    ranked = sorted(weights.items(), key=lambda pair: pair[1])
    harder = [(word, weight) for word, weight in reversed(ranked) if weight > 0][:WORDS_LISTED]
    easier = [(word, weight) for word, weight in ranked if weight < 0][:WORDS_LISTED]
    up, down = st.columns(2)
    up.markdown("**Raise the level**  \n" + (", ".join(f"{word} ({weight:+.2f})" for word, weight in harder) or "none"))
    down.markdown("**Lower the level**  \n" + (", ".join(f"{word} ({weight:+.2f})" for word, weight in easier) or "none"))

    st.divider()
    st.subheader("What if you change it?")
    st.caption("Edit the copy below. Both versions are scored by the same model.")
    edit, compare = st.columns(2, gap="large")
    with edit:
        question = st.session_state.get(f"original_{dataset}_question", example_question)
        options = st.session_state.get(f"original_{dataset}_options", example_options)
        edited_text = question_form(f"edited_{dataset}", dataset, question, options)
    with compare:
        if not edited_text.strip():
            st.info("The edited copy is empty.")
            return
        edited = score(predictor, edited_text)
        change = edited["expected"] - original["expected"]
        first, second = st.columns(2)
        first.metric("Original", f"{original['expected']:.2f}")
        second.metric("Edited", f"{edited['expected']:.2f}",
                      f"{change:+.2f} levels" if edited_text != text else None, delta_color="off")
        st.bar_chart(
            probability_table(names, {"Original": original["proba"], "Edited": edited["proba"]}),
            y_label="Probability", stack=False, height=260, color=["#eb6834", "#2a78d6"],
        )
        if edited_text != text:
            direction = "harder" if change > 0.05 else "easier" if change < -0.05 else "about the same"
            st.write(f"The model reads the edited version as **{direction}** "
                     f"({edited['level_name']}, was {original['level_name']}).")


st.set_page_config(page_title="LevelLens", page_icon="🔎", layout="wide")
st.navigation([
    st.Page(calibrate, title="Calibrate", icon="🔎", default=True),
    st.Page("pages/results.py", title="Results", icon="📊"),
]).run()
