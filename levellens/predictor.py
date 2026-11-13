"""One Predictor for the experiments and the app.

    from levellens.predictor import Predictor

    predictor = Predictor.load("stub")                    # no trained model needed
    result = predictor.predict(["What is 2 + 2? A. 3 B. 4 C. 5 D. 6"])
    result["proba"]            # shape [n, K]
    result["level"]            # most likely level of each question
    result["expected_level"]   # probability-weighted level of each question
    result["level_names"]      # names of the K levels

A trained model is saved by a run with `save_model=true` and loaded by its run id:
`Predictor.load("arc-distilbert-corn-s0")`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from levellens.data import info
from levellens.evaluate import interpret
from levellens.models.base import Model, load_class
from levellens.paths import cache_dir

CARD = "predictor.json"  # written next to the saved model: which class to load and what the levels are called
STUB_WORDS_AT_TOP_LEVEL = 60  # the stub reads a question of this many words as the highest level


class StubModel(Model):
    """Made-up probabilities, so the app can be built before any model is trained.

    Longer questions get a higher level, plus a small shift that depends on the text.
    The same text always gets the same answer.
    """

    def predict_proba(self, items) -> np.ndarray:
        """Return made-up probabilities centred on a level that grows with the number of words."""
        levels = np.arange(self.num_levels)
        proba = np.empty((len(items), self.num_levels))
        for row, item in enumerate(items):
            words = len(item["text"].split())
            shift = hashlib.sha1(item["text"].encode("utf-8")).digest()[0] / 255 - 0.5
            centre = min(words / STUB_WORDS_AT_TOP_LEVEL, 1.0) * (self.num_levels - 1) + shift
            weights = np.exp(-0.5 * (levels - centre) ** 2)
            proba[row] = weights / weights.sum()
        return proba


def model_folder(run_id: str) -> Path:
    """Return the folder in which the model of a run is saved."""
    return cache_dir() / "models" / run_id


def save_for_predictor(model: Model, run_id: str, dataset: str) -> Path:
    """Save a trained model so that `Predictor.load(run_id)` finds it. Return the folder."""
    folder = model_folder(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder)
    card = {
        "run_id": run_id,
        "dataset": dataset,
        "level_names": list(info(dataset).level_names),
        "target": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    (folder / CARD).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return folder


class Predictor:
    """Scores question texts with one model."""

    def __init__(self, model: Model, level_names: Sequence[str], name: str = ""):
        """Store the model, the names of its levels and the name of the predictor."""
        self.model = model
        self.level_names = list(level_names)
        self.name = name

    @classmethod
    def load(cls, name: str = "stub", dataset: str = "race") -> "Predictor":
        """Load a predictor.

        `name` is "stub", a run id, or the path of a saved model folder.
        `dataset` only matters for the stub: it sets the level names.
        """
        if name == "stub":
            levels = info(dataset).level_names
            return cls(StubModel(num_levels=len(levels)), levels, name="stub")
        folder = Path(name) if Path(name).is_dir() else model_folder(name)
        if not (folder / CARD).exists():
            raise FileNotFoundError(f"No saved model at {folder}. Train it with save_model=true, or use 'stub'.")
        card = json.loads((folder / CARD).read_text(encoding="utf-8"))
        return cls(load_class(card["target"]).load(folder), card["level_names"], name=card["run_id"])

    def predict(self, questions: Sequence[str | dict]) -> dict:
        """Score questions. Each one is a text (question plus options) or an item dict with a `text` field."""
        items = []
        for index, question in enumerate(questions):
            if isinstance(question, str):
                question = {"id": f"q{index}", "text": question, "question": question, "options": [], "meta": {}}
            items.append(question)
        level, proba, score = interpret(self.model.predict_proba(items), len(self.level_names))
        return {
            "proba": proba,
            "level": level,
            "expected_level": proba @ np.arange(len(self.level_names)),
            "score": score,
            "level_names": self.level_names,
        }
