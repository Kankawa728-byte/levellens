"""Floors: the scores that any real model must beat.

Only the majority class is here, as the worked example of the model contract.
Share D writes the stratified random floor and the subject prior in its own module (task D1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from levellens.models.base import Model


class Majority(Model):
    """Always predict the most common level of the training set.

    On a tie the highest level wins, as in the published baseline code
    (github.com/arthur-thuy/qde-ordinality, `src/trainer/baseline_trainer.py`).
    """

    def fit(self, train, dev, config) -> None:
        counts = np.bincount([item["level"] for item in train], minlength=self.num_levels)
        self.level_ = int(self.num_levels - 1 - np.argmax(counts[::-1]))

    def predict_proba(self, items) -> np.ndarray:
        proba = np.zeros((len(items), self.num_levels))
        proba[:, self.level_] = 1.0
        return proba

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "majority.json").write_text(json.dumps({"num_levels": self.num_levels, "level": self.level_}))

    @classmethod
    def load(cls, folder: Path) -> "Majority":
        state = json.loads((folder / "majority.json").read_text())
        model = cls(num_levels=state["num_levels"])
        model.level_ = state["level"]
        return model
