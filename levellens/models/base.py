"""The contract that every model follows, and the function that builds a model from a config.

To add a model:
1. Write a class with `fit` and `predict_proba` (subclass `Model`).
2. Add `configs/model/<name>.yaml` with the class path and its settings:

       target: levellens.models.classic.TfidfLogReg
       params:
         C: 1.0

3. Run it: `python -m levellens.run data=toy model=<name>`.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf


class Model:
    """Base class. `params` are the settings under `params:` in the model config."""

    def __init__(self, num_levels: int, **params):
        self.num_levels = num_levels
        self.params = params

    def fit(self, train: Sequence[dict], dev: Sequence[dict], config: DictConfig) -> None:
        """Train on `train`. Use `dev` for early stopping and for choosing settings. Never touch test.

        `config` is the whole run config: `config.seed`, `config.device`, `config.head`, `config.model`.
        """
        raise NotImplementedError

    def predict_proba(self, items: Sequence[dict]) -> np.ndarray:
        """Return an array of shape [n, K] whose rows sum to 1.

        A model that predicts one number per question (regression, R2DE) returns shape [n] instead.
        """
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        """Number of trained parameters, for the efficiency table. 0 if it does not apply."""
        return 0

    def save(self, folder: Path) -> None:
        """Write everything `load` needs into `folder`."""
        raise NotImplementedError(f"{type(self).__name__} has no save() yet")

    @classmethod
    def load(cls, folder: Path) -> "Model":
        """Return the model that `save` wrote into `folder`."""
        raise NotImplementedError(f"{cls.__name__} has no load() yet")


def texts(items: Sequence[dict]) -> list[str]:
    """Return the model input of each item."""
    return [item["text"] for item in items]


def load_class(target: str) -> type:
    """Return the class named by a dotted path, for example `levellens.models.floors.Majority`."""
    module_name, _, class_name = target.rpartition(".")
    return getattr(import_module(module_name), class_name)


def build_model(config: DictConfig, num_levels: int) -> Model:
    """Build the model named in `config.model.target` with the settings in `config.model.params`."""
    params = OmegaConf.to_container(config.model.get("params") or OmegaConf.create({}), resolve=True)
    return load_class(config.model.target)(num_levels=num_levels, **params)
