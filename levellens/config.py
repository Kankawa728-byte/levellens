"""Build the config of one run from the YAML files and the command line.

    python -m levellens.run data=arc model=majority head=none seed=1 model.params.alpha=0.5

- `data=`, `model=` and `head=` each pick one file: `configs/data/arc.yaml`,
  `configs/model/majority.yaml`, `configs/head/none.yaml`.
- Every other `key=value` overrides one setting after the files are loaded.
  Dots go into nested settings: `model.params.alpha=0.5`.
- `configs/train.yaml` holds the settings shared by every run and the default picks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from levellens.paths import config_dir

GROUPS = ("data", "model", "head")


def compose(overrides: Sequence[str] = (), folder: Path | None = None) -> DictConfig:
    """Return the config of one run.

    `overrides` are `key=value` strings, as typed on the command line.
    `folder` is the config folder; the default is `configs/` in the repository.
    """
    folder = Path(folder) if folder else config_dir()
    config = OmegaConf.load(folder / "train.yaml")
    picks = dict(config.pop("defaults"))

    settings = []
    for override in overrides:
        key, equals, value = override.partition("=")
        if not equals or not key:
            raise ValueError(f"Expected key=value, got {override!r}")
        if key in GROUPS:
            picks[key] = value
        else:
            settings.append(override)

    for group in GROUPS:
        path = folder / group / f"{picks[group]}.yaml"
        if not path.exists():
            known = sorted(p.stem for p in (folder / group).glob("*.yaml"))
            raise FileNotFoundError(f"No {group} config named {picks[group]!r}. Available: {known}")
        node = OmegaConf.load(path)
        if "name" not in node:
            node.name = picks[group]
        config[group] = node

    return OmegaConf.merge(config, OmegaConf.from_dotlist(settings))


def to_dict(config: DictConfig) -> dict:
    """Return the config as plain Python containers, ready for JSON."""
    return OmegaConf.to_container(config, resolve=True)
