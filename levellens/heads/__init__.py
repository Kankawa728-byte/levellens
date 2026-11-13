"""Output heads: the last layer of a neural model, its loss and its probabilities.

A head sits on top of an encoder (BiLSTM or transformer) that gives one vector per question.

    head = build_head(config.head, in_features=768, num_levels=5, class_weights=weights)
    logits = head(features)            # [n, head.out_features]
    loss = head.loss(logits, y)        # scalar
    proba = head.proba(logits)         # [n, K], every row sums to 1

Each head has a file in `configs/head/` that names its class:

    # configs/head/corn.yaml
    name: corn
    target: levellens.heads.corn.CornHead
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from levellens.heads.base import Head
from levellens.heads.ce import CrossEntropyHead
from levellens.heads.corn import CornHead
from levellens.models.base import load_class

__all__ = ["Head", "CrossEntropyHead", "CornHead", "build_head", "class_weights"]


def build_head(head_config: DictConfig, in_features: int, num_levels: int, class_weights: torch.Tensor | None = None) -> Head:
    """Build the head named in `head_config.target`."""
    if "target" not in head_config:
        raise ValueError(f"head={head_config.name} has no output layer. A neural model needs head=ce or head=corn.")
    return load_class(head_config.target)(in_features, num_levels, class_weights=class_weights)


def class_weights(levels, num_levels: int) -> torch.Tensor:
    """Return one weight per level, larger for rarer levels: n / (K * count), as scikit-learn's "balanced".

    With these weights every level contributes the same total to the loss.
    A level with no training question gets weight 0.
    """
    counts = torch.bincount(torch.as_tensor(levels, dtype=torch.long), minlength=num_levels).float()
    weights = counts.sum() / (num_levels * counts.clamp(min=1))
    return torch.where(counts > 0, weights, torch.zeros_like(weights))
