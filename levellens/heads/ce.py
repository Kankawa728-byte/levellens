"""Plain classification head: K logits, softmax, cross-entropy. It ignores the order of the levels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from levellens.heads.base import Head


class CrossEntropyHead(Head):
    """K logits. Predicting a neighbouring level costs as much as predicting a far one."""

    def __init__(self, in_features: int, num_levels: int, class_weights: torch.Tensor | None = None):
        super().__init__(in_features, num_levels, out_features=num_levels, class_weights=class_weights)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), y, weight=self.class_weights)

    def proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=-1)
