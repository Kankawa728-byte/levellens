"""The contract of an output head."""

from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    """One linear layer on the question vector, with the loss and the probabilities that go with it.

    `class_weights` is a tensor of shape [K] or None. A question of level k counts
    `class_weights[k]` times in the loss.
    """

    def __init__(self, in_features: int, num_levels: int, out_features: int, class_weights: torch.Tensor | None = None):
        """Build the linear layer and store the class weights."""
        super().__init__()
        self.num_levels = num_levels
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        # a buffer moves to the GPU with the model, but is not trained
        self.register_buffer("class_weights", None if class_weights is None else class_weights.float(), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the logits, shape [n, out_features]."""
        return self.linear(features)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the training loss (a scalar) for true levels `y`, shape [n]."""
        raise NotImplementedError

    def proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the probability of each level, shape [n, K]. Every row sums to 1."""
        raise NotImplementedError
