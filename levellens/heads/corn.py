"""CORN head: conditional ordinal regression (Shi, Cao and Raschka, 2023, arXiv:2111.08851).

The K-way problem is split into K-1 yes/no questions that follow the order of the levels:

    task 0:  is the level above 0?
    task k:  is the level above k, given that it is above k-1?

Task k is trained only on the questions whose level is above k-1, so its sigmoid output is the
conditional probability P(y > k | y > k-1). Multiplying them along the chain gives

    P(y > k) = P(y > 0) * P(y > 1 | y > 0) * ... * P(y > k | y > k-1)

which can only fall as k grows (rank consistency), and the probability of each level is the difference
of two neighbours: P(y = k) = P(y > k-1) - P(y > k), with P(y > -1) = 1 and P(y > K-1) = 0.
So the K probabilities are between 0 and 1 and sum to 1 by construction.

Written from the paper; the reference implementation is coral-pytorch (MIT), github.com/Raschka-research-group/coral-pytorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from levellens.heads.base import Head


class CornHead(Head):
    """K-1 logits, one per "is the level above k?" task."""

    def __init__(self, in_features: int, num_levels: int, class_weights: torch.Tensor | None = None):
        """Build a linear layer with K-1 logits, one per task."""
        super().__init__(in_features, num_levels, out_features=num_levels - 1, class_weights=class_weights)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy of every task, counted only on the questions that the task sees."""
        logits = logits.float()
        tasks = torch.arange(self.num_levels - 1, device=y.device)  # [K-1]
        target = (y[:, None] > tasks).float()  # [n, K-1]: is the level above k?
        in_task = (y[:, None] >= tasks).float()  # [n, K-1]: task k only sees levels above k-1
        if self.class_weights is not None:
            in_task = in_task * self.class_weights[y][:, None]
        per_entry = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (per_entry * in_task).sum() / in_task.sum().clamp(min=1e-8)

    def proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Multiply the conditional probabilities along the chain, then take the difference of neighbours."""
        above = torch.cumprod(torch.sigmoid(logits.float()), dim=-1)  # [n, K-1]: P(y > k)
        ones, zeros = torch.ones_like(above[:, :1]), torch.zeros_like(above[:, :1])
        return torch.cat([ones, above], dim=-1) - torch.cat([above, zeros], dim=-1)
