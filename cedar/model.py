"""Encoder and weak-label positive-pair sampler for CEDAR."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CedarEncoder(nn.Module):
    """MLP encoder mapping inputs to a unit-sphere embedding."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(input_dim, 4 * embedding_dim)

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden

        self.backbone = nn.Sequential(*layers)
        self.embedding = nn.Linear(hidden, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        pre_z = self.embedding(h)
        z = F.normalize(pre_z, p=2, dim=1, eps=1e-12)
        return z


class PositivePairSampler:
    """Sample dataset indices that share an anchor's weak (edge) label.

    For each anchor, returns ``num_positives`` indices drawn (without replacement
    when possible) from the set of flows carrying the same edge label, excluding
    the anchor itself. Anchors whose label has no other member in the dataset
    are reported as invalid and the loss is expected to skip them.
    """

    def __init__(self, weak_labels: np.ndarray, num_positives: int, seed: int):
        if num_positives < 1:
            raise ValueError("num_positives must be >= 1")
        self.labels = np.asarray(weak_labels, dtype=np.int64)
        if self.labels.ndim != 1 or self.labels.size == 0:
            raise ValueError("weak_labels must be a non-empty 1D array")
        self.num_positives = int(num_positives)
        self.label_to_indices = {
            int(label): np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self._rng = np.random.default_rng(int(seed))

    def sample(
        self,
        anchor_indices: np.ndarray,
        anchor_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(partner_indices, valid_mask)`` of shapes ``(B, C)`` and ``(B,)``."""

        anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
        anchor_labels = np.asarray(anchor_labels, dtype=np.int64)
        batch = anchor_indices.shape[0]
        partners = np.zeros((batch, self.num_positives), dtype=np.int64)
        valid = np.ones(batch, dtype=bool)

        for i in range(batch):
            pool = self.label_to_indices.get(int(anchor_labels[i]))
            if pool is None or pool.size <= 1:
                valid[i] = False
                partners[i] = int(anchor_indices[i])
                continue
            candidates = pool[pool != int(anchor_indices[i])]
            replace = candidates.size < self.num_positives
            partners[i] = self._rng.choice(candidates, size=self.num_positives, replace=replace)

        return partners, valid
