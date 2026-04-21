"""Encoder and label-aware batch sampler for CEDAR."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class LabelAwareBatchSampler(Sampler[List[int]]):
    """Ensure every batch provides multiple samples per weak label for contrastive learning."""

    def __init__(self,
                 labels: np.ndarray,
                 batch_size: int,
                 num_views: int = 2,
                 steps_per_epoch: Optional[int] = None,
                 seed: Optional[int] = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_views < 1:
            raise ValueError("num_views must be positive")

        self.labels = np.asarray(labels, dtype=np.int64)
        if self.labels.size == 0:
            raise ValueError("labels array must be non-empty")

        self.batch_size = batch_size
        self.num_views = num_views
        self.unique_labels = np.unique(self.labels)
        self.label_to_indices = {
            label: np.flatnonzero(self.labels == label)
            for label in self.unique_labels
        }
        self.steps_per_epoch = steps_per_epoch or max(1, math.ceil(len(self.labels) / self.batch_size))
        base_seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._base_seed = int(base_seed) % (2 ** 32)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self._base_seed + self._epoch)
        self._epoch += 1

        pairs_per_batch = max(1, self.batch_size // self.num_views)
        for _ in range(self.steps_per_epoch):
            batch_indices: List[int] = []

            chosen_labels = rng.choice(self.unique_labels, size=pairs_per_batch, replace=True)
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                replace = len(indices) < self.num_views
                sampled = rng.choice(indices, size=self.num_views, replace=replace)
                batch_indices.extend(int(idx) for idx in sampled.tolist())

            remainder = self.batch_size - len(batch_indices)
            if remainder > 0:
                label = rng.choice(self.unique_labels)
                indices = self.label_to_indices[label]
                replace = len(indices) < remainder
                sampled = rng.choice(indices, size=remainder, replace=replace)
                batch_indices.extend(int(idx) for idx in sampled.tolist())

            yield batch_indices


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
