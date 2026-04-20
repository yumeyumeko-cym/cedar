"""Configuration dataclass for CEDAR training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CedarConfig:
    # Model
    embedding_dim: int = 128
    hidden_dim: Optional[int] = None
    dropout: float = 0.1

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Weak-structure channel
    warmup_epochs: int = 40
    tau_weak: float = 0.1
    lambda_weak: float = 1.0

    # CEDAR loop
    num_clusters: int = 20
    em_iters: int = 8
    encoder_epochs: int = 8
    kappa: float = 50.0
    init_restarts: int = 1

    # Anti-collapse safeguards
    pi_smoothing: float = 0.01
    min_component_weight: float = 1e-3
    min_expected_count: float = 8.0
    reseed_low_mass: bool = True

    # Data
    batch_size: int = 512
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 13
