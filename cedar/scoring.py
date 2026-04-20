"""Anomaly scoring and evaluation metrics for CEDAR."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def cluster_diagnostics(pi: torch.Tensor, r_all: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Return simple occupancy diagnostics useful for collapse monitoring."""

    pi = pi / pi.sum().clamp_min(1e-12)
    entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum()
    effective_clusters = torch.exp(entropy)
    stats = {
        "pi_min": float(pi.min().item()),
        "pi_max": float(pi.max().item()),
        "pi_entropy": float(entropy.item()),
        "effective_clusters": float(effective_clusters.item()),
    }
    if r_all is not None:
        expected_counts = r_all.sum(dim=0)
        stats["expected_count_min"] = float(expected_counts.min().item())
        stats["expected_count_max"] = float(expected_counts.max().item())
    return stats


def score_samples_vmf(Z: torch.Tensor, mu: torch.Tensor, pi: torch.Tensor, kappa: float) -> np.ndarray:
    """Score samples by negative mixture log-probability."""

    Z = F.normalize(Z, dim=1, eps=1e-12)
    mu = F.normalize(mu, dim=1, eps=1e-12)
    log_pi = torch.log(pi.clamp_min(1e-12))
    logits = kappa * torch.matmul(Z, mu.T) + log_pi
    log_prob = torch.logsumexp(logits, dim=1)
    return (-log_prob).detach().cpu().numpy()


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, f1_percentile: float) -> Dict[str, float]:
    """Compute AUROC, AUPR, and point metrics at a score percentile threshold."""

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    thr = np.percentile(scores, f1_percentile)
    y_pred = (scores >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = f1_score(y_true, y_pred)

    return {
        "AUROC": float(auroc),
        "AUPR": float(aupr),
        "ACC": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        f"F1@{f1_percentile:.2f}": float(f1),
    }


def compute_f1_at_percentile(scores: np.ndarray, y_true: np.ndarray, f1_percentile: float) -> float:
    thr = np.percentile(scores, f1_percentile)
    y_pred = (scores >= thr).astype(int)
    return float(f1_score(y_true, y_pred))


def compute_f1_at_top_percent(scores: np.ndarray, y_true: np.ndarray, top_percent: float) -> float:
    if not 0.0 < top_percent < 100.0:
        raise ValueError("top_percent must be in (0, 100)")
    percentile = 100.0 - top_percent
    return compute_f1_at_percentile(scores, y_true, percentile)
