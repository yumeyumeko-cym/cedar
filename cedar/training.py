"""Training loops and vMF mixture EM for CEDAR."""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import CedarConfig
from .model import LabelAwareBatchSampler
from .scoring import cluster_diagnostics


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weak_structure_loss(z: torch.Tensor, weak_labels: torch.Tensor, tau_weak: float) -> torch.Tensor:
    """Anchor-balanced supervised-contrastive loss induced by weak labels."""

    if z.ndim != 2:
        raise ValueError("z must be 2D (batch, embedding_dim)")
    if weak_labels.ndim != 1 or weak_labels.shape[0] != z.shape[0]:
        raise ValueError("weak_labels must be a 1D tensor aligned with z")
    if tau_weak <= 0:
        raise ValueError("tau_weak must be positive")

    z = F.normalize(z, p=2, dim=1, eps=1e-12)
    logits = (z @ z.T) / tau_weak

    batch_size = logits.shape[0]
    logits_mask = torch.ones((batch_size, batch_size), device=logits.device, dtype=logits.dtype)
    logits_mask.fill_diagonal_(0.0)
    logits = logits.masked_fill(logits_mask == 0, float("-inf"))

    labels = weak_labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(dtype=logits.dtype) * logits_mask

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = positive_mask.sum(dim=1)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(positives, min=1.0)

    valid = positives > 0
    if not torch.any(valid):
        return z.sum() * 0.0
    return -mean_log_prob_pos[valid].mean()


def vmf_Q_loss(z_batch: torch.Tensor, r_batch: torch.Tensor, mu: torch.Tensor, kappa: float) -> torch.Tensor:
    """Negative reduced M-step objective for the vMF-mixture channel."""

    z_batch = F.normalize(z_batch, dim=1, eps=1e-12)
    mu = F.normalize(mu, dim=1, eps=1e-12)
    sim = torch.matmul(z_batch, mu.T)
    return -(r_batch * (kappa * sim)).sum(dim=1).mean()


def _build_embedding_loader(
    X: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )


def _label_aware_loader(X: np.ndarray, weak_labels: np.ndarray, cfg: CedarConfig) -> DataLoader:
    steps_per_epoch = max(1, math.ceil(len(weak_labels) / max(1, cfg.batch_size)))
    sampler = LabelAwareBatchSampler(
        labels=weak_labels,
        batch_size=max(2, cfg.batch_size),
        num_views=2,
        steps_per_epoch=steps_per_epoch,
        seed=cfg.seed,
    )
    idx_tensor = torch.arange(len(weak_labels), dtype=torch.long)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(weak_labels, dtype=torch.long),
        idx_tensor,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.num_workers > 0,
    )


@torch.no_grad()
def compute_embeddings(encoder: nn.Module, X: np.ndarray, batch_size: int, device: str) -> torch.Tensor:
    """Encode a full matrix into normalized embeddings."""

    encoder.eval()
    encoder.to(device)
    chunks: List[torch.Tensor] = []
    for (xb,) in _build_embedding_loader(X, batch_size, num_workers=0, shuffle=False):
        xb = xb.to(device)
        chunks.append(encoder(xb))
    Z = torch.cat(chunks, dim=0)
    return F.normalize(Z, dim=1, eps=1e-12)


def warmup_phase(encoder: nn.Module, X: np.ndarray, weak_labels: np.ndarray, cfg: CedarConfig) -> None:
    """Warm up the encoder using only the weak-structure objective."""

    encoder.to(cfg.device)
    loader = _label_aware_loader(X, weak_labels, cfg)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.warmup_epochs):
        encoder.train()
        for xb, yb, _ in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            z = encoder(xb)
            loss = weak_structure_loss(z, yb, cfg.tau_weak)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
            opt.step()


def _spherical_kmeanspp_init(Z: torch.Tensor, k: int) -> torch.Tensor:
    """k-means++ initialization on the unit sphere using cosine distance."""

    if Z.shape[0] < k:
        raise ValueError(f"Need at least k={k} samples for initialization")

    Z = F.normalize(Z, dim=1, eps=1e-12)
    centers = torch.empty((k, Z.shape[1]), device=Z.device, dtype=Z.dtype)

    first = torch.randint(0, Z.shape[0], (1,), device=Z.device).item()
    centers[0] = Z[first]

    dist = 1.0 - torch.matmul(Z, centers[0].unsqueeze(1)).squeeze(1)
    dist = dist.clamp_min(1e-12)

    for idx in range(1, k):
        prob = dist / dist.sum()
        next_idx = torch.multinomial(prob, 1).item()
        centers[idx] = Z[next_idx]
        new_dist = 1.0 - torch.matmul(Z, centers[idx].unsqueeze(1)).squeeze(1)
        dist = torch.minimum(dist, new_dist.clamp_min(1e-12))

    return F.normalize(centers, dim=1, eps=1e-12)


def _spherical_lloyd_refine(Z: torch.Tensor, mu: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Run a few Lloyd steps to spread prototypes before CEDAR."""

    if steps <= 0:
        return F.normalize(mu, dim=1, eps=1e-12)

    Z = F.normalize(Z, dim=1, eps=1e-12)
    mu = F.normalize(mu, dim=1, eps=1e-12)
    num_points = Z.shape[0]
    num_clusters = mu.shape[0]

    for _ in range(steps):
        sim = torch.matmul(Z, mu.T)
        assign = torch.argmax(sim, dim=1)
        new_mu = torch.zeros_like(mu)
        new_mu.index_add_(0, assign, Z)
        counts = torch.bincount(assign, minlength=num_clusters)
        empty = counts == 0
        if empty.any():
            rand_idx = torch.randint(0, num_points, (int(empty.sum().item()),), device=Z.device)
            new_mu[empty] = Z[rand_idx]
        mu = F.normalize(new_mu, dim=1, eps=1e-12)

    return mu


def _init_score(Z: torch.Tensor, mu: torch.Tensor, kappa: float) -> float:
    """Score initialization by the mean vMF Q value with uniform weights."""

    Z = F.normalize(Z, dim=1, eps=1e-12)
    mu = F.normalize(mu, dim=1, eps=1e-12)
    logits = kappa * torch.matmul(Z, mu.T)
    r = F.softmax(logits, dim=1)
    return float((r * logits).sum(dim=1).mean().item())


def select_init_mu(Z: torch.Tensor, cfg: CedarConfig) -> torch.Tensor:
    """Pick the best spherical k-means++ initialization over multiple restarts."""

    restarts = max(1, int(cfg.init_restarts))
    best_mu = None
    best_score = -float("inf")
    for _ in range(restarts):
        mu = _spherical_kmeanspp_init(Z, cfg.num_clusters)
        mu = _spherical_lloyd_refine(Z, mu, steps=3)
        score = _init_score(Z, mu, cfg.kappa)
        if score > best_score:
            best_score = score
            best_mu = mu
    return best_mu


def e_step_vmf(
    Z: torch.Tensor,
    mu: torch.Tensor,
    pi: torch.Tensor,
    kappa: float,
    pi_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute responsibilities and smoothed mixing weights."""

    Z = F.normalize(Z, dim=1, eps=1e-12)
    mu = F.normalize(mu, dim=1, eps=1e-12)
    log_pi = torch.log(pi.clamp_min(1e-12))
    scores = kappa * torch.matmul(Z, mu.T) + log_pi
    r = F.softmax(scores, dim=1)
    pi_new = r.mean(dim=0)
    if pi_smoothing > 0:
        k = pi_new.numel()
        pi_new = (1.0 - pi_smoothing) * pi_new + pi_smoothing / k
    pi_new = pi_new / pi_new.sum().clamp_min(1e-12)
    return r, pi_new


def update_mu(
    Z: torch.Tensor,
    r_all: torch.Tensor,
    cfg: CedarConfig,
    mu_prev: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Update prototypes and optionally reseed low-mass components."""

    Z = F.normalize(Z, dim=1, eps=1e-12)
    weighted_mu = torch.matmul(r_all.T, Z)
    mu = F.normalize(weighted_mu, dim=1, eps=1e-12)
    expected_counts = r_all.sum(dim=0)

    reseeded = 0
    if cfg.reseed_low_mass:
        low_mass = expected_counts < cfg.min_expected_count
        if low_mass.any():
            live_mask = ~low_mass
            if live_mask.any():
                live_mu = mu[live_mask]
                coverage = torch.matmul(Z, live_mu.T).max(dim=1).values
            elif mu_prev is not None:
                coverage = torch.matmul(Z, F.normalize(mu_prev, dim=1, eps=1e-12).T).max(dim=1).values
            else:
                coverage = torch.zeros(Z.shape[0], device=Z.device, dtype=Z.dtype)

            candidate_order = torch.argsort(coverage, descending=False)
            used_candidates = set()
            for cluster_idx in torch.nonzero(low_mass, as_tuple=False).view(-1).tolist():
                for candidate_idx in candidate_order.tolist():
                    if candidate_idx not in used_candidates:
                        mu[cluster_idx] = Z[candidate_idx]
                        used_candidates.add(candidate_idx)
                        reseeded += 1
                        break

    mu = F.normalize(mu, dim=1, eps=1e-12)
    stats = {
        "reseeded_components": float(reseeded),
        "min_expected_count": float(expected_counts.min().item()),
        "max_expected_count": float(expected_counts.max().item()),
    }
    return mu, stats


def fit_vmf_mixture_fixed_embeddings(
    Z: torch.Tensor | np.ndarray,
    cfg: CedarConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """Fit a vMF mixture on fixed embeddings without updating the encoder."""

    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.is_tensor(Z):
        Z = Z.detach().to(device)
    else:
        Z = torch.tensor(np.asarray(Z), dtype=torch.float32, device=device)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D tensor or array of shape (n_samples, embedding_dim)")
    if Z.shape[0] < cfg.num_clusters:
        raise ValueError(f"Need at least {cfg.num_clusters} samples to fit the vMF mixture")

    history: List[Dict[str, float]] = []
    with torch.no_grad():
        Z = F.normalize(Z, dim=1, eps=1e-12)
        mu = select_init_mu(Z, cfg)
        pi = torch.full((cfg.num_clusters,), 1.0 / cfg.num_clusters, device=device)

        for em_iter in range(cfg.em_iters):
            r_all, pi = e_step_vmf(Z, mu, pi, cfg.kappa, cfg.pi_smoothing)
            pi = torch.clamp(pi, min=cfg.min_component_weight)
            pi = pi / pi.sum().clamp_min(1e-12)
            mu, mu_stats = update_mu(Z, r_all, cfg, mu_prev=mu)
            if int(mu_stats["reseeded_components"]) > 0:
                r_all, pi = e_step_vmf(Z, mu, pi, cfg.kappa, cfg.pi_smoothing)
                pi = torch.clamp(pi, min=cfg.min_component_weight)
                pi = pi / pi.sum().clamp_min(1e-12)

            mean_q = float((r_all * (cfg.kappa * torch.matmul(Z, mu.T))).sum(dim=1).mean().item())
            diag = cluster_diagnostics(pi, r_all=r_all)
            history.append(
                {
                    "em_iter": float(em_iter),
                    "mean_q": mean_q,
                    **diag,
                    **mu_stats,
                }
            )
            print(
                "[vMF-fixed {iter}] mean_Q={mean_q:.4f} eff_k={eff:.2f} "
                "pi_min={pi_min:.4g} pi_max={pi_max:.4g} reseeded={reseeded:.0f}".format(
                    iter=em_iter,
                    mean_q=mean_q,
                    eff=diag["effective_clusters"],
                    pi_min=diag["pi_min"],
                    pi_max=diag["pi_max"],
                    reseeded=mu_stats["reseeded_components"],
                )
            )

        r_all, pi = e_step_vmf(Z, mu, pi, cfg.kappa, cfg.pi_smoothing)
        pi = torch.clamp(pi, min=cfg.min_component_weight)
        pi = pi / pi.sum().clamp_min(1e-12)
        mu, mu_stats = update_mu(Z, r_all, cfg, mu_prev=mu)
        final_diag = cluster_diagnostics(pi, r_all=r_all)
        history.append({"em_iter": float(cfg.em_iters), **final_diag, **mu_stats})

    return mu.detach(), pi.detach(), history


def composite_cedar_training_phase(
    encoder: nn.Module,
    X: np.ndarray,
    weak_labels: np.ndarray,
    cfg: CedarConfig,
    on_em_iter: Optional[Callable[[int, nn.Module], None]] = None,
    skip_warmup: bool = False,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """Train the encoder and vMF mixture with the composite CEDAR objective."""

    device = cfg.device
    encoder.to(device)

    if not skip_warmup:
        warmup_phase(encoder, X, weak_labels, cfg)

    with torch.no_grad():
        Z_full = compute_embeddings(encoder, X, cfg.batch_size, device)
        Z_full = F.normalize(Z_full, dim=1, eps=1e-12)
        mu = select_init_mu(Z_full, cfg)
    pi = torch.full((cfg.num_clusters,), 1.0 / cfg.num_clusters, device=device)

    loader = _label_aware_loader(X, weak_labels, cfg)
    history: List[Dict[str, float]] = []
    best_snapshot = {
        "loss": float("inf"),
        "encoder_state": None,
        "mu": None,
        "pi": None,
    }

    for em_iter in range(cfg.em_iters):
        with torch.no_grad():
            Z_full = compute_embeddings(encoder, X, cfg.batch_size, device)
            Z_full = F.normalize(Z_full, dim=1, eps=1e-12)
            r_all, pi = e_step_vmf(Z_full, mu, pi, cfg.kappa, cfg.pi_smoothing)
            pi = torch.clamp(pi, min=cfg.min_component_weight)
            pi = pi / pi.sum().clamp_min(1e-12)
            mu, mu_stats = update_mu(Z_full, r_all, cfg, mu_prev=mu)
            if int(mu_stats["reseeded_components"]) > 0:
                r_all, pi = e_step_vmf(Z_full, mu, pi, cfg.kappa, cfg.pi_smoothing)
                pi = torch.clamp(pi, min=cfg.min_component_weight)
                pi = pi / pi.sum().clamp_min(1e-12)

            mean_q = float((r_all * (cfg.kappa * torch.matmul(Z_full, mu.T))).sum(dim=1).mean().item())
            diag = cluster_diagnostics(pi, r_all=r_all)
            iter_stats = {
                "em_iter": float(em_iter),
                "mean_q": mean_q,
                **diag,
                **mu_stats,
            }
            history.append(iter_stats)
            print(
                "[CEDAR {iter}] mean_Q={mean_q:.4f} eff_k={eff:.2f} "
                "pi_min={pi_min:.4g} pi_max={pi_max:.4g} reseeded={reseeded:.0f}".format(
                    iter=em_iter,
                    mean_q=mean_q,
                    eff=diag["effective_clusters"],
                    pi_min=diag["pi_min"],
                    pi_max=diag["pi_max"],
                    reseeded=mu_stats["reseeded_components"],
                )
            )

        opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for _ in range(cfg.encoder_epochs):
            encoder.train()
            epoch_total = 0.0
            epoch_mix = 0.0
            epoch_weak = 0.0
            batches = 0

            for xb, yb, idxb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                idxb = idxb.to(device)

                opt.zero_grad(set_to_none=True)
                z = encoder(xb)
                r_batch = r_all[idxb].detach()
                mix_loss = vmf_Q_loss(z, r_batch, mu, cfg.kappa)
                weak_loss = weak_structure_loss(z, yb, cfg.tau_weak)
                total_loss = mix_loss + cfg.lambda_weak * weak_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
                opt.step()

                epoch_total += float(total_loss.item())
                epoch_mix += float(mix_loss.item())
                epoch_weak += float(weak_loss.item())
                batches += 1

            mean_total = epoch_total / max(1, batches)
            mean_mix = epoch_mix / max(1, batches)
            mean_weak = epoch_weak / max(1, batches)
            if mean_total < best_snapshot["loss"]:
                best_snapshot["loss"] = mean_total
                best_snapshot["encoder_state"] = {
                    key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
                }
                best_snapshot["mu"] = mu.detach().cpu().clone()
                best_snapshot["pi"] = pi.detach().cpu().clone()
            history.append(
                {
                    "em_iter": float(em_iter),
                    "encoder_total_loss": mean_total,
                    "encoder_mix_loss": mean_mix,
                    "encoder_weak_loss": mean_weak,
                }
            )
        if on_em_iter is not None:
            on_em_iter(em_iter, encoder)

    if best_snapshot["encoder_state"] is not None:
        state_on_device = {key: value.to(device) for key, value in best_snapshot["encoder_state"].items()}
        encoder.load_state_dict(state_on_device)
        encoder.eval()
        mu = best_snapshot["mu"].to(device)
        pi = best_snapshot["pi"].to(device)

    with torch.no_grad():
        Z_full = compute_embeddings(encoder, X, cfg.batch_size, device)
        Z_full = F.normalize(Z_full, dim=1, eps=1e-12)
        r_all, pi = e_step_vmf(Z_full, mu, pi, cfg.kappa, cfg.pi_smoothing)
        pi = torch.clamp(pi, min=cfg.min_component_weight)
        pi = pi / pi.sum().clamp_min(1e-12)
        mu, mu_stats = update_mu(Z_full, r_all, cfg, mu_prev=mu)
        final_diag = cluster_diagnostics(pi, r_all=r_all)
        history.append({"em_iter": float(cfg.em_iters), **final_diag, **mu_stats})

    return encoder, mu.detach(), pi.detach(), history
