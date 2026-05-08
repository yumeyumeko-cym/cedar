"""Training loops and vMF mixture EM for CEDAR."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import CedarConfig
from .model import PositivePairSampler
from .scoring import cluster_diagnostics


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_supcon_loss(
    z_anchor: torch.Tensor,
    z_partners: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """SimCLR-style InfoNCE where each anchor's positives are sampled same-edge partners.

    ``z_anchor`` are the batch embeddings ``(B, D)``. ``z_partners`` is shape
    ``(B, C, D)`` where partner ``c`` of anchor ``i`` was sampled from the full
    dataset (excluding the anchor itself) by matching weak label. ``valid`` is a
    bool mask ``(B,)`` flagging anchors whose label has at least one other flow
    in the dataset; anchors marked invalid are skipped from the loss.

    Negatives for anchor ``i`` are every other anchor and every partner that
    isn't one of anchor ``i``'s sampled positives.
    """

    if z_anchor.ndim != 2:
        raise ValueError("z_anchor must be 2D (B, D)")
    if z_partners.ndim != 3 or z_partners.shape[0] != z_anchor.shape[0] or z_partners.shape[2] != z_anchor.shape[1]:
        raise ValueError("z_partners must be (B, C, D) aligned with z_anchor")
    if tau <= 0:
        raise ValueError("tau must be positive")

    B, C, D = z_partners.shape
    z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-12)
    z_partners = F.normalize(z_partners, p=2, dim=2, eps=1e-12)

    keys = torch.cat([z_anchor, z_partners.reshape(B * C, D)], dim=0)  # (B + B*C, D)
    logits = (z_anchor @ keys.T) / tau  # (B, B + B*C)

    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_idx = torch.arange(B, device=logits.device)
    self_mask[self_idx, self_idx] = True

    pos_mask = torch.zeros_like(logits, dtype=torch.bool)
    pos_idx = B + torch.arange(B * C, device=logits.device).view(B, C)
    pos_mask.scatter_(1, pos_idx, True)

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_count = pos_mask.to(log_prob.dtype).sum(dim=1).clamp(min=1.0)
    loss_per = -masked_log_prob.sum(dim=1) / pos_count

    if not torch.any(valid):
        return z_anchor.sum() * 0.0
    return loss_per[valid].mean()


def vmf_Q_loss(z_batch: torch.Tensor, r_batch: torch.Tensor, mu: torch.Tensor, kappa: float) -> torch.Tensor:
    """Negative objective for the vMF-mixture channel."""

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


def _simple_loader(X: np.ndarray, weak_labels: np.ndarray, cfg: CedarConfig, shuffle: bool = True) -> DataLoader:
    idx_tensor = torch.arange(len(weak_labels), dtype=torch.long)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(weak_labels, dtype=torch.long),
        idx_tensor,
    )
    return DataLoader(
        dataset,
        batch_size=max(2, cfg.batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.num_workers > 0,
    )


def _encode_anchor_and_partners(
    encoder: nn.Module,
    xb: torch.Tensor,
    X: np.ndarray,
    partner_indices: np.ndarray,
    device: str,
    num_positives: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward anchors and their sampled partners through the encoder in one pass."""

    flat_partner_idx = partner_indices.reshape(-1)
    partner_x = torch.from_numpy(np.ascontiguousarray(X[flat_partner_idx])).float().to(device)
    all_x = torch.cat([xb, partner_x], dim=0)
    all_z = encoder(all_x)
    batch = xb.shape[0]
    z_anchor = all_z[:batch]
    z_partners = all_z[batch:].view(batch, num_positives, -1)
    return z_anchor, z_partners


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


def warmup_phase(
    encoder: nn.Module,
    X: np.ndarray,
    weak_labels: np.ndarray,
    cfg: CedarConfig,
    sampler: Optional[PositivePairSampler] = None,
) -> PositivePairSampler:
    """Warm up the encoder with the partner-pair supervised contrastive objective."""

    encoder.to(cfg.device)
    if sampler is None:
        sampler = PositivePairSampler(weak_labels, cfg.num_positives, cfg.seed)

    loader = _simple_loader(X, weak_labels, cfg, shuffle=True)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.warmup_epochs):
        encoder.train()
        for xb, yb, idxb in loader:
            partner_idx, valid_np = sampler.sample(idxb.numpy(), yb.numpy())
            xb = xb.to(cfg.device)
            valid = torch.from_numpy(valid_np).to(cfg.device)

            opt.zero_grad(set_to_none=True)
            z_anchor, z_partners = _encode_anchor_and_partners(
                encoder, xb, X, partner_idx, cfg.device, cfg.num_positives,
            )
            loss = pair_supcon_loss(z_anchor, z_partners, valid, cfg.tau_weak)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
            opt.step()
    return sampler


def _spherical_kmeanspp_init(Z: torch.Tensor, k: int) -> torch.Tensor:
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
    """Update vMF prototypes from soft responsibilities.

    Components whose weighted-sum direction has effectively zero norm retain
    their previous ``mu_prev`` row when one is supplied.
    """

    del cfg
    Z = F.normalize(Z, dim=1, eps=1e-12)
    weighted_mu = torch.matmul(r_all.T, Z)
    mu = F.normalize(weighted_mu, dim=1, eps=1e-12)

    if mu_prev is not None:
        bad = weighted_mu.norm(dim=1) < 1e-12
        if torch.any(bad):
            mu_prev_n = F.normalize(mu_prev, dim=1, eps=1e-12)
            mu = torch.where(bad.unsqueeze(1), mu_prev_n, mu)

    expected_counts = r_all.sum(dim=0)
    stats = {
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
                "pi_min={pi_min:.4g} pi_max={pi_max:.4g}".format(
                    iter=em_iter,
                    mean_q=mean_q,
                    eff=diag["effective_clusters"],
                    pi_min=diag["pi_min"],
                    pi_max=diag["pi_max"],
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

    sampler = PositivePairSampler(weak_labels, cfg.num_positives, cfg.seed)
    if not skip_warmup:
        warmup_phase(encoder, X, weak_labels, cfg, sampler=sampler)

    with torch.no_grad():
        Z_full = compute_embeddings(encoder, X, cfg.batch_size, device)
        Z_full = F.normalize(Z_full, dim=1, eps=1e-12)
        mu = select_init_mu(Z_full, cfg)
    pi = torch.full((cfg.num_clusters,), 1.0 / cfg.num_clusters, device=device)

    loader = _simple_loader(X, weak_labels, cfg, shuffle=True)
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
                "pi_min={pi_min:.4g} pi_max={pi_max:.4g}".format(
                    iter=em_iter,
                    mean_q=mean_q,
                    eff=diag["effective_clusters"],
                    pi_min=diag["pi_min"],
                    pi_max=diag["pi_max"],
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
                partner_idx, valid_np = sampler.sample(idxb.numpy(), yb.numpy())
                xb = xb.to(device)
                idxb_dev = idxb.to(device)
                valid = torch.from_numpy(valid_np).to(device)

                opt.zero_grad(set_to_none=True)
                z_anchor, z_partners = _encode_anchor_and_partners(
                    encoder, xb, X, partner_idx, device, cfg.num_positives,
                )
                r_batch = r_all[idxb_dev].detach()
                mix_loss = vmf_Q_loss(z_anchor, r_batch, mu, cfg.kappa)
                weak_loss = pair_supcon_loss(z_anchor, z_partners, valid, cfg.tau_weak)
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
