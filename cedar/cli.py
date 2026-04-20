"""CEDAR train + evaluate CLI for the composite weak-structure / vMF mixture objective."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import CedarConfig
from .data import DataBundle, SUPPORTED_DATASETS, load_and_sample_data, resolve_data_dir
from .model import CedarEncoder
from .scoring import (
    compute_f1_at_percentile,
    compute_f1_at_top_percent,
    compute_metrics,
    score_samples_vmf,
)
from .training import (
    composite_cedar_training_phase,
    compute_embeddings,
    set_seed,
)


def _format_run_tag(
    dataset: str,
    cut_off: float,
    seed: int,
    emb_dim: int,
    batch_size: int,
    num_clusters: int,
    tau_weak: float,
    lambda_weak: float,
    kappa: float,
) -> str:
    tag = (
        f"{dataset}_cut{cut_off:.2f}_seed{seed}_dim{emb_dim}_bs{batch_size}_"
        f"k{num_clusters}_tw{tau_weak:.3f}_lw{lambda_weak:.3f}_kappa{kappa:.1f}"
    )
    sanitized = (
        tag.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "m")
        .replace(".", "p")
    )
    return sanitized.lower()


def _build_cfg(
    args,
    emb_dim: int,
    batch_size: int,
    device: str,
    seed: int,
    tau_weak: float,
    lambda_weak: float,
    num_clusters: int,
    kappa: float,
) -> CedarConfig:
    return CedarConfig(
        embedding_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        tau_weak=tau_weak,
        lambda_weak=lambda_weak,
        num_clusters=num_clusters,
        em_iters=args.em_iters,
        encoder_epochs=args.encoder_epochs,
        kappa=kappa,
        init_restarts=args.init_restarts,
        pi_smoothing=args.pi_smoothing,
        min_component_weight=args.min_component_weight,
        min_expected_count=args.min_expected_count,
        reseed_low_mass=not args.disable_reseed,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed,
    )


def _full_numeric_matrix(db: DataBundle) -> np.ndarray:
    X_full = db.merged_df[db.columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return db.scaler.transform(X_full).astype(np.float32)


def _save_scores_parquet(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> Path:
    df = pd.DataFrame({"score": scores, "label": labels.astype(int)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def _save_checkpoint(
    encoder: CedarEncoder,
    mu: torch.Tensor,
    pi: torch.Tensor,
    cfg: CedarConfig,
    history,
    ckpt_path: Path,
) -> None:
    ckpt = {
        "encoder_state_dict": encoder.state_dict(),
        "mu": mu.detach().cpu(),
        "pi": pi.detach().cpu(),
        "cfg": asdict(cfg),
        "history": history,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def _load_checkpoint(
    input_dim: int,
    ckpt_path: Path,
    device: str,
) -> Tuple[CedarEncoder, torch.Tensor, torch.Tensor, CedarConfig, list]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    cfg = CedarConfig(**cfg_dict) if cfg_dict else CedarConfig()
    encoder = CedarEncoder(
        input_dim=input_dim,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    mu = ckpt["mu"].to(device)
    pi = ckpt["pi"].to(device)
    history = ckpt.get("history", [])
    return encoder, mu, pi, cfg, history


def train_single_run(
    db: DataBundle,
    args,
    emb_dim: int,
    batch_size: int,
    seed: int,
    tau_weak: float,
    lambda_weak: float,
    num_clusters: int,
    kappa: float,
    run_tag: str,
    ckpt_dir: Path,
) -> Path:
    set_seed(seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args, emb_dim, batch_size, device, seed, tau_weak, lambda_weak, num_clusters, kappa)
    encoder = CedarEncoder(
        input_dim=db.input_dim,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    encoder, mu, pi, history = composite_cedar_training_phase(encoder, db.X_benign_scaled, db.y_benign_edges, cfg)
    ckpt_path = ckpt_dir / f"cedar_vmf_{run_tag}.pth"
    _save_checkpoint(encoder, mu, pi, cfg, history, ckpt_path)
    return ckpt_path


def evaluate_run(
    db: DataBundle,
    ckpt_path: Path,
    scores_dir: Path,
    args,
    run_tag: str,
    primary_f1_percentile: float,
) -> Dict[str, object]:
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    encoder, mu, pi, cfg, history = _load_checkpoint(db.input_dim, ckpt_path, device)
    with torch.no_grad():
        X_full = _full_numeric_matrix(db)
        Z_full = compute_embeddings(encoder, X_full, cfg.batch_size, device)
        scores = score_samples_vmf(Z_full, mu, pi, cfg.kappa)

    y_true = db.merged_df["Label"].values.astype(int)
    metrics = compute_metrics(scores, y_true, primary_f1_percentile)
    auto_f1 = compute_f1_at_percentile(scores, y_true, db.f1_percentile)
    fixed_f1 = compute_f1_at_top_percent(scores, y_true, args.fixed_top_percent)
    parquet_path = scores_dir / f"cedar_vmf_{run_tag}_scores.parquet"
    _save_scores_parquet(scores, y_true, parquet_path)

    row = {
        "seed": cfg.seed,
        "cut_off": args.current_cut_off,
        "embedding_dim": cfg.embedding_dim,
        "batch_size": cfg.batch_size,
        "num_clusters": cfg.num_clusters,
        "tau_weak": cfg.tau_weak,
        "lambda_weak": cfg.lambda_weak,
        "kappa": cfg.kappa,
        "pi_smoothing": cfg.pi_smoothing,
        **metrics,
        f"F1@{db.f1_percentile:.2f}_auto": auto_f1,
        f"F1@top{args.fixed_top_percent:.2f}%": fixed_f1,
    }

    if history:
        final_diag = history[-1]
        for key in (
            "pi_min",
            "pi_max",
            "pi_entropy",
            "effective_clusters",
            "min_expected_count",
            "max_expected_count",
            "reseeded_components",
        ):
            if key in final_diag:
                row[f"final_{key}"] = final_diag[key]

    return row


def training_grid(args) -> None:
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.train_out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (d, b, tw, lw, nc, ka)
        for d in args.embedding_dim_list
        for b in args.batch_size_list
        for tw in args.tau_weak_list
        for lw in args.lambda_weak_list
        for nc in args.num_clusters_list
        for ka in args.kappa_list
    ]
    total = len(args.seeds) * len(args.cut_off_list) * len(configs)
    print(f"[INFO] Starting {total} CEDAR runs on {device}.")

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n========== Seed {seed} ==========")
        for cut_off in args.cut_off_list:
            args.current_cut_off = cut_off
            try:
                db = load_and_sample_data(args.dataset, resolve_data_dir(args.dataset, args.data_dir), cut_off, seed)
            except Exception as exc:
                print(f"[ERROR] Failed to load dataset (cut_off={cut_off}): {exc}")
                continue

            for emb_dim, batch_size, tau_weak, lambda_weak, num_clusters, kappa in configs:
                run_tag = _format_run_tag(
                    args.dataset,
                    cut_off,
                    seed,
                    emb_dim,
                    batch_size,
                    num_clusters,
                    tau_weak,
                    lambda_weak,
                    kappa,
                )
                print(f"[INFO] Training run: {run_tag}")
                try:
                    ckpt_path = train_single_run(
                        db,
                        args,
                        emb_dim,
                        batch_size,
                        seed,
                        tau_weak,
                        lambda_weak,
                        num_clusters,
                        kappa,
                        run_tag,
                        ckpt_dir,
                    )
                    print(f"[INFO] Saved best checkpoint to {ckpt_path}")
                except Exception as exc:
                    print(f"[ERROR] Training failed for {run_tag}: {exc}")


def evaluate_grid(args) -> None:
    ckpt_dir = Path(args.train_out_dir)
    out_dir = Path(args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores_parquet"
    scores_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (d, b, tw, lw, nc, ka)
        for d in args.embedding_dim_list
        for b in args.batch_size_list
        for tw in args.tau_weak_list
        for lw in args.lambda_weak_list
        for nc in args.num_clusters_list
        for ka in args.kappa_list
    ]
    rows = []

    for seed in tqdm(args.seeds, desc="Seeds"):
        set_seed(seed)
        for cut_off in tqdm(args.cut_off_list, desc=f"seed {seed} cutoffs", leave=False):
            args.current_cut_off = cut_off
            try:
                db = load_and_sample_data(args.dataset, resolve_data_dir(args.dataset, args.data_dir), cut_off, seed)
            except Exception as exc:
                print(f"[ERROR] Failed to load dataset (cut_off={cut_off}): {exc}")
                continue

            primary_f1_percentile = db.f1_percentile if args.f1_percentile.lower() == "auto" else float(args.f1_percentile)

            for emb_dim, batch_size, tau_weak, lambda_weak, num_clusters, kappa in tqdm(
                configs,
                desc=f"cut_off {cut_off:.2f} configs",
                leave=False,
            ):
                run_tag = _format_run_tag(
                    args.dataset,
                    cut_off,
                    seed,
                    emb_dim,
                    batch_size,
                    num_clusters,
                    tau_weak,
                    lambda_weak,
                    kappa,
                )
                ckpt_path = ckpt_dir / f"cedar_vmf_{run_tag}.pth"
                if not ckpt_path.is_file():
                    print(f"[WARN] Missing checkpoint for run {run_tag}, skipping.")
                    continue
                try:
                    row = evaluate_run(db, ckpt_path, scores_dir, args, run_tag, primary_f1_percentile)
                    rows.append(row)
                except Exception as exc:
                    print(f"[ERROR] Evaluation failed for {run_tag}: {exc}")

    if rows:
        results_df = pd.DataFrame(rows)
        results_path = out_dir / args.results_file
        results_df.to_csv(results_path, index=False)
        print(f"\n[INFO] Evaluation results saved to {results_path}")
    else:
        print("\n[WARN] No evaluation rows were produced.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="CEDAR train+eval utility for vMF mixture + weak structure")
    parser.add_argument("--dataset", type=str, default="o-unsw", choices=SUPPORTED_DATASETS, help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="auto", help="Path to dataset files. Use 'auto' for defaults")
    parser.add_argument("--train_out_dir", type=str, default="cedar_training_outputs", help="Directory for checkpoints")
    parser.add_argument("--eval_out_dir", type=str, default="cedar_eval_outputs", help="Directory for evaluation CSV files")

    parser.add_argument("--seeds", type=int, nargs="+", default=[13], help="List of random seeds")
    parser.add_argument("--cut_off_list", type=float, nargs="+", default=[1.0], help="Cutoff over [0,1] of max timestamp")

    parser.add_argument("--embedding_dim_list", type=int, nargs="+", default=[128], help="Embedding dimensions")
    parser.add_argument("--batch_size_list", type=int, nargs="+", default=[512], help="Batch sizes")

    parser.add_argument("--hidden_dim", type=int, default=None, help="Optional hidden dim override for encoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in encoder")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--warmup_epochs", type=int, default=40, help="Weak-only warm-up epochs")
    parser.add_argument("--tau_weak_list", type=float, nargs="+", default=[0.1], help="Weak-structure temperature values")
    parser.add_argument("--lambda_weak_list", type=float, nargs="+", default=[1.0], help="Weight on the weak objective")

    parser.add_argument("--num_clusters_list", type=int, nargs="+", default=[20], help="Mixture component counts")
    parser.add_argument("--em_iters", type=int, default=8, help="Number of CEDAR outer iterations")
    parser.add_argument("--encoder_epochs", type=int, default=8, help="Encoder epochs per CEDAR iteration")
    parser.add_argument("--kappa_list", type=float, nargs="+", default=[50.0], help="vMF concentration values")
    parser.add_argument("--init_restarts", type=int, default=1, help="Number of prototype init restarts")

    parser.add_argument("--pi_smoothing", type=float, default=0.01, help="Uniform smoothing added to mixing weights")
    parser.add_argument("--min_component_weight", type=float, default=1e-3, help="Floor applied to component weights")
    parser.add_argument("--min_expected_count", type=float, default=8.0, help="Reseed components below this soft count")
    parser.add_argument("--disable_reseed", action="store_true", help="Disable low-mass component reseeding")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="Device override: auto|cpu|cuda")

    parser.add_argument("--results_file", type=str, default="cedar_vmf_eval_results.csv", help="Filename for evaluation CSV")
    parser.add_argument("--f1_percentile", type=str, default="auto", help="Primary threshold percentile for point metrics ('auto' == benign ratio)")
    parser.add_argument("--fixed_top_percent", type=float, default=5.0, help="Additional fixed top-x%% anomaly rate reported alongside F1@auto")
    return parser.parse_args()


def main():
    args = parse_arguments()
    training_grid(args)
    evaluate_grid(args)


if __name__ == "__main__":
    main()
