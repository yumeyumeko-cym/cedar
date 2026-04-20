# CEDAR

Composite weak-structure + von Mises–Fisher (vMF) mixture training for **unsupervised network-traffic anomaly detection**.

CEDAR trains a small MLP encoder that maps flow features to a unit hypersphere, fits a vMF mixture over benign embeddings, and scores any sample by its negative mixture log-probability. A weak supervised-contrastive signal — derived from network *edges* (src→dst pairs) rather than attack labels — shapes the embedding space so that the unsupervised mixture has structure to latch onto.

## How it works

1. **Encoder.** An MLP with `LayerNorm + GELU + Dropout` blocks projects each flow into an `embedding_dim`-vector that is L2-normalized to live on the unit sphere.
2. **Warm-up.** The encoder is pre-trained for `warmup_epochs` with a *weak-structure* supervised-contrastive loss: positives are flows sharing the same edge, negatives are everything else in the batch (anchor-balanced via `LabelAwareBatchSampler`).
3. **Composite EM loop.** For `em_iters` outer iterations, CEDAR alternates:
   - **E-step + M-step** on a vMF mixture with `num_clusters` components, concentration `kappa`, smoothed mixing weights, and low-mass component reseeding to prevent collapse.
   - **Encoder gradient steps** against `L = vMF_Q + λ_weak · weak_structure`, mixing the M-step objective with the contrastive signal so the embedding moves toward better mixture geometry without abandoning the weak structure.
4. **Scoring.** At inference, each sample's anomaly score is `−log Σ_k π_k · vMF(z; μ_k, κ)`. Higher is more anomalous.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
└── cedar/
    ├── __init__.py        # public API
    ├── __main__.py        # `python -m cedar` entry point
    ├── config.py          # CedarConfig dataclass
    ├── model.py           # CedarEncoder, LabelAwareBatchSampler
    ├── training.py        # warmup, EM, composite training loop
    ├── scoring.py         # vMF scoring + AUROC/AUPR/F1 metrics
    ├── data.py            # DataBundle + per-dataset loaders
    └── cli.py             # training_grid, evaluate_grid, argparse
```

## Installation

```bash
pip install -r requirements.txt
```

If you need a specific PyTorch build (CUDA / ROCm / CPU-only), install the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`.

## Supported datasets

CEDAR expects each dataset to live in its own directory under `data/` with one or two parquet files. Use `--dataset <key>` on the CLI; pass `--data_dir <path>` to override the default location.

| `--dataset` | Default directory | Required parquet files |
|-------------|------------------|------------------------|
| `o-unsw`     | `data/UNSWNB15`           | `merged_df.parquet`, `training_df.parquet` |
| `nf-unsw`    | `data/NF-UNSWNB15-v3`     | `merged_df.parquet`                        |
| `cicids2017` | `data/CICIDS2017`         | `merged_df.parquet`, `data1.parquet`       |
| `cicids2018` | `data/NF-2018-v3`         | `merged_df.parquet`                        |

Every loader expects standard columns (`Source IP`, `Destination IP`, `Timestamp`, `Label`) plus dataset-specific feature columns; see [cedar/data.py](cedar/data.py) for the per-dataset text/categorical/numeric splits.

## Quick start

Train and evaluate one default-configuration run on UNSW-NB15:

```bash
python -m cedar --dataset o-unsw --seeds 13 --cut_off_list 1.0
```

This writes:

- a checkpoint to `cedar_training_outputs/cedar_vmf_<run_tag>.pth`
- per-sample scores to `cedar_eval_outputs/scores_parquet/cedar_vmf_<run_tag>_scores.parquet`
- summary metrics to `cedar_eval_outputs/cedar_vmf_eval_results.csv`

Sweep over a small grid:

```bash
python -m cedar \
  --dataset nf-unsw \
  --seeds 13 17 23 \
  --cut_off_list 0.5 1.0 \
  --num_clusters_list 10 20 40 \
  --kappa_list 25 50 100
```

Every combination of `seeds × cut_off_list × embedding_dim_list × batch_size_list × tau_weak_list × lambda_weak_list × num_clusters_list × kappa_list` is run as a separate training job and recorded as one row in the results CSV.

## CLI reference

Run `python -m cedar --help` for the full list. The most important groups:

**Data**
- `--dataset {o-unsw, nf-unsw, cicids2017, cicids2018}` — dataset key (default `o-unsw`).
- `--data_dir PATH` — override the default dataset directory; `auto` uses the table above.

**Grid sweep**
- `--seeds INT [INT …]` — random seeds (default `13`).
- `--cut_off_list FLOAT [FLOAT …]` — fraction of max benign timestamp to keep (default `1.0`).
- `--embedding_dim_list`, `--batch_size_list`, `--num_clusters_list`, `--kappa_list`, `--tau_weak_list`, `--lambda_weak_list` — hyperparameter grids.

**Encoder & optimization**
- `--hidden_dim`, `--dropout`, `--learning_rate`, `--weight_decay`, `--max_grad_norm`.

**Weak-structure channel**
- `--warmup_epochs` — weak-only pretraining epochs (default `40`).
- `--tau_weak_list`, `--lambda_weak_list` — temperature and loss weight for the contrastive term.

**vMF mixture loop**
- `--em_iters` — outer iterations (default `8`).
- `--encoder_epochs` — encoder gradient epochs per outer iteration (default `8`).
- `--init_restarts` — spherical k-means++ restarts when picking initial prototypes.

**Anti-collapse safeguards**
- `--pi_smoothing` — uniform smoothing added to mixing weights.
- `--min_component_weight` — floor on `π_k`.
- `--min_expected_count` — soft-count threshold below which a component is reseeded.
- `--disable_reseed` — turn off low-mass component reseeding.

**Runtime**
- `--device {auto, cpu, cuda}` — defaults to `cuda` if available.
- `--num_workers` — DataLoader workers.

**Output**
- `--train_out_dir` (default `cedar_training_outputs`) — checkpoint directory.
- `--eval_out_dir` (default `cedar_eval_outputs`) — evaluation outputs.
- `--results_file` (default `cedar_vmf_eval_results.csv`).
- `--f1_percentile` — `auto` uses the dataset's benign ratio; pass a float to fix it.
- `--fixed_top_percent` — additional reported F1 at top-x% anomaly rate (default `5.0`).

## Outputs

| Path | Contents |
|------|----------|
| `cedar_training_outputs/cedar_vmf_<tag>.pth` | Dict with `encoder_state_dict`, `mu`, `pi`, serialized `cfg`, training `history`. |
| `cedar_eval_outputs/scores_parquet/cedar_vmf_<tag>_scores.parquet` | Per-sample `score` (negative mixture log-prob) and binary `label`. |
| `cedar_eval_outputs/cedar_vmf_eval_results.csv` | One row per `(seed × cut_off × hyperparams)` with `AUROC`, `AUPR`, `ACC`, `Precision`, `Recall`, `F1@auto`, `F1@top-x%`, and final cluster diagnostics (`pi_min`, `pi_max`, `pi_entropy`, `effective_clusters`, `min/max_expected_count`, `reseeded_components`). |

The `<tag>` encodes the dataset, cutoff, seed, and key hyperparameters so a single output directory can hold an entire sweep.

## Programmatic API

```python
import numpy as np
from cedar import (
    CedarConfig,
    CedarEncoder,
    composite_cedar_training_phase,
    compute_embeddings,
    score_samples_vmf,
)

# X: (n, d) benign feature matrix; weak_labels: (n,) edge ids
cfg = CedarConfig(num_clusters=20, kappa=50.0, em_iters=8)
encoder = CedarEncoder(input_dim=X.shape[1], embedding_dim=cfg.embedding_dim)
encoder, mu, pi, history = composite_cedar_training_phase(encoder, X, weak_labels, cfg)

Z = compute_embeddings(encoder, X_eval, cfg.batch_size, cfg.device)
scores = score_samples_vmf(Z, mu, pi, cfg.kappa)  # higher = more anomalous
```

Lower-level pieces — `warmup_phase`, `e_step_vmf`, `update_mu`, `fit_vmf_mixture_fixed_embeddings`, `cluster_diagnostics` — are also exported from `cedar` for custom pipelines.

## Reproducibility

`set_seed(seed)` is called at the start of every run and seeds Python, NumPy, and PyTorch (including CUDA). Sweep over `--seeds 13 17 23 …` to estimate variance.

## License

MIT — see [LICENSE](LICENSE).
