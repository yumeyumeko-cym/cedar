# CEDAR

Composite weak-structure + von Mises–Fisher (vMF) mixture training for **unsupervised network-traffic anomaly detection**.

CEDAR trains a small MLP encoder that maps flow features to a unit hypersphere, fits a vMF mixture over benign embeddings, and scores any sample by its negative mixture log-probability. A weak supervised-contrastive signal — derived from network *edges* (src→dst pairs) rather than attack labels — shapes the embedding space so that the unsupervised mixture has structure to latch onto.


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
python -m cedar --dataset nf-unsw --seeds 13 --cut_off_list 1.0
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
  --cut_off_list 1.0 \
  --num_clusters_list 10 20 40 \
  --kappa_list 25 50 100
```

Every combination of `seeds × cut_off_list × embedding_dim_list × batch_size_list × tau_weak_list × lambda_weak_list × num_clusters_list × kappa_list` is run as a separate training job and recorded as one row in the results CSV.



## Outputs

| Path | Contents |
|------|----------|
| `cedar_training_outputs/cedar_vmf_<tag>.pth` | Dict with `encoder_state_dict`, `mu`, `pi`, serialized `cfg`, training `history`. |
| `cedar_eval_outputs/scores_parquet/cedar_vmf_<tag>_scores.parquet` | Per-sample `score` (negative mixture log-prob) and binary `label`. |
| `cedar_eval_outputs/cedar_vmf_eval_results.csv` | One row per `(seed × cut_off × hyperparams)` with `AUROC`, `AUPR`, `ACC`, `Precision`, `Recall`, `F1@auto`, `F1@top-x%`, and final cluster diagnostics (`pi_min`, `pi_max`, `pi_entropy`, `effective_clusters`, `min/max_expected_count`, `reseeded_components`). |

The `<tag>` encodes the dataset, cutoff, seed, and key hyperparameters so a single output directory can hold an entire sweep.



## License

MIT — see [LICENSE](LICENSE).
