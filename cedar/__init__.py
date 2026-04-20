"""CEDAR — composite weak-structure + vMF mixture for unsupervised anomaly detection."""

from .config import CedarConfig
from .data import (
    DATASET_DEFAULT_DIRS,
    DataBundle,
    SUPPORTED_DATASETS,
    load_and_sample_data,
    resolve_data_dir,
)
from .model import CedarEncoder, LabelAwareBatchSampler
from .scoring import (
    cluster_diagnostics,
    compute_f1_at_percentile,
    compute_f1_at_top_percent,
    compute_metrics,
    score_samples_vmf,
)
from .training import (
    composite_cedar_training_phase,
    compute_embeddings,
    e_step_vmf,
    fit_vmf_mixture_fixed_embeddings,
    select_init_mu,
    set_seed,
    update_mu,
    vmf_Q_loss,
    warmup_phase,
    weak_structure_loss,
)

__all__ = [
    "CedarConfig",
    "CedarEncoder",
    "DATASET_DEFAULT_DIRS",
    "DataBundle",
    "LabelAwareBatchSampler",
    "SUPPORTED_DATASETS",
    "cluster_diagnostics",
    "composite_cedar_training_phase",
    "compute_embeddings",
    "compute_f1_at_percentile",
    "compute_f1_at_top_percent",
    "compute_metrics",
    "e_step_vmf",
    "fit_vmf_mixture_fixed_embeddings",
    "load_and_sample_data",
    "resolve_data_dir",
    "score_samples_vmf",
    "select_init_mu",
    "set_seed",
    "update_mu",
    "vmf_Q_loss",
    "warmup_phase",
    "weak_structure_loss",
]
