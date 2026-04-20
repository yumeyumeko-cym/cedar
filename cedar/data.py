"""Dataset loading and preprocessing helpers for CEDAR."""

from dataclasses import dataclass
from typing import List
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


@dataclass
class DataBundle:
	benign_df: pd.DataFrame
	merged_df: pd.DataFrame
	columns_numeric: List[str]
	scaler: MinMaxScaler
	label_encoder: LabelEncoder
	X_benign_scaled: np.ndarray
	y_benign_edges: np.ndarray
	input_dim: int
	f1_percentile: float


SUPPORTED_DATASETS = ("o-unsw", "nf-unsw", "cicids2017", "cicids2018")

DATASET_DEFAULT_DIRS = {
	"o-unsw": os.path.join("data", "UNSWNB15"),
	"nf-unsw": os.path.join("data", "NF-UNSWNB15-v3"),
	"cicids2017": os.path.join("data", "CICIDS2017"),
	"cicids2018": os.path.join("data", "NF-2018-v3"),
}


def _load_o_unsw_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
	merged_path = os.path.join(data_dir, "merged_df.parquet")
	training_path = os.path.join(data_dir, "training_df.parquet")

	merged_df = pd.read_parquet(merged_path)
	data1 = pd.read_parquet(training_path)

	if "Edge" not in merged_df.columns:
		merged_df["Edge"] = merged_df["Source IP"] + "->" + merged_df["Destination IP"]

	if "Edge" not in data1.columns:
		data1["Edge"] = data1["Source IP"] + "->" + data1["Destination IP"]

	time_split = data1["Timestamp"].max()

	data1 = data1[data1["Timestamp"] <= time_split]
	benign_df = data1.groupby("Edge").filter(lambda df: df["Label"].max() == 0).copy()
	max_ts = benign_df["Timestamp"].max()
	cutoff_ts = cut_off * max_ts
	early_df = benign_df[benign_df["Timestamp"] <= cutoff_ts]

	ts_per_edge = early_df.groupby("Edge")["Timestamp"].nunique()
	eligible_edges = ts_per_edge[ts_per_edge >= 5].index
	benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

	columns_text = [
		"Source IP",
		"Destination IP",
		"Source Port",
		"Destination Port",
		"Timestamp",
		"Ltime",
		"Label",
		"attack_cat",
		"ct_ftp_cmd",
		"ct_flw_http_mthd",
		"Edge",
	]

	columns_cat = [
		"proto",
		"state",
		"service",
		"is_sm_ips_ports",
		"is_ftp_login",
	]

	columns_numeric = [col for col in merged_df.columns if col not in columns_text + columns_cat]
	X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	scaler = MinMaxScaler()
	X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
	edge_labels = benign_df["Edge"].copy()
	label_encoder = LabelEncoder()
	y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
	input_dim = X_benign_scaled.shape[1]

	benign_ratio = (merged_df["Label"] == 0).mean() * 100.0
	f1_percentile = benign_ratio

	return DataBundle(
		benign_df=benign_df,
		merged_df=merged_df,
		columns_numeric=columns_numeric,
		scaler=scaler,
		label_encoder=label_encoder,
		X_benign_scaled=X_benign_scaled,
		y_benign_edges=y_benign_edges,
		input_dim=input_dim,
		f1_percentile=f1_percentile,
	)


def _load_cicids2017_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
	merged_path = os.path.join(data_dir, "merged_df.parquet")
	training_path = os.path.join(data_dir, "data1.parquet")

	merged_df = pd.read_parquet(merged_path)
	data1 = pd.read_parquet(training_path)

	if "Edge" not in merged_df.columns:
		merged_df["Edge"] = merged_df["Source IP"] + "->" + merged_df["Destination IP"]

	if "Edge" not in data1.columns:
		data1["Edge"] = data1["Source IP"] + "->" + data1["Destination IP"]

	time_split = data1["Timestamp"].max()

	data1 = data1[data1["Timestamp"] <= time_split]
	benign_df = data1.groupby("Edge").filter(lambda df: df["Label"].max() == 0).copy()
	max_ts = benign_df["Timestamp"].max()
	cutoff_ts = cut_off * max_ts
	early_df = benign_df[benign_df["Timestamp"] <= cutoff_ts]

	ts_per_edge = early_df.groupby("Edge")["Timestamp"].nunique()
	eligible_edges = ts_per_edge[ts_per_edge >= 5].index
	benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

	columns_text = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
					"Timestamp", "Label", "Attack", "Edge"]
	columns_categorical = [col for col in merged_df if "Flag" in col]
	columns_numeric = [col for col in merged_df if col not in columns_text + columns_categorical]

	X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	scaler = MinMaxScaler()
	X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
	edge_labels = benign_df["Edge"].copy()
	label_encoder = LabelEncoder()
	y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
	input_dim = X_benign_scaled.shape[1]

	benign_ratio = (merged_df["Label"] == 0).mean() * 100.0
	f1_percentile = benign_ratio

	return DataBundle(
		benign_df=benign_df,
		merged_df=merged_df,
		columns_numeric=columns_numeric,
		scaler=scaler,
		label_encoder=label_encoder,
		X_benign_scaled=X_benign_scaled,
		y_benign_edges=y_benign_edges,
		input_dim=input_dim,
		f1_percentile=f1_percentile,
	)


def _load_nf_unsw_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
	merged_path = os.path.join(data_dir, "merged_df.parquet")
	merged_df = pd.read_parquet(merged_path)
	if "Edge" not in merged_df.columns:
		merged_df["Edge"] = merged_df["Source IP"] + "->" + merged_df["Destination IP"]
	time_split = 144

	data1 = merged_df[merged_df["Timestamp"] <= time_split]
	benign_df = data1.groupby("Edge").filter(lambda df: df["Label"].max() == 0).copy()
	max_ts = benign_df["Timestamp"].max()
	cutoff_ts = cut_off * max_ts
	early_df = benign_df[benign_df["Timestamp"] <= cutoff_ts]

	ts_per_edge = early_df.groupby("Edge")["Timestamp"].nunique()
	eligible_edges = ts_per_edge[ts_per_edge >= 5].index
	benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

	columns_text = [
		"Source IP",
		"Destination IP",
		"Source Port",
		"Destination Port",
		"Timestamp",
		"FLOW_END_MILLISECONDS",
		"DNS_QUERY_ID",
		"FTP_COMMAND_RET_CODE",
		"Label",
		"Attack",
		"Edge",
	]

	columns_cat = [
		"PROTOCOL",
		"L7_PROTO",
		"TCP_FLAGS",
		"ICMP_TYPE",
		"CLIENT_TCP_FLAGS",
		"SERVER_TCP_FLAGS",
		"DNS_QUERY_TYPE",
		"ICMP_IPV4_TYPE",
	]

	columns_numeric = [col for col in merged_df.columns if col not in columns_text + columns_cat]

	X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	scaler = MinMaxScaler()
	X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
	edge_labels = benign_df["Edge"].copy()
	label_encoder = LabelEncoder()
	y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
	input_dim = X_benign_scaled.shape[1]

	benign_ratio = (merged_df["Label"] == 0).mean() * 100.0
	f1_percentile = benign_ratio

	return DataBundle(
		benign_df=benign_df,
		merged_df=merged_df,
		columns_numeric=columns_numeric,
		scaler=scaler,
		label_encoder=label_encoder,
		X_benign_scaled=X_benign_scaled,
		y_benign_edges=y_benign_edges,
		input_dim=input_dim,
		f1_percentile=f1_percentile,
	)


def _load_cicids2018_data(data_dir: str, cut_off: float, seed: int) -> DataBundle:
	merged_path = os.path.join(data_dir, "merged_df.parquet")
	merged_df = pd.read_parquet(merged_path)

	# CICIDS2018 is much sparser under src->dst labels, so use a service-aware edge
	# and a less aggressive early-time window.
	merged_df["Edge"] = (
		merged_df["Destination IP"].astype(str)
		+ "|"
		+ merged_df["Destination Port"].astype(str)
		+ "|"
		+ merged_df["PROTOCOL"].astype(str)
	)
	time_split = float(merged_df["Timestamp"].quantile(0.20))

	data1 = merged_df[merged_df["Timestamp"] <= time_split].copy()
	benign_df = data1.groupby("Edge").filter(lambda df: df["Label"].max() == 0).copy()
	max_ts = benign_df["Timestamp"].max()
	cutoff_ts = cut_off * max_ts
	early_df = benign_df[benign_df["Timestamp"] <= cutoff_ts]

	ts_per_edge = early_df.groupby("Edge")["Timestamp"].nunique()
	eligible_edges = ts_per_edge[ts_per_edge >= 5].index
	benign_df = early_df[early_df["Edge"].isin(eligible_edges)].copy()

	columns_text = [
		"Source IP",
		"Destination IP",
		"Source Port",
		"Destination Port",
		"Timestamp",
		"FLOW_END_MILLISECONDS",
		"DNS_QUERY_ID",
		"FTP_COMMAND_RET_CODE",
		"Label",
		"Attack",
		"Edge",
	]

	columns_cat = [
		"PROTOCOL",
		"L7_PROTO",
		"TCP_FLAGS",
		"ICMP_TYPE",
		"CLIENT_TCP_FLAGS",
		"SERVER_TCP_FLAGS",
		"DNS_QUERY_TYPE",
		"ICMP_IPV4_TYPE",
	]

	columns_numeric = [col for col in merged_df.columns if col not in columns_text + columns_cat]

	X_benign_numeric = benign_df[columns_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	scaler = MinMaxScaler()
	X_benign_scaled = scaler.fit_transform(X_benign_numeric).astype(np.float32)
	edge_labels = benign_df["Edge"].copy()
	label_encoder = LabelEncoder()
	y_benign_edges = label_encoder.fit_transform(edge_labels).astype(np.int64)
	input_dim = X_benign_scaled.shape[1]

	benign_ratio = (merged_df["Label"] == 0).mean() * 100.0
	f1_percentile = benign_ratio

	return DataBundle(
		benign_df=benign_df,
		merged_df=merged_df,
		columns_numeric=columns_numeric,
		scaler=scaler,
		label_encoder=label_encoder,
		X_benign_scaled=X_benign_scaled,
		y_benign_edges=y_benign_edges,
		input_dim=input_dim,
		f1_percentile=f1_percentile,
	)


def resolve_data_dir(dataset: str, user_data_dir: str) -> str:
	dataset_key = dataset.lower()
	if dataset_key not in DATASET_DEFAULT_DIRS:
		raise ValueError(
			f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}"
		)

	if user_data_dir and user_data_dir.lower() != "auto":
		return user_data_dir

	return DATASET_DEFAULT_DIRS[dataset_key]


def load_and_sample_data(dataset: str, data_dir: str, cut_off: float, seed: int) -> DataBundle:
	dataset_key = dataset.lower()
	if dataset_key not in _DATASET_LOADERS:
		raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")

	if not os.path.isdir(data_dir):
		raise FileNotFoundError(f"Data directory not found for dataset '{dataset_key}': {data_dir}")

	return _DATASET_LOADERS[dataset_key](data_dir, cut_off, seed)


_DATASET_LOADERS = {
	"o-unsw": _load_o_unsw_data,
	"nf-unsw": _load_nf_unsw_data,
	"cicids2017": _load_cicids2017_data,
	"cicids2018": _load_cicids2018_data,
}


__all__ = [
	"DataBundle",
	"DATASET_DEFAULT_DIRS",
	"SUPPORTED_DATASETS",
	"resolve_data_dir",
	"load_and_sample_data",
]
