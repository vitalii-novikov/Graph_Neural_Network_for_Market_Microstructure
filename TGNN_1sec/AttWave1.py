# %% [markdown]
# # Temporal Graph Baseline for ETH 5-Minute Forecasting
#
# This notebook implements a clean baseline for the thesis:
# **Real-Time Market Microstructure Modeling with Temporal Graph Neural Networks**.
#
# Architecture used here:
# - **Nodes = assets**: ADA, BTC, ETH
# - **Target asset = ETH**
# - **Temporal encoder**: causal Transformer applied independently to each asset history
# - **Graph propagation**: message passing across assets using
#   1) identity support,
#   2) dynamic edge-conditioned adjacency from past-only rolling lead-lag correlations,
#   3) learned adaptive adjacency
# - **Prediction head**: ETH-only regression head for the fixed-horizon forward return
#
# Methodological principles enforced here:
# - Main supervised target is always the **fixed 5-minute ETH forward log return**
# - The forecast horizon is fixed in **real clock time** across all frequencies
# - Threshold selection is done on **validation only**
# - Trading evaluation uses a **non-overlapping sequential backtest**
# - Scaling is fit on **train only** for every fold
# - Walk-forward CV and final production refit use **purge/embargo gaps**
# - Triple-barrier labeling is intentionally **not used** in this baseline

# %%
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## Imports, seeds, and reproducibility

# %%
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

print("DEVICE:", DEVICE)

# %% [markdown]
# ## Config
#
# Notes:
# - `freq` may be `"1sec"`, `"1min"`, or `"5min"`
# - the **forecast horizon** is always 5 real minutes, converted by helper
# - lookback length is allowed to vary by frequency for practicality
# - all splitting logic uses a purge gap at least as large as the forecast horizon

# %%
CFG: Dict[str, Any] = {
    # Core data settings
    "freq": "1min",  # "1sec", "1min", or "5min"
    "data_dir": "../dataset",
    "artifact_dir": "./artifacts_temporal_graph_baseline_1min",
    "assets": ["ADA", "BTC", "ETH"],
    "target_asset": "ETH",
    "data_slice_start_frac": 0.00,
    "data_slice_end_frac": 0.75,

    # Final untouched holdout
    "final_holdout_frac": 0.10,

    # Frequency-specific lookback windows (hyperparameter, not target horizon)
    "lookback_bars_by_freq": {
        "1sec": 300,
        "1min": 240,
        "5min": 120,
    },

    # Order book / feature settings
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # Edge-feature windows in bars by frequency
    # These are strictly past-only rolling windows used to summarize cross-asset dependence.
    "edge_corr_windows_bars_by_freq": {
        "1sec": [60, 300, 900],
        "1min": [10, 30, 60],
        "5min": [6, 12, 24],
    },
    "edge_corr_lags_bars": [0, 1, 2, 5],
    "edge_use_fisher_z": True,

    # Split settings on the pre-holdout region
    "train_min_frac": 0.50,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,
    "purge_gap_extra_bars": 0,

    # Scaling
    "max_abs_node_feature": 8.0,
    "max_abs_edge_feature": 5.0,
    "scaler_quantile_low": 5.0,
    "scaler_quantile_high": 95.0,

    # Model
    "hidden_dim": 64,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "transformer_ff_mult": 4,
    "graph_layers": 2,
    "edge_hidden_dim": 32,
    "adaptive_adj_dim": 8,
    "dropout": 0.15,

    # Training
    "batch_size": 64,
    "epochs": 40,
    "patience": 6,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "huber_beta": 1.0e-3,

    # Trading / thresholding
    "cost_bps_per_side": 1.0,
    "trade_label_buffer_bps": 0.0,
    "threshold_grid_abs_return": [0.0, 0.0001, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030],
    "threshold_grid_quantiles": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    "min_validation_trades": 20,
}

ASSETS: List[str] = list(CFG["assets"])
TARGET_ASSET: str = str(CFG["target_asset"])
assert TARGET_ASSET in ASSETS
ASSET2IDX: Dict[str, int] = {a: i for i, a in enumerate(ASSETS)}
TARGET_NODE: int = ASSET2IDX[TARGET_ASSET]
ARTIFACT_DIR = Path(CFG["artifact_dir"])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("ASSETS:", ASSETS)
print("TARGET_ASSET:", TARGET_ASSET)
print("ARTIFACT_DIR:", ARTIFACT_DIR.resolve())

# %% [markdown]
# ## Helper functions for frequency and horizon conversion
#
# This is the key consistency rule:
# the model always predicts a **5-minute clock-time horizon**.
#
# Therefore:
# - `1sec  -> 300 bars`
# - `1min  -> 5 bars`
# - `5min  -> 1 bar`

# %%


FREQ = CFG["freq"]
HORIZON_MINUTES = 5
HORIZON_BARS = HORIZON_MINUTES
LOOKBACK_BARS = CFG["lookback_bars_by_freq"][FREQ]
EDGE_CORR_WINDOWS = CFG["edge_corr_windows_bars_by_freq"][FREQ]
EDGE_CORR_LAGS = [int(x) for x in CFG["edge_corr_lags_bars"]]
PURGE_GAP_BARS = HORIZON_BARS + int(CFG["purge_gap_extra_bars"])

print("FREQ:", FREQ)
print("HORIZON_MINUTES:", HORIZON_MINUTES)
print("HORIZON_BARS:", HORIZON_BARS)
print("LOOKBACK_BARS:", LOOKBACK_BARS)
print("EDGE_CORR_WINDOWS:", EDGE_CORR_WINDOWS)
print("EDGE_CORR_LAGS:", EDGE_CORR_LAGS)
print("PURGE_GAP_BARS:", PURGE_GAP_BARS)

# %% [markdown]
# ## Graph topology
#
# We use directed all-pairs edges with self-loops.
# This is a small graph with nodes = assets.

# %%
def build_edge_list(assets: List[str], add_self_loops: bool = True) -> List[Tuple[str, str]]:
    edges = [(src, dst) for src in assets for dst in assets if src != dst]
    if add_self_loops:
        edges += [(a, a) for a in assets]
    return edges


EDGE_LIST: List[Tuple[str, str]] = build_edge_list(ASSETS, add_self_loops=True)
EDGE_NAMES: List[str] = [f"{src}->{dst}" for src, dst in EDGE_LIST]
EDGE_INDEX = torch.tensor(
    [[ASSET2IDX[src], ASSET2IDX[dst]] for src, dst in EDGE_LIST],
    dtype=torch.long,
)

print("EDGE_NAMES:", EDGE_NAMES)
print("EDGE_INDEX:\n", EDGE_INDEX)

# %% [markdown]
# ## Data loading
#
# The loader is designed to support the CSV schema used in the old notebook:
# - timestamp column such as `system_time`
# - `midpoint`
# - `spread`
# - `buys`, `sells`
# - `bids_notional_0 ... bids_notional_{L-1}`
# - `asks_notional_0 ... asks_notional_{L-1}`
#
# The code also accepts a few safe aliases where possible, but standardized names
# in the notebook remain honest:
# `bids_notional`, `asks_notional`.

# %%
def choose_existing_column(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required column for {what}. Tried: {candidates}")


def infer_timestamp_column(df: pd.DataFrame) -> str:
    return choose_existing_column(
        df,
        ["system_time", "timestamp", "time", "datetime"],
        "timestamp",
    )


def infer_book_column(df: pd.DataFrame, side_prefix: str, level: int) -> str:
    candidates = [
        f"{side_prefix}_notional_{level}",
        f"{side_prefix}_vol_{level}",
        f"{side_prefix}_{level}",
    ]
    return choose_existing_column(df, candidates, f"{side_prefix} level {level}")


def load_one_asset_raw(asset: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    freq = FREQ
    data_dir = Path(cfg["data_dir"])
    path = data_dir / f"{asset}_{freq}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    start = int(len(df) * float(cfg["data_slice_start_frac"]))
    end = int(len(df) * float(cfg["data_slice_end_frac"]))
    df = df.iloc[start:end].copy()

    ts_col = infer_timestamp_column(df)
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.round('min')
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df[~df["timestamp"].duplicated(keep="last")].set_index("timestamp")

    midpoint_col = choose_existing_column(df, ["midpoint", "mid", "price"], "midpoint")
    spread_col = choose_existing_column(df, ["spread"], "spread")
    buys_col = choose_existing_column(df, ["buys"], "buys")
    sells_col = choose_existing_column(df, ["sells"], "sells")

    book_levels = int(cfg["book_levels"])
    standardized = pd.DataFrame(index=df.index)
    standardized[f"mid_{asset}"] = pd.to_numeric(df[midpoint_col], errors="coerce")
    standardized[f"spread_{asset}"] = pd.to_numeric(df[spread_col], errors="coerce")
    standardized[f"buys_{asset}"] = pd.to_numeric(df[buys_col], errors="coerce")
    standardized[f"sells_{asset}"] = pd.to_numeric(df[sells_col], errors="coerce")

    for level in range(book_levels):
        bid_col = infer_book_column(df, "bids", level)
        ask_col = infer_book_column(df, "asks", level)
        standardized[f"bids_notional_{asset}_{level}"] = pd.to_numeric(df[bid_col], errors="coerce")
        standardized[f"asks_notional_{asset}_{level}"] = pd.to_numeric(df[ask_col], errors="coerce")

    standardized = standardized.replace([np.inf, -np.inf], np.nan)
    standardized = standardized.dropna()
    return standardized


def load_and_align_assets(cfg: Dict[str, Any]) -> pd.DataFrame:
    aligned: Optional[pd.DataFrame] = None
    for asset in ASSETS:
        one = load_one_asset_raw(asset, cfg)
        aligned = one if aligned is None else aligned.join(one, how="inner")

    if aligned is None:
        raise RuntimeError("No asset data loaded")

    aligned = aligned.sort_index()
    aligned = aligned[~aligned.index.duplicated(keep="last")].copy()

    for asset in ASSETS:
        mid = aligned[f"mid_{asset}"].astype(float)
        aligned[f"lr_{asset}"] = np.log(mid).diff().fillna(0.0)

    aligned = aligned.reset_index().rename(columns={"index": "timestamp"})
    return aligned


df = load_and_align_assets(CFG)

print("Loaded aligned dataframe shape:", df.shape)
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
print(df.head(2))

# %% [markdown]
# ## Feature engineering
#
# Per-asset node features:
# - `lr_1bar`: current 1-bar log return
# - `rel_spread`: spread / midpoint
# - `log_buys`, `log_sells`
# - `flow_imbalance`
# - `depth_imbalance_total`
# - top-book level imbalances
# - near/far depth ratios
# - near/far depth imbalances
#
# Importantly, order-book fields are called **notional**, not “volume”, because the old
# schema used notional columns.

# %%
EPS = 1e-12


def safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def build_node_feature_tensor(df_: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    book_levels = int(cfg["book_levels"])
    top_levels = int(cfg["top_levels"])
    near_levels = int(cfg["near_levels"])

    if top_levels > book_levels:
        raise ValueError("top_levels must be <= book_levels")
    if near_levels >= book_levels:
        raise ValueError("near_levels must be < book_levels")

    feature_names = [
        "lr_1bar",
        "rel_spread",
        "log_buys",
        "log_sells",
        "flow_imbalance",
        "depth_imbalance_total",
        "top_imbalance_0",
        "top_imbalance_1",
        "top_imbalance_2",
        "top_imbalance_3",
        "top_imbalance_4",
        "bid_near_far_ratio",
        "ask_near_far_ratio",
        "depth_imbalance_near",
        "depth_imbalance_far",
    ]

    all_assets = []
    for asset in ASSETS:
        lr = df_[f"lr_{asset}"].to_numpy(dtype=np.float32)
        mid = df_[f"mid_{asset}"].to_numpy(dtype=np.float32)
        spread = df_[f"spread_{asset}"].to_numpy(dtype=np.float32)
        buys = df_[f"buys_{asset}"].to_numpy(dtype=np.float32)
        sells = df_[f"sells_{asset}"].to_numpy(dtype=np.float32)

        rel_spread = spread / (mid + EPS)
        log_buys = safe_log1p(buys).astype(np.float32)
        log_sells = safe_log1p(sells).astype(np.float32)
        flow_imb = ((buys - sells) / (buys + sells + EPS)).astype(np.float32)

        bids = np.stack(
            [df_[f"bids_notional_{asset}_{i}"].to_numpy(dtype=np.float32) for i in range(book_levels)],
            axis=1,
        )
        asks = np.stack(
            [df_[f"asks_notional_{asset}_{i}"].to_numpy(dtype=np.float32) for i in range(book_levels)],
            axis=1,
        )

        bid_total = bids.sum(axis=1)
        ask_total = asks.sum(axis=1)
        depth_imb_total = ((bid_total - ask_total) / (bid_total + ask_total + EPS)).astype(np.float32)

        top_imbals = []
        for i in range(top_levels):
            bi = bids[:, i]
            ai = asks[:, i]
            top_imbals.append(((bi - ai) / (bi + ai + EPS)).astype(np.float32))
        top_imbals = np.stack(top_imbals, axis=1)

        bid_near = bids[:, :near_levels].sum(axis=1)
        ask_near = asks[:, :near_levels].sum(axis=1)
        bid_far = bids[:, near_levels:].sum(axis=1)
        ask_far = asks[:, near_levels:].sum(axis=1)

        bid_near_far_ratio = (bid_near / (bid_far + EPS)).astype(np.float32)
        ask_near_far_ratio = (ask_near / (ask_far + EPS)).astype(np.float32)
        depth_imb_near = ((bid_near - ask_near) / (bid_near + ask_near + EPS)).astype(np.float32)
        depth_imb_far = ((bid_far - ask_far) / (bid_far + ask_far + EPS)).astype(np.float32)

        asset_features = np.column_stack(
            [
                lr,
                rel_spread,
                log_buys,
                log_sells,
                flow_imb,
                depth_imb_total,
                top_imbals[:, 0],
                top_imbals[:, 1],
                top_imbals[:, 2],
                top_imbals[:, 3],
                top_imbals[:, 4],
                bid_near_far_ratio,
                ask_near_far_ratio,
                depth_imb_near,
                depth_imb_far,
            ]
        ).astype(np.float32)

        all_assets.append(asset_features)

    x_node = np.stack(all_assets, axis=1).astype(np.float32)  # [T, N, F]
    x_node = np.nan_to_num(x_node, nan=0.0, posinf=0.0, neginf=0.0)
    return x_node, feature_names


X_NODE_RAW, NODE_FEATURE_NAMES = build_node_feature_tensor(df, CFG)
print("X_NODE_RAW shape:", X_NODE_RAW.shape)
print("NODE_FEATURE_NAMES:", NODE_FEATURE_NAMES)

# %% [markdown]
# ## Edge features
#
# For each directed edge, we compute a vector of past-only rolling lead-lag correlations
# between source and destination returns.
#
# Important note:
# - these edge features are **not** a full temporal edge sequence inside the model
# - the model uses **only the last edge snapshot at prediction time**
# - this is a deliberate simplification for a clean baseline

# %%
def fisher_z_transform(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -0.999999, 0.999999)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def build_edge_feature_tensor(
    df_: pd.DataFrame,
    assets: List[str],
    edge_list: List[Tuple[str, str]],
    windows: List[int],
    lags: List[int],
    use_fisher_z: bool = True,
) -> np.ndarray:
    t_len = len(df_)
    n_edges = len(edge_list)
    n_feat = len(windows) * len(lags)

    out = np.zeros((t_len, n_edges, n_feat), dtype=np.float32)
    lr_map = {asset: df_[f"lr_{asset}"].astype(float) for asset in assets}

    for e_idx, (src, dst) in enumerate(edge_list):
        if src == dst:
            out[:, e_idx, :] = 1.0
            continue

        src_series = lr_map[src]
        dst_series = lr_map[dst]

        feat_pos = 0
        for lag in lags:
            shifted_src = src_series.shift(int(lag)) if int(lag) > 0 else src_series
            for window in windows:
                min_periods = max(3, int(window) // 2)
                corr = shifted_src.rolling(window=int(window), min_periods=min_periods).corr(dst_series)
                arr = corr.to_numpy(dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if use_fisher_z:
                    arr = fisher_z_transform(arr)
                out[:, e_idx, feat_pos] = arr.astype(np.float32)
                feat_pos += 1

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


EDGE_FEATURE_RAW = build_edge_feature_tensor(
    df_=df,
    assets=ASSETS,
    edge_list=EDGE_LIST,
    windows=EDGE_CORR_WINDOWS,
    lags=EDGE_CORR_LAGS,
    use_fisher_z=bool(CFG["edge_use_fisher_z"]),
)

print("EDGE_FEATURE_RAW shape:", EDGE_FEATURE_RAW.shape)

# %% [markdown]
# ## Target construction
#
# Primary target:
# - **ETH forward log return over exactly 5 minutes of clock time**
#
# Auxiliary evaluation labels derived from the same target:
# - `direction_label`: whether the future ETH return is positive
# - `trade_label`: whether the future absolute ETH return exceeds a transaction-cost-aware threshold
#
# Definitions:
# - `dir_auc` uses the raw regression score `pred_return`
# - `trade_auc` uses the magnitude score `abs(pred_return)`
# - these labels are **evaluation-only**, not training targets

# %%
def fixed_horizon_forward_log_return(lr: np.ndarray, horizon_bars: int) -> np.ndarray:
    lr = np.asarray(lr, dtype=np.float64)
    out = np.full(len(lr), np.nan, dtype=np.float32)

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    csum = np.cumsum(np.insert(lr, 0, 0.0))
    last_valid_t = len(lr) - horizon_bars - 1
    if last_valid_t < 0:
        return out

    idx = np.arange(0, last_valid_t + 1)
    future_sum = csum[idx + horizon_bars + 1] - csum[idx + 1]
    out[idx] = future_sum.astype(np.float32)
    return out


def round_trip_cost_as_log_return(cost_bps_per_side: float) -> float:
    return 2.0 * float(cost_bps_per_side) * 1e-4


ETH_LR = df[f"lr_{TARGET_ASSET}"].to_numpy(dtype=np.float64)
Y_RET = fixed_horizon_forward_log_return(ETH_LR, horizon_bars=HORIZON_BARS)

TRADE_LABEL_ABS_RETURN_THRESHOLD = (
    round_trip_cost_as_log_return(CFG["cost_bps_per_side"])
    + float(CFG["trade_label_buffer_bps"]) * 1e-4
)

# Direction label: 1 if future return > 0, 0 if future return < 0, NaN if exactly 0 or invalid.
Y_DIR = np.full(len(Y_RET), np.nan, dtype=np.float32)
valid_dir_pos = np.isfinite(Y_RET) & (Y_RET > 0.0)
valid_dir_neg = np.isfinite(Y_RET) & (Y_RET < 0.0)
Y_DIR[valid_dir_pos] = 1.0
Y_DIR[valid_dir_neg] = 0.0

# Trade label: 1 if |future return| exceeds round-trip cost-aware threshold, else 0.
Y_TRADE = np.full(len(Y_RET), np.nan, dtype=np.float32)
valid_trade = np.isfinite(Y_RET)
Y_TRADE[valid_trade] = (np.abs(Y_RET[valid_trade]) > TRADE_LABEL_ABS_RETURN_THRESHOLD).astype(np.float32)

print("Y_RET finite count:", int(np.isfinite(Y_RET).sum()))
print("TRADE_LABEL_ABS_RETURN_THRESHOLD:", TRADE_LABEL_ABS_RETURN_THRESHOLD)

# %% [markdown]
# ## Valid sample range
#
# A sample at time `t` is valid only if:
# - its full lookback window exists
# - its full fixed-horizon target exists
#
# `sample_t[i]` maps sample index `i` to raw time index `t`.

# %%
T = len(df)
first_valid_t = LOOKBACK_BARS - 1
last_valid_t = T - HORIZON_BARS - 1

if last_valid_t < first_valid_t:
    raise RuntimeError(
        f"Not enough rows after slicing. Need at least lookback={LOOKBACK_BARS} and horizon={HORIZON_BARS}."
    )

SAMPLE_T = np.arange(first_valid_t, last_valid_t + 1, dtype=np.int64)
N_SAMPLES = len(SAMPLE_T)

print("T:", T)
print("first_valid_t:", first_valid_t)
print("last_valid_t:", last_valid_t)
print("N_SAMPLES:", N_SAMPLES)

# %% [markdown]
# ## Split generation with purge/embargo
#
# We first carve out a final untouched holdout.
#
# Importantly, we leave a **gap before the holdout** so that the pre-holdout region does not
# contain labels that overlap into holdout.
#
# Then, inside the pre-holdout region, walk-forward CV uses:
# - train
# - purge gap
# - validation
# - purge gap
# - test

# %%
def make_preholdout_and_holdout_split(
    n_samples: int,
    holdout_frac: float,
    gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < holdout_frac < 0.5):
        raise ValueError("holdout_frac must be in (0, 0.5)")

    holdout_n = max(1, int(round(n_samples * float(holdout_frac))))
    preholdout_n = n_samples - gap - holdout_n

    if preholdout_n <= 0:
        raise RuntimeError("Not enough samples for pre-holdout region after reserving gap and holdout.")

    idx_preholdout = np.arange(0, preholdout_n, dtype=np.int64)
    idx_holdout = np.arange(preholdout_n + gap, preholdout_n + gap + holdout_n, dtype=np.int64)

    if idx_holdout[-1] >= n_samples:
        raise RuntimeError("Holdout indices exceed available sample count.")

    return idx_preholdout, idx_holdout


def make_walk_forward_splits_with_gap(
    n_preholdout: int,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    step_window_frac: float,
    gap: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_min = max(1, int(round(n_preholdout * train_min_frac)))
    # Validation/test fractions are interpreted as full blocks that already include purge gaps:
    # block size = gap + effective window size.
    val_block_n = max(gap + 1, int(round(n_preholdout * val_window_frac)))
    test_block_n = max(gap + 1, int(round(n_preholdout * test_window_frac)))
    val_n = max(1, val_block_n - gap)
    test_n = max(1, test_block_n - gap)
    step_n = max(1, int(round(n_preholdout * step_window_frac)))

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    train_end = train_min

    while True:
        val_start = train_end + gap
        val_end = val_start + val_n
        test_start = val_end + gap
        test_end = test_start + test_n

        if test_end > n_preholdout:
            break

        idx_train = np.arange(0, train_end, dtype=np.int64)
        idx_val = np.arange(val_start, val_end, dtype=np.int64)
        idx_test = np.arange(test_start, test_end, dtype=np.int64)

        if len(idx_train) > 0 and len(idx_val) > 0 and len(idx_test) > 0:
            splits.append((idx_train, idx_val, idx_test))

        train_end += step_n

    return splits


IDX_PREHOLDOUT, IDX_HOLDOUT = make_preholdout_and_holdout_split(
    n_samples=N_SAMPLES,
    holdout_frac=float(CFG["final_holdout_frac"]),
    gap=PURGE_GAP_BARS,
)

WALK_FORWARD_SPLITS = make_walk_forward_splits_with_gap(
    n_preholdout=len(IDX_PREHOLDOUT),
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    step_window_frac=float(CFG["step_window_frac"]),
    gap=PURGE_GAP_BARS,
)

if len(WALK_FORWARD_SPLITS) == 0:
    raise RuntimeError("No valid walk-forward splits were created. Adjust split fractions or data size.")

print("Pre-holdout sample count:", len(IDX_PREHOLDOUT))
print("Holdout sample count:", len(IDX_HOLDOUT))
print("Number of CV folds:", len(WALK_FORWARD_SPLITS))
for i, (tr, va, te) in enumerate(WALK_FORWARD_SPLITS, start=1):
    print(f"Fold {i}: train={len(tr)}, val={len(va)}, test={len(te)}")

# %% [markdown]
# ## Dataset and DataLoaders
#
# Each sample returns:
# - node feature sequence: `[lookback, n_nodes, n_features]`
# - last edge snapshot: `[n_edges, n_edge_features]`
# - fixed-horizon target return

# %%
class TemporalGraphDataset(Dataset):
    def __init__(
        self,
        x_node: np.ndarray,
        x_edge: np.ndarray,
        y_ret: np.ndarray,
        sample_t: np.ndarray,
        sample_indices: np.ndarray,
        lookback_bars: int,
    ):
        self.x_node = x_node
        self.x_edge = x_edge
        self.y_ret = y_ret
        self.sample_t = sample_t.astype(np.int64)
        self.sample_indices = sample_indices.astype(np.int64)
        self.lookback_bars = int(lookback_bars)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, i: int):
        sample_idx = int(self.sample_indices[i])
        t = int(self.sample_t[sample_idx])
        start = t - self.lookback_bars + 1

        x_seq = self.x_node[start:t + 1]     # [L, N, F]
        e_last = self.x_edge[t]              # [E, D]
        y = self.y_ret[t]                    # scalar

        if not np.isfinite(y):
            raise RuntimeError(f"Encountered invalid target at t={t}")

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(e_last),
            torch.tensor(float(y), dtype=torch.float32),
            torch.tensor(sample_idx, dtype=torch.long),
        )


def temporal_graph_collate(batch):
    x_seq, e_last, y, sample_idx = zip(*batch)
    return (
        torch.stack(x_seq, dim=0),
        torch.stack(e_last, dim=0),
        torch.stack(y, dim=0),
        torch.stack(sample_idx, dim=0),
    )

# %% [markdown]
# ## Scaling utilities
#
# The scaler is fit only on timestamps that belong to the training region for each fold.
# Because the model uses lookback windows, fitting on all timestamps up to the last train time
# is leakage-safe and matches what would be available at training time.

# %%
def fit_robust_scaler_train_only(
    raw_array: np.ndarray,
    sample_t: np.ndarray,
    train_sample_indices: np.ndarray,
    max_abs_value: float,
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if raw_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={raw_array.shape}")

    last_train_t = int(sample_t[int(train_sample_indices[-1])])
    train_slice = raw_array[: last_train_t + 1]
    flat_train = train_slice.reshape(-1, raw_array.shape[-1])

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(float(q_low), float(q_high)),
    )
    scaler.fit(flat_train)

    flat_all = raw_array.reshape(-1, raw_array.shape[-1])
    scaled_all = scaler.transform(flat_all).reshape(raw_array.shape).astype(np.float32)
    scaled_all = np.clip(scaled_all, -float(max_abs_value), float(max_abs_value))
    scaled_all = np.nan_to_num(scaled_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {
        "center_": scaler.center_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "max_abs_value": float(max_abs_value),
    }
    return scaled_all, params


def apply_robust_scaler_params(raw_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(params["center_"], dtype=np.float32)
    scale = np.asarray(params["scale_"], dtype=np.float32)
    max_abs_value = float(params["max_abs_value"])

    flat = raw_array.reshape(-1, raw_array.shape[-1]).astype(np.float32)
    flat = (flat - center) / (scale + 1e-12)
    flat = np.clip(flat, -max_abs_value, max_abs_value)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.reshape(raw_array.shape).astype(np.float32)

# %% [markdown]
# ## Model definition
#
# Honest naming:
# - this is **not** graph attention
# - temporal part = **causal Transformer encoder**
# - graph part = **support-mixed graph propagation**
#
# The dynamic adjacency uses the **last edge feature snapshot only**.

# %%
class GraphPropagationLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H]
        # adj: [B, N, N]
        msg = torch.einsum("bij,bjh->bih", adj, h)
        out = self.self_lin(h) + self.msg_lin(msg)
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class TemporalGraphBaseline(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        target_node: int,
        lookback_bars: int,
        cfg: Dict[str, Any],
    ):
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        num_heads = int(cfg["transformer_heads"])
        num_layers = int(cfg["transformer_layers"])
        ff_mult = int(cfg["transformer_ff_mult"])
        graph_layers = int(cfg["graph_layers"])
        edge_hidden_dim = int(cfg["edge_hidden_dim"])
        adaptive_adj_dim = int(cfg["adaptive_adj_dim"])
        dropout = float(cfg["dropout"])

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")

        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)
        self.lookback_bars = int(lookback_bars)

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.lookback_bars, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
        )

        self.adaptive_src = nn.Parameter(torch.randn(self.n_nodes, adaptive_adj_dim) * 0.02)
        self.adaptive_dst = nn.Parameter(torch.randn(self.n_nodes, adaptive_adj_dim) * 0.02)

        # Support mix:
        # 0 = identity/self support
        # 1 = dynamic edge-conditioned support
        # 2 = learned adaptive support
        self.support_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        self.graph_layers = nn.ModuleList(
            [GraphPropagationLayer(hidden_dim, dropout) for _ in range(graph_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        identity_adj = torch.eye(self.n_nodes, dtype=torch.float32)
        self.register_buffer("identity_adj", identity_adj)

        allowed_mask = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float32)
        for src, dst in EDGE_INDEX.tolist():
            allowed_mask[src, dst] = 1.0
        self.register_buffer("allowed_mask", allowed_mask)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True entries are masked.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _dynamic_adjacency(self, edge_last: torch.Tensor) -> torch.Tensor:
        # edge_last: [B, E, D]
        bsz = edge_last.size(0)
        edge_logits = self.edge_mlp(edge_last).squeeze(-1)         # [B, E]
        edge_weights = F.softplus(edge_logits) + 1e-6              # positive weights

        adj = edge_last.new_zeros((bsz, self.n_nodes, self.n_nodes))
        src_idx = EDGE_INDEX[:, 0].to(edge_last.device)
        dst_idx = EDGE_INDEX[:, 1].to(edge_last.device)
        adj[:, src_idx, dst_idx] = edge_weights

        # Mask to allowed edges and row-normalize.
        adj = adj * self.allowed_mask.to(edge_last.device)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj

    def _adaptive_adjacency(self, batch_size: int, device: torch.device) -> torch.Tensor:
        logits = self.adaptive_src @ self.adaptive_dst.T
        logits = logits.masked_fill(self.allowed_mask.to(device) == 0.0, float("-inf"))
        adj = torch.softmax(logits, dim=-1)
        return adj.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x_seq: torch.Tensor, edge_last: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, L, N, F]
        # edge_last: [B, E, D]
        bsz, seq_len, n_nodes, node_dim = x_seq.shape
        assert n_nodes == self.n_nodes, f"Expected {self.n_nodes} nodes, got {n_nodes}"

        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        edge_last = torch.nan_to_num(edge_last, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.node_proj(x_seq)  # [B, L, N, H]
        h = h.permute(0, 2, 1, 3).contiguous().view(bsz * n_nodes, seq_len, -1)
        h = h + self.pos_emb[:, :seq_len, :]

        causal_mask = self._causal_mask(seq_len=seq_len, device=x_seq.device)
        h = self.temporal_encoder(h, mask=causal_mask)
        h_last = h[:, -1, :].view(bsz, n_nodes, -1)  # [B, N, H]

        adj_dynamic = self._dynamic_adjacency(edge_last)
        adj_adaptive = self._adaptive_adjacency(batch_size=bsz, device=x_seq.device)
        adj_identity = self.identity_adj.to(x_seq.device).unsqueeze(0).expand(bsz, -1, -1)

        support_mix = torch.softmax(self.support_logits, dim=0)
        adj = (
            support_mix[0] * adj_identity
            + support_mix[1] * adj_dynamic
            + support_mix[2] * adj_adaptive
        )
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        for layer in self.graph_layers:
            h_last = layer(h_last, adj)

        eth_repr = h_last[:, self.target_node, :]
        pred = self.head(eth_repr).squeeze(-1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred

# %% [markdown]
# ## Loss, prediction, and evaluation helpers

# %%
def regression_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    beta = float(cfg["huber_beta"])
    return F.smooth_l1_loss(pred.view(-1), target.view(-1), beta=beta)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def ic_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_roc_auc(y_true_binary: np.ndarray, score: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y_true_binary) & np.isfinite(score)
    y = y_true_binary[mask]
    s = score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


@torch.no_grad()
def predict_on_indices(
    model: nn.Module,
    x_node_scaled: np.ndarray,
    x_edge_scaled: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    ds = TemporalGraphDataset(
        x_node=x_node_scaled,
        x_edge=x_edge_scaled,
        y_ret=Y_RET,
        sample_t=SAMPLE_T,
        sample_indices=indices,
        lookback_bars=LOOKBACK_BARS,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_graph_collate,
    )

    model.eval()
    preds, targets, sample_positions = [], [], []

    for x_seq, e_last, y, sample_idx in loader:
        x_seq = x_seq.to(DEVICE).float()
        e_last = e_last.to(DEVICE).float()
        y = y.to(DEVICE).float()

        pred = model(x_seq, e_last)

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        sample_positions.append(sample_idx.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0).astype(np.float64)
    y_arr = np.concatenate(targets, axis=0).astype(np.float64)
    sample_positions = np.concatenate(sample_positions, axis=0).astype(np.int64)

    return {
        "pred": pred_arr,
        "target": y_arr,
        "sample_idx": sample_positions,
    }


def compute_benchmark_predictions(indices: np.ndarray) -> Dict[str, np.ndarray]:
    raw_t = SAMPLE_T[indices.astype(np.int64)]
    y_true = Y_RET[raw_t].astype(np.float64)
    eth_lr_now = df[f"lr_{TARGET_ASSET}"].to_numpy(dtype=np.float64)[raw_t]

    pred_zero = np.zeros_like(y_true, dtype=np.float64)
    pred_last = eth_lr_now * float(HORIZON_BARS)

    return {
        "y_true": y_true,
        "pred_zero": pred_zero,
        "pred_last": pred_last,
    }

# %% [markdown]
# ## Directional metrics and auxiliary AUCs
#
# Definitions:
# - `dir_auc`:
#   - label = `1` if future ETH return > 0, `0` if future ETH return < 0
#   - score = raw predicted ETH return
#
# - `trade_auc`:
#   - label = `1` if `abs(future ETH return)` exceeds the cost-aware threshold
#   - score = `abs(predicted ETH return)`
#
# Signal-level precision metrics are computed after applying the chosen validation threshold.

# %%
def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal_threshold: float,
    trade_label_abs_threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    dir_label = np.full(len(y_true), np.nan, dtype=np.float64)
    dir_label[y_true > 0.0] = 1.0
    dir_label[y_true < 0.0] = 0.0

    trade_label = (np.abs(y_true) > float(trade_label_abs_threshold)).astype(np.float64)

    dir_auc = safe_roc_auc(dir_label, y_pred)
    trade_auc = safe_roc_auc(trade_label, np.abs(y_pred))

    long_mask = y_pred >= float(signal_threshold)
    short_mask = y_pred <= -float(signal_threshold)
    active_mask = long_mask | short_mask

    true_sign = np.zeros(len(y_true), dtype=np.int8)
    true_sign[y_true > 0.0] = 1
    true_sign[y_true < 0.0] = -1

    pred_sign = np.zeros(len(y_true), dtype=np.int8)
    pred_sign[long_mask] = 1
    pred_sign[short_mask] = -1

    sign_acc = float((pred_sign[active_mask] == true_sign[active_mask]).mean()) if active_mask.any() else float("nan")
    long_precision = float((true_sign[long_mask] == 1).mean()) if long_mask.any() else float("nan")
    short_precision = float((true_sign[short_mask] == -1).mean()) if short_mask.any() else float("nan")
    coverage = float(active_mask.mean()) if len(active_mask) else float("nan")

    return {
        "dir_auc": dir_auc,
        "trade_auc": trade_auc,
        "sign_accuracy": sign_acc,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "coverage": coverage,
    }

# %% [markdown]
# ## Sequential non-overlapping backtest
#
# This is the critical correction versus the old notebook.
#
# Backtest logic:
# - scan timestamps in order
# - when flat:
#   - open long if prediction >= threshold
#   - open short if prediction <= -threshold
# - hold for exactly `horizon_bars`
# - close
# - advance directly to the first timestamp after the closed trade
#
# Because each trade holds the fixed horizon and the index jumps by that horizon,
# **positions cannot overlap**.
#
# Transaction costs:
# - `cost_bps_per_side` is interpreted as **one-way** cost
# - net PnL subtracts a **round-trip cost** = `2 * cost_bps_per_side * 1e-4`

# %%
def sequential_fixed_horizon_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal_threshold: float,
    horizon_bars: int,
    cost_bps_per_side: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        empty = pd.DataFrame(columns=["entry_idx", "side", "pred", "future_return", "gross_pnl", "net_pnl"])
        return {
            "gross_pnl": float("nan"),
            "net_pnl": float("nan"),
            "pnl_per_trade": float("nan"),
            "n_trades": 0,
            "trade_rate": float("nan"),
            "sharpe_like": float("nan"),
        }, empty

    round_trip_cost = round_trip_cost_as_log_return(cost_bps_per_side)
    i = 0
    records: List[Dict[str, Any]] = []

    while i < n:
        score = float(y_pred[i])

        if score >= float(signal_threshold):
            side = 1
        elif score <= -float(signal_threshold):
            side = -1
        else:
            i += 1
            continue

        realized_return = float(y_true[i])
        gross_pnl = float(side * realized_return)
        net_pnl = float(gross_pnl - round_trip_cost)

        records.append(
            {
                "entry_idx": i,
                "side": side,
                "pred": score,
                "future_return": realized_return,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
            }
        )

        i += int(horizon_bars)

    trades_df = pd.DataFrame(records)
    n_trades = int(len(trades_df))
    gross_pnl = float(trades_df["gross_pnl"].sum()) if n_trades else 0.0
    net_pnl = float(trades_df["net_pnl"].sum()) if n_trades else 0.0
    pnl_per_trade = float(net_pnl / n_trades) if n_trades else float("nan")

    if n_trades >= 2 and trades_df["net_pnl"].std(ddof=1) > 1e-12:
        sharpe_like = float(
            trades_df["net_pnl"].mean() / trades_df["net_pnl"].std(ddof=1) * np.sqrt(n_trades)
        )
    else:
        sharpe_like = float("nan")

    metrics = {
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "pnl_per_trade": pnl_per_trade,
        "n_trades": n_trades,
        "trade_rate": float(n_trades / n),
        "sharpe_like": sharpe_like,
    }
    return metrics, trades_df

# %% [markdown]
# ## Validation-only threshold selection
#
# The signal threshold is selected using only the validation set and the same
# non-overlapping sequential backtest used later on test/holdout.

# %%
def build_threshold_grid(y_pred: np.ndarray, cfg: Dict[str, Any]) -> List[float]:
    abs_pred = np.abs(np.asarray(y_pred, dtype=np.float64))
    abs_pred = abs_pred[np.isfinite(abs_pred)]

    thresholds = set(float(x) for x in cfg["threshold_grid_abs_return"])
    if len(abs_pred):
        for q in cfg["threshold_grid_quantiles"]:
            thresholds.add(float(np.quantile(abs_pred, float(q))))

    cleaned = sorted(x for x in thresholds if x >= 0.0)
    out: List[float] = []
    for x in cleaned:
        if not out or abs(x - out[-1]) > 1e-12:
            out.append(float(x))
    return out


def sweep_validation_thresholds(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    thresholds = build_threshold_grid(y_pred_val, cfg)

    for thr in thresholds:
        bt_metrics, _ = sequential_fixed_horizon_backtest(
            y_true=y_true_val,
            y_pred=y_pred_val,
            signal_threshold=float(thr),
            horizon_bars=HORIZON_BARS,
            cost_bps_per_side=float(cfg["cost_bps_per_side"]),
        )
        rows.append({"signal_threshold": float(thr), **bt_metrics})

    sweep_df = pd.DataFrame(rows)
    if len(sweep_df) == 0:
        raise RuntimeError("Threshold sweep produced no rows.")

    min_trades = int(cfg["min_validation_trades"])
    feasible = sweep_df[sweep_df["n_trades"] >= min_trades].copy()
    if len(feasible) == 0:
        feasible = sweep_df.copy()

    feasible = feasible.sort_values(
        by=["net_pnl", "pnl_per_trade", "signal_threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return feasible

# %% [markdown]
# ## End-to-end metric package for a split

# %%
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal_threshold: float,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    regression = {
        "rmse": rmse_np(y_true, y_pred),
        "mae": mae_np(y_true, y_pred),
        "ic": ic_np(y_true, y_pred),
    }

    classification = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        signal_threshold=float(signal_threshold),
        trade_label_abs_threshold=TRADE_LABEL_ABS_RETURN_THRESHOLD,
    )

    backtest_metrics, trades_df = sequential_fixed_horizon_backtest(
        y_true=y_true,
        y_pred=y_pred,
        signal_threshold=float(signal_threshold),
        horizon_bars=HORIZON_BARS,
        cost_bps_per_side=float(cfg["cost_bps_per_side"]),
    )

    return {
        **regression,
        **classification,
        **backtest_metrics,
        "trades_df": trades_df,
    }

# %% [markdown]
# ## Training loop

# %%
@dataclass
class SplitArtifacts:
    model_state: Dict[str, torch.Tensor]
    node_scaler_params: Dict[str, Any]
    edge_scaler_params: Dict[str, Any]
    best_epoch: int
    best_val_rmse: float
    signal_threshold: float
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    val_predictions: Dict[str, np.ndarray]
    test_predictions: Dict[str, np.ndarray]


def train_one_split(
    split_name: str,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict[str, Any],
) -> SplitArtifacts:
    # Train-only scaling
    x_node_scaled, node_scaler_params = fit_robust_scaler_train_only(
        raw_array=X_NODE_RAW,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_node_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )
    x_edge_scaled, edge_scaler_params = fit_robust_scaler_train_only(
        raw_array=EDGE_FEATURE_RAW,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_edge_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )

    train_ds = TemporalGraphDataset(
        x_node=x_node_scaled,
        x_edge=x_edge_scaled,
        y_ret=Y_RET,
        sample_t=SAMPLE_T,
        sample_indices=idx_train,
        lookback_bars=LOOKBACK_BARS,
    )
    val_ds = TemporalGraphDataset(
        x_node=x_node_scaled,
        x_edge=x_edge_scaled,
        y_ret=Y_RET,
        sample_t=SAMPLE_T,
        sample_indices=idx_val,
        lookback_bars=LOOKBACK_BARS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=temporal_graph_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_graph_collate,
    )

    model = TemporalGraphBaseline(
        node_dim=X_NODE_RAW.shape[-1],
        edge_dim=EDGE_FEATURE_RAW.shape[-1],
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        lookback_bars=LOOKBACK_BARS,
        cfg=cfg,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_epoch = -1
    best_val_rmse = float("inf")
    bad_epochs = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_losses: List[float] = []

        for x_seq, e_last, y, _sample_idx in train_loader:
            x_seq = x_seq.to(DEVICE).float()
            e_last = e_last.to(DEVICE).float()
            y = y.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, e_last)
            loss = regression_loss(pred, y, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_seq, e_last, y, _sample_idx in val_loader:
                x_seq = x_seq.to(DEVICE).float()
                e_last = e_last.to(DEVICE).float()
                y = y.to(DEVICE).float()

                pred = model(x_seq, e_last)
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(y.detach().cpu().numpy())

        val_pred_arr = np.concatenate(val_preds, axis=0).astype(np.float64)
        val_target_arr = np.concatenate(val_targets, axis=0).astype(np.float64)

        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_rmse = rmse_np(val_target_arr, val_pred_arr)
        val_mae = mae_np(val_target_arr, val_pred_arr)
        val_ic = ic_np(val_target_arr, val_pred_arr)

        scheduler.step(val_rmse)

        support_mix = torch.softmax(model.support_logits.detach().cpu(), dim=0).numpy().round(3).tolist()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[{split_name}] epoch={epoch:02d} "
            f"train_loss={train_loss_mean:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"val_mae={val_mae:.6f} "
            f"val_ic={val_ic:.4f} "
            f"lr={lr_now:.2e} "
            f"support_mix={support_mix}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = float(val_rmse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            print(f"[{split_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"[{split_name}] best_state is None")

    model.load_state_dict(best_state)

    val_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_edge_scaled=x_edge_scaled,
        indices=idx_val,
        batch_size=int(cfg["batch_size"]),
    )
    test_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_edge_scaled=x_edge_scaled,
        indices=idx_test,
        batch_size=int(cfg["batch_size"]),
    )

    threshold_sweep_df = sweep_validation_thresholds(
        y_true_val=val_pred_pack["target"],
        y_pred_val=val_pred_pack["pred"],
        cfg=cfg,
    )
    selected_threshold = float(threshold_sweep_df.iloc[0]["signal_threshold"])

    val_metrics = evaluate_predictions(
        y_true=val_pred_pack["target"],
        y_pred=val_pred_pack["pred"],
        signal_threshold=selected_threshold,
        cfg=cfg,
    )
    test_metrics = evaluate_predictions(
        y_true=test_pred_pack["target"],
        y_pred=test_pred_pack["pred"],
        signal_threshold=selected_threshold,
        cfg=cfg,
    )

    print(
        f"[{split_name}] best_epoch={best_epoch} "
        f"best_val_rmse={best_val_rmse:.6f} "
        f"selected_threshold={selected_threshold:.6f}"
    )
    print(
        f"[{split_name}] TEST "
        f"rmse={test_metrics['rmse']:.6f} "
        f"mae={test_metrics['mae']:.6f} "
        f"ic={test_metrics['ic']:.4f} "
        f"dir_auc={test_metrics['dir_auc']:.4f} "
        f"trade_auc={test_metrics['trade_auc']:.4f} "
        f"net_pnl={test_metrics['net_pnl']:.6f} "
        f"n_trades={test_metrics['n_trades']}"
    )

    return SplitArtifacts(
        model_state=copy.deepcopy(model.state_dict()),
        node_scaler_params=node_scaler_params,
        edge_scaler_params=edge_scaler_params,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        signal_threshold=selected_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_predictions=val_pred_pack,
        test_predictions=test_pred_pack,
    )

# %% [markdown]
# ## Artifact saving and loading

# %%
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def save_scaler_npz(path: Path, params: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        center_=np.asarray(params["center_"], dtype=np.float32),
        scale_=np.asarray(params["scale_"], dtype=np.float32),
        max_abs_value=np.asarray([float(params["max_abs_value"])], dtype=np.float32),
    )


def load_scaler_npz(path: Path) -> Dict[str, Any]:
    data = np.load(str(path))
    return {
        "center_": data["center_"].astype(np.float32),
        "scale_": data["scale_"].astype(np.float32),
        "max_abs_value": float(data["max_abs_value"][0]),
    }


def save_bundle(
    bundle_dir: Path,
    bundle_name: str,
    model_state: Dict[str, torch.Tensor],
    node_scaler_params: Dict[str, Any],
    edge_scaler_params: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    weights_path = bundle_dir / f"{bundle_name}_weights.pt"
    node_scaler_path = bundle_dir / f"{bundle_name}_node_scaler.npz"
    edge_scaler_path = bundle_dir / f"{bundle_name}_edge_scaler.npz"
    meta_path = bundle_dir / f"{bundle_name}_meta.json"

    torch.save(model_state, str(weights_path))
    save_scaler_npz(node_scaler_path, node_scaler_params)
    save_scaler_npz(edge_scaler_path, edge_scaler_params)

    meta_payload = {
        "bundle_name": bundle_name,
        "weights_file": weights_path.name,
        "node_scaler_file": node_scaler_path.name,
        "edge_scaler_file": edge_scaler_path.name,
        "cfg": _jsonable(CFG),
        "freq": FREQ,
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_bars": HORIZON_BARS,
        "lookback_bars": LOOKBACK_BARS,
        "assets": ASSETS,
        "target_asset": TARGET_ASSET,
        **_jsonable(meta),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    return {
        "weights": weights_path,
        "node_scaler": node_scaler_path,
        "edge_scaler": edge_scaler_path,
        "meta": meta_path,
    }


def load_bundle(bundle_dir: Path, bundle_name: str) -> Dict[str, Any]:
    meta_path = bundle_dir / f"{bundle_name}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    weights_path = bundle_dir / meta["weights_file"]
    node_scaler_path = bundle_dir / meta["node_scaler_file"]
    edge_scaler_path = bundle_dir / meta["edge_scaler_file"]

    return {
        "meta": meta,
        "state_dict": torch.load(str(weights_path), map_location="cpu"),
        "node_scaler_params": load_scaler_npz(node_scaler_path),
        "edge_scaler_params": load_scaler_npz(edge_scaler_path),
    }

# %% [markdown]
# ## Saved-bundle evaluation helper

# %%
@torch.no_grad()
def evaluate_saved_bundle_on_indices(
    bundle_dir: Path,
    bundle_name: str,
    indices: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    loaded = load_bundle(bundle_dir, bundle_name)
    meta = loaded["meta"]

    model = TemporalGraphBaseline(
        node_dim=X_NODE_RAW.shape[-1],
        edge_dim=EDGE_FEATURE_RAW.shape[-1],
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        lookback_bars=int(meta["lookback_bars"]),
        cfg=meta["cfg"],
    ).to(DEVICE)
    model.load_state_dict(loaded["state_dict"])
    model.eval()

    x_node_scaled = apply_robust_scaler_params(X_NODE_RAW, loaded["node_scaler_params"])
    x_edge_scaled = apply_robust_scaler_params(EDGE_FEATURE_RAW, loaded["edge_scaler_params"])

    pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_edge_scaled=x_edge_scaled,
        indices=indices.astype(np.int64),
        batch_size=int(meta["cfg"]["batch_size"]),
    )
    threshold = float(meta["selected_signal_threshold"])
    metrics = evaluate_predictions(
        y_true=pred_pack["target"],
        y_pred=pred_pack["pred"],
        signal_threshold=threshold,
        cfg=meta["cfg"],
    )

    bm = compute_benchmark_predictions(indices.astype(np.int64))
    bench = {
        "zero_rmse": rmse_np(bm["y_true"], bm["pred_zero"]),
        "zero_mae": mae_np(bm["y_true"], bm["pred_zero"]),
        "zero_ic": ic_np(bm["y_true"], bm["pred_zero"]),
        "last_rmse": rmse_np(bm["y_true"], bm["pred_last"]),
        "last_mae": mae_np(bm["y_true"], bm["pred_last"]),
        "last_ic": ic_np(bm["y_true"], bm["pred_last"]),
    }

    print("\n" + "=" * 100)
    print(label)
    print(f"bundle_name={bundle_name}")
    print(
        f"rmse={metrics['rmse']:.6f} "
        f"mae={metrics['mae']:.6f} "
        f"ic={metrics['ic']:.4f} "
        f"dir_auc={metrics['dir_auc']:.4f} "
        f"trade_auc={metrics['trade_auc']:.4f}"
    )
    print(
        f"sign_accuracy={metrics['sign_accuracy']:.4f} "
        f"long_precision={metrics['long_precision']:.4f} "
        f"short_precision={metrics['short_precision']:.4f} "
        f"coverage={metrics['coverage']:.4f}"
    )
    print(
        f"gross_pnl={metrics['gross_pnl']:.6f} "
        f"net_pnl={metrics['net_pnl']:.6f} "
        f"pnl_per_trade={metrics['pnl_per_trade']:.6f} "
        f"n_trades={metrics['n_trades']} "
        f"trade_rate={metrics['trade_rate']:.4f} "
        f"sharpe_like={metrics['sharpe_like']:.4f}"
    )
    print(
        f"bench_zero_rmse={bench['zero_rmse']:.6f} "
        f"bench_last_rmse={bench['last_rmse']:.6f}"
    )

    return {
        "pred_pack": pred_pack,
        "metrics": metrics,
        "benchmarks": bench,
        "threshold": threshold,
    }

# %% [markdown]
# ## Walk-forward CV pipeline
#
# For each fold:
# - fit scalers on train only
# - train model on train
# - early stop on validation RMSE
# - choose signal threshold on validation only
# - report validation and test metrics
#
# We also save one bundle per fold and track the overall best fold by validation RMSE.

# %%
cv_rows: List[Dict[str, Any]] = []
fold_bundle_names: List[str] = []
best_cv_bundle_name: Optional[str] = None
best_cv_val_rmse = float("inf")

for fold_idx, (idx_train, idx_val, idx_test) in enumerate(WALK_FORWARD_SPLITS, start=1):
    print("\n" + "=" * 100)
    print(
        f"FOLD {fold_idx}/{len(WALK_FORWARD_SPLITS)} "
        f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
    )

    artifacts = train_one_split(
        split_name=f"fold_{fold_idx:02d}",
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        cfg=CFG,
    )

    bundle_name = f"fold_{fold_idx:02d}_best"
    fold_bundle_names.append(bundle_name)

    save_bundle(
        bundle_dir=ARTIFACT_DIR,
        bundle_name=bundle_name,
        model_state=artifacts.model_state,
        node_scaler_params=artifacts.node_scaler_params,
        edge_scaler_params=artifacts.edge_scaler_params,
        meta={
            "kind": "cv_fold_best",
            "fold_idx": fold_idx,
            "best_epoch": artifacts.best_epoch,
            "best_val_rmse": artifacts.best_val_rmse,
            "selected_signal_threshold": artifacts.signal_threshold,
            "idx_train": idx_train.tolist(),
            "idx_val": idx_val.tolist(),
            "idx_test": idx_test.tolist(),
        },
    )

    artifacts.val_metrics["trades_df"].to_csv(
        ARTIFACT_DIR / f"{bundle_name}_val_trades.csv",
        index=False,
    )
    artifacts.test_metrics["trades_df"].to_csv(
        ARTIFACT_DIR / f"{bundle_name}_test_trades.csv",
        index=False,
    )

    val_bm = compute_benchmark_predictions(idx_val)
    test_bm = compute_benchmark_predictions(idx_test)

    row = {
        "fold": fold_idx,
        "best_epoch": artifacts.best_epoch,
        "best_val_rmse_checkpoint": artifacts.best_val_rmse,
        "selected_signal_threshold": artifacts.signal_threshold,

        "val_rmse": artifacts.val_metrics["rmse"],
        "val_mae": artifacts.val_metrics["mae"],
        "val_ic": artifacts.val_metrics["ic"],
        "val_dir_auc": artifacts.val_metrics["dir_auc"],
        "val_trade_auc": artifacts.val_metrics["trade_auc"],
        "val_sign_accuracy": artifacts.val_metrics["sign_accuracy"],
        "val_long_precision": artifacts.val_metrics["long_precision"],
        "val_short_precision": artifacts.val_metrics["short_precision"],
        "val_coverage": artifacts.val_metrics["coverage"],
        "val_gross_pnl": artifacts.val_metrics["gross_pnl"],
        "val_net_pnl": artifacts.val_metrics["net_pnl"],
        "val_pnl_per_trade": artifacts.val_metrics["pnl_per_trade"],
        "val_n_trades": artifacts.val_metrics["n_trades"],
        "val_trade_rate": artifacts.val_metrics["trade_rate"],
        "val_sharpe_like": artifacts.val_metrics["sharpe_like"],
        "val_bench_zero_rmse": rmse_np(val_bm["y_true"], val_bm["pred_zero"]),
        "val_bench_last_rmse": rmse_np(val_bm["y_true"], val_bm["pred_last"]),

        "test_rmse": artifacts.test_metrics["rmse"],
        "test_mae": artifacts.test_metrics["mae"],
        "test_ic": artifacts.test_metrics["ic"],
        "test_dir_auc": artifacts.test_metrics["dir_auc"],
        "test_trade_auc": artifacts.test_metrics["trade_auc"],
        "test_sign_accuracy": artifacts.test_metrics["sign_accuracy"],
        "test_long_precision": artifacts.test_metrics["long_precision"],
        "test_short_precision": artifacts.test_metrics["short_precision"],
        "test_coverage": artifacts.test_metrics["coverage"],
        "test_gross_pnl": artifacts.test_metrics["gross_pnl"],
        "test_net_pnl": artifacts.test_metrics["net_pnl"],
        "test_pnl_per_trade": artifacts.test_metrics["pnl_per_trade"],
        "test_n_trades": artifacts.test_metrics["n_trades"],
        "test_trade_rate": artifacts.test_metrics["trade_rate"],
        "test_sharpe_like": artifacts.test_metrics["sharpe_like"],
        "test_bench_zero_rmse": rmse_np(test_bm["y_true"], test_bm["pred_zero"]),
        "test_bench_last_rmse": rmse_np(test_bm["y_true"], test_bm["pred_last"]),
    }
    cv_rows.append(row)

    if artifacts.best_val_rmse < best_cv_val_rmse:
        best_cv_val_rmse = float(artifacts.best_val_rmse)
        best_cv_bundle_name = bundle_name

if best_cv_bundle_name is None:
    raise RuntimeError("best_cv_bundle_name is None")

CV_RESULTS_DF = pd.DataFrame(cv_rows)
CV_RESULTS_DF.to_csv(ARTIFACT_DIR / "cv_results_summary.csv", index=False)

print("\n" + "=" * 100)
print("CV_RESULTS_DF")
print(CV_RESULTS_DF)
print("\nCV mean metrics:")
print(CV_RESULTS_DF.mean(numeric_only=True))
print("\nBest CV bundle:", best_cv_bundle_name)

# %% [markdown]
# ## Post-CV holdout evaluation
#
# This uses the **best CV-selected model bundle** exactly as saved from CV.
# There is **no retuning on the holdout**.

# %%
POST_CV_HOLDOUT = evaluate_saved_bundle_on_indices(
    bundle_dir=ARTIFACT_DIR,
    bundle_name=best_cv_bundle_name,
    indices=IDX_HOLDOUT,
    label="POST-CV HOLDOUT EVALUATION USING BEST CV-SELECTED MODEL",
)

POST_CV_HOLDOUT_SUMMARY_DF = pd.DataFrame(
    [
        {
            "model_name": best_cv_bundle_name,
            "rmse": POST_CV_HOLDOUT["metrics"]["rmse"],
            "mae": POST_CV_HOLDOUT["metrics"]["mae"],
            "ic": POST_CV_HOLDOUT["metrics"]["ic"],
            "dir_auc": POST_CV_HOLDOUT["metrics"]["dir_auc"],
            "trade_auc": POST_CV_HOLDOUT["metrics"]["trade_auc"],
            "sign_accuracy": POST_CV_HOLDOUT["metrics"]["sign_accuracy"],
            "long_precision": POST_CV_HOLDOUT["metrics"]["long_precision"],
            "short_precision": POST_CV_HOLDOUT["metrics"]["short_precision"],
            "coverage": POST_CV_HOLDOUT["metrics"]["coverage"],
            "gross_pnl": POST_CV_HOLDOUT["metrics"]["gross_pnl"],
            "net_pnl": POST_CV_HOLDOUT["metrics"]["net_pnl"],
            "pnl_per_trade": POST_CV_HOLDOUT["metrics"]["pnl_per_trade"],
            "n_trades": POST_CV_HOLDOUT["metrics"]["n_trades"],
            "trade_rate": POST_CV_HOLDOUT["metrics"]["trade_rate"],
            "sharpe_like": POST_CV_HOLDOUT["metrics"]["sharpe_like"],
            "bench_zero_rmse": POST_CV_HOLDOUT["benchmarks"]["zero_rmse"],
            "bench_last_rmse": POST_CV_HOLDOUT["benchmarks"]["last_rmse"],
            "selected_signal_threshold": POST_CV_HOLDOUT["threshold"],
        }
    ]
)
POST_CV_HOLDOUT_SUMMARY_DF.to_csv(ARTIFACT_DIR / "post_cv_holdout_summary.csv", index=False)

print("\nPOST_CV_HOLDOUT_SUMMARY_DF")
print(POST_CV_HOLDOUT_SUMMARY_DF)

# %% [markdown]
# ## Full-refit / production-style evaluation
#
# Final protocol:
# - use the pre-holdout region only
# - split it into:
#   - `train_final`
#   - purge gap
#   - `val_final`
# - then leave another purge gap before the untouched holdout
#
# Threshold selection again uses **validation only**.
# Holdout is evaluated once at the end.

# %%
def make_final_production_split(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    val_window_frac: float,
    gap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pre = len(idx_preholdout)
    val_n = max(1, int(round(n_pre * float(val_window_frac))))

    train_end = n_pre - gap - val_n
    if train_end <= 0:
        raise RuntimeError("Not enough pre-holdout samples for final production split.")

    idx_train_final = np.arange(0, train_end, dtype=np.int64)
    idx_val_final = np.arange(train_end + gap, train_end + gap + val_n, dtype=np.int64)

    if idx_val_final[-1] >= n_pre:
        raise RuntimeError("Final validation split exceeds pre-holdout region.")

    return idx_train_final, idx_val_final, idx_holdout.astype(np.int64)


IDX_TRAIN_FINAL, IDX_VAL_FINAL, IDX_TEST_FINAL = make_final_production_split(
    idx_preholdout=IDX_PREHOLDOUT,
    idx_holdout=IDX_HOLDOUT,
    val_window_frac=float(CFG["val_window_frac"]),
    gap=PURGE_GAP_BARS,
)

print("Production split sizes:")
print("train_final:", len(IDX_TRAIN_FINAL))
print("val_final  :", len(IDX_VAL_FINAL))
print("holdout    :", len(IDX_TEST_FINAL))

# %%
PRODUCTION_ARTIFACTS = train_one_split(
    split_name="production_refit",
    idx_train=IDX_TRAIN_FINAL,
    idx_val=IDX_VAL_FINAL,
    idx_test=IDX_TEST_FINAL,
    cfg=CFG,
)

PRODUCTION_BUNDLE_NAME = "production_best"

save_bundle(
    bundle_dir=ARTIFACT_DIR,
    bundle_name=PRODUCTION_BUNDLE_NAME,
    model_state=PRODUCTION_ARTIFACTS.model_state,
    node_scaler_params=PRODUCTION_ARTIFACTS.node_scaler_params,
    edge_scaler_params=PRODUCTION_ARTIFACTS.edge_scaler_params,
    meta={
        "kind": "production_best",
        "best_epoch": PRODUCTION_ARTIFACTS.best_epoch,
        "best_val_rmse": PRODUCTION_ARTIFACTS.best_val_rmse,
        "selected_signal_threshold": PRODUCTION_ARTIFACTS.signal_threshold,
        "idx_train": IDX_TRAIN_FINAL.tolist(),
        "idx_val": IDX_VAL_FINAL.tolist(),
        "idx_test": IDX_TEST_FINAL.tolist(),
    },
)

PRODUCTION_ARTIFACTS.val_metrics["trades_df"].to_csv(
    ARTIFACT_DIR / f"{PRODUCTION_BUNDLE_NAME}_val_trades.csv",
    index=False,
)
PRODUCTION_ARTIFACTS.test_metrics["trades_df"].to_csv(
    ARTIFACT_DIR / f"{PRODUCTION_BUNDLE_NAME}_holdout_trades.csv",
    index=False,
)

PRODUCTION_HOLDOUT = evaluate_saved_bundle_on_indices(
    bundle_dir=ARTIFACT_DIR,
    bundle_name=PRODUCTION_BUNDLE_NAME,
    indices=IDX_HOLDOUT,
    label="FULL-REFIT / PRODUCTION HOLDOUT EVALUATION",
)

PRODUCTION_HOLDOUT_SUMMARY_DF = pd.DataFrame(
    [
        {
            "model_name": PRODUCTION_BUNDLE_NAME,
            "rmse": PRODUCTION_HOLDOUT["metrics"]["rmse"],
            "mae": PRODUCTION_HOLDOUT["metrics"]["mae"],
            "ic": PRODUCTION_HOLDOUT["metrics"]["ic"],
            "dir_auc": PRODUCTION_HOLDOUT["metrics"]["dir_auc"],
            "trade_auc": PRODUCTION_HOLDOUT["metrics"]["trade_auc"],
            "sign_accuracy": PRODUCTION_HOLDOUT["metrics"]["sign_accuracy"],
            "long_precision": PRODUCTION_HOLDOUT["metrics"]["long_precision"],
            "short_precision": PRODUCTION_HOLDOUT["metrics"]["short_precision"],
            "coverage": PRODUCTION_HOLDOUT["metrics"]["coverage"],
            "gross_pnl": PRODUCTION_HOLDOUT["metrics"]["gross_pnl"],
            "net_pnl": PRODUCTION_HOLDOUT["metrics"]["net_pnl"],
            "pnl_per_trade": PRODUCTION_HOLDOUT["metrics"]["pnl_per_trade"],
            "n_trades": PRODUCTION_HOLDOUT["metrics"]["n_trades"],
            "trade_rate": PRODUCTION_HOLDOUT["metrics"]["trade_rate"],
            "sharpe_like": PRODUCTION_HOLDOUT["metrics"]["sharpe_like"],
            "bench_zero_rmse": PRODUCTION_HOLDOUT["benchmarks"]["zero_rmse"],
            "bench_last_rmse": PRODUCTION_HOLDOUT["benchmarks"]["last_rmse"],
            "selected_signal_threshold": PRODUCTION_HOLDOUT["threshold"],
        }
    ]
)
PRODUCTION_HOLDOUT_SUMMARY_DF.to_csv(ARTIFACT_DIR / "production_holdout_summary.csv", index=False)

print("\nPRODUCTION_HOLDOUT_SUMMARY_DF")
print(PRODUCTION_HOLDOUT_SUMMARY_DF)

# %% [markdown]
# ## Final summary tables

# %%
FINAL_SUMMARY_DF = pd.concat(
    [
        POST_CV_HOLDOUT_SUMMARY_DF.assign(evaluation_protocol="post_cv_best_model_on_holdout"),
        PRODUCTION_HOLDOUT_SUMMARY_DF.assign(evaluation_protocol="full_refit_production_on_holdout"),
    ],
    axis=0,
    ignore_index=True,
)

FINAL_SUMMARY_DF.to_csv(ARTIFACT_DIR / "final_summary.csv", index=False)

print("\n" + "=" * 100)
print("FINAL_SUMMARY_DF")
print(FINAL_SUMMARY_DF)

# %% [markdown]
# ## Notes on interpretation
#
# - `rmse`, `mae`, `ic` evaluate the regression forecast of the fixed-horizon ETH return
# - `dir_auc` evaluates ranking of positive vs negative future ETH returns
# - `trade_auc` evaluates ranking of tradeable vs non-tradeable future moves, where
#   tradeability means `abs(future return)` exceeds the round-trip-cost-aware threshold
# - `coverage` is the fraction of timestamps where `abs(prediction) >= selected_threshold`
# - `trade_rate` is the number of executed sequential non-overlapping trades divided by
#   the number of timestamps in the evaluated split
# - `gross_pnl` and `net_pnl` are sums of per-trade log-return PnL across the split
#
# The benchmark forecasts are:
# - `zero-return`: predict 0 future return
# - `last-return extrapolation`: predict current ETH 1-bar return times `horizon_bars`

# %%
print("Notebook build complete.")