# %% [markdown]
# # Thesis-Grade Temporal Multigraph Fusion Notebook
#
# This notebook implements a methodologically strict multigraph fusion model for
# ETH 5-minute forecasting under the thesis setting:
#
# **Real-Time Market Microstructure Modeling with Temporal Graph Neural Networks**
#
# Core principles enforced throughout:
#
# - Nodes = assets: ADA, BTC, ETH
# - Target asset = ETH
# - Primary target = fixed-horizon ETH forward log return over exactly 5 minutes
# - Walk-forward CV with purge gaps
# - Validation-only threshold selection
# - Post-CV holdout evaluation using the best CV-selected saved bundle
# - Production / full-refit protocol:
#     train_final -> purge gap -> val_final -> purge gap -> holdout
# - Artifact saving/loading
# - Real multigraph fusion:
#     price dependence + order flow + liquidity
# - Graph-operator ablations under the same pipeline:
#     edge_mpnn, rel_conv, rel_gatv2
#
# Important methodological note:
# This notebook intentionally refuses to silently round timestamps or silently
# accept irregular time indices. If the aligned data are not regular at the
# configured frequency, the notebook raises an error rather than quietly
# changing the time base.

# %%
import copy
import json
import math
import os
import random
import warnings
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

warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Imports, seeds, determinism

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
# Assumptions documented in code comments:
#
# - CSV files are expected at:
#     {data_dir}/{ASSET}_{freq}.csv
#   where freq is one of {"1sec", "1min", "5min"}.
# - The CSV schema is expected to be compatible with the baseline:
#     system_time / timestamp, midpoint, spread, buys, sells,
#     bids_notional_*, asks_notional_*
# - Order-book fields are treated honestly as *notional* if that is how they
#   exist in the CSVs; the notebook does not relabel them as "volume".
# - The default run includes the full operator ablation suite. To reduce runtime
#   during experimentation, set `ablation_fast_mode=True`.
# - The graph is intentionally tiny (3 nodes), so all graph operators are
#   implemented directly in PyTorch without external graph libraries.

# %%
CFG: Dict[str, Any] = {
    # Data
    "freq": "1min",  # one of: "1sec", "1min", "5min"
    "data_dir": "../dataset",
    "artifact_root": "./artifacts_multigraph_fusion",
    "assets": ["ADA", "BTC", "ETH"],
    "target_asset": "ETH",
    "data_slice_start_frac": 0.00,
    "data_slice_end_frac": 0.75,
    "final_holdout_frac": 0.10,

    # Horizon is fixed in clock time. Mapping is handled below.
    "horizon_minutes": 5,

    # Lookback by frequency
    "lookback_bars_by_freq": {
        "1sec": 300,
        "1min": 240,
        "5min": 120,
    },

    # Book / feature config
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # Rolling dependence config used for all relation channels
    # These are trailing windows and lags, so edge features remain past-only.
    "relation_windows_bars_by_freq": {
        "1sec": [60, 300, 900],
        "1min": [10, 30, 60],
        "5min": [6, 12, 24],
    },
    "relation_lags_bars": [0, 1, 2, 5],
    "use_fisher_z_for_corr": True,

    # Splits
    "train_min_frac": 0.50,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,
    "purge_gap_extra_bars": 0,

    # Scaling
    "max_abs_node_feature": 8.0,
    "max_abs_edge_feature": 6.0,
    "scaler_quantile_low": 5.0,
    "scaler_quantile_high": 95.0,

    # Model
    "graph_operator": "edge_mpnn",  # one of {"edge_mpnn", "rel_conv", "rel_gatv2"}
    "hidden_dim": 64,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "transformer_ff_mult": 4,
    "graph_layers": 2,
    "dropout": 0.15,
    "edge_hidden_dim": 64,
    "fusion_hidden_dim": 32,
    "gat_heads": 4,

    # Training
    "batch_size": 64,
    "epochs": 35,
    "patience": 6,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "huber_beta": 1e-3,

    # Trading / thresholding
    "cost_bps_per_side": 1.0,
    "trade_label_buffer_bps": 0.0,
    "threshold_grid_abs_return": [0.0, 0.0001, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030],
    "threshold_grid_quantiles": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    "min_validation_trades": 20,

    # Ablation
    "operator_candidates": ["edge_mpnn", "rel_conv", "rel_gatv2"],
    "run_full_operator_ablation": True,
    "ablation_fast_mode": False,
    "ablation_epochs": 20,
    "ablation_patience": 4,
}

ASSETS: List[str] = list(CFG["assets"])
TARGET_ASSET: str = str(CFG["target_asset"])
assert TARGET_ASSET in ASSETS, "Target asset must be one of the configured assets."
ASSET2IDX: Dict[str, int] = {asset: i for i, asset in enumerate(ASSETS)}
TARGET_NODE: int = ASSET2IDX[TARGET_ASSET]

ARTIFACT_ROOT = Path(CFG["artifact_root"])
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

print("ASSETS:", ASSETS)
print("TARGET_ASSET:", TARGET_ASSET)
print("ARTIFACT_ROOT:", ARTIFACT_ROOT.resolve())

# %% [markdown]
# ## Frequency and horizon helpers
#
# This section explicitly fixes the baseline horizon bug.
#
# The forecast horizon is always **exactly 5 minutes of clock time**.
# Therefore:
#
# - 1sec -> 300 bars
# - 1min -> 5 bars
# - 5min -> 1 bar
#
# This mapping is used consistently in:
#
# - target construction
# - purge-gap logic
# - non-overlapping backtest holding period
# - benchmark construction
# - holdout / production refit logic

# %%
def normalize_freq_name(freq: str) -> str:
    f = str(freq).strip().lower()
    alias_map = {
        "1s": "1sec",
        "1sec": "1sec",
        "1second": "1sec",
        "1m": "1min",
        "1min": "1min",
        "1minute": "1min",
        "5m": "5min",
        "5min": "5min",
        "5minute": "5min",
    }
    if f not in alias_map:
        raise ValueError(f"Unsupported frequency: {freq}")
    return alias_map[f]


def freq_to_seconds(freq: str) -> int:
    f = normalize_freq_name(freq)
    return {"1sec": 1, "1min": 60, "5min": 300}[f]


def expected_timedelta(freq: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=freq_to_seconds(freq))


def horizon_bars_from_clock_minutes(freq: str, horizon_minutes: int = 5) -> int:
    horizon_seconds = int(horizon_minutes) * 60
    bar_seconds = freq_to_seconds(freq)
    if horizon_seconds % bar_seconds != 0:
        raise ValueError(
            f"Horizon of {horizon_minutes} minutes is not an integer number of bars for freq={freq}"
        )
    return horizon_seconds // bar_seconds


def get_freq_specific_lookback(cfg: Dict[str, Any]) -> int:
    freq = normalize_freq_name(cfg["freq"])
    return int(cfg["lookback_bars_by_freq"][freq])


def get_freq_specific_relation_windows(cfg: Dict[str, Any]) -> List[int]:
    freq = normalize_freq_name(cfg["freq"])
    return [int(x) for x in cfg["relation_windows_bars_by_freq"][freq]]


FREQ = normalize_freq_name(CFG["freq"])
HORIZON_MINUTES = int(CFG["horizon_minutes"])
HORIZON_BARS = horizon_bars_from_clock_minutes(FREQ, HORIZON_MINUTES)
LOOKBACK_BARS = get_freq_specific_lookback(CFG)
RELATION_WINDOWS = get_freq_specific_relation_windows(CFG)
RELATION_LAGS = [int(x) for x in CFG["relation_lags_bars"]]
PURGE_GAP_BARS = HORIZON_BARS + int(CFG["purge_gap_extra_bars"])
EXPECTED_DELTA = expected_timedelta(FREQ)

print("FREQ:", FREQ)
print("EXPECTED_DELTA:", EXPECTED_DELTA)
print("HORIZON_MINUTES:", HORIZON_MINUTES)
print("HORIZON_BARS:", HORIZON_BARS)
print("LOOKBACK_BARS:", LOOKBACK_BARS)
print("RELATION_WINDOWS:", RELATION_WINDOWS)
print("RELATION_LAGS:", RELATION_LAGS)
print("PURGE_GAP_BARS:", PURGE_GAP_BARS)

# %% [markdown]
# ## Graph topology and relation channels
#
# We use a directed all-pairs graph with self-loops:
#
# - nodes = assets
# - edges = all directed source -> destination pairs, including self loops
#
# The multigraph contains three **separate** relation channels:
#
# 1. `price_dep`
# 2. `order_flow`
# 3. `liquidity`
#
# These channels stay separate until the trainable relation-fusion stage.

# %%
RELATION_NAMES: List[str] = ["price_dep", "order_flow", "liquidity"]


def build_edge_list(assets: List[str], add_self_loops: bool = True) -> List[Tuple[str, str]]:
    edges = [(src, dst) for src in assets for dst in assets if src != dst]
    if add_self_loops:
        edges.extend([(a, a) for a in assets])
    return edges


EDGE_LIST: List[Tuple[str, str]] = build_edge_list(ASSETS, add_self_loops=True)
EDGE_NAMES: List[str] = [f"{src}->{dst}" for src, dst in EDGE_LIST]

EDGE_INDEX = torch.tensor(
    [[ASSET2IDX[src], ASSET2IDX[dst]] for src, dst in EDGE_LIST],
    dtype=torch.long,
)
EDGE_SRC_IDX = EDGE_INDEX[:, 0]
EDGE_DST_IDX = EDGE_INDEX[:, 1]

print("RELATION_NAMES:", RELATION_NAMES)
print("EDGE_NAMES:", EDGE_NAMES)
print("EDGE_INDEX:\n", EDGE_INDEX)

# %% [markdown]
# ## Data loading with frequency-aware timestamp handling
#
# Critical timestamp fix relative to the baseline:
#
# - timestamps are parsed and preserved as-is
# - they are **not rounded to minutes**
# - per-asset and aligned multi-asset time indices are checked against the
#   configured frequency
#
# If the time index is irregular after loading or alignment, this notebook raises
# an error rather than silently accepting a damaged time base.

# %%
EPS = 1e-12


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


def validate_regular_time_index(
    index: pd.DatetimeIndex,
    expected_delta: pd.Timedelta,
    name: str,
) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name}: expected DatetimeIndex, got {type(index)}")

    if len(index) < 2:
        raise ValueError(f"{name}: need at least 2 timestamps for regularity checks")

    if index.has_duplicates:
        dup_count = int(index.duplicated().sum())
        raise ValueError(f"{name}: time index contains {dup_count} duplicate timestamps")

    diffs = index.to_series().diff().dropna()
    bad_mask = diffs != expected_delta
    if bad_mask.any():
        bad_positions = np.where(bad_mask.to_numpy())[0][:5]
        examples = []
        for pos in bad_positions:
            examples.append(
                {
                    "prev": str(index[pos]),
                    "curr": str(index[pos + 1]),
                    "gap": str(diffs.iloc[pos]),
                }
            )
        raise ValueError(
            f"{name}: irregular time index for configured freq={FREQ}. "
            f"Expected every gap to be {expected_delta}, but found {int(bad_mask.sum())} irregular gaps. "
            f"Examples: {examples}"
        )

    inferred = diffs.mode().iloc[0]
    if inferred != expected_delta:
        raise ValueError(
            f"{name}: inferred step {inferred} does not match expected {expected_delta}"
        )


def load_one_asset_raw(asset: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    freq = normalize_freq_name(cfg["freq"])
    path = Path(cfg["data_dir"]) / f"{asset}_{freq}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for asset={asset}: {path}")

    raw = pd.read_csv(path)
    start = int(len(raw) * float(cfg["data_slice_start_frac"]))
    end = int(len(raw) * float(cfg["data_slice_end_frac"]))
    raw = raw.iloc[start:end].copy()

    ts_col = infer_timestamp_column(raw)
    raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Preserve exact timestamps; only exact duplicates are removed.
    raw = raw[~raw["timestamp"].duplicated(keep="last")].copy()
    raw = raw.set_index("timestamp")

    validate_regular_time_index(raw.index, EXPECTED_DELTA, name=f"{asset} raw")

    midpoint_col = choose_existing_column(raw, ["midpoint", "mid", "price"], "midpoint")
    spread_col = choose_existing_column(raw, ["spread"], "spread")
    buys_col = choose_existing_column(raw, ["buys"], "buys")
    sells_col = choose_existing_column(raw, ["sells"], "sells")

    out = pd.DataFrame(index=raw.index)
    out[f"mid_{asset}"] = pd.to_numeric(raw[midpoint_col], errors="coerce")
    out[f"spread_{asset}"] = pd.to_numeric(raw[spread_col], errors="coerce")
    out[f"buys_{asset}"] = pd.to_numeric(raw[buys_col], errors="coerce")
    out[f"sells_{asset}"] = pd.to_numeric(raw[sells_col], errors="coerce")

    book_levels = int(cfg["book_levels"])
    for level in range(book_levels):
        bid_col = infer_book_column(raw, "bids", level)
        ask_col = infer_book_column(raw, "asks", level)
        out[f"bids_notional_{asset}_{level}"] = pd.to_numeric(raw[bid_col], errors="coerce")
        out[f"asks_notional_{asset}_{level}"] = pd.to_numeric(raw[ask_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    missing_counts = out.isna().sum()
    if missing_counts.any():
        bad_cols = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            f"{asset}: required columns contain NaNs after parsing. "
            f"This notebook does not silently drop rows because that would damage time regularity. "
            f"Columns with missing values: {bad_cols}"
        )

    validate_regular_time_index(out.index, EXPECTED_DELTA, name=f"{asset} standardized")
    return out


def load_and_align_assets(cfg: Dict[str, Any]) -> pd.DataFrame:
    aligned: Optional[pd.DataFrame] = None

    for asset in ASSETS:
        one = load_one_asset_raw(asset, cfg)
        aligned = one if aligned is None else aligned.join(one, how="inner")

    if aligned is None or len(aligned) == 0:
        raise RuntimeError("No data available after multi-asset loading and alignment")

    # After inner alignment, re-check regularity. If one asset was missing bars,
    # the alignment can become irregular even if each asset was individually regular.
    aligned = aligned.sort_index()
    aligned = aligned[~aligned.index.duplicated(keep="last")].copy()
    validate_regular_time_index(aligned.index, EXPECTED_DELTA, name="aligned multigraph index")

    for asset in ASSETS:
        log_mid = np.log(aligned[f"mid_{asset}"].astype(float).to_numpy())
        lr = np.zeros(len(aligned), dtype=np.float64)
        lr[1:] = np.diff(log_mid)
        aligned[f"lr_{asset}"] = lr.astype(np.float32)

    aligned = aligned.reset_index().rename(columns={"index": "timestamp"})
    return aligned


df = load_and_align_assets(CFG)
TIMESTAMPS = pd.to_datetime(df["timestamp"], utc=True)

print("Aligned dataframe shape:", df.shape)
print("Aligned time range:", TIMESTAMPS.iloc[0], "->", TIMESTAMPS.iloc[-1])
print(df.head(2))

# %% [markdown]
# ## Node feature engineering
#
# Node features are built per asset and then stacked as:
#
# - `[T, N, F_node]`
#
# Features intentionally remain economically interpretable:
#
# - current 1-bar log return
# - relative spread
# - log buys / log sells
# - flow imbalance
# - total depth imbalance
# - top-book notional imbalances
# - near/far depth ratios
# - near/far depth imbalances
#
# In addition, we construct one relation-state series per asset for each explicit
# graph view:
#
# - `price_dep`: return state
# - `order_flow`: activity-weighted buy/sell imbalance
# - `liquidity`: composite liquidity state based on spread and depth structure

# %%
def safe_log1p_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def bounded_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log((num + eps) / (den + eps))


def build_node_features_and_relation_states(
    df_: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, np.ndarray]]]:
    book_levels = int(cfg["book_levels"])
    top_levels = int(cfg["top_levels"])
    near_levels = int(cfg["near_levels"])

    if top_levels > book_levels:
        raise ValueError("top_levels must be <= book_levels")
    if near_levels >= book_levels:
        raise ValueError("near_levels must be < book_levels")

    node_feature_names = [
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

    per_asset_node_features: List[np.ndarray] = []
    relation_state_map: Dict[str, Dict[str, np.ndarray]] = {
        rel: {} for rel in RELATION_NAMES
    }

    for asset in ASSETS:
        lr = df_[f"lr_{asset}"].to_numpy(dtype=np.float32)
        mid = df_[f"mid_{asset}"].to_numpy(dtype=np.float32)
        spread = df_[f"spread_{asset}"].to_numpy(dtype=np.float32)
        buys = df_[f"buys_{asset}"].to_numpy(dtype=np.float32)
        sells = df_[f"sells_{asset}"].to_numpy(dtype=np.float32)

        rel_spread = spread / (mid + EPS)
        log_buys = safe_log1p_np(buys).astype(np.float32)
        log_sells = safe_log1p_np(sells).astype(np.float32)
        flow_imbalance = ((buys - sells) / (buys + sells + EPS)).astype(np.float32)

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
        depth_imbalance_total = ((bid_total - ask_total) / (bid_total + ask_total + EPS)).astype(np.float32)

        top_imbalances = []
        for i in range(top_levels):
            bi = bids[:, i]
            ai = asks[:, i]
            top_imbalances.append(((bi - ai) / (bi + ai + EPS)).astype(np.float32))
        top_imbalances = np.stack(top_imbalances, axis=1)

        bid_near = bids[:, :near_levels].sum(axis=1)
        ask_near = asks[:, :near_levels].sum(axis=1)
        bid_far = bids[:, near_levels:].sum(axis=1)
        ask_far = asks[:, near_levels:].sum(axis=1)

        bid_near_far_ratio = (bid_near / (bid_far + EPS)).astype(np.float32)
        ask_near_far_ratio = (ask_near / (ask_far + EPS)).astype(np.float32)
        depth_imbalance_near = ((bid_near - ask_near) / (bid_near + ask_near + EPS)).astype(np.float32)
        depth_imbalance_far = ((bid_far - ask_far) / (bid_far + ask_far + EPS)).astype(np.float32)

        asset_node = np.column_stack(
            [
                lr,
                rel_spread,
                log_buys,
                log_sells,
                flow_imbalance,
                depth_imbalance_total,
                top_imbalances[:, 0],
                top_imbalances[:, 1],
                top_imbalances[:, 2],
                top_imbalances[:, 3],
                top_imbalances[:, 4],
                bid_near_far_ratio,
                ask_near_far_ratio,
                depth_imbalance_near,
                depth_imbalance_far,
            ]
        ).astype(np.float32)

        per_asset_node_features.append(asset_node)

        # Relation-state series
        # price_dep: direct return state
        price_state = lr.astype(np.float32)

        # order_flow: activity-weighted imbalance proxy
        turnover_log = safe_log1p_np(buys + sells).astype(np.float32)
        flow_state = (flow_imbalance * turnover_log).astype(np.float32)

        # liquidity: composite state based on spread + depth structure
        depth_shape = np.tanh(
            bounded_log_ratio(bid_near_far_ratio + 1.0, ask_near_far_ratio + 1.0)
        ).astype(np.float32)
        liquidity_state = (
            -np.log1p(np.maximum(rel_spread, 0.0) * 1e4)
            + 0.50 * depth_imbalance_total
            + 0.25 * depth_imbalance_near
            + 0.25 * depth_shape
        ).astype(np.float32)

        relation_state_map["price_dep"][asset] = np.nan_to_num(price_state, nan=0.0, posinf=0.0, neginf=0.0)
        relation_state_map["order_flow"][asset] = np.nan_to_num(flow_state, nan=0.0, posinf=0.0, neginf=0.0)
        relation_state_map["liquidity"][asset] = np.nan_to_num(liquidity_state, nan=0.0, posinf=0.0, neginf=0.0)

    x_node = np.stack(per_asset_node_features, axis=1).astype(np.float32)  # [T, N, F]
    x_node = np.nan_to_num(x_node, nan=0.0, posinf=0.0, neginf=0.0)

    return x_node, node_feature_names, relation_state_map


X_NODE_RAW, NODE_FEATURE_NAMES, RELATION_STATE_MAP = build_node_features_and_relation_states(df, CFG)

print("X_NODE_RAW shape:", X_NODE_RAW.shape)
print("NODE_FEATURE_NAMES:", NODE_FEATURE_NAMES)
for rel in RELATION_NAMES:
    print(rel, "example state shape:", RELATION_STATE_MAP[rel][TARGET_ASSET].shape)

# %% [markdown]
# ## Multigraph relation construction
#
# This is a real multigraph, not a concatenated single graph.
#
# For each relation channel, we construct a **separate** edge-feature tensor:
#
# - `price_dep`
# - `order_flow`
# - `liquidity`
#
# Each edge feature vector is built from trailing rolling dependence summaries
# over source and destination relation-state series.
#
# For each `(window, lag)` pair we compute:
#
# - rolling correlation
# - rolling beta-like slope
# - rolling mean product
#
# This yields a relation-specific edge tensor:
#
# - `[T, R, E, D_edge]`
#
# where relation channels remain separated until the relation-fusion module.

# %%
def fisher_z_transform(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -0.999999, 0.999999)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def rolling_dependence_feature_matrix(
    src_series: pd.Series,
    dst_series: pd.Series,
    windows: List[int],
    lags: List[int],
    use_fisher_z: bool = True,
) -> np.ndarray:
    features: List[np.ndarray] = []

    src_series = src_series.astype(float)
    dst_series = dst_series.astype(float)

    for lag in lags:
        shifted_src = src_series.shift(int(lag)) if int(lag) > 0 else src_series

        for window in windows:
            min_periods = max(3, int(window) // 2)

            corr = shifted_src.rolling(window=int(window), min_periods=min_periods).corr(dst_series)
            cov = shifted_src.rolling(window=int(window), min_periods=min_periods).cov(dst_series)
            var = shifted_src.rolling(window=int(window), min_periods=min_periods).var()
            mean_prod = (shifted_src * dst_series).rolling(window=int(window), min_periods=min_periods).mean()

            corr_arr = corr.to_numpy(dtype=np.float64)
            beta_arr = (cov / (var + EPS)).to_numpy(dtype=np.float64)
            mean_prod_arr = mean_prod.to_numpy(dtype=np.float64)

            corr_arr = np.nan_to_num(corr_arr, nan=0.0, posinf=0.0, neginf=0.0)
            beta_arr = np.nan_to_num(beta_arr, nan=0.0, posinf=0.0, neginf=0.0)
            mean_prod_arr = np.nan_to_num(mean_prod_arr, nan=0.0, posinf=0.0, neginf=0.0)

            if use_fisher_z:
                corr_arr = fisher_z_transform(corr_arr)

            features.extend(
                [
                    corr_arr.astype(np.float32),
                    beta_arr.astype(np.float32),
                    mean_prod_arr.astype(np.float32),
                ]
            )

    return np.stack(features, axis=1).astype(np.float32)


def build_multigraph_relation_tensor(
    relation_state_map: Dict[str, Dict[str, np.ndarray]],
    edge_list: List[Tuple[str, str]],
    windows: List[int],
    lags: List[int],
    use_fisher_z: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    relation_tensors: List[np.ndarray] = []
    edge_feature_names: List[str] = []

    # Build names once because all relations share the same edge-feature structure.
    for lag in lags:
        for window in windows:
            edge_feature_names.extend(
                [
                    f"lag{lag}_win{window}_corr",
                    f"lag{lag}_win{window}_beta",
                    f"lag{lag}_win{window}_meanprod",
                ]
            )

    for rel in RELATION_NAMES:
        per_edge = []
        for src, dst in edge_list:
            src_series = pd.Series(relation_state_map[rel][src])
            dst_series = pd.Series(relation_state_map[rel][dst])
            edge_mat = rolling_dependence_feature_matrix(
                src_series=src_series,
                dst_series=dst_series,
                windows=windows,
                lags=lags,
                use_fisher_z=use_fisher_z,
            )
            per_edge.append(edge_mat)

        rel_tensor = np.stack(per_edge, axis=1).astype(np.float32)  # [T, E, D]
        rel_tensor = np.nan_to_num(rel_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        relation_tensors.append(rel_tensor)

    x_rel_edge = np.stack(relation_tensors, axis=1).astype(np.float32)  # [T, R, E, D]
    return x_rel_edge, edge_feature_names


X_REL_EDGE_RAW, EDGE_FEATURE_NAMES = build_multigraph_relation_tensor(
    relation_state_map=RELATION_STATE_MAP,
    edge_list=EDGE_LIST,
    windows=RELATION_WINDOWS,
    lags=RELATION_LAGS,
    use_fisher_z=bool(CFG["use_fisher_z_for_corr"]),
)

print("X_REL_EDGE_RAW shape:", X_REL_EDGE_RAW.shape)
print("Number of edge features per relation:", len(EDGE_FEATURE_NAMES))
print("First 6 edge feature names:", EDGE_FEATURE_NAMES[:6])

# %% [markdown]
# ## Target construction
#
# Primary supervised target:
#
# - ETH forward log return over exactly 5 minutes of clock time
#
# This is constructed directly from ETH midpoint log prices:
#
# - `log(mid[t + horizon]) - log(mid[t])`
#
# Auxiliary evaluation labels:
#
# - direction label: future return > 0 vs < 0
# - trade label: abs(future return) > round-trip-cost-aware threshold
#
# These are evaluation-only labels. Training remains pure regression.

# %%
def forward_log_return_from_mid(mid: np.ndarray, horizon_bars: int) -> np.ndarray:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    mid = np.asarray(mid, dtype=np.float64)
    out = np.full(len(mid), np.nan, dtype=np.float32)

    log_mid = np.log(mid + EPS)
    if len(mid) <= horizon_bars:
        return out

    out[:-horizon_bars] = (log_mid[horizon_bars:] - log_mid[:-horizon_bars]).astype(np.float32)
    return out


def round_trip_cost_as_log_return(cost_bps_per_side: float) -> float:
    return 2.0 * float(cost_bps_per_side) * 1e-4


ETH_MID = df[f"mid_{TARGET_ASSET}"].to_numpy(dtype=np.float64)
ETH_LR_1BAR = df[f"lr_{TARGET_ASSET}"].to_numpy(dtype=np.float64)

Y_RET = forward_log_return_from_mid(ETH_MID, horizon_bars=HORIZON_BARS)

TRADE_LABEL_ABS_RETURN_THRESHOLD = (
    round_trip_cost_as_log_return(float(CFG["cost_bps_per_side"]))
    + float(CFG["trade_label_buffer_bps"]) * 1e-4
)

Y_DIR = np.full(len(Y_RET), np.nan, dtype=np.float32)
Y_DIR[np.isfinite(Y_RET) & (Y_RET > 0.0)] = 1.0
Y_DIR[np.isfinite(Y_RET) & (Y_RET < 0.0)] = 0.0

Y_TRADE = np.full(len(Y_RET), np.nan, dtype=np.float32)
valid_trade_mask = np.isfinite(Y_RET)
Y_TRADE[valid_trade_mask] = (
    np.abs(Y_RET[valid_trade_mask]) > TRADE_LABEL_ABS_RETURN_THRESHOLD
).astype(np.float32)

print("Finite target count:", int(np.isfinite(Y_RET).sum()))
print("Trade-label absolute-return threshold:", TRADE_LABEL_ABS_RETURN_THRESHOLD)

# %% [markdown]
# ## Valid sample range and raw/sample index mapping
#
# A sample at raw time index `t` is valid only if:
#
# - its full lookback exists
# - its full future target horizon exists
#
# `SAMPLE_T[sample_idx]` maps a sample index into the raw aligned time index.

# %%
T = len(df)
FIRST_VALID_T = LOOKBACK_BARS - 1
LAST_VALID_T = T - HORIZON_BARS - 1

if LAST_VALID_T < FIRST_VALID_T:
    raise RuntimeError(
        f"Not enough rows after slicing. Need lookback={LOOKBACK_BARS} and horizon={HORIZON_BARS}."
    )

SAMPLE_T = np.arange(FIRST_VALID_T, LAST_VALID_T + 1, dtype=np.int64)
N_SAMPLES = len(SAMPLE_T)

print("T:", T)
print("FIRST_VALID_T:", FIRST_VALID_T)
print("LAST_VALID_T:", LAST_VALID_T)
print("N_SAMPLES:", N_SAMPLES)

# %% [markdown]
# ## Split generation with purge/embargo logic
#
# The split structure is preserved from the baseline philosophy, but the horizon
# mapping bug is fixed. Purge gaps are in bars of the configured frequency and
# therefore remain consistent with the true 5-minute clock-time target.
#
# Process:
#
# 1. reserve a final untouched holdout
# 2. leave a purge gap before that holdout
# 3. run walk-forward CV on the pre-holdout region:
#    train -> purge gap -> val -> purge gap -> test
#
# All split arrays below are sample indices into `SAMPLE_T`, not raw time indices.

# %%
def make_preholdout_and_holdout_split(
    n_samples: int,
    holdout_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    holdout_n = max(1, int(round(n_samples * float(holdout_frac))))
    preholdout_n = n_samples - gap_bars - holdout_n

    if preholdout_n <= 0:
        raise RuntimeError("Not enough samples left after reserving gap and holdout.")

    idx_preholdout = np.arange(0, preholdout_n, dtype=np.int64)
    idx_holdout = np.arange(preholdout_n + gap_bars, preholdout_n + gap_bars + holdout_n, dtype=np.int64)

    if idx_holdout[-1] >= n_samples:
        raise RuntimeError("Holdout indices exceed available sample count.")

    return idx_preholdout, idx_holdout


def make_walk_forward_splits(
    idx_preholdout: np.ndarray,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    step_window_frac: float,
    gap_bars: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_pre = len(idx_preholdout)
    train_min = max(1, int(round(n_pre * float(train_min_frac))))
    val_n = max(1, int(round(n_pre * float(val_window_frac))))
    test_n = max(1, int(round(n_pre * float(test_window_frac))))
    step_n = max(1, int(round(n_pre * float(step_window_frac))))

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    train_end = train_min

    while True:
        val_start = train_end + gap_bars
        val_end = val_start + val_n
        test_start = val_end + gap_bars
        test_end = test_start + test_n

        if test_end > n_pre:
            break

        train_idx = idx_preholdout[:train_end].copy()
        val_idx = idx_preholdout[val_start:val_end].copy()
        test_idx = idx_preholdout[test_start:test_end].copy()

        if len(train_idx) and len(val_idx) and len(test_idx):
            splits.append((train_idx, val_idx, test_idx))

        train_end += step_n

    return splits


IDX_PREHOLDOUT, IDX_HOLDOUT = make_preholdout_and_holdout_split(
    n_samples=N_SAMPLES,
    holdout_frac=float(CFG["final_holdout_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

WALK_FORWARD_SPLITS = make_walk_forward_splits(
    idx_preholdout=IDX_PREHOLDOUT,
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    step_window_frac=float(CFG["step_window_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

if len(WALK_FORWARD_SPLITS) == 0:
    raise RuntimeError("No valid walk-forward splits were created. Adjust split fractions or data size.")

print("Pre-holdout samples:", len(IDX_PREHOLDOUT))
print("Holdout samples:", len(IDX_HOLDOUT))
print("Number of CV folds:", len(WALK_FORWARD_SPLITS))
for i, (tr, va, te) in enumerate(WALK_FORWARD_SPLITS, start=1):
    print(f"Fold {i}: train={len(tr)} val={len(va)} test={len(te)}")

# %% [markdown]
# ## Production / full-refit split
#
# Required final structure:
#
# - train_final
# - purge gap
# - val_final
# - purge gap
# - holdout
#
# Because the pre-holdout region was already separated from the final holdout by a
# purge gap, we only need to split the pre-holdout tail into:
#
# - train_final -> gap -> val_final
#
# which yields the full required sequence:
#
# - train_final -> gap -> val_final -> gap -> holdout

# %%
def make_final_production_split(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    val_window_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pre = len(idx_preholdout)
    val_n = max(1, int(round(n_pre * float(val_window_frac))))

    train_end = n_pre - gap_bars - val_n
    if train_end <= 0:
        raise RuntimeError("Not enough pre-holdout samples for final production split.")

    idx_train_final = idx_preholdout[:train_end].copy()
    idx_val_final = idx_preholdout[train_end + gap_bars: train_end + gap_bars + val_n].copy()
    idx_test_final = idx_holdout.copy()

    if len(idx_val_final) != val_n:
        raise RuntimeError("Final validation window was not created correctly.")

    return idx_train_final, idx_val_final, idx_test_final


IDX_TRAIN_FINAL, IDX_VAL_FINAL, IDX_TEST_FINAL = make_final_production_split(
    idx_preholdout=IDX_PREHOLDOUT,
    idx_holdout=IDX_HOLDOUT,
    val_window_frac=float(CFG["val_window_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

print("Production split sizes:")
print("train_final:", len(IDX_TRAIN_FINAL))
print("val_final  :", len(IDX_VAL_FINAL))
print("holdout    :", len(IDX_TEST_FINAL))

# %% [markdown]
# ## Dataset and DataLoaders
#
# Each sample returns:
#
# - node history: `[lookback, N, F_node]`
# - multigraph edge snapshot: `[R, E, D_edge]`
# - regression target
# - sample index and raw time index

# %%
class TemporalMultigraphDataset(Dataset):
    def __init__(
        self,
        x_node: np.ndarray,
        x_rel_edge: np.ndarray,
        y_ret: np.ndarray,
        sample_t: np.ndarray,
        sample_indices: np.ndarray,
        lookback_bars: int,
    ):
        self.x_node = x_node
        self.x_rel_edge = x_rel_edge
        self.y_ret = y_ret
        self.sample_t = sample_t.astype(np.int64)
        self.sample_indices = sample_indices.astype(np.int64)
        self.lookback_bars = int(lookback_bars)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, i: int):
        sample_idx = int(self.sample_indices[i])
        raw_t = int(self.sample_t[sample_idx])
        start = raw_t - self.lookback_bars + 1

        x_seq = self.x_node[start: raw_t + 1]          # [L, N, F]
        edge_last = self.x_rel_edge[raw_t]             # [R, E, D]
        y = float(self.y_ret[raw_t])

        if not np.isfinite(y):
            raise RuntimeError(f"Encountered invalid target at raw_t={raw_t}")

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(edge_last),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(sample_idx, dtype=torch.long),
            torch.tensor(raw_t, dtype=torch.long),
        )


def temporal_multigraph_collate(batch):
    x_seq, edge_last, y, sample_idx, raw_t = zip(*batch)
    return (
        torch.stack(x_seq, dim=0),
        torch.stack(edge_last, dim=0),
        torch.stack(y, dim=0),
        torch.stack(sample_idx, dim=0),
        torch.stack(raw_t, dim=0),
    )

# %% [markdown]
# ## Scaling utilities
#
# Scaling is fit on train only, using all raw timestamps up to the last train
# sample time. This is leakage-safe because every training sample could have
# observed those timestamps at training time.
#
# Node features and relation-channel edge features are scaled separately.
# Relation channels do **not** share a single scaler.

# %%
def fit_robust_scaler_train_only_3d(
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


def apply_robust_scaler_params_3d(raw_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(params["center_"], dtype=np.float32)
    scale = np.asarray(params["scale_"], dtype=np.float32)
    max_abs_value = float(params["max_abs_value"])

    flat = raw_array.reshape(-1, raw_array.shape[-1]).astype(np.float32)
    flat = (flat - center) / (scale + 1e-12)
    flat = np.clip(flat, -max_abs_value, max_abs_value)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.reshape(raw_array.shape).astype(np.float32)


def fit_relation_scalers_train_only(
    raw_rel_array: np.ndarray,
    relation_names: List[str],
    sample_t: np.ndarray,
    train_sample_indices: np.ndarray,
    max_abs_value: float,
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    if raw_rel_array.ndim != 4:
        raise ValueError(f"Expected 4D relation array, got shape={raw_rel_array.shape}")

    scaled = np.zeros_like(raw_rel_array, dtype=np.float32)
    params: Dict[str, Dict[str, Any]] = {}

    for r, rel in enumerate(relation_names):
        rel_scaled, rel_params = fit_robust_scaler_train_only_3d(
            raw_array=raw_rel_array[:, r, :, :],
            sample_t=sample_t,
            train_sample_indices=train_sample_indices,
            max_abs_value=max_abs_value,
            q_low=q_low,
            q_high=q_high,
        )
        scaled[:, r, :, :] = rel_scaled
        params[rel] = rel_params

    return scaled, params


def apply_relation_scalers(
    raw_rel_array: np.ndarray,
    relation_names: List[str],
    relation_scaler_params: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    if raw_rel_array.ndim != 4:
        raise ValueError(f"Expected 4D relation array, got shape={raw_rel_array.shape}")

    scaled = np.zeros_like(raw_rel_array, dtype=np.float32)
    for r, rel in enumerate(relation_names):
        scaled[:, r, :, :] = apply_robust_scaler_params_3d(
            raw_rel_array[:, r, :, :],
            relation_scaler_params[rel],
        )
    return scaled.astype(np.float32)

# %% [markdown]
# ## Graph operator blocks
#
# Honest operator family implementations:
#
# - `edge_mpnn`: edge-conditioned message passing
# - `rel_conv`: relation-aware convolution with scalar edge weights
# - `rel_gatv2`: edge-aware relational attention
#
# Important note:
# The graph is tiny (3 nodes), so we intentionally implement the operators
# directly instead of depending on a larger graph library.

# %%
def aggregate_messages_to_dst(msg: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    # msg: [B, E, H]
    out = msg.new_zeros(msg.size(0), n_nodes, msg.size(-1))
    for e in range(msg.size(1)):
        out[:, int(dst_idx[e].item()), :] += msg[:, e, :]
    return out


def edge_softmax_by_dst(logits: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    # logits: [B, E] or [B, E, Hh]
    out = torch.zeros_like(logits)

    if logits.ndim == 2:
        for node in range(n_nodes):
            mask = (dst_idx == node)
            out[:, mask] = torch.softmax(logits[:, mask], dim=1)
    elif logits.ndim == 3:
        for node in range(n_nodes):
            mask = (dst_idx == node)
            out[:, mask, :] = torch.softmax(logits[:, mask, :], dim=1)
    else:
        raise ValueError(f"Unsupported logits ndim={logits.ndim}")

    return out


class EdgeMPNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.n_nodes = int(n_nodes)

        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        indeg = torch.zeros(n_nodes, dtype=torch.float32)
        for dst in edge_index[:, 1].tolist():
            indeg[int(dst)] += 1.0
        self.register_buffer("indeg", indeg.clamp_min(1.0))

        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H], edge_attr: [B, E, D]
        h_src = h[:, self.src_idx, :]
        h_dst = h[:, self.dst_idx, :]

        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        edge_ctx = self.edge_net(edge_input)
        gate, shift = edge_ctx.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        msg = gate * self.src_proj(h_src) + shift
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)
        agg = agg / self.indeg.view(1, -1, 1)

        update = self.self_proj(h) + self.agg_proj(agg)
        out = self.norm(h + self.dropout(F.gelu(update)))
        return out


class RelConvLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.n_nodes = int(n_nodes)

        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        self.edge_score_net = nn.Sequential(
            nn.Linear(edge_dim, max(16, edge_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, edge_dim), 1),
        )
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h_src = h[:, self.src_idx, :]
        edge_score = self.edge_score_net(edge_attr).squeeze(-1)  # [B, E]
        alpha = edge_softmax_by_dst(edge_score, self.dst_idx, self.n_nodes)  # [B, E]

        msg = alpha.unsqueeze(-1) * self.src_proj(h_src)
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)

        update = self.self_proj(h) + self.agg_proj(agg)
        out = self.norm(h + self.dropout(F.gelu(update)))
        return out


class RelGATv2Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for rel_gatv2")

        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.n_nodes = int(n_nodes)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads

        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.e_proj = nn.Linear(edge_dim, hidden_dim)

        self.attn_vec = nn.Parameter(torch.randn(self.num_heads, self.head_dim) * 0.02)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H], edge_attr: [B, E, D]
        h_src = h[:, self.src_idx, :]
        h_dst = h[:, self.dst_idx, :]

        q = self.q_proj(h_dst).view(h.size(0), h_src.size(1), self.num_heads, self.head_dim)
        k = self.k_proj(h_src).view(h.size(0), h_src.size(1), self.num_heads, self.head_dim)
        v = self.v_proj(h_src).view(h.size(0), h_src.size(1), self.num_heads, self.head_dim)
        e = self.e_proj(edge_attr).view(h.size(0), h_src.size(1), self.num_heads, self.head_dim)

        # GATv2-style edge-aware attention logits
        logits = (torch.tanh(q + k + e) * self.attn_vec.view(1, 1, self.num_heads, self.head_dim)).sum(-1)
        logits = logits / math.sqrt(self.head_dim)

        alpha = edge_softmax_by_dst(logits, self.dst_idx, self.n_nodes)  # [B, E, heads]
        msg = alpha.unsqueeze(-1) * (v + e)
        msg = msg.reshape(h.size(0), h_src.size(1), self.hidden_dim)

        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)

        update = self.self_proj(h) + self.agg_proj(agg)
        out = self.norm(h + self.dropout(F.gelu(update)))
        return out


class RelationGraphBlock(nn.Module):
    def __init__(
        self,
        operator_name: str,
        hidden_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        num_layers: int,
        dropout: float,
        gat_heads: int,
    ):
        super().__init__()
        self.operator_name = str(operator_name)
        layers = []

        for _ in range(int(num_layers)):
            if self.operator_name == "edge_mpnn":
                layers.append(
                    EdgeMPNNLayer(
                        hidden_dim=hidden_dim,
                        edge_dim=edge_dim,
                        n_nodes=n_nodes,
                        edge_index=edge_index,
                        dropout=dropout,
                    )
                )
            elif self.operator_name == "rel_conv":
                layers.append(
                    RelConvLayer(
                        hidden_dim=hidden_dim,
                        edge_dim=edge_dim,
                        n_nodes=n_nodes,
                        edge_index=edge_index,
                        dropout=dropout,
                    )
                )
            elif self.operator_name == "rel_gatv2":
                layers.append(
                    RelGATv2Layer(
                        hidden_dim=hidden_dim,
                        edge_dim=edge_dim,
                        n_nodes=n_nodes,
                        edge_index=edge_index,
                        dropout=dropout,
                        num_heads=gat_heads,
                    )
                )
            else:
                raise ValueError(f"Unsupported graph operator: {self.operator_name}")

        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = h
        for layer in self.layers:
            out = layer(out, edge_attr)
        return out

# %% [markdown]
# ## Relation fusion module
#
# Each relation graph produces relation-specific node embeddings.
#
# This module performs explicit trainable relation fusion:
#
# - input: `[B, R, N, H]`
# - output:
#   - fused node embedding `[B, N, H]`
#   - relation weights `[B, R, N]`
#
# This is a semantic-attention style fusion over relation channels.

# %%
class RelationAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, fusion_hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_relations = int(num_relations)

        self.rel_emb = nn.Parameter(torch.randn(self.num_relations, self.hidden_dim) * 0.02)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, relation_node_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # relation_node_repr: [B, R, N, H]
        rel_bias = self.rel_emb.view(1, self.num_relations, 1, self.hidden_dim)
        score_input = relation_node_repr + rel_bias
        scores = self.score_mlp(score_input).squeeze(-1)  # [B, R, N]
        weights = torch.softmax(scores, dim=1)            # [B, R, N]
        fused = (weights.unsqueeze(-1) * relation_node_repr).sum(dim=1)  # [B, N, H]
        return fused, weights

# %% [markdown]
# ## Multigraph fusion model
#
# Honest architecture:
#
# - temporal encoder first
# - relation-specific graph reasoning second
# - trainable relation fusion third
# - ETH-only regression head last
#
# The model is called `MultigraphFusionModel` rather than anything involving
# "graph attention" unless the operator is actually `rel_gatv2`.

# %%
class MultigraphFusionModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        target_node: int,
        lookback_bars: int,
        relation_names: List[str],
        cfg: Dict[str, Any],
    ):
        super().__init__()

        hidden_dim = int(cfg["hidden_dim"])
        num_heads = int(cfg["transformer_heads"])
        num_layers = int(cfg["transformer_layers"])
        ff_mult = int(cfg["transformer_ff_mult"])
        graph_layers = int(cfg["graph_layers"])
        dropout = float(cfg["dropout"])
        gat_heads = int(cfg["gat_heads"])
        fusion_hidden_dim = int(cfg["fusion_hidden_dim"])

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")

        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)
        self.lookback_bars = int(lookback_bars)
        self.relation_names = list(relation_names)
        self.operator_name = str(cfg["graph_operator"])

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.asset_emb = nn.Parameter(torch.zeros(1, 1, n_nodes, hidden_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, lookback_bars, 1, hidden_dim))

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

        self.relation_blocks = nn.ModuleDict(
            {
                rel: RelationGraphBlock(
                    operator_name=self.operator_name,
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    n_nodes=n_nodes,
                    edge_index=EDGE_INDEX,
                    num_layers=graph_layers,
                    dropout=dropout,
                    gat_heads=gat_heads,
                )
                for rel in self.relation_names
            }
        )

        self.relation_fusion = RelationAttentionFusion(
            hidden_dim=hidden_dim,
            num_relations=len(self.relation_names),
            fusion_hidden_dim=fusion_hidden_dim,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.trunc_normal_(self.asset_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True entries are masked.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode_temporal(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, L, N, F]
        bsz, seq_len, n_nodes, _ = x_seq.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {n_nodes}")

        h = self.node_proj(x_seq)
        h = h + self.asset_emb[:, :, :n_nodes, :] + self.pos_emb[:, :seq_len, :, :]
        h = h.permute(0, 2, 1, 3).contiguous().view(bsz * n_nodes, seq_len, -1)

        causal_mask = self._causal_mask(seq_len=seq_len, device=x_seq.device)
        h = self.temporal_encoder(h, mask=causal_mask)
        h_last = h[:, -1, :].view(bsz, n_nodes, -1)
        return h_last

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_rel_snapshot: torch.Tensor,
        return_aux: bool = False,
    ) -> Any:
        # x_seq: [B, L, N, F]
        # edge_rel_snapshot: [B, R, E, D]
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        edge_rel_snapshot = torch.nan_to_num(edge_rel_snapshot, nan=0.0, posinf=0.0, neginf=0.0)

        h_temporal = self.encode_temporal(x_seq)  # [B, N, H]

        relation_outputs = []
        for r, rel in enumerate(self.relation_names):
            edge_attr_rel = edge_rel_snapshot[:, r, :, :]
            rel_h = self.relation_blocks[rel](h_temporal, edge_attr_rel)
            relation_outputs.append(rel_h)

        relation_stack = torch.stack(relation_outputs, dim=1)  # [B, R, N, H]
        fused_h, relation_weights = self.relation_fusion(relation_stack)

        eth_repr = fused_h[:, self.target_node, :]
        pred = self.head(eth_repr).squeeze(-1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            return {
                "pred": pred,
                "relation_weights": relation_weights,
                "relation_node_repr": relation_stack,
                "fused_node_repr": fused_h,
            }
        return pred

# %% [markdown]
# ## Loss, metrics, benchmarks

# %%
def regression_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    beta = float(cfg["huber_beta"])
    return F.smooth_l1_loss(pred.view(-1), target.view(-1), beta=beta)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


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


def build_threshold_grid(y_pred: np.ndarray, cfg: Dict[str, Any]) -> List[float]:
    abs_pred = np.abs(np.asarray(y_pred, dtype=np.float64))
    abs_pred = abs_pred[np.isfinite(abs_pred)]

    thresholds = set(float(x) for x in cfg["threshold_grid_abs_return"])
    if len(abs_pred):
        for q in cfg["threshold_grid_quantiles"]:
            thresholds.add(float(np.quantile(abs_pred, float(q))))

    thresholds = sorted(x for x in thresholds if x >= 0.0)
    deduped = []
    for x in thresholds:
        if not deduped or abs(x - deduped[-1]) > 1e-12:
            deduped.append(float(x))
    return deduped


def compute_benchmark_predictions(raw_t_indices: np.ndarray) -> Dict[str, np.ndarray]:
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)
    y_true = Y_RET[raw_t_indices].astype(np.float64)

    pred_zero = np.zeros_like(y_true, dtype=np.float64)
    pred_last = ETH_LR_1BAR[raw_t_indices].astype(np.float64) * float(HORIZON_BARS)

    return {
        "y_true": y_true,
        "pred_zero": pred_zero,
        "pred_last": pred_last,
    }


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

    sign_accuracy = float((pred_sign[active_mask] == true_sign[active_mask]).mean()) if active_mask.any() else float("nan")
    long_precision = float((true_sign[long_mask] == 1).mean()) if long_mask.any() else float("nan")
    short_precision = float((true_sign[short_mask] == -1).mean()) if short_mask.any() else float("nan")
    coverage = float(active_mask.mean()) if len(active_mask) else float("nan")

    return {
        "dir_auc": dir_auc,
        "trade_auc": trade_auc,
        "sign_accuracy": sign_accuracy,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "coverage": coverage,
    }

# %% [markdown]
# ## Non-overlapping sequential backtest
#
# This section explicitly fixes the baseline backtest horizon inconsistency.
#
# Procedure:
#
# - step through timestamps sequentially
# - when flat:
#     - open long if pred >= threshold
#     - open short if pred <= -threshold
# - hold for exactly `HORIZON_BARS`
# - close
# - jump directly to the first timestamp after the trade ends
#
# This ensures a one-position-at-a-time, non-overlapping sequential backtest.

# %%
def sequential_fixed_horizon_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_t_indices: np.ndarray,
    timestamps: pd.Series,
    signal_threshold: float,
    horizon_bars: int,
    cost_bps_per_side: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)

    n = len(y_true)
    if n == 0:
        empty = pd.DataFrame(
            columns=[
                "entry_local_idx",
                "entry_raw_t",
                "exit_raw_t",
                "entry_timestamp",
                "exit_timestamp",
                "side",
                "pred",
                "future_return",
                "gross_pnl",
                "net_pnl",
            ]
        )
        metrics = {
            "gross_pnl": float("nan"),
            "net_pnl": float("nan"),
            "pnl_per_trade": float("nan"),
            "n_trades": 0,
            "trade_rate": float("nan"),
            "sharpe_like": float("nan"),
        }
        return metrics, empty

    round_trip_cost = round_trip_cost_as_log_return(cost_bps_per_side)
    i = 0
    rows: List[Dict[str, Any]] = []

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

        entry_raw_t = int(raw_t_indices[i])
        exit_raw_t = int(entry_raw_t + horizon_bars)
        entry_ts = pd.Timestamp(timestamps.iloc[entry_raw_t])
        exit_ts = pd.Timestamp(timestamps.iloc[exit_raw_t]) if exit_raw_t < len(timestamps) else pd.NaT

        rows.append(
            {
                "entry_local_idx": i,
                "entry_raw_t": entry_raw_t,
                "exit_raw_t": exit_raw_t,
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "side": side,
                "pred": score,
                "future_return": realized_return,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
            }
        )

        i += int(horizon_bars)

    trades_df = pd.DataFrame(rows)
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_t_indices: np.ndarray,
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
        raw_t_indices=raw_t_indices,
        timestamps=TIMESTAMPS,
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


def sweep_validation_thresholds(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    raw_t_val: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for thr in build_threshold_grid(y_pred_val, cfg):
        bt_metrics, _ = sequential_fixed_horizon_backtest(
            y_true=y_true_val,
            y_pred=y_pred_val,
            raw_t_indices=raw_t_val,
            timestamps=TIMESTAMPS,
            signal_threshold=float(thr),
            horizon_bars=HORIZON_BARS,
            cost_bps_per_side=float(cfg["cost_bps_per_side"]),
        )
        rows.append({"signal_threshold": float(thr), **bt_metrics})

    sweep_df = pd.DataFrame(rows)
    if len(sweep_df) == 0:
        raise RuntimeError("Threshold sweep produced no candidate rows")

    feasible = sweep_df[sweep_df["n_trades"] >= int(cfg["min_validation_trades"])].copy()
    if len(feasible) == 0:
        feasible = sweep_df.copy()

    feasible = feasible.sort_values(
        by=["net_pnl", "pnl_per_trade", "signal_threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return feasible

# %% [markdown]
# ## Prediction helper

# %%
@torch.no_grad()
def predict_on_indices(
    model: nn.Module,
    x_node_scaled: np.ndarray,
    x_rel_edge_scaled: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> Dict[str, Any]:
    ds = TemporalMultigraphDataset(
        x_node=x_node_scaled,
        x_rel_edge=x_rel_edge_scaled,
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
        collate_fn=temporal_multigraph_collate,
    )

    model.eval()
    preds, targets, sample_idx_all, raw_t_all = [], [], [], []

    for x_seq, edge_last, y, sample_idx, raw_t in loader:
        x_seq = x_seq.to(DEVICE).float()
        edge_last = edge_last.to(DEVICE).float()
        pred = model(x_seq, edge_last)

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        sample_idx_all.append(sample_idx.detach().cpu().numpy())
        raw_t_all.append(raw_t.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0).astype(np.float64)
    target_arr = np.concatenate(targets, axis=0).astype(np.float64)
    sample_idx_arr = np.concatenate(sample_idx_all, axis=0).astype(np.int64)
    raw_t_arr = np.concatenate(raw_t_all, axis=0).astype(np.int64)

    return {
        "pred": pred_arr,
        "target": target_arr,
        "sample_idx": sample_idx_arr,
        "raw_t": raw_t_arr,
        "timestamp": TIMESTAMPS.iloc[raw_t_arr].reset_index(drop=True),
    }

# %% [markdown]
# ## Training loop and fold helper

# %%
@dataclass
class SplitArtifacts:
    model_state: Dict[str, torch.Tensor]
    node_scaler_params: Dict[str, Any]
    relation_scaler_params: Dict[str, Dict[str, Any]]
    best_epoch: int
    best_val_rmse: float
    signal_threshold: float
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    val_predictions: Dict[str, Any]
    test_predictions: Dict[str, Any]


def build_run_cfg(base_cfg: Dict[str, Any], operator_name: str, is_ablation_context: bool) -> Dict[str, Any]:
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["graph_operator"] = str(operator_name)

    if is_ablation_context and bool(base_cfg["ablation_fast_mode"]):
        run_cfg["epochs"] = int(base_cfg["ablation_epochs"])
        run_cfg["patience"] = int(base_cfg["ablation_patience"])

    return run_cfg


def build_model_for_cfg(cfg: Dict[str, Any]) -> MultigraphFusionModel:
    model = MultigraphFusionModel(
        node_dim=X_NODE_RAW.shape[-1],
        edge_dim=X_REL_EDGE_RAW.shape[-1],
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        lookback_bars=LOOKBACK_BARS,
        relation_names=RELATION_NAMES,
        cfg=cfg,
    ).to(DEVICE)
    return model


def train_one_split(
    split_name: str,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict[str, Any],
) -> SplitArtifacts:
    x_node_scaled, node_scaler_params = fit_robust_scaler_train_only_3d(
        raw_array=X_NODE_RAW,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_node_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )

    x_rel_edge_scaled, relation_scaler_params = fit_relation_scalers_train_only(
        raw_rel_array=X_REL_EDGE_RAW,
        relation_names=RELATION_NAMES,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_edge_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )

    train_ds = TemporalMultigraphDataset(
        x_node=x_node_scaled,
        x_rel_edge=x_rel_edge_scaled,
        y_ret=Y_RET,
        sample_t=SAMPLE_T,
        sample_indices=idx_train,
        lookback_bars=LOOKBACK_BARS,
    )
    val_ds = TemporalMultigraphDataset(
        x_node=x_node_scaled,
        x_rel_edge=x_rel_edge_scaled,
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
        collate_fn=temporal_multigraph_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_multigraph_collate,
    )

    model = build_model_for_cfg(cfg)
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
        train_losses = []

        for x_seq, edge_last, y, _sample_idx, _raw_t in train_loader:
            x_seq = x_seq.to(DEVICE).float()
            edge_last = edge_last.to(DEVICE).float()
            y = y.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, edge_last)
            loss = regression_loss(pred, y, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_seq, edge_last, y, _sample_idx, _raw_t in val_loader:
                x_seq = x_seq.to(DEVICE).float()
                edge_last = edge_last.to(DEVICE).float()
                y = y.to(DEVICE).float()

                pred = model(x_seq, edge_last)
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(y.detach().cpu().numpy())

        val_pred_arr = np.concatenate(val_preds, axis=0).astype(np.float64)
        val_target_arr = np.concatenate(val_targets, axis=0).astype(np.float64)

        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_rmse = rmse_np(val_target_arr, val_pred_arr)
        val_mae = mae_np(val_target_arr, val_pred_arr)
        val_ic = ic_np(val_target_arr, val_pred_arr)

        scheduler.step(val_rmse)

        print(
            f"[{split_name}][{cfg['graph_operator']}] "
            f"epoch={epoch:02d} "
            f"train_loss={train_loss_mean:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"val_mae={val_mae:.6f} "
            f"val_ic={val_ic:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = float(val_rmse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            print(f"[{split_name}][{cfg['graph_operator']}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"[{split_name}] no best_state captured during training")

    model.load_state_dict(best_state)

    val_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_val,
        batch_size=int(cfg["batch_size"]),
    )
    test_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_test,
        batch_size=int(cfg["batch_size"]),
    )

    threshold_sweep_df = sweep_validation_thresholds(
        y_true_val=val_pred_pack["target"],
        y_pred_val=val_pred_pack["pred"],
        raw_t_val=val_pred_pack["raw_t"],
        cfg=cfg,
    )
    selected_threshold = float(threshold_sweep_df.iloc[0]["signal_threshold"])

    val_metrics = evaluate_predictions(
        y_true=val_pred_pack["target"],
        y_pred=val_pred_pack["pred"],
        raw_t_indices=val_pred_pack["raw_t"],
        signal_threshold=selected_threshold,
        cfg=cfg,
    )
    test_metrics = evaluate_predictions(
        y_true=test_pred_pack["target"],
        y_pred=test_pred_pack["pred"],
        raw_t_indices=test_pred_pack["raw_t"],
        signal_threshold=selected_threshold,
        cfg=cfg,
    )

    print(
        f"[{split_name}][{cfg['graph_operator']}] "
        f"best_epoch={best_epoch} "
        f"best_val_rmse={best_val_rmse:.6f} "
        f"selected_threshold={selected_threshold:.6f}"
    )
    print(
        f"[{split_name}][{cfg['graph_operator']}] TEST "
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
        relation_scaler_params=relation_scaler_params,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        signal_threshold=selected_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_predictions=val_pred_pack,
        test_predictions=test_pred_pack,
    )

# %% [markdown]
# ## Artifact saving/loading
#
# For clarity and portability, each saved bundle contains:
#
# - model weights
# - node scaler params
# - per-relation scaler params
# - config
# - selected threshold
# - metadata on the split and operator

# %%
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
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


def save_bundle(
    bundle_dir: Path,
    bundle_name: str,
    model_state: Dict[str, torch.Tensor],
    node_scaler_params: Dict[str, Any],
    relation_scaler_params: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = bundle_dir / f"{bundle_name}.pt"
    meta_path = bundle_dir / f"{bundle_name}_meta.json"

    payload = {
        "bundle_name": bundle_name,
        "cfg": copy.deepcopy(cfg),
        "model_state": model_state,
        "node_scaler_params": node_scaler_params,
        "relation_scaler_params": relation_scaler_params,
        "relation_names": RELATION_NAMES,
        "assets": ASSETS,
        "target_asset": TARGET_ASSET,
        "freq": FREQ,
        "expected_delta_seconds": freq_to_seconds(FREQ),
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_bars": HORIZON_BARS,
        "lookback_bars": LOOKBACK_BARS,
        "meta": meta,
    }

    torch.save(payload, str(bundle_path))

    meta_json = {
        "bundle_name": bundle_name,
        "bundle_file": bundle_path.name,
        "cfg": _jsonable(cfg),
        "relation_names": RELATION_NAMES,
        "assets": ASSETS,
        "target_asset": TARGET_ASSET,
        "freq": FREQ,
        "expected_delta_seconds": freq_to_seconds(FREQ),
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_bars": HORIZON_BARS,
        "lookback_bars": LOOKBACK_BARS,
        **_jsonable(meta),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    return {"bundle": bundle_path, "meta": meta_path}


def load_bundle(bundle_dir: Path, bundle_name: str) -> Dict[str, Any]:
    bundle_path = bundle_dir / f"{bundle_name}.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    loaded = torch.load(str(bundle_path), map_location="cpu")
    return loaded

# %% [markdown]
# ## Saved-bundle evaluator
#
# This reuses saved artifacts exactly as persisted.
#
# No threshold retuning or rescaling is performed on the evaluation split.

# %%
@torch.no_grad()
def evaluate_saved_bundle_on_indices(
    bundle_dir: Path,
    bundle_name: str,
    indices: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    loaded = load_bundle(bundle_dir, bundle_name)
    cfg = loaded["cfg"]

    model = build_model_for_cfg(cfg)
    model.load_state_dict(loaded["model_state"])
    model.eval()

    x_node_scaled = apply_robust_scaler_params_3d(X_NODE_RAW, loaded["node_scaler_params"])
    x_rel_edge_scaled = apply_relation_scalers(
        raw_rel_array=X_REL_EDGE_RAW,
        relation_names=loaded["relation_names"],
        relation_scaler_params=loaded["relation_scaler_params"],
    )

    pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=indices.astype(np.int64),
        batch_size=int(cfg["batch_size"]),
    )

    threshold = float(loaded["meta"]["selected_signal_threshold"])
    metrics = evaluate_predictions(
        y_true=pred_pack["target"],
        y_pred=pred_pack["pred"],
        raw_t_indices=pred_pack["raw_t"],
        signal_threshold=threshold,
        cfg=cfg,
    )

    bm = compute_benchmark_predictions(pred_pack["raw_t"])
    benchmarks = {
        "zero_rmse": rmse_np(bm["y_true"], bm["pred_zero"]),
        "zero_mae": mae_np(bm["y_true"], bm["pred_zero"]),
        "zero_ic": ic_np(bm["y_true"], bm["pred_zero"]),
        "last_rmse": rmse_np(bm["y_true"], bm["pred_last"]),
        "last_mae": mae_np(bm["y_true"], bm["pred_last"]),
        "last_ic": ic_np(bm["y_true"], bm["pred_last"]),
    }

    print("\n" + "=" * 110)
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
        f"bench_zero_rmse={benchmarks['zero_rmse']:.6f} "
        f"bench_last_rmse={benchmarks['last_rmse']:.6f}"
    )

    return {
        "pred_pack": pred_pack,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "threshold": threshold,
    }

# %% [markdown]
# ## Single-operator experiment runner
#
# This runs the full required scaffold for one graph operator:
#
# 1. walk-forward CV
# 2. per-fold validation and test evaluation
# 3. saved-bundle post-CV holdout evaluation using best CV-selected model
# 4. production / full-refit evaluation
# 5. artifact saving/loading

# %%
def flatten_metrics_row(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in metrics.items():
        if k == "trades_df":
            continue
        row[f"{prefix}_{k}"] = v
    return row


def summarize_predictions_against_benchmarks(raw_t_idx: np.ndarray) -> Dict[str, float]:
    bm = compute_benchmark_predictions(raw_t_idx)
    return {
        "bench_zero_rmse": rmse_np(bm["y_true"], bm["pred_zero"]),
        "bench_zero_mae": mae_np(bm["y_true"], bm["pred_zero"]),
        "bench_zero_ic": ic_np(bm["y_true"], bm["pred_zero"]),
        "bench_last_rmse": rmse_np(bm["y_true"], bm["pred_last"]),
        "bench_last_mae": mae_np(bm["y_true"], bm["pred_last"]),
        "bench_last_ic": ic_np(bm["y_true"], bm["pred_last"]),
    }


def run_experiment_for_operator(
    operator_name: str,
    base_cfg: Dict[str, Any],
    is_ablation_context: bool = True,
) -> Dict[str, Any]:
    run_cfg = build_run_cfg(base_cfg=base_cfg, operator_name=operator_name, is_ablation_context=is_ablation_context)
    operator_dir = ARTIFACT_ROOT / operator_name
    operator_dir.mkdir(parents=True, exist_ok=True)

    cv_rows: List[Dict[str, Any]] = []
    best_cv_bundle_name: Optional[str] = None
    best_cv_val_rmse = float("inf")
    fold_bundle_names: List[str] = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(WALK_FORWARD_SPLITS, start=1):
        print("\n" + "=" * 110)
        print(
            f"OPERATOR={operator_name} | FOLD {fold_idx}/{len(WALK_FORWARD_SPLITS)} "
            f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
        )

        artifacts = train_one_split(
            split_name=f"{operator_name}_fold_{fold_idx:02d}",
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            cfg=run_cfg,
        )

        bundle_name = f"{operator_name}_fold_{fold_idx:02d}_best"
        fold_bundle_names.append(bundle_name)

        save_bundle(
            bundle_dir=operator_dir,
            bundle_name=bundle_name,
            model_state=artifacts.model_state,
            node_scaler_params=artifacts.node_scaler_params,
            relation_scaler_params=artifacts.relation_scaler_params,
            cfg=run_cfg,
            meta={
                "kind": "cv_fold_best",
                "operator_name": operator_name,
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
            operator_dir / f"{bundle_name}_val_trades.csv",
            index=False,
        )
        artifacts.test_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_test_trades.csv",
            index=False,
        )

        val_bench = summarize_predictions_against_benchmarks(artifacts.val_predictions["raw_t"])
        test_bench = summarize_predictions_against_benchmarks(artifacts.test_predictions["raw_t"])

        row = {
            "operator": operator_name,
            "fold": fold_idx,
            "best_epoch": artifacts.best_epoch,
            "best_val_rmse_checkpoint": artifacts.best_val_rmse,
            "selected_signal_threshold": artifacts.signal_threshold,
            **flatten_metrics_row("val", artifacts.val_metrics),
            **{f"val_{k}": v for k, v in val_bench.items()},
            **flatten_metrics_row("test", artifacts.test_metrics),
            **{f"test_{k}": v for k, v in test_bench.items()},
        }
        cv_rows.append(row)

        if artifacts.best_val_rmse < best_cv_val_rmse:
            best_cv_val_rmse = float(artifacts.best_val_rmse)
            best_cv_bundle_name = bundle_name

    if best_cv_bundle_name is None:
        raise RuntimeError(f"{operator_name}: no best CV bundle selected")

    cv_results_df = pd.DataFrame(cv_rows)
    cv_results_df.to_csv(operator_dir / f"{operator_name}_cv_results_summary.csv", index=False)

    cv_mean_df = cv_results_df.mean(numeric_only=True).to_frame(name="mean").T
    cv_mean_df.insert(0, "operator", operator_name)
    cv_mean_df.to_csv(operator_dir / f"{operator_name}_cv_mean_summary.csv", index=False)

    print("\n" + "=" * 110)
    print(f"CV_RESULTS_DF [{operator_name}]")
    print(cv_results_df)
    print("\nCV mean metrics:")
    print(cv_mean_df)

    post_cv_holdout = evaluate_saved_bundle_on_indices(
        bundle_dir=operator_dir,
        bundle_name=best_cv_bundle_name,
        indices=IDX_HOLDOUT,
        label=f"POST-CV HOLDOUT EVALUATION [{operator_name}] USING BEST CV-SELECTED MODEL",
    )
    post_cv_holdout_df = pd.DataFrame(
        [
            {
                "operator": operator_name,
                "model_name": best_cv_bundle_name,
                **flatten_metrics_row("", post_cv_holdout["metrics"]),
                "bench_zero_rmse": post_cv_holdout["benchmarks"]["zero_rmse"],
                "bench_last_rmse": post_cv_holdout["benchmarks"]["last_rmse"],
                "selected_signal_threshold": post_cv_holdout["threshold"],
            }
        ]
    )
    post_cv_holdout_df.to_csv(operator_dir / f"{operator_name}_post_cv_holdout_summary.csv", index=False)

    production_artifacts = train_one_split(
        split_name=f"{operator_name}_production_refit",
        idx_train=IDX_TRAIN_FINAL,
        idx_val=IDX_VAL_FINAL,
        idx_test=IDX_TEST_FINAL,
        cfg=run_cfg,
    )

    production_bundle_name = f"{operator_name}_production_best"
    save_bundle(
        bundle_dir=operator_dir,
        bundle_name=production_bundle_name,
        model_state=production_artifacts.model_state,
        node_scaler_params=production_artifacts.node_scaler_params,
        relation_scaler_params=production_artifacts.relation_scaler_params,
        cfg=run_cfg,
        meta={
            "kind": "production_best",
            "operator_name": operator_name,
            "best_epoch": production_artifacts.best_epoch,
            "best_val_rmse": production_artifacts.best_val_rmse,
            "selected_signal_threshold": production_artifacts.signal_threshold,
            "idx_train": IDX_TRAIN_FINAL.tolist(),
            "idx_val": IDX_VAL_FINAL.tolist(),
            "idx_test": IDX_TEST_FINAL.tolist(),
        },
    )

    production_artifacts.val_metrics["trades_df"].to_csv(
        operator_dir / f"{production_bundle_name}_val_trades.csv",
        index=False,
    )
    production_artifacts.test_metrics["trades_df"].to_csv(
        operator_dir / f"{production_bundle_name}_holdout_trades.csv",
        index=False,
    )

    production_holdout = evaluate_saved_bundle_on_indices(
        bundle_dir=operator_dir,
        bundle_name=production_bundle_name,
        indices=IDX_HOLDOUT,
        label=f"FULL-REFIT / PRODUCTION HOLDOUT EVALUATION [{operator_name}]",
    )
    production_holdout_df = pd.DataFrame(
        [
            {
                "operator": operator_name,
                "model_name": production_bundle_name,
                **flatten_metrics_row("", production_holdout["metrics"]),
                "bench_zero_rmse": production_holdout["benchmarks"]["zero_rmse"],
                "bench_last_rmse": production_holdout["benchmarks"]["last_rmse"],
                "selected_signal_threshold": production_holdout["threshold"],
            }
        ]
    )
    production_holdout_df.to_csv(operator_dir / f"{operator_name}_production_holdout_summary.csv", index=False)

    comparison_row = {
        "operator": operator_name,
        "graph_operator": operator_name,
        "cv_mean_test_rmse": float(cv_results_df["test_rmse"].mean()),
        "cv_mean_test_mae": float(cv_results_df["test_mae"].mean()),
        "cv_mean_test_ic": float(cv_results_df["test_ic"].mean()),
        "cv_mean_test_dir_auc": float(cv_results_df["test_dir_auc"].mean()),
        "cv_mean_test_trade_auc": float(cv_results_df["test_trade_auc"].mean()),
        "cv_mean_test_net_pnl": float(cv_results_df["test_net_pnl"].mean()),
        "cv_mean_test_sharpe_like": float(cv_results_df["test_sharpe_like"].mean()),
        "post_cv_holdout_rmse": float(post_cv_holdout_df.iloc[0]["_rmse"]) if "_rmse" in post_cv_holdout_df.columns else float(post_cv_holdout_df.iloc[0]["rmse"]),
        "post_cv_holdout_ic": float(post_cv_holdout_df.iloc[0]["_ic"]) if "_ic" in post_cv_holdout_df.columns else float(post_cv_holdout_df.iloc[0]["ic"]),
        "post_cv_holdout_net_pnl": float(post_cv_holdout_df.iloc[0]["_net_pnl"]) if "_net_pnl" in post_cv_holdout_df.columns else float(post_cv_holdout_df.iloc[0]["net_pnl"]),
        "production_holdout_rmse": float(production_holdout_df.iloc[0]["_rmse"]) if "_rmse" in production_holdout_df.columns else float(production_holdout_df.iloc[0]["rmse"]),
        "production_holdout_ic": float(production_holdout_df.iloc[0]["_ic"]) if "_ic" in production_holdout_df.columns else float(production_holdout_df.iloc[0]["ic"]),
        "production_holdout_net_pnl": float(production_holdout_df.iloc[0]["_net_pnl"]) if "_net_pnl" in production_holdout_df.columns else float(production_holdout_df.iloc[0]["net_pnl"]),
    }

    return {
        "operator_name": operator_name,
        "cfg": run_cfg,
        "artifact_dir": operator_dir,
        "fold_bundle_names": fold_bundle_names,
        "best_cv_bundle_name": best_cv_bundle_name,
        "production_bundle_name": production_bundle_name,
        "cv_results_df": cv_results_df,
        "cv_mean_df": cv_mean_df,
        "post_cv_holdout_df": post_cv_holdout_df,
        "production_holdout_df": production_holdout_df,
        "comparison_row": comparison_row,
    }

# %% [markdown]
# ## Operator ablation runner
#
# This runs the same experimental protocol for:
#
# - edge_mpnn
# - rel_conv
# - rel_gatv2
#
# The code is complete for all three operators. If runtime needs to be reduced
# during experimentation, set:
#
# - `CFG["ablation_fast_mode"] = True`
#
# which reduces epochs and patience for all ablation runs.

# %%
def normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        if col.startswith("_"):
            rename_map[col] = col[1:]
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def run_operator_ablation_suite(cfg: Dict[str, Any]) -> Dict[str, Any]:
    operator_runs: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []

    for operator_name in cfg["operator_candidates"]:
        print("\n" + "#" * 120)
        print(f"RUNNING OPERATOR ABLATION FOR: {operator_name}")
        print("#" * 120)

        run_out = run_experiment_for_operator(
            operator_name=operator_name,
            base_cfg=cfg,
            is_ablation_context=True,
        )

        operator_runs[operator_name] = run_out
        comparison_rows.append(run_out["comparison_row"])

    operator_comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["production_holdout_rmse", "production_holdout_net_pnl"],
        ascending=[True, False],
    ).reset_index(drop=True)

    operator_comparison_df.to_csv(ARTIFACT_ROOT / "operator_comparison_summary.csv", index=False)

    cv_summary_frames = []
    post_cv_frames = []
    production_frames = []

    for operator_name, run_out in operator_runs.items():
        cv_mean = run_out["cv_mean_df"].copy()
        cv_summary_frames.append(cv_mean)

        post_cv = normalize_summary_columns(run_out["post_cv_holdout_df"])
        post_cv_frames.append(post_cv)

        prod = normalize_summary_columns(run_out["production_holdout_df"])
        production_frames.append(prod)

    all_cv_summary_df = pd.concat(cv_summary_frames, axis=0, ignore_index=True)
    all_post_cv_holdout_df = pd.concat(post_cv_frames, axis=0, ignore_index=True)
    all_production_holdout_df = pd.concat(production_frames, axis=0, ignore_index=True)

    all_cv_summary_df.to_csv(ARTIFACT_ROOT / "all_operator_cv_mean_summary.csv", index=False)
    all_post_cv_holdout_df.to_csv(ARTIFACT_ROOT / "all_operator_post_cv_holdout_summary.csv", index=False)
    all_production_holdout_df.to_csv(ARTIFACT_ROOT / "all_operator_production_holdout_summary.csv", index=False)

    return {
        "operator_runs": operator_runs,
        "operator_comparison_df": operator_comparison_df,
        "all_cv_summary_df": all_cv_summary_df,
        "all_post_cv_holdout_df": all_post_cv_holdout_df,
        "all_production_holdout_df": all_production_holdout_df,
    }

# %% [markdown]
# ## Main execution
#
# By default, the notebook runs the full operator ablation suite.
#
# If you only want the default operator:
#
# - set `CFG["run_full_operator_ablation"] = False`
#
# In that case the notebook still runs the full required pipeline, but only for
# `CFG["graph_operator"]`.

# %%
if bool(CFG["run_full_operator_ablation"]):
    ABLATION_RESULTS = run_operator_ablation_suite(CFG)
    OPERATOR_RUNS = ABLATION_RESULTS["operator_runs"]
    OPERATOR_COMPARISON_DF = ABLATION_RESULTS["operator_comparison_df"]
    ALL_OPERATOR_CV_SUMMARY_DF = ABLATION_RESULTS["all_cv_summary_df"]
    ALL_OPERATOR_POST_CV_HOLDOUT_DF = ABLATION_RESULTS["all_post_cv_holdout_df"]
    ALL_OPERATOR_PRODUCTION_HOLDOUT_DF = ABLATION_RESULTS["all_production_holdout_df"]

    print("\n" + "=" * 110)
    print("ALL_OPERATOR_CV_SUMMARY_DF")
    print(ALL_OPERATOR_CV_SUMMARY_DF)

    print("\n" + "=" * 110)
    print("ALL_OPERATOR_POST_CV_HOLDOUT_DF")
    print(ALL_OPERATOR_POST_CV_HOLDOUT_DF)

    print("\n" + "=" * 110)
    print("ALL_OPERATOR_PRODUCTION_HOLDOUT_DF")
    print(ALL_OPERATOR_PRODUCTION_HOLDOUT_DF)

    print("\n" + "=" * 110)
    print("OPERATOR_COMPARISON_DF")
    print(OPERATOR_COMPARISON_DF)

else:
    DEFAULT_OPERATOR_RESULTS = run_experiment_for_operator(
        operator_name=str(CFG["graph_operator"]),
        base_cfg=CFG,
        is_ablation_context=False,
    )

    DEFAULT_CV_SUMMARY_DF = DEFAULT_OPERATOR_RESULTS["cv_mean_df"]
    DEFAULT_POST_CV_HOLDOUT_DF = normalize_summary_columns(DEFAULT_OPERATOR_RESULTS["post_cv_holdout_df"])
    DEFAULT_PRODUCTION_HOLDOUT_DF = normalize_summary_columns(DEFAULT_OPERATOR_RESULTS["production_holdout_df"])

    print("\n" + "=" * 110)
    print("DEFAULT_CV_SUMMARY_DF")
    print(DEFAULT_CV_SUMMARY_DF)

    print("\n" + "=" * 110)
    print("DEFAULT_POST_CV_HOLDOUT_DF")
    print(DEFAULT_POST_CV_HOLDOUT_DF)

    print("\n" + "=" * 110)
    print("DEFAULT_PRODUCTION_HOLDOUT_DF")
    print(DEFAULT_PRODUCTION_HOLDOUT_DF)

# %% [markdown]
# ## Final operator comparison tables
#
# The key output tables are:
#
# - per-operator CV summary
# - per-operator post-CV holdout summary
# - per-operator production holdout summary
# - operator comparison summary
#
# These are saved to `artifact_root` and each operator subdirectory.

# %%
if bool(CFG["run_full_operator_ablation"]):
    FINAL_OPERATOR_COMPARISON_DF = OPERATOR_COMPARISON_DF.copy()

    # A compact final table focused on the most decision-relevant metrics
    FINAL_OPERATOR_COMPARISON_COMPACT_DF = FINAL_OPERATOR_COMPARISON_DF[
        [
            "operator",
            "cv_mean_test_rmse",
            "cv_mean_test_ic",
            "cv_mean_test_net_pnl",
            "post_cv_holdout_rmse",
            "post_cv_holdout_ic",
            "post_cv_holdout_net_pnl",
            "production_holdout_rmse",
            "production_holdout_ic",
            "production_holdout_net_pnl",
        ]
    ].copy()

    FINAL_OPERATOR_COMPARISON_DF.to_csv(ARTIFACT_ROOT / "final_operator_comparison_full.csv", index=False)
    FINAL_OPERATOR_COMPARISON_COMPACT_DF.to_csv(ARTIFACT_ROOT / "final_operator_comparison_compact.csv", index=False)

    print("\n" + "=" * 110)
    print("FINAL_OPERATOR_COMPARISON_DF")
    print(FINAL_OPERATOR_COMPARISON_DF)

    print("\n" + "=" * 110)
    print("FINAL_OPERATOR_COMPARISON_COMPACT_DF")
    print(FINAL_OPERATOR_COMPARISON_COMPACT_DF)

# %% [markdown]
# ## Notes on interpretation
#
# - `rmse`, `mae`, `ic` evaluate the regression forecast of fixed-horizon ETH return
# - `dir_auc` evaluates ranking of positive vs negative future ETH returns
# - `trade_auc` evaluates ranking of tradeable vs non-tradeable future moves,
#   where tradeable means `abs(future return)` exceeds the round-trip-cost-aware threshold
# - `coverage` is the fraction of timestamps with `abs(prediction) >= selected_threshold`
# - `trade_rate` is executed non-overlapping trades divided by evaluated timestamps
# - `gross_pnl` and `net_pnl` are cumulative per-trade log-return PnL across the split
#
# Benchmarks:
#
# - `zero-return`: predict zero future return
# - `last-return extrapolation`: current ETH 1-bar return times `HORIZON_BARS`

# %%
print("Notebook build complete.")