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


# %%
def seed_everything(seed: int = 1001) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1001)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

print("DEVICE:", DEVICE)


# %%
CFG: Dict[str, Any] = {
    # Data settings
    "freq": "1min",
    "data_dir": "../dataset",
    "artifact_root": "./artifact_root_multigraph_temporal_multitask_pnl",
    "assets": ["ADA", "BTC", "ETH"],
    "target_asset": "ETH",
    "data_slice_start_frac": 0.00,
    "data_slice_end_frac": 0.75,
    "final_holdout_frac": 0.10,

    # Horizon settings
    "horizon_minutes": 5,

    # Lookback settings
    "lookback_bars_by_freq": {
        "1sec": 300,
        "1min": 240,
        "5min": 120,
    },

    # Raw book settings
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # Relation feature settings
    "relation_windows_bars_by_freq": {
        "1sec": [60, 300, 900],
        "1min": [10, 30, 60],
        "5min": [6, 12, 24],
    },
    "relation_lags_bars": [0, 1, 2, 5],
    "use_fisher_z_for_corr": True,

    # Split settings
    "preholdout_n_folds": 4,
    "train_min_frac": 0.50,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "purge_gap_extra_bars": 0,

    # Scaling settings
    "max_abs_node_feature": 8.0,
    "max_abs_edge_feature": 6.0,
    "scaler_quantile_low": 5.0,
    "scaler_quantile_high": 95.0,

    # Model settings
    "graph_operator": "dynamic_edge_mpnn",
    "node_hidden_dim": 48,
    "edge_hidden_dim": 32,
    "target_hidden_dim": 48,
    "node_temporal_layers": 3,
    "edge_temporal_layers": 3,
    "target_temporal_layers": 2,
    "graph_layers": 2,
    "temporal_kernel_size": 3,
    "dropout": 0.15,
    "fusion_hidden_dim": 32,

    # Training settings
    "batch_size": 64,
    "epochs": 35,
    "patience": 6,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "huber_beta": 5e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,

    # Multitask loss settings
    "loss_w_trade": 0.65,
    "loss_w_dir": 0.80,
    "loss_w_ret": 0.25,
    "loss_w_utility": 0.55,
    "utility_tanh_k": 1.50,
    "utility_bps_scale": 1e4,

    # Dynamic-operator regularization
    "adj_l1_lambda": 5e-5,
    "adj_prior_lambda": 1e-4,

    # Trade evaluation settings
    "cost_bps_per_side": 1.0,
    "trade_label_buffer_bps": 0.0,
    "threshold_trade_grid": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "threshold_dir_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "min_threshold_search_trades": 5,
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

print("RELATION_NAMES:", RELATION_NAMES)
print("EDGE_NAMES:", EDGE_NAMES)
print("EDGE_INDEX:\n", EDGE_INDEX)


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
    frame: pd.DataFrame,
    expected_delta: pd.Timedelta,
    name: str,
    *,
    fill_missing: bool = True,
    log_limit: int = 10,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: expected DatetimeIndex, got {type(frame.index)}")
    if len(frame.index) < 2:
        raise ValueError(f"{name}: need at least 2 timestamps for regularity checks")

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        dup_count = int(frame.index.duplicated().sum())
        raise ValueError(f"{name}: time index contains {dup_count} duplicate timestamps")

    diffs = frame.index.to_series().diff().dropna()
    bad_mask = diffs != expected_delta
    if bad_mask.any():
        bad_positions = np.where(bad_mask.to_numpy())[0][:5]
        irregular_examples = []
        for pos in bad_positions:
            irregular_examples.append(
                {
                    "prev": str(frame.index[pos]),
                    "curr": str(frame.index[pos + 1]),
                    "gap": str(diffs.iloc[pos]),
                }
            )

        if not fill_missing:
            raise ValueError(
                f"{name}: irregular time index for configured freq={FREQ}. "
                f"Expected every gap to be {expected_delta}, but found {int(bad_mask.sum())} irregular gaps. "
                f"Examples: {irregular_examples}"
            )

        full_index = pd.date_range(
            start=frame.index.min(),
            end=frame.index.max(),
            freq=expected_delta,
            tz=frame.index.tz,
        )
        missing_index = full_index.difference(frame.index)
        if len(missing_index) == 0:
            raise ValueError(
                f"{name}: irregular index detected but missing timestamps could not be inferred. "
                f"Examples: {irregular_examples}"
            )

        frame = frame.reindex(full_index).ffill()
        inserted_examples = [str(ts) for ts in missing_index[:log_limit]]
        print(
            f"[{name}] Forward-fill repair: inserted {len(missing_index)} missing rows "
            f"for freq={FREQ}. Example timestamps: {inserted_examples}"
        )

    repaired_diffs = frame.index.to_series().diff().dropna()
    if (repaired_diffs != expected_delta).any():
        raise RuntimeError(
            f"{name}: failed to regularize time index. Expected all deltas to be {expected_delta}."
        )

    inferred = repaired_diffs.mode().iloc[0]
    if inferred != expected_delta:
        raise ValueError(f"{name}: inferred step {inferred} does not match expected {expected_delta}")

    return frame


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
    if freq == "1sec":
        raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce").dt.floor("sec")
    else:
        raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce").dt.floor("min")

    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    raw = raw[~raw["timestamp"].duplicated(keep="last")].copy()
    raw = raw.set_index("timestamp")
    raw = validate_regular_time_index(raw, EXPECTED_DELTA, name=f"{asset} raw", fill_missing=True)

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
            f"{asset}: required columns contain NaNs after parsing. Columns with missing values: {bad_cols}"
        )

    out = validate_regular_time_index(out, EXPECTED_DELTA, name=f"{asset} standardized", fill_missing=True)
    return out


def load_and_align_assets(cfg: Dict[str, Any]) -> pd.DataFrame:
    aligned: Optional[pd.DataFrame] = None
    for asset in ASSETS:
        one = load_one_asset_raw(asset, cfg)
        aligned = one if aligned is None else aligned.join(one, how="inner")

    if aligned is None or len(aligned) == 0:
        raise RuntimeError("No data available after multi-asset loading and alignment")

    aligned = aligned.sort_index()
    aligned = aligned[~aligned.index.duplicated(keep="last")].copy()
    aligned = validate_regular_time_index(
        aligned,
        EXPECTED_DELTA,
        name="aligned multigraph index",
        fill_missing=True,
    )

    for asset in ASSETS:
        log_mid = np.log(aligned[f"mid_{asset}"].astype(float).to_numpy() + EPS)
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
    relation_state_map: Dict[str, Dict[str, np.ndarray]] = {rel: {} for rel in RELATION_NAMES}

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

        price_state = lr.astype(np.float32)
        turnover_log = safe_log1p_np(buys + sells).astype(np.float32)
        flow_state = (flow_imbalance * turnover_log).astype(np.float32)

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

    x_node = np.stack(per_asset_node_features, axis=1).astype(np.float32)
    x_node = np.nan_to_num(x_node, nan=0.0, posinf=0.0, neginf=0.0)
    return x_node, node_feature_names, relation_state_map


X_NODE_RAW, NODE_FEATURE_NAMES, RELATION_STATE_MAP = build_node_features_and_relation_states(df, CFG)

print("X_NODE_RAW shape:", X_NODE_RAW.shape)
print("NODE_FEATURE_NAMES:", NODE_FEATURE_NAMES)
for rel in RELATION_NAMES:
    print(rel, "example state shape:", RELATION_STATE_MAP[rel][TARGET_ASSET].shape)


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

        rel_tensor = np.stack(per_edge, axis=1).astype(np.float32)
        rel_tensor = np.nan_to_num(rel_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        relation_tensors.append(rel_tensor)

    x_rel_edge = np.stack(relation_tensors, axis=1).astype(np.float32)
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


TARGET_MID = df[f"mid_{TARGET_ASSET}"].to_numpy(dtype=np.float64)
Y_RET = forward_log_return_from_mid(TARGET_MID, horizon_bars=HORIZON_BARS)

TRADE_LABEL_ABS_RETURN_THRESHOLD = (
    round_trip_cost_as_log_return(float(CFG["cost_bps_per_side"]))
    + float(CFG["trade_label_buffer_bps"]) * 1e-4
)

Y_TRADE = np.full(len(Y_RET), np.nan, dtype=np.float32)
valid_trade_mask = np.isfinite(Y_RET)
Y_TRADE[valid_trade_mask] = (
    np.abs(Y_RET[valid_trade_mask]) > TRADE_LABEL_ABS_RETURN_THRESHOLD
).astype(np.float32)

Y_DIR = np.full(len(Y_RET), np.nan, dtype=np.float32)
Y_DIR[valid_trade_mask] = (Y_RET[valid_trade_mask] > 0.0).astype(np.float32)

print("Finite target count:", int(np.isfinite(Y_RET).sum()))
print("Trade-label absolute-return threshold:", TRADE_LABEL_ABS_RETURN_THRESHOLD)
print("Trade positives:", int(np.nansum(Y_TRADE)))


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


# %%
def make_preholdout_and_holdout_split(
    n_samples: int,
    holdout_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    holdout_n = max(1, int(round(n_samples * float(holdout_frac))))
    preholdout_n = n_samples - gap_bars - holdout_n
    if preholdout_n <= 0:
        raise RuntimeError("Not enough samples left after reserving purge gap and final holdout.")

    idx_preholdout = np.arange(0, preholdout_n, dtype=np.int64)
    idx_holdout = np.arange(preholdout_n + gap_bars, preholdout_n + gap_bars + holdout_n, dtype=np.int64)

    if len(idx_holdout) != holdout_n:
        raise RuntimeError("Holdout indices were not constructed correctly.")
    if idx_holdout[-1] >= n_samples:
        raise RuntimeError("Holdout indices exceed available sample count.")
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise RuntimeError("Pre-holdout and holdout indices overlap.")

    return idx_preholdout, idx_holdout


def make_strictly_increasing_integer_grid(start: int, stop: int, n_points: int) -> np.ndarray:
    if stop < start:
        raise RuntimeError(f"Invalid integer grid range: start={start}, stop={stop}")
    if (stop - start + 1) < n_points:
        raise RuntimeError(
            f"Cannot create {n_points} strictly increasing integer points inside [{start}, {stop}]"
        )

    grid = np.round(np.linspace(start, stop, num=n_points)).astype(np.int64)
    grid[0] = int(start)
    grid[-1] = int(stop)

    for i in range(1, n_points):
        grid[i] = max(int(grid[i]), int(grid[i - 1]) + 1)
    for i in range(n_points - 2, -1, -1):
        grid[i] = min(int(grid[i]), int(grid[i + 1]) - 1)

    if grid[0] < start or grid[-1] > stop or len(np.unique(grid)) != n_points:
        raise RuntimeError("Failed to construct the required strictly increasing train-end grid")
    return grid.astype(np.int64)


def make_exact_walk_forward_splits(
    idx_preholdout: np.ndarray,
    n_folds: int,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    gap_bars: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_pre = len(idx_preholdout)
    train_min = max(1, int(round(n_pre * float(train_min_frac))))
    val_n = max(1, int(round(n_pre * float(val_window_frac))))
    test_n = max(1, int(round(n_pre * float(test_window_frac))))

    max_train_end = n_pre - (2 * gap_bars + val_n + test_n)
    if max_train_end < train_min:
        raise RuntimeError(
            "Not enough pre-holdout samples to build the requested number of exact walk-forward folds."
        )

    train_ends = make_strictly_increasing_integer_grid(train_min, max_train_end, int(n_folds))
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for train_end in train_ends.tolist():
        val_start = int(train_end + gap_bars)
        val_end = int(val_start + val_n)
        test_start = int(val_end + gap_bars)
        test_end = int(test_start + test_n)

        idx_train = idx_preholdout[:train_end].copy()
        idx_val = idx_preholdout[val_start:val_end].copy()
        idx_test = idx_preholdout[test_start:test_end].copy()

        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            raise RuntimeError("Encountered an empty split while building exact walk-forward folds")
        splits.append((idx_train, idx_val, idx_test))

    if len(splits) != int(n_folds):
        raise RuntimeError(f"Expected exactly {n_folds} folds, got {len(splits)}")
    return splits


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
        raise RuntimeError("Not enough pre-holdout samples for the final production split")

    idx_train_final = idx_preholdout[:train_end].copy()
    idx_val_final = idx_preholdout[train_end + gap_bars: train_end + gap_bars + val_n].copy()
    idx_test_final = idx_holdout.copy()

    if len(idx_val_final) != val_n:
        raise RuntimeError("Final validation window was not created correctly")
    return idx_train_final, idx_val_final, idx_test_final


def assert_sorted_unique_indices(indices: np.ndarray, name: str) -> None:
    if len(indices) == 0:
        raise AssertionError(f"{name} is empty")
    if not np.all(indices[:-1] < indices[1:]):
        raise AssertionError(f"{name} must be strictly increasing and unique")


def assert_time_order_and_purge(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    gap_bars: int,
    label: str,
) -> None:
    assert_sorted_unique_indices(idx_train, f"{label}.train")
    assert_sorted_unique_indices(idx_val, f"{label}.val")
    assert_sorted_unique_indices(idx_test, f"{label}.test")

    if len(np.intersect1d(idx_train, idx_val)) > 0:
        raise AssertionError(f"{label}: train and val overlap")
    if len(np.intersect1d(idx_train, idx_test)) > 0:
        raise AssertionError(f"{label}: train and test overlap")
    if len(np.intersect1d(idx_val, idx_test)) > 0:
        raise AssertionError(f"{label}: val and test overlap")

    train_last = int(idx_train[-1])
    val_first = int(idx_val[0])
    val_last = int(idx_val[-1])
    test_first = int(idx_test[0])

    if not train_last < val_first < test_first:
        raise AssertionError(f"{label}: split windows are not strictly time ordered")
    if (val_first - train_last) <= gap_bars:
        raise AssertionError(f"{label}: purge gap between train and val is not respected")
    if (test_first - val_last) <= gap_bars:
        raise AssertionError(f"{label}: purge gap between val and test is not respected")


def validate_all_splits(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    idx_train_final: np.ndarray,
    idx_val_final: np.ndarray,
    idx_test_final: np.ndarray,
    gap_bars: int,
) -> None:
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise AssertionError("Pre-holdout and holdout overlap")

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(cv_splits, start=1):
        assert_time_order_and_purge(idx_train, idx_val, idx_test, gap_bars, label=f"cv_fold_{fold_idx}")

        if not np.all(np.isin(idx_train, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: train contains non-preholdout indices")
        if not np.all(np.isin(idx_val, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: val contains non-preholdout indices")
        if not np.all(np.isin(idx_test, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: test contains non-preholdout indices")
        if len(np.intersect1d(idx_train, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: holdout leaked into train")
        if len(np.intersect1d(idx_val, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: holdout leaked into val")
        if len(np.intersect1d(idx_test, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: holdout leaked into test")

    assert_time_order_and_purge(
        idx_train_final,
        idx_val_final,
        idx_test_final,
        gap_bars,
        label="final_production_split",
    )
    if not np.all(np.isin(idx_train_final, idx_preholdout)):
        raise AssertionError("Final train contains non-preholdout indices")
    if not np.all(np.isin(idx_val_final, idx_preholdout)):
        raise AssertionError("Final val contains non-preholdout indices")
    if not np.array_equal(idx_test_final, idx_holdout):
        raise AssertionError("Final production test must match the final holdout exactly")


IDX_PREHOLDOUT, IDX_HOLDOUT = make_preholdout_and_holdout_split(
    n_samples=N_SAMPLES,
    holdout_frac=float(CFG["final_holdout_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

WALK_FORWARD_SPLITS = make_exact_walk_forward_splits(
    idx_preholdout=IDX_PREHOLDOUT,
    n_folds=int(CFG["preholdout_n_folds"]),
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

IDX_TRAIN_FINAL, IDX_VAL_FINAL, IDX_TEST_FINAL = make_final_production_split(
    idx_preholdout=IDX_PREHOLDOUT,
    idx_holdout=IDX_HOLDOUT,
    val_window_frac=float(CFG["val_window_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

validate_all_splits(
    idx_preholdout=IDX_PREHOLDOUT,
    idx_holdout=IDX_HOLDOUT,
    cv_splits=WALK_FORWARD_SPLITS,
    idx_train_final=IDX_TRAIN_FINAL,
    idx_val_final=IDX_VAL_FINAL,
    idx_test_final=IDX_TEST_FINAL,
    gap_bars=PURGE_GAP_BARS,
)

print("Pre-holdout samples:", len(IDX_PREHOLDOUT))
print("Holdout samples:", len(IDX_HOLDOUT))
print("Number of CV folds:", len(WALK_FORWARD_SPLITS))
for i, (tr, va, te) in enumerate(WALK_FORWARD_SPLITS, start=1):
    print(f"Fold {i}: train={len(tr)} val={len(va)} test={len(te)}")
print("Production split sizes:")
print("train_final:", len(IDX_TRAIN_FINAL))
print("val_final  :", len(IDX_VAL_FINAL))
print("holdout    :", len(IDX_TEST_FINAL))


# %%
class TemporalMultigraphDataset(Dataset):
    def __init__(
        self,
        x_node: np.ndarray,
        x_rel_edge: np.ndarray,
        y_ret: np.ndarray,
        y_trade: np.ndarray,
        y_dir: np.ndarray,
        sample_t: np.ndarray,
        sample_indices: np.ndarray,
        lookback_bars: int,
    ):
        self.x_node = x_node
        self.x_rel_edge = x_rel_edge
        self.y_ret = y_ret
        self.y_trade = y_trade
        self.y_dir = y_dir
        self.sample_t = sample_t.astype(np.int64)
        self.sample_indices = sample_indices.astype(np.int64)
        self.lookback_bars = int(lookback_bars)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, i: int):
        sample_idx = int(self.sample_indices[i])
        raw_t = int(self.sample_t[sample_idx])
        start = raw_t - self.lookback_bars + 1

        x_node_seq = self.x_node[start: raw_t + 1]
        x_edge_seq = self.x_rel_edge[start: raw_t + 1]
        y_ret = float(self.y_ret[raw_t])
        y_trade = float(self.y_trade[raw_t])
        y_dir_raw = self.y_dir[raw_t]
        y_dir = float(0.0 if not np.isfinite(y_dir_raw) else y_dir_raw)
        dir_valid = float(y_trade > 0.5)

        if not np.isfinite(y_ret) or not np.isfinite(y_trade):
            raise RuntimeError(f"Encountered invalid target at raw_t={raw_t}")

        return (
            torch.from_numpy(x_node_seq),
            torch.from_numpy(x_edge_seq),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_trade, dtype=torch.float32),
            torch.tensor(y_dir, dtype=torch.float32),
            torch.tensor(dir_valid, dtype=torch.float32),
            torch.tensor(sample_idx, dtype=torch.long),
            torch.tensor(raw_t, dtype=torch.long),
        )


def temporal_multigraph_collate(batch):
    x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, dir_valid, sample_idx, raw_t = zip(*batch)
    return (
        torch.stack(x_node_seq, dim=0),
        torch.stack(x_edge_seq, dim=0),
        torch.stack(y_ret, dim=0),
        torch.stack(y_trade, dim=0),
        torch.stack(y_dir, dim=0),
        torch.stack(dir_valid, dim=0),
        torch.stack(sample_idx, dim=0),
        torch.stack(raw_t, dim=0),
    )


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
        "last_train_raw_t": last_train_t,
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


# %%
class CausalConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def _trim(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(-1) > target_len:
            return x[..., :target_len]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        y = self.conv1(x)
        y = self._trim(y, x.size(-1))
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self._trim(y, x.size(-1))
        y = self.dropout(y)

        y = y + residual
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y


class NodeTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        n_nodes: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.asset_emb = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.02)
        layers = []
        for i in range(int(num_layers)):
            dilation = 2 ** i
            layers.append(
                CausalConv1dBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_nodes, _ = x.shape
        h = self.input_proj(x)
        h = h + self.asset_emb.view(1, 1, n_nodes, -1)
        h = h.permute(0, 2, 3, 1).contiguous().view(bsz * n_nodes, -1, seq_len)
        for layer in self.layers:
            h = layer(h)
        h = h.view(bsz, n_nodes, -1, seq_len).permute(0, 3, 1, 2).contiguous()
        return h


class EdgeTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        for i in range(int(num_layers)):
            dilation = 2 ** i
            layers.append(
                CausalConv1dBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_rel, n_edges, _ = x.shape
        h = self.input_proj(x)
        h = h.permute(0, 2, 3, 4, 1).contiguous().view(bsz * n_rel * n_edges, -1, seq_len)
        for layer in self.layers:
            h = layer(h)
        h = h.view(bsz, n_rel, n_edges, -1, seq_len).permute(0, 4, 1, 2, 3).contiguous()
        return h


def aggregate_messages_to_dst(msg: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    out = msg.new_zeros(msg.size(0), n_nodes, msg.size(-1))
    for e in range(msg.size(1)):
        out[:, int(dst_idx[e].item()), :] += msg[:, e, :]
    return out


def edge_softmax_by_dst(logits: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    out = torch.zeros_like(logits)
    for node in range(n_nodes):
        mask = dst_idx == node
        if logits.ndim != 2:
            raise ValueError(f"Unsupported logits ndim={logits.ndim}")
        out[:, mask] = torch.softmax(logits[:, mask], dim=1)
    return out


class DynamicEdgeMPNNLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        indeg = torch.zeros(n_nodes, dtype=torch.float32)
        for dst in edge_index[:, 1].tolist():
            indeg[int(dst)] += 1.0
        indeg = indeg.clamp_min(1.0)
        self.register_buffer("indeg", indeg)
        self.register_buffer("edge_prior", (1.0 / indeg[edge_index[:, 1]]).float())

        self.src_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.self_proj = nn.Linear(node_dim, node_dim)
        self.agg_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_src = node_state[:, self.src_idx, :]
        h_dst = node_state[:, self.dst_idx, :]

        gate_input = torch.cat([h_src, h_dst, edge_state], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        msg = gate * self.src_proj(h_src) + self.edge_proj(edge_state)

        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)
        agg = agg / self.indeg.view(1, -1, 1)

        update = self.self_proj(node_state) + self.agg_proj(agg)
        out = self.norm(node_state + self.dropout(F.gelu(update)))

        raw_strength = gate.mean(dim=-1)
        norm_strength = edge_softmax_by_dst(raw_strength, self.dst_idx, self.n_nodes)
        adj_l1 = raw_strength.abs().mean()
        adj_prior = ((norm_strength - self.edge_prior.view(1, -1)) ** 2).mean()

        aux = {
            "adj_l1": adj_l1,
            "adj_prior": adj_prior,
        }
        return out, aux


class DynamicRelConvLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        indeg = torch.zeros(n_nodes, dtype=torch.float32)
        for dst in edge_index[:, 1].tolist():
            indeg[int(dst)] += 1.0
        indeg = indeg.clamp_min(1.0)
        self.register_buffer("edge_prior", (1.0 / indeg[edge_index[:, 1]]).float())

        self.src_proj = nn.Linear(node_dim, node_dim)
        self.edge_score_net = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
        )
        self.edge_shift = nn.Linear(edge_dim, node_dim)
        self.self_proj = nn.Linear(node_dim, node_dim)
        self.agg_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_src = node_state[:, self.src_idx, :]
        logits = self.edge_score_net(edge_state).squeeze(-1)
        alpha = edge_softmax_by_dst(logits, self.dst_idx, self.n_nodes)
        msg = alpha.unsqueeze(-1) * (self.src_proj(h_src) + self.edge_shift(edge_state))
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)

        update = self.self_proj(node_state) + self.agg_proj(agg)
        out = self.norm(node_state + self.dropout(F.gelu(update)))

        adj_l1 = alpha.new_tensor(0.0)
        adj_prior = ((alpha - self.edge_prior.view(1, -1)) ** 2).mean()
        aux = {
            "adj_l1": adj_l1,
            "adj_prior": adj_prior,
        }
        return out, aux


class RelationGraphBlock(nn.Module):
    def __init__(
        self,
        operator_name: str,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.operator_name = str(operator_name)
        layers = []
        for _ in range(int(num_layers)):
            if self.operator_name == "dynamic_edge_mpnn":
                layers.append(
                    DynamicEdgeMPNNLayer(
                        node_dim=node_dim,
                        edge_dim=edge_dim,
                        n_nodes=n_nodes,
                        edge_index=edge_index,
                        dropout=dropout,
                    )
                )
            elif self.operator_name == "dynamic_rel_conv":
                layers.append(
                    DynamicRelConvLayer(
                        node_dim=node_dim,
                        edge_dim=edge_dim,
                        n_nodes=n_nodes,
                        edge_index=edge_index,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unsupported graph operator: {self.operator_name}")
        self.layers = nn.ModuleList(layers)

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, n_nodes, node_dim = node_seq.shape
        _, _, n_edges, edge_dim = edge_seq.shape

        flat_nodes = node_seq.reshape(bsz * seq_len, n_nodes, node_dim)
        flat_edges = edge_seq.reshape(bsz * seq_len, n_edges, edge_dim)

        out = flat_nodes
        total_adj_l1 = flat_nodes.new_tensor(0.0)
        total_adj_prior = flat_nodes.new_tensor(0.0)

        for layer in self.layers:
            out, aux = layer(out, flat_edges)
            total_adj_l1 = total_adj_l1 + aux["adj_l1"]
            total_adj_prior = total_adj_prior + aux["adj_prior"]

        out = out.reshape(bsz, seq_len, n_nodes, node_dim)
        aux_out = {
            "adj_l1": total_adj_l1 / max(len(self.layers), 1),
            "adj_prior": total_adj_prior / max(len(self.layers), 1),
        }
        return out, aux_out


class RelationAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, fusion_hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_relations = int(num_relations)
        self.rel_emb = nn.Parameter(torch.randn(num_relations, hidden_dim) * 0.02)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, relation_node_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_bias = self.rel_emb.view(1, 1, self.num_relations, 1, self.hidden_dim)
        score_input = relation_node_seq + rel_bias
        scores = self.score_mlp(score_input).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        fused = (weights.unsqueeze(-1) * relation_node_seq).sum(dim=2)
        return fused, weights


class TargetTemporalTower(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        for i in range(int(num_layers)):
            dilation = 2 ** i
            layers.append(
                CausalConv1dBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, target_seq: torch.Tensor) -> torch.Tensor:
        h = target_seq.transpose(1, 2)
        for layer in self.layers:
            h = layer(h)
        h = h.transpose(1, 2)
        last_h = h[:, -1, :]
        return self.post(last_h)


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class MultigraphTemporalFusionModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        target_node: int,
        relation_names: List[str],
        cfg: Dict[str, Any],
    ):
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)
        self.relation_names = list(relation_names)

        node_hidden_dim = int(cfg["node_hidden_dim"])
        edge_hidden_dim = int(cfg["edge_hidden_dim"])
        target_hidden_dim = int(cfg["target_hidden_dim"])
        graph_layers = int(cfg["graph_layers"])
        kernel_size = int(cfg["temporal_kernel_size"])
        dropout = float(cfg["dropout"])
        operator_name = str(cfg["graph_operator"])

        self.node_encoder = NodeTemporalEncoder(
            input_dim=node_dim,
            hidden_dim=node_hidden_dim,
            num_layers=int(cfg["node_temporal_layers"]),
            kernel_size=kernel_size,
            dropout=dropout,
            n_nodes=n_nodes,
        )
        self.edge_encoder = EdgeTemporalEncoder(
            input_dim=edge_dim,
            hidden_dim=edge_hidden_dim,
            num_layers=int(cfg["edge_temporal_layers"]),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.relation_blocks = nn.ModuleDict(
            {
                rel: RelationGraphBlock(
                    operator_name=operator_name,
                    node_dim=node_hidden_dim,
                    edge_dim=edge_hidden_dim,
                    n_nodes=n_nodes,
                    edge_index=EDGE_INDEX,
                    num_layers=graph_layers,
                    dropout=dropout,
                )
                for rel in self.relation_names
            }
        )

        self.fusion = RelationAttentionFusion(
            hidden_dim=node_hidden_dim,
            num_relations=len(self.relation_names),
            fusion_hidden_dim=int(cfg["fusion_hidden_dim"]),
        )

        self.target_proj = nn.Linear(node_hidden_dim, target_hidden_dim)
        self.target_tower = TargetTemporalTower(
            hidden_dim=target_hidden_dim,
            num_layers=int(cfg["target_temporal_layers"]),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.trade_head = PredictionHead(target_hidden_dim, dropout)
        self.dir_head = PredictionHead(target_hidden_dim, dropout)
        self.fixed_head = PredictionHead(target_hidden_dim, dropout)

    def forward(
        self,
        x_node_seq: torch.Tensor,
        x_edge_seq: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x_node_seq = torch.nan_to_num(x_node_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x_edge_seq = torch.nan_to_num(x_edge_seq, nan=0.0, posinf=0.0, neginf=0.0)

        node_seq = self.node_encoder(x_node_seq)
        edge_seq = self.edge_encoder(x_edge_seq)

        relation_outputs = []
        total_adj_l1 = x_node_seq.new_tensor(0.0)
        total_adj_prior = x_node_seq.new_tensor(0.0)

        for r, rel in enumerate(self.relation_names):
            rel_node_seq, rel_aux = self.relation_blocks[rel](node_seq, edge_seq[:, :, r, :, :])
            relation_outputs.append(rel_node_seq)
            total_adj_l1 = total_adj_l1 + rel_aux["adj_l1"]
            total_adj_prior = total_adj_prior + rel_aux["adj_prior"]

        relation_stack = torch.stack(relation_outputs, dim=2)
        fused_node_seq, relation_weights = self.fusion(relation_stack)

        target_seq = fused_node_seq[:, :, self.target_node, :]
        target_seq = self.target_proj(target_seq)
        shared_target_state = self.target_tower(target_seq)

        trade_logit = torch.nan_to_num(self.trade_head(shared_target_state), nan=0.0, posinf=0.0, neginf=0.0)
        dir_logit = torch.nan_to_num(self.dir_head(shared_target_state), nan=0.0, posinf=0.0, neginf=0.0)
        fixed_ret = torch.nan_to_num(self.fixed_head(shared_target_state), nan=0.0, posinf=0.0, neginf=0.0)

        result: Dict[str, torch.Tensor] = {
            "trade_logit": trade_logit,
            "dir_logit": dir_logit,
            "fixed_ret": fixed_ret,
            "adj_l1_penalty": total_adj_l1 / max(len(self.relation_names), 1),
            "adj_prior_penalty": total_adj_prior / max(len(self.relation_names), 1),
        }
        if return_aux:
            result["relation_weights"] = relation_weights
            result["relation_node_seq"] = relation_stack
            result["fused_node_seq"] = fused_node_seq
            result["shared_target_state"] = shared_target_state
        return result


# %%
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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def finite_or_default(value: Any, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, str, bool)) or value is None


def flatten_metrics_row(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in metrics.items():
        if k in {"trades_df", "threshold_grid_df"}:
            continue
        if is_scalar_metric(v):
            row[f"{prefix}{k}" if prefix else k] = v
    return row


def scalarize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if is_scalar_metric(v):
            out[k] = v
    return out


def compute_binary_pos_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    labels = labels[np.isfinite(labels)]
    if len(labels) == 0:
        return 1.0

    pos = float((labels > 0.5).sum())
    neg = float((labels <= 0.5).sum())
    if pos <= 0.0 or neg <= 0.0:
        return 1.0
    return float(np.clip(neg / max(pos, 1.0), 0.25, 25.0))


def compute_soft_utility_np(
    trade_logit: np.ndarray,
    dir_logit: np.ndarray,
    y_ret: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, float, float]:
    trade_prob = sigmoid_np(trade_logit)
    dir_term = np.tanh(float(cfg["utility_tanh_k"]) * np.asarray(dir_logit, dtype=np.float64))
    utility = trade_prob * dir_term * np.asarray(y_ret, dtype=np.float64)
    if len(utility) == 0:
        return utility, float("nan"), float("nan")
    soft_utility_mean = float(np.mean(utility))
    scaled_soft_utility = float(soft_utility_mean * float(cfg["utility_bps_scale"]))
    return utility, soft_utility_mean, scaled_soft_utility


def positions_from_threshold_pair(
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    thr_trade: float,
    thr_dir: float,
) -> np.ndarray:
    trade_prob = np.asarray(trade_prob, dtype=np.float64)
    dir_prob = np.asarray(dir_prob, dtype=np.float64)

    long_mask = (trade_prob >= float(thr_trade)) & (dir_prob >= float(thr_dir))
    short_mask = (trade_prob >= float(thr_trade)) & (dir_prob <= (1.0 - float(thr_dir)))

    position = np.zeros(len(trade_prob), dtype=np.int8)
    position[long_mask] = 1
    position[short_mask] = -1

    both = long_mask & short_mask
    if both.any():
        tie_idx = np.where(both)[0]
        choose_long = dir_prob[tie_idx] >= 0.5
        position[tie_idx[choose_long]] = 1
        position[tie_idx[~choose_long]] = -1
    return position


def sequential_threshold_backtest(
    y_ret: np.ndarray,
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    fixed_ret_pred: np.ndarray,
    raw_t_indices: np.ndarray,
    timestamps: pd.Series,
    thr_trade: float,
    thr_dir: float,
    horizon_bars: int,
    cost_bps_per_side: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_ret = np.asarray(y_ret, dtype=np.float64)
    trade_prob = np.asarray(trade_prob, dtype=np.float64)
    dir_prob = np.asarray(dir_prob, dtype=np.float64)
    fixed_ret_pred = np.asarray(fixed_ret_pred, dtype=np.float64)
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)

    position = positions_from_threshold_pair(trade_prob, dir_prob, thr_trade, thr_dir)
    round_trip_cost = round_trip_cost_as_log_return(cost_bps_per_side)

    rows: List[Dict[str, Any]] = []
    i = 0
    n = len(y_ret)

    while i < n:
        pos = int(position[i])
        if pos == 0:
            i += 1
            continue

        realized_ret = float(y_ret[i])
        gross_pnl = float(pos * realized_ret)
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
                "position": pos,
                "trade_prob": float(trade_prob[i]),
                "dir_prob": float(dir_prob[i]),
                "fixed_ret_pred": float(fixed_ret_pred[i]),
                "future_return": realized_ret,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
            }
        )
        i += int(horizon_bars)

    trades_df = pd.DataFrame(rows)
    n_trades = int(len(trades_df))

    if n_trades == 0:
        empty_metrics = {
            "thr_trade": float(thr_trade),
            "thr_dir": float(thr_dir),
            "gross_pnl_sum": 0.0,
            "pnl_sum": 0.0,
            "pnl_per_trade": float("nan"),
            "n_trades": 0,
            "trade_rate": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl_sum": 0.0,
            "short_pnl_sum": 0.0,
            "win_rate": float("nan"),
            "directional_hit_rate": float("nan"),
            "long_precision": float("nan"),
            "short_precision": float("nan"),
            "sharpe_like": float("nan"),
        }
        return empty_metrics, trades_df

    gross_pnl_sum = float(trades_df["gross_pnl"].sum())
    pnl_sum = float(trades_df["net_pnl"].sum())
    pnl_per_trade = float(pnl_sum / n_trades)

    if n_trades >= 2 and float(trades_df["net_pnl"].std(ddof=1)) > 1e-12:
        sharpe_like = float(
            trades_df["net_pnl"].mean() / trades_df["net_pnl"].std(ddof=1) * np.sqrt(n_trades)
        )
    else:
        sharpe_like = float("nan")

    long_trades = int((trades_df["position"] == 1).sum())
    short_trades = int((trades_df["position"] == -1).sum())
    long_pnl_sum = float(trades_df.loc[trades_df["position"] == 1, "net_pnl"].sum())
    short_pnl_sum = float(trades_df.loc[trades_df["position"] == -1, "net_pnl"].sum())
    win_rate = float((trades_df["net_pnl"] > 0.0).mean())

    long_precision = float((trades_df.loc[trades_df["position"] == 1, "future_return"] > 0.0).mean()) if long_trades else float("nan")
    short_precision = float((trades_df.loc[trades_df["position"] == -1, "future_return"] < 0.0).mean()) if short_trades else float("nan")
    directional_hit_rate = float((trades_df["position"] * trades_df["future_return"] > 0.0).mean())

    metrics = {
        "thr_trade": float(thr_trade),
        "thr_dir": float(thr_dir),
        "gross_pnl_sum": gross_pnl_sum,
        "pnl_sum": pnl_sum,
        "pnl_per_trade": pnl_per_trade,
        "n_trades": n_trades,
        "trade_rate": float(n_trades / max(len(y_ret), 1)),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pnl_sum": long_pnl_sum,
        "short_pnl_sum": short_pnl_sum,
        "win_rate": win_rate,
        "directional_hit_rate": directional_hit_rate,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "sharpe_like": sharpe_like,
    }
    return metrics, trades_df


def find_best_threshold_pair(
    y_ret: np.ndarray,
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    fixed_ret_pred: np.ndarray,
    raw_t_indices: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[float, ...]] = None
    best_thresholds: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_trades_df: Optional[pd.DataFrame] = None

    for thr_trade in [float(x) for x in cfg["threshold_trade_grid"]]:
        for thr_dir in [float(x) for x in cfg["threshold_dir_grid"]]:
            bt_metrics, trades_df = sequential_threshold_backtest(
                y_ret=y_ret,
                trade_prob=trade_prob,
                dir_prob=dir_prob,
                fixed_ret_pred=fixed_ret_pred,
                raw_t_indices=raw_t_indices,
                timestamps=TIMESTAMPS,
                thr_trade=thr_trade,
                thr_dir=thr_dir,
                horizon_bars=HORIZON_BARS,
                cost_bps_per_side=float(cfg["cost_bps_per_side"]),
            )
            feasible = int(bt_metrics["n_trades"]) >= int(cfg["min_threshold_search_trades"])
            row = {**bt_metrics, "feasible": bool(feasible)}
            rows.append(row)

            pnl_sum = finite_or_default(bt_metrics["pnl_sum"], -1e18)
            pnl_per_trade = finite_or_default(bt_metrics["pnl_per_trade"], -1e18)
            win_rate = finite_or_default(bt_metrics["win_rate"], -1e18)
            n_trades = finite_or_default(bt_metrics["n_trades"], -1e18)
            feasibility_rank = 1.0 if feasible else 0.0
            key = (feasibility_rank, pnl_sum, pnl_per_trade, win_rate, n_trades)

            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = {"thr_trade": float(thr_trade), "thr_dir": float(thr_dir)}
                best_metrics = bt_metrics
                best_trades_df = trades_df.copy()

    grid_df = pd.DataFrame(rows)
    if best_thresholds is None or best_metrics is None or best_trades_df is None:
        raise RuntimeError("Threshold-pair search failed to produce a result")

    return best_thresholds, grid_df, best_metrics, best_trades_df


def evaluate_predictions(
    y_ret: np.ndarray,
    y_trade: np.ndarray,
    y_dir: np.ndarray,
    trade_logit: np.ndarray,
    dir_logit: np.ndarray,
    fixed_ret_pred: np.ndarray,
    raw_t_indices: np.ndarray,
    cfg: Dict[str, Any],
    fixed_threshold_pair: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    y_ret = np.asarray(y_ret, dtype=np.float64)
    y_trade = np.asarray(y_trade, dtype=np.float64)
    y_dir = np.asarray(y_dir, dtype=np.float64)
    trade_logit = np.asarray(trade_logit, dtype=np.float64)
    dir_logit = np.asarray(dir_logit, dtype=np.float64)
    fixed_ret_pred = np.asarray(fixed_ret_pred, dtype=np.float64)
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)

    trade_prob = sigmoid_np(trade_logit)
    dir_prob = sigmoid_np(dir_logit)

    dir_mask = (y_trade > 0.5) & np.isfinite(y_dir)
    trade_auc = safe_roc_auc(y_trade, trade_prob)
    dir_auc = safe_roc_auc(y_dir[dir_mask], dir_prob[dir_mask]) if dir_mask.any() else float("nan")
    dir_auc_for_selection = finite_or_default(dir_auc, 0.5)

    utility_vector, soft_utility_mean, scaled_soft_utility = compute_soft_utility_np(
        trade_logit=trade_logit,
        dir_logit=dir_logit,
        y_ret=y_ret,
        cfg=cfg,
    )
    selection_score = float(scaled_soft_utility + 0.55 * dir_auc_for_selection)

    regression_metrics = {
        "rmse": rmse_np(y_ret, fixed_ret_pred),
        "mae": mae_np(y_ret, fixed_ret_pred),
        "ic": ic_np(y_ret, fixed_ret_pred),
    }

    if fixed_threshold_pair is None:
        selected_threshold_pair, threshold_grid_df, threshold_metrics, trades_df = find_best_threshold_pair(
            y_ret=y_ret,
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            fixed_ret_pred=fixed_ret_pred,
            raw_t_indices=raw_t_indices,
            cfg=cfg,
        )
    else:
        selected_threshold_pair = {
            "thr_trade": float(fixed_threshold_pair["thr_trade"]),
            "thr_dir": float(fixed_threshold_pair["thr_dir"]),
        }
        threshold_metrics, trades_df = sequential_threshold_backtest(
            y_ret=y_ret,
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            fixed_ret_pred=fixed_ret_pred,
            raw_t_indices=raw_t_indices,
            timestamps=TIMESTAMPS,
            thr_trade=float(selected_threshold_pair["thr_trade"]),
            thr_dir=float(selected_threshold_pair["thr_dir"]),
            horizon_bars=HORIZON_BARS,
            cost_bps_per_side=float(cfg["cost_bps_per_side"]),
        )
        threshold_grid_df = pd.DataFrame()

    out = {
        **regression_metrics,
        "trade_auc": trade_auc,
        "dir_auc": dir_auc,
        "dir_auc_for_selection": dir_auc_for_selection,
        "soft_utility_mean": soft_utility_mean,
        "scaled_soft_utility": scaled_soft_utility,
        "selection_score": selection_score,
        **threshold_metrics,
        "threshold_grid_df": threshold_grid_df,
        "trades_df": trades_df,
    }
    return out


def compute_multitask_loss(
    model_out: Dict[str, torch.Tensor],
    y_ret: torch.Tensor,
    y_trade: torch.Tensor,
    y_dir: torch.Tensor,
    dir_valid: torch.Tensor,
    trade_pos_weight: torch.Tensor,
    dir_pos_weight: torch.Tensor,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    trade_logit = model_out["trade_logit"].view(-1)
    dir_logit = model_out["dir_logit"].view(-1)
    fixed_ret = model_out["fixed_ret"].view(-1)

    y_ret = y_ret.view(-1)
    y_trade = y_trade.view(-1)
    y_dir = y_dir.view(-1)
    dir_valid = dir_valid.view(-1) > 0.5

    trade_loss = F.binary_cross_entropy_with_logits(
        trade_logit,
        y_trade,
        pos_weight=trade_pos_weight.to(trade_logit.device),
    )

    if dir_valid.any():
        dir_loss = F.binary_cross_entropy_with_logits(
            dir_logit[dir_valid],
            y_dir[dir_valid],
            pos_weight=dir_pos_weight.to(dir_logit.device),
        )
    else:
        dir_loss = trade_logit.new_tensor(0.0)

    ret_loss = F.smooth_l1_loss(
        fixed_ret,
        y_ret,
        beta=float(cfg["huber_beta"]),
        reduction="mean",
    )

    utility = torch.sigmoid(trade_logit) * torch.tanh(float(cfg["utility_tanh_k"]) * dir_logit) * y_ret
    utility_mean = utility.mean()
    scaled_soft_utility = utility_mean * float(cfg["utility_bps_scale"])
    utility_loss = -scaled_soft_utility

    adj_l1_penalty = model_out.get("adj_l1_penalty", trade_logit.new_tensor(0.0))
    adj_prior_penalty = model_out.get("adj_prior_penalty", trade_logit.new_tensor(0.0))

    total_loss = (
        float(cfg["loss_w_trade"]) * trade_loss
        + float(cfg["loss_w_dir"]) * dir_loss
        + float(cfg["loss_w_ret"]) * ret_loss
        + float(cfg["loss_w_utility"]) * utility_loss
        + float(cfg["adj_l1_lambda"]) * adj_l1_penalty
        + float(cfg["adj_prior_lambda"]) * adj_prior_penalty
    )

    return {
        "total_loss": total_loss,
        "trade_loss": trade_loss.detach(),
        "dir_loss": dir_loss.detach(),
        "ret_loss": ret_loss.detach(),
        "utility_loss": utility_loss.detach(),
        "soft_utility_mean": utility_mean.detach(),
        "scaled_soft_utility": scaled_soft_utility.detach(),
        "adj_l1_penalty": adj_l1_penalty.detach(),
        "adj_prior_penalty": adj_prior_penalty.detach(),
    }


# %%
@torch.no_grad()
def collect_predictions_from_loader(model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
    model.eval()

    trade_logits: List[np.ndarray] = []
    dir_logits: List[np.ndarray] = []
    fixed_preds: List[np.ndarray] = []
    y_ret_all: List[np.ndarray] = []
    y_trade_all: List[np.ndarray] = []
    y_dir_all: List[np.ndarray] = []
    sample_idx_all: List[np.ndarray] = []
    raw_t_all: List[np.ndarray] = []

    for x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, _dir_valid, sample_idx, raw_t in loader:
        x_node_seq = x_node_seq.to(DEVICE).float()
        x_edge_seq = x_edge_seq.to(DEVICE).float()

        out = model(x_node_seq, x_edge_seq)

        trade_logits.append(out["trade_logit"].detach().cpu().numpy())
        dir_logits.append(out["dir_logit"].detach().cpu().numpy())
        fixed_preds.append(out["fixed_ret"].detach().cpu().numpy())
        y_ret_all.append(y_ret.detach().cpu().numpy())
        y_trade_all.append(y_trade.detach().cpu().numpy())
        y_dir_all.append(y_dir.detach().cpu().numpy())
        sample_idx_all.append(sample_idx.detach().cpu().numpy())
        raw_t_all.append(raw_t.detach().cpu().numpy())

    trade_logit = np.concatenate(trade_logits, axis=0).astype(np.float64)
    dir_logit = np.concatenate(dir_logits, axis=0).astype(np.float64)
    fixed_ret = np.concatenate(fixed_preds, axis=0).astype(np.float64)
    target_ret = np.concatenate(y_ret_all, axis=0).astype(np.float64)
    target_trade = np.concatenate(y_trade_all, axis=0).astype(np.float64)
    target_dir = np.concatenate(y_dir_all, axis=0).astype(np.float64)
    sample_idx = np.concatenate(sample_idx_all, axis=0).astype(np.int64)
    raw_t = np.concatenate(raw_t_all, axis=0).astype(np.int64)

    return {
        "trade_logit": trade_logit,
        "dir_logit": dir_logit,
        "fixed_ret": fixed_ret,
        "trade_prob": sigmoid_np(trade_logit),
        "dir_prob": sigmoid_np(dir_logit),
        "target_ret": target_ret,
        "target_trade": target_trade,
        "target_dir": target_dir,
        "sample_idx": sample_idx,
        "raw_t": raw_t,
        "timestamp": TIMESTAMPS.iloc[raw_t].reset_index(drop=True),
    }


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
        y_trade=Y_TRADE,
        y_dir=Y_DIR,
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
    return collect_predictions_from_loader(model, loader)


# %%
@dataclass
class SplitArtifacts:
    model_state: Dict[str, torch.Tensor]
    node_scaler_params: Dict[str, Any]
    relation_scaler_params: Dict[str, Dict[str, Any]]
    best_epoch: int
    best_checkpoint_key: Tuple[float, ...]
    best_checkpoint_summary: Dict[str, Any]
    training_history_df: pd.DataFrame
    selected_threshold_pair: Dict[str, float]
    validation_threshold_grid: pd.DataFrame
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    val_predictions: Dict[str, Any]
    test_predictions: Dict[str, Any]


# %%
def build_model_for_cfg(cfg: Dict[str, Any]) -> MultigraphTemporalFusionModel:
    model = MultigraphTemporalFusionModel(
        node_dim=X_NODE_RAW.shape[-1],
        edge_dim=X_REL_EDGE_RAW.shape[-1],
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
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
        y_trade=Y_TRADE,
        y_dir=Y_DIR,
        sample_t=SAMPLE_T,
        sample_indices=idx_train,
        lookback_bars=LOOKBACK_BARS,
    )
    val_ds = TemporalMultigraphDataset(
        x_node=x_node_scaled,
        x_rel_edge=x_rel_edge_scaled,
        y_ret=Y_RET,
        y_trade=Y_TRADE,
        y_dir=Y_DIR,
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

    train_trade_labels = Y_TRADE[SAMPLE_T[idx_train]].astype(np.float64)
    trade_pos_weight = torch.tensor(compute_binary_pos_weight(train_trade_labels), dtype=torch.float32, device=DEVICE)

    train_dir_mask = train_trade_labels > 0.5
    train_dir_labels = Y_DIR[SAMPLE_T[idx_train]][train_dir_mask].astype(np.float64)
    dir_pos_weight = torch.tensor(compute_binary_pos_weight(train_dir_labels), dtype=torch.float32, device=DEVICE)

    model = build_model_for_cfg(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["scheduler_factor"]),
        patience=int(cfg["scheduler_patience"]),
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    best_checkpoint_key: Optional[Tuple[float, ...]] = None
    best_checkpoint_summary: Dict[str, Any] = {}
    bad_epochs = 0
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        batch_rows: List[Dict[str, float]] = []

        for x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, dir_valid, _sample_idx, _raw_t in train_loader:
            x_node_seq = x_node_seq.to(DEVICE).float()
            x_edge_seq = x_edge_seq.to(DEVICE).float()
            y_ret = y_ret.to(DEVICE).float()
            y_trade = y_trade.to(DEVICE).float()
            y_dir = y_dir.to(DEVICE).float()
            dir_valid = dir_valid.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            out = model(x_node_seq, x_edge_seq)
            loss_pack = compute_multitask_loss(
                model_out=out,
                y_ret=y_ret,
                y_trade=y_trade,
                y_dir=y_dir,
                dir_valid=dir_valid,
                trade_pos_weight=trade_pos_weight,
                dir_pos_weight=dir_pos_weight,
                cfg=cfg,
            )
            total_loss = loss_pack["total_loss"]
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            batch_rows.append(
                {
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "trade_loss": float(loss_pack["trade_loss"].cpu().item()),
                    "dir_loss": float(loss_pack["dir_loss"].cpu().item()),
                    "ret_loss": float(loss_pack["ret_loss"].cpu().item()),
                    "utility_loss": float(loss_pack["utility_loss"].cpu().item()),
                    "soft_utility_mean": float(loss_pack["soft_utility_mean"].cpu().item()),
                    "scaled_soft_utility": float(loss_pack["scaled_soft_utility"].cpu().item()),
                    "adj_l1_penalty": float(loss_pack["adj_l1_penalty"].cpu().item()),
                    "adj_prior_penalty": float(loss_pack["adj_prior_penalty"].cpu().item()),
                }
            )

        val_pred_pack = collect_predictions_from_loader(model, val_loader)
        val_metrics_epoch = evaluate_predictions(
            y_ret=val_pred_pack["target_ret"],
            y_trade=val_pred_pack["target_trade"],
            y_dir=val_pred_pack["target_dir"],
            trade_logit=val_pred_pack["trade_logit"],
            dir_logit=val_pred_pack["dir_logit"],
            fixed_ret_pred=val_pred_pack["fixed_ret"],
            raw_t_indices=val_pred_pack["raw_t"],
            cfg=cfg,
            fixed_threshold_pair=None,
        )

        selection_score = float(val_metrics_epoch["selection_score"])
        checkpoint_key = (
            selection_score,
            finite_or_default(val_metrics_epoch["pnl_sum"], -1e18),
            finite_or_default(val_metrics_epoch["dir_auc_for_selection"], 0.5),
            finite_or_default(val_metrics_epoch["trade_auc"], -1e18),
            -finite_or_default(val_metrics_epoch["rmse"], 1e18),
        )
        scheduler.step(selection_score)

        train_means = pd.DataFrame(batch_rows).mean(numeric_only=True).to_dict() if batch_rows else {}
        history_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": float(v) for k, v in train_means.items()},
            **{f"val_{k}": v for k, v in scalarize_dict(val_metrics_epoch).items()},
        }
        history_rows.append(history_row)

        print(
            f"[{split_name}] "
            f"epoch={epoch:02d} "
            f"train_total={finite_or_default(train_means.get('total_loss'), float('nan')):.6f} "
            f"trade={finite_or_default(train_means.get('trade_loss'), float('nan')):.4f} "
            f"dir={finite_or_default(train_means.get('dir_loss'), float('nan')):.4f} "
            f"ret={finite_or_default(train_means.get('ret_loss'), float('nan')):.6f} "
            f"utility={finite_or_default(train_means.get('utility_loss'), float('nan')):.4f} "
            f"val_sel={selection_score:.4f} "
            f"val_soft_util_bps={finite_or_default(val_metrics_epoch.get('scaled_soft_utility'), float('nan')):.4f} "
            f"val_dir_auc={finite_or_default(val_metrics_epoch.get('dir_auc'), float('nan')):.4f} "
            f"val_trade_auc={finite_or_default(val_metrics_epoch.get('trade_auc'), float('nan')):.4f} "
            f"val_pnl={finite_or_default(val_metrics_epoch.get('pnl_sum'), float('nan')):.6f} "
            f"val_trades={int(finite_or_default(val_metrics_epoch.get('n_trades'), 0))} "
            f"thr_trade={finite_or_default(val_metrics_epoch.get('thr_trade'), float('nan')):.2f} "
            f"thr_dir={finite_or_default(val_metrics_epoch.get('thr_dir'), float('nan')):.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if best_checkpoint_key is None or checkpoint_key > best_checkpoint_key:
            best_checkpoint_key = checkpoint_key
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint_summary = {
                "epoch": int(epoch),
                "checkpoint_key": [float(x) for x in checkpoint_key],
                "selected_threshold_pair": {
                    "thr_trade": float(val_metrics_epoch["thr_trade"]),
                    "thr_dir": float(val_metrics_epoch["thr_dir"]),
                },
                "val_metrics": scalarize_dict(val_metrics_epoch),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            print(f"[{split_name}] early stopping at epoch {epoch}")
            break

    if best_state is None or best_checkpoint_key is None:
        raise RuntimeError(f"[{split_name}] no best checkpoint was captured during training")

    model.load_state_dict(best_state)

    val_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_val,
        batch_size=int(cfg["batch_size"]),
    )
    val_metrics = evaluate_predictions(
        y_ret=val_pred_pack["target_ret"],
        y_trade=val_pred_pack["target_trade"],
        y_dir=val_pred_pack["target_dir"],
        trade_logit=val_pred_pack["trade_logit"],
        dir_logit=val_pred_pack["dir_logit"],
        fixed_ret_pred=val_pred_pack["fixed_ret"],
        raw_t_indices=val_pred_pack["raw_t"],
        cfg=cfg,
        fixed_threshold_pair=None,
    )

    selected_threshold_pair = {
        "thr_trade": float(val_metrics["thr_trade"]),
        "thr_dir": float(val_metrics["thr_dir"]),
    }

    test_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_test,
        batch_size=int(cfg["batch_size"]),
    )
    test_metrics = evaluate_predictions(
        y_ret=test_pred_pack["target_ret"],
        y_trade=test_pred_pack["target_trade"],
        y_dir=test_pred_pack["target_dir"],
        trade_logit=test_pred_pack["trade_logit"],
        dir_logit=test_pred_pack["dir_logit"],
        fixed_ret_pred=test_pred_pack["fixed_ret"],
        raw_t_indices=test_pred_pack["raw_t"],
        cfg=cfg,
        fixed_threshold_pair=selected_threshold_pair,
    )

    print(
        f"[{split_name}] best_epoch={best_epoch} "
        f"best_selection_score={best_checkpoint_key[0]:.4f} "
        f"selected_thr_trade={selected_threshold_pair['thr_trade']:.2f} "
        f"selected_thr_dir={selected_threshold_pair['thr_dir']:.2f}"
    )
    print(
        f"[{split_name}] TEST "
        f"selection_score={finite_or_default(test_metrics['selection_score'], float('nan')):.4f} "
        f"scaled_soft_utility={finite_or_default(test_metrics['scaled_soft_utility'], float('nan')):.4f} "
        f"dir_auc={finite_or_default(test_metrics['dir_auc'], float('nan')):.4f} "
        f"trade_auc={finite_or_default(test_metrics['trade_auc'], float('nan')):.4f} "
        f"pnl_sum={finite_or_default(test_metrics['pnl_sum'], float('nan')):.6f} "
        f"pnl_per_trade={finite_or_default(test_metrics['pnl_per_trade'], float('nan')):.6f} "
        f"n_trades={int(finite_or_default(test_metrics['n_trades'], 0))}"
    )

    return SplitArtifacts(
        model_state=copy.deepcopy(model.state_dict()),
        node_scaler_params=node_scaler_params,
        relation_scaler_params=relation_scaler_params,
        best_epoch=best_epoch,
        best_checkpoint_key=best_checkpoint_key,
        best_checkpoint_summary=best_checkpoint_summary,
        training_history_df=pd.DataFrame(history_rows),
        selected_threshold_pair=selected_threshold_pair,
        validation_threshold_grid=val_metrics["threshold_grid_df"].copy(),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_predictions=val_pred_pack,
        test_predictions=test_pred_pack,
    )


# %%
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
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


# %%
OPERATOR_NAME = str(CFG["graph_operator"])
OPERATOR_DIR = ARTIFACT_ROOT / OPERATOR_NAME
OPERATOR_DIR.mkdir(parents=True, exist_ok=True)

cv_rows: List[Dict[str, Any]] = []
cv_fold_artifacts: List[SplitArtifacts] = []

for fold_idx, (idx_train, idx_val, idx_test) in enumerate(WALK_FORWARD_SPLITS, start=1):
    print("\n" + "=" * 110)
    print(
        f"RUNNING CV FOLD {fold_idx}/{len(WALK_FORWARD_SPLITS)} | "
        f"operator={OPERATOR_NAME} | train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
    )

    artifacts = train_one_split(
        split_name=f"{OPERATOR_NAME}_fold_{fold_idx:02d}",
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        cfg=CFG,
    )
    cv_fold_artifacts.append(artifacts)

    bundle_name = f"{OPERATOR_NAME}_fold_{fold_idx:02d}_best"
    save_bundle(
        bundle_dir=OPERATOR_DIR,
        bundle_name=bundle_name,
        model_state=artifacts.model_state,
        node_scaler_params=artifacts.node_scaler_params,
        relation_scaler_params=artifacts.relation_scaler_params,
        cfg=CFG,
        meta={
            "kind": "cv_fold_best",
            "operator_name": OPERATOR_NAME,
            "fold_idx": fold_idx,
            "best_epoch": artifacts.best_epoch,
            "best_checkpoint_key": list(artifacts.best_checkpoint_key),
            "best_checkpoint_summary": artifacts.best_checkpoint_summary,
            "selected_threshold_pair": artifacts.selected_threshold_pair,
            "idx_train": idx_train.tolist(),
            "idx_val": idx_val.tolist(),
            "idx_test": idx_test.tolist(),
        },
    )

    artifacts.training_history_df.to_csv(
        OPERATOR_DIR / f"{bundle_name}_training_history.csv",
        index=False,
    )
    artifacts.validation_threshold_grid.to_csv(
        OPERATOR_DIR / f"{bundle_name}_validation_threshold_grid.csv",
        index=False,
    )
    artifacts.val_metrics["trades_df"].to_csv(
        OPERATOR_DIR / f"{bundle_name}_val_trades.csv",
        index=False,
    )
    artifacts.test_metrics["trades_df"].to_csv(
        OPERATOR_DIR / f"{bundle_name}_test_trades.csv",
        index=False,
    )

    cv_row = {
        "operator": OPERATOR_NAME,
        "fold": fold_idx,
        "best_epoch": artifacts.best_epoch,
        "best_checkpoint_key": json.dumps(list(artifacts.best_checkpoint_key)),
        "selected_thr_trade": float(artifacts.selected_threshold_pair["thr_trade"]),
        "selected_thr_dir": float(artifacts.selected_threshold_pair["thr_dir"]),
        **flatten_metrics_row("val_", artifacts.val_metrics),
        **flatten_metrics_row("test_", artifacts.test_metrics),
    }
    cv_rows.append(cv_row)

CV_RESULTS_DF = pd.DataFrame(cv_rows)
CV_RESULTS_DF.to_csv(OPERATOR_DIR / f"{OPERATOR_NAME}_cv_results_summary.csv", index=False)

cv_mean_numeric = CV_RESULTS_DF.mean(numeric_only=True).to_dict()
CV_MEAN_DF = pd.DataFrame(
    [
        {
            "operator": OPERATOR_NAME,
            "graph_operator": OPERATOR_NAME,
            "cv_mean_test_selection_score": float(cv_mean_numeric.get("test_selection_score", np.nan)),
            "cv_mean_test_scaled_soft_utility": float(cv_mean_numeric.get("test_scaled_soft_utility", np.nan)),
            "cv_mean_test_dir_auc": float(cv_mean_numeric.get("test_dir_auc", np.nan)),
            "cv_mean_test_trade_auc": float(cv_mean_numeric.get("test_trade_auc", np.nan)),
            "cv_mean_test_pnl_sum": float(cv_mean_numeric.get("test_pnl_sum", np.nan)),
            "cv_mean_test_pnl_per_trade": float(cv_mean_numeric.get("test_pnl_per_trade", np.nan)),
            "cv_mean_test_trade_rate": float(cv_mean_numeric.get("test_trade_rate", np.nan)),
            "cv_mean_test_win_rate": float(cv_mean_numeric.get("test_win_rate", np.nan)),
            "cv_mean_test_directional_hit_rate": float(cv_mean_numeric.get("test_directional_hit_rate", np.nan)),
            "cv_mean_test_rmse": float(cv_mean_numeric.get("test_rmse", np.nan)),
            "cv_mean_test_mae": float(cv_mean_numeric.get("test_mae", np.nan)),
            "cv_mean_test_ic": float(cv_mean_numeric.get("test_ic", np.nan)),
        }
    ]
)
CV_MEAN_DF.to_csv(OPERATOR_DIR / f"{OPERATOR_NAME}_cv_mean_summary.csv", index=False)

print("\n" + "=" * 110)
print("CV_RESULTS_DF")
print(CV_RESULTS_DF)
print("\nCV_MEAN_DF")
print(CV_MEAN_DF)


# %%
print("\n" + "=" * 110)
print(
    f"RUNNING PRODUCTION REFIT | operator={OPERATOR_NAME} | "
    f"train={len(IDX_TRAIN_FINAL)} val={len(IDX_VAL_FINAL)} holdout={len(IDX_TEST_FINAL)}"
)

PRODUCTION_ARTIFACTS = train_one_split(
    split_name=f"{OPERATOR_NAME}_production_refit",
    idx_train=IDX_TRAIN_FINAL,
    idx_val=IDX_VAL_FINAL,
    idx_test=IDX_TEST_FINAL,
    cfg=CFG,
)

PRODUCTION_BUNDLE_NAME = f"{OPERATOR_NAME}_production_best"
save_bundle(
    bundle_dir=OPERATOR_DIR,
    bundle_name=PRODUCTION_BUNDLE_NAME,
    model_state=PRODUCTION_ARTIFACTS.model_state,
    node_scaler_params=PRODUCTION_ARTIFACTS.node_scaler_params,
    relation_scaler_params=PRODUCTION_ARTIFACTS.relation_scaler_params,
    cfg=CFG,
    meta={
        "kind": "production_best",
        "operator_name": OPERATOR_NAME,
        "best_epoch": PRODUCTION_ARTIFACTS.best_epoch,
        "best_checkpoint_key": list(PRODUCTION_ARTIFACTS.best_checkpoint_key),
        "best_checkpoint_summary": PRODUCTION_ARTIFACTS.best_checkpoint_summary,
        "selected_threshold_pair": PRODUCTION_ARTIFACTS.selected_threshold_pair,
        "idx_train": IDX_TRAIN_FINAL.tolist(),
        "idx_val": IDX_VAL_FINAL.tolist(),
        "idx_test": IDX_TEST_FINAL.tolist(),
    },
)

PRODUCTION_ARTIFACTS.training_history_df.to_csv(
    OPERATOR_DIR / f"{PRODUCTION_BUNDLE_NAME}_training_history.csv",
    index=False,
)
PRODUCTION_ARTIFACTS.validation_threshold_grid.to_csv(
    OPERATOR_DIR / f"{PRODUCTION_BUNDLE_NAME}_validation_threshold_grid.csv",
    index=False,
)
PRODUCTION_ARTIFACTS.val_metrics["trades_df"].to_csv(
    OPERATOR_DIR / f"{PRODUCTION_BUNDLE_NAME}_val_trades.csv",
    index=False,
)
PRODUCTION_ARTIFACTS.test_metrics["trades_df"].to_csv(
    OPERATOR_DIR / f"{PRODUCTION_BUNDLE_NAME}_holdout_trades.csv",
    index=False,
)

PRODUCTION_HOLDOUT_DF = pd.DataFrame(
    [
        {
            "operator": OPERATOR_NAME,
            "model_name": PRODUCTION_BUNDLE_NAME,
            "best_epoch": PRODUCTION_ARTIFACTS.best_epoch,
            "selected_thr_trade": float(PRODUCTION_ARTIFACTS.selected_threshold_pair["thr_trade"]),
            "selected_thr_dir": float(PRODUCTION_ARTIFACTS.selected_threshold_pair["thr_dir"]),
            **flatten_metrics_row("val_", PRODUCTION_ARTIFACTS.val_metrics),
            **flatten_metrics_row("holdout_", PRODUCTION_ARTIFACTS.test_metrics),
        }
    ]
)
PRODUCTION_HOLDOUT_DF.to_csv(OPERATOR_DIR / f"{OPERATOR_NAME}_production_holdout_summary.csv", index=False)

FINAL_SUMMARY_DF = pd.DataFrame(
    [
        {
            "operator": OPERATOR_NAME,
            "graph_operator": OPERATOR_NAME,
            "cv_mean_test_selection_score": float(CV_MEAN_DF.iloc[0]["cv_mean_test_selection_score"]),
            "cv_mean_test_scaled_soft_utility": float(CV_MEAN_DF.iloc[0]["cv_mean_test_scaled_soft_utility"]),
            "cv_mean_test_dir_auc": float(CV_MEAN_DF.iloc[0]["cv_mean_test_dir_auc"]),
            "cv_mean_test_trade_auc": float(CV_MEAN_DF.iloc[0]["cv_mean_test_trade_auc"]),
            "cv_mean_test_pnl_sum": float(CV_MEAN_DF.iloc[0]["cv_mean_test_pnl_sum"]),
            "cv_mean_test_pnl_per_trade": float(CV_MEAN_DF.iloc[0]["cv_mean_test_pnl_per_trade"]),
            "production_best_epoch": int(PRODUCTION_ARTIFACTS.best_epoch),
            "production_selected_thr_trade": float(PRODUCTION_ARTIFACTS.selected_threshold_pair["thr_trade"]),
            "production_selected_thr_dir": float(PRODUCTION_ARTIFACTS.selected_threshold_pair["thr_dir"]),
            "production_holdout_selection_score": float(PRODUCTION_ARTIFACTS.test_metrics["selection_score"]),
            "production_holdout_scaled_soft_utility": float(PRODUCTION_ARTIFACTS.test_metrics["scaled_soft_utility"]),
            "production_holdout_dir_auc": float(PRODUCTION_ARTIFACTS.test_metrics["dir_auc"]),
            "production_holdout_trade_auc": float(PRODUCTION_ARTIFACTS.test_metrics["trade_auc"]),
            "production_holdout_pnl_sum": float(PRODUCTION_ARTIFACTS.test_metrics["pnl_sum"]),
            "production_holdout_pnl_per_trade": float(PRODUCTION_ARTIFACTS.test_metrics["pnl_per_trade"]),
            "production_holdout_trade_rate": float(PRODUCTION_ARTIFACTS.test_metrics["trade_rate"]),
            "production_holdout_win_rate": float(PRODUCTION_ARTIFACTS.test_metrics["win_rate"]),
            "production_holdout_directional_hit_rate": float(PRODUCTION_ARTIFACTS.test_metrics["directional_hit_rate"]),
            "production_holdout_rmse": float(PRODUCTION_ARTIFACTS.test_metrics["rmse"]),
            "production_holdout_mae": float(PRODUCTION_ARTIFACTS.test_metrics["mae"]),
            "production_holdout_ic": float(PRODUCTION_ARTIFACTS.test_metrics["ic"]),
        }
    ]
)
FINAL_SUMMARY_DF.to_csv(ARTIFACT_ROOT / "final_summary.csv", index=False)

print("\n" + "=" * 110)
print("PRODUCTION_HOLDOUT_DF")
print(PRODUCTION_HOLDOUT_DF)

print("\n" + "=" * 110)
print("FINAL_SUMMARY_DF")
print(FINAL_SUMMARY_DF)


# %%
print("Notebook build complete.")
