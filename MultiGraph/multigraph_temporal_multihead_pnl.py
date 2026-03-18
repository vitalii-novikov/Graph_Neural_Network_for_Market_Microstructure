# %%
import copy
import json
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
    "artifact_root": "./artifact_root_multigraph_temporal_edgehistory_multihead_pnl",
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
    "num_train_folds": 4,
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

    # Multitask loss weights
    "loss_w_trade": 0.65,
    "loss_w_dir": 0.80,
    "loss_w_ret": 0.25,
    "loss_w_utility": 0.55,
    "utility_tanh_k": 1.75,

    # Dynamic adjacency regularization
    "adj_l1_lambda": 1e-4,
    "adj_prior_lambda": 2e-4,

    # Trade evaluation settings
    "cost_bps_per_side": 1.0,
    "trade_label_buffer_bps": 0.0,
    "thr_trade_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "thr_dir_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "min_validation_trades": 20,
    "min_validation_coverage": 0.0,

    # Optional operator ablation
    "operator_candidates": ["dynamic_edge_mpnn", "dynamic_rel_conv"],
    "run_full_operator_ablation": False,
    "ablation_fast_mode": True,
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
EDGE_SRC_IDX = EDGE_INDEX[:, 0]
EDGE_DST_IDX = EDGE_INDEX[:, 1]

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
TARGET_LR_1BAR = df[f"lr_{TARGET_ASSET}"].to_numpy(dtype=np.float64)

Y_RET = forward_log_return_from_mid(TARGET_MID, horizon_bars=HORIZON_BARS)

TRADE_LABEL_ABS_RETURN_THRESHOLD = (
    round_trip_cost_as_log_return(float(CFG["cost_bps_per_side"]))
    + float(CFG["trade_label_buffer_bps"]) * 1e-4
)

Y_DIR = np.full(len(Y_RET), np.nan, dtype=np.float32)
Y_DIR[np.isfinite(Y_RET) & (Y_RET > 0.0)] = 1.0
Y_DIR[np.isfinite(Y_RET) & (Y_RET < 0.0)] = 0.0
Y_DIR[np.isfinite(Y_RET) & (Y_RET == 0.0)] = 0.5

Y_TRADE = np.full(len(Y_RET), np.nan, dtype=np.float32)
valid_trade_mask = np.isfinite(Y_RET)
Y_TRADE[valid_trade_mask] = (
    np.abs(Y_RET[valid_trade_mask]) > TRADE_LABEL_ABS_RETURN_THRESHOLD
).astype(np.float32)

print("Finite target count:", int(np.isfinite(Y_RET).sum()))
print("Trade-label absolute-return threshold:", TRADE_LABEL_ABS_RETURN_THRESHOLD)

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


def make_exact_walk_forward_splits(
    idx_preholdout: np.ndarray,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    gap_bars: int,
    num_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_pre = len(idx_preholdout)
    num_folds = int(num_folds)
    train_min = max(1, int(round(n_pre * float(train_min_frac))))
    val_n = max(1, int(round(n_pre * float(val_window_frac))))
    test_n = max(1, int(round(n_pre * float(test_window_frac))))

    max_train_end = n_pre - (2 * gap_bars) - val_n - test_n
    if max_train_end <= train_min:
        raise RuntimeError(
            "Not enough pre-holdout data to create the requested number of exact walk-forward folds."
        )

    train_ends = np.linspace(train_min, max_train_end, num=num_folds)
    train_ends = np.round(train_ends).astype(np.int64)

    if len(np.unique(train_ends)) != num_folds:
        raise RuntimeError(
            "Exact fold construction produced duplicate train end points. "
            "Reduce num_train_folds or adjust split fractions."
        )

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, train_end in enumerate(train_ends, start=1):
        val_start = int(train_end) + gap_bars
        val_end = val_start + val_n
        test_start = val_end + gap_bars
        test_end = test_start + test_n

        if test_end > n_pre:
            raise RuntimeError(f"Fold {fold_idx} exceeds the pre-holdout boundary.")

        idx_train = idx_preholdout[: int(train_end)].copy()
        idx_val = idx_preholdout[val_start:val_end].copy()
        idx_test = idx_preholdout[test_start:test_end].copy()

        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            raise RuntimeError(f"Fold {fold_idx} produced an empty split.")

        splits.append((idx_train, idx_val, idx_test))

    if len(splits) != num_folds:
        raise RuntimeError(f"Expected exactly {num_folds} walk-forward folds, got {len(splits)}.")
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
        raise RuntimeError("Not enough pre-holdout samples for final production split.")

    idx_train_final = idx_preholdout[:train_end].copy()
    idx_val_final = idx_preholdout[train_end + gap_bars: train_end + gap_bars + val_n].copy()
    idx_test_final = idx_holdout.copy()

    if len(idx_val_final) != val_n:
        raise RuntimeError("Final validation window was not created correctly.")
    return idx_train_final, idx_val_final, idx_test_final


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

    if len(cv_splits) != int(CFG["num_train_folds"]):
        raise AssertionError(
            f"Expected exactly {int(CFG['num_train_folds'])} CV folds, got {len(cv_splits)}"
        )

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(cv_splits, start=1):
        assert_time_order_and_purge(idx_train, idx_val, idx_test, gap_bars, label=f"cv_fold_{fold_idx}")

        if not np.all(np.isin(idx_train, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: train contains non-preholdout indices")
        if not np.all(np.isin(idx_val, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: val contains non-preholdout indices")
        if not np.all(np.isin(idx_test, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: test contains non-preholdout indices")
        if len(np.intersect1d(idx_test, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV test")
        if len(np.intersect1d(idx_train, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV train")
        if len(np.intersect1d(idx_val, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV val")

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
        raise AssertionError("Final production test must equal the final holdout exactly")


IDX_PREHOLDOUT, IDX_HOLDOUT = make_preholdout_and_holdout_split(
    n_samples=N_SAMPLES,
    holdout_frac=float(CFG["final_holdout_frac"]),
    gap_bars=PURGE_GAP_BARS,
)

WALK_FORWARD_SPLITS = make_exact_walk_forward_splits(
    idx_preholdout=IDX_PREHOLDOUT,
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    gap_bars=PURGE_GAP_BARS,
    num_folds=int(CFG["num_train_folds"]),
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
    print(f"Fold {i}: train={len(tr)} val={len(va)} test={len(te)} source=preholdout_test")
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
        y_dir = float(self.y_dir[raw_t])

        if not np.isfinite(y_ret):
            raise RuntimeError(f"Encountered invalid return target at raw_t={raw_t}")
        if not np.isfinite(y_trade):
            raise RuntimeError(f"Encountered invalid trade target at raw_t={raw_t}")
        if not np.isfinite(y_dir):
            raise RuntimeError(f"Encountered invalid direction target at raw_t={raw_t}")

        return (
            torch.from_numpy(x_node_seq),
            torch.from_numpy(x_edge_seq),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_trade, dtype=torch.float32),
            torch.tensor(y_dir, dtype=torch.float32),
            torch.tensor(sample_idx, dtype=torch.long),
            torch.tensor(raw_t, dtype=torch.long),
        )


def temporal_multigraph_collate(batch):
    x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, sample_idx, raw_t = zip(*batch)
    return (
        torch.stack(x_node_seq, dim=0),
        torch.stack(x_edge_seq, dim=0),
        torch.stack(y_ret, dim=0),
        torch.stack(y_trade, dim=0),
        torch.stack(y_dir, dim=0),
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
    if last_train_t > int(sample_t[int(train_sample_indices[-1])]):
        raise AssertionError("Scaler fit window exceeded train boundary")
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
        out[:, mask] = torch.softmax(logits[:, mask], dim=1)
    return out


def build_incoming_uniform_prior(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    indeg = torch.zeros(n_nodes, dtype=torch.float32)
    for dst in edge_index[:, 1].tolist():
        indeg[int(dst)] += 1.0

    prior = torch.zeros(edge_index.size(0), dtype=torch.float32)
    for e, dst in enumerate(edge_index[:, 1].tolist()):
        prior[e] = 1.0 / max(float(indeg[int(dst)]), 1.0)
    return prior


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
        self.register_buffer("adj_prior", build_incoming_uniform_prior(edge_index, n_nodes))

        indeg = torch.zeros(n_nodes, dtype=torch.float32)
        for dst in edge_index[:, 1].tolist():
            indeg[int(dst)] += 1.0
        self.register_buffer("indeg", indeg.clamp_min(1.0))

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

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor, return_aux: bool = False):
        h_src = node_state[:, self.src_idx, :]
        h_dst = node_state[:, self.dst_idx, :]

        gate_input = torch.cat([h_src, h_dst, edge_state], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        msg = gate * self.src_proj(h_src) + self.edge_proj(edge_state)

        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)
        agg = agg / self.indeg.view(1, -1, 1)

        update = self.self_proj(node_state) + self.agg_proj(agg)
        out = self.norm(node_state + self.dropout(F.gelu(update)))

        if not return_aux:
            return out

        adj_weight = gate.mean(dim=-1)
        prior = self.adj_prior.view(1, -1).expand_as(adj_weight)
        aux = {
            "adj_l1": adj_weight.mean(),
            "adj_prior": F.mse_loss(adj_weight, prior),
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
        self.register_buffer("adj_prior", build_incoming_uniform_prior(edge_index, n_nodes))

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

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor, return_aux: bool = False):
        h_src = node_state[:, self.src_idx, :]
        logits = self.edge_score_net(edge_state).squeeze(-1)
        alpha = edge_softmax_by_dst(logits, self.dst_idx, self.n_nodes)
        msg = alpha.unsqueeze(-1) * (self.src_proj(h_src) + self.edge_shift(edge_state))
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)

        update = self.self_proj(node_state) + self.agg_proj(agg)
        out = self.norm(node_state + self.dropout(F.gelu(update)))

        if not return_aux:
            return out

        reg_adj = torch.sigmoid(logits)
        prior = self.adj_prior.view(1, -1).expand_as(reg_adj)
        aux = {
            "adj_l1": reg_adj.mean(),
            "adj_prior": F.mse_loss(reg_adj, prior),
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

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor, collect_regularization: bool = False):
        bsz, seq_len, n_nodes, node_dim = node_seq.shape
        _, _, n_edges, edge_dim = edge_seq.shape

        flat_nodes = node_seq.reshape(bsz * seq_len, n_nodes, node_dim)
        flat_edges = edge_seq.reshape(bsz * seq_len, n_edges, edge_dim)

        out = flat_nodes
        adj_l1_terms: List[torch.Tensor] = []
        adj_prior_terms: List[torch.Tensor] = []

        for layer in self.layers:
            if collect_regularization:
                out, aux = layer(out, flat_edges, return_aux=True)
                adj_l1_terms.append(aux["adj_l1"])
                adj_prior_terms.append(aux["adj_prior"])
            else:
                out = layer(out, flat_edges, return_aux=False)

        out = out.reshape(bsz, seq_len, n_nodes, node_dim)

        if not collect_regularization:
            return out

        reg = {
            "adj_l1": torch.stack(adj_l1_terms).mean() if adj_l1_terms else out.new_zeros(()),
            "adj_prior": torch.stack(adj_prior_terms).mean() if adj_prior_terms else out.new_zeros(()),
        }
        return out, reg


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


class TargetTemporalTrunk(nn.Module):
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


def make_prediction_head(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


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
        self.target_trunk = TargetTemporalTrunk(
            hidden_dim=target_hidden_dim,
            num_layers=int(cfg["target_temporal_layers"]),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.trade_head = make_prediction_head(target_hidden_dim, dropout)
        self.dir_head = make_prediction_head(target_hidden_dim, dropout)
        self.fixed_head = make_prediction_head(target_hidden_dim, dropout)

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
        adj_l1_terms: List[torch.Tensor] = []
        adj_prior_terms: List[torch.Tensor] = []

        for r, rel in enumerate(self.relation_names):
            rel_node_seq, rel_reg = self.relation_blocks[rel](
                node_seq,
                edge_seq[:, :, r, :, :],
                collect_regularization=True,
            )
            relation_outputs.append(rel_node_seq)
            adj_l1_terms.append(rel_reg["adj_l1"])
            adj_prior_terms.append(rel_reg["adj_prior"])

        relation_stack = torch.stack(relation_outputs, dim=2)
        fused_node_seq, relation_weights = self.fusion(relation_stack)

        target_seq = fused_node_seq[:, :, self.target_node, :]
        target_seq = self.target_proj(target_seq)
        shared_state = self.target_trunk(target_seq)

        trade_logit = self.trade_head(shared_state).squeeze(-1)
        dir_logit = self.dir_head(shared_state).squeeze(-1)
        fixed_pred = self.fixed_head(shared_state).squeeze(-1)

        outputs = {
            "trade_logit": torch.nan_to_num(trade_logit, nan=0.0, posinf=0.0, neginf=0.0),
            "dir_logit": torch.nan_to_num(dir_logit, nan=0.0, posinf=0.0, neginf=0.0),
            "fixed_pred": torch.nan_to_num(fixed_pred, nan=0.0, posinf=0.0, neginf=0.0),
            "adj_l1": torch.stack(adj_l1_terms).mean() if adj_l1_terms else target_seq.new_zeros(()),
            "adj_prior": torch.stack(adj_prior_terms).mean() if adj_prior_terms else target_seq.new_zeros(()),
        }

        if return_aux:
            outputs["relation_weights"] = relation_weights
            outputs["relation_node_seq"] = relation_stack
            outputs["fused_node_seq"] = fused_node_seq

        return outputs

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


def finite_or_default(value: Any, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_soft_position_torch(trade_logit: torch.Tensor, dir_logit: torch.Tensor, k: float) -> torch.Tensor:
    return torch.sigmoid(trade_logit) * torch.tanh(float(k) * dir_logit)


def compute_soft_utility_numpy(trade_logit: np.ndarray, dir_logit: np.ndarray, returns: np.ndarray, k: float) -> np.ndarray:
    p_trade = sigmoid_np(trade_logit)
    soft_dir = np.tanh(float(k) * np.asarray(dir_logit, dtype=np.float64))
    ret = np.asarray(returns, dtype=np.float64)
    return p_trade * soft_dir * ret


@dataclass
class LossState:
    pos_weight_trade: float
    pos_weight_dir: float


def compute_positive_class_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    labels = labels[np.isfinite(labels)]
    if len(labels) == 0:
        return 1.0
    pos = float((labels > 0.5).sum())
    neg = float(len(labels) - pos)
    if pos <= 0.0:
        return 1.0
    return float(min(10.0, max(1.0, neg / pos)))


def build_loss_state(idx_train: np.ndarray) -> LossState:
    raw_t_train = SAMPLE_T[idx_train]
    trade_train = Y_TRADE[raw_t_train].astype(np.float64)
    dir_train = Y_DIR[raw_t_train][trade_train > 0.5].astype(np.float64)

    return LossState(
        pos_weight_trade=compute_positive_class_weight(trade_train),
        pos_weight_dir=compute_positive_class_weight(dir_train) if len(dir_train) > 0 else 1.0,
    )


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    y_ret: torch.Tensor,
    y_trade: torch.Tensor,
    y_dir: torch.Tensor,
    loss_state: LossState,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    trade_logit = outputs["trade_logit"].view(-1)
    dir_logit = outputs["dir_logit"].view(-1)
    fixed_pred = outputs["fixed_pred"].view(-1)

    y_ret = y_ret.view(-1).float()
    y_trade = y_trade.view(-1).float()
    y_dir = y_dir.view(-1).float()

    trade_pos_weight = torch.tensor(loss_state.pos_weight_trade, dtype=torch.float32, device=trade_logit.device)
    dir_pos_weight = torch.tensor(loss_state.pos_weight_dir, dtype=torch.float32, device=trade_logit.device)

    trade_loss = F.binary_cross_entropy_with_logits(
        trade_logit,
        y_trade,
        pos_weight=trade_pos_weight,
    )

    trade_mask = y_trade > 0.5
    if trade_mask.any():
        dir_loss = F.binary_cross_entropy_with_logits(
            dir_logit[trade_mask],
            y_dir[trade_mask],
            pos_weight=dir_pos_weight,
        )
    else:
        dir_loss = trade_logit.new_zeros(())

    ret_loss = F.smooth_l1_loss(
        fixed_pred,
        y_ret,
        beta=float(cfg["huber_beta"]),
    )

    soft_position = compute_soft_position_torch(
        trade_logit=trade_logit,
        dir_logit=dir_logit,
        k=float(cfg["utility_tanh_k"]),
    )
    soft_utility = soft_position * y_ret
    utility_loss = -soft_utility.mean()

    adj_reg = (
        float(cfg["adj_l1_lambda"]) * outputs["adj_l1"]
        + float(cfg["adj_prior_lambda"]) * outputs["adj_prior"]
    )

    total_loss = (
        float(cfg["loss_w_trade"]) * trade_loss
        + float(cfg["loss_w_dir"]) * dir_loss
        + float(cfg["loss_w_ret"]) * ret_loss
        + float(cfg["loss_w_utility"]) * utility_loss
        + adj_reg
    )

    return {
        "total_loss": total_loss,
        "trade_loss": trade_loss.detach(),
        "dir_loss": dir_loss.detach(),
        "ret_loss": ret_loss.detach(),
        "utility_loss": utility_loss.detach(),
        "adj_reg": adj_reg.detach(),
        "soft_utility_mean": soft_utility.mean().detach(),
    }

# %%
def apply_threshold_pair(
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    thr_trade: float,
    thr_dir: float,
) -> Tuple[np.ndarray, np.ndarray]:
    trade_prob = np.asarray(trade_prob, dtype=np.float64)
    dir_prob = np.asarray(dir_prob, dtype=np.float64)

    long_mask = (trade_prob >= float(thr_trade)) & (dir_prob > float(thr_dir))
    short_mask = (trade_prob >= float(thr_trade)) & (dir_prob < (1.0 - float(thr_dir)))
    return long_mask.astype(bool), short_mask.astype(bool)


def sequential_fixed_horizon_backtest_from_masks(
    y_true: np.ndarray,
    raw_t_indices: np.ndarray,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    horizon_bars: int,
    cost_bps_per_side: float,
    timestamps: Optional[pd.Series] = None,
    build_trades: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.float64)
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)
    long_mask = np.asarray(long_mask, dtype=bool)
    short_mask = np.asarray(short_mask, dtype=bool)

    n = len(y_true)
    round_trip_cost = round_trip_cost_as_log_return(cost_bps_per_side)

    rows: List[Dict[str, Any]] = []
    pnl_list: List[float] = []
    gross_list: List[float] = []
    side_list: List[int] = []
    win_list: List[int] = []
    correct_list: List[int] = []

    i = 0
    while i < n:
        go_long = bool(long_mask[i])
        go_short = bool(short_mask[i])

        if go_long and go_short:
            go_long = False
            go_short = False

        if go_long:
            side = 1
        elif go_short:
            side = -1
        else:
            i += 1
            continue

        realized_return = float(y_true[i])
        gross_pnl = float(side * realized_return)
        net_pnl = float(gross_pnl - round_trip_cost)

        pnl_list.append(net_pnl)
        gross_list.append(gross_pnl)
        side_list.append(side)
        win_list.append(int(net_pnl > 0.0))
        correct_list.append(int(side * realized_return > 0.0))

        if build_trades:
            entry_raw_t = int(raw_t_indices[i])
            exit_raw_t = int(entry_raw_t + horizon_bars)
            entry_ts = pd.Timestamp(timestamps.iloc[entry_raw_t]) if timestamps is not None else pd.NaT
            exit_ts = pd.Timestamp(timestamps.iloc[exit_raw_t]) if (timestamps is not None and exit_raw_t < len(timestamps)) else pd.NaT
            rows.append(
                {
                    "entry_local_idx": i,
                    "entry_raw_t": entry_raw_t,
                    "exit_raw_t": exit_raw_t,
                    "entry_timestamp": entry_ts,
                    "exit_timestamp": exit_ts,
                    "side": side,
                    "future_return": realized_return,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                }
            )

        i += int(horizon_bars)

    n_trades = len(pnl_list)
    pnl_sum = float(np.sum(pnl_list)) if n_trades else 0.0
    gross_pnl_sum = float(np.sum(gross_list)) if n_trades else 0.0
    pnl_per_trade = float(pnl_sum / n_trades) if n_trades else float("nan")
    sign_accuracy = float(np.mean(correct_list)) if n_trades else float("nan")
    win_rate = float(np.mean(win_list)) if n_trades else float("nan")
    long_trades = int(sum(1 for side in side_list if side == 1))
    short_trades = int(sum(1 for side in side_list if side == -1))
    long_pnl_sum = float(np.sum([p for p, side in zip(pnl_list, side_list) if side == 1])) if n_trades else 0.0
    short_pnl_sum = float(np.sum([p for p, side in zip(pnl_list, side_list) if side == -1])) if n_trades else 0.0
    trade_rate = float(n_trades / n) if n > 0 else float("nan")

    if n_trades >= 2 and np.std(pnl_list, ddof=1) > 1e-12:
        sharpe_like = float(np.mean(pnl_list) / np.std(pnl_list, ddof=1) * np.sqrt(n_trades))
    else:
        sharpe_like = float("nan")

    metrics = {
        "gross_pnl_sum": gross_pnl_sum,
        "pnl_sum": pnl_sum,
        "pnl_per_trade": pnl_per_trade,
        "n_trades": int(n_trades),
        "trade_rate": trade_rate,
        "sign_accuracy": sign_accuracy,
        "win_rate": win_rate,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pnl_sum": long_pnl_sum,
        "short_pnl_sum": short_pnl_sum,
        "sharpe_like": sharpe_like,
    }

    trades_df = pd.DataFrame(rows)
    return metrics, trades_df


def search_best_threshold_pair(
    y_ret: np.ndarray,
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    raw_t_indices: np.ndarray,
    cfg: Dict[str, Any],
    timestamps: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_pair: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, ...]] = None

    for thr_trade in cfg["thr_trade_grid"]:
        for thr_dir in cfg["thr_dir_grid"]:
            long_mask, short_mask = apply_threshold_pair(
                trade_prob=trade_prob,
                dir_prob=dir_prob,
                thr_trade=float(thr_trade),
                thr_dir=float(thr_dir),
            )
            active_mask = long_mask | short_mask
            coverage = float(active_mask.mean()) if len(active_mask) else float("nan")

            bt_metrics, _ = sequential_fixed_horizon_backtest_from_masks(
                y_true=y_ret,
                raw_t_indices=raw_t_indices,
                long_mask=long_mask,
                short_mask=short_mask,
                horizon_bars=HORIZON_BARS,
                cost_bps_per_side=float(cfg["cost_bps_per_side"]),
                timestamps=None,
                build_trades=False,
            )

            feasible = (
                bt_metrics["n_trades"] >= int(cfg["min_validation_trades"])
                and finite_or_default(coverage, 0.0) >= float(cfg["min_validation_coverage"])
            )

            row = {
                "thr_trade": float(thr_trade),
                "thr_dir": float(thr_dir),
                "coverage": coverage,
                "feasible": bool(feasible),
                **bt_metrics,
            }
            rows.append(row)

            key = (
                1.0 if feasible else 0.0,
                finite_or_default(bt_metrics["pnl_sum"], -1e9),
                finite_or_default(bt_metrics["pnl_per_trade"], -1e9),
                finite_or_default(bt_metrics["sign_accuracy"], -1e9),
                float(bt_metrics["n_trades"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = {
                    "thr_trade": float(thr_trade),
                    "thr_dir": float(thr_dir),
                    "coverage": coverage,
                    "feasible": bool(feasible),
                }
                best_metrics = copy.deepcopy(bt_metrics)

    if best_pair is None or best_metrics is None:
        raise RuntimeError("Threshold-pair search failed to identify a valid validation pair.")

    best_long_mask, best_short_mask = apply_threshold_pair(
        trade_prob=trade_prob,
        dir_prob=dir_prob,
        thr_trade=float(best_pair["thr_trade"]),
        thr_dir=float(best_pair["thr_dir"]),
    )
    best_metrics, best_trades_df = sequential_fixed_horizon_backtest_from_masks(
        y_true=y_ret,
        raw_t_indices=raw_t_indices,
        long_mask=best_long_mask,
        short_mask=best_short_mask,
        horizon_bars=HORIZON_BARS,
        cost_bps_per_side=float(cfg["cost_bps_per_side"]),
        timestamps=timestamps,
        build_trades=True,
    )
    best_metrics["coverage"] = float((best_long_mask | best_short_mask).mean()) if len(best_long_mask) else float("nan")

    grid_df = pd.DataFrame(rows).sort_values(
        by=["feasible", "pnl_sum", "pnl_per_trade", "sign_accuracy", "n_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return best_pair, grid_df, best_metrics, best_trades_df


def evaluate_prediction_pack(
    pred_pack: Dict[str, Any],
    cfg: Dict[str, Any],
    selected_threshold_pair: Optional[Dict[str, Any]] = None,
    search_threshold_pair_on_pack: bool = False,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Dict[str, Any]]:
    y_ret = np.asarray(pred_pack["y_ret"], dtype=np.float64)
    y_trade = np.asarray(pred_pack["y_trade"], dtype=np.float64)
    y_dir = np.asarray(pred_pack["y_dir"], dtype=np.float64)
    fixed_pred = np.asarray(pred_pack["fixed_pred"], dtype=np.float64)
    trade_logit = np.asarray(pred_pack["trade_logit"], dtype=np.float64)
    dir_logit = np.asarray(pred_pack["dir_logit"], dtype=np.float64)
    trade_prob = np.asarray(pred_pack["trade_prob"], dtype=np.float64)
    dir_prob = np.asarray(pred_pack["dir_prob"], dtype=np.float64)
    raw_t = np.asarray(pred_pack["raw_t"], dtype=np.int64)

    trade_auc = safe_roc_auc(y_trade, trade_prob)
    dir_mask = y_trade > 0.5
    dir_auc = safe_roc_auc(y_dir[dir_mask], dir_prob[dir_mask]) if dir_mask.any() else float("nan")

    soft_utility_vec = compute_soft_utility_numpy(
        trade_logit=trade_logit,
        dir_logit=dir_logit,
        returns=y_ret,
        k=float(cfg["utility_tanh_k"]),
    )
    soft_utility_mean = float(np.mean(soft_utility_vec)) if len(soft_utility_vec) else float("nan")
    denom = float(np.mean(np.abs(y_ret)) + 1e-12) if len(y_ret) else 1.0
    scaled_soft_utility = float(soft_utility_mean / denom) if np.isfinite(soft_utility_mean) else float("nan")
    selection_score = float(
        finite_or_default(scaled_soft_utility, 0.0) + 0.55 * finite_or_default(dir_auc, 0.5)
    )

    threshold_grid_df: Optional[pd.DataFrame] = None
    if search_threshold_pair_on_pack or selected_threshold_pair is None:
        selected_threshold_pair, threshold_grid_df, threshold_metrics, trades_df = search_best_threshold_pair(
            y_ret=y_ret,
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            raw_t_indices=raw_t,
            cfg=cfg,
            timestamps=TIMESTAMPS,
        )
    else:
        long_mask, short_mask = apply_threshold_pair(
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            thr_trade=float(selected_threshold_pair["thr_trade"]),
            thr_dir=float(selected_threshold_pair["thr_dir"]),
        )
        threshold_metrics, trades_df = sequential_fixed_horizon_backtest_from_masks(
            y_true=y_ret,
            raw_t_indices=raw_t,
            long_mask=long_mask,
            short_mask=short_mask,
            horizon_bars=HORIZON_BARS,
            cost_bps_per_side=float(cfg["cost_bps_per_side"]),
            timestamps=TIMESTAMPS,
            build_trades=True,
        )
        threshold_metrics["coverage"] = float((long_mask | short_mask).mean()) if len(long_mask) else float("nan")

    metrics = {
        "rmse": rmse_np(y_ret, fixed_pred),
        "mae": mae_np(y_ret, fixed_pred),
        "ic": ic_np(y_ret, fixed_pred),
        "trade_auc": trade_auc,
        "dir_auc": dir_auc,
        "soft_utility_mean": soft_utility_mean,
        "scaled_soft_utility": scaled_soft_utility,
        "selection_score": selection_score,
        **threshold_metrics,
        "trades_df": trades_df,
        "selected_threshold_pair": copy.deepcopy(selected_threshold_pair),
    }
    return metrics, threshold_grid_df, selected_threshold_pair


def checkpoint_key_from_metrics(metrics: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        finite_or_default(metrics.get("selection_score"), -1e9),
        finite_or_default(metrics.get("scaled_soft_utility"), -1e9),
        finite_or_default(metrics.get("dir_auc"), -1e9),
        finite_or_default(metrics.get("pnl_sum"), -1e9),
        finite_or_default(metrics.get("pnl_per_trade"), -1e9),
    )


def better_selection_key(candidate: Tuple[float, ...], incumbent: Optional[Tuple[float, ...]]) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent

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

    model.eval()
    trade_logits, dir_logits, fixed_preds = [], [], []
    y_ret_all, y_trade_all, y_dir_all = [], [], []
    sample_idx_all, raw_t_all = [], []

    for x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, sample_idx, raw_t in loader:
        x_node_seq = x_node_seq.to(DEVICE).float()
        x_edge_seq = x_edge_seq.to(DEVICE).float()

        outputs = model(x_node_seq, x_edge_seq, return_aux=False)
        trade_logits.append(outputs["trade_logit"].detach().cpu().numpy())
        dir_logits.append(outputs["dir_logit"].detach().cpu().numpy())
        fixed_preds.append(outputs["fixed_pred"].detach().cpu().numpy())

        y_ret_all.append(y_ret.detach().cpu().numpy())
        y_trade_all.append(y_trade.detach().cpu().numpy())
        y_dir_all.append(y_dir.detach().cpu().numpy())
        sample_idx_all.append(sample_idx.detach().cpu().numpy())
        raw_t_all.append(raw_t.detach().cpu().numpy())

    trade_logit_arr = np.concatenate(trade_logits, axis=0).astype(np.float64)
    dir_logit_arr = np.concatenate(dir_logits, axis=0).astype(np.float64)
    fixed_pred_arr = np.concatenate(fixed_preds, axis=0).astype(np.float64)
    y_ret_arr = np.concatenate(y_ret_all, axis=0).astype(np.float64)
    y_trade_arr = np.concatenate(y_trade_all, axis=0).astype(np.float64)
    y_dir_arr = np.concatenate(y_dir_all, axis=0).astype(np.float64)
    sample_idx_arr = np.concatenate(sample_idx_all, axis=0).astype(np.int64)
    raw_t_arr = np.concatenate(raw_t_all, axis=0).astype(np.int64)

    return {
        "trade_logit": trade_logit_arr,
        "dir_logit": dir_logit_arr,
        "fixed_pred": fixed_pred_arr,
        "trade_prob": sigmoid_np(trade_logit_arr),
        "dir_prob": sigmoid_np(dir_logit_arr),
        "y_ret": y_ret_arr,
        "y_trade": y_trade_arr,
        "y_dir": y_dir_arr,
        "sample_idx": sample_idx_arr,
        "raw_t": raw_t_arr,
        "timestamp": TIMESTAMPS.iloc[raw_t_arr].reset_index(drop=True),
    }

# %%
@dataclass
class SplitArtifacts:
    model_state: Dict[str, torch.Tensor]
    node_scaler_params: Dict[str, Any]
    relation_scaler_params: Dict[str, Dict[str, Any]]
    loss_state: Dict[str, float]
    best_epoch: int
    best_checkpoint_key: Tuple[float, ...]
    best_checkpoint_summary: Dict[str, Any]
    selected_threshold_pair: Dict[str, Any]
    validation_threshold_grid: pd.DataFrame
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

    loss_state = build_loss_state(idx_train)

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
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
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
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_epoch = -1
    best_checkpoint_key: Optional[Tuple[float, ...]] = None
    best_checkpoint_summary: Dict[str, Any] = {}
    bad_epochs = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_total_loss = []
        train_trade_loss = []
        train_dir_loss = []
        train_ret_loss = []
        train_utility_loss = []
        train_adj_reg = []
        train_soft_utility = []

        for x_node_seq, x_edge_seq, y_ret, y_trade, y_dir, _sample_idx, _raw_t in train_loader:
            x_node_seq = x_node_seq.to(DEVICE).float()
            x_edge_seq = x_edge_seq.to(DEVICE).float()
            y_ret = y_ret.to(DEVICE).float()
            y_trade = y_trade.to(DEVICE).float()
            y_dir = y_dir.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x_node_seq, x_edge_seq, return_aux=False)
            loss_pack = compute_multitask_loss(
                outputs=outputs,
                y_ret=y_ret,
                y_trade=y_trade,
                y_dir=y_dir,
                loss_state=loss_state,
                cfg=cfg,
            )
            loss_pack["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            train_total_loss.append(float(loss_pack["total_loss"].detach().cpu().item()))
            train_trade_loss.append(float(loss_pack["trade_loss"].cpu().item()))
            train_dir_loss.append(float(loss_pack["dir_loss"].cpu().item()))
            train_ret_loss.append(float(loss_pack["ret_loss"].cpu().item()))
            train_utility_loss.append(float(loss_pack["utility_loss"].cpu().item()))
            train_adj_reg.append(float(loss_pack["adj_reg"].cpu().item()))
            train_soft_utility.append(float(loss_pack["soft_utility_mean"].cpu().item()))

        val_pred_pack = predict_on_indices(
            model=model,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
            indices=idx_val,
            batch_size=int(cfg["batch_size"]),
        )
        val_metrics, _val_grid_df, val_threshold_pair = evaluate_prediction_pack(
            pred_pack=val_pred_pack,
            cfg=cfg,
            selected_threshold_pair=None,
            search_threshold_pair_on_pack=True,
        )

        checkpoint_key = checkpoint_key_from_metrics(val_metrics)
        scheduler.step(float(val_metrics["selection_score"]))

        print(
            f"[{split_name}][{cfg['graph_operator']}] "
            f"epoch={epoch:02d} "
            f"loss={np.mean(train_total_loss):.6f} "
            f"trade_bce={np.mean(train_trade_loss):.6f} "
            f"dir_bce={np.mean(train_dir_loss):.6f} "
            f"ret_huber={np.mean(train_ret_loss):.6f} "
            f"utility_loss={np.mean(train_utility_loss):.6f} "
            f"adj_reg={np.mean(train_adj_reg):.6f} "
            f"train_soft_util={np.mean(train_soft_utility):.6f} "
            f"val_selection={val_metrics['selection_score']:.6f} "
            f"val_soft_util={val_metrics['scaled_soft_utility']:.6f} "
            f"val_dir_auc={finite_or_default(val_metrics['dir_auc'], float('nan')):.4f} "
            f"val_trade_auc={finite_or_default(val_metrics['trade_auc'], float('nan')):.4f} "
            f"val_pnl_sum={finite_or_default(val_metrics['pnl_sum'], float('nan')):.6f} "
            f"val_ppt={finite_or_default(val_metrics['pnl_per_trade'], float('nan')):.6f} "
            f"val_trades={int(val_metrics['n_trades'])} "
            f"thr_trade={val_threshold_pair['thr_trade']:.2f} "
            f"thr_dir={val_threshold_pair['thr_dir']:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if better_selection_key(checkpoint_key, best_checkpoint_key):
            best_checkpoint_key = checkpoint_key
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint_summary = {
                "epoch": int(epoch),
                "checkpoint_key": list(checkpoint_key),
                "selected_threshold_pair": copy.deepcopy(val_threshold_pair),
                "val_metrics": {
                    k: v
                    for k, v in val_metrics.items()
                    if k not in {"trades_df", "selected_threshold_pair"}
                },
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            print(f"[{split_name}][{cfg['graph_operator']}] early stopping at epoch {epoch}")
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
    test_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_test,
        batch_size=int(cfg["batch_size"]),
    )

    val_metrics, validation_threshold_grid, selected_threshold_pair = evaluate_prediction_pack(
        pred_pack=val_pred_pack,
        cfg=cfg,
        selected_threshold_pair=None,
        search_threshold_pair_on_pack=True,
    )
    test_metrics, _, _ = evaluate_prediction_pack(
        pred_pack=test_pred_pack,
        cfg=cfg,
        selected_threshold_pair=selected_threshold_pair,
        search_threshold_pair_on_pack=False,
    )

    print(
        f"[{split_name}][{cfg['graph_operator']}] "
        f"best_epoch={best_epoch} "
        f"best_key={best_checkpoint_key}"
    )
    print(
        f"[{split_name}][{cfg['graph_operator']}] "
        f"selected_threshold_pair="
        f"(thr_trade={selected_threshold_pair['thr_trade']:.2f}, "
        f"thr_dir={selected_threshold_pair['thr_dir']:.2f})"
    )
    print(
        f"[{split_name}][{cfg['graph_operator']}] TEST "
        f"selection={test_metrics['selection_score']:.6f} "
        f"soft_util={test_metrics['scaled_soft_utility']:.6f} "
        f"dir_auc={finite_or_default(test_metrics['dir_auc'], float('nan')):.4f} "
        f"trade_auc={finite_or_default(test_metrics['trade_auc'], float('nan')):.4f} "
        f"rmse={test_metrics['rmse']:.6f} "
        f"ic={finite_or_default(test_metrics['ic'], float('nan')):.4f} "
        f"pnl_sum={finite_or_default(test_metrics['pnl_sum'], float('nan')):.6f} "
        f"pnl_per_trade={finite_or_default(test_metrics['pnl_per_trade'], float('nan')):.6f} "
        f"n_trades={int(test_metrics['n_trades'])}"
    )

    return SplitArtifacts(
        model_state=copy.deepcopy(model.state_dict()),
        node_scaler_params=node_scaler_params,
        relation_scaler_params=relation_scaler_params,
        loss_state={
            "pos_weight_trade": float(loss_state.pos_weight_trade),
            "pos_weight_dir": float(loss_state.pos_weight_dir),
        },
        best_epoch=best_epoch,
        best_checkpoint_key=best_checkpoint_key,
        best_checkpoint_summary=best_checkpoint_summary,
        selected_threshold_pair=copy.deepcopy(selected_threshold_pair),
        validation_threshold_grid=validation_threshold_grid.copy(),
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


def load_bundle(bundle_dir: Path, bundle_name: str) -> Dict[str, Any]:
    bundle_path = bundle_dir / f"{bundle_name}.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    try:
        loaded = torch.load(str(bundle_path), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(str(bundle_path), map_location="cpu")
    return loaded

# %%
def is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, str, bool)) or value is None


def flatten_metrics_row(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in metrics.items():
        if k in {"trades_df", "selected_threshold_pair"}:
            continue
        if is_scalar_metric(v):
            row[f"{prefix}{k}" if prefix else k] = v
    return row


def flatten_threshold_pair(prefix: str, pair: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in pair.items():
        if is_scalar_metric(v):
            row[f"{prefix}{k}" if prefix else k] = v
    return row


def run_cv_for_operator(
    operator_name: str,
    base_cfg: Dict[str, Any],
    is_ablation_context: bool = True,
) -> Dict[str, Any]:
    run_cfg = build_run_cfg(base_cfg=base_cfg, operator_name=operator_name, is_ablation_context=is_ablation_context)
    operator_dir = ARTIFACT_ROOT / operator_name
    operator_dir.mkdir(parents=True, exist_ok=True)

    cv_rows: List[Dict[str, Any]] = []
    best_cv_bundle_name: Optional[str] = None
    best_cv_key: Optional[Tuple[float, ...]] = None
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
                "best_checkpoint_key": list(artifacts.best_checkpoint_key),
                "best_checkpoint_summary": artifacts.best_checkpoint_summary,
                "loss_state": artifacts.loss_state,
                "selected_threshold_pair": artifacts.selected_threshold_pair,
                "idx_train": idx_train.tolist(),
                "idx_val": idx_val.tolist(),
                "idx_test": idx_test.tolist(),
            },
        )

        artifacts.validation_threshold_grid.to_csv(
            operator_dir / f"{bundle_name}_validation_threshold_grid.csv",
            index=False,
        )
        artifacts.val_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_val_trades.csv",
            index=False,
        )
        artifacts.test_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_test_trades.csv",
            index=False,
        )

        row = {
            "operator": operator_name,
            "fold": fold_idx,
            "best_epoch": artifacts.best_epoch,
            "best_checkpoint_key": json.dumps(list(artifacts.best_checkpoint_key)),
            **flatten_threshold_pair("selected_", artifacts.selected_threshold_pair),
            **flatten_metrics_row("val_", artifacts.val_metrics),
            **flatten_metrics_row("test_", artifacts.test_metrics),
        }
        cv_rows.append(row)

        fold_key = (
            finite_or_default(artifacts.test_metrics["pnl_sum"], -1e9),
            finite_or_default(artifacts.test_metrics["selection_score"], -1e9),
            finite_or_default(artifacts.test_metrics["dir_auc"], -1e9),
            finite_or_default(artifacts.test_metrics["pnl_per_trade"], -1e9),
        )
        if better_selection_key(fold_key, best_cv_key):
            best_cv_key = fold_key
            best_cv_bundle_name = bundle_name

    if best_cv_bundle_name is None or best_cv_key is None:
        raise RuntimeError(f"{operator_name}: no best CV bundle selected")

    cv_results_df = pd.DataFrame(cv_rows)
    cv_results_df.to_csv(operator_dir / f"{operator_name}_cv_results_summary.csv", index=False)

    cv_mean_numeric = cv_results_df.mean(numeric_only=True).to_dict()
    cv_mean_summary_row = {
        "operator": operator_name,
        "graph_operator": operator_name,
        "cv_mean_test_selection_score": float(cv_mean_numeric.get("test_selection_score", np.nan)),
        "cv_mean_test_scaled_soft_utility": float(cv_mean_numeric.get("test_scaled_soft_utility", np.nan)),
        "cv_mean_test_dir_auc": float(cv_mean_numeric.get("test_dir_auc", np.nan)),
        "cv_mean_test_trade_auc": float(cv_mean_numeric.get("test_trade_auc", np.nan)),
        "cv_mean_test_rmse": float(cv_mean_numeric.get("test_rmse", np.nan)),
        "cv_mean_test_ic": float(cv_mean_numeric.get("test_ic", np.nan)),
        "cv_mean_test_pnl_sum": float(cv_mean_numeric.get("test_pnl_sum", np.nan)),
        "cv_mean_test_pnl_per_trade": float(cv_mean_numeric.get("test_pnl_per_trade", np.nan)),
        "cv_mean_test_sharpe_like": float(cv_mean_numeric.get("test_sharpe_like", np.nan)),
    }
    cv_mean_df = pd.DataFrame([cv_mean_summary_row])
    cv_mean_df.to_csv(operator_dir / f"{operator_name}_cv_mean_summary.csv", index=False)

    print("\n" + "=" * 110)
    print(f"CV_RESULTS_DF [{operator_name}]")
    print(cv_results_df)
    print("\nCV mean metrics:")
    print(cv_mean_df)

    return {
        "operator_name": operator_name,
        "cfg": run_cfg,
        "artifact_dir": operator_dir,
        "fold_bundle_names": fold_bundle_names,
        "best_cv_bundle_name": best_cv_bundle_name,
        "best_cv_key": best_cv_key,
        "cv_results_df": cv_results_df,
        "cv_mean_df": cv_mean_df,
    }


def select_best_operator_from_cv_runs(operator_runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    best_operator_name: Optional[str] = None
    best_key: Optional[Tuple[float, ...]] = None

    for operator_name, run_out in operator_runs.items():
        row = run_out["cv_mean_df"].iloc[0].to_dict()
        key = (
            finite_or_default(row["cv_mean_test_pnl_sum"], -1e9),
            finite_or_default(row["cv_mean_test_selection_score"], -1e9),
            finite_or_default(row["cv_mean_test_dir_auc"], -1e9),
            finite_or_default(row["cv_mean_test_pnl_per_trade"], -1e9),
        )
        row["cv_operator_selection_key"] = json.dumps(list(key))
        rows.append(row)

        if better_selection_key(key, best_key):
            best_key = key
            best_operator_name = operator_name

    if best_operator_name is None:
        raise RuntimeError("No operator selected from CV runs")

    operator_comparison_df = pd.DataFrame(rows).sort_values(
        by=[
            "cv_mean_test_pnl_sum",
            "cv_mean_test_selection_score",
            "cv_mean_test_dir_auc",
            "cv_mean_test_pnl_per_trade",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    operator_comparison_df.to_csv(ARTIFACT_ROOT / "operator_cv_comparison_summary.csv", index=False)

    selected_operator_run = operator_runs[best_operator_name]
    print("\n" + "=" * 110)
    print("OPERATOR_CV_COMPARISON_DF")
    print(operator_comparison_df)
    print("\nSELECTED_OPERATOR_FROM_CV:", best_operator_name)
    print("SELECTED_OPERATOR_KEY:", best_key)

    return {
        "selected_operator_name": best_operator_name,
        "selected_operator_run": selected_operator_run,
        "selected_operator_key": best_key,
        "operator_comparison_df": operator_comparison_df,
    }

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

    selected_threshold_pair = copy.deepcopy(loaded["meta"]["selected_threshold_pair"])

    metrics, _, _ = evaluate_prediction_pack(
        pred_pack=pred_pack,
        cfg=cfg,
        selected_threshold_pair=selected_threshold_pair,
        search_threshold_pair_on_pack=False,
    )

    print("\n" + "=" * 110)
    print(label)
    print(f"bundle_name={bundle_name}")
    print(
        f"selected_threshold_pair=(thr_trade={selected_threshold_pair['thr_trade']:.2f}, "
        f"thr_dir={selected_threshold_pair['thr_dir']:.2f})"
    )
    print(
        f"selection={metrics['selection_score']:.6f} "
        f"soft_util={metrics['scaled_soft_utility']:.6f} "
        f"dir_auc={finite_or_default(metrics['dir_auc'], float('nan')):.4f} "
        f"trade_auc={finite_or_default(metrics['trade_auc'], float('nan')):.4f}"
    )
    print(
        f"rmse={metrics['rmse']:.6f} "
        f"mae={metrics['mae']:.6f} "
        f"ic={finite_or_default(metrics['ic'], float('nan')):.4f}"
    )
    print(
        f"pnl_sum={finite_or_default(metrics['pnl_sum'], float('nan')):.6f} "
        f"pnl_per_trade={finite_or_default(metrics['pnl_per_trade'], float('nan')):.6f} "
        f"n_trades={int(metrics['n_trades'])} "
        f"trade_rate={finite_or_default(metrics['trade_rate'], float('nan')):.4f} "
        f"sign_accuracy={finite_or_default(metrics['sign_accuracy'], float('nan')):.4f}"
    )

    return {
        "pred_pack": pred_pack,
        "metrics": metrics,
        "selected_threshold_pair": selected_threshold_pair,
    }


def run_selected_operator_post_cv_and_production(
    operator_run: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    operator_name = str(operator_run["operator_name"])
    operator_dir = Path(operator_run["artifact_dir"])
    run_cfg = copy.deepcopy(operator_run["cfg"])
    best_cv_bundle_name = str(operator_run["best_cv_bundle_name"])

    post_cv_holdout = evaluate_saved_bundle_on_indices(
        bundle_dir=operator_dir,
        bundle_name=best_cv_bundle_name,
        indices=IDX_HOLDOUT,
        label=f"POST-CV HOLDOUT EVALUATION [{operator_name}] USING SELECTED CV WINNER",
    )
    post_cv_holdout_df = pd.DataFrame(
        [
            {
                "operator": operator_name,
                "model_name": best_cv_bundle_name,
                **flatten_metrics_row("", post_cv_holdout["metrics"]),
                **flatten_threshold_pair("selected_", post_cv_holdout["selected_threshold_pair"]),
            }
        ]
    )
    post_cv_holdout_df.to_csv(operator_dir / f"{operator_name}_post_cv_holdout_summary.csv", index=False)
    post_cv_holdout["metrics"]["trades_df"].to_csv(
        operator_dir / f"{operator_name}_post_cv_holdout_trades.csv",
        index=False,
    )

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
            "best_checkpoint_key": list(production_artifacts.best_checkpoint_key),
            "best_checkpoint_summary": production_artifacts.best_checkpoint_summary,
            "loss_state": production_artifacts.loss_state,
            "selected_threshold_pair": production_artifacts.selected_threshold_pair,
            "idx_train": IDX_TRAIN_FINAL.tolist(),
            "idx_val": IDX_VAL_FINAL.tolist(),
            "idx_test": IDX_TEST_FINAL.tolist(),
        },
    )

    production_artifacts.validation_threshold_grid.to_csv(
        operator_dir / f"{production_bundle_name}_validation_threshold_grid.csv",
        index=False,
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
                **flatten_threshold_pair("selected_", production_holdout["selected_threshold_pair"]),
            }
        ]
    )
    production_holdout_df.to_csv(operator_dir / f"{operator_name}_production_holdout_summary.csv", index=False)

    summary_row = {
        "operator": operator_name,
        "graph_operator": operator_name,
        "selected_cv_bundle_name": best_cv_bundle_name,
        "production_bundle_name": production_bundle_name,
        "cv_mean_test_selection_score": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_selection_score"]),
        "cv_mean_test_dir_auc": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_dir_auc"]),
        "cv_mean_test_pnl_sum": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_pnl_sum"]),
        "post_cv_holdout_selection_score": float(post_cv_holdout_df.iloc[0]["selection_score"]),
        "post_cv_holdout_dir_auc": float(post_cv_holdout_df.iloc[0]["dir_auc"]),
        "post_cv_holdout_pnl_sum": float(post_cv_holdout_df.iloc[0]["pnl_sum"]),
        "production_holdout_selection_score": float(production_holdout_df.iloc[0]["selection_score"]),
        "production_holdout_dir_auc": float(production_holdout_df.iloc[0]["dir_auc"]),
        "production_holdout_trade_auc": float(production_holdout_df.iloc[0]["trade_auc"]),
        "production_holdout_rmse": float(production_holdout_df.iloc[0]["rmse"]),
        "production_holdout_ic": float(production_holdout_df.iloc[0]["ic"]),
        "production_holdout_pnl_sum": float(production_holdout_df.iloc[0]["pnl_sum"]),
        "production_holdout_pnl_per_trade": float(production_holdout_df.iloc[0]["pnl_per_trade"]),
        "production_thr_trade": float(production_holdout_df.iloc[0]["selected_thr_trade"]),
        "production_thr_dir": float(production_holdout_df.iloc[0]["selected_thr_dir"]),
    }
    selected_operator_summary_df = pd.DataFrame([summary_row])
    selected_operator_summary_df.to_csv(ARTIFACT_ROOT / "selected_operator_final_summary.csv", index=False)

    return {
        "operator_name": operator_name,
        "artifact_dir": operator_dir,
        "post_cv_holdout_df": post_cv_holdout_df,
        "production_holdout_df": production_holdout_df,
        "selected_operator_summary_df": selected_operator_summary_df,
    }

# %%
if bool(CFG["run_full_operator_ablation"]):
    OPERATOR_RUNS: Dict[str, Dict[str, Any]] = {}
    for operator_name in CFG["operator_candidates"]:
        print("\n" + "#" * 120)
        print(f"RUNNING CV FOR OPERATOR: {operator_name}")
        print("#" * 120)
        OPERATOR_RUNS[operator_name] = run_cv_for_operator(
            operator_name=operator_name,
            base_cfg=CFG,
            is_ablation_context=True,
        )

    OPERATOR_SELECTION = select_best_operator_from_cv_runs(OPERATOR_RUNS)
    SELECTED_OPERATOR_RUN = OPERATOR_SELECTION["selected_operator_run"]
    OPERATOR_CV_COMPARISON_DF = OPERATOR_SELECTION["operator_comparison_df"]

    FINAL_SELECTED_RUN = run_selected_operator_post_cv_and_production(
        operator_run=SELECTED_OPERATOR_RUN,
        cfg=CFG,
    )

    FINAL_OPERATOR_CV_SUMMARY_DF = pd.concat(
        [run_out["cv_mean_df"] for run_out in OPERATOR_RUNS.values()],
        axis=0,
        ignore_index=True,
    )
    FINAL_OPERATOR_CV_SUMMARY_DF.to_csv(ARTIFACT_ROOT / "all_operator_cv_mean_summary.csv", index=False)

    print("\n" + "=" * 110)
    print("FINAL_OPERATOR_CV_SUMMARY_DF")
    print(FINAL_OPERATOR_CV_SUMMARY_DF)

    print("\n" + "=" * 110)
    print("OPERATOR_CV_COMPARISON_DF")
    print(OPERATOR_CV_COMPARISON_DF)

    print("\n" + "=" * 110)
    print("SELECTED_OPERATOR_POST_CV_HOLDOUT_DF")
    print(FINAL_SELECTED_RUN["post_cv_holdout_df"])

    print("\n" + "=" * 110)
    print("SELECTED_OPERATOR_PRODUCTION_HOLDOUT_DF")
    print(FINAL_SELECTED_RUN["production_holdout_df"])

    print("\n" + "=" * 110)
    print("SELECTED_OPERATOR_FINAL_SUMMARY_DF")
    print(FINAL_SELECTED_RUN["selected_operator_summary_df"])

else:
    DEFAULT_OPERATOR_RESULTS = run_cv_for_operator(
        operator_name=str(CFG["graph_operator"]),
        base_cfg=CFG,
        is_ablation_context=False,
    )

    FINAL_SELECTED_RUN = run_selected_operator_post_cv_and_production(
        operator_run=DEFAULT_OPERATOR_RESULTS,
        cfg=CFG,
    )

    DEFAULT_CV_SUMMARY_DF = DEFAULT_OPERATOR_RESULTS["cv_mean_df"].copy()
    DEFAULT_POST_CV_HOLDOUT_DF = FINAL_SELECTED_RUN["post_cv_holdout_df"].copy()
    DEFAULT_PRODUCTION_HOLDOUT_DF = FINAL_SELECTED_RUN["production_holdout_df"].copy()
    DEFAULT_FINAL_SUMMARY_DF = FINAL_SELECTED_RUN["selected_operator_summary_df"].copy()

    print("\n" + "=" * 110)
    print("DEFAULT_CV_SUMMARY_DF")
    print(DEFAULT_CV_SUMMARY_DF)

    print("\n" + "=" * 110)
    print("DEFAULT_POST_CV_HOLDOUT_DF")
    print(DEFAULT_POST_CV_HOLDOUT_DF)

    print("\n" + "=" * 110)
    print("DEFAULT_PRODUCTION_HOLDOUT_DF")
    print(DEFAULT_PRODUCTION_HOLDOUT_DF)

    print("\n" + "=" * 110)
    print("DEFAULT_FINAL_SUMMARY_DF")
    print(DEFAULT_FINAL_SUMMARY_DF)

# %%
print("Notebook build complete.")
