
# %% [markdown]
# This script is a single-file percent-format version of the original notebook.
#
# Design choices in this rewrite:
#
# 1. The early pipeline structure is preserved: imports/config, graph definition,
#    data loading, edge features, labels, node features, and chronological splits.
# 2. The middle modeling section is rewritten so the temporal module is based on
#    causal self-attention instead of dilated TCN blocks.
# 3. The late-stage workflow is preserved: walk-forward CV, fold bundle saving,
#    overall-best bundle creation, saved-bundle evaluation, and production refit.
# 4. To avoid future-period leakage with horizon-based labels, the split logic
#    inserts a configurable purge/embargo gap between train, validation, test,
#    and final holdout segments.
# 5. The code is intentionally minimal: removed unused sampler paths and legacy
#    compatibility code that is not needed by the current training flow.

# %%
# Step 0: Imports, seed, config
# ======================================================================

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 4))
print("DEVICE:", DEVICE)

CFG: Dict[str, Any] = {
    # ----------------------
    # data
    # ----------------------
    "freq": "1min",
    "data_dir": "../dataset",
    "final_test_frac": 0.10,

    # ----------------------
    # order book
    # ----------------------
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # ----------------------
    # walk-forward windows
    # windows are chronological; purge/embargo gaps are inserted
    # between train->val, val->test, and cv->final holdout
    # ----------------------
    "train_min_frac": 0.55,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,

    # ----------------------
    # scaling
    # ----------------------
    "max_abs_feat": 8.0,
    "max_abs_edge": 4.0,

    # ----------------------
    # graph / edge features
    # edge features are rolling lead-lag correlations
    # ----------------------
    "corr_windows": [10, 30, 60, 120],
    "corr_lags": list(range(0, 11)),
    "edges_mode": "all_pairs",
    "add_self_loops": True,
    "edge_transform": "fisher",
    "edge_scale": True,

    # ----------------------
    # labels
    # TB labels define trade / direction classification,
    # fixed-horizon return is used for regression + utility
    # ----------------------
    "tb_horizon": 30,
    "lookback": 240,
    "tb_pt_mult": 1.70,
    "tb_sl_mult": 1.70,
    "tb_min_barrier": 0.0035,
    "tb_max_barrier": 0.0150,

    "fixed_horizon": 30,
    "fixed_ret_clip": 0.010,

    # ----------------------
    # split safety
    # purge prevents overlap between label horizons across adjacent splits
    # embargo adds an optional extra dead zone after purge
    # for 1-minute data, using horizon-sized gaps is the safest default
    # ----------------------
    "split_purge": 30,
    "split_embargo": 30,

    # ----------------------
    # training
    # ----------------------
    "batch_size": 64,
    "epochs": 80,
    "lr": 2.0e-4,
    "weight_decay": 2.0e-3,
    "grad_clip": 1.0,
    "dropout": 0.20,

    # scheduler
    "use_onecycle": True,
    "onecycle_pct_start": 0.25,
    "onecycle_div_factor": 40.0,
    "onecycle_final_div": 500.0,

    # ----------------------
    # model: graph + temporal attention
    # each block performs graph propagation, then causal temporal attention
    # ----------------------
    "model_dim": 64,
    "temporal_layers": 3,
    "temporal_heads": 4,
    "ff_mult": 2,
    "attn_dropout": 0.10,

    # ----------------------
    # adaptive adjacency
    # ----------------------
    "adj_emb_dim": 16,
    "adj_temperature": 1.00,
    "adaptive_topk": 3,
    "adj_l1_lambda": 2e-3,
    "adj_prior_lambda": 1e-2,

    # prior adjacency
    "prior_use_abs": True,
    "prior_diag_boost": 1.0,
    "prior_row_normalize": True,

    # ----------------------
    # trading eval / thresholds
    # ----------------------
    "cost_bps": 1.0,
    "thr_trade_grid": [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97],
    "thr_dir_grid": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "eval_min_trades": 50,
    "max_trade_rate_val": 0.15,
    "trade_rate_penalty": 3.5,
    "thr_objective": "pnl_sum",
    "proxy_target_trades": [20, 40, 60, 80, 100],
    "thr_pairs_check": [],

    # ----------------------
    # selection metric
    # ----------------------
    "sel_b_dir_auc": 0.55,

    # ----------------------
    # loss weights
    # ----------------------
    "loss_w_trade": 0.65,
    "loss_w_dir": 0.80,
    "loss_w_ret": 0.25,
    "loss_w_utility": 0.55,

    # regression / utility
    "ret_huber_delta": 0.003,
    "utility_k": 3.0,
    "utility_scale": 220.0,
    "utility_mask_true_trades": False,
    "trade_prob_penalty": 1.00,

    # artifacts
    "artifact_dir": "./artifacts_attention",
}

ASSETS = ["ADA", "BTC", "ETH"]
ASSET2IDX = {a: i for i, a in enumerate(ASSETS)}
TARGET_ASSET = "ETH"
TARGET_NODE = ASSET2IDX[TARGET_ASSET]

ART_DIR = Path(CFG["artifact_dir"])
ART_DIR.mkdir(parents=True, exist_ok=True)

print("Assets:", ASSETS, "| Target:", TARGET_ASSET)
print("Artifacts dir:", str(ART_DIR.resolve()))


# %% [markdown]
# This block defines the directed graph used by the model.
#
# The graph is built from all ordered asset pairs plus self-loops. Self-loops are
# kept because both the learned adaptive adjacency and the prior adjacency are more
# stable when each node can preserve some self-information explicitly.
#
# `EDGE_INDEX` is stored as integer node pairs and is reused later both for prior
# adjacency construction and for static topology support.

# %%
# Step 0.1: Graph edges
# ======================================================================

def build_edge_list(cfg: Dict[str, Any], assets: List[str]) -> List[Tuple[str, str]]:
    mode = str(cfg.get("edges_mode", "manual"))
    if mode == "manual":
        edges = list(cfg["edges"])
    elif mode == "all_pairs":
        edges = [(s, t) for s in assets for t in assets if s != t]
    else:
        raise ValueError(f"Unknown edges_mode={mode}")

    if bool(cfg.get("add_self_loops", True)):
        edges = edges + [(a, a) for a in assets]
    return edges


EDGE_LIST = build_edge_list(CFG, ASSETS)
EDGE_NAMES = [f"{s}->{t}" for (s, t) in EDGE_LIST]
EDGE_INDEX = torch.tensor(
    [[ASSET2IDX[s], ASSET2IDX[t]] for (s, t) in EDGE_LIST],
    dtype=torch.long,
)

print("EDGE_LIST:", EDGE_NAMES)
print("EDGE_INDEX:", EDGE_INDEX.tolist())


# %% [markdown]
# This block loads and merges the raw per-asset CSV files.
#
# The loader keeps only the fields that are required downstream:
# midpoint, spread, buys, sells, and per-level bid/ask notionals.
# Timestamps are rounded to the minute and all assets are joined on the same index.
#
# After the join, per-asset log-returns are added because they are used both for
# node features and for lead-lag edge features.

# %%
# Step 1: Data loading
# ======================================================================

def load_asset(
    asset: str,
    freq: str,
    data_dir: Path,
    book_levels: int,
    part: Tuple[int, int] = (0, 80),
) -> pd.DataFrame:
    path = data_dir / f"{asset}_{freq}.csv"
    df = pd.read_csv(path)
    df = df.iloc[int(len(df) * part[0] / 100): int(len(df) * part[1] / 100)]

    df["timestamp"] = pd.to_datetime(df["system_time"], utc=True).dt.round("min")
    df = df.sort_values("timestamp").set_index("timestamp")

    bid_cols = [f"bids_notional_{i}" for i in range(book_levels)]
    ask_cols = [f"asks_notional_{i}" for i in range(book_levels)]
    needed = ["midpoint", "spread", "buys", "sells"] + bid_cols + ask_cols

    missing = [c for c in needed if c not in df.columns]
    if missing:
        tail = "..." if len(missing) > 10 else ""
        raise ValueError(f"{asset}: missing columns in CSV: {missing[:10]}{tail}")

    return df[needed]


def load_all_assets() -> pd.DataFrame:
    freq = CFG["freq"]
    data_dir = Path(CFG["data_dir"])
    book_levels = int(CFG["book_levels"])

    def rename_cols(df_one: pd.DataFrame, asset: str) -> pd.DataFrame:
        rename_map = {
            "midpoint": asset,
            "buys": f"buys_{asset}",
            "sells": f"sells_{asset}",
            "spread": f"spread_{asset}",
        }
        for i in range(book_levels):
            rename_map[f"bids_notional_{i}"] = f"bids_vol_{asset}_{i}"
            rename_map[f"asks_notional_{i}"] = f"asks_vol_{asset}_{i}"
        return df_one.rename(columns=rename_map)

    frames = []
    for asset in ASSETS:
        df_asset = load_asset(asset, freq, data_dir, book_levels, part=(0, 75))
        frames.append(rename_cols(df_asset, asset))

    df_out = frames[0]
    for frame in frames[1:]:
        df_out = df_out.join(frame, how="inner")

    return df_out.reset_index()


df = load_all_assets()
for a in ASSETS:
    df[f"lr_{a}"] = np.log(df[a]).diff().fillna(0.0)

print("Loaded df:", df.shape)
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
print(df.head(2))


# %% [markdown]
# This block constructs edge features from rolling lead-lag correlations.
#
# For each directed edge `source -> target`, and for each lag in `corr_lags`,
# the code computes a rolling correlation between:
#
# - source return shifted by `lag`
# - target return at the current timestamp
#
# This preserves causal direction for positive lags: source information comes from
# the past. Self-loop edges are filled with constant 1.0 features.
#
# These edge features are later used only for adjacency priors, not as inputs to
# a temporal convolution. That is one of the main differences in this attention
# rewrite.

# %%
# Step 2: Edge features (rolling corr with lead-lag 0..10 minutes)
# ======================================================================

def _fisher_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1.0 + x + eps) / (1.0 - x + eps))


def build_corr_array(
    df_: pd.DataFrame,
    corr_windows: List[int],
    edges: List[Tuple[str, str]],
    lags: List[int],
    transform: str = "fisher",
) -> np.ndarray:
    """
    Edge features per time:
      for edge s->t:
        for lag in lags:
          corr(lr_s.shift(lag), lr_t) over rolling window

    No leakage:
      positive lag means the source is shifted into the past.
    """
    T_ = len(df_)
    E_ = len(edges)
    W_ = len(corr_windows)
    Lg = len(lags)
    out = np.zeros((T_, E_, W_ * Lg), dtype=np.float32)

    lr_map = {a: df_[f"lr_{a}"].astype(float) for a in ASSETS}

    for ei, (s, t) in enumerate(edges):
        if s == t:
            out[:, ei, :] = 1.0
            continue

        src0 = lr_map[s]
        dst0 = lr_map[t]

        feat_idx = 0
        for lag in lags:
            src = src0.shift(int(lag)) if int(lag) > 0 else src0
            for w in corr_windows:
                r = src.rolling(int(w), min_periods=1).corr(dst0)
                r = np.nan_to_num(
                    r.to_numpy(dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if transform == "fisher":
                    r = _fisher_z(r).astype(np.float32)
                out[:, ei, feat_idx] = r
                feat_idx += 1

    return out.astype(np.float32)


edge_feat = build_corr_array(
    df,
    CFG["corr_windows"],
    EDGE_LIST,
    CFG["corr_lags"],
    transform=str(CFG.get("edge_transform", "fisher")),
)

print("edge_feat shape:", edge_feat.shape, "(T,E,edge_dim)")
print("edge_dim =", edge_feat.shape[-1])


# %% [markdown]
# This block builds the targets.
#
# The classification targets come from triple-barrier labeling on ETH returns:
#
# - `y_trade`: whether a trade should be taken
# - `y_dir`: trade direction for true-trade cases
#
# A second continuous target, `fixed_ret`, is the fixed-horizon future log-return.
# It is used by the regression head and by the soft utility term in the loss.
#
# The longest label horizon is later reused by the split builder to insert a
# leakage-safe gap between chronological blocks.

# %%
# Step 3: Labels (TB for trade/dir) + fixed-horizon return
# ======================================================================

EPS = 1e-6


def triple_barrier_labels_from_lr(
    lr: pd.Series,
    horizon: int,
    vol_window: int,
    pt_mult: float,
    sl_mult: float,
    min_barrier: float,
    max_barrier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_tb: {0=down, 1=flat/no-trade, 2=up}
      exit_ret: realized log-return to exit (tp/sl/timeout)
      exit_t: exit index
      thr: barrier per t
    """
    lr = lr.astype(float).copy()
    T = len(lr)

    vol = lr.rolling(vol_window, min_periods=max(10, vol_window // 10)).std().shift(1)
    thr = (vol * math.sqrt(horizon)).clip(lower=min_barrier, upper=max_barrier)

    y = np.ones(T, dtype=np.int64)
    exit_ret = np.zeros(T, dtype=np.float32)
    exit_t = np.arange(T, dtype=np.int64)

    lr_np = lr.fillna(0.0).to_numpy(dtype=np.float64)
    thr_np = thr.fillna(min_barrier).to_numpy(dtype=np.float64)

    for t in range(T - horizon - 1):
        up = pt_mult * thr_np[t]
        dn = -sl_mult * thr_np[t]

        cum = 0.0
        hit = 1
        et = t + horizon
        er = 0.0

        for dt in range(1, horizon + 1):
            cum += lr_np[t + dt]
            if cum >= up:
                hit, et, er = 2, t + dt, cum
                break
            if cum <= dn:
                hit, et, er = 0, t + dt, cum
                break

        if hit == 1:
            er = float(np.sum(lr_np[t + 1: t + horizon + 1]))
            et = t + horizon

        y[t] = hit
        exit_ret[t] = er
        exit_t[t] = et

    return y, exit_ret, exit_t, thr_np


def fixed_horizon_future_return(lr: np.ndarray, horizon: int) -> np.ndarray:
    """
    fixed return at time t:
      r_H(t) = sum_{i=1..H} lr[t+i]
    """
    lr = np.asarray(lr, dtype=np.float64)
    T = lr.shape[0]
    out = np.zeros(T, dtype=np.float32)
    if horizon <= 0:
        return out

    for t in range(0, T - horizon - 1):
        out[t] = float(lr[t + 1: t + horizon + 1].sum())
    return out


y_tb, exit_ret, exit_t, tb_thr = triple_barrier_labels_from_lr(
    df["lr_ETH"],
    horizon=int(CFG["tb_horizon"]),
    vol_window=int(CFG["lookback"]),
    pt_mult=float(CFG["tb_pt_mult"]),
    sl_mult=float(CFG["tb_sl_mult"]),
    min_barrier=float(CFG["tb_min_barrier"]),
    max_barrier=float(CFG["tb_max_barrier"]),
)

y_trade = (y_tb != 1).astype(np.int64)
y_dir = (y_tb == 2).astype(np.int64)
fixed_ret = fixed_horizon_future_return(
    df["lr_ETH"].to_numpy(dtype=np.float64),
    int(CFG["fixed_horizon"]),
)

LABEL_HORIZON = max(int(CFG["tb_horizon"]), int(CFG["fixed_horizon"]))
if int(CFG["split_purge"]) < LABEL_HORIZON:
    CFG["split_purge"] = LABEL_HORIZON

dist = np.bincount(y_tb, minlength=3)
print("TB dist [down,flat,up]:", dist)
print("True trade ratio:", float(y_trade.mean()))
print("fixed_ret stats: mean=", float(np.mean(fixed_ret)), "std=", float(np.std(fixed_ret)))
print("Effective label horizon:", LABEL_HORIZON)


# %% [markdown]
# This block builds per-node features for each asset.
#
# The feature set remains close to the original notebook:
#
# - return and spread
# - log-transformed buys / sells and OFI
# - depth imbalance over all 15 levels
# - top-level imbalance features
# - near-vs-far depth ratios and imbalances
#
# The final tensor shape is `(T, N, F)`. Sample anchors are then restricted to
# timestamps that have enough lookback history and enough future room for both
# triple-barrier and fixed-horizon targets.

# %%
# Step 4: Node features
# ======================================================================

def safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def build_node_tensor(df_: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    book_levels = int(CFG["book_levels"])
    top_k = int(CFG["top_levels"])
    near_k = int(CFG["near_levels"])

    if near_k >= book_levels:
        raise ValueError("CFG['near_levels'] must be < CFG['book_levels']")

    feat_names = [
        "lr",
        "spread",
        "log_buys",
        "log_sells",
        "ofi",
        "DI_15",
        "DI_L0",
        "DI_L1",
        "DI_L2",
        "DI_L3",
        "DI_L4",
        "near_ratio_bid",
        "near_ratio_ask",
        "di_near",
        "di_far",
    ]

    feats_all = []
    for a in ASSETS:
        lr = df_[f"lr_{a}"].values.astype(np.float32)
        spread = df_[f"spread_{a}"].values.astype(np.float32)

        buys = df_[f"buys_{a}"].values.astype(np.float32)
        sells = df_[f"sells_{a}"].values.astype(np.float32)

        log_buys = safe_log1p(buys).astype(np.float32)
        log_sells = safe_log1p(sells).astype(np.float32)
        ofi = ((buys - sells) / (buys + sells + EPS)).astype(np.float32)

        bids_lvls = np.stack(
            [df_[f"bids_vol_{a}_{i}"].values.astype(np.float32) for i in range(book_levels)],
            axis=1,
        )
        asks_lvls = np.stack(
            [df_[f"asks_vol_{a}_{i}"].values.astype(np.float32) for i in range(book_levels)],
            axis=1,
        )

        bid_sum = bids_lvls.sum(axis=1)
        ask_sum = asks_lvls.sum(axis=1)
        di_15 = ((bid_sum - ask_sum) / (bid_sum + ask_sum + EPS)).astype(np.float32)

        di_levels = []
        for i in range(top_k):
            b = bids_lvls[:, i]
            s = asks_lvls[:, i]
            di_levels.append(((b - s) / (b + s + EPS)).astype(np.float32))
        di_l0_4 = np.stack(di_levels, axis=1)

        bid_near = bids_lvls[:, :near_k].sum(axis=1)
        ask_near = asks_lvls[:, :near_k].sum(axis=1)
        bid_far = bids_lvls[:, near_k:].sum(axis=1)
        ask_far = asks_lvls[:, near_k:].sum(axis=1)

        near_ratio_bid = (bid_near / (bid_far + EPS)).astype(np.float32)
        near_ratio_ask = (ask_near / (ask_far + EPS)).astype(np.float32)

        di_near = ((bid_near - ask_near) / (bid_near + ask_near + EPS)).astype(np.float32)
        di_far = ((bid_far - ask_far) / (bid_far + ask_far + EPS)).astype(np.float32)

        xa = np.column_stack(
            [
                lr,
                spread,
                log_buys,
                log_sells,
                ofi,
                di_15,
                di_l0_4[:, 0],
                di_l0_4[:, 1],
                di_l0_4[:, 2],
                di_l0_4[:, 3],
                di_l0_4[:, 4],
                near_ratio_bid,
                near_ratio_ask,
                di_near,
                di_far,
            ]
        ).astype(np.float32)

        feats_all.append(xa)

    X = np.stack(feats_all, axis=1).astype(np.float32)
    return X, feat_names


X_node_raw, node_feat_names = build_node_tensor(df)

T = len(df)
L = int(CFG["lookback"])
t_min = L - 1
t_max = T - LABEL_HORIZON - 2
sample_t = np.arange(t_min, t_max + 1)
n_samples = len(sample_t)

print("X_node_raw:", X_node_raw.shape, "edge_feat:", edge_feat.shape)
print("n_samples:", n_samples, "| t range:", int(sample_t[0]), "->", int(sample_t[-1]))


# %% [markdown]
# This block creates leakage-safe chronological splits.
#
# The critical change relative to the original notebook is that splits are built
# with an explicit `purge + embargo` gap between adjacent blocks.
#
# Why this matters:
#
# - each sample at anchor time `t` uses future returns up to `t + horizon`
# - if validation starts immediately after training, the last training labels can
#   still depend on future timestamps that sit inside the validation period
# - the same issue appears between validation and test, and between CV and the
#   final holdout
#
# The implementation below preserves the same high-level walk-forward logic while
# leaving dead zones between chronological blocks to prevent that leakage.

# %%
# Step 5: Purged / embargoed chronological splits
# ======================================================================

def make_final_holdout_split(
    n_samples_: int,
    final_test_frac: float,
    purge: int,
    embargo: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < final_test_frac < 0.5):
        raise ValueError("final_test_frac should be in (0, 0.5)")

    n_final = max(1, int(round(final_test_frac * n_samples_)))
    gap = int(purge) + int(embargo)
    cv_end = n_samples_ - n_final - gap

    if cv_end <= 50:
        raise ValueError("Too few samples left for CV after applying final holdout gap.")

    idx_cv = np.arange(0, cv_end, dtype=np.int64)
    idx_final = np.arange(cv_end + gap, n_samples_, dtype=np.int64)

    if len(idx_final) == 0:
        raise ValueError("Final holdout is empty after applying purge/embargo.")
    return idx_cv, idx_final


def make_walk_forward_splits(
    n_samples_: int,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    step_window_frac: float,
    purge: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_min = int(train_min_frac * n_samples_)
    val_w = max(1, int(val_window_frac * n_samples_))
    test_w = max(1, int(test_window_frac * n_samples_))
    step_w = max(1, int(step_window_frac * n_samples_))
    gap = int(purge) + int(embargo)

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    train_end = train_min

    while True:
        val_start = train_end + gap
        val_end = val_start + val_w

        test_start = val_end + gap
        test_end = test_start + test_w

        if test_end > n_samples_:
            break

        idx_train = np.arange(0, train_end, dtype=np.int64)
        idx_val = np.arange(val_start, val_end, dtype=np.int64)
        idx_test = np.arange(test_start, test_end, dtype=np.int64)

        splits.append((idx_train, idx_val, idx_test))
        train_end += step_w

    if not splits:
        raise ValueError("No walk-forward splits were produced. Reduce windows or gaps.")
    return splits


idx_cv_all, idx_final_test = make_final_holdout_split(
    n_samples,
    float(CFG["final_test_frac"]),
    purge=int(CFG["split_purge"]),
    embargo=int(CFG["split_embargo"]),
)
n_samples_cv = len(idx_cv_all)

walk_splits = make_walk_forward_splits(
    n_samples_=n_samples_cv,
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    step_window_frac=float(CFG["step_window_frac"]),
    purge=int(CFG["split_purge"]),
    embargo=int(CFG["split_embargo"]),
)

print("Holdout split:")
print(f"  n_samples total: {n_samples}")
print(f"  n_samples CV   : {len(idx_cv_all)}")
print(f"  n_samples FINAL: {len(idx_final_test)}")
print(f"  purge          : {int(CFG['split_purge'])}")
print(f"  embargo        : {int(CFG['split_embargo'])}")
print("\nWalk-forward folds:", len(walk_splits))
for i, (a, b, c) in enumerate(walk_splits, 1):
    print(f"  fold {i}: train={len(a)} | val={len(b)} | test={len(c)}")


# %% [markdown]
# This block defines the dataset and train-only scaling helpers.
#
# Each sample returns:
#
# - a node sequence of shape `(L, N, F)`
# - an edge-feature sequence of shape `(L, E, D)`
# - classification and regression targets
# - the original sample index
#
# Scaling is fitted only on timestamps that belong to the training region of a fold.
# That preserves chronological integrity and avoids leaking future distributional
# information from validation or test periods into the training transform.

# %%
# Step 6: Dataset and scaling helpers
# ======================================================================

class LobGraphSequenceDatasetTwoHeadFixedH(Dataset):
    """
    Returns:
      x_seq:      (L, N, F)
      e_seq:      (L, E, D)
      y_trade:    scalar in {0, 1}
      y_dir:      scalar in {0, 1}, meaningful only when y_trade=1
      exit_ret:   scalar, TB realized return for threshold / PnL evaluation
      fixed_ret:  scalar, fixed-horizon return for regression / utility
      sidx:       scalar sample index
    """

    def __init__(
        self,
        X_node: np.ndarray,
        E_feat: np.ndarray,
        y_trade_arr: np.ndarray,
        y_dir_arr: np.ndarray,
        exit_ret_arr: np.ndarray,
        fixed_ret_arr: np.ndarray,
        sample_t_: np.ndarray,
        indices: np.ndarray,
        lookback: int,
    ):
        self.X_node = X_node
        self.E_feat = E_feat
        self.y_trade = y_trade_arr
        self.y_dir = y_dir_arr
        self.exit_ret = exit_ret_arr
        self.fixed_ret = fixed_ret_arr
        self.sample_t = sample_t_
        self.indices = indices.astype(np.int64)
        self.L = int(lookback)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        sidx = int(self.indices[i])
        t = int(self.sample_t[sidx])
        t0 = t - self.L + 1

        x_seq = self.X_node[t0: t + 1]
        e_seq = self.E_feat[t0: t + 1]

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(e_seq),
            torch.tensor(int(self.y_trade[t]), dtype=torch.float32),
            torch.tensor(int(self.y_dir[t]), dtype=torch.float32),
            torch.tensor(float(self.exit_ret[t]), dtype=torch.float32),
            torch.tensor(float(self.fixed_ret[t]), dtype=torch.float32),
            torch.tensor(sidx, dtype=torch.long),
        )


def collate_fn_twohead(batch):
    xs, es, ytr, ydir, er_exit, er_fixed, sidxs = zip(*batch)
    return (
        torch.stack(xs, 0),
        torch.stack(es, 0),
        torch.stack(ytr, 0),
        torch.stack(ydir, 0),
        torch.stack(er_exit, 0),
        torch.stack(er_fixed, 0),
        torch.stack(sidxs, 0),
    )


def fit_scale_nodes_train_only(
    X_node_raw_: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    max_abs: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    X_train_time = X_node_raw_[train_time_mask]
    _, _, Fdim = X_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(X_train_time.reshape(-1, Fdim))

    X_scaled = scaler.transform(X_node_raw_.reshape(-1, Fdim)).reshape(X_node_raw_.shape).astype(np.float32)
    X_scaled = np.clip(X_scaled, -max_abs, max_abs).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {
        "center_": scaler.center_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "max_abs": float(max_abs),
    }
    return X_scaled, params


def fit_scale_edges_train_only(
    E_raw_: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    max_abs: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    E_train_time = E_raw_[train_time_mask]
    _, _, D = E_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(E_train_time.reshape(-1, D))

    E_scaled = scaler.transform(E_raw_.reshape(-1, D)).reshape(E_raw_.shape).astype(np.float32)
    E_scaled = np.clip(E_scaled, -max_abs, max_abs).astype(np.float32)
    E_scaled = np.nan_to_num(E_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {
        "center_": scaler.center_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "max_abs": float(max_abs),
    }
    return E_scaled, params


def apply_scaler_params(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(params["center_"], dtype=np.float32)
    scale = np.asarray(params["scale_"], dtype=np.float32)
    max_abs = float(params["max_abs"])

    X2 = (X - center) / (scale + 1e-12)
    X2 = np.clip(X2, -max_abs, max_abs)
    return np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def split_trade_ratio(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> float:
    tt = sample_t_[indices]
    return float(y_trade_arr[tt].mean()) if len(tt) else float("nan")


# %% [markdown]
# This block contains the core model rewrite.
#
# The original temporal module used dilated TCN / Graph WaveNet style blocks.
# In this version, the temporal part is replaced by causal self-attention:
#
# 1. node features are projected to `model_dim`
# 2. graph propagation mixes information across assets via a weighted combination of
#    - static adjacency from the edge list
#    - prior adjacency from edge features
#    - learned adaptive adjacency
# 3. after graph mixing, each node sequence is processed by causal temporal
#    self-attention, not by a temporal convolution
# 4. the final representation of the target node at the last timestamp is sent to
#    the three heads: trade, direction, and fixed-horizon return
#
# The adjacency regularizers from the original notebook are preserved.

# %%
# Step 7: Graph + temporal attention model
# ======================================================================

def build_static_adjacency_from_edges(
    edge_index: torch.Tensor,
    n_nodes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    src = edge_index[:, 0].long()
    dst = edge_index[:, 1].long()
    A[src, dst] = 1.0
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    return A


def build_adj_prior_from_edge_attr(
    edge_attr_last: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    use_abs: bool,
    diag_boost: float,
    row_normalize: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    edge_attr_last = torch.nan_to_num(edge_attr_last, nan=0.0, posinf=0.0, neginf=0.0)
    B, E, _ = edge_attr_last.shape

    score = edge_attr_last.mean(dim=-1)
    if use_abs:
        score = score.abs()
    weight = torch.sigmoid(score)

    A = torch.zeros((B, n_nodes, n_nodes), device=edge_attr_last.device, dtype=edge_attr_last.dtype)
    src = edge_index[:, 0].to(edge_attr_last.device)
    dst = edge_index[:, 1].to(edge_attr_last.device)
    A[:, src, dst] = weight

    diag = torch.arange(n_nodes, device=edge_attr_last.device)
    A[:, diag, diag] = torch.maximum(
        A[:, diag, diag],
        torch.full_like(A[:, diag, diag], float(diag_boost)),
    )

    if row_normalize:
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

    return torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes: int, cfg: Dict[str, Any]):
        super().__init__()
        self.n = int(n_nodes)
        emb_dim = int(cfg.get("adj_emb_dim", 8))
        self.E1 = nn.Parameter(0.01 * torch.randn(self.n, emb_dim))
        self.E2 = nn.Parameter(0.01 * torch.randn(self.n, emb_dim))
        self.temp = max(float(cfg.get("adj_temperature", 1.0)), 1e-3)
        self.topk = int(cfg.get("adaptive_topk", self.n))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.relu(self.E1 @ self.E2.t()) / self.temp

        if 0 < self.topk < self.n:
            vals, idx = torch.topk(logits, k=self.topk, dim=-1)
            masked = torch.full_like(logits, fill_value=float("-inf"))
            masked.scatter_(-1, idx, vals)
            logits = masked

        A = torch.softmax(logits, dim=-1)
        sparsity_proxy = torch.sigmoid(torch.nan_to_num(logits, neginf=-20.0))
        return A, sparsity_proxy, logits


class LearnableSupportMix(nn.Module):
    def __init__(self, n_supports: int = 3):
        super().__init__()
        self.w_logits = nn.Parameter(torch.zeros(n_supports, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.w_logits, dim=0)


def make_causal_attn_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), fill_value=float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = int(ff_mult) * int(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphTemporalAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.graph_norm = nn.LayerNorm(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        self.graph_proj = nn.Linear(dim * 2, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ff = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, C), A: (B, N, N)
        msg = torch.einsum("blnc,bnm->blmc", x, A)
        mix = torch.cat([x, msg], dim=-1)
        x = self.graph_norm(x + self.dropout(self.graph_proj(mix)))

        B, L_, N, C = x.shape
        y = x.permute(0, 2, 1, 3).reshape(B * N, L_, C)
        mask = make_causal_attn_mask(L_, device=y.device)

        attn_out, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        y = self.attn_norm(y + self.dropout(attn_out))
        y = self.ff_norm(y + self.ff(y))

        x = y.reshape(B, N, L_, C).permute(0, 2, 1, 3).contiguous()
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class GraphTemporalAttentionTwoHeadFixedH(nn.Module):
    """
    Input:
      x_seq: (B, L, N, F)
      e_seq: (B, L, E, D)

    Output for the target node at the last timestamp:
      trade_logit: (B,)
      dir_logit:   (B,)
      fixed_hat:   (B,)
    """

    def __init__(self, node_in: int, cfg: Dict[str, Any], n_nodes: int, target_node: int):
        super().__init__()
        self.cfg = cfg
        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)

        dim = int(cfg["model_dim"])
        heads = int(cfg["temporal_heads"])
        layers = int(cfg["temporal_layers"])
        ff_mult = int(cfg["ff_mult"])
        dropout = float(cfg.get("dropout", 0.0))
        attn_dropout = float(cfg.get("attn_dropout", 0.0))

        if dim % heads != 0:
            raise ValueError("model_dim must be divisible by temporal_heads")

        self.in_proj = nn.Linear(int(node_in), dim)
        self.input_norm = nn.LayerNorm(dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, int(cfg["lookback"]), 1, dim))

        A_static = build_static_adjacency_from_edges(EDGE_INDEX, n_nodes=self.n_nodes)
        self.register_buffer("A_static", A_static)

        self.adapt = AdaptiveAdjacency(n_nodes=self.n_nodes, cfg=cfg)
        self.support_mix = LearnableSupportMix(n_supports=3)

        self.blocks = nn.ModuleList(
            [
                GraphTemporalAttentionBlock(
                    dim=dim,
                    heads=heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(dim)
        self.trade_head = nn.Linear(dim, 1)
        self.dir_head = nn.Linear(dim, 1)
        self.fixed_head = nn.Linear(dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_supports(self, e_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = e_seq.shape[0]
        e_last = e_seq[:, -1, :, :]

        A_prior = build_adj_prior_from_edge_attr(
            edge_attr_last=e_last,
            edge_index=EDGE_INDEX.to(e_seq.device),
            n_nodes=self.n_nodes,
            use_abs=bool(self.cfg.get("prior_use_abs", False)),
            diag_boost=float(self.cfg.get("prior_diag_boost", 1.0)),
            row_normalize=bool(self.cfg.get("prior_row_normalize", True)),
        )

        A_adapt_base, sparsity_proxy, _ = self.adapt()
        A_adapt = A_adapt_base.unsqueeze(0).expand(B, -1, -1)

        w = self.support_mix()
        A_static = self.A_static.to(e_seq.device).to(e_seq.dtype).unsqueeze(0).expand(B, -1, -1)

        A_mix = w[0] * A_static + w[1] * A_prior + w[2] * A_adapt
        A_mix = A_mix / (A_mix.sum(dim=-1, keepdim=True) + 1e-8)

        N = self.n_nodes
        offdiag = 1.0 - torch.eye(N, device=e_seq.device, dtype=e_seq.dtype)
        l1_off = (sparsity_proxy.to(e_seq.dtype) * offdiag).abs().mean()
        mse_prior = ((A_adapt - A_prior) ** 2 * offdiag).mean()

        aux = {
            "support_w": w.detach().cpu().numpy().tolist(),
            "_l1_off_t": l1_off,
            "_mse_prior_t": mse_prior,
        }
        return A_mix, aux

    def forward(self, x_seq: torch.Tensor, e_seq: torch.Tensor, return_aux: bool = False):
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        e_seq = torch.nan_to_num(e_seq, nan=0.0, posinf=0.0, neginf=0.0)

        B, L_, N, _ = x_seq.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")

        x = self.in_proj(x_seq)
        x = self.input_norm(x + self.pos_emb[:, :L_])

        A_mix, aux = self._compute_supports(e_seq)

        for block in self.blocks:
            x = block(x, A_mix)

        feat = self.final_norm(x[:, -1, self.target_node, :])

        trade_logit = self.trade_head(feat).squeeze(-1)
        dir_logit = self.dir_head(feat).squeeze(-1)
        fixed_hat = self.fixed_head(feat).squeeze(-1)

        trade_logit = torch.nan_to_num(trade_logit, nan=0.0, posinf=0.0, neginf=0.0)
        dir_logit = torch.nan_to_num(dir_logit, nan=0.0, posinf=0.0, neginf=0.0)
        fixed_hat = torch.nan_to_num(fixed_hat, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            return trade_logit, dir_logit, fixed_hat, aux
        return trade_logit, dir_logit, fixed_hat


# %% [markdown]
# This block implements metrics, threshold search, and the multitask loss.
#
# The loss matches the original high-level setup:
#
# - trade BCE on all samples
# - direction BCE only on true-trade samples
# - Huber regression on fixed-horizon return
# - soft utility term based on predicted trade intensity and predicted direction
#
# The output of the two-head classifier is converted into a 3-class distribution:
# short / flat / long. That distribution is then used by threshold search and
# PnL evaluation exactly like in the original workflow.

# %%
# Step 8: Metrics, thresholding, losses
# ======================================================================

def _safe_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def probs3_from_twohead(p_trade: np.ndarray, p_dir: np.ndarray) -> np.ndarray:
    p_trade = np.asarray(p_trade, dtype=np.float64)
    p_dir = np.asarray(p_dir, dtype=np.float64)

    p_flat = 1.0 - p_trade
    p_long = p_trade * p_dir
    p_short = p_trade * (1.0 - p_dir)

    prob3 = np.stack([p_short, p_flat, p_long], axis=1)
    prob3 = np.clip(prob3, 1e-12, 1.0)
    prob3 = prob3 / prob3.sum(axis=1, keepdims=True)
    return prob3


def compute_trade_dir_auc_from_twohead(
    y_trade_true: np.ndarray,
    y_tb_true: np.ndarray,
    p_trade: np.ndarray,
    p_dir: np.ndarray,
) -> Tuple[float, float]:
    trade_auc = _safe_auc_binary(y_trade_true, p_trade)

    mask_trade = (y_tb_true != 1)
    y_dir_bin = (y_tb_true[mask_trade] == 2).astype(np.int64)
    dir_auc = _safe_auc_binary(y_dir_bin, p_dir[mask_trade])

    return trade_auc, dir_auc


def pnl_from_probs_3class(
    prob3: np.ndarray,
    exit_ret_arr: np.ndarray,
    thr_trade: float,
    thr_dir: float,
    cost_bps: float,
) -> Dict[str, Any]:
    prob3 = np.asarray(prob3, dtype=np.float64)
    exit_ret_arr = np.asarray(exit_ret_arr, dtype=np.float64)

    p_short = prob3[:, 0]
    p_flat = prob3[:, 1]
    p_long = prob3[:, 2]

    trade_conf = 1.0 - p_flat
    dir_prob = p_long / (p_long + p_short + 1e-12)
    dir_conf = np.maximum(dir_prob, 1.0 - dir_prob)

    mask = (trade_conf >= float(thr_trade)) & (dir_conf >= float(thr_dir))

    action = np.zeros_like(exit_ret_arr, dtype=np.float64)
    action[mask] = np.where(dir_prob[mask] >= 0.5, 1.0, -1.0)

    cost = (float(cost_bps) * 1e-4) * mask.astype(np.float64)
    pnl = action * exit_ret_arr - cost

    n = int(len(exit_ret_arr))
    n_tr = int(mask.sum())

    return {
        "n": n,
        "n_trades": n_tr,
        "trade_rate": float(n_tr / max(1, n)),
        "pnl_sum": float(pnl.sum()),
        "pnl_mean": float(pnl.mean()) if n else float("nan"),
        "pnl_per_trade": float(pnl.sum() / max(1, n_tr)),
    }


def build_trade_threshold_grid(
    p_trade: np.ndarray,
    base_grid: Optional[List[float]],
    target_trades_list: Optional[List[int]],
) -> List[float]:
    p_trade = np.asarray(p_trade, dtype=np.float64)
    p_trade = p_trade[np.isfinite(p_trade)]
    if p_trade.size == 0:
        return base_grid or [0.5]

    thrs = set(float(t) for t in (base_grid or []))
    if target_trades_list:
        N = int(p_trade.size)
        for k in target_trades_list:
            k = int(k)
            if k <= 0:
                continue
            if k >= N:
                thr = float(np.min(p_trade))
            else:
                q = 1.0 - (k / N)
                thr = float(np.quantile(p_trade, q))
            thrs.add(float(np.clip(thr, 0.01, 0.99)))

    out = sorted(thrs)
    cleaned: List[float] = []
    for t in out:
        if not cleaned or abs(t - cleaned[-1]) > 1e-6:
            cleaned.append(float(t))
    return cleaned


def sweep_thresholds_3class(
    prob3: np.ndarray,
    exit_ret_arr: np.ndarray,
    cfg: Dict[str, Any],
    min_trades: int,
    target_trade_rate: Optional[float],
) -> pd.DataFrame:
    prob3 = np.asarray(prob3, dtype=np.float64)
    p_trade = 1.0 - prob3[:, 1]

    thr_trade_grid = build_trade_threshold_grid(
        p_trade=p_trade,
        base_grid=cfg.get("thr_trade_grid", [0.5]),
        target_trades_list=cfg.get("proxy_target_trades", None),
    )
    thr_dir_grid = cfg.get("thr_dir_grid", [0.5])

    obj = str(cfg.get("thr_objective", "pnl_sum"))
    max_rate = cfg.get("max_trade_rate_val", None)
    penalty = float(cfg.get("trade_rate_penalty", 0.0))

    rows = []
    for thr_t in thr_trade_grid:
        for thr_d in thr_dir_grid:
            m = pnl_from_probs_3class(prob3, exit_ret_arr, thr_t, thr_d, cfg["cost_bps"])
            if int(m["n_trades"]) < int(min_trades):
                continue
            if max_rate is not None and float(m["trade_rate"]) > float(max_rate):
                continue

            base = float(m.get(obj, np.nan))
            if not np.isfinite(base):
                continue

            if target_trade_rate is not None:
                score = base - penalty * abs(float(m["trade_rate"]) - float(target_trade_rate))
            else:
                score = base - penalty * float(m["trade_rate"])

            rows.append({"thr_trade": float(thr_t), "thr_dir": float(thr_d), "score": float(score), **m})

    if not rows:
        return sweep_thresholds_3class(
            prob3=prob3,
            exit_ret_arr=exit_ret_arr,
            cfg=cfg,
            min_trades=1,
            target_trade_rate=target_trade_rate,
        )

    return pd.DataFrame(rows).sort_values(["score", "pnl_sum"], ascending=False)


def total_loss_with_adj_reg(loss: torch.Tensor, aux: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
    lam_l1 = float(cfg.get("adj_l1_lambda", 0.0))
    lam_pr = float(cfg.get("adj_prior_lambda", 0.0))
    reg = 0.0
    if lam_l1 > 0:
        reg = reg + lam_l1 * aux["_l1_off_t"]
    if lam_pr > 0:
        reg = reg + lam_pr * aux["_mse_prior_t"]
    return loss + reg


def compute_pos_weights_binary(y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    n_pos = max(n_pos, 1.0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)


def multitask_loss_twohead_fixedH(
    trade_logit: torch.Tensor,
    dir_logit: torch.Tensor,
    fixed_hat: torch.Tensor,
    y_trade_t: torch.Tensor,
    y_dir_t: torch.Tensor,
    fixed_ret_t: torch.Tensor,
    bce_trade_fn: nn.Module,
    bce_dir_fn: nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    trade_logit = trade_logit.reshape(-1)
    dir_logit = dir_logit.reshape(-1)
    fixed_hat = fixed_hat.reshape(-1)
    y_trade_t = y_trade_t.reshape(-1)
    y_dir_t = y_dir_t.reshape(-1)
    fixed_ret_t = fixed_ret_t.reshape(-1)

    device = trade_logit.device
    dtype = trade_logit.dtype

    def _cfg_float(key: str, default: float) -> float:
        v = cfg.get(key, default)
        return float(v) if v is not None else float(default)

    wT = _cfg_float("loss_w_trade", 1.0)
    wD = _cfg_float("loss_w_dir", 1.0)
    wR = _cfg_float("loss_w_ret", 1.0)
    wU = _cfg_float("loss_w_utility", 1.0)

    bceT = bce_trade_fn(trade_logit, y_trade_t.to(dtype=torch.float32))

    maskT = (y_trade_t == 1)
    if maskT.any():
        bceD = bce_dir_fn(dir_logit[maskT], y_dir_t[maskT].to(dtype=torch.float32))
    else:
        bceD = torch.zeros((), device=device, dtype=dtype)

    clip_val = _cfg_float("fixed_ret_clip", 0.0)
    fr = fixed_ret_t.to(dtype=fixed_hat.dtype)
    if clip_val > 0:
        fr = torch.clamp(fr, -clip_val, clip_val)

    delta = _cfg_float("ret_huber_delta", 0.01)
    huber = F.huber_loss(fixed_hat, fr, delta=delta, reduction="mean")

    p_trade = torch.sigmoid(trade_logit)
    k = _cfg_float("utility_k", 2.0)
    soft_dir = torch.tanh(k * dir_logit)
    pos = p_trade * soft_dir

    fee = _cfg_float("cost_bps", 0.0) * 1e-4
    utility_vec = pos * fr - fee * pos.abs()

    if bool(cfg.get("utility_mask_true_trades", False)):
        util = utility_vec[maskT].mean() if maskT.any() else torch.zeros((), device=device, dtype=dtype)
    else:
        util = utility_vec.mean()

    util_scale = _cfg_float("utility_scale", 1.0)
    utilS = util_scale * util
    util_loss = -utilS

    total = (wT * bceT) + (wD * bceD) + (wR * huber) + (wU * util_loss)

    parts = {
        "bce_trade": bceT.detach(),
        "bce_dir": bceD.detach(),
        "huber": huber.detach(),
        "util": util.detach(),
        "util_scaled": utilS.detach(),
        "pos_abs": pos.abs().mean().detach(),
        "p_trade_mean": p_trade.mean().detach(),
    }
    return total, parts


# %% [markdown]
# This block keeps the threshold-pairs diagnostic logic.
#
# It is used in exactly two places:
#
# - once after walk-forward CV ends, on the concatenated fold predictions
# - once after the production refit, on the final holdout
#
# The diagnostic table is intentionally separated from model selection. It provides
# a stable, explicit view of how threshold pairs behave under the current scores.

# %%
# Step 8.5: Threshold-pairs check utilities
# ======================================================================

def _normalize_pairs_list(pairs: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if pairs is None:
        return out
    if isinstance(pairs, (list, tuple)):
        for p in pairs:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            try:
                out.append((float(p[0]), float(p[1])))
            except Exception:
                continue
    return out


def build_threshold_pairs_for_check(prob3: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[float, float]]:
    prob3 = np.asarray(prob3, dtype=np.float64)
    p_trade = 1.0 - prob3[:, 1]

    thr_trade_grid = build_trade_threshold_grid(
        p_trade=p_trade,
        base_grid=cfg.get("thr_trade_grid", [0.5]),
        target_trades_list=cfg.get("proxy_target_trades", None),
    )
    thr_dir_grid = list(cfg.get("thr_dir_grid", [0.5]))

    pairs = {(float(tt), float(td)) for tt in thr_trade_grid for td in thr_dir_grid}
    for tt, td in _normalize_pairs_list(cfg.get("thr_pairs_check", [])):
        pairs.add((float(tt), float(td)))

    return sorted(pairs)


def check_threshold_pairs_once(
    prob3: np.ndarray,
    exit_ret_arr: np.ndarray,
    cfg: Dict[str, Any],
    label: str,
    min_trades: Optional[int] = None,
    target_trade_rate: Optional[float] = None,
    top_k: int = 20,
    save_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    prob3 = np.asarray(prob3, dtype=np.float64)
    exit_ret_arr = np.asarray(exit_ret_arr, dtype=np.float64)

    min_trades_eff = int(min_trades) if min_trades is not None else int(cfg.get("eval_min_trades", 1))
    pairs = build_threshold_pairs_for_check(prob3, cfg)

    thr_trade_vals = sorted({p[0] for p in pairs})
    thr_dir_vals = sorted({p[1] for p in pairs})

    cfg_tmp = dict(cfg)
    cfg_tmp["thr_trade_grid"] = thr_trade_vals
    cfg_tmp["thr_dir_grid"] = thr_dir_vals

    df_tbl = sweep_thresholds_3class(
        prob3=prob3,
        exit_ret_arr=exit_ret_arr,
        cfg=cfg_tmp,
        min_trades=min_trades_eff,
        target_trade_rate=target_trade_rate,
    ).copy()

    cols = ["thr_trade", "thr_dir", "score", "pnl_sum", "pnl_per_trade", "trade_rate", "n_trades", "n"]
    cols = [c for c in cols if c in df_tbl.columns]

    print("\n" + "=" * 90)
    print(label)
    print(f"checked pairs: {len(df_tbl)} | min_trades={min_trades_eff} | target_trade_rate={target_trade_rate}")
    print(df_tbl[cols].head(int(top_k)).to_string(index=False))

    if save_csv_path is not None:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_tbl.to_csv(save_csv_path, index=False)
        print(f"Saved threshold check table -> {save_csv_path}")

    return df_tbl


def check_threshold_pairs_after_cv_once(
    fold_artifacts: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    split: str = "test",
    top_k: int = 20,
    save_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")

    prob3_all = []
    exit_all = []
    for art in fold_artifacts:
        ev = art["val_eval"] if split == "val" else art["test_eval"]
        prob3_all.append(np.asarray(ev["prob3"], dtype=np.float64))
        exit_all.append(np.asarray(ev["exit_ret"], dtype=np.float64))

    prob3_cat = np.concatenate(prob3_all, axis=0) if prob3_all else np.zeros((0, 3), dtype=np.float64)
    exit_cat = np.concatenate(exit_all, axis=0) if exit_all else np.zeros((0,), dtype=np.float64)

    return check_threshold_pairs_once(
        prob3=prob3_cat,
        exit_ret_arr=exit_cat,
        cfg=cfg,
        label=f"THRESHOLD CHECK (AFTER CV) | aggregated split={split.upper()} over folds",
        min_trades=int(cfg.get("eval_min_trades", 1)),
        target_trade_rate=None,
        top_k=top_k,
        save_csv_path=save_csv_path,
    )


# %% [markdown]
# This block evaluates a model on an arbitrary index set.
#
# It reuses the same loss decomposition as training, collects probabilities, and
# converts the two-head outputs into a three-class distribution for downstream
# thresholding and PnL evaluation.
#
# The returned dictionary is intentionally rich because it is reused both inside
# walk-forward CV and in the saved-bundle evaluation flow.

# %%
# Step 9: Evaluation
# ======================================================================

@torch.no_grad()
def eval_twohead_on_indices(
    model: nn.Module,
    X_scaled: np.ndarray,
    edge_scaled: np.ndarray,
    indices: np.ndarray,
    bce_trade: nn.Module,
    bce_dir: nn.Module,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ds = LobGraphSequenceDatasetTwoHeadFixedH(
        X_node=X_scaled,
        E_feat=edge_scaled,
        y_trade_arr=y_trade,
        y_dir_arr=y_dir,
        exit_ret_arr=exit_ret,
        fixed_ret_arr=fixed_ret,
        sample_t_=sample_t,
        indices=indices.astype(np.int64),
        lookback=int(cfg["lookback"]),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_twohead,
        num_workers=0,
    )

    model.eval()

    tot_loss = 0.0
    tot_tr = 0.0
    tot_dr = 0.0
    tot_huber = 0.0
    tot_util = 0.0
    tot_util_s = 0.0
    tot_pos_abs = 0.0
    tot_ptr = 0.0
    n = 0

    p_trade_all = []
    p_dir_all = []
    y_trade_all = []
    exit_all = []

    for x, e, yt, yd, er_exit, er_fixed, _sidx in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        yt = yt.to(DEVICE).float()
        yd = yd.to(DEVICE).float()
        er_fixed = er_fixed.to(DEVICE).float()

        trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
        loss, parts = multitask_loss_twohead_fixedH(
            trade_logit,
            dir_logit,
            fixed_hat,
            yt,
            yd,
            er_fixed,
            bce_trade,
            bce_dir,
            cfg,
        )
        loss = total_loss_with_adj_reg(loss, aux, cfg)

        B = int(yt.size(0))
        tot_loss += float(loss.item()) * B
        tot_tr += float(parts["bce_trade"].item()) * B
        tot_dr += float(parts["bce_dir"].item()) * B
        tot_huber += float(parts["huber"].item()) * B
        tot_util += float(parts["util"].item()) * B
        tot_util_s += float(parts["util_scaled"].item()) * B
        tot_pos_abs += float(parts["pos_abs"].item()) * B
        tot_ptr += float(parts["p_trade_mean"].item()) * B
        n += B

        p_trade_all.append(torch.sigmoid(trade_logit).detach().cpu().numpy())
        p_dir_all.append(torch.sigmoid(dir_logit).detach().cpu().numpy())
        y_trade_all.append(yt.detach().cpu().numpy())
        exit_all.append(er_exit.detach().cpu().numpy())

    p_trade_np = np.concatenate(p_trade_all, axis=0) if p_trade_all else np.zeros((0,), dtype=np.float64)
    p_dir_np = np.concatenate(p_dir_all, axis=0) if p_dir_all else np.zeros((0,), dtype=np.float64)
    y_trade_np = (np.concatenate(y_trade_all, axis=0) > 0.5).astype(np.int64) if y_trade_all else np.zeros((0,), dtype=np.int64)
    er_exit_np = np.concatenate(exit_all, axis=0) if exit_all else np.zeros((0,), dtype=np.float64)

    t_idx = sample_t[indices.astype(np.int64)]
    y_tb_np = y_tb[t_idx].astype(np.int64)

    trade_auc, dir_auc = compute_trade_dir_auc_from_twohead(y_trade_np, y_tb_np, p_trade_np, p_dir_np)
    prob3 = probs3_from_twohead(p_trade_np, p_dir_np)

    return {
        "loss": float(tot_loss / max(1, n)),
        "loss_trade": float(tot_tr / max(1, n)),
        "loss_dir": float(tot_dr / max(1, n)),
        "loss_huber": float(tot_huber / max(1, n)),
        "soft_util_mean": float(tot_util / max(1, n)),
        "soft_util_scaled_mean": float(tot_util_s / max(1, n)),
        "pos_abs_mean": float(tot_pos_abs / max(1, n)),
        "p_trade_mean": float(tot_ptr / max(1, n)),
        "trade_auc": float(trade_auc) if np.isfinite(trade_auc) else float("nan"),
        "dir_auc": float(dir_auc) if np.isfinite(dir_auc) else float("nan"),
        "p_trade": p_trade_np,
        "p_dir": p_dir_np,
        "prob3": prob3,
        "y_tb": y_tb_np,
        "exit_ret": er_exit_np,
    }


# %% [markdown]
# This block saves and loads compact training bundles.
#
# The bundle format is intentionally simple:
#
# - model weights: `.pt`
# - node and edge scalers: `.npz`
# - metadata and config: `.json`
#
# No pickle-based custom objects are required for inference or later evaluation.

# %%
# Step 10: Artifact saving/loading
# ======================================================================

def _to_jsonable_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in cfg.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def save_scaler_npz(path: Path, params: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        center_=np.asarray(params["center_"], dtype=np.float32),
        scale_=np.asarray(params["scale_"], dtype=np.float32),
        max_abs=np.asarray([float(params["max_abs"])], dtype=np.float32),
    )


def load_scaler_npz(path: Path) -> Dict[str, Any]:
    data = np.load(str(path))
    return {
        "center_": data["center_"].astype(np.float32),
        "scale_": data["scale_"].astype(np.float32),
        "max_abs": float(data["max_abs"][0]),
    }


def save_bundle(
    bundle_dir: Path,
    name: str,
    model_state: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    node_scaler_params: Dict[str, Any],
    edge_scaler_params: Optional[Dict[str, Any]],
    extra_meta: Dict[str, Any],
) -> Dict[str, Optional[Path]]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / f"{name}_weights.pt"
    node_scaler_path = bundle_dir / f"{name}_node_scaler.npz"
    edge_scaler_path = bundle_dir / f"{name}_edge_scaler.npz"
    meta_path = bundle_dir / f"{name}_meta.json"

    torch.save(model_state, str(weights_path))
    save_scaler_npz(node_scaler_path, node_scaler_params)

    edge_scaler_file = None
    if edge_scaler_params is not None:
        save_scaler_npz(edge_scaler_path, edge_scaler_params)
        edge_scaler_file = edge_scaler_path.name

    meta = {
        "name": name,
        "weights_file": weights_path.name,
        "node_scaler_file": node_scaler_path.name,
        "edge_scaler_file": edge_scaler_file,
        "cfg": _to_jsonable_cfg(cfg),
        **extra_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "weights": weights_path,
        "node_scaler": node_scaler_path,
        "edge_scaler": edge_scaler_path if edge_scaler_params is not None else None,
        "meta": meta_path,
    }


def load_bundle(bundle_dir: Path, name: str) -> Dict[str, Any]:
    meta_path = bundle_dir / f"{name}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    weights_path = bundle_dir / meta["weights_file"]
    node_scaler_path = bundle_dir / meta["node_scaler_file"]
    edge_scaler_file = meta.get("edge_scaler_file", None)
    edge_scaler_path = (bundle_dir / edge_scaler_file) if edge_scaler_file else None

    state = torch.load(str(weights_path), map_location="cpu")
    node_scaler_params = load_scaler_npz(node_scaler_path)
    edge_scaler_params = load_scaler_npz(edge_scaler_path) if edge_scaler_path else None

    return {
        "meta": meta,
        "state": state,
        "node_scaler_params": node_scaler_params,
        "edge_scaler_params": edge_scaler_params,
    }


# %% [markdown]
# This block trains one walk-forward fold.
#
# The model selection logic is kept from the original workflow:
#
# `selection = soft_utility_scaled_mean + b * dir_auc`
#
# Validation thresholds are selected only on the validation block, and those
# thresholds are then applied unchanged to the fold test block.
#
# The code also logs support weights so that the contribution of static, prior,
# and adaptive adjacency can be monitored over time.

# %%
# Step 11: Train one fold
# ======================================================================

def train_one_fold_twohead_fixedH(
    fold_id: int,
    X_scaled: np.ndarray,
    edge_scaled: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    node_scaler_params: Dict[str, Any],
    edge_scaler_params: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    t_train = sample_t[idx_train]
    ytr_train = y_trade[t_train].astype(np.int64)
    ytb_train = y_tb[t_train].astype(np.int64)

    tr_ds = LobGraphSequenceDatasetTwoHeadFixedH(
        X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_train, int(cfg["lookback"])
    )
    va_ds = LobGraphSequenceDatasetTwoHeadFixedH(
        X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_val, int(cfg["lookback"])
    )
    te_ds = LobGraphSequenceDatasetTwoHeadFixedH(
        X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_test, int(cfg["lookback"])
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate_fn_twohead,
        num_workers=0,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_twohead,
        num_workers=0,
    )

    model = GraphTemporalAttentionTwoHeadFixedH(
        node_in=int(X_scaled.shape[-1]),
        cfg=cfg,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
    ).to(DEVICE)

    pos_w_trade = compute_pos_weights_binary(ytr_train)
    bce_trade = nn.BCEWithLogitsLoss(pos_weight=pos_w_trade)

    mask_tr = (ytb_train != 1)
    ydir_train = (ytb_train[mask_tr] == 2).astype(np.int64)
    pos_w_dir = compute_pos_weights_binary(ydir_train) if ydir_train.size else torch.tensor([1.0], device=DEVICE)
    bce_dir = nn.BCEWithLogitsLoss(pos_weight=pos_w_dir)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    use_onecycle = bool(cfg.get("use_onecycle", True))
    if use_onecycle:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            steps_per_epoch=max(1, len(tr_loader)),
            pct_start=float(cfg.get("onecycle_pct_start", 0.20)),
            div_factor=float(cfg.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(cfg.get("onecycle_final_div", 200.0)),
        )
    else:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    b_dir = float(cfg.get("sel_b_dir_auc", 0.10))
    trade_pen = float(cfg.get("trade_prob_penalty", 0.0))

    best_sel = -1e18
    best_state = None
    best_epoch = -1
    patience = 7
    bad = 0

    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()

        tot = 0.0
        tot_tr = 0.0
        tot_dr = 0.0
        tot_h = 0.0
        tot_u = 0.0
        tot_us = 0.0
        n_ = 0

        for x, e, yt, yd, _er_exit, er_fixed, _sidx in tr_loader:
            x = x.to(DEVICE).float()
            e = e.to(DEVICE).float()
            yt = yt.to(DEVICE).float()
            yd = yd.to(DEVICE).float()
            er_fixed = er_fixed.to(DEVICE).float()

            opt.zero_grad(set_to_none=True)

            trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
            loss_mt, parts = multitask_loss_twohead_fixedH(
                trade_logit,
                dir_logit,
                fixed_hat,
                yt,
                yd,
                er_fixed,
                bce_trade,
                bce_dir,
                cfg,
            )

            if trade_pen > 0:
                loss_mt = loss_mt + trade_pen * parts["p_trade_mean"]

            loss = total_loss_with_adj_reg(loss_mt, aux, cfg)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            if use_onecycle:
                sch.step()

            B = int(yt.size(0))
            tot += float(loss.item()) * B
            tot_tr += float(parts["bce_trade"].item()) * B
            tot_dr += float(parts["bce_dir"].item()) * B
            tot_h += float(parts["huber"].item()) * B
            tot_u += float(parts["util"].item()) * B
            tot_us += float(parts["util_scaled"].item()) * B
            n_ += B

        tr_loss = tot / max(1, n_)
        tr_tr = tot_tr / max(1, n_)
        tr_dr = tot_dr / max(1, n_)
        tr_h = tot_h / max(1, n_)
        tr_u = tot_u / max(1, n_)
        tr_us = tot_us / max(1, n_)

        model.eval()
        v_tot = 0.0
        v_tr = 0.0
        v_dr = 0.0
        v_h = 0.0
        v_u = 0.0
        v_us = 0.0
        v_pa = 0.0
        v_ptr = 0.0
        v_n = 0

        p_trade_list = []
        p_dir_list = []
        y_trade_list = []

        for x, e, yt, yd, _er_exit, er_fixed, _sidx in va_loader:
            x = x.to(DEVICE).float()
            e = e.to(DEVICE).float()
            yt = yt.to(DEVICE).float()
            yd = yd.to(DEVICE).float()
            er_fixed = er_fixed.to(DEVICE).float()

            trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
            loss_mt, parts = multitask_loss_twohead_fixedH(
                trade_logit,
                dir_logit,
                fixed_hat,
                yt,
                yd,
                er_fixed,
                bce_trade,
                bce_dir,
                cfg,
            )
            loss_val = total_loss_with_adj_reg(loss_mt, aux, cfg)

            B = int(yt.size(0))
            v_tot += float(loss_val.item()) * B
            v_tr += float(parts["bce_trade"].item()) * B
            v_dr += float(parts["bce_dir"].item()) * B
            v_h += float(parts["huber"].item()) * B
            v_u += float(parts["util"].item()) * B
            v_us += float(parts["util_scaled"].item()) * B
            v_pa += float(parts["pos_abs"].item()) * B
            v_ptr += float(parts["p_trade_mean"].item()) * B
            v_n += B

            p_trade_list.append(torch.sigmoid(trade_logit).detach().cpu().numpy())
            p_dir_list.append(torch.sigmoid(dir_logit).detach().cpu().numpy())
            y_trade_list.append((yt.detach().cpu().numpy() > 0.5).astype(np.int64))

        p_trade_np = np.concatenate(p_trade_list, axis=0) if p_trade_list else np.zeros((0,), dtype=np.float64)
        p_dir_np = np.concatenate(p_dir_list, axis=0) if p_dir_list else np.zeros((0,), dtype=np.float64)
        y_trade_np = np.concatenate(y_trade_list, axis=0) if y_trade_list else np.zeros((0,), dtype=np.int64)

        t_val = sample_t[idx_val]
        y_tb_val = y_tb[t_val].astype(np.int64)
        trade_auc, dir_auc = compute_trade_dir_auc_from_twohead(y_trade_np, y_tb_val, p_trade_np, p_dir_np)

        val_loss = v_tot / max(1, v_n)
        val_tr = v_tr / max(1, v_n)
        val_dr = v_dr / max(1, v_n)
        val_h = v_h / max(1, v_n)
        val_u = v_u / max(1, v_n)
        val_us = v_us / max(1, v_n)
        val_pa = v_pa / max(1, v_n)
        val_ptr = v_ptr / max(1, v_n)

        sel = float(val_us) + b_dir * (float(dir_auc) if np.isfinite(dir_auc) else 0.0)

        if sel > best_sel:
            best_sel = sel
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if not use_onecycle:
            sch.step(sel)

        lr_now = opt.param_groups[0]["lr"]
        w_support = model.support_mix().detach().cpu().numpy().tolist()

        print(
            f"[fold {fold_id:02d}] ep {ep:02d} lr={lr_now:.2e} "
            f"tr_loss={tr_loss:.4f} (bceT={tr_tr:.4f}, bceD={tr_dr:.4f}, huber={tr_h:.4f}, util={tr_u:.5f}, utilS={tr_us:.5f}) "
            f"val_loss={val_loss:.4f} (bceT={val_tr:.4f}, bceD={val_dr:.4f}, huber={val_h:.4f}, util={val_u:.5f}, utilS={val_us:.5f}, posAbs={val_pa:.3f}, pT={val_ptr:.3f}) "
            f"val_trade_auc={trade_auc:.3f} val_dir_auc={dir_auc:.3f} sel={sel:.5f} "
            f"best={best_sel:.5f}@ep{best_epoch:02d} supports={np.round(w_support, 3).tolist()}"
        )

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_eval = eval_twohead_on_indices(model, X_scaled, edge_scaled, idx_val, bce_trade, bce_dir, cfg)
    test_eval = eval_twohead_on_indices(model, X_scaled, edge_scaled, idx_test, bce_trade, bce_dir, cfg)

    true_val_trade_rate = split_trade_ratio(idx_val, sample_t, y_trade)
    sweep_val = sweep_thresholds_3class(
        prob3=val_eval["prob3"],
        exit_ret_arr=val_eval["exit_ret"],
        cfg=cfg,
        min_trades=int(cfg["eval_min_trades"]),
        target_trade_rate=float(true_val_trade_rate),
    )
    best_thr = sweep_val.iloc[0].to_dict()
    thr_trade = float(best_thr["thr_trade"])
    thr_dir = float(best_thr["thr_dir"])

    pnl_val = pnl_from_probs_3class(val_eval["prob3"], val_eval["exit_ret"], thr_trade, thr_dir, cfg["cost_bps"])
    pnl_test = pnl_from_probs_3class(test_eval["prob3"], test_eval["exit_ret"], thr_trade, thr_dir, cfg["cost_bps"])

    print(
        f"[fold {fold_id:02d}] chosen thresholds on VAL: thr_trade={thr_trade:.3f} thr_dir={thr_dir:.3f} "
        f"| val pnl_sum={pnl_val['pnl_sum']:.4f} val trade_rate={pnl_val['trade_rate']:.3f}"
    )
    print(
        f"[fold {fold_id:02d}] TEST (fixed thresholds from VAL): "
        f"trade_auc={test_eval['trade_auc']:.3f} dir_auc={test_eval['dir_auc']:.3f} "
        f"soft_utilS={test_eval['soft_util_scaled_mean']:.5f} pnl_sum={pnl_test['pnl_sum']:.4f} "
        f"trade_rate={pnl_test['trade_rate']:.3f} trades={pnl_test['n_trades']}"
    )

    return {
        "fold": int(fold_id),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "node_scaler_params": node_scaler_params,
        "edge_scaler_params": edge_scaler_params,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "best_epoch": int(best_epoch),
        "best_sel": float(best_sel),
        "val_eval": val_eval,
        "test_eval": test_eval,
        "thr_trade": thr_trade,
        "thr_dir": thr_dir,
        "pnl_val": pnl_val,
        "pnl_test": pnl_test,
    }


# %% [markdown]
# This block runs walk-forward CV, saves per-fold bundles, and also materializes
# an `overall_best` bundle that simply copies the best validation-selected fold.
#
# The external workflow is intentionally preserved from the original notebook:
#
# - fit each fold independently
# - save the best state of each fold
# - summarize fold metrics
# - create a final `overall_best` bundle for later evaluation

# %%
# Step 12: Walk-forward CV run + saving per-fold bundles + overall best
# ======================================================================

def run_walk_forward_cv_twohead_fixedH() -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    fold_artifacts: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    best_overall_sel = -1e18
    best_overall_name: Optional[str] = None

    for fi, (idx_tr, idx_va, idx_te) in enumerate(walk_splits, 1):
        print("\n" + "=" * 90)
        print(f"FOLD {fi}/{len(walk_splits)} sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
        print(f"True trade ratio (val):  {split_trade_ratio(idx_va, sample_t, y_trade):.3f}")
        print(f"True trade ratio (test): {split_trade_ratio(idx_te, sample_t, y_trade):.3f}")

        X_scaled, node_params = fit_scale_nodes_train_only(
            X_node_raw,
            sample_t,
            idx_tr,
            max_abs=float(CFG["max_abs_feat"]),
        )
        if bool(CFG.get("edge_scale", True)):
            edge_scaled, edge_params = fit_scale_edges_train_only(
                edge_feat,
                sample_t,
                idx_tr,
                max_abs=float(CFG["max_abs_edge"]),
            )
        else:
            edge_scaled = edge_feat.astype(np.float32)
            edge_params = None

        artifact = train_one_fold_twohead_fixedH(
            fold_id=fi,
            X_scaled=X_scaled,
            edge_scaled=edge_scaled,
            idx_train=idx_tr,
            idx_val=idx_va,
            idx_test=idx_te,
            node_scaler_params=node_params,
            edge_scaler_params=edge_params,
            cfg=CFG,
        )

        fold_name = f"fold_{fi:02d}"
        extra_meta = {
            "kind": "fold_best",
            "fold": fi,
            "best_epoch": artifact["best_epoch"],
            "best_sel": artifact["best_sel"],
            "thr_trade": artifact["thr_trade"],
            "thr_dir": artifact["thr_dir"],
            "idx_train": artifact["idx_train"].tolist(),
            "idx_val": artifact["idx_val"].tolist(),
            "idx_test": artifact["idx_test"].tolist(),
        }
        saved = save_bundle(
            bundle_dir=ART_DIR,
            name=fold_name,
            model_state=artifact["model_state"],
            cfg=CFG,
            node_scaler_params=artifact["node_scaler_params"],
            edge_scaler_params=artifact["edge_scaler_params"],
            extra_meta=extra_meta,
        )
        print("Saved fold bundle:", saved["meta"].name)

        if float(artifact["best_sel"]) > best_overall_sel:
            best_overall_sel = float(artifact["best_sel"])
            best_overall_name = fold_name

        fold_artifacts.append(artifact)

        rows.append(
            {
                "fold": fi,
                "val_trade_auc": artifact["val_eval"]["trade_auc"],
                "val_dir_auc": artifact["val_eval"]["dir_auc"],
                "val_soft_utilS": artifact["val_eval"]["soft_util_scaled_mean"],
                "val_loss": artifact["val_eval"]["loss"],
                "test_trade_auc": artifact["test_eval"]["trade_auc"],
                "test_dir_auc": artifact["test_eval"]["dir_auc"],
                "test_soft_utilS": artifact["test_eval"]["soft_util_scaled_mean"],
                "test_loss": artifact["test_eval"]["loss"],
                "thr_trade": artifact["thr_trade"],
                "thr_dir": artifact["thr_dir"],
                "test_trade_rate_pred": artifact["pnl_test"]["trade_rate"],
                "test_pnl_sum": artifact["pnl_test"]["pnl_sum"],
                "test_n_trades": artifact["pnl_test"]["n_trades"],
                "best_sel": artifact["best_sel"],
            }
        )

    cv_summary = pd.DataFrame(rows)

    if best_overall_name is None:
        raise RuntimeError("No best fold was selected.")

    overall_name = "overall_best"
    best_bundle = load_bundle(ART_DIR, best_overall_name)

    extra_meta = {
        "kind": "overall_best",
        "source_name": best_overall_name,
        "source_fold": best_bundle["meta"].get("fold", None),
        "thr_trade": best_bundle["meta"]["thr_trade"],
        "thr_dir": best_bundle["meta"]["thr_dir"],
        "idx_train": best_bundle["meta"]["idx_train"],
        "idx_val": best_bundle["meta"]["idx_val"],
        "idx_test": best_bundle["meta"]["idx_test"],
    }
    save_bundle(
        bundle_dir=ART_DIR,
        name=overall_name,
        model_state=best_bundle["state"],
        cfg=CFG,
        node_scaler_params=best_bundle["node_scaler_params"],
        edge_scaler_params=best_bundle["edge_scaler_params"],
        extra_meta=extra_meta,
    )
    print("\nSaved overall best bundle as:", overall_name)

    return cv_summary, fold_artifacts, overall_name


cv_summary_twohead, fold_artifacts_twohead, overall_best_name = run_walk_forward_cv_twohead_fixedH()

print("\n" + "=" * 90)
print("CV summary (two-head; utility selection; TEST uses thresholds selected on VAL):")
print(cv_summary_twohead)
print("\nMeans:")
print(cv_summary_twohead.mean(numeric_only=True))


# %% [markdown]
# This block runs the diagnostic threshold-pair check once after CV finishes.
#
# The predictions are concatenated across folds and evaluated jointly. This keeps
# the original intent of the notebook while making threshold behavior easier to
# inspect after the full walk-forward run.

# %%
# Threshold check ONCE after CV finishes
# ======================================================================

_ = check_threshold_pairs_after_cv_once(
    fold_artifacts=fold_artifacts_twohead,
    cfg=CFG,
    split="test",
    top_k=20,
    save_csv_path=(ART_DIR / "threshold_check_after_cv_aggregate_TEST.csv"),
)


# %% [markdown]
# This block evaluates any saved bundle on an arbitrary index set, including the
# final holdout.
#
# It reconstructs the model, reapplies the saved scalers, recomputes losses and
# metrics, and optionally runs the threshold-pairs diagnostic check.
#
# This preserves the late-stage evaluation flow from the original notebook.

# %%
# Step 13: Evaluate a saved bundle on FINAL holdout
# ======================================================================

@torch.no_grad()
def evaluate_bundle_on_indices(
    bundle_dir: Path,
    name: str,
    indices: np.ndarray,
    label: str,
    do_threshold_check: bool = False,
    threshold_check_top_k: int = 20,
    threshold_check_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle = load_bundle(bundle_dir, name)
    cfg_loaded = bundle["meta"]["cfg"]

    model = GraphTemporalAttentionTwoHeadFixedH(
        node_in=int(X_node_raw.shape[-1]),
        cfg=cfg_loaded,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
    ).to(DEVICE)
    model.load_state_dict(bundle["state"])
    model.eval()

    X_scaled = apply_scaler_params(X_node_raw.astype(np.float32), bundle["node_scaler_params"])
    if bundle["edge_scaler_params"] is not None:
        E_scaled = apply_scaler_params(edge_feat.astype(np.float32), bundle["edge_scaler_params"])
    else:
        E_scaled = edge_feat.astype(np.float32)

    idx_train_saved = np.asarray(bundle["meta"]["idx_train"], dtype=np.int64)
    t_train = sample_t[idx_train_saved]
    ytr_train = y_trade[t_train].astype(np.int64)
    ytb_train = y_tb[t_train].astype(np.int64)

    pos_w_trade = compute_pos_weights_binary(ytr_train)
    bce_trade = nn.BCEWithLogitsLoss(pos_weight=pos_w_trade)

    mask_tr = (ytb_train != 1)
    ydir_train = (ytb_train[mask_tr] == 2).astype(np.int64)
    pos_w_dir = compute_pos_weights_binary(ydir_train) if ydir_train.size else torch.tensor([1.0], device=DEVICE)
    bce_dir = nn.BCEWithLogitsLoss(pos_weight=pos_w_dir)

    ev = eval_twohead_on_indices(
        model,
        X_scaled,
        E_scaled,
        indices.astype(np.int64),
        bce_trade,
        bce_dir,
        cfg_loaded,
    )

    thr_trade = float(bundle["meta"]["thr_trade"])
    thr_dir = float(bundle["meta"]["thr_dir"])
    pnl = pnl_from_probs_3class(ev["prob3"], ev["exit_ret"], thr_trade, thr_dir, float(cfg_loaded["cost_bps"]))

    print("\n" + "=" * 90)
    print(label)
    print(f"bundle: {name}")
    print(f"trade_auc={ev['trade_auc']:.3f} | dir_auc={ev['dir_auc']:.3f} | loss={ev['loss']:.4f}")
    print(
        f"soft_util={ev['soft_util_mean']:.6f} | soft_utilS={ev['soft_util_scaled_mean']:.5f} "
        f"| pos_abs={ev['pos_abs_mean']:.4f} | p_trade_mean={ev['p_trade_mean']:.3f}"
    )
    print(f"pnl_sum={pnl['pnl_sum']:.4f} | trade_rate={pnl['trade_rate']:.3f} | trades={pnl['n_trades']}")

    if do_threshold_check:
        _ = check_threshold_pairs_once(
            prob3=ev["prob3"],
            exit_ret_arr=ev["exit_ret"],
            cfg=cfg_loaded,
            label=f"THRESHOLD CHECK (bundle={name}) on: {label}",
            min_trades=int(cfg_loaded.get("eval_min_trades", 1)),
            target_trade_rate=None,
            top_k=threshold_check_top_k,
            save_csv_path=threshold_check_csv,
        )

    return {"eval": ev, "pnl": pnl}


holdout_indices = idx_final_test.astype(np.int64)
_ = evaluate_bundle_on_indices(
    ART_DIR,
    overall_best_name,
    holdout_indices,
    label="FINAL HOLDOUT using overall_best",
    do_threshold_check=False,
)


# %% [markdown]
# This block performs the production refit workflow.
#
# The behavior matches the original notebook:
#
# - train on the CV portion
# - reserve the last validation block inside CV for threshold selection
# - evaluate the resulting production model on the final holdout
# - save `production_best`
#
# A single threshold-pairs diagnostic check is then run once on the holdout,
# which preserves the "one check after production refit" behavior.

# %%
# Step 14: Production fit on CV -> select thresholds on val_final -> eval holdout
# ======================================================================

def production_fit_and_save() -> str:
    print("\n" + "=" * 90)
    print("PRODUCTION FIT: train on CV -> select thresholds on val_final -> eval on FINAL holdout")

    gap = int(CFG["split_purge"]) + int(CFG["split_embargo"])
    val_w = max(1, int(float(CFG["val_window_frac"]) * n_samples_cv))
    train_end = n_samples_cv - val_w - gap

    if train_end <= 0:
        raise ValueError("Production split is empty after applying gap and val window.")

    idx_train_final = np.arange(0, train_end, dtype=np.int64)
    idx_val_final = np.arange(train_end + gap, n_samples_cv, dtype=np.int64)
    idx_holdout = idx_final_test.astype(np.int64)

    print("Sizes:")
    print("  train_final:", len(idx_train_final))
    print("  val_final  :", len(idx_val_final))
    print("  holdout    :", len(idx_holdout))
    print(f"True trade ratio (val_final): {split_trade_ratio(idx_val_final, sample_t, y_trade):.3f}")
    print(f"True trade ratio (holdout):   {split_trade_ratio(idx_holdout, sample_t, y_trade):.3f}")

    X_scaled, node_params = fit_scale_nodes_train_only(
        X_node_raw,
        sample_t,
        idx_train_final,
        max_abs=float(CFG["max_abs_feat"]),
    )
    if bool(CFG.get("edge_scale", True)):
        edge_scaled, edge_params = fit_scale_edges_train_only(
            edge_feat,
            sample_t,
            idx_train_final,
            max_abs=float(CFG["max_abs_edge"]),
        )
    else:
        edge_scaled = edge_feat.astype(np.float32)
        edge_params = None

    artifact = train_one_fold_twohead_fixedH(
        fold_id=99,
        X_scaled=X_scaled,
        edge_scaled=edge_scaled,
        idx_train=idx_train_final,
        idx_val=idx_val_final,
        idx_test=idx_holdout,
        node_scaler_params=node_params,
        edge_scaler_params=edge_params,
        cfg=CFG,
    )

    production_name = "production_best"
    extra_meta = {
        "kind": "production_best",
        "fold": 99,
        "best_epoch": artifact["best_epoch"],
        "best_sel": artifact["best_sel"],
        "thr_trade": artifact["thr_trade"],
        "thr_dir": artifact["thr_dir"],
        "idx_train": artifact["idx_train"].tolist(),
        "idx_val": artifact["idx_val"].tolist(),
        "idx_test": artifact["idx_test"].tolist(),
    }
    save_bundle(
        bundle_dir=ART_DIR,
        name=production_name,
        model_state=artifact["model_state"],
        cfg=CFG,
        node_scaler_params=artifact["node_scaler_params"],
        edge_scaler_params=artifact["edge_scaler_params"],
        extra_meta=extra_meta,
    )

    print("\nSaved production bundle as:", production_name)

    _ = check_threshold_pairs_once(
        prob3=artifact["test_eval"]["prob3"],
        exit_ret_arr=artifact["test_eval"]["exit_ret"],
        cfg=CFG,
        label="THRESHOLD CHECK (AFTER PRODUCTION REFIT) | HOLDOUT",
        min_trades=int(CFG.get("eval_min_trades", 1)),
        target_trade_rate=None,
        top_k=20,
        save_csv_path=(ART_DIR / "threshold_check_after_production_holdout.csv"),
    )

    return production_name


production_best_name = production_fit_and_save()

_ = evaluate_bundle_on_indices(
    ART_DIR,
    production_best_name,
    idx_final_test.astype(np.int64),
    label="TEST-ONLY FROM PRODUCTION BUNDLE (holdout)",
    do_threshold_check=False,
)
