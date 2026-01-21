# %%
"""
Graph WaveNet / MTGNN-style model (adaptive adjacency + dilated gated TCN)
Two-head setup: trade? -> direction? + fixed-horizon return head.

Requested changes preserved:
1) Fixed-horizon return for regression + soft utility (TB labels remain as ground truth source).
2) Stronger influence of utility (scaling + higher weight).
3) Selection metric: sel = soft_utility_scaled_mean + b * dir_auc (utility emphasized).
4) Correct lead-lag edges for 1–10 minutes.

Logged each epoch:
- val_trade_auc and val_dir_auc
- losses and soft utility

Artifacts:
- .pt stores ONLY model weights (state_dict)
- scalers stored as .npz (no pickle)
- meta stored as .json
"""

# %%
# ======================================================================
# Step 0: Imports, seed, config (single unified block)
# ======================================================================

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

torch.set_num_threads(max(1, os.cpu_count() or 4))

CFG: Dict[str, Any] = {
    # data
    "freq": "1min",
    "data_dir": "../dataset",
    "final_test_frac": 0.10,

    # order book
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # walk-forward windows (in sample-space)
    "train_min_frac": 0.50,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,

    # scaling
    "max_abs_feat": 10.0,
    "max_abs_edge": 6.0,

    # correlations / graph
    "corr_windows": [6 * 5, 12 * 5, 24 * 5, 48 * 5, 84 * 5],  # 30m,1h,2h,4h,7h
    "corr_lags": list(range(0, 11)),  # lead-lag horizons: 0..10 minutes
    "edges_mode": "all_pairs",        # "manual" | "all_pairs"
    "edges": [("ADA", "BTC"), ("ADA", "ETH"), ("ETH", "BTC")],  # used if edges_mode="manual"
    "add_self_loops": True,
    "edge_transform": "fisher",       # "none" | "fisher"
    "edge_scale": True,

    # triple-barrier labels (source of ground truth)
    "tb_horizon": 1 * 30,
    "lookback": 4 * 12 * 5,
    "tb_pt_mult": 1.2,
    "tb_sl_mult": 1.1,
    "tb_min_barrier": 0.001,
    "tb_max_barrier": 0.006,

    # fixed-horizon return for regression + utility
    "fixed_horizon": 30,              # minutes: r_H(t) = sum lr_{t+1..t+H}
    "fixed_ret_clip": 0.02,           # clip future return for ret/utility stability

    # training
    "batch_size": 128,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "grad_clip": 1.0,
    "dropout": 0.15,

    # stability tricks
    "use_weighted_sampler": True,
    "use_onecycle": True,

    # model channels
    "gwn_residual_channels": 64,
    "gwn_dilation_channels": 64,
    "gwn_skip_channels": 128,
    "gwn_end_channels": 128,
    "gwn_blocks": 3,
    "gwn_layers_per_block": 2,
    "gwn_kernel_size": 2,

    # adaptive adjacency
    "adj_emb_dim": 8,
    "adj_temperature": 1.0,
    "adaptive_topk": 3,

    # adjacency regularization
    "adj_l1_lambda": 1e-3,
    "adj_prior_lambda": 1e-2,

    # prior adjacency from edge_attr (last timestep)
    "prior_use_abs": False,
    "prior_diag_boost": 1.0,
    "prior_row_normalize": True,

    # trading eval
    "cost_bps": 1.0,

    # threshold sweep grids (val only)
    "thr_trade_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "thr_dir_grid":   [0.50, 0.55, 0.60, 0.65, 0.70],
    "eval_min_trades": 50,
    "max_trade_rate_val": 0.65,
    "trade_rate_penalty": 0.10,
    "thr_objective": "pnl_sum",
    "proxy_target_trades": [50, 100, 200],

    # selection metric: sel = soft_utility_scaled_mean + b * dir_auc
    "sel_b_dir_auc": 0.10,

    # optional anti-overtrading penalty on mean p_trade
    "trade_prob_penalty": 0.01,

    # loss weights (stronger utility)
    "loss_w_trade": 1.0,              # BCE trade
    "loss_w_dir": 1.0,                # BCE direction (only on true trades)
    "loss_w_ret": 0.50,               # Huber fixed-horizon
    "loss_w_utility": 1.00,           # utility weight increased

    # regression / utility stability
    "ret_huber_delta": 0.01,
    "utility_k": 2.0,
    "utility_scale": 50.0,

    # artifact saving
    "artifact_dir": "./artifacts_gwnet_twohead_fixedH",
}

ASSETS = ["ADA", "BTC", "ETH"]
ASSET2IDX = {a: i for i, a in enumerate(ASSETS)}
TARGET_ASSET = "ETH"
TARGET_NODE = ASSET2IDX[TARGET_ASSET]

ART_DIR = Path(CFG["artifact_dir"])
ART_DIR.mkdir(parents=True, exist_ok=True)

print("Assets:", ASSETS, "| Target:", TARGET_ASSET)
print("Artifacts dir:", str(ART_DIR.resolve()))

# %%
# ======================================================================
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
EDGE_INDEX = torch.tensor([[ASSET2IDX[s], ASSET2IDX[t]] for (s, t) in EDGE_LIST], dtype=torch.long)

print("EDGE_LIST:", EDGE_NAMES)
print("EDGE_INDEX:", EDGE_INDEX.tolist())

# %%
# ======================================================================
# Step 1: Data loading
# ======================================================================

def load_asset(asset: str, freq: str, data_dir: Path, book_levels: int, part: Tuple[int, int] = (0, 80)) -> pd.DataFrame:
    path = data_dir / f"{asset}_{freq}.csv"
    df = pd.read_csv(path)
    df = df.iloc[int(len(df) * part[0] / 100): int(len(df) * part[1] / 100)]

    df["timestamp"] = pd.to_datetime(df["system_time"]).dt.round("min")
    df = df.sort_values("timestamp").set_index("timestamp")

    bid_cols = [f"bids_notional_{i}" for i in range(book_levels)]
    ask_cols = [f"asks_notional_{i}" for i in range(book_levels)]

    needed = ["midpoint", "spread", "buys", "sells"] + bid_cols + ask_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: missing columns in CSV: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return df[needed]


def load_all_assets() -> pd.DataFrame:
    freq = CFG["freq"]
    data_dir = Path(CFG["data_dir"])
    book_levels = CFG["book_levels"]

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

    df_ada = rename_cols(load_asset("ADA", freq, data_dir, book_levels, part=(0, 75)), "ADA")
    df_btc = rename_cols(load_asset("BTC", freq, data_dir, book_levels, part=(0, 75)), "BTC")
    df_eth = rename_cols(load_asset("ETH", freq, data_dir, book_levels, part=(0, 75)), "ETH")

    df = df_ada.join(df_btc).join(df_eth).reset_index()
    return df


df = load_all_assets()
for a in ASSETS:
    df[f"lr_{a}"] = np.log(df[a]).diff().fillna(0.0)

print("Loaded df:", df.shape)
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
print(df.head(2))

# %%
# ======================================================================
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
    No leakage: shift(lag>0) uses past of source.
    Self-loop edges a->a: constant 1.0.
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
                r = np.nan_to_num(r.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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

# %%
# ======================================================================
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
    thr = (vol * np.sqrt(horizon)).clip(lower=min_barrier, upper=max_barrier)

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


def fixed_horizon_future_return(lr: np.ndarray, H: int) -> np.ndarray:
    """
    r_H(t) = sum_{i=1..H} lr[t+i]
    For last H timesteps, return 0.0 (not used by sampling).
    """
    lr = np.asarray(lr, dtype=np.float64)
    T = lr.shape[0]
    out = np.zeros(T, dtype=np.float32)
    if H <= 0:
        return out
    for t in range(0, T - H - 1):
        out[t] = float(lr[t + 1: t + H + 1].sum())
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

y_trade = (y_tb != 1).astype(np.int64)                 # 1 if trade (SHORT or LONG)
y_dir = (y_tb == 2).astype(np.int64)                   # 1 if LONG, 0 if SHORT (meaningful only when y_trade=1)

fixed_ret = fixed_horizon_future_return(df["lr_ETH"].to_numpy(dtype=np.float64), int(CFG["fixed_horizon"]))

dist = np.bincount(y_tb, minlength=3)
print("TB dist [down,flat,up]:", dist)
print("True trade ratio:", float(y_trade.mean()))
print("fixed_ret stats: mean=", float(np.mean(fixed_ret)), "std=", float(np.std(fixed_ret)))

# %%
# ======================================================================
# Step 4: Node features
# ======================================================================

def safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def build_node_tensor(df_: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Features per asset:
      lr, spread,
      log_buys, log_sells, ofi,
      DI_15,
      DI_L0..DI_L4,
      near_ratio_bid, near_ratio_ask,
      di_near, di_far
    """
    book_levels = int(CFG["book_levels"])
    top_k = int(CFG["top_levels"])
    near_k = int(CFG["near_levels"])

    if near_k >= book_levels:
        raise ValueError("CFG['near_levels'] must be < CFG['book_levels']")

    feat_names = [
        "lr", "spread",
        "log_buys", "log_sells", "ofi",
        "DI_15",
        "DI_L0", "DI_L1", "DI_L2", "DI_L3", "DI_L4",
        "near_ratio_bid", "near_ratio_ask",
        "di_near", "di_far",
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

        bids_lvls = np.stack([df_[f"bids_vol_{a}_{i}"].values.astype(np.float32) for i in range(book_levels)], axis=1)
        asks_lvls = np.stack([df_[f"asks_vol_{a}_{i}"].values.astype(np.float32) for i in range(book_levels)], axis=1)

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

        Xa = np.column_stack([
            lr, spread,
            log_buys, log_sells, ofi,
            di_15,
            di_l0_4[:, 0], di_l0_4[:, 1], di_l0_4[:, 2], di_l0_4[:, 3], di_l0_4[:, 4],
            near_ratio_bid, near_ratio_ask,
            di_near, di_far,
        ]).astype(np.float32)

        feats_all.append(Xa)

    X = np.stack(feats_all, axis=1).astype(np.float32)  # (T,N,F)
    return X, feat_names


X_node_raw, node_feat_names = build_node_tensor(df)

T = len(df)
L = int(CFG["lookback"])
H_tb = int(CFG["tb_horizon"])
H_fixed = int(CFG["fixed_horizon"])

t_min = L - 1
t_max = T - max(H_tb, H_fixed) - 2
sample_t = np.arange(t_min, t_max + 1)
n_samples = len(sample_t)

print("X_node_raw:", X_node_raw.shape, "edge_feat:", edge_feat.shape)
print("n_samples:", n_samples, "| t range:", int(sample_t[0]), "->", int(sample_t[-1]))

# %%
# ======================================================================
# Step 5: Splits
# ======================================================================

def make_final_holdout_split(n_samples_: int, final_test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < final_test_frac < 0.5):
        raise ValueError("final_test_frac should be in (0, 0.5)")
    n_final = max(1, int(round(final_test_frac * n_samples_)))
    n_cv = n_samples_ - n_final
    if n_cv <= 50:
        raise ValueError("Too few samples left for CV after holdout split.")
    idx_cv = np.arange(0, n_cv, dtype=np.int64)
    idx_final = np.arange(n_cv, n_samples_, dtype=np.int64)
    return idx_cv, idx_final


def make_walk_forward_splits(
    n_samples_: int,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    step_window_frac: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_min = int(train_min_frac * n_samples_)
    val_w = max(1, int(val_window_frac * n_samples_))
    test_w = max(1, int(test_window_frac * n_samples_))
    step_w = max(1, int(step_window_frac * n_samples_))

    splits = []
    start = train_min
    while True:
        tr_end = start
        va_end = tr_end + val_w
        te_end = va_end + test_w
        if te_end > n_samples_:
            break

        idx_train = np.arange(0, tr_end, dtype=np.int64)
        idx_val = np.arange(tr_end, va_end, dtype=np.int64)
        idx_test = np.arange(va_end, te_end, dtype=np.int64)
        splits.append((idx_train, idx_val, idx_test))

        start += step_w

    return splits


idx_cv_all, idx_final_test = make_final_holdout_split(n_samples, float(CFG["final_test_frac"]))
n_samples_cv = len(idx_cv_all)

walk_splits = make_walk_forward_splits(
    n_samples_=n_samples_cv,
    train_min_frac=float(CFG["train_min_frac"]),
    val_window_frac=float(CFG["val_window_frac"]),
    test_window_frac=float(CFG["test_window_frac"]),
    step_window_frac=float(CFG["step_window_frac"]),
)

print("Holdout split:")
print(f"  n_samples total: {n_samples}")
print(f"  n_samples CV   : {len(idx_cv_all)}")
print(f"  n_samples FINAL: {len(idx_final_test)}")
print("\nWalk-forward folds:", len(walk_splits))
for i, (a, b, c) in enumerate(walk_splits, 1):
    print(f"  fold {i}: train={len(a)} | val={len(b)} | test={len(c)}")

# %%
# ======================================================================
# Step 6: Dataset, scaling helpers, sampler
# ======================================================================

class LobGraphSequenceDatasetTwoHeadFixedH(Dataset):
    """
    Returns:
      x_seq:      (L,N,F)
      e_seq:      (L,E,D)
      y_trade:    scalar in {0,1}
      y_dir:      scalar in {0,1} (meaningful only when y_trade=1)
      exit_ret:   scalar (TB realized return to exit) - for threshold sweep / PnL eval
      fixed_ret:  scalar (fixed-horizon future return) - for regression + utility
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

        x_seq = self.X_node[t0:t + 1]
        e_seq = self.E_feat[t0:t + 1]

        yt = int(self.y_trade[t])
        yd = int(self.y_dir[t])

        er_exit = float(self.exit_ret[t])
        er_fixed = float(self.fixed_ret[t])

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(e_seq),
            torch.tensor(yt, dtype=torch.float32),
            torch.tensor(yd, dtype=torch.float32),
            torch.tensor(er_exit, dtype=torch.float32),
            torch.tensor(er_fixed, dtype=torch.float32),
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


def make_weighted_sampler_trade(y_trade_np: np.ndarray) -> WeightedRandomSampler:
    """
    Weighted sampler for binary trade label.
    """
    y = np.asarray(y_trade_np, dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    sample_w = w[y].astype(np.float64)
    return WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), num_samples=len(sample_w), replacement=True)


def fit_scale_nodes_train_only(X_node_raw_: np.ndarray, sample_t_: np.ndarray, idx_train: np.ndarray, max_abs: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    X_train_time = X_node_raw_[train_time_mask]
    _, _, Fdim = X_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(X_train_time.reshape(-1, Fdim))

    X_scaled = scaler.transform(X_node_raw_.reshape(-1, Fdim)).reshape(X_node_raw_.shape).astype(np.float32)
    X_scaled = np.clip(X_scaled, -max_abs, max_abs).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {"center_": scaler.center_.astype(np.float32), "scale_": scaler.scale_.astype(np.float32), "max_abs": float(max_abs)}
    return X_scaled, params


def fit_scale_edges_train_only(E_raw_: np.ndarray, sample_t_: np.ndarray, idx_train: np.ndarray, max_abs: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    E_train_time = E_raw_[train_time_mask]
    _, _, D = E_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(E_train_time.reshape(-1, D))

    E_scaled = scaler.transform(E_raw_.reshape(-1, D)).reshape(E_raw_.shape).astype(np.float32)
    E_scaled = np.clip(E_scaled, -max_abs, max_abs).astype(np.float32)
    E_scaled = np.nan_to_num(E_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {"center_": scaler.center_.astype(np.float32), "scale_": scaler.scale_.astype(np.float32), "max_abs": float(max_abs)}
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

# %%
# ======================================================================
# Step 7: Graph WaveNet model (trade logit + dir logit + fixed_ret_hat)
# ======================================================================

def build_static_adjacency_from_edges(edge_index: torch.Tensor, n_nodes: int, eps: float = 1e-8) -> torch.Tensor:
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
    B, E, D = edge_attr_last.shape
    r = edge_attr_last.mean(dim=-1)
    if use_abs:
        r = r.abs()
    w = torch.sigmoid(r)

    A = torch.zeros((B, n_nodes, n_nodes), device=edge_attr_last.device, dtype=edge_attr_last.dtype)
    src = edge_index[:, 0].to(edge_attr_last.device)
    dst = edge_index[:, 1].to(edge_attr_last.device)
    A[:, src, dst] = w

    diag = torch.arange(n_nodes, device=edge_attr_last.device)
    A[:, diag, diag] = torch.maximum(A[:, diag, diag], torch.full_like(A[:, diag, diag], float(diag_boost)))

    if row_normalize:
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

    return torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes: int, cfg: Dict[str, Any]):
        super().__init__()
        self.n = int(n_nodes)
        k = int(cfg.get("adj_emb_dim", 8))
        self.E1 = nn.Parameter(0.01 * torch.randn(self.n, k))
        self.E2 = nn.Parameter(0.01 * torch.randn(self.n, k))
        temp = float(cfg.get("adj_temperature", 1.0))
        self.temp = max(temp, 1e-3)
        self.topk = int(cfg.get("adaptive_topk", self.n))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = (self.E1 @ self.E2.t())
        logits = F.relu(logits) / self.temp

        if 0 < self.topk < self.n:
            vals, idx = torch.topk(logits, k=self.topk, dim=-1)
            mask = torch.full_like(logits, fill_value=float("-inf"))
            mask.scatter_(-1, idx, vals)
            logits = mask

        A = torch.softmax(logits, dim=-1)
        sparsity_proxy = torch.sigmoid(logits)
        return A, sparsity_proxy, logits


class LearnableSupportMix(nn.Module):
    def __init__(self, n_supports: int = 3):
        super().__init__()
        self.w_logits = nn.Parameter(torch.zeros(n_supports, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.w_logits, dim=0)


class CausalConv2dTime(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.k = int(kernel_size)
        self.d = int(dilation)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, self.k), dilation=(1, self.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0, 0, 0))
        return self.conv(x)


def graph_message_passing(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bcnt,bnm->bcmt", x, A)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, residual_ch: int, dilation_ch: int, skip_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.filter_conv = CausalConv2dTime(residual_ch, dilation_ch, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = CausalConv2dTime(residual_ch, dilation_ch, kernel_size=kernel_size, dilation=dilation)

        self.residual_conv = nn.Conv2d(dilation_ch, residual_ch, kernel_size=(1, 1))
        self.skip_conv = nn.Conv2d(dilation_ch, skip_ch, kernel_size=(1, 1))

        self.dropout = nn.Dropout(float(dropout))
        self.bn = nn.BatchNorm2d(residual_ch)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        residual = x

        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = f * g
        z = self.dropout(z)

        skip = self.skip_conv(z)
        out = self.residual_conv(z)
        out = graph_message_passing(out, A)
        out = out + residual
        out = self.bn(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        skip = torch.nan_to_num(skip, nan=0.0, posinf=0.0, neginf=0.0)
        return out, skip


class GraphWaveNetTwoHeadFixedH(nn.Module):
    """
    Input:
      x_seq: (B,L,N,F)
      e_seq: (B,L,E,D) used only for building A_prior from last step

    Output (target node):
      trade_logit: (B,) trade probability logit
      dir_logit:   (B,) direction logit (prob LONG given trade)
      fixed_hat:   (B,) predicted fixed-horizon future return
    """
    def __init__(self, node_in: int, edge_dim: int, cfg: Dict[str, Any], n_nodes: int, target_node: int):
        super().__init__()
        self.cfg = cfg
        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)

        residual_ch = int(cfg["gwn_residual_channels"])
        dilation_ch = int(cfg["gwn_dilation_channels"])
        skip_ch = int(cfg["gwn_skip_channels"])
        end_ch = int(cfg["gwn_end_channels"])
        k = int(cfg["gwn_kernel_size"])
        blocks = int(cfg["gwn_blocks"])
        layers_per_block = int(cfg["gwn_layers_per_block"])
        drop = float(cfg.get("dropout", 0.0))

        self.in_proj = nn.Linear(int(node_in), residual_ch)

        A_static = build_static_adjacency_from_edges(EDGE_INDEX, n_nodes=self.n_nodes)
        self.register_buffer("A_static", A_static)

        self.adapt = AdaptiveAdjacency(n_nodes=self.n_nodes, cfg=cfg)
        self.support_mix = LearnableSupportMix(n_supports=3)

        self.blocks = nn.ModuleList()
        for _b in range(blocks):
            for l in range(layers_per_block):
                dilation = 2 ** l
                self.blocks.append(GraphWaveNetBlock(
                    residual_ch=residual_ch,
                    dilation_ch=dilation_ch,
                    skip_ch=skip_ch,
                    kernel_size=k,
                    dilation=dilation,
                    dropout=drop,
                ))

        self.end1 = nn.Conv2d(skip_ch, end_ch, kernel_size=(1, 1))
        self.trade_head = nn.Linear(end_ch, 1)
        self.dir_head = nn.Linear(end_ch, 1)
        self.fixed_head = nn.Linear(end_ch, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_supports(self, e_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, L_, E, D = e_seq.shape
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
        offdiag = (1.0 - torch.eye(N, device=e_seq.device, dtype=e_seq.dtype))
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

        B, L_, N, Fdim = x_seq.shape
        assert N == self.n_nodes

        x = self.in_proj(x_seq)                 # (B,L,N,C)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,C,N,T)

        A_mix, aux = self._compute_supports(e_seq)

        skip_sum = None
        for blk in self.blocks:
            x, skip = blk(x, A_mix)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        y = F.relu(skip_sum)
        y_end = F.relu(self.end1(y))            # (B,end_ch,N,T)
        feat = y_end[:, :, self.target_node, -1]  # (B,end_ch)

        trade_logit = self.trade_head(feat).squeeze(-1)
        dir_logit = self.dir_head(feat).squeeze(-1)
        fixed_hat = self.fixed_head(feat).squeeze(-1)

        trade_logit = torch.nan_to_num(trade_logit, nan=0.0, posinf=0.0, neginf=0.0)
        dir_logit = torch.nan_to_num(dir_logit, nan=0.0, posinf=0.0, neginf=0.0)
        fixed_hat = torch.nan_to_num(fixed_hat, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            return trade_logit, dir_logit, fixed_hat, aux
        return trade_logit, dir_logit, fixed_hat

# %%
# ======================================================================
# Step 8: Metrics, thresholding, losses (two-head + fixed-horizon utility)
# ======================================================================

def _safe_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def probs3_from_twohead(p_trade: np.ndarray, p_dir: np.ndarray) -> np.ndarray:
    """
    Convert two-head outputs to a 3-class distribution:
      p_flat = 1 - p_trade
      p_long = p_trade * p_dir
      p_short = p_trade * (1 - p_dir)
    """
    p_trade = np.asarray(p_trade, dtype=np.float64)
    p_dir = np.asarray(p_dir, dtype=np.float64)
    p_flat = 1.0 - p_trade
    p_long = p_trade * p_dir
    p_short = p_trade * (1.0 - p_dir)
    prob3 = np.stack([p_short, p_flat, p_long], axis=1)
    prob3 = np.clip(prob3, 1e-12, 1.0)
    prob3 = prob3 / prob3.sum(axis=1, keepdims=True)
    return prob3


def compute_trade_dir_auc_from_twohead(y_trade_true: np.ndarray, y_tb_true: np.ndarray, p_trade: np.ndarray, p_dir: np.ndarray) -> Tuple[float, float]:
    """
    trade_auc: y_trade_true vs p_trade
    dir_auc: LONG vs SHORT only on true trades; score = p_dir
    """
    y_trade_true = np.asarray(y_trade_true, dtype=np.int64)
    y_tb_true = np.asarray(y_tb_true, dtype=np.int64)
    p_trade = np.asarray(p_trade, dtype=np.float64)
    p_dir = np.asarray(p_dir, dtype=np.float64)

    trade_auc = _safe_auc_binary(y_trade_true, p_trade)

    mask_trade = (y_tb_true != 1)
    y_dir_bin = (y_tb_true[mask_trade] == 2).astype(np.int64)
    dir_auc = _safe_auc_binary(y_dir_bin, p_dir[mask_trade])

    return trade_auc, dir_auc


def pnl_from_probs_3class(prob3: np.ndarray, exit_ret_arr: np.ndarray, thr_trade: float, thr_dir: float, cost_bps: float) -> Dict[str, Any]:
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


def build_trade_threshold_grid(p_trade: np.ndarray, base_grid: Optional[List[float]], target_trades_list: Optional[List[int]]) -> List[float]:
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
    cleaned = []
    for t in out:
        if not cleaned or abs(t - cleaned[-1]) > 1e-6:
            cleaned.append(float(t))
    return cleaned


def sweep_thresholds_3class(prob3: np.ndarray, exit_ret_arr: np.ndarray, cfg: Dict[str, Any], min_trades: int, target_trade_rate: Optional[float]) -> pd.DataFrame:
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
        return sweep_thresholds_3class(prob3, exit_ret_arr, cfg, min_trades=1, target_trade_rate=target_trade_rate)

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
    """
    pos_weight for BCEWithLogitsLoss:
      pos_weight = n_neg / n_pos
    """
    y = np.asarray(y, dtype=np.int64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    n_pos = max(n_pos, 1.0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)


def multitask_loss_twohead_fixedH(
    trade_logit: torch.Tensor,
    dir_logit: torch.Tensor,
    fixed_hat: torch.Tensor,
    y_trade_batch: torch.Tensor,
    y_dir_batch: torch.Tensor,
    fixed_ret_batch: torch.Tensor,
    bce_trade: nn.Module,
    bce_dir: nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Total = w_trade*BCE(trade) + w_dir*BCE(dir | trade) + w_ret*Huber + w_util*(-scaled_utility)

    Utility uses fixed_ret:
      p_trade = sigmoid(trade_logit)
      p_dir   = sigmoid(dir_logit)
      signed_dir = 2*p_dir - 1 in [-1,1]
      pos = p_trade * tanh(k*signed_dir)
      utility = pos * fixed_ret - fee * |pos|
      utility_scaled = utility_scale * utility
      util_loss = -mean(utility_scaled)
    """
    w_tr = float(cfg["loss_w_trade"])
    w_dr = float(cfg["loss_w_dir"])
    w_ret = float(cfg["loss_w_ret"])
    w_ut = float(cfg["loss_w_utility"])

    loss_trade = bce_trade(trade_logit, y_trade_batch)

    mask = (y_trade_batch > 0.5)
    if mask.any():
        loss_dir = bce_dir(dir_logit[mask], y_dir_batch[mask])
    else:
        loss_dir = torch.zeros((), device=trade_logit.device)

    clip_val = float(cfg.get("fixed_ret_clip", 0.0))
    fr = torch.clamp(fixed_ret_batch, -clip_val, clip_val) if (clip_val and clip_val > 0) else fixed_ret_batch

    delta = float(cfg.get("ret_huber_delta", 0.01))
    loss_ret = F.huber_loss(fixed_hat, fr, delta=delta)

    p_trade = torch.sigmoid(trade_logit)
    p_dir = torch.sigmoid(dir_logit)
    signed_dir = 2.0 * p_dir - 1.0
    k = float(cfg.get("utility_k", 2.0))
    pos = p_trade * torch.tanh(k * signed_dir)

    fee = float(cfg.get("cost_bps", 0.0)) * 1e-4
    utility = pos * fr - fee * torch.abs(pos)

    util_scale = float(cfg.get("utility_scale", 1.0))
    utility_scaled = util_scale * utility
    util_loss = -utility_scaled.mean()

    total = w_tr * loss_trade + w_dr * loss_dir + w_ret * loss_ret + w_ut * util_loss

    parts = {
        "bce_trade": loss_trade.detach(),
        "bce_dir": loss_dir.detach(),
        "huber": loss_ret.detach(),
        "util": utility.mean().detach(),
        "util_scaled": utility_scaled.mean().detach(),
        "util_loss": util_loss.detach(),
        "pos_abs": torch.abs(pos).mean().detach(),
        "p_trade_mean": p_trade.mean().detach(),
    }
    return total, parts

# %%
# ======================================================================
# Step 9: Evaluation (two-head -> prob3 for thresholds, log AUCs + utility)
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
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_twohead, num_workers=0)

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
    y_tb_all = []
    exit_all = []

    for x, e, yt, yd, er_exit, er_fixed, _sidx in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        yt = yt.to(DEVICE).float()
        yd = yd.to(DEVICE).float()
        er_fixed = er_fixed.to(DEVICE).float()

        trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
        loss, parts = multitask_loss_twohead_fixedH(trade_logit, dir_logit, fixed_hat, yt, yd, er_fixed, bce_trade, bce_dir, cfg)
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

        p_trade = torch.sigmoid(trade_logit).detach().cpu().numpy()
        p_dir = torch.sigmoid(dir_logit).detach().cpu().numpy()

        p_trade_all.append(p_trade)
        p_dir_all.append(p_dir)

        y_trade_all.append(yt.detach().cpu().numpy())

        # TB label reconstructed from y_trade/y_dir is not possible for flat; keep original via index
        # Here we recover TB labels by mapping sidx->t->y_tb outside the loader:
        # We use exit_ret tensor order and dataset indices; easiest: compute y_tb in dataset using sample_t.
        # To keep eval simple, recompute y_tb from current indices and loader batch positions:
        # We'll store TB labels from dataset inside the batch by reconstructing using sidx in the dataset.
        # (Here we hack: use er_exit length to slice TB labels by advancing pointer.)
        # Instead, we do an exact method by re-evaluating TB labels from the same indices outside loop later.
        exit_all.append(er_exit.detach().cpu().numpy())

    p_trade_np = np.concatenate(p_trade_all, axis=0) if p_trade_all else np.zeros((0,), dtype=np.float64)
    p_dir_np = np.concatenate(p_dir_all, axis=0) if p_dir_all else np.zeros((0,), dtype=np.float64)
    y_trade_np = (np.concatenate(y_trade_all, axis=0) > 0.5).astype(np.int64) if y_trade_all else np.zeros((0,), dtype=np.int64)
    er_exit_np = np.concatenate(exit_all, axis=0) if exit_all else np.zeros((0,), dtype=np.float64)

    # Exact TB labels for these indices:
    t_idx = sample_t[indices.astype(np.int64)]
    y_tb_np = y_tb[t_idx].astype(np.int64)

    trade_auc, dir_auc = compute_trade_dir_auc_from_twohead(y_trade_np, y_tb_np, p_trade_np, p_dir_np)

    prob3 = probs3_from_twohead(p_trade_np, p_dir_np)

    out = {
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
    return out

# %%
# ======================================================================
# Step 10: Artifact saving/loading (weights .pt + scaler .npz + meta .json)
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
) -> Dict[str, Path]:
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

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    node_scaler_params = load_scaler_npz(node_scaler_path)
    edge_scaler_params = load_scaler_npz(edge_scaler_path) if edge_scaler_path else None

    return {
        "meta": meta,
        "state": state,
        "node_scaler_params": node_scaler_params,
        "edge_scaler_params": edge_scaler_params,
    }

# %%
# ======================================================================
# Step 11: Train one fold (two-head)
# Selection metric: sel = soft_util_scaled_mean + b * dir_auc
# Still log trade_auc and dir_auc each epoch
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

    tr_ds = LobGraphSequenceDatasetTwoHeadFixedH(X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_train, int(cfg["lookback"]))
    va_ds = LobGraphSequenceDatasetTwoHeadFixedH(X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_val,   int(cfg["lookback"]))
    te_ds = LobGraphSequenceDatasetTwoHeadFixedH(X_scaled, edge_scaled, y_trade, y_dir, exit_ret, fixed_ret, sample_t, idx_test,  int(cfg["lookback"]))

    sampler = None
    shuffle = True
    if bool(cfg.get("use_weighted_sampler", True)):
        sampler = make_weighted_sampler_trade(ytr_train)
        shuffle = False

    tr_loader = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), shuffle=shuffle, sampler=sampler, collate_fn=collate_fn_twohead, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_twohead, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_twohead, num_workers=0)

    model = GraphWaveNetTwoHeadFixedH(
        node_in=int(X_scaled.shape[-1]),
        edge_dim=int(edge_scaled.shape[-1]),
        cfg=cfg,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
    ).to(DEVICE)

    # BCE losses with pos_weight for class imbalance
    pos_w_trade = compute_pos_weights_binary(ytr_train)
    bce_trade = nn.BCEWithLogitsLoss(pos_weight=pos_w_trade)

    # Direction pos_weight computed on trade samples only
    mask_tr = (ytb_train != 1)
    ydir_train = (ytb_train[mask_tr] == 2).astype(np.int64)
    pos_w_dir = compute_pos_weights_binary(ydir_train) if ydir_train.size else torch.tensor([1.0], device=DEVICE)
    bce_dir = nn.BCEWithLogitsLoss(pos_weight=pos_w_dir)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    use_onecycle = bool(cfg.get("use_onecycle", True))
    if use_onecycle:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            steps_per_epoch=max(1, len(tr_loader)),
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=50.0,
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
        n = 0

        for x, e, yt, yd, _er_exit, er_fixed, _sidx in tr_loader:
            x = x.to(DEVICE).float()
            e = e.to(DEVICE).float()
            yt = yt.to(DEVICE).float()
            yd = yd.to(DEVICE).float()
            er_fixed = er_fixed.to(DEVICE).float()

            opt.zero_grad(set_to_none=True)

            trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
            loss_mt, parts = multitask_loss_twohead_fixedH(
                trade_logit, dir_logit, fixed_hat,
                yt, yd, er_fixed,
                bce_trade, bce_dir,
                cfg
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
            n += B

        tr_loss = tot / max(1, n)
        tr_tr = tot_tr / max(1, n)
        tr_dr = tot_dr / max(1, n)
        tr_h = tot_h / max(1, n)
        tr_u = tot_u / max(1, n)
        tr_us = tot_us / max(1, n)

        # Validation
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
        exit_list = []

        for x, e, yt, yd, er_exit, er_fixed, _sidx in va_loader:
            x = x.to(DEVICE).float()
            e = e.to(DEVICE).float()
            yt = yt.to(DEVICE).float()
            yd = yd.to(DEVICE).float()
            er_fixed = er_fixed.to(DEVICE).float()

            trade_logit, dir_logit, fixed_hat, aux = model(x, e, return_aux=True)
            loss_mt, parts = multitask_loss_twohead_fixedH(
                trade_logit, dir_logit, fixed_hat,
                yt, yd, er_fixed,
                bce_trade, bce_dir,
                cfg
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
            exit_list.append(er_exit.detach().cpu().numpy())

        p_trade_np = np.concatenate(p_trade_list, axis=0) if p_trade_list else np.zeros((0,), dtype=np.float64)
        p_dir_np = np.concatenate(p_dir_list, axis=0) if p_dir_list else np.zeros((0,), dtype=np.float64)
        y_trade_np = np.concatenate(y_trade_list, axis=0) if y_trade_list else np.zeros((0,), dtype=np.int64)
        er_exit_np = np.concatenate(exit_list, axis=0) if exit_list else np.zeros((0,), dtype=np.float64)

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

    # Eval with best epoch model
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

# %%
# ======================================================================
# Step 12: Walk-forward CV run + saving per-fold bundles + overall best
# ======================================================================

def run_walk_forward_cv_twohead_fixedH() -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    fold_artifacts: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    best_overall_sel = -1e18
    best_overall_name = None

    for fi, (idx_tr, idx_va, idx_te) in enumerate(walk_splits, 1):
        print("\n" + "=" * 90)
        print(f"FOLD {fi}/{len(walk_splits)} sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
        print(f"True trade ratio (val):  {split_trade_ratio(idx_va, sample_t, y_trade):.3f}")
        print(f"True trade ratio (test): {split_trade_ratio(idx_te, sample_t, y_trade):.3f}")

        X_scaled, node_params = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_tr, max_abs=float(CFG["max_abs_feat"]))
        if bool(CFG.get("edge_scale", True)):
            edge_scaled, edge_params = fit_scale_edges_train_only(edge_feat, sample_t, idx_tr, max_abs=float(CFG["max_abs_edge"]))
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

        rows.append({
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
        })

    cv_summary = pd.DataFrame(rows)

    assert best_overall_name is not None
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

# %%
# ======================================================================
# Step 13: Evaluate a saved bundle on FINAL holdout
# ======================================================================

@torch.no_grad()
def evaluate_bundle_on_indices(bundle_dir: Path, name: str, indices: np.ndarray, label: str) -> Dict[str, Any]:
    bundle = load_bundle(bundle_dir, name)
    cfg = bundle["meta"]["cfg"]

    model = GraphWaveNetTwoHeadFixedH(
        node_in=int(X_node_raw.shape[-1]),
        edge_dim=int(edge_feat.shape[-1]),
        cfg=cfg,
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

    ev = eval_twohead_on_indices(model, X_scaled, E_scaled, indices.astype(np.int64), bce_trade, bce_dir, cfg)

    thr_trade = float(bundle["meta"]["thr_trade"])
    thr_dir = float(bundle["meta"]["thr_dir"])
    pnl = pnl_from_probs_3class(ev["prob3"], ev["exit_ret"], thr_trade, thr_dir, float(cfg["cost_bps"]))

    print("\n" + "=" * 90)
    print(label)
    print(f"bundle: {name}")
    print(f"trade_auc={ev['trade_auc']:.3f} | dir_auc={ev['dir_auc']:.3f} | loss={ev['loss']:.4f}")
    print(f"soft_util={ev['soft_util_mean']:.6f} | soft_utilS={ev['soft_util_scaled_mean']:.5f} | pos_abs={ev['pos_abs_mean']:.4f} | p_trade_mean={ev['p_trade_mean']:.3f}")
    print(f"pnl_sum={pnl['pnl_sum']:.4f} | trade_rate={pnl['trade_rate']:.3f} | trades={pnl['n_trades']}")
    return {"eval": ev, "pnl": pnl}


holdout_indices = idx_final_test.astype(np.int64)
_ = evaluate_bundle_on_indices(ART_DIR, overall_best_name, holdout_indices, label="FINAL HOLDOUT (10%) using overall_best")

# %%
# ======================================================================
# Step 14: Production fit on CV(90%) -> select thresholds on val_final -> eval holdout(10%)
# ======================================================================

def production_fit_and_save() -> str:
    print("\n" + "=" * 90)
    print("PRODUCTION FIT: train on CV(90%) -> select thresholds on val_final -> eval on FINAL holdout(10%)")

    val_w = max(1, int(float(CFG["val_window_frac"]) * n_samples_cv))
    train_end = n_samples_cv - val_w

    idx_train_final = np.arange(0, train_end, dtype=np.int64)
    idx_val_final = np.arange(train_end, n_samples_cv, dtype=np.int64)
    idx_holdout = idx_final_test.astype(np.int64)

    print("Sizes:")
    print("  train_final:", len(idx_train_final))
    print("  val_final  :", len(idx_val_final))
    print("  holdout    :", len(idx_holdout))
    print(f"True trade ratio (val_final): {split_trade_ratio(idx_val_final, sample_t, y_trade):.3f}")
    print(f"True trade ratio (holdout):   {split_trade_ratio(idx_holdout, sample_t, y_trade):.3f}")

    X_scaled, node_params = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_train_final, max_abs=float(CFG["max_abs_feat"]))
    if bool(CFG.get("edge_scale", True)):
        edge_scaled, edge_params = fit_scale_edges_train_only(edge_feat, sample_t, idx_train_final, max_abs=float(CFG["max_abs_edge"]))
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

    prod_name = "production_best"
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
        name=prod_name,
        model_state=artifact["model_state"],
        cfg=CFG,
        node_scaler_params=artifact["node_scaler_params"],
        edge_scaler_params=artifact["edge_scaler_params"],
        extra_meta=extra_meta,
    )
    print("\nSaved production bundle as:", prod_name)
    return prod_name


production_name = production_fit_and_save()
prod_bundle = load_bundle(ART_DIR, production_name)
idx_eval = np.asarray(prod_bundle["meta"]["idx_test"], dtype=np.int64)
_ = evaluate_bundle_on_indices(ART_DIR, production_name, idx_eval, label="TEST-ONLY FROM PRODUCTION BUNDLE (holdout)")
