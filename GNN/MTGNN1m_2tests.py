# %%
"""
Two-stage LOB GNN — MTGNN-style experiment
VS Code / Jupyter compatible (.py) via #%% cells

Key principles:
- Train-only scaling (time-ordered, no leakage)
- Stage A: trade / no-trade (AUC)
- Stage B: direction on trade-only (AUC on trade-only)
- Thresholds (thr_trade, thr_dir) are selected ONLY on val, never tuned on test/holdout

This notebook implements:
- Learnable adjacency (A_learned) + prior adjacency from edge_attr (A_prior)
- Regularization:
  (a) L1 on off-diagonal (implemented on sigmoid(adj_logits) for meaningful sparsity pressure)
  (b) penalty for deviation from A_prior (MSE on adjacency)
- Final adjacency:
  A_final = alpha * A_prior + (1 - alpha) * A_learned
  alpha is fixed or learned (clipped), controlled by CFG
- Temporal block:
  Conv (dilated) -> Attention pooling over time

New (per your comment):
- We DO NOT use mean/median thresholds as the default "final check".
- Step 10 now stores per-fold artifacts (models + thresholds + val preds).
- Step 11 implements 3 post-CV holdout checks WITHOUT any extra refit:
    1) LAST fold model + LAST fold thresholds
    2) BEST-VAL fold model + BEST-VAL thresholds
    3) LAST fold model + GLOBAL thresholds fitted on concatenated fold-VAL predictions
- We fix the variable naming: m_trade_last / m_dir_last are explicitly saved in Step 10.
- Step 12 is production-fit: train on CV(90%) with final val window, then evaluate on FINAL holdout(10%).
"""

# %% [markdown]
# ## Step 0 — Imports + reproducibility + config

# %%
import os
import math
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score


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
    "data_dir": Path("../dataset"),
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
    "corr_lags": [0, 1, 2, 5],  # lead-lag (no leakage)
    "edges_mode": "all_pairs",  # "manual" | "all_pairs"
    "edges": [("ADA", "BTC"), ("ADA", "ETH"), ("ETH", "BTC")],  # used if edges_mode="manual"
    "add_self_loops": True,
    "edge_transform": "fisher",  # "none" | "fisher"
    "edge_scale": True,
    "edge_dropout": 0.10,

    # triple-barrier
    "tb_horizon": 1 * 30,
    "lookback": 4 * 12 * 5,
    "tb_pt_mult": 1.2,
    "tb_sl_mult": 1.1,
    "tb_min_barrier": 0.001,
    "tb_max_barrier": 0.006,

    # training
    "batch_size": 128,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "grad_clip": 1.0,
    "dropout": 0.15,

    # stability tricks
    "label_smoothing": 0.02,
    "use_weighted_sampler": True,
    "use_onecycle": True,

    # model dims
    "hidden": 128,
    "gnn_layers": 3,

    # --- Temporal (Conv -> AttnPool)
    "tcn_channels": 128,
    "tcn_layers": 3,
    "tcn_kernel": 2,
    "tcn_dropout": 0.20,
    "tcn_causal": True,

    "attn_pool_hidden": 128,
    "attn_pool_dropout": 0.10,

    # --- Learnable adjacency (MTGNN-style)
    # A_learned options:
    #   "emb": A = softmax((E1 @ E2^T)/temp)
    #   "matrix": A = softmax(A_logits/temp)
    "adj_mode": "emb",
    "adj_emb_dim": 8,
    "adj_temperature": 1.0,

    # A_prior from edge_attr (last timestep of the sequence)
    "prior_use_abs": False,       # if True: use abs(mean(edge_attr)) for weights
    "prior_diag_boost": 1.0,      # ensure diag >= this before row-normalization
    "prior_row_normalize": True,

    # mixing alpha
    "alpha_mode": "learned",      # "fixed" | "learned"
    "adj_alpha": 0.50,            # used if alpha_mode="fixed"
    "adj_alpha_min": 0.05,        # clamp if learned
    "adj_alpha_max": 0.95,

    # adjacency regularization
    "adj_l1_lambda": 1e-3,
    "adj_prior_lambda": 1e-2,

    # trading eval
    "cost_bps": 1.0,

    # threshold sweep grids (val only)
    "thr_trade_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "thr_dir_grid":   [0.50, 0.55, 0.60, 0.65, 0.70],

    # min trades constraints
    "eval_min_trades": 50,

    # anti-overtrading threshold selection
    "max_trade_rate_val": 0.65,
    "trade_rate_penalty": 0.10,
    "thr_objective": "pnl_sum",  # "pnl_sum" | "pnl_sharpe" | "pnl_per_trade"

    # dynamic quantile thresholds for thr_trade
    "proxy_target_trades": [50, 100, 200],
}

ASSETS = ["ADA", "BTC", "ETH"]
ASSET2IDX = {a: i for i, a in enumerate(ASSETS)}
TARGET_ASSET = "ETH"
TARGET_NODE = ASSET2IDX[TARGET_ASSET]


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
EDGE_NAMES = [f"{s}->{t}" for s, t in EDGE_LIST]
EDGE_INDEX = torch.tensor([[ASSET2IDX[s], ASSET2IDX[t]] for (s, t) in EDGE_LIST], dtype=torch.long)

print("EDGE_LIST:", EDGE_NAMES)
print("EDGE_INDEX:", EDGE_INDEX.tolist())

# %% [markdown]
# ## Step 1 — Load data + log returns

# %%
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
    data_dir = CFG["data_dir"]
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
print("Columns example:", df.columns[:20].tolist())
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
print(df.head(2))

# %% [markdown]
# ## Step 2 — Multi-window correlations → edge features (T,E,D)

# %%
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
print("edge_dim =", edge_feat.shape[-1], " = windows * lags =", len(CFG["corr_windows"]) * len(CFG["corr_lags"]))
print("Edge names:", EDGE_NAMES)
print("edge_feat sample [t=100, first 3 edges]:\n", edge_feat[100, :3, :])
print("edge_feat stats: mean=", float(edge_feat.mean()), "std=", float(edge_feat.std()))

# %% [markdown]
# ## Step 3 — Triple-barrier labels → two-stage labels + exit_ret

# %%
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
      thr: barrier per t (float array, len T)
    No leakage: vol is shift(1).
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


y_tb, exit_ret, exit_t, tb_thr = triple_barrier_labels_from_lr(
    df["lr_ETH"],
    horizon=CFG["tb_horizon"],
    vol_window=CFG["lookback"],
    pt_mult=CFG["tb_pt_mult"],
    sl_mult=CFG["tb_sl_mult"],
    min_barrier=CFG["tb_min_barrier"],
    max_barrier=CFG["tb_max_barrier"],
)

# two-stage labels
y_trade = (y_tb != 1).astype(np.int64)  # 1=trade, 0=no-trade
y_dir = (y_tb == 2).astype(np.int64)    # 1=up, 0=down (meaningful only when y_trade==1)

dist = np.bincount(y_tb, minlength=3)
print("TB dist [down,flat,up]:", dist)
print("Trade ratio (true):", float(y_trade.mean()))

# %% [markdown]
# ## Step 4 — Build node tensor (T,N,F) + sample_t

# %%
EPS = 1e-6


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
    book_levels = CFG["book_levels"]
    top_k = CFG["top_levels"]
    near_k = CFG["near_levels"]

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
        di_l0_4 = np.stack(di_levels, axis=1)  # (T,5)

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
L = CFG["lookback"]
H = CFG["tb_horizon"]

t_min = L - 1
t_max = T - H - 2
sample_t = np.arange(t_min, t_max + 1)
n_samples = len(sample_t)

print("X_node_raw:", X_node_raw.shape, "edge_feat:", edge_feat.shape)
print("node_feat_names:", node_feat_names)
print("n_samples:", n_samples, "| t range:", int(sample_t[0]), "->", int(sample_t[-1]))
print(
    "Feature stats (TARGET asset, lr):",
    "mean=", float(X_node_raw[:, TARGET_NODE, node_feat_names.index("lr")].mean()),
    "std=", float(X_node_raw[:, TARGET_NODE, node_feat_names.index("lr")].std()),
)

# %% [markdown]
# ## Step 5 — Final holdout split + walk-forward splits (CV-part only)

# %%
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


idx_cv_all, idx_final_test = make_final_holdout_split(n_samples, CFG["final_test_frac"])
n_samples_cv = len(idx_cv_all)
n_samples_final = len(idx_final_test)

print("Holdout split:")
print(f"  n_samples total: {n_samples}")
print(f"  n_samples CV   : {n_samples_cv} ({100 * n_samples_cv / n_samples:.1f}%)")
print(f"  n_samples FINAL: {n_samples_final} ({100 * n_samples_final / n_samples:.1f}%)")
print("  CV range   :", int(idx_cv_all[0]), int(idx_cv_all[-1]))
print("  FINAL range:", int(idx_final_test[0]), int(idx_final_test[-1]))

walk_splits = make_walk_forward_splits(
    n_samples_=n_samples_cv,
    train_min_frac=CFG["train_min_frac"],
    val_window_frac=CFG["val_window_frac"],
    test_window_frac=CFG["test_window_frac"],
    step_window_frac=CFG["step_window_frac"],
)

print("\nWalk-forward folds:", len(walk_splits))
for i, (a, b, c) in enumerate(walk_splits, 1):
    print(f"  fold {i}: train={len(a)} | val={len(b)} | test={len(c)}")

# %% [markdown]
# ## Step 6 — Dataset + scaling (train-only) + helpers

# %%
class LobGraphSequenceDataset2Stage(Dataset):
    """
    Returns:
      x_seq: (L,N,F)
      e_seq: (L,E,edge_dim)
      y_trade: scalar
      y_dir: scalar
      exit_ret: scalar
    """
    def __init__(
        self,
        X_node: np.ndarray,
        E_feat: np.ndarray,
        y_trade_arr: np.ndarray,
        y_dir_arr: np.ndarray,
        exit_ret_arr: np.ndarray,
        sample_t_: np.ndarray,
        indices: np.ndarray,
        lookback: int,
    ):
        self.X_node = X_node
        self.E_feat = E_feat
        self.y_trade = y_trade_arr
        self.y_dir = y_dir_arr
        self.exit_ret = exit_ret_arr
        self.sample_t = sample_t_
        self.indices = indices.astype(np.int64)
        self.L = int(lookback)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        sidx = int(self.indices[i])
        t = int(self.sample_t[sidx])
        t0 = t - self.L + 1

        x_seq = self.X_node[t0:t + 1]  # (L,N,F)
        e_seq = self.E_feat[t0:t + 1]  # (L,E,D)

        yt = int(self.y_trade[t])
        yd = int(self.y_dir[t])
        er = float(self.exit_ret[t])

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(e_seq),
            torch.tensor(yt, dtype=torch.long),
            torch.tensor(yd, dtype=torch.long),
            torch.tensor(er, dtype=torch.float32),
        )


def collate_fn_2stage(batch):
    xs, es, yts, yds, ers = zip(*batch)
    return (
        torch.stack(xs, 0),   # (B,L,N,F)
        torch.stack(es, 0),   # (B,L,E,D)
        torch.stack(yts, 0),  # (B,)
        torch.stack(yds, 0),  # (B,)
        torch.stack(ers, 0),  # (B,)
    )


def fit_scale_nodes_train_only(
    X_node_raw_: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    max_abs: float = 10.0
) -> Tuple[np.ndarray, RobustScaler]:
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    X_train_time = X_node_raw_[train_time_mask]  # (Ttr,N,F)
    _, _, Fdim = X_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(X_train_time.reshape(-1, Fdim))

    X_scaled = scaler.transform(X_node_raw_.reshape(-1, Fdim)).reshape(X_node_raw_.shape).astype(np.float32)
    X_scaled = np.clip(X_scaled, -max_abs, max_abs).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X_scaled, scaler


def fit_scale_edges_train_only(
    E_raw_: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    max_abs: float = 6.0
) -> Tuple[np.ndarray, RobustScaler]:
    """
    Robust-scale edge features per fold (train timeline only).
    Fisher-transformed correlations can be heavy-tailed.
    """
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    E_train_time = E_raw_[train_time_mask]  # (Ttr,E,D)
    _, _, D = E_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(E_train_time.reshape(-1, D))

    E_scaled = scaler.transform(E_raw_.reshape(-1, D)).reshape(E_raw_.shape).astype(np.float32)
    E_scaled = np.clip(E_scaled, -max_abs, max_abs).astype(np.float32)
    E_scaled = np.nan_to_num(E_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return E_scaled, scaler


def subset_trade_indices(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> np.ndarray:
    tt = sample_t_[indices]
    mask = (y_trade_arr[tt] == 1)
    return indices[mask]


def split_trade_ratio(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> float:
    tt = sample_t_[indices]
    return float(y_trade_arr[tt].mean()) if len(tt) else float("nan")

# %% [markdown]
# ## Step 7 — MTGNN-style model: (Conv -> AttnPool) + learnable adjacency

# %%
def build_adj_prior_from_edge_attr(
    edge_attr_last: torch.Tensor,    # (B,E,D)
    edge_index: torch.Tensor,        # (E,2) [src,dst]
    n_nodes: int,
    use_abs: bool = False,
    diag_boost: float = 1.0,
    row_normalize: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Build A_prior (B,N,N) from edge_attr at the last timestep.
    Default mapping:
      w = sigmoid(mean(edge_attr)) in [0,1]
    Then fill A_prior[src,dst] = w, enforce diag >= diag_boost, row-normalize.
    """
    edge_attr_last = torch.nan_to_num(edge_attr_last, nan=0.0, posinf=0.0, neginf=0.0)
    B, E, D = edge_attr_last.shape
    r = edge_attr_last.mean(dim=-1)  # (B,E)
    if use_abs:
        r = r.abs()

    w = torch.sigmoid(r)  # (B,E) in [0,1]

    A = torch.zeros((B, n_nodes, n_nodes), device=edge_attr_last.device, dtype=edge_attr_last.dtype)
    src = edge_index[:, 0].to(edge_attr_last.device)
    dst = edge_index[:, 1].to(edge_attr_last.device)
    A[:, src, dst] = w

    diag = torch.arange(n_nodes, device=edge_attr_last.device)
    A[:, diag, diag] = torch.maximum(A[:, diag, diag], torch.full_like(A[:, diag, diag], float(diag_boost)))

    if row_normalize:
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

    return torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


class LearnableAdjacency(nn.Module):
    """
    Produces A_learned (N,N) as row-softmax over logits.
    Also returns a "sparsity proxy" matrix for L1 regularization (sigmoid(logits)).
    """
    def __init__(self, n_nodes: int, cfg: Dict[str, Any]):
        super().__init__()
        self.n = int(n_nodes)
        self.mode = str(cfg.get("adj_mode", "emb"))
        self.temp = float(cfg.get("adj_temperature", 1.0))
        self.temp = max(self.temp, 1e-3)

        if self.mode == "matrix":
            self.adj_logits = nn.Parameter(0.01 * torch.randn(self.n, self.n))
        elif self.mode == "emb":
            k = int(cfg.get("adj_emb_dim", 8))
            self.E1 = nn.Parameter(0.01 * torch.randn(self.n, k))
            self.E2 = nn.Parameter(0.01 * torch.randn(self.n, k))
        else:
            raise ValueError(f"Unknown adj_mode={self.mode}")

        alpha_mode = str(cfg.get("alpha_mode", "fixed"))
        self.alpha_mode = alpha_mode
        if self.alpha_mode == "learned":
            init_alpha = float(cfg.get("adj_alpha", 0.5))
            init_alpha = float(np.clip(init_alpha, 1e-3, 1 - 1e-3))
            self.alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha / (1 - init_alpha)), dtype=torch.float32))
        else:
            self.register_buffer("alpha_fixed", torch.tensor(float(cfg.get("adj_alpha", 0.5)), dtype=torch.float32))

        self.alpha_min = float(cfg.get("adj_alpha_min", 0.05))
        self.alpha_max = float(cfg.get("adj_alpha_max", 0.95))

    def _get_logits(self) -> torch.Tensor:
        if self.mode == "matrix":
            return self.adj_logits
        logits = self.E1 @ self.E2.t()  # (N,N)
        return logits

    def alpha(self) -> torch.Tensor:
        if self.alpha_mode == "learned":
            a = torch.sigmoid(self.alpha_logit)
            return torch.clamp(a, min=self.alpha_min, max=self.alpha_max)
        return torch.clamp(self.alpha_fixed, min=0.0, max=1.0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._get_logits() / self.temp  # (N,N)
        A = torch.softmax(logits, dim=-1)        # row-stochastic
        sparsity_proxy = torch.sigmoid(logits)   # used for L1 on off-diagonal
        return A, sparsity_proxy


class GraphMixLayer(nn.Module):
    """
    Simple adjacency-based message passing:
      m_j = sum_i A[i,j] * h_i
      out = GELU(W_self h + W_nei m)
      gated residual
    """
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(hidden, hidden)
        self.lin_nei = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        for m in [self.lin_self, self.lin_nei, self.gate]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        h: (B,L,N,H)
        A: (B,N,N) with A[src,dst]
        """
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        m = torch.einsum("bij,blih->bljh", A, h)  # aggregate to dst=j
        out = F.gelu(self.lin_self(h) + self.lin_nei(m))
        out = self.drop(out)

        g = torch.sigmoid(self.gate(torch.cat([h, m], dim=-1)))
        y = g * out + (1.0 - g) * h
        return torch.nan_to_num(self.ln(y), nan=0.0, posinf=0.0, neginf=0.0)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, dilation=self.dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float, causal: bool = True):
        super().__init__()
        self.causal = bool(causal)

        if self.causal:
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        else:
            pad = ((kernel_size - 1) * dilation) // 2
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)

        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.downsample = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        res = self.downsample(x)
        return torch.nan_to_num(self.act(y + res), nan=0.0, posinf=0.0, neginf=0.0)


class TemporalConvNet(nn.Module):
    def __init__(self, in_ch: int, channels: List[int], kernel_size: int, dropout: float, causal: bool = True):
        super().__init__()
        layers = []
        cur = int(in_ch)
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(cur, int(out_ch), int(kernel_size), int(dilation), float(dropout), causal=causal))
            cur = int(out_ch)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttnPool1D(nn.Module):
    """
    Lightweight attention pooling over time.
    Input:  y (B,C,L)
    Output: pooled (B,C)
    """
    def __init__(self, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self.proj(y)                 # (B,1,L)
        attn = torch.softmax(scores, dim=-1)  # (B,1,L)
        pooled = (attn * y).sum(dim=-1)       # (B,C)
        return pooled, attn.squeeze(1)        # (B,C), (B,L)


class MTGNN_ConvAttn_Classifier(nn.Module):
    """
    forward(x_seq, e_seq, edge_index) -> logits (B,2)
    Also supports returning aux losses:
      forward(..., return_aux=True) -> (logits, aux_dict)
    """
    def __init__(self, node_in: int, edge_dim: int, cfg: Dict[str, Any], n_nodes: int, target_node: int, n_classes: int = 2):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)

        hidden = int(cfg["hidden"])
        dropout = float(cfg["dropout"])

        # adjacency modules
        self.learn_adj = LearnableAdjacency(self.n_nodes, cfg)

        # node feature projection
        self.in_proj = nn.Sequential(
            nn.Linear(int(node_in), hidden),
            nn.LayerNorm(hidden),
        )

        # graph layers (use A_final for message passing)
        self.gnn_layers = nn.ModuleList([GraphMixLayer(hidden, dropout=dropout) for _ in range(int(cfg["gnn_layers"]))])

        # temporal conv + attention pooling on target node trajectory
        tcn_channels = int(cfg["tcn_channels"])
        tcn_layers_n = int(cfg["tcn_layers"])
        tcn_kernel = int(cfg["tcn_kernel"])
        tcn_dropout = float(cfg["tcn_dropout"])
        tcn_causal = bool(cfg["tcn_causal"])

        self.tcn_in = nn.Linear(hidden, tcn_channels)
        self.tcn = TemporalConvNet(
            in_ch=tcn_channels,
            channels=[tcn_channels] * tcn_layers_n,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
            causal=tcn_causal,
        )

        self.pool = AttnPool1D(
            channels=tcn_channels,
            hidden=int(cfg.get("attn_pool_hidden", tcn_channels)),
            dropout=float(cfg.get("attn_pool_dropout", 0.1)),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(tcn_channels),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels, tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_A_final(self, e_seq: torch.Tensor, edge_index: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        e_seq: (B,L,E,D)
        Build A_prior from last timestep, mix with A_learned.
        Return A_final (B,N,N) and aux dict with reg terms.
        """
        B, L_, E, D = e_seq.shape
        e_last = e_seq[:, -1, :, :]  # (B,E,D)

        A_prior = build_adj_prior_from_edge_attr(
            edge_attr_last=e_last,
            edge_index=edge_index,
            n_nodes=self.n_nodes,
            use_abs=bool(cfg.get("prior_use_abs", False)),
            diag_boost=float(cfg.get("prior_diag_boost", 1.0)),
            row_normalize=bool(cfg.get("prior_row_normalize", True)),
        )  # (B,N,N)

        A_learned_base, sparsity_proxy = self.learn_adj()  # (N,N), (N,N)
        A_learned = A_learned_base.unsqueeze(0).expand(B, -1, -1)  # (B,N,N)

        alpha = self.learn_adj.alpha().to(e_seq.device).to(e_seq.dtype)  # scalar
        A_final = alpha * A_prior + (1.0 - alpha) * A_learned

        # regularization
        N = self.n_nodes
        offdiag = (1.0 - torch.eye(N, device=e_seq.device, dtype=e_seq.dtype))
        l1_off = (sparsity_proxy * offdiag).abs().mean()
        mse_prior = ((A_learned - A_prior) ** 2 * offdiag).mean()

        aux = {
            "alpha": float(alpha.detach().cpu().item()),
            "l1_off": float(l1_off.detach().cpu().item()),
            "mse_prior": float(mse_prior.detach().cpu().item()),
            # keep tensors for loss composition
            "_l1_off_t": l1_off,
            "_mse_prior_t": mse_prior,
        }
        return A_final, aux

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        cfg: Optional[Dict[str, Any]] = None,
        return_aux: bool = False
    ):
        cfg = CFG if cfg is None else cfg

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

        B, L_, N, Fin = x.shape
        assert N == self.n_nodes, f"Expected N={self.n_nodes}, got {N}"

        A_final, aux = self._compute_A_final(e, edge_index, cfg)

        h = self.in_proj(x)  # (B,L,N,H)
        for gnn in self.gnn_layers:
            h = gnn(h, A_final)  # (B,L,N,H)

        h_tgt = h[:, :, self.target_node, :]  # (B,L,H)
        z = self.tcn_in(h_tgt)                # (B,L,C)
        z = z.transpose(1, 2)                 # (B,C,L)
        y = self.tcn(z)                       # (B,C,L)
        emb, attn_w = self.pool(y)            # (B,C), (B,L)

        logits = self.head(emb)               # (B,2)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            aux["attn_mean"] = float(attn_w.mean().detach().cpu().item())
            aux["attn_max"] = float(attn_w.max().detach().cpu().item())
            return logits, aux
        return logits


# sanity
B_ = 2
Fdim = X_node_raw.shape[-1]
E_ = EDGE_INDEX.shape[0]
Dedge = edge_feat.shape[-1]
x_dummy = torch.randn(B_, L, len(ASSETS), Fdim)
e_dummy = torch.randn(B_, L, E_, Dedge)
m_dummy = MTGNN_ConvAttn_Classifier(node_in=Fdim, edge_dim=Dedge, cfg=CFG, n_nodes=len(ASSETS), target_node=TARGET_NODE).to(DEVICE)
with torch.no_grad():
    out, aux = m_dummy(x_dummy.to(DEVICE), e_dummy.to(DEVICE), EDGE_INDEX.to(DEVICE), cfg=CFG, return_aux=True)
print("Model sanity logits:", out.shape, "| finite:", bool(torch.isfinite(out).all().item()))
print("Aux sanity:", {k: aux[k] for k in ["alpha", "l1_off", "mse_prior", "attn_mean", "attn_max"]})

# %% [markdown]
# ## Step 8 — Train/Eval helpers (AUC-oriented) + adjacency regularization

# %%
def make_ce_weights_binary(y_np: np.ndarray) -> torch.Tensor:
    y_np = np.asarray(y_np, dtype=np.int64)
    counts = np.bincount(y_np, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def make_weighted_sampler_from_labels(y_np: np.ndarray) -> WeightedRandomSampler:
    y_np = np.asarray(y_np, dtype=np.int64)
    counts = np.bincount(y_np, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = counts.sum() / (2.0 * counts)
    sample_w = class_w[y_np].astype(np.float64)
    sample_w = torch.tensor(sample_w, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def total_loss_with_adj_reg(ce_loss: torch.Tensor, aux: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
    lam_l1 = float(cfg.get("adj_l1_lambda", 0.0))
    lam_pr = float(cfg.get("adj_prior_lambda", 0.0))
    reg = 0.0
    if lam_l1 > 0:
        reg = reg + lam_l1 * aux["_l1_off_t"]
    if lam_pr > 0:
        reg = reg + lam_pr * aux["_mse_prior_t"]
    return ce_loss + reg


@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, y_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0

    ys = []
    probs = []
    ers = []
    aux_accum = {"alpha": [], "l1_off": [], "mse_prior": []}

    for x, e, y_trade_b, y_dir_b, er in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        y = (y_trade_b if y_key == "trade" else y_dir_b).to(DEVICE).long()

        logits, aux = model(x, e, EDGE_INDEX.to(DEVICE), cfg=cfg, return_aux=True)
        ce = loss_fn(logits, y)
        loss = total_loss_with_adj_reg(ce, aux, cfg)

        total_loss += float(loss.item()) * int(y.size(0))
        n += int(y.size(0))

        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        probs.append(p)
        ers.append(er.detach().cpu().numpy())

        aux_accum["alpha"].append(aux["alpha"])
        aux_accum["l1_off"].append(aux["l1_off"])
        aux_accum["mse_prior"].append(aux["mse_prior"])

    ys = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    probs = np.concatenate(probs) if probs else np.zeros((0, 2), dtype=np.float32)
    ers = np.concatenate(ers) if ers else np.array([], dtype=np.float32)

    if len(ys) == 0:
        return {"loss": np.nan, "acc": np.nan, "f1m": np.nan, "auc": np.nan, "cm": None, "y": ys, "prob": probs, "er": ers}

    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(ys, y_pred)
    f1m = f1_score(ys, y_pred, average="macro")
    auc = roc_auc_score(ys, probs[:, 1]) if len(np.unique(ys)) == 2 else np.nan
    cm = confusion_matrix(ys, y_pred)

    out = {
        "loss": total_loss / max(1, n),
        "acc": float(acc),
        "f1m": float(f1m),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "cm": cm,
        "y": ys,
        "prob": probs,
        "er": ers,
    }
    if aux_accum["alpha"]:
        out["adj_alpha_mean"] = float(np.mean(aux_accum["alpha"]))
        out["adj_l1_off_mean"] = float(np.mean(aux_accum["l1_off"]))
        out["adj_mse_prior_mean"] = float(np.mean(aux_accum["mse_prior"]))
    return out


def train_binary_classifier(
    X_scaled: np.ndarray,
    edge_scaled: np.ndarray,
    y_trade_arr: np.ndarray,
    y_dir_arr: np.ndarray,
    exit_ret_arr: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict[str, Any],
    stage_name: str,  # "trade" or "dir"
) -> Tuple[nn.Module, Dict[str, Any]]:
    assert stage_name in ("trade", "dir")

    L_ = int(cfg["lookback"])
    bs = int(cfg["batch_size"])

    tr_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_scaled, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_train, L_)
    va_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_scaled, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_val,   L_)
    te_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_scaled, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_test,  L_)

    # labels for sampler/weights (TRAIN only)
    t_train = sample_t_[idx_train]
    y_train_np = (y_trade_arr[t_train] if stage_name == "trade" else y_dir_arr[t_train]).astype(np.int64)

    sampler = None
    shuffle = True
    if bool(cfg.get("use_weighted_sampler", True)):
        sampler = make_weighted_sampler_from_labels(y_train_np)
        shuffle = False

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=shuffle, sampler=sampler, drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)

    node_in = int(X_scaled.shape[-1])
    edge_dim = int(edge_scaled.shape[-1])

    model = MTGNN_ConvAttn_Classifier(
        node_in=node_in,
        edge_dim=edge_dim,
        cfg=cfg,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        n_classes=2,
    ).to(DEVICE)

    ce_w = make_ce_weights_binary(y_train_np)
    loss_fn = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=float(cfg.get("label_smoothing", 0.0)))

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

    best_auc = -1e18
    best_state = None
    best_epoch = -1
    patience = 7
    bad = 0

    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        tot_loss = 0.0
        n = 0

        for x, e, y_trade_b, y_dir_b, _er in tr_loader:
            x = x.to(DEVICE).float()
            e = e.to(DEVICE).float()
            y = (y_trade_b if stage_name == "trade" else y_dir_b).to(DEVICE).long()

            opt.zero_grad(set_to_none=True)

            logits, aux = model(x, e, EDGE_INDEX.to(DEVICE), cfg=cfg, return_aux=True)
            ce = loss_fn(logits, y)
            loss = total_loss_with_adj_reg(ce, aux, cfg)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()

            if use_onecycle:
                sch.step()

            tot_loss += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))

        tr_loss = tot_loss / max(1, n)

        va = eval_binary(model, va_loader, loss_fn, y_key=stage_name, cfg=cfg)
        va_auc = va["auc"]
        sel_auc = float(va_auc) if np.isfinite(va_auc) else -1e18

        if sel_auc > best_auc:
            best_auc = sel_auc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if not use_onecycle:
            sch.step(sel_auc)

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[{stage_name}] ep {ep:02d} lr={lr_now:.2e} "
            f"tr_loss={tr_loss:.4f} va_loss={va['loss']:.4f} va_auc={va_auc:.3f} "
            f"alpha={va.get('adj_alpha_mean', float('nan')):.3f} "
            f"reg(l1={va.get('adj_l1_off_mean', float('nan')):.4f}, prior={va.get('adj_mse_prior_mean', float('nan')):.4f}) "
            f"best={best_auc:.3f}@ep{best_epoch:02d}"
        )

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    va = eval_binary(model, va_loader, loss_fn, y_key=stage_name, cfg=cfg)
    te = eval_binary(model, te_loader, loss_fn, y_key=stage_name, cfg=cfg)

    res = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc) if np.isfinite(best_auc) else np.nan,
        "val": va,
        "test": te,
    }
    return model, res

# %% [markdown]
# ## Step 9 — Two-stage PnL + threshold sweep (val only)

# %%
def build_trade_threshold_grid(
    p_trade: np.ndarray,
    base_grid: Optional[List[float]] = None,
    target_trades_list: Optional[List[int]] = None,
    min_thr: float = 0.01,
    max_thr: float = 0.99,
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
            thrs.add(float(np.clip(thr, min_thr, max_thr)))

    out = sorted(thrs)
    cleaned = []
    for t in out:
        if not cleaned or abs(t - cleaned[-1]) > 1e-6:
            cleaned.append(float(t))
    return cleaned


def two_stage_trade_mask(prob_trade: np.ndarray, prob_dir: np.ndarray, thr_trade: float, thr_dir: float) -> np.ndarray:
    p_trade = prob_trade[:, 1]
    p_up = prob_dir[:, 1]
    conf_dir = np.maximum(p_up, 1.0 - p_up)
    return (p_trade >= float(thr_trade)) & (conf_dir >= float(thr_dir))


def two_stage_pnl_by_threshold(
    prob_trade: np.ndarray,
    prob_dir: np.ndarray,
    exit_ret_arr: np.ndarray,
    thr_trade: float,
    thr_dir: float,
    cost_bps: float,
) -> Dict[str, Any]:
    p_up = prob_dir[:, 1]
    mask = two_stage_trade_mask(prob_trade, prob_dir, thr_trade, thr_dir)

    action = np.zeros_like(exit_ret_arr, dtype=np.float32)
    action[mask] = np.where(p_up[mask] >= 0.5, 1.0, -1.0).astype(np.float32)

    cost = (float(cost_bps) * 1e-4) * mask.astype(np.float32)
    pnl = action * exit_ret_arr - cost

    n = int(len(exit_ret_arr))
    n_tr = int(mask.sum())

    return {
        "n": n,
        "n_trades": n_tr,
        "trade_rate": float(n_tr / max(1, n)),
        "pnl_sum": float(pnl.sum()),
        "pnl_mean": float(pnl.mean()) if n else np.nan,
        "pnl_per_trade": float(pnl.sum() / max(1, n_tr)),
        "pnl_sharpe": float((pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(288)) if n else np.nan,
    }


def sweep_thresholds(
    prob_trade: np.ndarray,
    prob_dir: np.ndarray,
    exit_ret_arr: np.ndarray,
    cfg: Dict[str, Any],
    min_trades: int = 0,
    target_trade_rate: Optional[float] = None,
) -> pd.DataFrame:
    p_trade = prob_trade[:, 1]
    thr_trade_grid = build_trade_threshold_grid(
        p_trade=p_trade,
        base_grid=cfg.get("thr_trade_grid", [0.5]),
        target_trades_list=cfg.get("proxy_target_trades", None),
        min_thr=0.01,
        max_thr=0.99,
    )
    thr_dir_grid = cfg.get("thr_dir_grid", [0.5])

    obj = str(cfg.get("thr_objective", "pnl_sum"))
    max_rate = cfg.get("max_trade_rate_val", None)
    penalty = float(cfg.get("trade_rate_penalty", 0.0))

    rows = []
    for thr_t in thr_trade_grid:
        for thr_d in thr_dir_grid:
            m = two_stage_pnl_by_threshold(prob_trade, prob_dir, exit_ret_arr, thr_t, thr_d, cfg["cost_bps"])
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
        return sweep_thresholds(prob_trade, prob_dir, exit_ret_arr, cfg, min_trades=1, target_trade_rate=target_trade_rate)

    df_ = pd.DataFrame(rows).sort_values(["score", "pnl_sum"], ascending=False)
    return df_


@torch.no_grad()
def predict_probs_on_indices(
    model: nn.Module,
    X_scaled: np.ndarray,
    edge_scaled: np.ndarray,
    indices: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    ds = LobGraphSequenceDataset2Stage(X_scaled, edge_scaled, y_trade, y_dir, exit_ret, sample_t, indices, cfg["lookback"])
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_2stage, num_workers=0)

    model.eval()
    probs = []
    ers = []
    for x, e, _yt, _yd, er in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        logits = model(x, e, EDGE_INDEX.to(DEVICE), cfg=cfg, return_aux=False)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
        ers.append(er.cpu().numpy())

    return np.concatenate(probs, axis=0), np.concatenate(ers, axis=0)

# %% [markdown]
# ## Step 10 — Run walk-forward folds (CV-part) + store fold artifacts

# %%
def _state_dict_to_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def run_walk_forward_cv() -> Tuple[pd.DataFrame, List[Dict[str, Any]], nn.Module, nn.Module]:
    """
    Returns:
      - cv_summary: per-fold TEST metrics (using thresholds selected on that fold VAL)
      - fold_artifacts: list of per-fold dicts (models + thresholds + VAL preds)
      - m_trade_last, m_dir_last: last fold trained models
    """
    rows: List[Dict[str, Any]] = []
    fold_artifacts: List[Dict[str, Any]] = []

    m_trade_last: Optional[nn.Module] = None
    m_dir_last: Optional[nn.Module] = None

    for fi, (idx_tr, idx_va, idx_te) in enumerate(walk_splits, 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fi}/{len(walk_splits)} sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
        true_val_trade = split_trade_ratio(idx_va, sample_t, y_trade)
        true_te_trade = split_trade_ratio(idx_te, sample_t, y_trade)
        print(f"True trade ratio (val):  {true_val_trade:.3f}")
        print(f"True trade ratio (test): {true_te_trade:.3f}")

        # fold scaling (fit only on fold train timeline)
        X_scaled, _ = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_tr, max_abs=CFG["max_abs_feat"])
        if bool(CFG.get("edge_scale", True)):
            edge_scaled, _ = fit_scale_edges_train_only(edge_feat, sample_t, idx_tr, max_abs=CFG["max_abs_edge"])
        else:
            edge_scaled = edge_feat

        # Stage A
        m_trade, r_trade = train_binary_classifier(
            X_scaled, edge_scaled, y_trade, y_dir, exit_ret, sample_t,
            idx_tr, idx_va, idx_te, CFG, stage_name="trade"
        )

        # Stage B (trade-only indices)
        idx_tr_T = subset_trade_indices(idx_tr, sample_t, y_trade)
        idx_va_T = subset_trade_indices(idx_va, sample_t, y_trade)
        idx_te_T = subset_trade_indices(idx_te, sample_t, y_trade)

        if len(idx_tr_T) < max(200, 2 * CFG["batch_size"]) or len(idx_va_T) < 50 or len(idx_te_T) < 50:
            print("[dir] skip: not enough trade samples in this fold.")

            rows.append({
                "fold": fi,
                "trade_test_auc": r_trade["test"]["auc"],
                "dir_test_auc": np.nan,
                "test_trade_rate_pred": np.nan,
                "test_pnl_sum": np.nan,
                "test_pnl_mean": np.nan,
                "thr_trade": np.nan,
                "thr_dir": np.nan,
                "n_trades": np.nan,
                "best_val_score": np.nan,
            })

            fold_artifacts.append({
                "fold": fi,
                "idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te,
                "thr_trade": np.nan, "thr_dir": np.nan,
                "best_val_score": np.nan,
                "trade_state": _state_dict_to_cpu(m_trade.state_dict()),
                "dir_state": None,
                "prob_trade_val": None, "prob_dir_val": None, "er_val": None,
                "val_true_trade_rate": float(true_val_trade),
            })

            m_trade_last = m_trade
            m_dir_last = None
            continue

        m_dir, r_dir = train_binary_classifier(
            X_scaled, edge_scaled, y_trade, y_dir, exit_ret, sample_t,
            idx_tr_T, idx_va_T, idx_te_T, CFG, stage_name="dir"
        )

        # Choose thresholds on VAL (VAL only)
        prob_trade_val, er_val = predict_probs_on_indices(m_trade, X_scaled, edge_scaled, idx_va, CFG)
        prob_dir_val, _ = predict_probs_on_indices(m_dir, X_scaled, edge_scaled, idx_va, CFG)

        sweep_val = sweep_thresholds(
            prob_trade_val, prob_dir_val, er_val, CFG,
            min_trades=int(CFG["eval_min_trades"]),
            target_trade_rate=float(true_val_trade),
        )
        best_val = sweep_val.iloc[0].to_dict()
        thr_trade_star = float(best_val["thr_trade"])
        thr_dir_star = float(best_val["thr_dir"])

        val_metrics = two_stage_pnl_by_threshold(prob_trade_val, prob_dir_val, er_val, thr_trade_star, thr_dir_star, CFG["cost_bps"])
        print("\nChosen thresholds (from VAL):")
        print(f"  thr_trade*={thr_trade_star:.3f} thr_dir*={thr_dir_star:.3f} | score={best_val['score']:.4f}")
        print(f"  val trade_rate(pred)={val_metrics['trade_rate']:.3f} | val pnl_sum={val_metrics['pnl_sum']:.4f} | val sharpe={val_metrics['pnl_sharpe']:.3f}")

        print("\nTop-5 VAL threshold candidates:")
        print(sweep_val.head(5)[["thr_trade", "thr_dir", "score", "trade_rate", "pnl_sum", "pnl_sharpe", "n_trades"]])

        # Evaluate on TEST with fixed thresholds
        prob_trade_te, er_te = predict_probs_on_indices(m_trade, X_scaled, edge_scaled, idx_te, CFG)
        prob_dir_te, _ = predict_probs_on_indices(m_dir, X_scaled, edge_scaled, idx_te, CFG)
        te_metrics = two_stage_pnl_by_threshold(prob_trade_te, prob_dir_te, er_te, thr_trade_star, thr_dir_star, CFG["cost_bps"])

        print("\nTEST (fixed thr from VAL):")
        print(f"  trade_rate(pred)={te_metrics['trade_rate']:.3f} | pnl_sum={te_metrics['pnl_sum']:.4f} | pnl_mean={te_metrics['pnl_mean']:.6f} | trades={te_metrics['n_trades']}")

        rows.append({
            "fold": fi,
            "trade_test_auc": r_trade["test"]["auc"],
            "dir_test_auc": r_dir["test"]["auc"],
            "test_trade_rate_pred": te_metrics["trade_rate"],
            "test_pnl_sum": te_metrics["pnl_sum"],
            "test_pnl_mean": te_metrics["pnl_mean"],
            "thr_trade": thr_trade_star,
            "thr_dir": thr_dir_star,
            "n_trades": te_metrics["n_trades"],
            "best_val_score": float(best_val["score"]),
        })

        fold_artifacts.append({
            "fold": fi,
            "idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te,
            "thr_trade": thr_trade_star, "thr_dir": thr_dir_star,
            "best_val_score": float(best_val["score"]),
            "trade_state": _state_dict_to_cpu(m_trade.state_dict()),
            "dir_state": _state_dict_to_cpu(m_dir.state_dict()),
            "prob_trade_val": prob_trade_val,
            "prob_dir_val": prob_dir_val,
            "er_val": er_val,
            "val_true_trade_rate": float(true_val_trade),
        })

        m_trade_last = m_trade
        m_dir_last = m_dir

    if m_trade_last is None:
        raise RuntimeError("No folds were trained; check your split configuration.")

    cv_summary = pd.DataFrame(rows)
    return cv_summary, fold_artifacts, m_trade_last, m_dir_last


cv_summary, fold_artifacts, m_trade_last, m_dir_last = run_walk_forward_cv()

print("\n" + "=" * 80)
print("CV summary (fold TEST, fixed thresholds from fold-VAL):")
print(cv_summary)
print("\nMeans (just for debugging, NOT a final decision rule):")
print(cv_summary.mean(numeric_only=True))

# %% [markdown]
# ## Step 11 — Post-CV checks on FINAL holdout (10%) WITHOUT refit (3 methods)

# %%
def _safe_auc_binary(y_true: np.ndarray, p1: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    p1 = np.asarray(p1, dtype=np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p1))


def _build_model_from_state(node_in: int, edge_dim: int, cfg: Dict[str, Any], state: Dict[str, torch.Tensor]) -> nn.Module:
    m = MTGNN_ConvAttn_Classifier(
        node_in=node_in,
        edge_dim=edge_dim,
        cfg=cfg,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        n_classes=2,
    ).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m


def _get_scaled_arrays_for_fold(idx_tr_fold: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled, _ = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_tr_fold, max_abs=CFG["max_abs_feat"])
    if bool(CFG.get("edge_scale", True)):
        edge_scaled, _ = fit_scale_edges_train_only(edge_feat, sample_t, idx_tr_fold, max_abs=CFG["max_abs_edge"])
    else:
        edge_scaled = edge_feat
    return X_scaled, edge_scaled


def _eval_holdout_with_models_and_thresholds(
    method: str,
    m_trade: nn.Module,
    m_dir: nn.Module,
    thr_trade: float,
    thr_dir: float,
    X_scaled: np.ndarray,
    edge_scaled: np.ndarray,
    idx_holdout: np.ndarray,
) -> Dict[str, Any]:
    prob_trade_hold, er_hold = predict_probs_on_indices(m_trade, X_scaled, edge_scaled, idx_holdout, CFG)
    prob_dir_hold, _ = predict_probs_on_indices(m_dir, X_scaled, edge_scaled, idx_holdout, CFG)

    t_hold = sample_t[idx_holdout]
    y_trade_hold = y_trade[t_hold].astype(np.int64)
    y_dir_hold = y_dir[t_hold].astype(np.int64)

    trade_auc = _safe_auc_binary(y_trade_hold, prob_trade_hold[:, 1])
    mask_true_trade = (y_trade_hold == 1)
    dir_auc = _safe_auc_binary(y_dir_hold[mask_true_trade], prob_dir_hold[mask_true_trade, 1])

    pnl = two_stage_pnl_by_threshold(
        prob_trade_hold, prob_dir_hold, er_hold,
        thr_trade=thr_trade, thr_dir=thr_dir,
        cost_bps=CFG["cost_bps"],
    )

    return {
        "method": method,
        "thr_trade": float(thr_trade),
        "thr_dir": float(thr_dir),
        "holdout_trade_auc": float(trade_auc) if np.isfinite(trade_auc) else np.nan,
        "holdout_dir_auc": float(dir_auc) if np.isfinite(dir_auc) else np.nan,
        "trade_rate_pred": float(pnl["trade_rate"]),
        "pnl_sum": float(pnl["pnl_sum"]),
        "pnl_sharpe": float(pnl["pnl_sharpe"]),
        "n_trades": int(pnl["n_trades"]),
    }


def run_post_cv_holdout_checks() -> pd.DataFrame:
    idx_holdout = idx_final_test.astype(np.int64)
    node_in = int(X_node_raw.shape[-1])
    edge_dim = int(edge_feat.shape[-1])

    ok_folds = [fa for fa in fold_artifacts if fa.get("dir_state") is not None and np.isfinite(fa.get("thr_trade", np.nan))]
    if len(ok_folds) == 0:
        raise RuntimeError("No folds with a trained DIR model were stored; cannot run Step 11 checks.")

    # 1) LAST fold model + LAST fold thresholds
    fa_last = ok_folds[-1]
    X_last, E_last = _get_scaled_arrays_for_fold(fa_last["idx_tr"])
    m_trade_last_ = _build_model_from_state(node_in, edge_dim, CFG, fa_last["trade_state"])
    m_dir_last_ = _build_model_from_state(node_in, edge_dim, CFG, fa_last["dir_state"])
    r1 = _eval_holdout_with_models_and_thresholds(
        method=f"1) LAST fold model + LAST fold thresholds (fold={fa_last['fold']})",
        m_trade=m_trade_last_,
        m_dir=m_dir_last_,
        thr_trade=float(fa_last["thr_trade"]),
        thr_dir=float(fa_last["thr_dir"]),
        X_scaled=X_last,
        edge_scaled=E_last,
        idx_holdout=idx_holdout,
    )

    # 2) BEST-VAL fold model + BEST-VAL thresholds
    fa_best = max(ok_folds, key=lambda d: float(d.get("best_val_score", -1e18)))
    X_best, E_best = _get_scaled_arrays_for_fold(fa_best["idx_tr"])
    m_trade_best = _build_model_from_state(node_in, edge_dim, CFG, fa_best["trade_state"])
    m_dir_best = _build_model_from_state(node_in, edge_dim, CFG, fa_best["dir_state"])
    r2 = _eval_holdout_with_models_and_thresholds(
        method=f"2) BEST-VAL fold model + BEST-VAL thresholds (fold={fa_best['fold']}, val_score={fa_best['best_val_score']:.4f})",
        m_trade=m_trade_best,
        m_dir=m_dir_best,
        thr_trade=float(fa_best["thr_trade"]),
        thr_dir=float(fa_best["thr_dir"]),
        X_scaled=X_best,
        edge_scaled=E_best,
        idx_holdout=idx_holdout,
    )

    # 3) LAST fold model + GLOBAL thresholds on concatenated fold-VAL predictions
    prob_trade_all = np.concatenate([fa["prob_trade_val"] for fa in ok_folds], axis=0)
    prob_dir_all = np.concatenate([fa["prob_dir_val"] for fa in ok_folds], axis=0)
    er_all = np.concatenate([fa["er_val"] for fa in ok_folds], axis=0)

    idx_va_all = np.concatenate([fa["idx_va"] for fa in ok_folds], axis=0)
    true_trade_rate_all = split_trade_ratio(idx_va_all, sample_t, y_trade)

    sweep_global = sweep_thresholds(
        prob_trade_all, prob_dir_all, er_all, CFG,
        min_trades=int(CFG["eval_min_trades"]),
        target_trade_rate=float(true_trade_rate_all),
    )
    best_global = sweep_global.iloc[0].to_dict()
    thr_trade_global = float(best_global["thr_trade"])
    thr_dir_global = float(best_global["thr_dir"])

    r3 = _eval_holdout_with_models_and_thresholds(
        method=f"3) LAST fold model + GLOBAL thresholds (VAL-concat; true_val_trade={true_trade_rate_all:.3f}) (model_fold={fa_last['fold']})",
        m_trade=m_trade_last_,
        m_dir=m_dir_last_,
        thr_trade=thr_trade_global,
        thr_dir=thr_dir_global,
        X_scaled=X_last,
        edge_scaled=E_last,
        idx_holdout=idx_holdout,
    )

    out = pd.DataFrame([r1, r2, r3])

    print("\n" + "=" * 80)
    print("STEP 11 — HOLDOUT CHECKS WITHOUT ANY REFIT (3 METHODS)")
    print(f"Holdout size: {len(idx_holdout)} samples | sidx {int(idx_holdout[0])}..{int(idx_holdout[-1])}")

    print("\nResults (compare these, not mean/median over folds):")
    print(out[[
        "method", "thr_trade", "thr_dir",
        "holdout_trade_auc", "holdout_dir_auc",
        "trade_rate_pred", "pnl_sum", "pnl_sharpe", "n_trades"
    ]].to_string(index=False))

    print("\nGlobal thresholds (method 3) top-5 candidates (VAL-concat):")
    print(sweep_global.head(5)[["thr_trade", "thr_dir", "score", "trade_rate", "pnl_sum", "pnl_sharpe", "n_trades"]])

    print("=" * 80)
    return out


post_cv_holdout = run_post_cv_holdout_checks()

# %% [markdown]
# ## Step 12 — Production-fit: train on CV(90%) → select thresholds on val_final → eval on FINAL holdout(10%)

# %%
def run_production_fit() -> Dict[str, Any]:
    """
    Train on the full CV-part (90%) with a final validation window (val_final),
    select thresholds on val_final, then evaluate on FINAL holdout (10%).
    """
    print("\n" + "=" * 80)
    print("STEP 12 — PRODUCTION-FIT (TRAIN ON CV(90%) → SELECT THR ON val_final → EVAL ON FINAL HOLDOUT(10%))")

    val_w = max(1, int(CFG["val_window_frac"] * n_samples_cv))
    train_end = n_samples_cv - val_w

    idx_train_final = np.arange(0, train_end, dtype=np.int64)
    idx_val_final = np.arange(train_end, n_samples_cv, dtype=np.int64)
    idx_holdout = idx_final_test.astype(np.int64)

    true_val_trade = split_trade_ratio(idx_val_final, sample_t, y_trade)
    true_hold_trade = split_trade_ratio(idx_holdout, sample_t, y_trade)

    print("Final split sizes:")
    print("  train_final:", len(idx_train_final))
    print("  val_final  :", len(idx_val_final))
    print("  holdout    :", len(idx_holdout))
    print(f"True trade ratio (val_final): {true_val_trade:.3f}")
    print(f"True trade ratio (holdout):   {true_hold_trade:.3f}")

    X_scaled_final, _ = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_train_final, max_abs=CFG["max_abs_feat"])
    if bool(CFG.get("edge_scale", True)):
        edge_scaled_final, _ = fit_scale_edges_train_only(edge_feat, sample_t, idx_train_final, max_abs=CFG["max_abs_edge"])
    else:
        edge_scaled_final = edge_feat

    # Stage A
    m_trade_f, r_trade = train_binary_classifier(
        X_scaled_final, edge_scaled_final, y_trade, y_dir, exit_ret, sample_t,
        idx_train_final, idx_val_final, idx_holdout, CFG, stage_name="trade"
    )

    # Stage B (trade-only)
    idx_train_T = subset_trade_indices(idx_train_final, sample_t, y_trade)
    idx_val_T = subset_trade_indices(idx_val_final, sample_t, y_trade)
    idx_hold_T = subset_trade_indices(idx_holdout, sample_t, y_trade)

    print("\nTrade-only sizes for DIR:")
    print("  train_final_T:", len(idx_train_T))
    print("  val_final_T  :", len(idx_val_T))
    print("  holdout_T    :", len(idx_hold_T))

    m_dir_f, r_dir = train_binary_classifier(
        X_scaled_final, edge_scaled_final, y_trade, y_dir, exit_ret, sample_t,
        idx_train_T, idx_val_T, idx_hold_T, CFG, stage_name="dir"
    )

    # Choose thresholds on val_final (VAL only)
    prob_trade_val, er_val = predict_probs_on_indices(m_trade_f, X_scaled_final, edge_scaled_final, idx_val_final, CFG)
    prob_dir_val, _ = predict_probs_on_indices(m_dir_f, X_scaled_final, edge_scaled_final, idx_val_final, CFG)

    sweep_val = sweep_thresholds(
        prob_trade_val, prob_dir_val, er_val, CFG,
        min_trades=int(CFG["eval_min_trades"]),
        target_trade_rate=float(true_val_trade),
    )
    best_val = sweep_val.iloc[0].to_dict()
    thr_trade_star = float(best_val["thr_trade"])
    thr_dir_star = float(best_val["thr_dir"])

    val_metrics = two_stage_pnl_by_threshold(prob_trade_val, prob_dir_val, er_val, thr_trade_star, thr_dir_star, CFG["cost_bps"])
    print("\nChosen thresholds on val_final:")
    print(f"  thr_trade*={thr_trade_star:.3f} thr_dir*={thr_dir_star:.3f} | score={best_val['score']:.4f}")
    print(f"  val trade_rate(pred)={val_metrics['trade_rate']:.3f} | val pnl_sum={val_metrics['pnl_sum']:.4f} | val sharpe={val_metrics['pnl_sharpe']:.3f} | trades={val_metrics['n_trades']}")

    # Evaluate holdout with fixed thresholds
    prob_trade_hold, er_hold = predict_probs_on_indices(m_trade_f, X_scaled_final, edge_scaled_final, idx_holdout, CFG)
    prob_dir_hold, _ = predict_probs_on_indices(m_dir_f, X_scaled_final, edge_scaled_final, idx_holdout, CFG)
    hold_metrics = two_stage_pnl_by_threshold(prob_trade_hold, prob_dir_hold, er_hold, thr_trade_star, thr_dir_star, CFG["cost_bps"])

    # AUCs on holdout
    t_hold = sample_t[idx_holdout]
    y_trade_hold = y_trade[t_hold].astype(np.int64)
    y_dir_hold = y_dir[t_hold].astype(np.int64)
    trade_auc_hold = _safe_auc_binary(y_trade_hold, prob_trade_hold[:, 1])
    mask_true_trade = (y_trade_hold == 1)
    dir_auc_hold = _safe_auc_binary(y_dir_hold[mask_true_trade], prob_dir_hold[mask_true_trade, 1])

    print("\nFINAL HOLDOUT RESULT (production-fit, fixed thresholds from val_final):")
    print(f"  AUC trade={trade_auc_hold:.3f} | AUC dir(trade-only)={dir_auc_hold:.3f}")
    print(f"  trade_rate(pred)={hold_metrics['trade_rate']:.3f}")
    print(f"  pnl_sum={hold_metrics['pnl_sum']:.4f} | pnl_mean={hold_metrics['pnl_mean']:.6f} | trades={hold_metrics['n_trades']}")
    print(f"  sharpe(per-bar proxy)={hold_metrics['pnl_sharpe']:.3f}")

    print("\nAUC summary (val_final vs holdout):")
    print(f"  TRADE: val_auc={r_trade['val']['auc']:.3f} | holdout_auc={trade_auc_hold:.3f}")
    print(f"  DIR  : val_auc={r_dir['val']['auc']:.3f} | holdout_auc={dir_auc_hold:.3f}")

    print("\nTop-5 val_final threshold candidates:")
    print(sweep_val.head(5)[["thr_trade", "thr_dir", "score", "trade_rate", "pnl_sum", "pnl_sharpe", "n_trades"]])
    print("=" * 80)

    return {
        "thr_trade": thr_trade_star,
        "thr_dir": thr_dir_star,
        "val_true_trade_rate": float(true_val_trade),
        "hold_true_trade_rate": float(true_hold_trade),
        "holdout_trade_auc": float(trade_auc_hold) if np.isfinite(trade_auc_hold) else np.nan,
        "holdout_dir_auc": float(dir_auc_hold) if np.isfinite(dir_auc_hold) else np.nan,
        **hold_metrics,
    }


prod_fit_result = run_production_fit()
print("\nProduction-fit summary dict:")
print(prod_fit_result)

# %% [markdown]
# ## Notes

# %%
"""
Options for selecting a "final" configuration after CV (conceptually):

- "LAST fold" (method 1):
  Most realistic if your production regime is closest to the latest market state.
  Uses that fold's preprocessing (scalers), models, and thresholds.

- "BEST-VAL fold" (method 2):
  Picks the fold whose own VAL threshold-sweep score is best (still VAL-only).
  Often helps if some folds are noisy / bad regime.

- "GLOBAL thresholds" (method 3):
  Keeps the LAST model, but stabilizes threshold selection by aggregating all fold-VAL predictions.
  This can be less brittle than using only the last fold VAL.

Adjacency knobs:
- CFG["adj_mode"] = "emb" or "matrix"
- CFG["alpha_mode"] = "learned" typically works better than fixed alpha
- CFG["adj_l1_lambda"] increases sparsity pressure (on sigmoid(logits) off-diagonal)
- CFG["adj_prior_lambda"] enforces consistency with A_prior from edge_attr
- CFG["prior_use_abs"] controls whether negative corr strengthens adjacency (abs) or weakens it (no abs)
- CFG["adj_temperature"] controls softness of A_learned row-softmax

Temporal knobs:
- CFG["tcn_layers"], CFG["tcn_kernel"], CFG["tcn_dropout"]
- CFG["attn_pool_hidden"] for pooling MLP capacity
"""
