# %% [markdown]
# # Two-stage LOB GNN (SGA-TCN) — VS Code / Jupyter compatible (.py)
# Формат: логические шаги через `# %%`.
# Важные принципы:
# - Scaling только по train (time-ordered, без leakage)
# - Stage A: trade/no-trade (AUC)
# - Stage B: direction на trade-only (AUC на trade-only)
# - Пороги (thr_trade, thr_dir) выбираем ТОЛЬКО на val, на holdout НЕ подбираем
#
# Рекомендовано для Mac M2 (CPU): batch_size=64..128, hidden=64..128, epochs=15..25, AMP отключён на CPU

# %% [markdown]
# ## Step 0 — Imports + reproducibility + config

# %%
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuda seed safe (no-op on cpu-only)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# Для CPU на Mac часто полезно:
torch.set_num_threads(max(1, os.cpu_count() or 4))

CFG: Dict = {
    # data
    "freq": "1min",
    "data_dir": Path("../dataset"),
    "final_test_frac": 0.10,  # 10% final holdout by time

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

    # correlations
    "corr_windows": [6 * 5, 12 * 5, 24 * 5, 48 * 5, 84 * 5],  # 30m,1h,2h,4h,7h
    "edges": [("ADA", "BTC"), ("ADA", "ETH"), ("ETH", "BTC")],

    # triple-barrier
    "tb_horizon": 1 * 15,      # 15 min
    "lookback": 5 * 12 * 5,    # 5 hours => 300
    "tb_pt_mult": 1.2,
    "tb_sl_mult": 1.1,
    "tb_min_barrier": 0.001,
    "tb_max_barrier": 0.006,

    # training (CPU-friendly defaults)
    "batch_size": 64,          # для M2 CPU обычно лучше 64..128
    "epochs": 20,
    "lr": 2e-4,
    "weight_decay": 1e-3,
    "grad_clip": 1.0,
    "dropout": 0.20,

    # model
    "hidden": 96,              # 64..128
    "gnn_layers": 3,

    # --- SGA (spatial)
    "gat_heads": 2,

    # --- TCN (temporal)
    "tcn_channels": 96,
    "tcn_layers": 3,
    "tcn_kernel": 2,
    "tcn_dropout": 0.20,
    "tcn_causal": True,
    "tcn_pool": "last",        # "last" (causal-safe) or "mean"

    # trading eval
    "cost_bps": 1.0,

    # threshold sweep grids (val only)
    "thr_trade_grid": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "thr_dir_grid":   [0.50, 0.55, 0.60, 0.65, 0.70],

    # min trades constraints (чтобы избегать "лучше всего = 0 сделок")
    "eval_min_trades": 50,

    # динамические квантильные пороги для thr_trade (чтобы адаптироваться к некалиброванным p_trade)
    "proxy_target_trades": [50, 100, 200],
}

ASSETS = ["ADA", "BTC", "ETH"]
ASSET2IDX = {a: i for i, a in enumerate(ASSETS)}
TARGET_ASSET = "ETH"
TARGET_NODE = ASSET2IDX[TARGET_ASSET]

EDGE_INDEX = torch.tensor([[ASSET2IDX[s], ASSET2IDX[t]] for (s, t) in CFG["edges"]], dtype=torch.long)  # (E,2)


def add_self_loops_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, dtype=edge_index.dtype).view(-1, 1)
    loops = torch.cat([loops, loops], dim=1)
    return torch.cat([edge_index, loops], dim=0)


EDGE_INDEX = add_self_loops_edge_index(EDGE_INDEX, num_nodes=len(ASSETS))
print("EDGE_INDEX (with self-loops):", EDGE_INDEX.tolist())

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

    df_ada = rename_cols(load_asset("ADA", freq, data_dir, book_levels, part=(0, 80)), "ADA")
    df_btc = rename_cols(load_asset("BTC", freq, data_dir, book_levels, part=(0, 80)), "BTC")
    df_eth = rename_cols(load_asset("ETH", freq, data_dir, book_levels, part=(0, 80)), "ETH")

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
# ## Step 2 — Multi-window correlations → edge features (T,E,W)

# %%
def build_corr_array(df_: pd.DataFrame, corr_windows: List[int], edges: List[Tuple[str, str]]) -> np.ndarray:
    T_ = len(df_)
    n_edges = len(edges)
    n_w = len(corr_windows)

    out = np.zeros((T_, n_edges, n_w), dtype=np.float32)

    # для твоего набора edges = (ADA,BTC), (ADA,ETH), (ETH,BTC) — оставляем быстро и явно
    # если захочешь обобщить, можно сделать циклом с rolling().corr()
    for wi, w in enumerate(corr_windows):
        r_ada_btc = df_["lr_ADA"].rolling(w, min_periods=1).corr(df_["lr_BTC"])
        r_ada_eth = df_["lr_ADA"].rolling(w, min_periods=1).corr(df_["lr_ETH"])
        r_eth_btc = df_["lr_ETH"].rolling(w, min_periods=1).corr(df_["lr_BTC"])

        out[:, 0, wi] = np.nan_to_num(r_ada_btc)
        out[:, 1, wi] = np.nan_to_num(r_ada_eth)
        out[:, 2, wi] = np.nan_to_num(r_eth_btc)

    return out


corr_array = build_corr_array(df, CFG["corr_windows"], CFG["edges"])
edge_feat = np.nan_to_num(corr_array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print("corr_array shape:", corr_array.shape, "(T,E,W)")
print("edge_feat sample [t=100, all edges, all windows]:\n", edge_feat[100])

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
y_trade = (y_tb != 1).astype(np.int64)   # 1=trade, 0=no-trade
y_dir = (y_tb == 2).astype(np.int64)     # 1=up, 0=down (meaningful only when y_trade==1)

dist = np.bincount(y_tb, minlength=3)
print("TB dist [down,flat,up]:", dist)
print("Trade ratio (true):", float(y_trade.mean()))

# %% [markdown]
# ## Step 4 — Build node tensor (T,N,F) + sample_t (valid indices in sample-space)

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

# quick feature sanity
print("Feature stats (TARGET asset, lr):",
      "mean=", float(X_node_raw[:, TARGET_NODE, node_feat_names.index("lr")].mean()),
      "std=", float(X_node_raw[:, TARGET_NODE, node_feat_names.index("lr")].std()))

# %% [markdown]
# ## Step 5 — Final holdout split (time-ordered) + walk-forward splits (CV-part only)

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
      e_seq: (L,E,W)
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
        e_seq = self.E_feat[t0:t + 1]  # (L,E,W)

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
        torch.stack(es, 0),   # (B,L,E,W)
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
    """
    Fit scaler on all times up to last train sample time (no leakage).
    """
    last_train_t = int(sample_t_[int(idx_train[-1])])
    train_time_mask = np.arange(0, last_train_t + 1)

    X_train_time = X_node_raw_[train_time_mask]  # (Ttr,N,F)
    Ttr, N, Fdim = X_train_time.shape

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    scaler.fit(X_train_time.reshape(-1, Fdim))

    X_scaled = scaler.transform(X_node_raw_.reshape(-1, Fdim)).reshape(X_node_raw_.shape).astype(np.float32)
    X_scaled = np.clip(X_scaled, -max_abs, max_abs).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X_scaled, scaler


def subset_trade_indices(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> np.ndarray:
    tt = sample_t_[indices]
    mask = (y_trade_arr[tt] == 1)
    return indices[mask]


def split_trade_ratio(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> float:
    tt = sample_t_[indices]
    return float(y_trade_arr[tt].mean()) if len(tt) else float("nan")


# %% [markdown]
# ## Step 7 — Model (SGA-TCN) — drop-in logits (B,2)

# %%
class SpatialGraphAttentionLayer(nn.Module):
    """
    Graph Attention with edge_attr in attention scorer:
      score_e = a^T [h_src || h_dst || edge_emb]
      attn normalized per-dst over incoming edges
      msg = W_msg(h_src)
      agg_dst = sum(attn * msg)
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = max(1, int(heads))
        self.dropout = float(dropout)

        self.head_dim = max(1, int(math.ceil(out_dim / self.heads)))
        self.inner_dim = self.heads * self.head_dim

        self.lin_node = nn.Linear(self.in_dim, self.inner_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, self.inner_dim, bias=False)
        self.lin_msg = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

        self.attn_vec = nn.Parameter(torch.empty(self.heads, 3 * self.head_dim))

        self.out_proj = nn.Linear(self.inner_dim, self.out_dim, bias=False)
        self.res_proj = nn.Identity() if self.in_dim == self.out_dim else nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.ln = nn.LayerNorm(self.out_dim)
        self.attn_drop = nn.Dropout(self.dropout)
        self.out_drop = nn.Dropout(self.dropout)
        self.act = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in [self.lin_node, self.lin_edge, self.lin_msg, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,Fin)
        edge_attr: (B,E_attr,W)  (может быть меньше чем E_index из-за self-loops)
        edge_index: (E_index,2)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        B, N, _ = x.shape
        E_index = edge_index.shape[0]
        E_attr = edge_attr.shape[1]
        W = edge_attr.shape[2]

        # pad edge_attr for self-loops if needed
        if E_attr < E_index:
            pad = torch.zeros((B, E_index - E_attr, W), device=edge_attr.device, dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif E_attr > E_index:
            edge_attr = edge_attr[:, :E_index, :]

        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]

        h = self.lin_node(x).view(B, N, self.heads, self.head_dim)           # (B,N,Hh,dh)
        eemb = self.lin_edge(edge_attr).view(B, E_index, self.heads, self.head_dim)  # (B,E,Hh,dh)

        h_src = h[:, src_idx, :, :]  # (B,E,Hh,dh)
        h_dst = h[:, dst_idx, :, :]  # (B,E,Hh,dh)

        cat = torch.cat([h_src, h_dst, eemb], dim=-1)  # (B,E,Hh,3*dh)
        scores = (cat * self.attn_vec[None, None, :, :]).sum(dim=-1)         # (B,E,Hh)
        scores = self.act(scores)

        # softmax per dst-node (N маленький => простой цикл ок)
        alphas = torch.zeros_like(scores)
        for n in range(N):
            mask = (dst_idx == n)
            if int(mask.sum()) == 0:
                continue
            s = scores[:, mask, :]
            a = torch.softmax(s, dim=1)
            a = self.attn_drop(a)
            alphas[:, mask, :] = a

        msg = self.lin_msg(h_src.reshape(B, E_index, self.inner_dim)).view(B, E_index, self.heads, self.head_dim)

        agg = torch.zeros((B, N, self.heads, self.head_dim), device=x.device, dtype=x.dtype)
        for e_i in range(E_index):
            dst = int(dst_idx[e_i].item())
            agg[:, dst, :, :] += alphas[:, e_i, :].unsqueeze(-1) * msg[:, e_i, :, :]

        out = agg.reshape(B, N, self.inner_dim)
        out = self.out_proj(out)
        out = self.out_drop(out)

        res = self.res_proj(x)
        y = self.ln(res + out)
        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


class SpatialGraphAttentionMP(nn.Module):
    """Applies SpatialGraphAttentionLayer independently at each timestep."""
    def __init__(self, in_dim: int, hidden: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        self.gat = SpatialGraphAttentionLayer(in_dim=in_dim, out_dim=hidden, edge_dim=edge_dim, heads=heads, dropout=dropout)

    def forward(self, x_seq: torch.Tensor, e_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, L_, N, F_ = x_seq.shape
        x_flat = x_seq.reshape(B * L_, N, F_)
        e_flat = e_seq.reshape(B * L_, e_seq.size(2), e_seq.size(3))
        h_flat = self.gat(x_flat, e_flat, edge_index)  # (B*L,N,H)
        return h_flat.reshape(B, L_, N, -1)


class CausalConv1d(nn.Module):
    """Causal Conv1d: pads only on the left => no future leakage."""
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


class GNN_TCN_Classifier(nn.Module):
    """
    input:
      x_seq (B,L,N,F), e_seq (B,L,E,W), edge_index (E,2)
    output:
      logits (B,2)
    """
    def __init__(self, node_in: int, edge_dim: int, cfg: Dict, target_node: int, n_classes: int = 2):
        super().__init__()
        self.target_node = int(target_node)

        hidden = int(cfg["hidden"])
        dropout = float(cfg["dropout"])

        gat_heads = int(cfg["gat_heads"])
        tcn_channels = int(cfg["tcn_channels"])
        tcn_layers_n = int(cfg["tcn_layers"])
        tcn_kernel = int(cfg["tcn_kernel"])
        tcn_dropout = float(cfg["tcn_dropout"])
        tcn_causal = bool(cfg["tcn_causal"])
        self.tcn_pool = str(cfg["tcn_pool"])

        # spatial stack
        self.gnns = nn.ModuleList()
        for i in range(int(cfg["gnn_layers"])):
            in_dim = int(node_in) if i == 0 else hidden
            self.gnns.append(SpatialGraphAttentionMP(in_dim=in_dim, hidden=hidden, edge_dim=int(edge_dim),
                                                     heads=gat_heads, dropout=dropout))

        # temporal
        self.tcn_in = nn.Linear(hidden, tcn_channels)
        self.tcn = TemporalConvNet(
            in_ch=tcn_channels,
            channels=[tcn_channels] * tcn_layers_n,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
            causal=tcn_causal,
        )

        # head
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

    def forward(self, x: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

        h = x
        for gnn in self.gnns:
            h = gnn(h, e, edge_index)  # (B,L,N,H)

        h_tgt = h[:, :, self.target_node, :]  # (B,L,H)
        z = self.tcn_in(h_tgt)                # (B,L,C)
        z = z.transpose(1, 2)                 # (B,C,L)

        y = self.tcn(z)                       # (B,C,L)
        emb = y[:, :, -1] if self.tcn_pool == "last" else y.mean(dim=-1)
        logits = self.head(emb)               # (B,2)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


# quick sanity check
B_ = 2
Fdim = X_node_raw.shape[-1]
E_ = EDGE_INDEX.shape[0]
W_ = edge_feat.shape[-1]
x_dummy = torch.randn(B_, L, len(ASSETS), Fdim)
e_dummy = torch.randn(B_, L, E_, W_)
m_dummy = GNN_TCN_Classifier(node_in=Fdim, edge_dim=W_, cfg=CFG, target_node=TARGET_NODE).to(DEVICE)
with torch.no_grad():
    out = m_dummy(x_dummy.to(DEVICE), e_dummy.to(DEVICE), EDGE_INDEX.to(DEVICE))
print("Model sanity logits:", out.shape, "| finite:", bool(torch.isfinite(out).all().item()))

# %% [markdown]
# ## Step 8 — Train/Eval helpers (AUC-oriented)

# %%
@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, y_key: str) -> Dict:
    model.eval()
    total_loss = 0.0
    n = 0

    ys = []
    probs = []
    ers = []

    for x, e, y_trade_b, y_dir_b, er in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        y = (y_trade_b if y_key == "trade" else y_dir_b).to(DEVICE).long()

        logits = model(x, e, EDGE_INDEX.to(DEVICE))
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * int(y.size(0))
        n += int(y.size(0))

        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        probs.append(p)
        ers.append(er.detach().cpu().numpy())

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

    return {
        "loss": total_loss / max(1, n),
        "acc": float(acc),
        "f1m": float(f1m),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "cm": cm,
        "y": ys,
        "prob": probs,
        "er": ers,
    }


def make_ce_weights_binary(y_np: np.ndarray) -> torch.Tensor:
    y_np = np.asarray(y_np, dtype=np.int64)
    counts = np.bincount(y_np, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_binary_classifier(
    X_scaled: np.ndarray,
    edge_feat_: np.ndarray,
    y_trade_arr: np.ndarray,
    y_dir_arr: np.ndarray,
    exit_ret_arr: np.ndarray,
    sample_t_: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict,
    stage_name: str,  # "trade" or "dir"
) -> Tuple[nn.Module, Dict]:
    assert stage_name in ("trade", "dir")

    L_ = int(cfg["lookback"])
    bs = int(cfg["batch_size"])

    tr_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_feat_, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_train, L_)
    va_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_feat_, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_val,   L_)
    te_ds = LobGraphSequenceDataset2Stage(X_scaled, edge_feat_, y_trade_arr, y_dir_arr, exit_ret_arr, sample_t_, idx_test,  L_)

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_fn_2stage, num_workers=0)

    node_in = int(X_scaled.shape[-1])
    edge_dim = int(edge_feat_.shape[-1])
    model = GNN_TCN_Classifier(node_in=node_in, edge_dim=edge_dim, cfg=cfg, target_node=TARGET_NODE).to(DEVICE)

    # class weights from TRAIN only
    t_train = sample_t_[idx_train]
    y_train_np = (y_trade_arr[t_train] if stage_name == "trade" else y_dir_arr[t_train]).astype(np.int64)
    ce_w = make_ce_weights_binary(y_train_np)
    loss_fn = nn.CrossEntropyLoss(weight=ce_w)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
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
            logits = model(x, e, EDGE_INDEX.to(DEVICE))
            loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()

            tot_loss += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))

        tr_loss = tot_loss / max(1, n)

        va = eval_binary(model, va_loader, loss_fn, y_key=stage_name)
        va_auc = va["auc"]
        sel_auc = float(va_auc) if np.isfinite(va_auc) else -1e18

        if sel_auc > best_auc:
            best_auc = sel_auc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        sch.step(sel_auc)

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[{stage_name}] ep {ep:02d} lr={lr_now:.2e} "
            f"tr_loss={tr_loss:.4f} va_loss={va['loss']:.4f} va_auc={va_auc:.3f} "
            f"best={best_auc:.3f}@ep{best_epoch:02d}"
        )

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    va = eval_binary(model, va_loader, loss_fn, y_key=stage_name)
    te = eval_binary(model, te_loader, loss_fn, y_key=stage_name)

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
) -> Dict:
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
        "pnl_sharpe": float((pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(288)) if n else np.nan,  # per-bar proxy
    }


def sweep_thresholds(
    prob_trade: np.ndarray,
    prob_dir: np.ndarray,
    exit_ret_arr: np.ndarray,
    cfg: Dict,
    min_trades: int = 0,
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

    rows = []
    for thr_t in thr_trade_grid:
        for thr_d in thr_dir_grid:
            m = two_stage_pnl_by_threshold(prob_trade, prob_dir, exit_ret_arr, thr_t, thr_d, cfg["cost_bps"])
            if int(m["n_trades"]) < int(min_trades):
                continue
            rows.append({"thr_trade": float(thr_t), "thr_dir": float(thr_d), **m})

    if not rows and min_trades > 0:
        # soften constraint
        return sweep_thresholds(prob_trade, prob_dir, exit_ret_arr, cfg, min_trades=1)

    if not rows:
        # last fallback: allow 0 trades
        for thr_t in thr_trade_grid:
            for thr_d in thr_dir_grid:
                m = two_stage_pnl_by_threshold(prob_trade, prob_dir, exit_ret_arr, thr_t, thr_d, cfg["cost_bps"])
                rows.append({"thr_trade": float(thr_t), "thr_dir": float(thr_d), **m})

    df_ = pd.DataFrame(rows).sort_values(["pnl_sum", "pnl_mean"], ascending=False)
    return df_


@torch.no_grad()
def predict_probs_on_indices(model: nn.Module, X_scaled: np.ndarray, edge_feat_: np.ndarray, indices: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ds = LobGraphSequenceDataset2Stage(X_scaled, edge_feat_, y_trade, y_dir, exit_ret, sample_t, indices, cfg["lookback"])
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_2stage, num_workers=0)

    model.eval()
    probs = []
    ers = []
    for x, e, _yt, _yd, er in loader:
        x = x.to(DEVICE).float()
        e = e.to(DEVICE).float()
        logits = model(x, e, EDGE_INDEX.to(DEVICE))
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
        ers.append(er.cpu().numpy())

    return np.concatenate(probs, axis=0), np.concatenate(ers, axis=0)


# %% [markdown]
# ## Step 10 — Run walk-forward folds (CV-part): train trade → train dir → test PnL + trade share

# %%
def run_walk_forward_cv() -> pd.DataFrame:
    rows = []

    for fi, (idx_tr, idx_va, idx_te) in enumerate(walk_splits, 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fi}/{len(walk_splits)} sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
        print(f"True trade ratio (val):  {split_trade_ratio(idx_va, sample_t, y_trade):.3f}")
        print(f"True trade ratio (test): {split_trade_ratio(idx_te, sample_t, y_trade):.3f}")

        # scale per fold (fit only on train timeline)
        X_scaled, _scaler = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_tr, max_abs=CFG["max_abs_feat"])

        # Stage A
        m_trade, r_trade = train_binary_classifier(
            X_scaled, edge_feat, y_trade, y_dir, exit_ret, sample_t,
            idx_tr, idx_va, idx_te, CFG, stage_name="trade"
        )

        # Stage B (train on trade-only indices)
        idx_tr_T = subset_trade_indices(idx_tr, sample_t, y_trade)
        idx_va_T = subset_trade_indices(idx_va, sample_t, y_trade)
        idx_te_T = subset_trade_indices(idx_te, sample_t, y_trade)

        if len(idx_tr_T) < max(200, 2 * CFG["batch_size"]) or len(idx_te_T) < 50:
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
            })
            continue

        m_dir, r_dir = train_binary_classifier(
            X_scaled, edge_feat, y_trade, y_dir, exit_ret, sample_t,
            idx_tr_T, idx_va_T, idx_te_T, CFG, stage_name="dir"
        )

        # Choose thresholds on VAL (важно!)
        prob_trade_val, er_val = predict_probs_on_indices(m_trade, X_scaled, edge_feat, idx_va, CFG)
        prob_dir_val, _ = predict_probs_on_indices(m_dir, X_scaled, edge_feat, idx_va, CFG)
        sweep_val = sweep_thresholds(prob_trade_val, prob_dir_val, er_val, CFG, min_trades=int(CFG["eval_min_trades"]))
        best_val = sweep_val.iloc[0].to_dict()

        thr_trade_star = float(best_val["thr_trade"])
        thr_dir_star = float(best_val["thr_dir"])

        val_metrics = two_stage_pnl_by_threshold(prob_trade_val, prob_dir_val, er_val, thr_trade_star, thr_dir_star, CFG["cost_bps"])
        print("\nChosen thresholds (from VAL):")
        print(f"  thr_trade*={thr_trade_star:.3f} thr_dir*={thr_dir_star:.3f} | val trade_rate(pred)={val_metrics['trade_rate']:.3f} | val pnl_sum={val_metrics['pnl_sum']:.4f}")

        # Evaluate on TEST with fixed thresholds from VAL
        prob_trade_te, er_te = predict_probs_on_indices(m_trade, X_scaled, edge_feat, idx_te, CFG)
        prob_dir_te, _ = predict_probs_on_indices(m_dir, X_scaled, edge_feat, idx_te, CFG)
        te_metrics = two_stage_pnl_by_threshold(prob_trade_te, prob_dir_te, er_te, thr_trade_star, thr_dir_star, CFG["cost_bps"])

        print("TEST (fixed thr from VAL):")
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
        })

    return pd.DataFrame(rows)


cv_summary = run_walk_forward_cv()
print("\n" + "=" * 80)
print("CV summary (fold TEST, fixed thresholds from VAL):")
print(cv_summary)
print("\nMeans:")
print(cv_summary.mean(numeric_only=True))

# %% [markdown]
# ## Step 11 — Final train on CV(90%) and evaluate on FINAL holdout (10%)
# Здесь печатаем:
# - AUC по trade/dir
# - пороги, выбранные на val_final
# - долю трейдинга на holdout при текущих параметрах (trade_rate(pred))
# - и oracle на holdout (только как верхняя оценка, НЕ использовать для выбора)

# %%
def run_final_train_holdout() -> None:
    print("\n" + "=" * 80)
    print("FINAL TRAIN/VAL on CV-part (90%) -> EVAL on FINAL holdout (10%)")

    # final train/val split inside CV-part
    val_w = max(1, int(CFG["val_window_frac"] * n_samples_cv))
    train_end = n_samples_cv - val_w

    idx_train_final = np.arange(0, train_end, dtype=np.int64)
    idx_val_final = np.arange(train_end, n_samples_cv, dtype=np.int64)
    idx_holdout = idx_final_test.astype(np.int64)

    print("Final split sizes:")
    print("  train_final:", len(idx_train_final))
    print("  val_final  :", len(idx_val_final))
    print("  holdout    :", len(idx_holdout))
    print(f"True trade ratio (val_final):  {split_trade_ratio(idx_val_final, sample_t, y_trade):.3f}")
    print(f"True trade ratio (holdout):    {split_trade_ratio(idx_holdout, sample_t, y_trade):.3f}")

    # scaling on train_final timeline only
    X_scaled_final, _ = fit_scale_nodes_train_only(X_node_raw, sample_t, idx_train_final, max_abs=CFG["max_abs_feat"])

    # Stage A
    m_trade, r_trade = train_binary_classifier(
        X_scaled_final, edge_feat, y_trade, y_dir, exit_ret, sample_t,
        idx_train_final, idx_val_final, idx_holdout, CFG, stage_name="trade"
    )

    # Stage B on trade-only
    idx_train_T = subset_trade_indices(idx_train_final, sample_t, y_trade)
    idx_val_T = subset_trade_indices(idx_val_final, sample_t, y_trade)
    idx_hold_T = subset_trade_indices(idx_holdout, sample_t, y_trade)

    print("\nTrade-only sizes for DIR:")
    print("  train_final_T:", len(idx_train_T))
    print("  val_final_T  :", len(idx_val_T))
    print("  holdout_T    :", len(idx_hold_T))

    m_dir, r_dir = train_binary_classifier(
        X_scaled_final, edge_feat, y_trade, y_dir, exit_ret, sample_t,
        idx_train_T, idx_val_T, idx_hold_T, CFG, stage_name="dir"
    )

    # choose thresholds on val_final
    prob_trade_val, er_val = predict_probs_on_indices(m_trade, X_scaled_final, edge_feat, idx_val_final, CFG)
    prob_dir_val, _ = predict_probs_on_indices(m_dir, X_scaled_final, edge_feat, idx_val_final, CFG)
    sweep_val = sweep_thresholds(prob_trade_val, prob_dir_val, er_val, CFG, min_trades=int(CFG["eval_min_trades"]))
    best_val = sweep_val.iloc[0].to_dict()
    thr_trade_star = float(best_val["thr_trade"])
    thr_dir_star = float(best_val["thr_dir"])

    val_metrics = two_stage_pnl_by_threshold(prob_trade_val, prob_dir_val, er_val, thr_trade_star, thr_dir_star, CFG["cost_bps"])

    print("\nChosen thresholds on val_final:")
    print(f"  thr_trade*={thr_trade_star:.3f}")
    print(f"  thr_dir*  ={thr_dir_star:.3f}")
    print(f"  val trade_rate(pred)={val_metrics['trade_rate']:.3f} | val pnl_sum={val_metrics['pnl_sum']:.4f} | val pnl_mean={val_metrics['pnl_mean']:.6f} | trades={val_metrics['n_trades']}")

    # evaluate holdout with fixed thresholds from val_final
    prob_trade_hold, er_hold = predict_probs_on_indices(m_trade, X_scaled_final, edge_feat, idx_holdout, CFG)
    prob_dir_hold, _ = predict_probs_on_indices(m_dir, X_scaled_final, edge_feat, idx_holdout, CFG)
    hold_metrics = two_stage_pnl_by_threshold(prob_trade_hold, prob_dir_hold, er_hold, thr_trade_star, thr_dir_star, CFG["cost_bps"])

    print("\nFINAL HOLDOUT RESULT (fixed thresholds from val_final):")
    print(f"  trade_rate(pred)={hold_metrics['trade_rate']:.3f}  <-- доля трейдинга на holdout при текущих параметрах")
    print(f"  pnl_sum={hold_metrics['pnl_sum']:.4f} | pnl_mean={hold_metrics['pnl_mean']:.6f} | trades={hold_metrics['n_trades']}")
    print(f"  sharpe(per-bar proxy)={hold_metrics['pnl_sharpe']:.3f}")

    # oracle (DO NOT USE for selection)
    sweep_hold_oracle = sweep_thresholds(prob_trade_hold, prob_dir_hold, er_hold, CFG, min_trades=int(CFG["eval_min_trades"]))
    best_hold_oracle = sweep_hold_oracle.iloc[0].to_dict()
    print("\n[ORACLE] best possible on holdout by sweeping thresholds (DO NOT USE for selection):")
    print(f"  thr_trade={best_hold_oracle['thr_trade']:.3f} thr_dir={best_hold_oracle['thr_dir']:.3f}")
    print(f"  trade_rate(pred)={best_hold_oracle['trade_rate']:.3f} | pnl_sum={best_hold_oracle['pnl_sum']:.4f} | trades={int(best_hold_oracle['n_trades'])}")

    # quick AUC summary
    print("\nAUC summary:")
    print(f"  TRADE: val_auc={r_trade['val']['auc']:.3f} | holdout_auc={r_trade['test']['auc']:.3f}")
    print(f"  DIR  : val_auc={r_dir['val']['auc']:.3f} | holdout_auc={r_dir['test']['auc']:.3f}")


run_final_train_holdout()

# %% [markdown]
# ## Notes (коротко)
# - “доля трейдинга на тестовом датасете при текущих параметрах” выводится как `trade_rate(pred)`:
#   - на fold-test: “TEST (fixed thr from VAL)”
#   - на финальном holdout: “FINAL HOLDOUT RESULT … trade_rate(pred)=…”
#
# Если захочешь — в следующем шаге могу:
# 1) добавить калибровку вероятностей (temperature scaling) только по val,
# 2) сделать альтернативный objective для threshold selection (например max pnl_sum при фикс. trade_rate),
# 3) или сделать single-head 3-class (down/flat/up) с метрикой AUC/PR-AUC по trade и down/up отдельно.
