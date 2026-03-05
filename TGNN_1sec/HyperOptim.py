# %% [markdown]
# # Ultra-fast Mac M2 notebook (no pandas rolling.corr)
#
# Что именно делаем для скорости (без потери логики):
# 1) **Убираем pandas `rolling().corr()`** полностью.
# 2) Считаем edge-features **только для моментов, которые реально участвуют в обучении** (`sample_t`),
#    и **только `e_last`** (потому что модель использует только последний срез для adjacency prior).
# 3) Edge-features кешируем на диск как `float16` memmap (≈150–200MB вместо сотен/гигабайт).
# 4) Сохраняем остальные speed-фиксы: укороченный lookback + stride, subsample train/val, eval реже.
#
# Формат: код с `# %%` как в ноутбуке.

# %% ======================================================================
# Step 0: Imports, device, seed, config
# ======================================================================

import os
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

# Optional: accelerate triple-barrier loop
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(100)

# Prefer Apple Silicon GPU via Metal (MPS)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", DEVICE)

# Threads: too many on Mac can hurt
cpu_cnt = os.cpu_count() or 4
torch.set_num_threads(min(8, max(1, cpu_cnt)))

USE_AMP = (DEVICE.type == "mps")
print("USE_AMP:", USE_AMP, "| NUMBA:", NUMBA_AVAILABLE)

CFG: Dict[str, Any] = {
    # ----------------------
    # data
    # ----------------------
    "freq": "1sec",
    "data_dir": "../dataset",
    "final_test_frac": 0.10,
    "data_part": (0, 75),         # (0,100) when ready
    "cache_dir": "./cache_1s",

    # ----------------------
    # assets / target
    # ----------------------
    "assets": ["ADA", "BTC", "ETH"],
    "target_asset": "ETH",

    # ----------------------
    # order book
    # ----------------------
    "book_levels": 15,
    "top_levels": 5,
    "near_levels": 5,

    # ----------------------
    # walk-forward windows
    # ----------------------
    "train_min_frac": 0.55,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,
    "max_folds": 1,               # Mac iteration: 1; full CV later: 3

    # ----------------------
    # edge features (FAST rolling corr)  <<< MAIN CHANGE
    # ----------------------
    "corr_windows": [60, 180],            # seconds
    "corr_lags": [0, 1, 2, 5, 10, 30],     # seconds
    # If you insist on 1–10 minute lags, add: [60,120,180,300,600] (will slow feature build)
    "edges_mode": "all_pairs",
    "add_self_loops": True,
    "edge_transform": "fisher",           # fisher z on corr
    "edge_dtype": "float16",              # cache dtype

    # ----------------------
    # triple-barrier labels
    # ----------------------
    "tb_horizon": 300,          # 5 minutes
    "tb_vol_window": 900,       # decoupled from model lookback
    "tb_pt_mult": 1.70,
    "tb_sl_mult": 1.70,
    "tb_min_barrier": 0.0014,
    "tb_max_barrier": 0.0060,

    # ----------------------
    # fixed-horizon return head
    # ----------------------
    "fixed_horizon": 300,
    "fixed_ret_clip": 0.0040,

    # ----------------------
    # model input length (KEY SPEED)
    # ----------------------
    "model_lookback": 240,      # seconds
    "frame_stride": 2,          # effective length ~120

    # ----------------------
    # training speed controls (KEY SPEED)
    # ----------------------
    "batch_size": 64,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 5e-3,
    "grad_clip": 1.0,
    "dropout": 0.35,

    "train_samples_per_epoch": 25_000,
    "val_samples_per_eval": 10_000,
    "eval_every": 2,
    "patience": 6,

    "use_onecycle": True,
    "onecycle_pct_start": 0.20,
    "onecycle_div_factor": 40.0,
    "onecycle_final_div": 800.0,

    # ----------------------
    # model (smaller)
    # ----------------------
    "gwn_residual_channels": 32,
    "gwn_dilation_channels": 32,
    "gwn_skip_channels": 128,
    "gwn_end_channels": 128,
    "gwn_blocks": 1,
    "gwn_layers_per_block": 7,
    "gwn_kernel_size": 2,

    # ----------------------
    # adaptive adjacency
    # ----------------------
    "adj_emb_dim": 16,
    "adj_temperature": 1.25,
    "adaptive_topk": 3,
    "adj_l1_lambda": 3e-3,
    "adj_prior_lambda": 2e-2,
    "prior_use_abs": True,
    "prior_diag_boost": 1.0,
    "prior_row_normalize": True,

    # ----------------------
    # loss weights + utility
    # ----------------------
    "loss_w_trade": 0.65,
    "loss_w_dir": 0.80,
    "loss_w_ret": 0.25,
    "loss_w_utility": 0.55,

    "ret_huber_delta": 0.0015,

    "utility_k": 3.0,
    "utility_scale": 550.0,
    "utility_mask_true_trades": False,
    "trade_prob_penalty": 2.0,

    # ----------------------
    # thresholds (subsampled sweep at end)
    # ----------------------
    "cost_bps": 1.0,
    "thr_trade_grid": [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97],
    "thr_dir_grid":   [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "eval_min_trades": 300,
    "max_trade_rate_val": 0.02,
    "trade_rate_penalty": 8.0,
    "thr_objective": "pnl_sum",
    "proxy_target_trades": [200, 500, 1000],
    "sel_b_dir_auc": 0.55,

    # ----------------------
    # artifacts / cache
    # ----------------------
    "artifact_dir": "./artifacts_1s",
}

ASSETS = list(CFG["assets"])
ASSET2IDX = {a: i for i, a in enumerate(ASSETS)}
TARGET_ASSET = str(CFG["target_asset"])
TARGET_NODE = ASSET2IDX[TARGET_ASSET]

ART_DIR = Path(CFG["artifact_dir"])
ART_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(CFG["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Assets:", ASSETS, "| Target:", TARGET_ASSET)
print("Artifacts dir:", str(ART_DIR.resolve()))
print("Cache dir:", str(CACHE_DIR.resolve()))

# %% ======================================================================
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

# %% ======================================================================
# Step 1: Data loading (1-second)
# ======================================================================

def load_asset(asset: str, freq: str, data_dir: Path, book_levels: int, part: Tuple[int, int]) -> pd.DataFrame:
    path = data_dir / f"{asset}_{freq}.csv"
    df = pd.read_csv(path)
    df = df.iloc[int(len(df) * part[0] / 100): int(len(df) * part[1] / 100)]

    ts = pd.to_datetime(df["system_time"], utc=True)
    df["timestamp"] = ts.dt.floor("S")
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.groupby(level=0).last()  # one row per second

    bid_cols = [f"bids_notional_{i}" for i in range(book_levels)]
    ask_cols = [f"asks_notional_{i}" for i in range(book_levels)]
    needed = ["midpoint", "spread", "buys", "sells"] + bid_cols + ask_cols

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return df[needed]


def load_all_assets() -> pd.DataFrame:
    freq = CFG["freq"]
    data_dir = Path(CFG["data_dir"])
    book_levels = int(CFG["book_levels"])
    part = tuple(CFG.get("data_part", (0, 100)))

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

    df_a = rename_cols(load_asset("ADA", freq, data_dir, book_levels, part=part), "ADA")
    df_b = rename_cols(load_asset("BTC", freq, data_dir, book_levels, part=part), "BTC")
    df_e = rename_cols(load_asset("ETH", freq, data_dir, book_levels, part=part), "ETH")

    df = df_a.join(df_b, how="inner").join(df_e, how="inner").reset_index()
    df = df.rename(columns={"index": "timestamp"})

    for a in ASSETS:
        px = pd.to_numeric(df[a], errors="coerce").replace(0, np.nan).ffill()
        df[a] = px
        df[f"lr_{a}"] = np.log(px).diff().fillna(0.0).astype(np.float32)

    return df


t0 = time.time()
df = load_all_assets()
print("Loaded df:", df.shape)
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
print(df.head(2))
print(f"Load time: {time.time() - t0:.2f}s")

# %% ======================================================================
# Step 2: Node features + caching
# ======================================================================

EPS = 1e-6

def safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def build_node_tensor(df_: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    book_levels = int(CFG["book_levels"])
    top_k = int(CFG["top_levels"])
    near_k = int(CFG["near_levels"])

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


node_cache_path = CACHE_DIR / f"X_node_T{len(df)}.npy"
if node_cache_path.exists():
    X_node_raw = np.load(node_cache_path, mmap_mode="r")
    node_feat_names = [
        "lr","spread","log_buys","log_sells","ofi","DI_15",
        "DI_L0","DI_L1","DI_L2","DI_L3","DI_L4",
        "near_ratio_bid","near_ratio_ask","di_near","di_far"
    ]
    print("Loaded cached X_node_raw:", X_node_raw.shape)
else:
    t0 = time.time()
    X_node_raw, node_feat_names = build_node_tensor(df)
    np.save(node_cache_path, X_node_raw)
    print("Built X_node_raw:", X_node_raw.shape)
    print(f"Node feat build time: {time.time() - t0:.2f}s")

# %% ======================================================================
# Step 3: Labels (TB + fixed return) FAST
# ======================================================================

def fixed_horizon_future_return_fast(lr: np.ndarray, H: int) -> np.ndarray:
    lr = np.asarray(lr, dtype=np.float64)
    T = lr.shape[0]
    out = np.zeros(T, dtype=np.float32)
    if H <= 0 or T < (H + 2):
        return out
    cs = np.zeros(T + 1, dtype=np.float64)
    cs[1:] = np.cumsum(lr)
    end_idx = T - H - 1
    out[:end_idx] = (cs[1 + H : 1 + H + end_idx] - cs[1 : 1 + end_idx]).astype(np.float32)
    return out


if NUMBA_AVAILABLE:
    @njit
    def _tb_loop_numba(lr_np, thr_np, horizon, pt_mult, sl_mult):
        T = lr_np.shape[0]
        y = np.ones(T, dtype=np.int64)
        exit_ret = np.zeros(T, dtype=np.float32)
        exit_t = np.arange(T, dtype=np.int64)

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
                    hit = 2
                    et = t + dt
                    er = cum
                    break
                if cum <= dn:
                    hit = 0
                    et = t + dt
                    er = cum
                    break

            if hit == 1:
                s = 0.0
                for k in range(t + 1, t + horizon + 1):
                    s += lr_np[k]
                er = s
                et = t + horizon

            y[t] = hit
            exit_ret[t] = er
            exit_t[t] = et

        return y, exit_ret, exit_t


def triple_barrier_labels_from_lr(
    lr: pd.Series,
    horizon: int,
    vol_window: int,
    pt_mult: float,
    sl_mult: float,
    min_barrier: float,
    max_barrier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lr = lr.astype(float).copy()
    T = len(lr)

    vol = lr.rolling(vol_window, min_periods=max(10, vol_window // 10)).std().shift(1)
    thr = (vol * np.sqrt(horizon)).clip(lower=min_barrier, upper=max_barrier)

    lr_np = lr.fillna(0.0).to_numpy(dtype=np.float64)
    thr_np = thr.fillna(min_barrier).to_numpy(dtype=np.float64)

    if NUMBA_AVAILABLE:
        y, exit_ret, exit_t = _tb_loop_numba(lr_np, thr_np, int(horizon), float(pt_mult), float(sl_mult))
        return y, exit_ret, exit_t, thr_np

    # fallback (slow)
    y = np.ones(T, dtype=np.int64)
    exit_ret = np.zeros(T, dtype=np.float32)
    exit_t = np.arange(T, dtype=np.int64)

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
            er = float(np.sum(lr_np[t + 1 : t + horizon + 1]))
            et = t + horizon

        y[t] = hit
        exit_ret[t] = er
        exit_t[t] = et

    return y, exit_ret, exit_t, thr_np


t0 = time.time()
y_tb, exit_ret, exit_t, tb_thr = triple_barrier_labels_from_lr(
    df["lr_ETH"],
    horizon=int(CFG["tb_horizon"]),
    vol_window=int(CFG["tb_vol_window"]),
    pt_mult=float(CFG["tb_pt_mult"]),
    sl_mult=float(CFG["tb_sl_mult"]),
    min_barrier=float(CFG["tb_min_barrier"]),
    max_barrier=float(CFG["tb_max_barrier"]),
)
print(f"TB build time: {time.time() - t0:.2f}s (numba={NUMBA_AVAILABLE})")

y_trade = (y_tb != 1).astype(np.int64)
y_dir = (y_tb == 2).astype(np.int64)

fixed_ret = fixed_horizon_future_return_fast(df["lr_ETH"].to_numpy(dtype=np.float64), int(CFG["fixed_horizon"]))

dist = np.bincount(y_tb, minlength=3)
print("TB dist [down,flat,up]:", dist)
print("True trade ratio:", float(y_trade.mean()))
print("fixed_ret stats: mean=", float(np.mean(fixed_ret)), "std=", float(np.std(fixed_ret)))

# %% ======================================================================
# Step 4: sample_t indices (training samples)
# ======================================================================

T = len(df)
H_tb = int(CFG["tb_horizon"])
H_fixed = int(CFG["fixed_horizon"])

model_lookback = int(CFG["model_lookback"])
frame_stride = int(CFG["frame_stride"])
L_eff = int(np.ceil(model_lookback / frame_stride))

t_min = model_lookback
t_max = T - max(H_tb, H_fixed) - 2
sample_t = np.arange(t_min, t_max + 1, dtype=np.int64)
n_samples = len(sample_t)

print("T:", T)
print("model_lookback:", model_lookback, "frame_stride:", frame_stride, "L_eff:", L_eff)
print("n_samples:", n_samples, "| t range:", int(sample_t[0]), "->", int(sample_t[-1]))

# %% ======================================================================
# Step 5: FAST edge features computed ONLY for sample_t and only e_last
# ======================================================================

def _fisher_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1.0 + x + eps) / (1.0 - x + eps))


def rolling_corr_all_t_from_cumsums(
    sx: np.ndarray, sy: np.ndarray,
    sxx: np.ndarray, syy: np.ndarray,
    sxy: np.ndarray,
    w: int,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Vectorized rolling corr for ALL t using cumulative sums.
    Uses variable window size m=min(w, t+1) like pandas min_periods=1,
    but we force corr=0 when m<2 (pandas would NaN -> later we used 0).
    """
    T_ = sx.shape[0]
    idx = np.arange(T_, dtype=np.int64)
    start = idx - (w - 1)
    start_clip = np.clip(start, 0, None)
    m = (idx - start_clip + 1).astype(np.float64)

    def win_sum(cs: np.ndarray) -> np.ndarray:
        prev = np.zeros_like(cs)
        mask = start_clip > 0
        prev[mask] = cs[start_clip[mask] - 1]
        return cs - prev

    sum_x = win_sum(sx)
    sum_y = win_sum(sy)
    sum_x2 = win_sum(sxx)
    sum_y2 = win_sum(syy)
    sum_xy = win_sum(sxy)

    num = m * sum_xy - sum_x * sum_y
    den_x = m * sum_x2 - sum_x * sum_x
    den_y = m * sum_y2 - sum_y * sum_y
    den = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0)) + eps

    corr = num / den
    corr[m < 2.0] = 0.0
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def build_edge_last_features_for_sample_t(
    lr_mat: np.ndarray,               # (T,N) float32/float64
    edges: List[Tuple[str, str]],
    corr_windows: List[int],
    corr_lags: List[int],
    sample_t_idx: np.ndarray,         # (n_samples,) time indices in [0..T-1]
    transform: str = "fisher",
    cache_path: Optional[Path] = None,
    out_dtype: str = "float16",
) -> np.ndarray:
    """
    Returns edge_last_sample: (n_samples, E, D) where D=len(corr_windows)*len(corr_lags)
    Only computed for sample_t positions (massive speed/memory win vs (T,E,D)).
    """
    nT, nN = lr_mat.shape
    E = len(edges)
    W = len(corr_windows)
    Lg = len(corr_lags)
    D = W * Lg

    out_dtype_np = np.float16 if out_dtype == "float16" else np.float32

    # Cache
    if cache_path is not None and cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        print("Loaded cached edge_last_sample:", arr.shape, arr.dtype)
        return arr

    # Create memmap .npy to avoid RAM peaks
    if cache_path is None:
        cache_path = CACHE_DIR / "edge_last_sample.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(str(cache_path), mode="w+", dtype=out_dtype_np, shape=(len(sample_t_idx), E, D))

    # Prepare y cumsums per dst node
    lr64 = np.asarray(lr_mat, dtype=np.float64)

    y_cs: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dst in range(nN):
        y = lr64[:, dst]
        sy = np.cumsum(y)
        syy = np.cumsum(y * y)
        y_cs[dst] = (y, sy, syy)

    # Prepare x-shifts and their cumsums per (src,lag)
    x_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for src in range(nN):
        x0 = lr64[:, src]
        for lag in corr_lags:
            lag = int(lag)
            x = np.zeros(nT, dtype=np.float64)
            if lag == 0:
                x[:] = x0
            else:
                x[lag:] = x0[: nT - lag]
            sx = np.cumsum(x)
            sxx = np.cumsum(x * x)
            x_cache[(src, lag)] = (x, sx, sxx)

    # Main loop (E * Lg): compute sxy once, then corr for each window
    t0 = time.time()
    for ei, (s_name, t_name) in enumerate(edges):
        s = ASSET2IDX[s_name]
        t = ASSET2IDX[t_name]

        if s == t:
            # self-loop: constant 1.0 for all D
            mm[:, ei, :] = out_dtype_np(1.0)
            continue

        y, sy, syy = y_cs[t]

        feat_idx = 0
        for lag in corr_lags:
            lag = int(lag)
            x, sx, sxx = x_cache[(s, lag)]

            # compute xy cumsum once per (src,lag,dst)
            xy = x * y
            sxy = np.cumsum(xy)

            for w in corr_windows:
                w = int(w)
                corr_all = rolling_corr_all_t_from_cumsums(sx, sy, sxx, syy, sxy, w=w)

                if transform == "fisher":
                    corr_all = _fisher_z(corr_all).astype(np.float32)

                # take only sample_t positions
                corr_s = corr_all[sample_t_idx]  # (n_samples,)
                mm[:, ei, feat_idx] = corr_s.astype(out_dtype_np)
                feat_idx += 1

    mm.flush()
    print(f"Built edge_last_sample -> {cache_path.name} shape={mm.shape} dtype={mm.dtype} in {time.time()-t0:.2f}s")

    # Load back as memmap read-only
    arr = np.load(cache_path, mmap_mode="r")
    return arr


# Build lr matrix (T,N)
lr_mat = np.stack([df[f"lr_{a}"].to_numpy(dtype=np.float32) for a in ASSETS], axis=1)  # (T,N)

edge_cache_name = (
    f"edge_last_sample_ns{n_samples}_W{'-'.join(map(str,CFG['corr_windows']))}"
    f"_L{'-'.join(map(str,CFG['corr_lags']))}_tr{CFG['edge_transform']}_dt{CFG['edge_dtype']}.npy"
)
edge_cache_path = CACHE_DIR / edge_cache_name

edge_last_sample = build_edge_last_features_for_sample_t(
    lr_mat=lr_mat,
    edges=EDGE_LIST,
    corr_windows=list(CFG["corr_windows"]),
    corr_lags=list(CFG["corr_lags"]),
    sample_t_idx=sample_t,
    transform=str(CFG.get("edge_transform", "fisher")),
    cache_path=edge_cache_path,
    out_dtype=str(CFG.get("edge_dtype", "float16")),
)

print("edge_last_sample:", edge_last_sample.shape, edge_last_sample.dtype, "(n_samples,E,D)")
print("edge_dim D =", edge_last_sample.shape[-1], "| E =", edge_last_sample.shape[1])

# %% ======================================================================
# Step 6: Splits (walk-forward)
# ======================================================================

def make_final_holdout_split(n_samples_: int, final_test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    n_final = max(1, int(round(final_test_frac * n_samples_)))
    n_cv = n_samples_ - n_final
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
        splits.append((
            np.arange(0, tr_end, dtype=np.int64),
            np.arange(tr_end, va_end, dtype=np.int64),
            np.arange(va_end, te_end, dtype=np.int64),
        ))
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

walk_splits = walk_splits[: int(CFG.get("max_folds", len(walk_splits)))]

print("Holdout split:")
print(f"  n_samples total: {n_samples}")
print(f"  n_samples CV   : {len(idx_cv_all)}")
print(f"  n_samples FINAL: {len(idx_final_test)}")
print("\nWalk-forward folds:", len(walk_splits))
for i, (a, b, c) in enumerate(walk_splits, 1):
    print(f"  fold {i}: train={len(a)} | val={len(b)} | test={len(c)}")

# %% ======================================================================
# Step 7: Dataset (uses edge_last_sample by sidx)  <<< MAIN CHANGE
# ======================================================================

class LobGraphDatasetTwoHeadFixedHUltraFast(Dataset):
    """
    Returns:
      x_seq:  (L_eff,N,F)   from X_node_raw
      e_last: (E,D)         from edge_last_sample[sidx]  (already aligned to sample_t)
      labels: by real time index t=sample_t[sidx]
    """
    def __init__(
        self,
        X_node: np.ndarray,             # (T,N,F) (memmap ok)
        edge_last_s: np.ndarray,        # (n_samples,E,D) (memmap ok)
        y_trade_arr: np.ndarray,
        y_dir_arr: np.ndarray,
        exit_ret_arr: np.ndarray,
        fixed_ret_arr: np.ndarray,
        sample_t_: np.ndarray,          # (n_samples,)
        indices: np.ndarray,            # indices in [0..n_samples-1]
        model_lookback: int,
        frame_stride: int,
    ):
        self.X_node = X_node
        self.edge_last_s = edge_last_s
        self.y_trade = y_trade_arr
        self.y_dir = y_dir_arr
        self.exit_ret = exit_ret_arr
        self.fixed_ret = fixed_ret_arr
        self.sample_t = sample_t_
        self.indices = indices.astype(np.int64)
        self.model_lookback = int(model_lookback)
        self.frame_stride = int(frame_stride)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        sidx = int(self.indices[i])
        t = int(self.sample_t[sidx])

        t0 = t - self.model_lookback
        x_seq = self.X_node[t0 : t + 1 : self.frame_stride]          # (L_eff,N,F)
        e_last = self.edge_last_s[sidx]                               # (E,D) aligned

        yt = int(self.y_trade[t])
        yd = int(self.y_dir[t])
        er_exit = float(self.exit_ret[t])
        er_fixed = float(self.fixed_ret[t])

        return (
            torch.from_numpy(np.asarray(x_seq, dtype=np.float32)),
            torch.from_numpy(np.asarray(e_last, dtype=np.float32)),   # float16->float32 ok
            torch.tensor(yt, dtype=torch.float32),
            torch.tensor(yd, dtype=torch.float32),
            torch.tensor(er_exit, dtype=torch.float32),
            torch.tensor(er_fixed, dtype=torch.float32),
            torch.tensor(sidx, dtype=torch.long),
        )


def collate_fn_ultrafast(batch):
    xs, e_last, ytr, ydir, er_exit, er_fixed, sidxs = zip(*batch)
    return (
        torch.stack(xs, 0),
        torch.stack(e_last, 0),
        torch.stack(ytr, 0),
        torch.stack(ydir, 0),
        torch.stack(er_exit, 0),
        torch.stack(er_fixed, 0),
        torch.stack(sidxs, 0),
    )


def split_trade_ratio(indices: np.ndarray, sample_t_: np.ndarray, y_trade_arr: np.ndarray) -> float:
    tt = sample_t_[indices]
    return float(y_trade_arr[tt].mean()) if len(tt) else float("nan")


def subsample_indices(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=int(max_n), replace=False)
    sel.sort()
    return sel.astype(np.int64)

# %% ======================================================================
# Step 8: Model (same as before, takes e_last)
# ======================================================================

def build_static_adjacency_from_edges(edge_index: torch.Tensor, n_nodes: int, eps: float = 1e-8) -> torch.Tensor:
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    src = edge_index[:, 0].long()
    dst = edge_index[:, 1].long()
    A[src, dst] = 1.0
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    return A


def build_adj_prior_from_edge_attr(
    edge_attr_last: torch.Tensor,  # (B,E,D)
    edge_index: torch.Tensor,
    n_nodes: int,
    use_abs: bool,
    diag_boost: float,
    row_normalize: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    edge_attr_last = torch.nan_to_num(edge_attr_last, nan=0.0, posinf=0.0, neginf=0.0)
    r = edge_attr_last.mean(dim=-1)
    if use_abs:
        r = r.abs()
    w = torch.sigmoid(r)

    B, E, _ = edge_attr_last.shape
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
        residual = x
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = self.dropout(f * g)

        skip = self.skip_conv(z)
        out = self.residual_conv(z)
        out = graph_message_passing(out, A)
        out = self.bn(out + residual)
        return out, skip


class GraphWaveNetTwoHeadFixedH(nn.Module):
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

    def _compute_supports(self, e_last: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, E, D = e_last.shape

        A_prior = build_adj_prior_from_edge_attr(
            edge_attr_last=e_last,
            edge_index=EDGE_INDEX.to(e_last.device),
            n_nodes=self.n_nodes,
            use_abs=bool(self.cfg.get("prior_use_abs", False)),
            diag_boost=float(self.cfg.get("prior_diag_boost", 1.0)),
            row_normalize=bool(self.cfg.get("prior_row_normalize", True)),
        )

        A_adapt_base, sparsity_proxy, _ = self.adapt()
        A_adapt = A_adapt_base.unsqueeze(0).expand(B, -1, -1)

        w = self.support_mix()
        A_static = self.A_static.to(e_last.device).to(e_last.dtype).unsqueeze(0).expand(B, -1, -1)

        A_mix = w[0] * A_static + w[1] * A_prior + w[2] * A_adapt
        A_mix = A_mix / (A_mix.sum(dim=-1, keepdim=True) + 1e-8)

        N = self.n_nodes
        offdiag = (1.0 - torch.eye(N, device=e_last.device, dtype=e_last.dtype))
        l1_off = (sparsity_proxy.to(e_last.dtype) * offdiag).abs().mean()
        mse_prior = ((A_adapt - A_prior) ** 2 * offdiag).mean()

        aux = {"support_w": w.detach().cpu().numpy().tolist(), "_l1_off_t": l1_off, "_mse_prior_t": mse_prior}
        return A_mix, aux

    def forward(self, x_seq: torch.Tensor, e_last: torch.Tensor, return_aux: bool = False):
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        e_last = torch.nan_to_num(e_last, nan=0.0, posinf=0.0, neginf=0.0)

        B, L, N, Fdim = x_seq.shape
        x = self.in_proj(x_seq)                 # (B,L,N,C)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,C,N,T)

        A_mix, aux = self._compute_supports(e_last)

        skip_sum = None
        for blk in self.blocks:
            x, skip = blk(x, A_mix)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        y = F.relu(skip_sum)
        y_end = F.relu(self.end1(y))
        feat = y_end[:, :, self.target_node, -1]

        trade_logit = self.trade_head(feat).squeeze(-1)
        dir_logit = self.dir_head(feat).squeeze(-1)
        fixed_hat = self.fixed_head(feat).squeeze(-1)

        if return_aux:
            return trade_logit, dir_logit, fixed_hat, aux
        return trade_logit, dir_logit, fixed_hat

# %% ======================================================================
# Step 9: Loss + metrics + threshold sweep (same logic)
# ======================================================================

def _safe_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def compute_trade_dir_auc_from_twohead(y_trade_true: np.ndarray, y_tb_true: np.ndarray, p_trade: np.ndarray, p_dir: np.ndarray) -> Tuple[float, float]:
    trade_auc = _safe_auc_binary(y_trade_true, p_trade)
    mask_trade = (y_tb_true != 1)
    y_dir_bin = (y_tb_true[mask_trade] == 2).astype(np.int64)
    dir_auc = _safe_auc_binary(y_dir_bin, p_dir[mask_trade])
    return trade_auc, dir_auc


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

    def _cfg_float(key: str, default: float) -> float:
        v = cfg.get(key, default)
        return float(v) if v is not None else float(default)

    wT = _cfg_float("loss_w_trade", 1.0)
    wD = _cfg_float("loss_w_dir", 1.0)
    wR = _cfg_float("loss_w_ret", 1.0)
    wU = _cfg_float("loss_w_utility", 1.0)

    bceT = bce_trade_fn(trade_logit, y_trade_t)

    maskT = (y_trade_t == 1)
    if maskT.any():
        bceD = bce_dir_fn(dir_logit[maskT], y_dir_t[maskT])
    else:
        bceD = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)

    clip_val = _cfg_float("fixed_ret_clip", 0.0)
    fr = fixed_ret_t
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
        util = utility_vec[maskT].mean() if maskT.any() else torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
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
        "util_scaled": utilS.detach(),
        "pos_abs": pos.abs().mean().detach(),
        "p_trade_mean": p_trade.mean().detach(),
    }
    return total, parts

# %% ======================================================================
# Step 10: Fast eval (subsampled)
# ======================================================================

@torch.no_grad()
def eval_twohead_on_indices_ultrafast(
    model: nn.Module,
    indices: np.ndarray,
    bce_trade: nn.Module,
    bce_dir: nn.Module,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ds = LobGraphDatasetTwoHeadFixedHUltraFast(
        X_node=X_node_raw,
        edge_last_s=edge_last_sample,
        y_trade_arr=y_trade,
        y_dir_arr=y_dir,
        exit_ret_arr=exit_ret,
        fixed_ret_arr=fixed_ret,
        sample_t_=sample_t,
        indices=indices.astype(np.int64),
        model_lookback=int(cfg["model_lookback"]),
        frame_stride=int(cfg["frame_stride"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn_ultrafast, num_workers=0)

    model.eval()

    tot_loss = 0.0
    tot_us = 0.0
    tot_ptr = 0.0
    tot_pos_abs = 0.0
    n = 0

    p_trade_all, p_dir_all, y_trade_all, exit_all = [], [], [], []

    for x, e_last, yt, yd, er_exit, er_fixed, _sidx in loader:
        x = x.to(DEVICE).float()
        e_last = e_last.to(DEVICE).float()
        yt = yt.to(DEVICE).float()
        yd = yd.to(DEVICE).float()
        er_fixed = er_fixed.to(DEVICE).float()

        trade_logit, dir_logit, fixed_hat, aux = model(x, e_last, return_aux=True)
        loss, parts = multitask_loss_twohead_fixedH(trade_logit, dir_logit, fixed_hat, yt, yd, er_fixed, bce_trade, bce_dir, cfg)
        loss = total_loss_with_adj_reg(loss, aux, cfg)

        B = int(yt.size(0))
        tot_loss += float(loss.item()) * B
        tot_us += float(parts["util_scaled"].item()) * B
        tot_ptr += float(parts["p_trade_mean"].item()) * B
        tot_pos_abs += float(parts["pos_abs"].item()) * B
        n += B

        p_trade_all.append(torch.sigmoid(trade_logit).detach().cpu().numpy())
        p_dir_all.append(torch.sigmoid(dir_logit).detach().cpu().numpy())
        y_trade_all.append((yt.detach().cpu().numpy() > 0.5).astype(np.int64))
        exit_all.append(er_exit.detach().cpu().numpy())

    p_trade_np = np.concatenate(p_trade_all, axis=0) if p_trade_all else np.zeros((0,), dtype=np.float64)
    p_dir_np = np.concatenate(p_dir_all, axis=0) if p_dir_all else np.zeros((0,), dtype=np.float64)
    y_trade_np = np.concatenate(y_trade_all, axis=0) if y_trade_all else np.zeros((0,), dtype=np.int64)
    er_exit_np = np.concatenate(exit_all, axis=0) if exit_all else np.zeros((0,), dtype=np.float64)

    t_idx = sample_t[indices.astype(np.int64)]
    y_tb_np = y_tb[t_idx].astype(np.int64)

    trade_auc, dir_auc = compute_trade_dir_auc_from_twohead(y_trade_np, y_tb_np, p_trade_np, p_dir_np)
    prob3 = probs3_from_twohead(p_trade_np, p_dir_np)

    return {
        "loss": float(tot_loss / max(1, n)),
        "soft_util_scaled_mean": float(tot_us / max(1, n)),
        "pos_abs_mean": float(tot_pos_abs / max(1, n)),
        "p_trade_mean": float(tot_ptr / max(1, n)),
        "trade_auc": float(trade_auc) if np.isfinite(trade_auc) else float("nan"),
        "dir_auc": float(dir_auc) if np.isfinite(dir_auc) else float("nan"),
        "prob3": prob3,
        "exit_ret": er_exit_np,
    }

# %% ======================================================================
# Step 11: Train one fold (ultrafast)
# ======================================================================

def train_one_fold_ultrafast(
    fold_id: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    t_train = sample_t[idx_train]
    ytr_train = y_trade[t_train].astype(np.int64)
    ytb_train = y_tb[t_train].astype(np.int64)

    pos_w_trade = compute_pos_weights_binary(ytr_train)
    bce_trade = nn.BCEWithLogitsLoss(pos_weight=pos_w_trade)

    mask_tr = (ytb_train != 1)
    ydir_train = (ytb_train[mask_tr] == 2).astype(np.int64)
    pos_w_dir = compute_pos_weights_binary(ydir_train) if ydir_train.size else torch.tensor([1.0], device=DEVICE)
    bce_dir = nn.BCEWithLogitsLoss(pos_weight=pos_w_dir)

    model = GraphWaveNetTwoHeadFixedH(
        node_in=int(X_node_raw.shape[-1]),
        edge_dim=int(edge_last_sample.shape[-1]),
        cfg=cfg,
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    steps_per_epoch = max(1, int(np.ceil(cfg["train_samples_per_epoch"] / cfg["batch_size"])))
    use_onecycle = bool(cfg.get("use_onecycle", True))
    if use_onecycle:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            steps_per_epoch=steps_per_epoch,
            pct_start=float(cfg.get("onecycle_pct_start", 0.20)),
            div_factor=float(cfg.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(cfg.get("onecycle_final_div", 200.0)),
        )
    else:
        sch = None

    b_dir = float(cfg.get("sel_b_dir_auc", 0.10))
    trade_pen = float(cfg.get("trade_prob_penalty", 0.0))

    best_sel = -1e18
    best_state = None
    best_epoch = -1
    bad = 0
    patience = int(cfg.get("patience", 7))

    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()

        idx_train_ep = subsample_indices(idx_train, int(cfg["train_samples_per_epoch"]), seed=1000 * fold_id + ep)

        tr_ds = LobGraphDatasetTwoHeadFixedHUltraFast(
            X_node=X_node_raw,
            edge_last_s=edge_last_sample,
            y_trade_arr=y_trade,
            y_dir_arr=y_dir,
            exit_ret_arr=exit_ret,
            fixed_ret_arr=fixed_ret,
            sample_t_=sample_t,
            indices=idx_train_ep,
            model_lookback=int(cfg["model_lookback"]),
            frame_stride=int(cfg["frame_stride"]),
        )
        tr_loader = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate_fn_ultrafast, num_workers=0)

        tot = 0.0
        tot_us = 0.0
        tot_ptr = 0.0
        n_ = 0

        autocast_ctx = torch.autocast(device_type=DEVICE.type, dtype=torch.float16) if USE_AMP else torch.autocast(device_type="cpu", enabled=False)

        for x, e_last, yt, yd, _er_exit, er_fixed, _sidx in tr_loader:
            x = x.to(DEVICE).float()
            e_last = e_last.to(DEVICE).float()
            yt = yt.to(DEVICE).float()
            yd = yd.to(DEVICE).float()
            er_fixed = er_fixed.to(DEVICE).float()

            opt.zero_grad(set_to_none=True)

            with autocast_ctx:
                trade_logit, dir_logit, fixed_hat, aux = model(x, e_last, return_aux=True)
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
            tot_us += float(parts["util_scaled"].item()) * B
            tot_ptr += float(parts["p_trade_mean"].item()) * B
            n_ += B

        tr_loss = tot / max(1, n_)
        tr_us = tot_us / max(1, n_)
        tr_ptr = tot_ptr / max(1, n_)

        do_eval = (ep == 1) or (ep % int(cfg.get("eval_every", 1)) == 0) or (ep == int(cfg["epochs"]))
        sel = -1e18
        val_loss = float("nan")
        val_us = float("nan")
        trade_auc = float("nan")
        dir_auc = float("nan")

        if do_eval:
            idx_val_ep = subsample_indices(idx_val, int(cfg["val_samples_per_eval"]), seed=2000 * fold_id + ep)
            ev = eval_twohead_on_indices_ultrafast(model, idx_val_ep, bce_trade, bce_dir, cfg)
            trade_auc = ev["trade_auc"]
            dir_auc = ev["dir_auc"]
            val_loss = ev["loss"]
            val_us = ev["soft_util_scaled_mean"]

            sel = float(val_us) + b_dir * (float(dir_auc) if np.isfinite(dir_auc) else 0.0)

            if sel > best_sel:
                best_sel = sel
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

        lr_now = opt.param_groups[0]["lr"]
        w_support = model.support_mix().detach().cpu().numpy().tolist()

        print(
            f"[fold {fold_id:02d}] ep {ep:02d} lr={lr_now:.2e} "
            f"tr_loss={tr_loss:.4f} tr_utilS={tr_us:.5f} tr_pT={tr_ptr:.3f} "
            + (f"| val_loss={val_loss:.4f} val_utilS={val_us:.5f} val_trade_auc={trade_auc:.3f} val_dir_auc={dir_auc:.3f} sel={sel:.5f} "
               f"best={best_sel:.5f}@ep{best_epoch:02d} supports={np.round(w_support, 3).tolist()}"
               if do_eval else
               f"| (skip val; eval_every={cfg.get('eval_every',1)}) supports={np.round(w_support, 3).tolist()}")
        )

        if do_eval and bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold selection on bigger subsample once
    idx_val_eval = subsample_indices(idx_val, 30_000, seed=777 + fold_id)
    idx_test_eval = subsample_indices(idx_test, 30_000, seed=888 + fold_id)

    val_eval = eval_twohead_on_indices_ultrafast(model, idx_val_eval, bce_trade, bce_dir, cfg)
    test_eval = eval_twohead_on_indices_ultrafast(model, idx_test_eval, bce_trade, bce_dir, cfg)

    true_val_trade_rate = split_trade_ratio(idx_val_eval, sample_t, y_trade)

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
        f"[fold {fold_id:02d}] chosen thresholds on VAL(subsample): thr_trade={thr_trade:.3f} thr_dir={thr_dir:.3f} "
        f"| val pnl_sum={pnl_val['pnl_sum']:.4f} val trade_rate={pnl_val['trade_rate']:.3f}"
    )
    print(
        f"[fold {fold_id:02d}] TEST(subsample): trade_auc={test_eval['trade_auc']:.3f} dir_auc={test_eval['dir_auc']:.3f} "
        f"soft_utilS={test_eval['soft_util_scaled_mean']:.5f} pnl_sum={pnl_test['pnl_sum']:.4f} "
        f"trade_rate={pnl_test['trade_rate']:.3f} trades={pnl_test['n_trades']}"
    )

    return {
        "fold": int(fold_id),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "best_epoch": int(best_epoch),
        "best_sel": float(best_sel),
        "val_eval": val_eval,
        "test_eval": test_eval,
        "thr_trade": thr_trade,
        "thr_dir": thr_dir,
        "pnl_val": pnl_val,
        "pnl_test": pnl_test,
    }

# %% ======================================================================
# Step 12: Run CV (ultrafast)
# ======================================================================

def run_walk_forward_cv_ultrafast() -> pd.DataFrame:
    rows = []
    for fi, (idx_tr, idx_va, idx_te) in enumerate(walk_splits, 1):
        print("\n" + "=" * 90)
        print(f"FOLD {fi}/{len(walk_splits)} sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
        print(f"True trade ratio (val):  {split_trade_ratio(idx_va, sample_t, y_trade):.3f}")
        print(f"True trade ratio (test): {split_trade_ratio(idx_te, sample_t, y_trade):.3f}")

        art = train_one_fold_ultrafast(fi, idx_tr, idx_va, idx_te, CFG)

        rows.append({
            "fold": fi,
            "val_trade_auc": art["val_eval"]["trade_auc"],
            "val_dir_auc": art["val_eval"]["dir_auc"],
            "val_soft_utilS": art["val_eval"]["soft_util_scaled_mean"],
            "val_loss": art["val_eval"]["loss"],
            "test_trade_auc": art["test_eval"]["trade_auc"],
            "test_dir_auc": art["test_eval"]["dir_auc"],
            "test_soft_utilS": art["test_eval"]["soft_util_scaled_mean"],
            "test_loss": art["test_eval"]["loss"],
            "thr_trade": art["thr_trade"],
            "thr_dir": art["thr_dir"],
            "test_trade_rate_pred": art["pnl_test"]["trade_rate"],
            "test_pnl_sum": art["pnl_test"]["pnl_sum"],
            "test_n_trades": art["pnl_test"]["n_trades"],
            "best_sel": art["best_sel"],
        })

    return pd.DataFrame(rows)


cv_summary = run_walk_forward_cv_ultrafast()

print("\n" + "=" * 90)
print("CV summary (ULTRAFAST):")
print(cv_summary)
print("\nMeans:")
print(cv_summary.mean(numeric_only=True))