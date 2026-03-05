# 1-second LOB config for the same GraphWaveNet/MTGNN-style model
# Key idea: keep *time horizons* reasonable for 1s data and make the pipeline robust to
# lots of zeros / repeated values (flat prints, unchanged book, etc.).

CFG = {
    # ----------------------
    # data
    # ----------------------
    "freq": "1s",                       # 1-second bars (IMPORTANT: in loader you must round to "S" not "min")
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
    # ----------------------
    "train_min_frac": 0.55,
    "val_window_frac": 0.10,
    "test_window_frac": 0.10,
    "step_window_frac": 0.10,

    # ----------------------
    # scaling
    # ----------------------
    "max_abs_feat": 6.0,                # 1s data is noisier; tighter clipping reduces rare spikes / bad scaler tails
    "max_abs_edge": 3.5,                # Fisher(z) on short-window corr can explode near ±1 -> slightly tighter clip

    # ----------------------
    # correlations / graph
    # ----------------------
    # Old (1min): windows [10,30,60,120] minutes and lags 0..10 minutes.
    # New (1s): use seconds; keep a mix of short micro lags + the requested 1–10 minute lags.
    "corr_windows": [60, 180, 600, 1800],      # 1m/3m/10m/30m windows (seconds). Shorter windows help microstructure, longer stabilize zeros.
    "corr_lags": [0, 1, 2, 5, 10, 30, 60, 120, 300, 600],  # includes 1–10 minutes (60..600s) + small lags for HFT lead/lag
    "edges_mode": "all_pairs",
    "add_self_loops": True,
    "edge_transform": "fisher",
    "edge_scale": True,

    # ----------------------
    # triple-barrier labels
    # ----------------------
    # Old: horizon=30 steps @ 1min => 30 minutes.
    # For 1s data, keeping 30 minutes (1800) makes sequences huge and training heavy.
    # A practical starting point is 5 minutes (300s) for 1s microstructure.
    "tb_horizon": 300,                  # 5 minutes ahead in 1s steps (balances microstructure + tractable training)
    "lookback": 900,                    # 15 minutes history for BOTH: model input length and TB vol_window (enough context + stable vol with many zeros)
    "tb_pt_mult": 1.70,
    "tb_sl_mult": 1.70,
    # Barriers scaled down roughly by sqrt(5min/30min) ~ 0.41 vs the old 1min config.
    "tb_min_barrier": 0.0014,           # smaller min barrier because horizon is shorter; avoids "always flat" labels at 1s
    "tb_max_barrier": 0.0060,           # cap also scaled down; prevents extreme barriers during short-lived volatility bursts

    # ----------------------
    # fixed-horizon return head
    # ----------------------
    "fixed_horizon": 300,               # match TB horizon (5 minutes); keeps utility/return target consistent
    "fixed_ret_clip": 0.0040,           # clip reduced vs 30min because typical 5min returns are smaller; stabilizes Huber + utility

    # ----------------------
    # training
    # ----------------------
    # 1s dataset has far more samples; fewer epochs are usually enough.
    # Sequence length is bigger (L=900), so keep batch size conservative on CPU.
    "batch_size": 32,                   # 1s sequences are longer -> memory heavier; raise to 64/128 if you train on GPU
    "epochs": 40,                       # more samples per epoch -> fewer epochs needed vs 1min
    "lr": 3.0e-4,                       # slightly higher max_lr often works with OneCycle on larger datasets
    "weight_decay": 5.0e-3,             # stronger regularization: 1s data is noisier + more repeats
    "grad_clip": 1.0,
    "dropout": 0.35,                    # bump dropout for noisy 1s microstructure

    "use_weighted_sampler": False,      # keep off; with huge 1s sample size, weighting can overtrade / miscalibrate

    # utility
    "utility_mask_true_trades": False,  # keep False: penalize bad "would-trade" decisions even if TB label is flat

    # scheduler
    "use_onecycle": True,
    "onecycle_pct_start": 0.20,         # slightly faster warmup; lots of steps per epoch anyway
    "onecycle_div_factor": 40.0,
    "onecycle_final_div": 800.0,        # slower tail helps stabilize at the end on noisy data

    # ----------------------
    # model
    # ----------------------
    # IMPORTANT: receptive field must cover meaningful part of the horizon in *steps*.
    # Old (k=2, blocks=2, layers=4) gave RF ~ 31 steps (~31 minutes at 1min).
    # New target: RF ~ 512 seconds (~8.5 min) to be in the same ballpark as 5-min horizon.
    "gwn_residual_channels": 48,        # slightly smaller channels to offset deeper dilation stack
    "gwn_dilation_channels": 48,
    "gwn_skip_channels": 192,
    "gwn_end_channels": 192,
    "gwn_blocks": 1,                    # fewer blocks; we increase layers_per_block for RF instead
    "gwn_layers_per_block": 9,          # dilations 1..256 => RF ~ 512 steps with k=2 (good for 1s / 5min horizon)
    "gwn_kernel_size": 2,

    # ----------------------
    # adaptive adjacency
    # ----------------------
    "adj_emb_dim": 16,
    "adj_temperature": 1.25,            # soften adjacency logits; 1s corr priors are noisier -> avoid overly peaky A_adapt
    "adaptive_topk": 3,                 # keep full connectivity for 3 nodes (no sparsity mask)

    "adj_l1_lambda": 3e-3,              # modest sparsity pressure (off-diagonal) to reduce unstable rapid flipping
    "adj_prior_lambda": 2e-2,           # stronger prior anchoring: 1s corr features are noisier / more often 0 due to flat lr

    # prior adjacency
    "prior_use_abs": True,
    "prior_diag_boost": 1.0,
    "prior_row_normalize": True,

    # ----------------------
    # trading eval / thresholds
    # ----------------------
    "cost_bps": 1.0,
    # 1s predictions are much noisier -> prefer higher trade confidence thresholds by default.
    "thr_trade_grid": [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
    "thr_dir_grid":   [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "eval_min_trades": 500,             # more samples at 1s -> require more trades to make PnL estimates less noisy
    "max_trade_rate_val": 0.02,         # 2% of seconds traded is already very high frequency; prevents degenerate always-trade models
    "trade_rate_penalty": 8.0,          # stronger penalty: discourages overtrading when many repeated values inflate false signals
    "thr_objective": "pnl_sum",
    "proxy_target_trades": [200, 500, 1000, 2000],  # scale up vs 1min; val windows contain many more timestamps
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

    # regression / utility stability
    "ret_huber_delta": 0.0015,          # smaller delta because fixed_ret magnitude is smaller at 5-min horizon on 1s data

    # utility
    "utility_k": 3.0,
    # Returns are smaller at shorter horizon -> scale utility up to keep gradient signal comparable.
    "utility_scale": 550.0,             # ~2.5x of 220 (roughly inverse of sqrt(5/30)) to compensate smaller 5-min returns
    "utility_return_source": "fixed_ret",

    # discourage too frequent trading directly through logits
    "trade_prob_penalty": 2.0,          # stronger than 1min: with 1s noise it's easy to learn "always trade"

    # keep (unused by current code path, retained for compatibility)
    "trade_rate_target": 0.02,
    "trade_rate_lambda": 0.0,

    # artifacts
    "artifact_dir": "./artifacts_1s",
}