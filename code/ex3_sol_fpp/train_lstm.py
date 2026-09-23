"""Train the Gaussian-head LSTM baseline for Example 3 (torch; workstation).

memdiff.lstm.train_lstm on the SAME training trajectories as the
diffusion pipeline, inputs (y_n, dy_n) (strictly more than the bank
conditioning), maximum likelihood, single Gaussian head, one model per
training seed in LSTM_SEEDS. Wall time and parameter count are recorded
for the cost comparison.

    python train_lstm.py [--smoke]     -> out/lstm_s<seed>/model.pt
                                          (smoke: out/smoke/lstm_s<seed>/)
"""

import os
import sys
import time

import numpy as np
import torch

import config as cfg
from memdiff.lstm import train_lstm

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        cfg.apply_smoke()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_train"]
    for seed in cfg.LSTM_SEEDS:
        cfg.SEED_LSTM = int(seed)
        print(f"== LSTM + Gaussian head, seed {seed} (device {device}) ==")
        print(f"  hidden {cfg.LSTM_HIDDEN}, layers {cfg.LSTM_LAYERS}, mixture "
              f"{cfg.LSTM_MIX}, sequence {cfg.LSTM_SEQ}, burn {cfg.LSTM_BURN}")
        t0 = time.perf_counter()
        net, meta = train_lstm(X_train, cfg, device)
        wall = time.perf_counter() - t0
        n_par = sum(p.numel() for p in net.parameters())
        meta["wall_train_s"] = wall
        meta["tokens"] = meta["epochs_run"] * meta["batch"] * meta["seq"]
        print(f"  parameters {n_par}, wall {wall:.0f} s, epochs "
              f"{meta['epochs_run']}, val NLL {meta['val_nll']:.4f}")
        vdir = os.path.join(cfg.OUT_DIR, f"lstm_s{seed}")
        os.makedirs(vdir, exist_ok=True)
        torch.save({"state": net.state_dict(), "meta": meta,
                    "n_parameters": n_par}, os.path.join(vdir, "model.pt"))
        print(f"  saved -> {vdir}/model.pt")
