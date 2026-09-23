"""Train the recurrent baseline for Example 4 (torch; workstation).

An LSTM with a Gaussian head, fitted by maximum likelihood on the SAME
training trajectories the diffusion pipeline uses. It is the standard
machine-learning alternative for a non-Markovian series and is trained
independently of the training-free sampler.

    python train_lstm.py

Writes out/lstm/model.pt (state dict + scaling meta).
"""

import os

import numpy as np
import torch

import config as cfg
from memdiff.lstm import train_lstm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_train"]
    print(f"== LSTM baseline (device {device}) ==")
    print(f"  hidden {cfg.LSTM_HIDDEN}, layers {cfg.LSTM_LAYERS}, "
          f"mixture components {cfg.LSTM_MIX}, sequence {cfg.LSTM_SEQ}")
    net, meta = train_lstm(X_train, cfg, device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  parameters {n_par}  (flow map: "
          f"~{cfg.N_LAYERS * cfg.HIDDEN ** 2})")

    vdir = os.path.join(cfg.OUT_DIR, "lstm")
    os.makedirs(vdir, exist_ok=True)
    torch.save({"state": net.state_dict(), "meta": meta,
                "n_parameters": n_par}, os.path.join(vdir, "model.pt"))
    print(f"  saved -> {vdir}/model.pt")
