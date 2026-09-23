"""Exact stationary trajectories for ex2_gle2d (numpy only).

Simulates the exact discrete transition (A_d, Q_d) of the frozen
regime; writes out/data.npz with X_train (N_TRAJ, L+1, 2) and X_test
(N_TEST, L+1, 2) — the observed velocity component only.
"""

import os

import numpy as np

import config as cfg
from exact_refs import references


def simulate(n_traj, L, seed, refs):
    rng = np.random.default_rng(seed)
    Ad, cQ, sqSig = refs["Ad"], refs["cQ"], refs["sqSig"]
    Y = np.empty((n_traj, L + 1, 4))
    Y[:, 0] = rng.standard_normal((n_traj, 4)) * sqSig
    for t in range(L):
        Y[:, t + 1] = Y[:, t] @ Ad.T + rng.standard_normal(
            (n_traj, 4)) @ cQ.T
    return Y[:, :, :2]


def main():
    refs = references()
    print(f"simulating {cfg.N_TRAJ} train + {cfg.N_TEST} test "
          f"trajectories of {cfg.L_TRAJ} steps ...")
    X_train = simulate(cfg.N_TRAJ, cfg.L_TRAJ, cfg.SEED_DATA, refs)
    X_test = simulate(cfg.N_TEST, cfg.L_TRAJ, cfg.SEED_DATA + 1, refs)
    gibbs = float(np.max(np.abs(
        np.cov(X_train.reshape(-1, 2).T) - cfg.KBT * np.eye(2))))
    print(f"Gibbs check |Cov(v) - I|_max = {gibbs:.4f}")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.OUT_DIR, "data.npz"),
                        X_train=X_train, X_test=X_test,
                        gibbs_dev=gibbs)
    print(f"saved -> {cfg.OUT_DIR}/data.npz")


if __name__ == "__main__":
    main()
