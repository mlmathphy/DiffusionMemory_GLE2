"""Exact observation data for Example 4, multiscale GLE (numpy only).

Trajectories are drawn from the EXACT discrete-time linear-Gaussian
transition (A_d, Q_d) of the Markovian embedding, initialized at
stationarity, and only the X component is recorded. There is therefore
no time-integration error in the data: every discrepancy measured later
belongs to the learning pipeline, not to the data generator.

    python generate_data.py        ->  out/data.npz
"""

import os

import numpy as np

import config as cfg
from multiscale_exact import discrete_system


def simulate(n_traj, n_steps, rng):
    """(n_traj, n_steps+1) observed X, exact stationary sampling."""
    A_d, Q_d, Sigma = discrete_system()
    s = A_d.shape[0]
    Ls = np.linalg.cholesky(Sigma + 1e-14 * np.eye(s))
    Lq = np.linalg.cholesky(Q_d + 1e-14 * np.eye(s))
    Y = rng.standard_normal((n_traj, s)) @ Ls.T
    X = np.empty((n_traj, n_steps + 1))
    X[:, 0] = Y[:, 0]
    for n in range(1, n_steps + 1):
        Y = Y @ A_d.T + rng.standard_normal((n_traj, s)) @ Lq.T
        X[:, n] = Y[:, 0]
    return X


def main():
    rng = np.random.default_rng(cfg.SEED_DATA)
    X_train = simulate(cfg.N_TRAJ, cfg.L_TRAJ, rng)
    X_test = simulate(cfg.N_TEST, cfg.L_TRAJ, rng)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUT_DIR, "data.npz")
    np.savez_compressed(path, X_train=X_train, X_test=X_test, dt=cfg.DT)
    print(f"data: train {X_train.shape}, test {X_test.shape} -> {path}")


if __name__ == "__main__":
    main()
