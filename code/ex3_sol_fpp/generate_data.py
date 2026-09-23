"""Exact observation data for Example 3 (numpy only).

Trajectories of the filtered-Poisson model are drawn by the EXACT
simulator (model.simulate: exact decay, exact compound-Poisson arrivals
inside each step, white observation noise), after a stationary burn-in.
Only the observed scalar y is recorded for training and testing: the
hidden pulse states never reach any model or the evaluation (the
observable-history reference is a particle filter on y alone).

    python generate_data.py [--smoke]    ->  out/data.npz  (smoke: out/smoke/)
"""

import os
import sys

import numpy as np

import config as cfg
import model as md

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        cfg.apply_smoke()
    P = md.params()
    Y_tr, _, _ = md.simulate(cfg.N_TRAJ, cfg.L_TRAJ, cfg.SEED_DATA, P=P)
    Y_te, _, _ = md.simulate(cfg.N_TEST, cfg.L_TRAJ, cfg.SEED_DATA + 1, P=P)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUT_DIR, "data.npz")
    np.savez_compressed(path, X_train=Y_tr, X_test=Y_te, dt=cfg.DT,
                        regime=np.array([cfg.TAU_D[0], cfg.TAU_D[1],
                                         cfg.GAMMA[0], cfg.GAMMA[1], cfg.TAU_R,
                                         cfg.EPS_NOISE, cfg.DT]),
                        seed=cfg.SEED_DATA)
    print(f"data: train {Y_tr.shape}, test {Y_te.shape} -> {path}")
    print(f"  closed form: mean {P['mean']:.3f} var {P['var']:.3f}; sample "
          f"mean {Y_tr.mean():.3f} var {Y_tr.var():.3f}")
