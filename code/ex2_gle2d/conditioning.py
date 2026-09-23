"""ex2_gle2d conditioning: tuples, the OLS compression map, variants.

The deployed conditioning coordinate for the 'mem' variant is the 2D
DATA-ESTIMATED increment prediction q_hat = [1, v, m] @ coef, fitted
by OLS on the training tuples (Stage-B evidence: q_hat matches the
oracle to 0.84% of the innovation std held out). The full memory bank
m is still updated at every step of any rollout; compression happens
at query time only.
"""

import os

import numpy as np

import config as cfg
from memdiff.features import ema_features


def bank_features(X, rates):
    """EMA bank on the observed velocities (rate-major layout)."""
    return ema_features(X, np.asarray(rates), cfg.DT, cfg.BURN_TOL)


def raw_tuples(X, rates):
    """Full tuples zeta = (v, m) and raw increments, burn-in removed.

    Returns Z (N, 2 + 2k), D (N, 2), plus per-row (traj, time) so
    rollout warm-ups can find prehistories.
    """
    m, nb = bank_features(X, rates)
    nb = max(nb, 1)
    Z = np.concatenate([X[:, nb:-1], m[:, nb:-1]], axis=2)
    D = X[:, nb + 1:] - X[:, nb:-1]
    return (Z.reshape(-1, Z.shape[-1]), D.reshape(-1, 2), nb)


def fit_compression(X_train, rates):
    """Pool-fitted OLS map: increments on tuples (intercept included).

    Deterministic given the data; saved to out/compression.npz so the
    numpy stage-0 and the torch stages use identical coefficients.
    """
    Z, D, nb = raw_tuples(X_train, rates)
    stride = max(1, int(np.ceil(len(Z) / cfg.N_REF)))
    Zs, Ds = Z[::stride][:cfg.N_REF], D[::stride][:cfg.N_REF]
    Xd = np.hstack([np.ones((len(Zs), 1)), Zs])
    coef, *_ = np.linalg.lstsq(Xd, Ds, rcond=None)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    np.savez(os.path.join(cfg.OUT_DIR, "compression.npz"),
             coef=coef, rates=np.asarray(rates), n_fit=len(Zs),
             stride=stride, n_burn=nb)
    return coef, nb, stride


def qhat(Z, coef):
    """Compressed coordinate for tuple rows Z (N, 2+2k) -> (N, 2)."""
    Z = np.atleast_2d(Z)
    return np.hstack([np.ones((len(Z), 1)), Z]) @ coef


def build_tuples(X, rates, variant, coef=None, kappa_s=None):
    """Conditioning tuples for one variant.

    'mem'    C = q_hat (2D compressed predictive coordinate)
    'markov' C = v (memoryless baseline)
    Labels S = increments * kappa_s (pipeline convention); the
    reference pool is strided to cfg.N_REF rows.
    """
    Z, D, nb = raw_tuples(X, rates if variant == "mem"
                          else np.array([]))
    if kappa_s is None:
        kappa_s = float(1.0 / D.std())
    stride = max(1, int(np.ceil(len(Z) / cfg.N_REF)))
    Z, D = Z[::stride][:cfg.N_REF], D[::stride][:cfg.N_REF]
    if variant == "mem":
        C = qhat(Z, coef)
    elif variant == "markov":
        C = Z[:, :2].copy()
    else:
        raise ValueError(variant)
    return {"C": C, "S": D * kappa_s, "kappa_s": kappa_s,
            "n_burn": nb, "stride": stride, "rates": np.asarray(rates)}
