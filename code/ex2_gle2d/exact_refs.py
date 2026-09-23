"""Closed-form references for the frozen ex2_gle2d regime (numpy only).

Thin layer over the scout machinery (scout.py) so the pipeline and the
scout share one implementation of the model, the discretization, the
bank conditionals, and the Kalman full-history reference. Run as a
script for a pre-vet printout of the frozen regime's reference
numbers.
"""

import numpy as np

import config as cfg
from scout import (build_ct, discretize, sv_series, bank_analysis,
                   kalman_full_reference, psd_sqrt)
from memdiff.features import rates_from_band


def model():
    """A, Qc, Sigma, params for the frozen regime."""
    return build_ct(cfg.F_MEM, cfg.TAU2, cfg.THETA)


def discrete():
    A, Qc, Sigma, par = model()
    Ad, Qd = discretize(A, Sigma, cfg.DT)
    return Ad, Qd, Sigma, par


def band_rates():
    """The scout's declared band rule (exact state ACF) + frozen k."""
    Ad, Qd, Sigma, _ = discrete()
    Sv = sv_series(Ad, Sigma, cfg.BAND_MAX_LAG + 2)
    r = np.trace(Sv, axis1=1, axis2=2) / np.trace(Sv[0])
    sig = np.nonzero(np.abs(r[1:cfg.BAND_MAX_LAG + 1]) > cfg.ACF_SIG)[0]
    l_max = int(sig[-1]) + 1 if len(sig) else 1
    band = (1.0 / (l_max * cfg.DT), 1.0 / cfg.DT)
    return band, rates_from_band(band, cfg.K_BANK)


def references(n_sv=None):
    """Everything the pipeline compares against, in one dict."""
    Ad, Qd, Sigma, par = discrete()
    m_acf = int(round(3 * cfg.TAU2 / cfg.DT))
    n_sv = n_sv or max(cfg.EVAL_MAX_LAG + 2, m_acf + 2,
                       cfg.BAND_MAX_LAG + 2)
    Sv = sv_series(Ad, Sigma, n_sv)
    band, rates = band_rates()
    bk = bank_analysis(Ad, Qd, Sigma, Sv, rates, [1], m_acf)
    V_kal, kal_iters, kal_delta = kalman_full_reference(
        Ad, Qd, Sigma, [1])
    Sv0 = Sv[0]
    V_state = Sv0 - Sv[1] @ np.linalg.solve(Sv0, Sv[1].T)
    # sampled-displacement diffusion tensor (SCOUT.md convention)
    S1e = (Ad @ np.linalg.solve(np.eye(4) - Ad, Sigma))[:2, :2]
    D_ex = 0.5 * cfg.DT * (Sv0 + S1e + S1e.T)
    return dict(Ad=Ad, Qd=Qd, Sigma=Sigma, params=par, Sv=Sv,
                band=band, rates=rates, Gamma=bk["Gamma"],
                Vb1=bk["Vb1"], chain=bk["chain"], V_state=V_state,
                V_kal1=V_kal[1], kal_iters=kal_iters,
                kal_delta=kal_delta, D_exact=D_ex, m_acf=m_acf,
                sqSig=np.sqrt(np.diag(Sigma)), cQ=psd_sqrt(Qd))


def markov_coeffs(Sv):
    """Exact memoryless one-step law: E[v'|v] = Phi v, cov V_mk."""
    Phi = Sv[1] @ np.linalg.inv(Sv[0])
    V_mk = Sv[0] - Sv[1] @ np.linalg.solve(Sv[0], Sv[1].T)
    return Phi, V_mk


def axis_rotation(Sv, m_max, gap_frac=0.05, norm_frac=0.02):
    """Principal-axis angle of the symmetrized VACF matrix vs lag,
    masked (None) where the eigenvalue gap or the norm is too small
    for the angle to be reliable (codeX correction on record)."""
    S0 = 0.5 * (Sv[0] + Sv[0].T)
    lam0 = float(np.linalg.eigvalsh(S0).max())
    n0 = float(np.linalg.norm(S0))
    out = []
    for m in range(m_max + 1):
        S = 0.5 * (Sv[m] + Sv[m].T)
        w, U = np.linalg.eigh(S)
        gap, nrm = float(w[-1] - w[0]), float(np.linalg.norm(S))
        ang = None
        if gap >= gap_frac * lam0 and nrm >= norm_frac * n0:
            v = U[:, -1]
            ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
        out.append(dict(lag=m, angle_deg=ang, gap=gap, norm=nrm))
    return out


if __name__ == "__main__":
    r = references()
    p = r["params"]
    print("frozen regime:", p)
    print("band:", r["band"], " rates:", np.round(r["rates"], 4))
    print("gain_total(h=1) nats:",
          0.5 * (np.linalg.slogdet(r["V_state"])[1]
                 - np.linalg.slogdet(r["V_kal1"])[1]))
    print("Vb1:", r["Vb1"])
    print("ideal-bank chain:", {k: v for k, v in r["chain"].items()
                                if not isinstance(v, list)})
    print("D_exact (sampled-displacement):", r["D_exact"])
