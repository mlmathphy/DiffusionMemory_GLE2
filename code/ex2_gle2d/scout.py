"""ex2_gle2d scout: closed-form gates + sampler feasibility (SCOUT.md).

Single command, numpy/scipy only. All gates, thresholds, the grid, and
the deterministic selection rule are predeclared in SCOUT.md and
encoded below; every grid point's full record is written to the JSON.

    python3 scout.py --quick     # ~1 min smoke, out/scout_quick.json,
                                 # reduced settings, NO selection
    python3 scout.py             # full grid,   out/scout_results.json

Revision after codeX code review (2026-09-07): steady-state Kalman
full-history reference (nested with the bank, so the mean-gap identity
holds at every horizon; raw gaps recorded, never silently clipped);
pipeline kappa_s label scaling; distribution-sensitive projected-W1
check in gate D calibrated on exact-reference draws; sampled-
displacement diffusion-tensor normalization DT/2; quick mode never
selects and records its actual reduced settings.

PROTOCOL v2 (2026-09-07, after the v1 full run): the v1 grid was a
NO-GO solely on gate B (cross-component history gain <= 2.8% of the
total everywhere; A passed at all four f_mem = 0.9 points, C passed
with k = 4-5, D never ran). Three-way agreed amendment: the paper
claim is ACCURATE LEARNING OF A COUPLED 2D GLE, not cross-component
history transfer, so B becomes a REPORTED quantity (kept in the
record as cross_gate_v1 against the v1 thresholds) while A, C, D
remain required. The v1 selection rule was tied to gate B and is
replaced: max gain_total; tie: min k_sel; tie: min index. The v1
NO-GO record (out/scout_results.json) is preserved; v2 writes
out/scout_results_v2.json. Thresholds were NOT changed.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import expm, cho_factor, cho_solve
from scipy.special import ndtri

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from memdiff.features import ema_features, rates_from_band
from memdiff.sampler_np import TrainingFreeSamplerNP

# ---------------------------------------------------------- declared protocol
DT = 0.2                       # observation interval (declared)
KBT = 1.0
TAU1 = 1.0
TRACE_SHARE = 1.0              # a_i tau_i for BOTH modes -> tr G_mem = 2
F_MEM_GRID = (0.5, 0.75, 0.9)
TAU2_GRID = (5.0, 20.0)
THETA_GRID = (45.0, 60.0)      # degrees between u1 and u2

ACF_SIG = 0.01                 # state-ACF significance level for the band
BAND_MAX_LAG = 2000
K_CAP = 5                      # conditioning dim 2 + 2k <= 12
BURN_TOL = 1e-3

# gates (SCOUT.md table)
GATE_A_NATS = 0.05
GATE_B_EACH = 0.005
GATE_B_FRAC = 0.10
GATE_C_RETENTION = 0.90        # fraction of the h=1 information gain
GATE_C_LOSTFRAC = 0.10
GATE_C_VACF = 0.05
GATE_C_DIFF = 0.10
GATE_D_W2 = 0.05
GATE_D_PROJ_FACTOR = 2.0       # vs exact-reference calibration draws

# predeclared projection directions for the distributional check,
# applied in the frame whitened by the exact conditional covariance
PROJ_DIRS = np.array([[1.0, 0.0], [0.0, 1.0],
                      [np.sqrt(0.5), np.sqrt(0.5)],
                      [np.sqrt(0.5), -np.sqrt(0.5)]])

# validity thresholds
VAL_LYAP = 1e-8
VAL_QD_EIG = -1e-10
VAL_WINDOW_NATS = 1e-3         # window used for gate B / reporting
VAL_KAL_TOL = 1e-12            # Kalman fixed-point tolerance
VAL_NESTING = -1e-6            # min allowed lost-frac before clipping
VAL_GIBBS = 0.10

# sampler settings, mirroring the pipeline (ex1 config); label scaling
# follows build_tuples: S = increments * kappa_s, kappa_s = 1/std(D)
NU = 0.2
EPS_METRIC = 0.3

SIZES_FULL = dict(n_traj=8, l_use=25000, n_lab=100000, n_anchor=48,
                  n_samp=1024, n_ode=400, j=1024, m_full=1024,
                  m_half=512, band_lag=BAND_MAX_LAG)
SIZES_QUICK = dict(n_traj=2, l_use=4000, n_lab=20000, n_anchor=8,
                   n_samp=256, n_ode=50, j=256, m_full=256,
                   m_half=128, band_lag=600)
SEED_BASE = 20260907

DT_SENSITIVITY = (0.1, 0.4)    # reported at the selected regime only


# ---------------------------------------------------------------- linear alg

def dlyap(T, Q):
    """Solve S = T S T^T + Q (row-major vectorization; small dense T)."""
    n = T.shape[0]
    x = np.linalg.solve(np.eye(n * n) - np.kron(T, T), Q.reshape(-1))
    S = x.reshape(n, n)
    return 0.5 * (S + S.T)


def psd_sqrt(V):
    w, U = np.linalg.eigh(V)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T


def psd_inv_sqrt(V):
    w, U = np.linalg.eigh(V)
    return (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T


def gauss_w2sq(mu0, V0, mu1, V1):
    """Squared Wasserstein-2 distance between two Gaussians."""
    s0 = psd_sqrt(V0)
    wm = np.linalg.eigvalsh(s0 @ V1 @ s0)
    cross = np.sqrt(np.clip(wm, 0.0, None)).sum()
    return (float(np.sum((mu1 - mu0) ** 2))
            + float(np.trace(V0) + np.trace(V1)) - 2.0 * float(cross))


def proj_w1_stat(y):
    """Max over PROJ_DIRS of the 1D empirical-vs-N(0,1) W1 distance.

    y: (n, 2) whitened samples. Distribution-sensitive: any conditional
    shape error (skew, tails, multimodality) survives this statistic
    even when mean and covariance are exact.
    """
    n = len(y)
    q = ndtri((np.arange(n) + 0.5) / n)
    stat = 0.0
    for u in PROJ_DIRS:
        p = np.sort(y @ u)
        stat = max(stat, float(np.mean(np.abs(p - q))))
    return stat


def logdet(V):
    sign, ld = np.linalg.slogdet(V)
    if sign <= 0:
        raise FloatingPointError("non-PD conditional covariance")
    return float(ld)


# ---------------------------------------------------------------- the model

def build_ct(f_mem, tau2, theta_deg):
    """Continuous-time embedding: A, Q_c, exact Gibbs Sigma, params."""
    th = np.deg2rad(theta_deg)
    u1 = np.array([1.0, 0.0])
    u2 = np.array([np.cos(th), np.sin(th)])
    a1 = TRACE_SHARE / TAU1
    a2 = TRACE_SHARE / tau2
    g_tr = 2.0 * TRACE_SHARE
    gamma = g_tr * (1.0 - f_mem) / (2.0 * f_mem)
    A = np.zeros((4, 4))
    A[:2, :2] = -gamma * np.eye(2)
    A[:2, 2] = u1
    A[:2, 3] = u2
    A[2, :2] = -a1 * u1
    A[3, :2] = -a2 * u2
    A[2, 2] = -1.0 / TAU1
    A[3, 3] = -1.0 / tau2
    Qc = np.diag([2 * gamma * KBT, 2 * gamma * KBT,
                  2 * a1 * KBT / TAU1, 2 * a2 * KBT / tau2])
    Sigma = np.diag([KBT, KBT, a1 * KBT, a2 * KBT])
    G_mem = (a1 * TAU1 * np.outer(u1, u1) + a2 * tau2 * np.outer(u2, u2))
    par = dict(gamma=gamma, a1=a1, a2=a2, tau1=TAU1, tau2=tau2,
               theta_deg=theta_deg, f_mem=f_mem,
               u2=u2.tolist(), g_mem_eigs=np.linalg.eigvalsh(G_mem).tolist())
    return A, Qc, Sigma, par


def discretize(A, Sigma, dt):
    Ad = expm(A * dt)
    Qd = Sigma - Ad @ Sigma @ Ad.T
    return Ad, 0.5 * (Qd + Qd.T)


def sv_series(Ad, Sigma, n):
    """Sv[m] = Cov(v_{t+m}, v_t) = [A_d^m Sigma]_{vv}, m = 0..n-1."""
    Sv = np.empty((n, 2, 2))
    C = Sigma.copy()
    for m in range(n):
        Sv[m] = C[:2, :2]
        C = Ad @ C
    return Sv


# ------------------------------------------------ full-history reference

def kalman_full_reference(Ad, Qd, Sigma, hs, max_iter=200000):
    """Steady-state exact-observation Kalman filter reference.

    v is observed exactly, so the recursion is: predict
    P- = A_d P A_d^T + Q_d, then condition on the v-block. The fixed
    point P_inf = Cov(Y_n | v_{-inf..n}) gives the INFINITE-history
    conditional at every horizon,

        V_full(h) = [A_d^h P_inf A_d^hT + sum_{j<h} A_d^j Q_d A_d^jT]_vv.

    The bank sigma-algebra is strictly nested inside the infinite past,
    so tr(V_bank - V_full) >= 0 holds exactly (mean-gap identity);
    negative values can only be roundoff and are recorded raw.
    """
    def condition_v(M):
        G = np.linalg.solve(M[:2, :2], M[:2, :])
        Mc = M - M[:, :2] @ G
        return 0.5 * (Mc + Mc.T)

    P = condition_v(Sigma)
    delta = np.inf
    for it in range(max_iter):
        P_new = condition_v(Ad @ P @ Ad.T + Qd)
        delta = float(np.max(np.abs(P_new - P)))
        P = P_new
        if delta < VAL_KAL_TOL:
            break
    V = {}
    M = P.copy()
    for h in range(1, max(hs) + 1):
        M = Ad @ M @ Ad.T + Qd
        if h in hs:
            V[h] = M[:2, :2].copy()
    return V, it + 1, delta


# ------------------------------------------------------- window conditionals

def build_window_cov(Sv, M):
    """Block-Toeplitz Cov of w = (v_n, v_{n-1}, ..., v_{n-M+1})."""
    G4 = np.empty((M, 2, M, 2))
    for d in range(M):
        i = np.arange(M - d)
        G4[i, :, i + d, :] = Sv[d]
        if d:
            G4[i + d, :, i, :] = Sv[d].T
    return G4.reshape(2 * M, 2 * M)


def cross_rows(Sv, h, M):
    """Cov(v_{n+h}, w) as a (2, 2M) matrix."""
    return np.concatenate([Sv[h + j] for j in range(M)], axis=1)


def window_analysis(Sv, M_full, M_half):
    """Finite-window conditionals: cross-component gains (gate B, a
    comparison of two NESTED windows), the window-vs-Kalman consistency
    number, and the reported window-at-bank-dim gain."""
    Sv0 = Sv[0]
    G = build_window_cov(Sv, M_full)
    cho = cho_factor(G, lower=True)
    c1 = cross_rows(Sv, 1, M_full)
    V_win1 = Sv0 - c1 @ cho_solve(cho, c1.T)
    V_win1 = 0.5 * (V_win1 + V_win1.T)
    V_state1 = Sv0 - Sv[1] @ np.linalg.solve(Sv0, Sv[1].T)
    gain_win = 0.5 * (logdet(V_state1) - logdet(V_win1))

    ch = cho_factor(G[:2 * M_half, :2 * M_half], lower=True)
    c1h = cross_rows(Sv, 1, M_half)
    V_half = Sv0 - c1h @ cho_solve(ch, c1h.T)
    gain_half = 0.5 * (logdet(V_state1) - logdet(V_half))

    cross = {}
    for alpha in (0, 1):
        own = np.r_[np.array([0, 1]), 2 * np.arange(1, M_full) + alpha]
        Gs = G[np.ix_(own, own)]
        cs = c1[alpha, own]
        v_own = Sv0[alpha, alpha] - cs @ np.linalg.solve(Gs, cs)
        v_both = V_win1[alpha, alpha]
        cross[alpha] = 0.5 * float(np.log(v_own / v_both))

    def window_at_dim(W):
        """Reported: ideal raw window of W blocks (dim 2W)."""
        n = 2 * W
        V = Sv0 - c1[:, :n] @ np.linalg.solve(G[:n, :n], c1[:, :n].T)
        return 0.5 * (logdet(V_state1) - logdet(V))

    return dict(gain_win=gain_win, gain_half=gain_half,
                cross_x=cross[0], cross_y=cross[1],
                window_at_dim=window_at_dim)


# ---------------------------------------------------------- bank conditionals

def bank_analysis(Ad, Qd, Sigma, Sv, rates, hs, m_acf):
    """Exact bank conditionals + ideal-bank closed-loop chain."""
    k = len(rates)
    rho = np.repeat(np.exp(-np.asarray(rates) * DT), 2)   # rate-major
    R = np.diag(rho)
    E = np.tile(np.eye(2), (k, 1))                        # (2k, 2)
    P = np.eye(4)[:2]                                     # v-selector
    T_aug = np.block([[Ad, np.zeros((4, 2 * k))],
                      [E @ P @ (Ad - np.eye(4)), R]])
    L = np.vstack([np.eye(4), E @ P])
    Sz = dlyap(T_aug, L @ Qd @ L.T)
    zidx = np.r_[np.arange(2), np.arange(4, 4 + 2 * k)]
    Czz = Sz[np.ix_(zidx, zidx)]
    Sv0 = Sv[0]

    h_max = max(hs)
    V_bank, c_h = {}, {}
    Adh = np.eye(4)
    CyZ = Sz[:4, :]                                       # Cov(Y_n, Z_n)
    for h in range(1, h_max + 1):
        Adh = Ad @ Adh
        if h in hs:
            c = (P @ Adh @ CyZ)[:, zidx]
            V = Sv0 - c @ np.linalg.solve(Czz, c.T)
            V_bank[h] = 0.5 * (V + V.T)
            c_h[h] = c
    Gamma = np.linalg.solve(Czz, c_h[1].T).T              # (2, 2+2k)
    Vb1 = V_bank[1]

    # ideal-bank chain on xi = (v, m)
    Sel = np.hstack([np.eye(2), np.zeros((2, 2 * k))])
    T_ch = np.vstack([Gamma,
                      E @ (Gamma - Sel)
                      + np.hstack([np.zeros((2 * k, 2)), R])])
    rho_ch = float(np.max(np.abs(np.linalg.eigvals(T_ch))))
    chain = dict(rho=rho_ch, stable=bool(rho_ch < 1.0))
    if rho_ch < 1.0:
        Lc = np.vstack([np.eye(2), E])
        S_ch = dlyap(T_ch, Lc @ Vb1 @ Lc.T)
        # VACF matrix of the chain vs exact, m = 0..m_acf
        scale = float(np.max(np.abs(Sv0)))
        err, C = 0.0, S_ch.copy()
        for m in range(m_acf + 1):
            err = max(err, float(np.max(np.abs(C[:2, :2] - Sv[m]))))
            C = T_ch @ C
        chain["vacf_err"] = err / scale
        # SAMPLED-DISPLACEMENT diffusion tensor: for x_N = DT sum v_n,
        # Cov(x_N)/(2 N DT) -> (DT/2)[Sv(0) + sum_{m>=1}(Sv(m)+Sv(m)^T)]
        # (exact geometric sums). This is the discrete-sampling
        # transport tensor, NOT the continuous-time integral of the
        # velocity ACF; the two differ at finite DT.
        S1e = (Ad @ np.linalg.solve(np.eye(4) - Ad, Sigma))[:2, :2]
        D_ex = 0.5 * DT * (Sv0 + S1e + S1e.T)
        n = 2 + 2 * k
        S1c = (T_ch @ np.linalg.solve(np.eye(n) - T_ch, S_ch))[:2, :2]
        D_ch = 0.5 * DT * (S_ch[:2, :2] + S1c + S1c.T)
        chain["diff_sampled_exact"] = D_ex.tolist()
        chain["diff_sampled_chain"] = D_ch.tolist()
        chain["diff_err"] = float(np.linalg.norm(D_ch - D_ex)
                                  / np.linalg.norm(D_ex))
    return dict(V_bank=V_bank, Gamma=Gamma, Vb1=Vb1, chain=chain)


# ------------------------------------------------------------------- gate D

def gate_d_sampler(Ad, Qd, Sigma, rates, Gamma, Vb1, seed, sizes):
    """Real-sampler feasibility at the pipeline's settings.

    Labels follow the pipeline convention (build_tuples): the sampler
    sees increments scaled by kappa_s = 1/std(increments); sampled
    labels are divided by kappa_s before every comparison. Two checks
    per anchor: Gaussian-moment W2 (mean+cov) and the projected-W1
    statistic (distribution-sensitive), the latter calibrated against
    exact-reference draws of the same size.
    """
    lam_min = float(np.min(rates))
    n_burn = int(np.ceil(-np.log(BURN_TOL) / (lam_min * DT)))
    L = n_burn + sizes["l_use"]
    cQ = psd_sqrt(Qd)
    sqSig = np.sqrt(np.diag(Sigma))                       # Sigma is diagonal

    def sim(nt, sd):
        rng = np.random.default_rng(sd)
        Y = np.empty((nt, L + 1, 4))
        Y[:, 0] = rng.standard_normal((nt, 4)) * sqSig
        for t in range(L):
            Y[:, t + 1] = (Y[:, t] @ Ad.T
                           + rng.standard_normal((nt, 4)) @ cQ.T)
        return Y[:, :, :2]

    v = sim(sizes["n_traj"], seed)
    m, nb = ema_features(v, np.asarray(rates), DT, BURN_TOL)
    nb = max(nb, 1)
    C_all = np.concatenate([v[:, nb:L], m[:, nb:L]], axis=2)
    S_all = v[:, nb + 1:L + 1] - v[:, nb:L]
    C_all = C_all.reshape(-1, C_all.shape[-1])
    S_all = S_all.reshape(-1, 2)
    kappa_s = float(1.0 / S_all.std())                    # pipeline scaling
    gibbs_dev = float(np.max(np.abs(np.cov(v.reshape(-1, 2).T)
                                    - KBT * np.eye(2))))
    stride = max(1, int(np.ceil(len(C_all) / sizes["n_lab"])))
    C_pool = C_all[::stride][:sizes["n_lab"]]
    S_pool = S_all[::stride][:sizes["n_lab"]] * kappa_s

    va = sim(1, seed + 1)
    ma, _ = ema_features(va, np.asarray(rates), DT, BURN_TOL)
    t_anchor = np.linspace(nb, L - 1, sizes["n_anchor"]).astype(int)
    sampler = TrainingFreeSamplerNP(C_pool, S_pool, sizes["j"], NU,
                                    EPS_METRIC, sizes["n_ode"])
    Winv = psd_inv_sqrt(Vb1)
    sqV = psd_sqrt(Vb1)
    tr_ref = float(np.trace(Vb1))
    w2n, w2_floor, proj, proj_cal, ess, rad = [], [], [], [], [], []
    for i, t in enumerate(t_anchor):
        zeta = np.concatenate([va[0, t], ma[0, t]])
        mu_dv = Gamma @ zeta - va[0, t]
        samp, diag = sampler.sample_conditional(
            zeta[None, :], sizes["n_samp"], seed=seed + 100 + i)
        dv = samp / kappa_s
        w2n.append(gauss_w2sq(mu_dv, Vb1, dv.mean(axis=0),
                              np.cov(dv.T)) / tr_ref)
        proj.append(proj_w1_stat((dv - mu_dv) @ Winv))
        # calibration: exact-reference draws of the same size
        g = np.random.default_rng(seed + 500 + i).standard_normal(
            (sizes["n_samp"], 2))
        dv_cal = mu_dv + g @ sqV.T
        w2_floor.append(gauss_w2sq(mu_dv, Vb1, dv_cal.mean(axis=0),
                                   np.cov(dv_cal.T)) / tr_ref)
        proj_cal.append(proj_w1_stat(g))
        ess.append(diag["ess"])
        rad.append(diag["radius"])
        if (i + 1) % 16 == 0:
            print(f"      anchor {i + 1}/{sizes['n_anchor']}")
    return dict(kappa_s=kappa_s,
                w2_moment_median=float(np.median(w2n)),
                w2_moment_p90=float(np.quantile(w2n, 0.9)),
                w2_moment_all=list(map(float, w2n)),
                w2_moment_floor_measured=float(np.median(w2_floor)),
                proj_w1_median=float(np.median(proj)),
                proj_w1_p90=float(np.quantile(proj, 0.9)),
                proj_w1_all=list(map(float, proj)),
                proj_w1_calib_median=float(np.median(proj_cal)),
                proj_dirs=PROJ_DIRS.tolist(),
                ess_median=float(np.median(ess)),
                ess_p5=float(np.quantile(ess, 0.05)),
                radius_median=float(np.median(rad)),
                n_labels=int(len(C_pool)), stride=int(stride),
                gibbs_dev=gibbs_dev)


# ---------------------------------------------------------------- one point

def run_point(idx, f_mem, tau2, theta, quick, sizes):
    A, Qc, Sigma, par = build_ct(f_mem, tau2, theta)
    rec = dict(index=idx, params=par)
    validity = {}
    validity["lyap_resid"] = float(np.max(np.abs(A @ Sigma + Sigma @ A.T
                                                 + Qc)))
    validity["a_hurwitz"] = bool(np.max(np.linalg.eigvals(A).real) < 0)
    Ad, Qd = discretize(A, Sigma, DT)
    validity["qd_min_eig"] = float(np.linalg.eigvalsh(Qd).min())

    hs = sorted({1, int(round(tau2 / (2 * DT))), int(round(tau2 / DT)),
                 int(round(2 * tau2 / DT))})
    m_acf = int(round(3 * tau2 / DT))
    n_sv = max(sizes["m_full"] + 2, sizes["band_lag"] + 2, m_acf + 2)
    Sv = sv_series(Ad, Sigma, n_sv)
    V_state = {h: Sv[0] - Sv[h] @ np.linalg.solve(Sv[0], Sv[h].T)
               for h in hs}

    # declared rate band: slow end from the exact STATE ACF
    r = np.trace(Sv, axis1=1, axis2=2) / np.trace(Sv[0])
    sig = np.nonzero(np.abs(r[1:sizes["band_lag"] + 1]) > ACF_SIG)[0]
    l_max = int(sig[-1]) + 1 if len(sig) else 1
    lam_lo, lam_hi = 1.0 / (l_max * DT), 1.0 / DT
    if lam_lo >= lam_hi:
        lam_lo = lam_hi / 10.0
    rec["band"] = [lam_lo, lam_hi]

    # full-history reference: converged steady-state Kalman filter
    V_kal, kal_iters, kal_delta = kalman_full_reference(Ad, Qd, Sigma, hs)
    validity["kalman_iters"] = kal_iters
    validity["kalman_delta"] = kal_delta
    validity["kalman_converged"] = bool(kal_delta < VAL_KAL_TOL)
    gain_total = 0.5 * (logdet(V_state[1]) - logdet(V_kal[1]))
    rec["gain_total_nats"] = gain_total

    # finite-window machinery: gate B (nested windows) + reporting
    win = window_analysis(Sv, sizes["m_full"], sizes["m_half"])
    rec["gain_window_nats"] = win["gain_win"]
    rec["window_convergence_nats"] = abs(win["gain_win"]
                                         - win["gain_half"])
    rec["window_vs_kalman_nats"] = abs(win["gain_win"] - gain_total)
    validity["window_converged"] = bool(
        rec["window_convergence_nats"] < VAL_WINDOW_NATS)
    rec["cross_x_nats"] = win["cross_x"]
    rec["cross_y_nats"] = win["cross_y"]

    gate_a = bool(gain_total >= GATE_A_NATS)
    # v2: cross-component gain is REPORTED, not gated; the boolean
    # against the v1 thresholds is kept in the record for continuity.
    cross_sum = win["cross_x"] + win["cross_y"]
    cross_gate_v1 = bool(min(win["cross_x"], win["cross_y"]) >= GATE_B_EACH
                         and cross_sum >= GATE_B_FRAC * gain_total)
    rec["cross_gate_v1"] = cross_gate_v1

    # gate C: smallest k <= K_CAP passing every sub-check
    k_table, k_sel, bank_sel = {}, None, None
    nesting_ok = True
    for k in range(1, K_CAP + 1):
        rates = rates_from_band((lam_lo, lam_hi), k)
        bk = bank_analysis(Ad, Qd, Sigma, Sv, rates, hs, m_acf)
        gain_bank = 0.5 * (logdet(V_state[1]) - logdet(bk["Vb1"]))
        retention = gain_bank / gain_total if gain_total > 0 else 0.0
        lost, lost_raw = {}, {}
        for h in hs:
            num = float(np.trace(bk["V_bank"][h] - V_kal[h]))
            den = float(np.trace(V_state[h] - V_kal[h]))
            lost_raw[h] = num / den if den > 1e-12 else None
            if lost_raw[h] is not None and lost_raw[h] < VAL_NESTING:
                nesting_ok = False        # identity violated: not roundoff
            lost[h] = (max(0.0, num) / den) if den > 1e-12 else None
        lost_vals = [x for x in lost.values() if x is not None]
        ch = bk["chain"]
        ok = (retention >= GATE_C_RETENTION
              and all(x <= GATE_C_LOSTFRAC for x in lost_vals)
              and ch["stable"]
              and ch.get("vacf_err", np.inf) <= GATE_C_VACF
              and ch.get("diff_err", np.inf) <= GATE_C_DIFF)
        k_table[k] = dict(rates=list(map(float, rates)),
                          dim=2 + 2 * k,
                          retention_frac=retention,
                          gain_bank_nats=gain_bank,
                          lost_frac={str(h): lost[h] for h in hs},
                          lost_frac_raw={str(h): lost_raw[h] for h in hs},
                          chain_rho=ch["rho"],
                          vacf_err=ch.get("vacf_err"),
                          diff_err=ch.get("diff_err"),
                          diff_sampled_exact=ch.get("diff_sampled_exact"),
                          diff_sampled_chain=ch.get("diff_sampled_chain"),
                          passes=bool(ok))
        if ok and k_sel is None:
            k_sel, bank_sel = k, bk
    gate_c = k_sel is not None
    validity["nesting_ok"] = bool(nesting_ok)
    rec["k_table"] = k_table
    rec["k_sel"] = k_sel
    if k_sel is not None:
        rec["window_gain_at_dim_nats"] = win["window_at_dim"](k_sel + 1)

    validity["valid"] = bool(validity["lyap_resid"] < VAL_LYAP
                             and validity["a_hurwitz"]
                             and validity["qd_min_eig"] > VAL_QD_EIG
                             and validity["kalman_converged"]
                             and validity["window_converged"]
                             and nesting_ok)
    rec["validity"] = validity
    rec["gates"] = dict(A=gate_a, C=gate_c)

    # gate D: real sampler. Full run (v2): where A and C pass and the
    # point is valid. Quick: point 0 only, as an API/shape smoke -- its
    # output is marked smoke_only, sets no gate, never feeds a selection.
    run_d = (validity["valid"] and gate_a and gate_c) \
        if not quick else (idx == 0)
    if run_d:
        rates = rates_from_band((lam_lo, lam_hi),
                                k_sel if k_sel else 2)
        bk = bank_sel if bank_sel is not None else bank_analysis(
            Ad, Qd, Sigma, Sv, rates, hs, m_acf)
        print("    gate D: simulating + sampling ...")
        d = gate_d_sampler(Ad, Qd, Sigma, rates, bk["Gamma"], bk["Vb1"],
                           SEED_BASE + idx, sizes)
        d["gibbs_ok"] = bool(d["gibbs_dev"] < VAL_GIBBS)
        rec["gate_d"] = d
        if quick:
            rec["gate_d"]["smoke_only"] = True
        else:
            rec["gates"]["D"] = bool(
                d["w2_moment_median"] <= GATE_D_W2
                and d["proj_w1_median"]
                <= GATE_D_PROJ_FACTOR * d["proj_w1_calib_median"]
                and d["gibbs_ok"])
    rec["pass_all"] = bool((not quick) and validity["valid"] and gate_a
                           and gate_c and rec["gates"].get("D", False))
    return rec


def dt_sensitivity(f_mem, tau2, theta):
    out = {}
    A, Qc, Sigma, _ = build_ct(f_mem, tau2, theta)
    for dt in DT_SENSITIVITY:
        Ad, Qd = discretize(A, Sigma, dt)
        Sv1 = sv_series(Ad, Sigma, 2)
        V_state1 = Sv1[0] - Sv1[1] @ np.linalg.solve(Sv1[0], Sv1[1].T)
        V, iters, delta = kalman_full_reference(Ad, Qd, Sigma, [1])
        out[str(dt)] = dict(
            gain_total_nats=0.5 * (logdet(V_state1) - logdet(V[1])),
            kalman_iters=iters, kalman_delta=delta)
    return out


# --------------------------------------------------------------------- main

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def print_diagnostics():
    """Print the saved gate-D diagnostics (reads JSON only, no compute)."""
    for name in ("out/scout_results_v2.json", "out/scout_results.json"):
        if os.path.exists(name):
            break
    else:
        print("no saved scout results found under out/")
        return
    with open(name) as f:
        result = json.load(f)
    print(f"gate-D diagnostics from {name} "
          f"(protocol v{result['protocol'].get('version', 1)})")
    for p in result["points"]:
        d = p.get("gate_d")
        if not d:
            continue
        pr = p["params"]
        print(f"\npoint {p['index']} (f_mem={pr['f_mem']} "
              f"tau2={pr['tau2']} theta={pr['theta_deg']}deg) "
              f"k={p['k_sel']}"
              + ("  [smoke_only]" if d.get("smoke_only") else ""))
        print(f"  moment W2 med/p90: {d['w2_moment_median']:.6f} / "
              f"{d['w2_moment_p90']:.6f}  (limit {GATE_D_W2}; "
              f"measured floor {d['w2_moment_floor_measured']:.6f})")
        print(f"  projected W1 med/p90: {d['proj_w1_median']:.6f} / "
              f"{d['proj_w1_p90']:.6f}  (limit "
              f"{GATE_D_PROJ_FACTOR * d['proj_w1_calib_median']:.6f} "
              f"= {GATE_D_PROJ_FACTOR} x calib "
              f"{d['proj_w1_calib_median']:.6f})")
        print(f"  Gibbs deviation: {d['gibbs_dev']:.6f} "
              f"(limit {VAL_GIBBS})")
        print(f"  ESS median/p5: {d['ess_median']:.1f} / "
              f"{d['ess_p5']:.1f}   radius median: "
              f"{d['radius_median']:.6f}")
        print(f"  kappa_s {d['kappa_s']:.4f}  reference tuples "
              f"{d['n_labels']} (stride {d['stride']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="print saved gate-D diagnostics and exit")
    args = ap.parse_args()
    if args.diag:
        print_diagnostics()
        return
    sizes = SIZES_QUICK if args.quick else SIZES_FULL

    grid = [(f, t, th) for f in F_MEM_GRID for t in TAU2_GRID
            for th in THETA_GRID]
    records = []
    for idx, (f, t, th) in enumerate(grid):
        print(f"[{idx + 1:2d}/{len(grid)}] f_mem={f} tau2={t} "
              f"theta={th}deg")
        rec = run_point(idx, f, t, th, args.quick, sizes)
        g = rec["gates"]
        print(f"    gain={rec['gain_total_nats']:.4f} nats  "
              f"cross=({rec['cross_x_nats']:.4f},"
              f"{rec['cross_y_nats']:.4f} reported)  "
              f"k_sel={rec['k_sel']}  "
              f"A={g['A']} C={g['C']} D={g.get('D', '-')}")
        records.append(rec)

    # quick mode NEVER selects: reduced windows/labels are not a
    # scientific basis for freezing a regime.
    passing = [] if args.quick else [r for r in records if r["pass_all"]]
    selected = None
    if passing:
        # v2 rule (the v1 cross-gain rule was tied to retired gate B)
        selected = sorted(
            passing,
            key=lambda r: (-r["gain_total_nats"], r["k_sel"],
                           r["index"]))[0]

    result = dict(
        protocol=dict(version=2,
                      amendment=("2026-09-07: gate B (cross-component "
                                 "history gain) demoted to REPORTED "
                                 "after the v1 NO-GO; paper claim is "
                                 "accurate coupled-2D-GLE learning, "
                                 "not cross-history transfer; A/C/D "
                                 "unchanged, thresholds unchanged; v1 "
                                 "record preserved in "
                                 "out/scout_results.json"),
                      DT=DT, KBT=KBT, tau1=TAU1, trace_share=TRACE_SHARE,
                      grid_f_mem=list(F_MEM_GRID),
                      grid_tau2=list(TAU2_GRID),
                      grid_theta=list(THETA_GRID),
                      acf_sig=ACF_SIG, k_cap=K_CAP, dim_cap=2 + 2 * K_CAP,
                      gates=dict(A_nats=GATE_A_NATS,
                                 C_retention_frac=GATE_C_RETENTION,
                                 C_lostfrac=GATE_C_LOSTFRAC,
                                 C_vacf=GATE_C_VACF, C_diff=GATE_C_DIFF,
                                 D_w2_moment=GATE_D_W2,
                                 D_proj_factor=GATE_D_PROJ_FACTOR),
                      reported_v1_cross_gate=dict(B_each=GATE_B_EACH,
                                                  B_frac=GATE_B_FRAC),
                      sampler=dict(J=sizes["j"], nu=NU, eps=EPS_METRIC,
                                   n_ode=sizes["n_ode"],
                                   n_anchors=sizes["n_anchor"],
                                   n_samples=sizes["n_samp"],
                                   label_scaling="kappa_s = 1/std(dv), "
                                                 "build_tuples convention"),
                      sizes=dict(sizes),
                      proj_dirs=PROJ_DIRS.tolist(),
                      selection_rule=("v2: max gain_total; tie: min "
                                      "k_sel; tie: min index"),
                      quick=bool(args.quick)),
        points=records,
        n_passing=len(passing),
        selected=None if selected is None else dict(
            index=selected["index"], params=selected["params"],
            k_sel=selected["k_sel"], band=selected["band"],
            cross_sum_nats=(selected["cross_x_nats"]
                            + selected["cross_y_nats"]),
            gain_total_nats=selected["gain_total_nats"]))
    if args.quick:
        result["note"] = ("smoke run: reduced settings recorded in "
                          "protocol.sizes; no scientific selection")

    if selected is not None and not args.quick:
        p = selected["params"]
        print("computing DT sensitivity at the selected regime ...")
        result["dt_sensitivity"] = dt_sensitivity(
            p["f_mem"], p["tau2"], p["theta_deg"])

    os.makedirs("out", exist_ok=True)
    name = "out/scout_quick.json" if args.quick else \
        "out/scout_results_v2.json"   # v1 NO-GO record kept alongside
    with open(name, "w") as f:
        json.dump(jsonable(result), f, indent=1)

    print("\n================ DECISION ================")
    if args.quick:
        print("smoke complete: shapes/APIs exercised; quick mode "
              "never selects a regime.")
    else:
        print(f"passing points: {len(passing)} / {len(grid)}")
        if selected is None:
            print("NO-GO: no grid point passes all gates.")
        else:
            p = selected["params"]
            print(f"SELECTED index {selected['index']}: "
                  f"f_mem={p['f_mem']} tau2={p['tau2']} "
                  f"theta={p['theta_deg']}deg k={selected['k_sel']} "
                  f"(dim {2 + 2 * selected['k_sel']})")
    print(f"record: {name}")


if __name__ == "__main__":
    main()
