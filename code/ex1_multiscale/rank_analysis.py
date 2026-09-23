"""Predictive-rank analysis of the selected bank for Example 4 (numpy).

The multiscale analogue of Example 1's rank analysis. The embedding
Y = (X, zeta, F_1, ..., F_4) has five hidden coordinates, so by the
linear-Gaussian bound the predictive rank of ANY bank is at most five;
the analysis measures the spectrum of the selected k = 6 bank, the loss
of the rank-p projection, and the reconstruction of the hidden slow
forcings from the observed component alone.

Outputs: a printed report and figs/ex4_rank.pdf (three panels, same
layout as Example 1's rank figure).
"""

import json
import os

import numpy as np

import config as cfg
from multiscale_exact import (_memory_joint, acf_x, discrete_system,
                              cond_given_memory)
from memdiff.features import ema_features, rates_from_band

H_MAX = 5000          # horizons h*DT = 0.2 .. 1000 (five slow times)

FALLBACK_BAND = (0.037610339330866153, 15.0)
FALLBACK_K = 6


def selected_bank():
    """Exact selected bank: selection.json -> re-run selection -> fallback."""
    sel_path = os.path.join(cfg.OUT_DIR, "mem", "selection.json")
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            sel = json.load(f)
        return tuple(sel["band"]), int(sel["k"]), "selection.json"
    data_path = os.path.join(cfg.OUT_DIR, "data.npz")
    if os.path.exists(data_path):
        from conditioning import select_rates_by_prediction
        sel = select_rates_by_prediction(np.load(data_path)["X_train"],
                                         verbose=False)
        return tuple(sel["band"]), int(sel["k"]), "re-run selection"
    return FALLBACK_BAND, FALLBACK_K, "recorded pilot selection"


BAND, K, BAND_SRC = selected_bank()


def joint_blocks():
    """Covariance blocks of (X_n, m_n) + horizon cross-covariances,
    everything partialled on X_n (closed form)."""
    rates = rates_from_band(BAND, K)
    J = _memory_joint(rates, extra_lags=H_MAX)
    S, C, L, W = J["S"], J["C"], J["L"], J["W"]
    Sxx, sxm, Smm = S[0, 0], S[0, 1:], S[1:, 1:]
    Sm_x = Smm - np.outer(sxm, sxm) / Sxx
    G = np.empty((K, H_MAX))
    for h in range(1, H_MAX + 1):
        cm = W @ (C[h:h + L + 1] - C[h + 1:h + L + 2])
        G[:, h - 1] = cm - C[h] * sxm / Sxx
    return rates, J, S, Sm_x, G


def hidden_cross_cov(rates, L):
    """Cov(m_i, Y_n) for the s-dimensional embedding, closed form."""
    A_d, _, Sigma = discrete_system()
    s = A_d.shape[0]
    rho = np.exp(-np.asarray(rates) * cfg.DT)
    e0 = np.zeros(s); e0[0] = 1.0
    cross = np.zeros((len(rates), s))
    Al = np.eye(s)
    for l in range(L + 1):
        vec = (Al - Al @ A_d) @ (Sigma @ e0)
        cross += rho[:, None] ** l * vec[None, :]
        Al = Al @ A_d
    return cross, Sigma


def reconstruction_r2(S, cross, Sigma, T=None):
    """R^2 of the best linear reconstruction of the hidden coordinates
    from v = (X_n, m_n), optionally through the projection v -> T v."""
    cv = np.vstack([Sigma[0, 1:], cross[:, 1:]])
    Sv = S
    if T is not None:
        Sv, cv = T @ S @ T.T, T @ cv
    coef = np.linalg.solve(Sv, cv)
    r2 = np.einsum("ij,ij->j", cv, coef) / np.diag(Sigma)[1:]
    return r2, coef


def spectrum_loss():
    """(sv, eps_p, n_hid): predictive spectrum and worst-horizon loss of
    the rank-p projection for the selected bank (closed form; numpy).
    Shared with make_figs.py for the combined manuscript figure."""
    _, _, _, Sm_x, G = joint_blocks()
    n_hid = discrete_system()[0].shape[0] - 1
    Lc = np.linalg.cholesky(Sm_x)
    with np.errstate(all="ignore"):
        B = np.linalg.solve(Lc, G)
        U, sv, _ = np.linalg.svd(B, full_matrices=False)
    if not np.isfinite(sv).all():
        raise FloatingPointError("non-finite predictive spectrum")
    full_gain = (B ** 2).sum(axis=0)
    nz = full_gain > 0
    eps_p = np.empty(K)
    for p in range(1, K + 1):
        kept = ((U[:, :p].T @ B) ** 2).sum(axis=0)
        eps_p[p - 1] = np.max(1.0 - kept[nz] / full_gain[nz])
    return sv, eps_p, n_hid


def main():
    rates, J, S, Sm_x, G = joint_blocks()
    s_dim = discrete_system()[0].shape[0]
    n_hid = s_dim - 1
    print(f"selected bank: k = {K}, band = {np.round(BAND, 5)} [{BAND_SRC}]")
    print(f"embedding dim {s_dim}: hidden coordinates {n_hid} "
          f"(zeta + {n_hid - 1} forcings) -> predictive rank <= {n_hid}")

    Lc = np.linalg.cholesky(Sm_x)
    with np.errstate(all="ignore"):
        B = np.linalg.solve(Lc, G)
        U, sv, _ = np.linalg.svd(B, full_matrices=False)
    if not np.isfinite(sv).all():
        raise FloatingPointError("non-finite predictive spectrum")
    energy = np.cumsum(sv ** 2) / (sv ** 2).sum()
    print("predictive spectrum (singular values):",
          np.array2string(sv, precision=5))
    print("cumulative explained energy (squared):", np.round(energy, 5))

    full_gain = (B ** 2).sum(axis=0)
    nz = full_gain > 0
    eps_p = np.empty(K)
    for p in range(1, K + 1):
        kept = ((U[:, :p].T @ B) ** 2).sum(axis=0)
        eps_p[p - 1] = np.max(1.0 - kept[nz] / full_gain[nz])
    print("worst-horizon loss of the rank-p SVD projection:",
          np.array2string(eps_p, precision=4))

    # analytic factorization: rank <= n_hid for every horizon set
    A_d = discrete_system()[0]
    cross, Sigma = hidden_cross_cov(rates, J["L"])
    sxm = S[0, 1:]
    CmH_x = cross[:, 1:] - np.outer(sxm, Sigma[0, 1:]) / S[0, 0]
    Gam = np.empty((n_hid, H_MAX))
    Ah = np.eye(s_dim)
    for h in range(1, H_MAX + 1):
        Ah = Ah @ A_d
        Gam[:, h - 1] = Ah[0, 1:]
    with np.errstate(all="ignore"):
        fact_resid = (np.linalg.norm(B - np.linalg.solve(Lc, CmH_x) @ Gam)
                      / np.linalg.norm(B))
    print(f"analytic rank-{n_hid} factorization residual: {fact_resid:.2e}")

    # numerical rank and the projection
    p_rank = int(np.sum(sv > 1e-6 * sv[0]))
    print(f"numerical rank: {p_rank} (threshold 1e-6 sigma_1)")
    P = np.linalg.solve(Lc.T, U[:, :p_rank]).T

    _, V_full = cond_given_memory(rates)
    T = np.zeros((1 + p_rank, 1 + K)); T[0, 0] = 1.0; T[1:, 1:] = P
    C, RD, W_, L = J["C"], J["RD"], J["W"], J["L"]
    c = np.empty(1 + K)
    c[0] = C[1] - C[0]
    c[1:] = W_ @ RD[1:L + 2]
    S3, c3 = T @ S @ T.T, T @ c
    V_proj = RD[0] - c3 @ np.linalg.solve(S3, c3)
    print(f"one-step predictive variance: full bank {V_full:.8f}   "
          f"projected p={p_rank} {V_proj:.8f}   "
          f"rel diff {abs(V_proj - V_full) / V_full:.2e}")

    r2_full, _ = reconstruction_r2(S, cross, Sigma)
    r2_proj, coef3 = reconstruction_r2(S, cross, Sigma, T)
    names = ["zeta"] + [f"F(T={T_:g})" for T_ in cfg.FORCING_TIMES]
    print("R^2 reconstruction of the hidden coordinates from (X, m):")
    for nm, a, b in zip(names, r2_full, r2_proj):
        print(f"  {nm:10s} {a:.4f}   through (X, z): {b:.4f}")

    # exact-trajectory verification, focused on the two slowest forcings
    rng = np.random.default_rng(cfg.SEED_EVAL + 41)
    A_d, Q_d, Sig = discrete_system()
    n_burn_est = int(np.ceil(-np.log(cfg.BURN_TOL) / (rates.min() * cfg.DT)))
    n_show = int(400.0 / cfg.DT)                  # plotted window: 400 t.u.
    # R^2 must be estimated over many slow correlation times, not just the
    # plotted window (2 slow times give a uselessly noisy estimate)
    n_steps = n_burn_est + int(4000.0 / cfg.DT)
    Y = np.empty((n_steps, s_dim))
    Ls = np.linalg.cholesky(Sig + 1e-14 * np.eye(s_dim))
    Lq = np.linalg.cholesky(Q_d + 1e-14 * np.eye(s_dim))
    Y[0] = Ls @ rng.standard_normal(s_dim)
    for n in range(1, n_steps):
        Y[n] = A_d @ Y[n - 1] + Lq @ rng.standard_normal(s_dim)
    m, n_burn = ema_features(Y[None, :, 0], rates, cfg.DT, cfg.BURN_TOL)
    with np.errstate(all="ignore"):
        v3 = np.column_stack([Y[:, 0], m[0] @ P.T])
        Hrec = v3 @ coef3
    if not np.isfinite(Hrec).all():
        raise FloatingPointError("non-finite trajectory reconstruction")
    r2_emp = 1.0 - ((Hrec[n_burn:] - Y[n_burn:, 1:]) ** 2).mean(0) \
        / Y[n_burn:, 1:].var(0)
    print("empirical R^2 on an exact trajectory:", np.round(r2_emp, 4))

    # the IDENTIFIABLE objects: zeta and the total hidden forcing (the
    # observation responds only to the sum; individual components are
    # partially identifiable at best, see the table above)
    F_tot = Y[:, 2:].sum(1)
    F_rec = Hrec[:, 1:].sum(1)
    zeta, z_rec = Y[:, 1], Hrec[:, 0]
    r2_ftot = 1.0 - ((F_rec[n_burn:] - F_tot[n_burn:]) ** 2).mean() \
        / F_tot[n_burn:].var()
    r2_zeta = 1.0 - ((z_rec[n_burn:] - zeta[n_burn:]) ** 2).mean() \
        / zeta[n_burn:].var()
    print(f"empirical R^2: zeta {r2_zeta:.4f}, total forcing {r2_ftot:.4f}")

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, MODEL_ALPHA
    setup_style()
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    FLOOR = 1e-9
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))

    j = np.arange(1, K + 1)
    ax[0].semilogy(j, np.maximum(sv, FLOOR), "ko-")
    ax[0].axhline(FLOOR, color="k", ls=":", lw=1)
    ax[0].axvline(n_hid + 0.5, color="tab:red", ls="--", lw=1)
    ax[0].text(n_hid + 0.55, sv[0] * 0.3, f"rank $\\leq {n_hid}$",
               fontsize=8, color="tab:red")
    ax[0].set(xlabel="direction $j$", ylabel="singular value",
              title="predictive spectrum")
    ax[0].set_xticks(j)

    ax[1].semilogy(j, np.maximum(eps_p, FLOOR), "ko-")
    ax[1].axhline(FLOOR, color="k", ls=":", lw=1)
    ax[1].set(xlabel="kept dimensions $p$",
              ylabel=r"worst-horizon $\epsilon_p$",
              title="loss of the rank-$p$ SVD projection")
    ax[1].set_xticks(j)

    n_win = int(30.0 / cfg.DT)                # display window: tau rule
    t = (np.arange(n_steps)[n_burn:n_burn + n_win] - n_burn) * cfg.DT
    sl = slice(n_burn, n_burn + n_win)
    off = 4.0 * zeta.std()
    ax[2].plot(t, zeta[sl] + off, "k-", lw=0.9)
    ax[2].plot(t, z_rec[sl] + off, "-", color="tab:red",
               alpha=MODEL_ALPHA)
    ax[2].plot(t, F_tot[sl], "k-", lw=0.9, label="exact")
    ax[2].plot(t, F_rec[sl], "-", color="tab:red",
               alpha=MODEL_ALPHA, label=r"from $(x_n, z_n)$")
    ax[2].text(0.02, 0.97,
               rf"$\zeta_t$ (offset), $R^2={r2_zeta:.4f}$",
               transform=ax[2].transAxes, va="top", fontsize=8)
    ax[2].text(0.02, 0.06,
               rf"total forcing, $R^2={r2_ftot:.3f}$",
               transform=ax[2].transAxes, va="bottom", fontsize=8)
    ax[2].set(xlabel=r"$t$", title="hidden variables recovered")
    ax[2].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{cfg.FIG_DIR}/ex4_rank.pdf")
    print(f"figure -> {cfg.FIG_DIR}/ex4_rank.pdf")


if __name__ == "__main__":
    main()
