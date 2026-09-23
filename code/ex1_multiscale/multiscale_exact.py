"""Closed-form reference model for Example 4, multiscale GLE (numpy only).

The GLE of config.py has the exact Markovian embedding
Y = (X, Z, F_1, ..., F_nf) with one OU block per forcing component;
linear-Gaussian, so the same closed-form machinery as Example 1 applies.

Additional closed-form references, specific to the multiscale design:
  * ar_ideal_acf(M):   stationary ACF of the population AR(M) process =
    the IDEAL raw-window model of length M (Yule-Walker + extension);
  * bank_ideal_acf(rates): stationary ACF of the IDEAL bank chain.
Together they give the closed-loop error of the ideal window and ideal
bank at any conditioning dimension, BEFORE any training: the design
tables printed by the pre-vet.

Everything the example measures is derived from (A, B):

  * stationary covariance Sigma (Lyapunov equation),
  * exact discrete-time transition (A_d, Q_d) at step DT -> exact data,
  * ACF of the observed component C(m) = [A_d^m Sigma]_{00},
  * conditional of the next increment given (X_n, m_n) for the EMA bank
    actually used by the pipeline (stationary geometric sums, truncated
    at TAIL_TOL),
  * conditional given a finite history window; V_full is the value at
    the largest configured window, with the convergence of the window
    sweep reported alongside,
  * exact Markovian one-step projection (AR(1) with phi = C(1)/C(0)).

The transition, ACF, and projection are exact linear-Gaussian formulas;
the memory/full-history conditionals are numerically converged
references (geometric-tail and history-window truncations with reported
tolerances), not symbolic infinite-history quantities.

Run as a script for the PRE-VET report (method lesson: vet a new example
with the numpy ideal-model ACF test before any GPU stage):

    python gle1d_exact.py
"""

import numpy as np

import config as cfg
from memdiff.features import acf_estimate, rates_from_band


# ---------------------------------------------------------------- linear alg

def expm(M):
    """Matrix exponential by scaling-and-squaring Taylor (small dense M)."""
    n_sq = max(0, int(np.ceil(np.log2(max(1.0, np.linalg.norm(M, 1))))) + 4)
    Ms = M / (2 ** n_sq)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 20):
        term = term @ Ms / k
        E = E + term
    for _ in range(n_sq):
        E = E @ E
    return E


def lyapunov(A, Q):
    """Solve A S + S A^T + Q = 0 for S (row-major vectorization)."""
    n = A.shape[0]
    M = np.kron(A, np.eye(n)) + np.kron(np.eye(n), A)
    S = np.linalg.solve(M, -Q.flatten()).reshape(n, n)
    return 0.5 * (S + S.T)


# ---------------------------------------------------------------- the model

def build_model():
    nf = len(cfg.FORCING_TIMES)
    n = 2 + nf
    A = np.zeros((n, n))
    q = np.zeros(n)
    A[0, 0] = -cfg.GAMMA
    A[0, 1] = -cfg.C_K
    A[1, 0] = 1.0
    A[1, 1] = -cfg.LAM_K
    for j, (T, s) in enumerate(zip(cfg.FORCING_TIMES, cfg.FORCING_SIGS)):
        A[0, 2 + j] = 1.0
        A[2 + j, 2 + j] = -1.0 / T
        q[2 + j] = 2.0 * s ** 2 / T
    return A, np.diag(q)


def discrete_system(dt=None):
    """Exact one-step system at step dt: Y_{n+1} = A_d Y_n + N(0, Q_d)."""
    dt = cfg.DT if dt is None else dt
    A, BBt = build_model()
    Sigma = lyapunov(A, BBt)
    A_d = expm(A * dt)
    Q_d = Sigma - A_d @ Sigma @ A_d.T
    Q_d = 0.5 * (Q_d + Q_d.T)
    return A_d, Q_d, Sigma


def acf_x(n_lags, dt=None):
    """C(m) = Cov(X_n, X_{n+m}) = [A_d^m Sigma]_{00}, m = 0..n_lags."""
    A_d, _, Sigma = discrete_system(dt)
    s0 = Sigma[:, 0]
    r = np.zeros(A_d.shape[0]); r[0] = 1.0
    C = np.empty(n_lags + 1)
    C[0] = s0[0]
    for m in range(1, n_lags + 1):
        r = r @ A_d
        C[m] = r @ s0
    return C


def incr_acov(C):
    """R_D(h) = Cov(D_n, D_{n+h}) from C; D_n = X_n - X_{n-1}. h = 0..len-2."""
    L = len(C) - 2
    h = np.arange(L + 1)
    Cm1 = C[np.abs(h - 1)]
    return 2.0 * C[h] - Cm1 - C[h + 1]


# ------------------------------------------------- exact conditional laws

def tail_length(rates, dt):
    """Lags needed for the geometric feature sums to reach TAIL_TOL."""
    rho_max = np.exp(-min(rates) * dt)
    return int(np.ceil(np.log(cfg.TAIL_TOL) / np.log(rho_max)))


def _memory_joint(rates, dt=None, extra_lags=0):
    """Covariance assembly for v = (X_n, m_n) (stationary EMA bank of the
    note's Eqs. (7)-(8), geometric sums truncated at TAIL_TOL).

    Returns dict with C (ACF), RD (increment acov), rho, W (geometric
    weights), L (tail length), S (Cov of v), k.
    """
    dt = cfg.DT if dt is None else dt
    rates = np.asarray(rates, dtype=float)
    k = len(rates)
    L = tail_length(rates, dt) if k > 0 else 4
    C = acf_x(L + 3 + extra_lags, dt)
    RD = incr_acov(C)

    S = np.empty((1 + k, 1 + k))
    S[0, 0] = C[0]
    rho = W = None
    if k > 0:
        # truncation of the geometric sums is controlled by the WEIGHT
        # rho^L (= TAIL_TOL by construction), not by raw ACF decay; warn
        # only if the weighted tail contribution is non-negligible
        rho = np.exp(-rates * dt)        # (k,)
        tail = cfg.TAIL_TOL * abs(RD[L]) / RD[0] / (1.0 - rho.max())
        if tail > 1e-8:
            print(f"WARNING: weighted geometric tail {tail:.1e} at L={L}; "
                  f"tighten TAIL_TOL.")
        with np.errstate(all="ignore"):   # subnormal-heavy sums are benign
            l = np.arange(L + 1)          # geometric sum index
            W = rho[:, None] ** l[None, :]   # (k, L+1)
            # Cov(m_i, X_n) = sum_l rho_i^l (C(l) - C(l+1))
            S[0, 1:] = S[1:, 0] = W @ (C[:L + 1] - C[1:L + 2])
            # Cov(m_i, m_j) = sum_h R_D(|h|) * g_ij(h),
            # g_ij(h>=0) = rho_i^h/(1 - rho_i rho_j), g_ij(h<0) = rho_j^{-h}/(...)
            denom = 1.0 - rho[:, None] * rho[None, :]
            h = np.arange(1, L + 1)
            pos = (rho[:, None, None] ** h[None, None, :])       # (k,1,L)
            neg = (rho[None, :, None] ** h[None, None, :])       # (1,k,L)
            cross = np.tensordot(pos * np.ones((1, k, 1)), RD[1:L + 1],
                                 axes=([2], [0])) \
                + np.tensordot(neg * np.ones((k, 1, 1)), RD[1:L + 1],
                               axes=([2], [0]))
            S[1:, 1:] = (RD[0] + cross) / denom
    if not np.isfinite(S).all():
        raise FloatingPointError("non-finite covariance assembly in "
                                 f"_memory_joint(rates={rates})")
    return {"C": C, "RD": RD, "rho": rho, "W": W, "L": L, "S": S, "k": k}


def cond_given_memory(rates, dt=None):
    """Exact conditional of D_{n+1} = X_{n+1} - X_n given v = (X_n, m_n).

    Returns (beta, V) with E[D|v] = beta @ v and Var[D|v] = V.
    rates = [] gives the Markovian projection restricted to X_n.
    """
    J = _memory_joint(rates, dt)
    C, RD, W, L, k = J["C"], J["RD"], J["W"], J["L"], J["k"]
    c = np.empty(1 + k)                  # Cov(v, D_{n+1})
    c[0] = C[1] - C[0]
    if k > 0:
        with np.errstate(all="ignore"):
            # Cov(m_i, D_{n+1}) = sum_l rho_i^l R_D(l+1)
            c[1:] = W @ RD[1:L + 2]
    if not np.isfinite(c).all():
        raise FloatingPointError("non-finite covariance in cond_given_memory")
    beta = np.linalg.solve(J["S"], c)
    V = RD[0] - c @ beta
    return beta, float(V)


def cond_horizon_given_memory(rates, n_horizon, dt=None):
    """Exact conditional of X_{n+h} given v = (X_n, m_n), h = 1..n_horizon.

    Returns (A, Vh): E[X_{n+h}|v] = A[h-1] @ v, Var[X_{n+h}|v] = Vh[h-1].
    The exact reference for the conditional-ensemble ('particle release')
    figure. rates = [] gives the conditional given X_n alone.
    """
    J = _memory_joint(rates, dt, extra_lags=n_horizon)
    C, W, L, k = J["C"], J["W"], J["L"], J["k"]
    A = np.empty((n_horizon, 1 + k))
    Vh = np.empty(n_horizon)
    for h in range(1, n_horizon + 1):
        cov = np.empty(1 + k)
        cov[0] = C[h]                    # Cov(X_{n+h}, X_n)
        if k > 0:
            with np.errstate(all="ignore"):
                # Cov(X_{n+h}, m_i) = sum_l rho_i^l (C(h+l) - C(h+l+1))
                cov[1:] = W @ (C[h:h + L + 1] - C[h + 1:h + L + 2])
        a = np.linalg.solve(J["S"], cov)
        A[h - 1] = a
        Vh[h - 1] = C[0] - cov @ a
    if not (np.isfinite(A).all() and np.isfinite(Vh).all()):
        raise FloatingPointError("non-finite horizon conditional")
    return A, Vh


def cond_given_window(M, n_lags=None):
    """V_hist(M) = Var(X_{n+1} | X_n, ..., X_{n-M+1}) via Toeplitz solve."""
    if n_lags is None:
        n_lags = M + 1
    C = acf_x(max(n_lags, M + 1))
    idx = np.arange(M)
    Gamma = C[np.abs(idx[:, None] - idx[None, :])]
    cvec = C[1:M + 1]
    coef = np.linalg.solve(Gamma, cvec)
    return float(C[0] - cvec @ coef)


def markov_projection():
    """Exact one-step projection p(X_{n+1}|X_n): AR(1) phi, innovation var."""
    C = acf_x(2)
    phi = C[1] / C[0]
    return float(phi), float(C[0] - C[1] ** 2 / C[0]), float(C[0])


def v_full(windows=None):
    """Full-history predictive variance: numerically converged reference,
    defined as V_hist at the largest configured window."""
    windows = cfg.HIST_WINDOWS if windows is None else windows
    vs = np.array([cond_given_window(M) for M in windows])
    tail = abs(vs[-1] - vs[-2]) / vs[-1]
    # a residual sweep change of ~1e-8 shifts eps_k by < 1e-5, negligible
    # against PASS_EPS_K; warn only above that level
    if tail > 5e-8:
        print(f"WARNING: V_hist window sweep not converged "
              f"(last relative change {tail:.1e}); enlarge HIST_WINDOWS.")
    return vs, float(vs[-1])


def sufficiency_curve(rates_for_k, ks=None):
    """epsilon_k = (V_mem(k) - V_full) / (V_markov - V_full) in [0, 1].

    rates_for_k: callable k -> rates array (the pipeline's log-spaced rates).
    """
    ks = cfg.K_SWEEP if ks is None else ks
    _, Vinf = v_full()
    Vmk = cond_given_window(1)
    out = []
    for k in ks:
        rates = rates_for_k(k)
        _, V = cond_given_memory(rates)
        eps = (V - Vinf) / (Vmk - Vinf)
        out.append((k, V, eps))
    return out, Vinf, Vmk


# ------------------------------------------------- ideal-model rollout

def ideal_rollout(rates, n_traj, n_steps, seed=0, dt=None):
    """Roll out the EXACT conditional given (X_n, m_n): the ideal memory
    model with zero estimation/distillation error. Its ACF vs the exact C
    isolates the memory-sufficiency error alone (pre-vet ACF test)."""
    dt = cfg.DT if dt is None else dt
    rng = np.random.default_rng(seed)
    beta, V = cond_given_memory(rates, dt)
    rho = np.exp(-np.asarray(rates) * dt)
    k = len(rho)
    C0 = acf_x(1)[0]
    x = rng.normal(0.0, np.sqrt(C0), n_traj)
    m = np.zeros((n_traj, k))
    burn = int(np.ceil(np.log(cfg.BURN_TOL) / np.log(rho.max()))) if k else 0
    X = np.empty((n_traj, n_steps))
    for n in range(burn + n_steps):
        v = np.concatenate([x[:, None], m], axis=1)
        d = v @ beta + np.sqrt(V) * rng.standard_normal(n_traj)
        x = x + d
        if k:
            m = rho[None, :] * m + d[:, None]
        if n >= burn:
            X[:, n - burn] = x
    return X


# ---------------------------------------- closed-loop ideal references

def ar_ideal_acf(M, n_out, C=None):
    """Stationary ACF of the population AR(M) process: the IDEAL raw
    last-M-window model. Matches C at lags <= M, extrapolates beyond."""
    if C is None:
        C = acf_x(max(M + 1, n_out))
    if M >= n_out:            # the window covers the whole horizon:
        return C[:n_out + 1].copy()   # AR(M) matches C at lags <= M
    idx = np.arange(M)
    G = C[np.abs(idx[:, None] - idx[None, :])]
    a = np.linalg.solve(G, C[1:M + 1])
    Ce = np.empty(n_out + 1)
    Ce[:M + 1] = C[:M + 1]
    for m in range(M + 1, n_out + 1):
        Ce[m] = a @ Ce[m - 1:m - M - 1:-1]
    return Ce


def bank_ideal_acf(rates, n_out):
    """Stationary ACF of the IDEAL bank chain x' = x + beta.(x, m) + e:
    a linear-Gaussian AR(1) on (x, m), closed form."""
    rates = np.asarray(rates, float)
    k = len(rates)
    rho = np.exp(-rates * cfg.DT)
    beta, V = cond_given_memory(rates)
    T = np.zeros((1 + k, 1 + k))
    B = np.ones(1 + k)
    T[0] = beta
    T[0, 0] += 1.0
    for i in range(k):
        T[1 + i] = beta
        T[1 + i, 1 + i] += rho[i]
    Qc = V * np.outer(B, B)
    P = np.zeros_like(Qc)
    for _ in range(500000):
        Pn = T @ P @ T.T + Qc
        if np.max(np.abs(Pn - P)) < 1e-14 * max(1.0, np.max(np.abs(Pn))):
            P = Pn
            break
        P = Pn
    Ce = np.empty(n_out + 1)
    Ce[0] = P[0, 0]
    v = P[:, 0].copy()
    for m in range(1, n_out + 1):
        v = T @ v
        Ce[m] = v[0]
    return Ce


# --------------------------------- exact full-history forecast (Kalman)

def kalman_filter_exact(X_hist):
    """Exact posterior of the embedding Y_n given an observed history.

    X_hist: (n_q, W+1) observed states; the filter conditions on ALL of
    them exactly (the observation is the X component itself, noiseless),
    initialized from the stationary law conditioned on the first value.
    Returns (mu, P): mu (n_q, s) posterior means of Y at the last
    history index, and P (s, s) the posterior covariance -- which is
    data independent, hence shared across queries.
    """
    A_d, Q_d, Sigma = discrete_system()
    X_hist = np.atleast_2d(np.asarray(X_hist, float))
    # initialize: stationary prior conditioned on X_0 (exact observation)
    mu = np.outer(X_hist[:, 0], Sigma[:, 0] / Sigma[0, 0])
    P = Sigma - np.outer(Sigma[:, 0], Sigma[0, :]) / Sigma[0, 0]
    for n in range(1, X_hist.shape[1]):
        mu = mu @ A_d.T
        P = A_d @ P @ A_d.T + Q_d
        # noiseless update on the X component; the innovation variance
        # P[0, 0] = Var(X-step innovation) > 0 because Q_d[0, 0] > 0
        K = P[:, 0] / P[0, 0]
        mu = mu + (X_hist[:, n] - mu[:, 0])[:, None] * K[None, :]
        P = P - np.outer(K, P[0, :])
        P = 0.5 * (P + P.T)
    if not (np.isfinite(mu).all() and np.isfinite(P).all()):
        raise FloatingPointError("non-finite Kalman filter state")
    return mu, P


def kalman_forecast(X_hist, n_horizon):
    """Exact conditional of X_{n+h} given the FULL observed history,
    h = 1..n_horizon: the fair forecast reference for ANY model, since
    it conditions on exactly the information every model receives.

    Returns (mean, sd): mean (n_q, H) per query; sd (H,) shared across
    queries (the forecast variance of a linear-Gaussian filter does not
    depend on the data).
    """
    A_d, Q_d, _ = discrete_system()
    mu, P = kalman_filter_exact(X_hist)
    means = np.empty((mu.shape[0], n_horizon))
    var = np.empty(n_horizon)
    for h in range(n_horizon):
        mu = mu @ A_d.T
        P = A_d @ P @ A_d.T + Q_d
        P = 0.5 * (P + P.T)
        means[:, h] = mu[:, 0]
        var[h] = P[0, 0]
    if not (np.isfinite(means).all() and np.all(var > 0)):
        raise FloatingPointError("non-finite Kalman forecast")
    return means, np.sqrt(var)


# ------------------------------------------------- pre-vet report

def _prevet():
    from conditioning import select_rates_by_prediction
    from generate_data import simulate

    A, _ = build_model()
    eig = np.linalg.eigvals(A)
    print("=" * 72)
    print("Example 4 pre-vet: multiscale GLE (closed form / numpy)")
    print("=" * 72)
    print(f"embedding dim {A.shape[0]}; eigenvalues re < 0: "
          f"{bool(np.all(eig.real < 0))}")
    if not np.all(eig.real < 0):
        raise ValueError("UNSTABLE parameters -- fix config first")

    n_show = int(5.0 * max(cfg.FORCING_TIMES) / cfg.DT)
    C = acf_x(n_show)
    print(f"stationary Var(X) = {C[0]:.4f}; forcing times "
          f"{cfg.FORCING_TIMES} (= "
          f"{[float(f'{t / cfg.DT:g}') for t in cfg.FORCING_TIMES]} steps)")

    vs, Vinf = v_full()
    Vmk = cond_given_window(1)
    red = Vmk - Vinf
    print(f"V_markov = {Vmk:.6f}  V_full = {Vinf:.6f}  "
          f"(reducible {red / Vmk:.1%}; window sweep conv "
          f"{np.abs(vs[-1] - vs[-2]) / vs[-1]:.1e})")
    prof = [(M, (cond_given_window(M) - Vinf) / red)
            for M in (2, 4, 8, 16, 64, 256)]
    print("ONE-STEP window residual (the trap -- looks sufficient): "
          + "  ".join(f"M{M}={e:.3f}" for M, e in prof))

    print("\nDESIGN TABLE: closed-loop ACF error of the IDEAL models "
          f"(max over tau <= {n_show * cfg.DT:.0f}):")
    for M in (9, 17, 33, 65):     # AR(M) conditions on M states = dim M
        Ce = ar_ideal_acf(M, n_show, C=acf_x(max(n_show, M + 1)))
        err = np.max(np.abs(Ce - C)) / C[0]
        print(f"  ideal WINDOW AR({M:3d})  (dim {M:3d}): {err:.3f}")
    lam_lo_design = 1.0 / max(cfg.FORCING_TIMES)
    for k in (4, 6, 8):
        rates = rates_from_band((lam_lo_design, 3.0 / cfg.DT), k)
        err = np.max(np.abs(bank_ideal_acf(rates, n_show) - C)) / C[0]
        print(f"  ideal BANK   k={k:3d}  (dim {k + 1:3d}): {err:.3f}")

    # stage-1 selection on synthetic data through the pipeline code path
    print(f"\nstage-1 selection on {cfg.N_TRAJ_SELECT} exact synthetic "
          f"trajectories:")
    rng = np.random.default_rng(cfg.SEED_DATA + 7)
    Xs = simulate(cfg.N_TRAJ_SELECT, cfg.L_TRAJ, rng)
    sel = select_rates_by_prediction(Xs)
    band, k, rates = sel["band"], sel["k"], sel["rates"]

    _, Vsel = cond_given_memory(rates)
    eps_sel = (Vsel - Vinf) / red
    print(f"  exact V_mem at selection = {Vsel:.6f} "
          f"(holdout residual {sel['resid_chosen']:.6f}) -> eps = {eps_sel:.4f}")

    e_bank = np.max(np.abs(bank_ideal_acf(rates, n_show) - C)) / C[0]
    e_wind = np.max(np.abs(ar_ideal_acf(k + 1, n_show,
                                        C=acf_x(max(n_show, k + 2))) - C)) / C[0]
    gap = e_wind / e_bank
    print(f"  at the SELECTED k = {k} (dim {k + 1}): ideal bank ACF err "
          f"{e_bank:.3f}, ideal window AR({k + 1}) at the same dimension "
          f"{e_wind:.3f}  -> design gap {gap:.1f}x")

    X = ideal_rollout(rates, n_traj=50, n_steps=cfg.L_GEN, seed=0)
    acf_gen = acf_estimate(X, n_show)
    err_meas = np.max(np.abs(acf_gen - C)) / C[0]
    print(f"  ideal-model MEASURED ACF test: {err_meas:.4f} "
          f"(closed-loop closed form {e_bank:.4f}; the difference is the "
          f"finite-sample ACF noise floor)")

    go = (eps_sel < cfg.PASS_EPS_K and e_bank < cfg.PASS_ACF_IDEAL
          and gap > cfg.PASS_GAP)
    print(f"\nGO criteria: eps < {cfg.PASS_EPS_K} ({eps_sel:.4f}), "
          f"ideal bank err < {cfg.PASS_ACF_IDEAL} ({e_bank:.3f}), "
          f"design gap > {cfg.PASS_GAP} ({gap:.1f})  ->  "
          f"{'GO' if go else 'NO-GO'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        from memdiff.plotting import setup_style, MODEL_ALPHA
        setup_style()
        os.makedirs(cfg.FIG_DIR, exist_ok=True)
        tau = np.arange(n_show + 1) * cfg.DT
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].semilogx(tau[1:], (C / C[0])[1:], "k-", lw=2, label="exact")
        ax[0].semilogx(tau[1:], (bank_ideal_acf(rates, n_show) / C[0])[1:],
                       "-", color="tab:red", alpha=MODEL_ALPHA,
                       label=f"ideal bank $k={k}$")
        ax[0].semilogx(tau[1:], (ar_ideal_acf(k + 1, n_show,
                       C=acf_x(max(n_show, k + 2))) / C[0])[1:],
                       "-", color="tab:orange", alpha=MODEL_ALPHA,
                       label=f"ideal window, same dim (AR({k + 1}))")
        ax[0].set(xlabel=r"lag time $\tau$", ylabel="normalized ACF",
                  title="ideal models, closed form")
        ax[0].legend()
        dims_w, errs_w = [], []
        for M in (5, 9, 17, 33, 65, 129):
            Ce = ar_ideal_acf(M, n_show, C=acf_x(max(n_show, M + 1)))
            dims_w.append(M)
            errs_w.append(np.max(np.abs(Ce - C)) / C[0])
        dims_b, errs_b = [], []
        for kk in (2, 4, 6, 8, 10):
            rr = rates_from_band((lam_lo_design, 3.0 / cfg.DT), kk)
            dims_b.append(kk + 1)
            errs_b.append(np.max(np.abs(bank_ideal_acf(rr, n_show) - C))
                          / C[0])
        ax[1].loglog(dims_w, errs_w, "o-", color="tab:orange",
                     label="raw window")
        ax[1].loglog(dims_b, errs_b, "s-", color="tab:red", label="EMA bank")
        ax[1].set(xlabel="conditioning dimension",
                  ylabel="closed-loop ACF error (ideal)",
                  title="memory representation vs dimension")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f"{cfg.FIG_DIR}/ex4_prevet.pdf")
        print(f"figure -> {cfg.FIG_DIR}/ex4_prevet.pdf")
    except ImportError:
        print("(matplotlib not available; tables above are the report)")
    return go


if __name__ == "__main__":
    import sys
    sys.exit(0 if _prevet() else 1)
