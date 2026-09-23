"""Example-1 conditioning: tuples and data-driven selection (numpy only).

Example-4 adaptation: the SLOW END of the stage-A band grid comes from
the significance range of the STATE autocovariance rather than the
increment autocovariance. Multi-decade modes are invisible in the
increments (amplitude ~ (DT/T)^2) but plain in the state ACF; stage B's
closed-loop criterion then decides what is kept.

The generic primitives (ACF estimator, rate band, EMA bank, log-spaced
rates) live in memdiff.features; this module keeps what is specific to
the 1D general branch, c_n = (x_n, m_n): the regrouping of trajectories
into conditioning tuples and the two-stage (band, k) selection of
stage 1 of Algorithm 1.
"""

import numpy as np

import config as cfg
from memdiff.features import (acf_estimate, rates_from_band,
                             ema_features as _ema_features,
                             select_rate_band as _select_rate_band)


def select_rate_band(incr_acf_norm, dt, thresh=None):
    return _select_rate_band(incr_acf_norm, dt,
                             cfg.ACF_SIG_THRESH if thresh is None else thresh)


def ema_features(X, rates, dt=None):
    """Common EMA bank with this example's DT and burn-in tolerance."""
    return _ema_features(X, rates, cfg.DT if dt is None else dt,
                         cfg.BURN_TOL)


# ---------------------------------------------- data-driven selection

def select_rates_by_prediction(X, dt=None, verbose=True):
    """Stage 1 of Algorithm 1, data-driven: choose (band, k).

    Stage A: for each k <= K_MAX, shortlist the band minimizing the
    HELD-OUT one-step linear (OLS) prediction error of the next
    increment given (x_n, m_n), over a grid of log-spaced bands snapped to
    a rate dictionary. The OLS residual is a model-free proxy for the
    memory-sufficiency error (for linear-Gaussian processes it estimates
    V_mem exactly, which the Example-1 pre-vet cross-checks).

    Stage B: one-step residuals under-select k -- closed-loop error
    concentrates on slow modes -- so k is chosen CLOSED-LOOP: refit the
    linear surrogate at the exact log-spaced rates, roll it out with its
    fitted noise, and pick the smallest k whose rollout ACF matches the
    data ACF within SEL_ACF_TOL (fallback: the k minimizing that
    error). This is the data-driven form of the ideal-model ACF test;
    for linear-Gaussian processes the two coincide.

    Returns dict with band, k, rates (exact log-spaced rates), the per-k
    table (residual, band, closed-loop ACF error), resid_markov,
    resid_chosen, acf_err_chosen, n_burn.
    """
    dt = cfg.DT if dt is None else dt
    X = np.asarray(X, float)[:cfg.N_TRAJ_SELECT]

    # slow end of the grid from the STATE-ACF significant range
    D = np.diff(X, axis=1)
    max_lag = min(cfg.ACF_MAX_LAG, X.shape[1] // 4)
    rs = acf_estimate(X, max_lag)
    lam_lo, _ = select_rate_band(rs / rs[0], dt, thresh=cfg.ACF_SIG_THRESH)
    lam_hi = max(cfg.LMAX_FACTORS) / dt
    lam_dict = np.geomspace(lam_lo, lam_hi, cfg.N_RATE_DICT)

    m, _ = ema_features(X, lam_dict, dt)
    # rho_lo^n < BURN_TOL  <=>  n > -log(BURN_TOL) / (lam_lo dt);
    # cap so a very slow lam_lo cannot consume the selection data
    n_burn = int(np.ceil(-np.log(cfg.BURN_TOL) / (lam_lo * dt)))
    n_burn = min(n_burn, X.shape[1] // 5)
    x_n = X[:, n_burn:-1]
    y = D[:, n_burn:]
    mm = m[:, n_burn:-1]

    n_fit = max(1, int(0.6 * X.shape[0]))

    def flat(a, sl):
        return a[sl].reshape(-1, a.shape[-1]) if a.ndim == 3 \
            else a[sl].reshape(-1, 1)

    # Every band/k candidate is a column subset of [1, x, dictionary],
    # so the normal equations are accumulated ONCE per split; each
    # candidate then costs a small dense solve instead of a fresh
    # full-size least-squares. Columns are standardized (fit-split
    # stats) first — same holdout residual algebraically, but without
    # it the squared condition number of the correlated EMA dictionary
    # can reorder near-tied candidates.
    _mm_fit = flat(mm, slice(None, n_fit))
    _x_fit = flat(x_n, slice(None, n_fit))
    mu_c = np.concatenate([_x_fit.mean(0), _mm_fit.mean(0)])
    sd_c = np.maximum(np.concatenate([_x_fit.std(0), _mm_fit.std(0)]),
                      1e-300)

    def gram(sl):
        C_p = np.hstack([flat(x_n, sl), flat(mm, sl)])
        C_p = (C_p - mu_c) / sd_c
        y_p = y[sl].reshape(-1)
        A = np.hstack([np.ones((C_p.shape[0], 1)), C_p])
        with np.errstate(all="ignore"):   # subnormal BLAS noise is benign
            G = A.T @ A
            b = A.T @ y_p
        return G, b, float(y_p @ y_p), len(y_p)

    G_f, b_f, _, _ = gram(slice(None, n_fit))
    G_h, b_h, yy_h, n_h = gram(slice(n_fit, None))

    def resid(cols):
        idx = np.concatenate([[0, 1], 2 + np.asarray(cols, int)]) \
            if len(cols) else np.array([0, 1])
        with np.errstate(all="ignore"):
            coef, *_ = np.linalg.lstsq(G_f[np.ix_(idx, idx)], b_f[idx],
                                       rcond=None)
            r = (yy_h - 2.0 * float(coef @ b_h[idx])
                 + float(coef @ (G_h[np.ix_(idx, idx)] @ coef))) / n_h
        if not np.isfinite(r):
            raise FloatingPointError("non-finite OLS residual in selection")
        return float(r)

    # ---------------- stage A: band shortlist per k (one-step residual)
    r_markov = resid([])
    lmin_grid = np.geomspace(lam_lo, 1.0 / dt, 8)
    lmax_grid = [f / dt for f in cfg.LMAX_FACTORS]
    log_dict = np.log(lam_dict)

    # Example-4 amendment: one-step residuals are nearly FLAT across bands
    # when the predictive memory is slow (the one-step trap applies to the
    # selection itself), so stage A forwards ALL bands whose held-out
    # residual is within TIE_TOL of the best to stage B, which decides by
    # the closed-loop criterion.
    cand = {}      # k -> [(resid, band), ...] near-tied shortlist
    for k in range(1, cfg.K_MAX + 1):
        rows = []
        for lmin in lmin_grid:
            for lmax in lmax_grid:
                if lmin >= lmax:
                    continue
                rates = rates_from_band((lmin, lmax), k)
                cols = np.unique(np.argmin(
                    np.abs(np.log(rates)[:, None] - log_dict[None, :]),
                    axis=1))
                if len(cols) < k:      # dictionary collision: skip
                    continue
                rows.append((resid(list(cols)), (float(lmin), float(lmax))))
        rows.sort(key=lambda z: z[0])
        cand[k] = ([z for z in rows if z[0] <= rows[0][0] * (1 + cfg.TIE_TOL)]
                   if rows else [])

    # ---------------- stage B: closed-loop selection of k
    sel_lag = min(cfg.ACF_MAX_LAG, X.shape[1] // 10)
    acf_ref = acf_estimate(X, sel_lag)
    x_std = X.std()

    def ols_exact(rates):
        """Refit at the exact log-spaced rates; coef for [1, x, m]."""
        mk, _ = ema_features(X, rates, dt)
        mmk = mk[:, n_burn:-1]
        def design(sl):
            xf = flat(x_n, sl)
            return np.hstack([np.ones((xf.shape[0], 1)), xf, flat(mmk, sl)])
        with np.errstate(all="ignore"):
            coef, *_ = np.linalg.lstsq(design(slice(None, n_fit)),
                                       y[:n_fit].reshape(-1), rcond=None)
            r = float(np.mean((y[n_fit:].reshape(-1)
                               - design(slice(n_fit, None)) @ coef) ** 2))
        return coef, r

    def surrogate_acf_err(rates, coef, sig, seed=0):
        """Roll out the fitted linear surrogate; max ACF error vs data."""
        rng = np.random.default_rng(seed)
        rho = np.exp(-np.asarray(rates) * dt)
        n_traj, n_steps = 25, 20000
        burn = min(int(np.ceil(-np.log(cfg.BURN_TOL) / (min(rates) * dt))),
                   2000)
        x = rng.normal(0.0, x_std, n_traj)
        mloc = np.zeros((n_traj, len(rho)))
        Xg = np.empty((n_traj, n_steps))
        for n in range(burn + n_steps):
            v = np.concatenate([np.ones((n_traj, 1)), x[:, None], mloc],
                               axis=1)
            d = v @ coef + sig * rng.standard_normal(n_traj)
            x = x + d
            mloc = rho[None, :] * mloc + d[:, None]
            if n % 500 == 0 and np.abs(x).max() > 1e6:
                return np.inf                      # unstable fit
            if n >= burn:
                Xg[:, n - burn] = x
        if not np.isfinite(Xg).all():
            return np.inf
        acf_g = acf_estimate(Xg, sel_lag)
        return float(np.max(np.abs(acf_g - acf_ref)) / acf_ref[0])

    table = {}     # k -> [resid, band, acf_err] of the closed-loop winner
    k_sel, chosen = None, None
    for k in sorted(cand):
        if not cand[k]:
            table[k] = [np.inf, None, None]
            continue
        best = [np.inf, None, np.inf]          # [resid, band, err]
        for r_ols, band in cand[k]:
            rates = rates_from_band(band, k)
            coef, r_exact = ols_exact(rates)
            err = surrogate_acf_err(rates, coef, np.sqrt(r_exact))
            if err < best[2]:
                best = [r_exact, band, err]
        table[k] = best
        if verbose:
            print(f"    k={k:2d}: {len(cand[k])} tied bands -> "
                  f"band [{best[1][0]:.4f}, {best[1][1]:.4f}]  "
                  f"acfErr {best[2]:.4f}")
        if best[2] <= cfg.SEL_ACF_TOL:
            k_sel = k
            chosen = (rates_from_band(best[1], k), best[0], best[2])
            break
    if k_sel is None:      # fallback: best closed-loop error among tried
        tried = {k: v for k, v in table.items() if v[2] is not None}
        k_sel = min(tried, key=lambda k: tried[k][2])
        rates = rates_from_band(table[k_sel][1], k_sel)
        chosen = (rates, table[k_sel][0], table[k_sel][2])
    rates_sel, r_sel, err_sel = chosen
    band_sel = table[k_sel][1]

    if verbose:
        print(f"  selection: markov residual {r_markov:.6f}")
        for k in sorted(table):
            r, b, e = table[k]
            if b is None:
                print(f"    k={k:2d}: (no valid candidate)")
                continue
            estr = f"  acfErr {e:.4f}" if e is not None else ""
            mark = " <-- selected" if k == k_sel else ""
            print(f"    k={k:2d}: resid {r:.6f}  "
                  f"band [{b[0]:.3f}, {b[1]:.3f}]{estr}{mark}")
        print(f"  chosen: k={k_sel}, band "
              f"[{band_sel[0]:.4f}, {band_sel[1]:.4f}], "
              f"rates {np.round(rates_sel, 3)}, "
              f"closed-loop ACF err {err_sel:.4f}")

    return {"band": band_sel, "k": k_sel, "rates": rates_sel,
            "resid_markov": r_markov, "resid_chosen": r_sel,
            "acf_err_chosen": err_sel,
            "table": {k: {"resid": v[0], "band": v[1], "acf_err": v[2]}
                      for k, v in table.items()},
            "n_burn": n_burn}


# ------------------------------------------------------------- tuples

def build_window_tuples(X, n_lags, kappa_s=None):
    """Raw last-k history conditioning: the brute-force memory baseline.

    c_n = (x_n, x_{n-1}, ..., x_{n-n_lags}), so with n_lags = k the
    conditioning has the SAME dimension 1 + k as the EMA bank of size k.
    Any difference between the two variants is therefore attributable to
    the representation of the history, not to the dimension the score
    estimator has to work in.

    Same layout and same kappa_s convention as build_tuples.
    """
    X = np.asarray(X, float)
    D = np.diff(X, axis=1)
    if kappa_s is None:
        kappa_s = 1.0 / D.std()
    n0, L1 = n_lags, X.shape[1]
    cols = [X[:, n0 - l:L1 - 1 - l] for l in range(n_lags + 1)]
    C = np.stack(cols, axis=2).reshape(-1, n_lags + 1)
    S = (D[:, n0:] * kappa_s).reshape(-1, 1)
    return {
        "C": C, "S": S, "X0": X[:, n0:-1].reshape(-1),
        "kappa_s": float(kappa_s),
        "rates": np.arange(1, n_lags + 1, dtype=float),   # retained lags
        "n_burn": n0,
    }




def build_tuples(X, rates, dt=None, kappa_s=None):
    """Regroup trajectories into conditioning tuples (paper, Section 3.2).

    Returns dict with
      C  : (N, 1+k) conditions c_n = (x_n, m_n)        [k = 0: just x_n]
      S  : (N, 1)   scaled displacements s_n = (x_{n+1} - x_n) kappa_s
      X0 : (N,)     x_n (to reconstruct states from displacements)
      kappa_s, rates, n_burn
    """
    dt = cfg.DT if dt is None else dt
    m, n_burn = ema_features(X, rates, dt)
    D = np.diff(X, axis=1)                     # (n_traj, L)
    if kappa_s is None:
        kappa_s = 1.0 / D.std()
    n0 = n_burn                                 # first usable n
    x_n = X[:, n0:-1]
    m_n = m[:, n0:-1]
    d_n = D[:, n0:]
    C = np.concatenate([x_n[..., None], m_n], axis=2).reshape(-1, 1 + len(rates))
    S = (d_n * kappa_s).reshape(-1, 1)
    return {
        "C": C, "S": S, "X0": x_n.reshape(-1),
        "kappa_s": float(kappa_s), "rates": np.asarray(rates, float),
        "n_burn": n_burn,
    }
