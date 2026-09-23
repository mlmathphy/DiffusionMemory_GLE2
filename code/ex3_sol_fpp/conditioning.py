"""Example-3 conditioning: data-driven bank selection and tuples (numpy).

The selection is the paper's two-stage rule as implemented for Example 1
(ex1_multiscale/conditioning.py): stage A shortlists rate bands per k by
the held-out one-step OLS residual (forwarding near-ties, because the
one-step trap flattens the residuals), stage B chooses the smallest k
whose linear closed-loop surrogate reproduces the data ACF within
SEL_ACF_TOL. The slow end of the grid comes from the STATE-ACF
significance range. Then the predictive-rank reduction of App. A.3
(data-estimated, mean + second-moment targets) gives the coordinates
z = P (m - mu_m - Gam (y - mu_y)) deployed as the conditioning
c_n = (y_n, z_n) when REDUCE is on (else c_n = (y_n, m_n)).
"""

import numpy as np

import config as cfg
from memdiff.features import (acf_estimate, rates_from_band,
                             ema_features as _ema_features,
                             select_rate_band as _select_rate_band)


def ema_features(X, rates, dt=None):
    return _ema_features(X, rates, cfg.DT if dt is None else dt,
                         cfg.BURN_TOL)


# ---------------------------------------------- data-driven selection

def select_rates_by_prediction(X, dt=None, verbose=True):
    """Stage 1 of Algorithm 1 (band, k); see module docstring."""
    dt = cfg.DT if dt is None else dt
    X = np.asarray(X, float)[:cfg.N_TRAJ_SELECT]
    D = np.diff(X, axis=1)
    max_lag = min(cfg.ACF_MAX_LAG, X.shape[1] // 4)
    rs = acf_estimate(X, max_lag)
    lam_lo, _ = _select_rate_band(rs / rs[0], dt, cfg.ACF_SIG_THRESH)
    lam_hi = max(cfg.LMAX_FACTORS) / dt
    lam_dict = np.geomspace(lam_lo, lam_hi, cfg.N_RATE_DICT)

    m, _ = ema_features(X, lam_dict, dt)
    n_burn = int(np.ceil(-np.log(cfg.BURN_TOL) / (lam_lo * dt)))
    n_burn = min(n_burn, X.shape[1] // 5)
    x_n = X[:, n_burn:-1]
    y = D[:, n_burn:]
    mm = m[:, n_burn:-1]
    n_fit = max(1, int(0.6 * X.shape[0]))

    def flat(a, sl):
        return a[sl].reshape(-1, a.shape[-1]) if a.ndim == 3 \
            else a[sl].reshape(-1, 1)

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
        with np.errstate(all="ignore"):
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

    r_markov = resid([])
    lmin_grid = np.geomspace(lam_lo, 1.0 / dt, 8)
    lmax_grid = [f / dt for f in cfg.LMAX_FACTORS]
    log_dict = np.log(lam_dict)
    cand = {}
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
                if len(cols) < k:
                    continue
                rows.append((resid(list(cols)), (float(lmin), float(lmax))))
        rows.sort(key=lambda z: z[0])
        cand[k] = ([z for z in rows if z[0] <= rows[0][0] * (1 + cfg.TIE_TOL)]
                   if rows else [])

    sel_lag = min(cfg.ACF_MAX_LAG, X.shape[1] // 10)
    acf_ref = acf_estimate(X, sel_lag)
    x_std = X.std()

    def ols_exact(rates):
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
        rng = np.random.default_rng(seed)
        rho = np.exp(-np.asarray(rates) * dt)
        n_traj, n_steps = 25, 20000
        burn = min(int(np.ceil(-np.log(cfg.BURN_TOL) / (min(rates) * dt))),
                   2000)
        x = rng.normal(0.0, x_std, n_traj)
        mloc = np.zeros((n_traj, len(rho)))
        Xg = np.empty((n_traj, n_steps))
        for n in range(burn + n_steps):
            v = np.concatenate([np.ones((n_traj, 1)), x[:, None], mloc], axis=1)
            d = v @ coef + sig * rng.standard_normal(n_traj)
            x = x + d
            mloc = rho[None, :] * mloc + d[:, None]
            if n % 500 == 0 and np.abs(x).max() > 1e6:
                return np.inf
            if n >= burn:
                Xg[:, n - burn] = x
        if not np.isfinite(Xg).all():
            return np.inf
        acf_g = acf_estimate(Xg, sel_lag)
        return float(np.max(np.abs(acf_g - acf_ref)) / acf_ref[0])

    table = {}
    k_sel, chosen = None, None
    for k in sorted(cand):
        if not cand[k]:
            table[k] = [np.inf, None, None]
            continue
        best = [np.inf, None, np.inf]
        for r_ols, band in cand[k]:
            rates = rates_from_band(band, k)
            coef, r_exact = ols_exact(rates)
            err = surrogate_acf_err(rates, coef, np.sqrt(r_exact))
            if err < best[2]:
                best = [r_exact, band, err]
        table[k] = best
        if verbose:
            print(f"    k={k:2d}: {len(cand[k])} tied bands -> band "
                  f"[{best[1][0]:.4f}, {best[1][1]:.4f}]  acfErr {best[2]:.4f}")
        if best[2] <= cfg.SEL_ACF_TOL:
            k_sel = k
            chosen = (rates_from_band(best[1], k), best[0], best[2])
            break
    if k_sel is None:
        tried = {k: v for k, v in table.items() if v[2] is not None}
        k_sel = min(tried, key=lambda k: tried[k][2])
        rates = rates_from_band(table[k_sel][1], k_sel)
        chosen = (rates, table[k_sel][0], table[k_sel][2])
    rates_sel, r_sel, err_sel = chosen
    band_sel = table[k_sel][1]
    if verbose:
        print(f"  selection: markov residual {r_markov:.6f}; chosen k={k_sel}, "
              f"band [{band_sel[0]:.4f}, {band_sel[1]:.4f}], rates "
              f"{np.round(rates_sel, 3)}, closed-loop ACF err {err_sel:.4f}")
    return {"band": band_sel, "k": k_sel, "rates": rates_sel,
            "resid_markov": r_markov, "resid_chosen": r_sel,
            "acf_err_chosen": err_sel,
            "table": {k: {"resid": v[0], "band": v[1], "acf_err": v[2]}
                      for k, v in table.items()},
            "n_burn": n_burn}


# ------------------------------------------ predictive-rank reduction

def predictive_rank(x, M, FUT, energy_rule):
    """Data-estimated predictive spectrum of the bank (App. A.3): residualize
    m on y, whiten, cross-covariances with mean and second-moment targets of
    the future increments over the horizons, SVD. Returns the projection
    dict used by `project`."""
    x = np.asarray(x, float).reshape(-1, 1)
    mu_x, mu_m = x.mean(0), M.mean(0)
    with np.errstate(all="ignore"):        # subnormal BLAS flags are benign
        Gam, *_ = np.linalg.lstsq(x - mu_x, M - mu_m, rcond=None)
        Mt = (M - mu_m) - (x - mu_x) @ Gam
        cov_m = np.atleast_2d(np.cov(Mt.T))      # k = 1 gives a scalar cov
        ev, U = np.linalg.eigh(cov_m)
        ev = np.maximum(ev, 1e-12 * ev.max())
        Wm = U @ np.diag(ev ** -0.5) @ U.T
        Uw = Mt @ Wm
        blocks = []
        for h in range(FUT.shape[1]):
            d = FUT[:, h]
            fut = np.stack([d, d ** 2], 1)
            B, *_ = np.linalg.lstsq(x - mu_x, fut - fut.mean(0), rcond=None)
            fr = fut - fut.mean(0) - (x - mu_x) @ B
            fr = fr / np.maximum(fr.std(0), 1e-12)
            blocks.append(Uw.T @ fr / len(fr))
        Gs = np.hstack(blocks)
    if not np.isfinite(Gs).all():
        raise FloatingPointError("non-finite predictive-rank matrix")
    Ug, sg, _ = np.linalg.svd(Gs, full_matrices=False)
    energy = np.cumsum(sg ** 2) / np.sum(sg ** 2)
    p = max(1, min(int(np.searchsorted(energy, energy_rule) + 1), len(sg)))
    return dict(mu_x=mu_x, mu_m=mu_m, Gam=Gam, Pmap=Ug[:, :p].T @ Wm,
                spectrum=sg.tolist(), energy=energy.tolist(), p=int(p))


def project(x, M, proj):
    """z = P (m - mu_m - Gam (y - mu_y)); x (n,), M (n, k) -> (n, p)."""
    x = np.asarray(x, float).reshape(-1, 1)
    with np.errstate(all="ignore"):
        Mt = (M - proj["mu_m"]) - (x - proj["mu_x"]) @ proj["Gam"]
        z = Mt @ proj["Pmap"].T
    if not np.isfinite(z).all():
        raise FloatingPointError("non-finite predictive coordinates")
    return z


# ------------------------------------------------------------- tuples

def build_tuples(X, rates, dt=None, kappa_s=None, proj=None, reduce=None):
    """Conditioning tuples c_n = (y_n, z_n) [or (y_n, m_n)], targets
    s_n = (y_{n+1} - y_n) kappa_s. With reduce on and proj None the
    projection is estimated here (and returned)."""
    dt = cfg.DT if dt is None else dt
    reduce = cfg.REDUCE if reduce is None else reduce
    m, n_burn = ema_features(X, rates, dt)
    D = np.diff(X, axis=1)
    if kappa_s is None:
        kappa_s = 1.0 / D.std()
    n0 = n_burn
    H = cfg.H_RANK
    x_n = X[:, n0:-1].reshape(-1)
    m_n = m[:, n0:-1].reshape(-1, len(rates))
    d_n = D[:, n0:].reshape(-1)
    if reduce and len(rates):
        if proj is None:
            xs = X[:, n0:-H]
            ms = m[:, n0:-H].reshape(-1, len(rates))
            FUT = np.stack([D[:, n0 + h - 1:D.shape[1] - H + h]
                            for h in range(1, H + 1)], 2).reshape(-1, H)
            proj = predictive_rank(xs.reshape(-1), ms, FUT, cfg.ENERGY_RULE)
        C = np.hstack([x_n[:, None], project(x_n, m_n, proj)])
    else:
        C = np.hstack([x_n[:, None], m_n])
    S = (d_n * kappa_s).reshape(-1, 1)
    return {"C": C, "S": S, "X0": x_n, "kappa_s": float(kappa_s),
            "rates": np.asarray(rates, float), "n_burn": n_burn,
            "proj": proj}
