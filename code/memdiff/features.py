"""Shared numpy primitives: ACF/MSD estimators, rate bands, EMA bank.

All functions are config-free; every parameter is explicit.
"""

import numpy as np


# ------------------------------------------------------------- ACF / MSD

def acf_estimate(X, max_lag):
    """Autocovariance of a stationary series, averaged over rows of X.

    X: (n_traj, L). Returns acov[0..max_lag] (not normalized), FFT-based.
    """
    X = np.asarray(X, dtype=float)
    n_traj, L = X.shape
    max_lag = min(max_lag, L - 1)
    Xc = X - X.mean()
    nfft = int(2 ** np.ceil(np.log2(2 * L)))
    F = np.fft.rfft(Xc, n=nfft, axis=1)
    acov = np.fft.irfft((F * np.conj(F)).mean(axis=0), n=nfft)[:max_lag + 1]
    return acov / (L - np.arange(max_lag + 1))


def vel_acf(D, max_lag):
    """Per-component increment (velocity) autocovariance.

    D: (n_traj, L, d). Returns (d, max_lag+1).
    """
    D = np.asarray(D, float)
    return np.stack([acf_estimate(D[:, :, c], max_lag)
                     for c in range(D.shape[2])])


def _msd_1d(x, max_lag):
    """FFT MSD (standard S1 - 2*S2 algorithm), x: (n_traj, L)."""
    x = np.asarray(x, float)
    n, L = x.shape
    max_lag = min(max_lag, L - 1)
    nfft = int(2 ** np.ceil(np.log2(2 * L)))
    F = np.fft.rfft(x, n=nfft, axis=1)
    S2 = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :max_lag + 1]
    D = x ** 2
    Q = 2.0 * D.sum(axis=1)
    S1 = np.empty((n, max_lag + 1))
    for m in range(max_lag + 1):
        if m > 0:
            Q = Q - D[:, m - 1] - D[:, L - m]
        S1[:, m] = Q / (L - m)
    return (S1 - 2.0 * S2 / (L - np.arange(max_lag + 1))).mean(axis=0)


def msd(X, max_lag):
    """Ensemble MSD, MSD(l) = <|x_{n+l} - x_n|^2>. X: (n_traj, L, d)."""
    X = np.asarray(X, float)
    return sum(_msd_1d(X[:, :, c], max_lag) for c in range(X.shape[2]))


# ------------------------------------------------------------- rate band

def select_rate_band(incr_acf_norm, dt, thresh):
    """Band [lam_min, lam_max] from the normalized increment ACF.

    lam_min: inverse of the largest lag at which |ACF| is distinguishable
    from sampling noise (threshold `thresh`); lam_max: fastest scale
    resolvable at the sampling interval, 1/dt (paper, Section 3.1).
    """
    r = np.asarray(incr_acf_norm)
    sig = np.nonzero(np.abs(r[1:]) > thresh)[0]
    l_max = int(sig[-1]) + 1 if len(sig) else 1
    lam_min = 1.0 / (l_max * dt)
    lam_max = 1.0 / dt
    if lam_min >= lam_max:
        lam_min = lam_max / 10.0
    return lam_min, lam_max


def rates_from_band(band, k):
    """Log-spaced rates across the selected band (paper, Section 3.1);
    geometric center for k = 1."""
    lam_min, lam_max = band
    if k == 0:
        return np.array([])
    if k == 1:
        return np.array([np.sqrt(lam_min * lam_max)])
    i = np.arange(k)
    return lam_min * (lam_max / lam_min) ** (i / (k - 1))


# ------------------------------------------------------------- EMA bank

def ema_features(X, rates, dt, burn_tol):
    """EMA memory bank (paper, Section 3.1):
    m_n = rho m_{n-1} + (X_n - X_{n-1}), m_0 = 0.

    X: (n_traj, L+1) states, or (n_traj, L+1, d) for d-dimensional
    states (bank applied per component, rate-major layout
    [rate0_c0, rate0_c1, ..., rate1_c0, ...]).

    Returns m aligned with X -- (n_traj, L+1, k) in the 1D case,
    (n_traj, L+1, d*k) otherwise -- and the burn-in index n_burn below
    which the finite start is not yet a good proxy for the stationary
    functional (rho_max^n > burn_tol).
    """
    X = np.asarray(X, float)
    onedim = X.ndim == 2
    if onedim:
        X = X[..., None]
    n_traj, Lp1, d = X.shape
    rates = np.asarray(rates, dtype=float)
    k = len(rates)
    m = np.zeros((n_traj, Lp1, d * k))
    if k == 0:
        return m, 0
    rho = np.repeat(np.exp(-rates * dt), d)        # per component
    D = np.tile(np.diff(X, axis=1), (1, 1, k))     # (n, L, d*k)
    for t in range(1, Lp1):
        m[:, t] = rho[None, :] * m[:, t - 1] + D[:, t - 1]
    n_burn = int(np.ceil(-np.log(burn_tol) / (rates.min() * dt)))
    return m, n_burn
