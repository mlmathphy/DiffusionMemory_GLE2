"""Observable-history oracle: particle filters for the filtered-Poisson
model (codeX rulings, 2026-08-22).

The hidden state H_n = (X_d1, X_r1, X_d2, X_r2) has an exact transition
(model.step) and a Gaussian observation y_n = signal(H_n) + sigma_n xi.
Filtering with ONLY y_{0:n} gives particles ~ p(H_n | y_{0:n});
propagating every particle through the exact compound-Poisson step (with
fresh observation noise) gives draws of the observable-history
predictive law p(y_{n+1} | y_{0:n}) -- the target of an observable-
history model, independent of kNN.

Two proposals:
  * bootstrap: prior transition draws, likelihood weights (degenerates
    at large bursts: only particles that drew a matching amplitude
    survive -- weight ESS reaches ~1 at those steps);
  * guided (default): the arrivals' counts and times are drawn from the
    prior, but the amplitude of ONE designated arrival (the first fast
    arrival if any, else the first slow one) is drawn from its exact
    conditional given the observation and everything else -- a
    truncated normal, because the likelihood is Gaussian in the
    amplitude and the prior is exponential -- and the importance weight
    is corrected by prior/proposal. This is the locally optimal
    proposal for the amplitude; the residual degeneracy comes only from
    the discrete arrival configuration.
Systematic resampling at every step; the weight ESS is recorded at every
step.  Initial particles: stationary draws of the model; y_0 assimilated
without propagation.  Predictive calibration: rank of the realized
y_{t+1} among predictive draws (PIT), collected after `pit_from`.
"""

import numpy as np
from scipy.stats import truncnorm, norm

import config as cfg
import model as md


def _systematic(w, rng):
    n = len(w)
    u = (rng.uniform() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), u).clip(0, n - 1)


def step_guided(state, rng, P, y_obs, dt=None):
    """One guided step for n particles. Returns (new_state, logw) where
    logw = log p(y_obs | H') + log p(A*) - log q(A*) for the designated
    arrival amplitude A* (or log p(y_obs | H') without one)."""
    dt = cfg.DT if dt is None else dt
    n = state.shape[0]
    npop = len(P["a"])
    b = P["b"]
    Xd = [state[:, 2 * j] * np.exp(-dt / P["a"][j]) for j in range(npop)]
    Xr = [state[:, 2 * j + 1] * np.exp(-dt / b) for j in range(npop)]
    sig_rest = sum(P["c"][j] * (Xd[j] - Xr[j]) for j in range(npop))
    # designated arrival: (population, time, gain g) per particle, or none
    des_pop = np.full(n, -1)
    des_s = np.zeros(n)
    counts = [rng.poisson(P["nu"][j] * dt, n) for j in range(npop)]
    for j in range(npop):
        a = P["a"][j]
        N = counts[j]
        for i in range(int(N.max()) if n else 0):
            m = N > i
            nm = int(m.sum())
            s = rng.uniform(0.0, dt, nm)
            # the first arrival of the first population with an arrival
            # is designated (amplitude drawn later); others from the prior
            newdes = m & (des_pop < 0) if i == 0 else np.zeros(n, bool)
            if i == 0:
                newdes = newdes & m
            idx_m = np.nonzero(m)[0]
            isdes = newdes[idx_m]
            A = rng.exponential(P["amp"][j], nm)
            # prior amplitudes for non-designated arrivals
            contrib_d = A * np.exp(-(dt - s) / a)
            contrib_r = A * np.exp(-(dt - s) / b)
            Xd[j][idx_m[~isdes]] += contrib_d[~isdes]
            Xr[j][idx_m[~isdes]] += contrib_r[~isdes]
            if isdes.any():
                des_pop[idx_m[isdes]] = j
                des_s[idx_m[isdes]] = s[isdes]
        sig_rest = sum(P["c"][jj] * (Xd[jj] - Xr[jj]) for jj in range(npop))
    logw = np.empty(n)
    nodes = des_pop < 0
    logw[nodes] = norm.logpdf(y_obs, sig_rest[nodes], P["sig_n"])
    for j in range(npop):
        m = des_pop == j
        if not m.any():
            continue
        a = P["a"][j]
        s = des_s[m]
        ed, er = np.exp(-(dt - s) / a), np.exp(-(dt - s) / b)
        g = P["c"][j] * (ed - er)                     # signal gain of A
        r = y_obs - sig_rest[m]
        mu_q = r / g - P["sig_n"] ** 2 / (g ** 2 * P["amp"][j])
        sd_q = P["sig_n"] / g
        alpha = -mu_q / sd_q
        A = truncnorm.rvs(alpha, np.inf, loc=mu_q, scale=sd_q,
                          random_state=rng)
        logq = truncnorm.logpdf(A, alpha, np.inf, loc=mu_q, scale=sd_q)
        logp = -np.log(P["amp"][j]) - A / P["amp"][j]
        sig_new = sig_rest[m] + g * A
        logw[m] = norm.logpdf(y_obs, sig_new, P["sig_n"]) + logp - logq
        idx = np.nonzero(m)[0]
        Xd[j][idx] += A * ed
        Xr[j][idx] += A * er
    new = np.empty_like(state)
    for j in range(npop):
        new[:, 2 * j], new[:, 2 * j + 1] = Xd[j], Xr[j]
    if not np.isfinite(logw).all():
        raise FloatingPointError("non-finite guided-proposal weights")
    return new, logw


class ParticleFilter:
    def __init__(self, P, n_part, rng, guided=True, burn_t=None, pit_n=256,
                 pit_from=None):
        self.P, self.N, self.rng, self.guided = P, int(n_part), rng, guided
        burn = int(round((cfg.BURN_T if burn_t is None else burn_t) / cfg.DT))
        st = np.zeros((self.N, 2 * len(P["a"])))
        for _ in range(burn):
            st, _ = md.step(st, rng, P)
        self.st = st
        self.ess = []                       # weight ESS per assimilation
        self.pit = []                       # (t, PIT of y_{t+1})
        self.pit_n, self.pit_from = pit_n, pit_from
        self.t = -1

    def assimilate(self, y_obs, propagate=True):
        if propagate and self.pit_from is not None and self.t >= self.pit_from:
            d = self.predictive(self.pit_n)
            self.pit.append((self.t, float(np.mean(d < y_obs))))
        if propagate and self.guided:
            self.st, logw = step_guided(self.st, self.rng, self.P, y_obs)
        else:
            if propagate:
                self.st, _ = md.step(self.st, self.rng, self.P)
            sig = md.signal(self.st, self.P)
            logw = -0.5 * ((y_obs - sig) / self.P["sig_n"]) ** 2
        logw = logw - logw.max()
        w = np.exp(logw)
        w /= w.sum()
        self.ess.append(float(1.0 / np.sum(w ** 2)))
        self.st = self.st[_systematic(w, self.rng)]
        self.t += 1

    def run(self, y_series, t_stop):
        """Assimilate y_0 .. y_{t_stop} (inclusive)."""
        for t in range(self.t + 1, t_stop + 1):
            self.assimilate(y_series[t], propagate=(t > 0))

    def predictive(self, n_s):
        """n_s draws of y_{t+1} | y_{0:t} (fresh arrivals and noise)."""
        idx = self.rng.integers(0, self.N, n_s)
        _, y = md.step(self.st[idx], self.rng, self.P)
        return y

    def predictive_path(self, n_s, H):
        """n_s exact multi-step predictive paths y_{t+1..t+H} | y_{0:t}:
        resampled particles propagated H steps with fresh arrivals and
        observation noise (the reference for conditional-forecast figures)."""
        idx = self.rng.integers(0, self.N, n_s)
        st = self.st[idx]
        out = np.empty((n_s, H))
        for h in range(H):
            st, y = md.step(st, self.rng, self.P)
            out[:, h] = y
        return out

    def moments(self):
        """Predictive mean/variance of y_{t+1} | y_{0:t} given the particle
        approximation (law of total variance over particles)."""
        mu, var = md.oracle_moments(self.st, self.P)
        return float(mu.mean()), float(var.mean() + mu.var())

    def ess_near(self, t, half=10):
        e = np.asarray(self.ess)
        lo, hi = max(0, t - half), min(len(e), t + half + 1)
        return e[lo:hi]
