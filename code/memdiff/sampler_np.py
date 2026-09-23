"""NumPy implementation of the training-free conditional sampler.

This is the torch-free mirror of :mod:`memdiff.sampler`.  It implements
the same supervised conditioning metric and the same regularized reverse
probability-flow ODE.  NumPy-only design scans and pre-vets should import
this module instead of reaching into an experiment or archive directory.
"""

import numpy as np
from scipy.spatial import cKDTree


class TrainingFreeSamplerNP:
    """Training-free conditional sampler for NumPy-only workflows."""

    def __init__(self, C_data, S_data, j_neighbors, nu, eps, n_ode):
        C_data = np.asarray(C_data, float)
        S_data = np.asarray(S_data, float)
        if C_data.ndim == 1:
            C_data = C_data[:, None]
        if S_data.ndim == 1:
            S_data = S_data[:, None]
        if C_data.ndim != 2 or S_data.ndim != 2:
            raise ValueError("C_data and S_data must be one- or two-dimensional")
        if len(C_data) != len(S_data):
            raise ValueError("C_data and S_data must have the same row count")
        if not 1 <= int(j_neighbors) <= len(C_data):
            raise ValueError("j_neighbors must lie between 1 and len(C_data)")
        if float(nu) <= 0 or float(eps) < 0 or int(n_ode) <= 0:
            raise ValueError("nu and n_ode must be positive; eps must be nonnegative")
        self.J = int(j_neighbors)
        self.nu = float(nu)
        self.eps = float(eps)
        self.n_ode = int(n_ode)
        self._fit_metric(C_data, S_data)
        self.Chat = (C_data - self.mu_c) @ self.Lmap.T
        self.tree = cKDTree(self.Chat)
        self.S_data = S_data
        self.d_s = S_data.shape[1]

    def _fit_metric(self, C, S):
        self.mu_c = C.mean(axis=0)
        sigma = np.atleast_2d(np.cov(C.T))
        evals, evecs = np.linalg.eigh(sigma)
        whiten = (evecs
                  @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12)))
                  @ evecs.T)
        U = (C - self.mu_c) @ whiten
        B, *_ = np.linalg.lstsq(U, S - S.mean(axis=0), rcond=None)
        U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
        smax = s_B.max() if len(s_B) else 0.0
        rows = ((s_B / smax)[:, None] * U_B.T if smax > 0
                else np.zeros_like(U_B.T))
        self.Lmap = np.vstack([rows, self.eps * np.eye(C.shape[1])]) @ whiten
        self.metric_M = B

    def transform(self, C_query):
        return (np.atleast_2d(np.asarray(C_query, float)) - self.mu_c) \
            @ self.Lmap.T

    def neighbors(self, C_query):
        """Return neighbor indices, normalized weights, ESS, and radius."""
        dist, idx = self.tree.query(self.transform(C_query), k=self.J,
                                    workers=-1)
        if self.J == 1:
            dist, idx = dist[:, None], idx[:, None]
        logw = -0.5 * (dist / self.nu) ** 2
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        ess = 1.0 / (w ** 2).sum(axis=1)
        return idx, w, ess, dist[:, -1]

    def _reverse_ode(self, z0, s_neigh, logw_cond, n_ode):
        h = 1.0 / n_ode
        tau = np.linspace(1.0, 0.0, n_ode + 1)
        s = np.asarray(z0, float).copy()
        for j in range(n_ode):
            t = tau[j + 1]
            dtau = tau[j] - tau[j + 1]
            alpha = 1.0 - t + h
            beta2 = t + h
            f = -1.0 / alpha
            g2 = 1.0 - 2.0 * f * beta2
            resid = alpha * s_neigh - s[:, None, :]
            logw = -0.5 * (resid ** 2).sum(-1) / beta2 + logw_cond
            logw -= logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            w /= w.sum(axis=1, keepdims=True)
            score = (w[:, :, None] * resid).sum(1) / beta2
            s = s - (f * s - 0.5 * g2 * score) * dtau
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("non-finite reverse-ODE output")
        return s

    def sample_labels(self, C_query, z=None, rng=None, n_ode=None,
                      batch=4096):
        """Draw one label per condition; return ``(samples, z, diag)``."""
        n_ode = self.n_ode if n_ode is None else int(n_ode)
        C_query = np.atleast_2d(np.asarray(C_query, float))
        nq = len(C_query)
        if z is None:
            rng = np.random.default_rng() if rng is None else rng
            z = rng.standard_normal((nq, self.d_s))
        z = np.atleast_2d(np.asarray(z, float))
        if z.shape != (nq, self.d_s):
            raise ValueError("z must have shape (n_query, displacement_dim)")
        out = np.empty((nq, self.d_s))
        ess, radius = np.empty(nq), np.empty(nq)
        for a in range(0, nq, int(batch)):
            b = min(a + int(batch), nq)
            idx, w, e, rad = self.neighbors(C_query[a:b])
            out[a:b] = self._reverse_ode(
                z[a:b], self.S_data[idx], np.log(w + 1e-300), n_ode)
            ess[a:b], radius[a:b] = e, rad
        return out, z, {"ess": ess, "radius": radius}

    def sample_at(self, condition, n_samples, rng, idx=None, w=None,
                  n_ode=None):
        """Draw many samples at one condition from a shared neighbor set."""
        n_ode = self.n_ode if n_ode is None else int(n_ode)
        if idx is None or w is None:
            idx_all, w_all, _, _ = self.neighbors(condition)
            idx, w = idx_all[0], w_all[0]
        idx, w = np.asarray(idx), np.asarray(w, float)
        z0 = rng.standard_normal((int(n_samples), self.d_s))
        s_neigh = np.broadcast_to(
            self.S_data[idx][None, :, :],
            (int(n_samples), len(idx), self.d_s))
        logw = np.broadcast_to(
            np.log(w + 1e-300)[None, :], (int(n_samples), len(idx)))
        return self._reverse_ode(z0, s_neigh, logw, n_ode)

    def sample_conditional(self, condition, n_samples, n_ode=None, seed=None):
        """Torch-class-compatible wrapper for many draws at one condition."""
        rng = np.random.default_rng(seed)
        idx, w, ess, radius = self.neighbors(condition)
        samples = self.sample_at(condition, n_samples, rng, idx=idx[0],
                                 w=w[0], n_ode=n_ode)
        return samples, {"ess": float(ess[0]), "radius": float(radius[0])}


def parity_check(C_data, S_data, C_query, j_neighbors, nu, eps, n_ode,
                 seed=0):
    """Return the maximum label difference from the Torch implementation.

    ``None`` is returned when Torch is unavailable.  Torch uses float32, so
    parity is expected to numerical precision rather than bit-for-bit.
    """
    try:
        from memdiff.sampler import TrainingFreeSampler
    except (ImportError, ModuleNotFoundError):
        return None
    C_query = np.asarray(C_query, float)
    S_data = np.asarray(S_data, float)
    if C_query.ndim == 1:
        C_query = C_query[None, :]
    if S_data.ndim == 1:
        S_data = S_data[:, None]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(C_query), S_data.shape[1]))
    sampler_np = TrainingFreeSamplerNP(C_data, S_data, j_neighbors, nu,
                                       eps, n_ode)
    sample_np, _, _ = sampler_np.sample_labels(C_query, z=z)
    sampler_torch = TrainingFreeSampler(C_data, S_data, j_neighbors, nu,
                                        eps, n_ode, device="cpu")
    sample_torch, _, _ = sampler_torch.sample_labels(C_query, z=z)
    return float(np.max(np.abs(sample_np - sample_torch)))
