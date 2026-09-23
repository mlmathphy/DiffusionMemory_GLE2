"""Training-free memory-conditioned diffusion sampler (torch).

Implements Section 3.2 of the paper: the supervised conditioning metric,
the kNN-truncated score estimator, and the regularized reverse
probability-flow ODE with h = 1/n_ODE (coefficients evaluated at the
lower node of each backward step). The memory
enters only through the conditioning kernel; the memoryless baseline is
the same class fed baseline conditions.

Config-free: all parameters are explicit (the examples fill them from
their config via BaseExample.make_sampler).

Diagnostics per query (method lesson: always monitor these):
  ess    -- effective sample size of the conditioning weights
  radius -- metric distance to the J-th neighbor
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


class TrainingFreeSampler:

    def __init__(self, C_data, S_data, j_neighbors, nu, eps, n_ode,
                 label_batch=4096, device=None,
                 metric_mode="supervised"):
        self.J = j_neighbors
        self.nu = nu
        self.eps = eps
        self.n_ode = n_ode
        self.label_batch = label_batch
        # "supervised" (default, unchanged) or "isotropic": force the
        # eps-scaled whitened metric (the same form as the degenerate
        # smax = 0 branch).  Needed when the conditional mean is known
        # to be zero: a finite sample yields a small but nonzero
        # regression matrix that the s/smax normalization would
        # promote to a full-strength spurious supervised direction.
        self.metric_mode = metric_mode
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        C_data = np.asarray(C_data, float)
        S_data = np.asarray(S_data, float)
        self._fit_metric(C_data, S_data)
        self.Chat = (C_data - self.mu_c) @ self.Lmap.T
        self.tree = cKDTree(self.Chat)
        self.S_data = torch.tensor(S_data, dtype=torch.float32,
                                   device=self.device)
        self.d_s = S_data.shape[1]

    # ------------------------------------------------ supervised metric

    def _fit_metric(self, C, S):
        """Supervised conditioning metric M (paper, Section 3.2): whiten
        jointly, one least-squares solve, SVD."""
        self.mu_c = C.mean(axis=0)
        Sig = np.atleast_2d(np.cov(C.T))
        evals, evecs = np.linalg.eigh(Sig)
        W = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
        if self.metric_mode == "isotropic":
            self.Lmap = self.eps * W
            self.metric_M = None
            return
        U = (C - self.mu_c) @ W
        B, *_ = np.linalg.lstsq(U, S - S.mean(axis=0), rcond=None)
        U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
        smax = s_B.max()
        # degenerate fallback: if the regression detects no conditional-mean
        # dependence (smax = 0), the supervised rows vanish and the metric
        # reduces to the isotropic whitened distance scaled by eps
        rows_sup = (s_B / smax)[:, None] * U_B.T if smax > 0 \
            else np.zeros_like(U_B.T)                          # (r, d_c)
        self.Lmap = np.vstack([rows_sup,
                               self.eps * np.eye(C.shape[1])]) @ W
        self.metric_M = B          # paper notation: M

    def transform(self, C):
        return (np.asarray(C, float) - self.mu_c) @ self.Lmap.T

    # ------------------------------------------------ reverse ODE core

    def _reverse_ode(self, z0, s_neigh, logw_cond, n_ode):
        """Explicit Euler on the regularized schedule (paper, Section 3.2).

        z0: (B, d_s) latents; s_neigh: (B, J, d_s); logw_cond: (B, J).
        """
        h = 1.0 / n_ode
        tau = torch.linspace(1.0, 0.0, n_ode + 1)
        s = z0
        for j in range(n_ode):
            t = tau[j + 1].item()
            dtau = (tau[j] - tau[j + 1]).item()
            alpha = 1.0 - t + h
            beta2 = t + h
            f = -1.0 / alpha
            g2 = 1.0 - 2.0 * f * beta2
            resid = alpha * s_neigh - s[:, None, :]            # (B, J, d_s)
            logw = -0.5 * (resid ** 2).sum(-1) / beta2 + logw_cond
            w = torch.softmax(logw, dim=1)
            score = (w[:, :, None] * resid).sum(1) / beta2
            s = s - (f * s - 0.5 * g2 * score) * dtau
        return s

    def _neighbor_logw(self, C_query):
        """kNN indices, conditioning log-weights, and diagnostics."""
        Chat_q = self.transform(C_query)
        # workers=-1: exact same neighbors, parallel over queries — the
        # whitened metric space defeats KD-tree pruning, so single-core
        # queries dominate the label-generation wall clock
        dist, idx = self.tree.query(Chat_q, k=self.J, workers=-1)
        logw = torch.tensor(-0.5 * (dist / self.nu) ** 2,
                            dtype=torch.float32, device=self.device)
        wc = torch.softmax(logw, dim=1)
        ess = (1.0 / (wc ** 2).sum(dim=1)).cpu().numpy()
        radius = dist[:, -1]
        return idx, logw, {"ess": ess, "radius": radius}

    # ------------------------------------------------ public sampling

    def sample_labels(self, C_query, z=None, n_ode=None, batch=None):
        """One sample per query condition (label generation, Alg. 1).

        Returns (s, diag): s (Nq, d_s) scaled displacements, diagnostics.
        """
        n_ode = self.n_ode if n_ode is None else n_ode
        batch = self.label_batch if batch is None else batch
        C_query = np.asarray(C_query, float)
        nq = C_query.shape[0]
        if z is None:
            z = np.random.randn(nq, self.d_s)
        out = np.empty((nq, self.d_s))
        ess = np.empty(nq)
        radius = np.empty(nq)
        for a in range(0, nq, batch):
            b = min(a + batch, nq)
            idx, logw, diag = self._neighbor_logw(C_query[a:b])
            s_neigh = self.S_data[torch.tensor(idx, device=self.device)]
            z0 = torch.tensor(z[a:b], dtype=torch.float32, device=self.device)
            s = self._reverse_ode(z0, s_neigh, logw, n_ode)
            out[a:b] = s.cpu().numpy()
            ess[a:b], radius[a:b] = diag["ess"], diag["radius"]
        return out, z, {"ess": ess, "radius": radius}

    def sample_conditional(self, c, n_samples, n_ode=None, seed=None):
        """Many samples at ONE condition c (one-step verification, E2).

        Neighbors are found once and shared across latents.
        """
        n_ode = self.n_ode if n_ode is None else n_ode
        rng = np.random.default_rng(seed)
        idx, logw, diag = self._neighbor_logw(np.asarray(c, float)[None, :])
        s_neigh = self.S_data[torch.tensor(idx[0], device=self.device)]
        z0 = torch.tensor(rng.standard_normal((n_samples, self.d_s)),
                          dtype=torch.float32, device=self.device)
        s = self._reverse_ode(z0,
                              s_neigh[None, :, :].expand(n_samples, -1, -1),
                              logw.expand(n_samples, -1), n_ode)
        return s.cpu().numpy(), {"ess": diag["ess"][0],
                                 "radius": diag["radius"][0]}
