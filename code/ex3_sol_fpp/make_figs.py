"""Conditional phase-space evolution figure for Example 3 (torch + particle
filter; workstation, after evaluate.py) -- the honest analogue of the
Example-2 cloud-evolution figure for a scalar SOL signal (codeX spec).

  figs/ex3_cloud.pdf   3 x 3 grid. Rows: observable-history PF reference /
                       memory-conditioned diffusion / Gaussian-head LSTM.
                       Columns: forecast horizons t - t_n = 0.5, 2, 5 tau_d1.
                       Coordinates (y_{n+h}, (y_{n+h} - y_{n+h-1})/dt).
                       Initial condition: the predeclared active held-out
                       history (seeded rule); N_ENS paths from the same
                       history for all rows. Rendering: 2-D histogram on ONE
                       fixed grid smoothed with ONE Gaussian kernel (physical
                       widths KDE_SIG), filled 50/80/95% highest-density
                       contours OF THE IN-FRAME MASS, identical axes;
                       annotation y_99 (99th percentile of y_{n+h}); red
                       line: burst threshold; "X% outside" annotation when
                       more than 1% of a panel's samples fall outside the
                       fixed grid (Example-2 safeguard). The three
                       ensembles are fingerprint-cached under out/cache/
                       (cloud_ens.npz): a fresh run performs one PF
                       assimilation and two short generations; styling
                       reruns are cache hits.

    python make_figs.py [--smoke]
    python make_figs.py --plot   # render existing cache without loading models
"""

import os
import sys

import numpy as np
import json
from scipy.ndimage import gaussian_filter

import config as cfg
import model as md
from pfilter import ParticleFilter
from conditioning import ema_features
if "--plot" not in sys.argv:
    import torch
    from evaluate import load_mem, load_lstms, rollout_mem
    from memdiff.lstm import ensemble_rollout_multi

H_CLOUD_T = (0.5, 2.0, 5.0)          # horizons in tau_d1
N_ENS = 4000
KDE_SIG = (0.35, 0.7)                 # kernel widths in (y, dy/dt) units
COL = {"ref": "0.35", "ours": "C3", "lstm": "C0"}
LAB = {"ref": "observable-history\nreference (PF)",
       "ours": "memory-conditioned\ndiffusion", "lstm": "LSTM +\nGaussian head"}


def hdr_levels(D, masses=(0.5, 0.8, 0.95)):
    v = np.sort(D.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    lv = [v[min(int(np.searchsorted(c, p)), len(v) - 1)] for p in masses]
    return np.unique([x for x in lv if x > 0])


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        cfg.apply_smoke()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from memdiff.plotting import setup_style, paper_typography
    setup_style()
    plot_only = "--plot" in sys.argv
    device = None if plot_only else ("cuda" if torch.cuda.is_available() else "cpu")
    P = md.params()
    mu, sd = P["mean"], np.sqrt(P["var"])
    thr = mu + cfg.BURST_SIGMA * sd
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]
    if plot_only:
        with open(os.path.join(cfg.OUT_DIR, "mem", "selection.json")) as f:
            ck = json.load(f)
    else:
        net, ck = load_mem(device)
        lstm = next(iter(load_lstms(device).values()))
    m_te, n_burn = ema_features(X_test, np.asarray(ck["rates"]))
    lo = max(n_burn, cfg.LSTM_WARM, cfg.PF_WARM) + 1
    rng = np.random.default_rng(cfg.SEED_EVAL + 77)
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    hz = [int(round(h / cfg.DT)) for h in H_CLOUD_T]
    H = max(hz)
    n_ens = N_ENS if not smoke else 400
    # predeclared active history: seeded draw among times >= lo with y above
    # the 95% quantile (same rule and seed as before)
    yq = X_test[:, lo:-H - 1]
    qlo, qhi = np.quantile(yq, [0.3, 0.95])
    cand_q = np.argwhere(yq < qlo)
    _ = cand_q[rng.integers(len(cand_q))]           # (quiet draw kept for seed parity)
    cand = np.argwhere(yq > qhi)
    i, t = cand[rng.integers(len(cand))]
    i, t = int(i), int(t) + lo
    print(f"active history: trajectory {i}, time index {t}, y = {X_test[i, t]:.3f}")

    # the three ensembles are fingerprint-cached (history, checkpoints,
    # horizon, ensemble size, PF size, seeds) so that styling revisions do
    # not redo the PF assimilation and the generation
    if plot_only:
        E3 = np.load(os.path.join(cfg.OUT_DIR, "cache", f"cloud_ens{cfg.TAG}.npz"))["value"]
        if E3.shape != (3, n_ens, H):
            raise ValueError(f"Unexpected cloud cache shape: {E3.shape}")
    else:
        from memdiff.cache import cached, fingerprint, state_fingerprint
        n_pf = cfg.PF_N_EVAL if not smoke else 1000
        seeds = [int(rng.integers(1 << 31)) for _ in range(3)]
        key = fingerprint(X_test[i, :t + 1], i, t, H, n_ens, n_pf, seeds, cfg.DT,
                          cfg.TAU_D, cfg.GAMMA, cfg.TAU_R, cfg.EPS_NOISE,
                          cfg.LSTM_WARM, state_fingerprint(ck["state"]), ck["norm"],
                          ck["kappa_s"], np.asarray(ck["rates"]),
                          ck["selection"].get("proj") or {},
                          state_fingerprint(lstm["net"].state_dict()), lstm["meta"],
                          "cloud-v1")

        def compute():
            pf = ParticleFilter(P, n_pf, np.random.default_rng(seeds[0]), guided=True)
            pf.run(X_test[i], t)
            Ypf = pf.predictive_path(n_ens, H)
            x0 = np.full(n_ens, X_test[i, t])
            m0 = np.repeat(m_te[i, t][None, :], n_ens, 0)
            Yo = rollout_mem(net, ck, x0, m0, H, seeds[1], device)
            pre = X_test[i, t - cfg.LSTM_WARM:t + 1][None, :]
            Yl = ensemble_rollout_multi(lstm["net"], lstm["meta"], pre, n_ens, H,
                                        seeds[2], device)[0]
            return np.stack([Ypf, Yo, Yl])                  # (3, n_ens, H)
        E3 = cached(os.path.join(cfg.OUT_DIR, "cache", f"cloud_ens{cfg.TAG}.npz"),
                    key, compute)
    ens = {"ref": E3[0], "ours": E3[1], "lstm": E3[2]}
    y0 = X_test[i, t]

    def rate(Y, h):
        prev = Y[:, h - 2] if h >= 2 else np.full(Y.shape[0], y0)
        return (Y[:, h - 1] - prev) / cfg.DT

    # one fixed grid and one kernel for every panel
    ye = np.linspace(-1.0, mu + 7 * sd, 121)
    ve = np.linspace(-3.0 * sd / cfg.DT * 0.6, 4.0 * sd / cfg.DT * 0.6, 121)
    yc, vc = 0.5 * (ye[1:] + ye[:-1]), 0.5 * (ve[1:] + ve[:-1])
    sig_bins = (KDE_SIG[0] / (ye[1] - ye[0]), KDE_SIG[1] / (ve[1] - ve[0]))
    dens, outside = {}, {}
    for k in ens:
        dens[k], outside[k] = [], []
        for h in hz:
            yy, vv = ens[k][:, h - 1], rate(ens[k], h)
            inside = (yy >= ye[0]) & (yy < ye[-1]) & (vv >= ve[0]) & (vv < ve[-1])
            outside[k].append(1.0 - inside.mean())      # outside-frame fraction
            dens[k].append(gaussian_filter(
                np.histogram2d(yy, vv, bins=[ye, ve], density=True)[0], sig_bins))
    fig, ax = plt.subplots(3, 3, figsize=(10.5, 10.5), sharex=True, sharey=True)
    for r, k in enumerate(("ref", "ours", "lstm")):
        rgb = to_rgb(COL[k])
        for c, h in enumerate(hz):
            a = ax[r, c]
            D = dens[k][c]
            # Example-2 safeguard: HDR contours enclose percentages of the
            # IN-FRAME mass; annotate when > 1% of the samples lie outside
            if outside[k][c] > 0.01:
                a.text(0.03, 0.04, f"{100 * outside[k][c]:.1f}% outside",
                       transform=a.transAxes, fontsize=7, color="0.3")
            lv = hdr_levels(D)
            if len(lv) >= 2:
                levels = np.concatenate([lv, [D.max() * 1.0001]])
                cols = [(*rgb, al) for al in np.linspace(0.25, 0.85, len(levels) - 1)]
                a.contourf(yc, vc, D.T, levels=levels, colors=cols)
                a.contour(yc, vc, D.T, levels=lv, colors=[rgb], linewidths=0.6)
            a.axvline(thr, color="r", lw=0.9)
            a.axhline(0, color="0.8", lw=0.6)
            y99 = np.quantile(ens[k][:, h - 1], 0.99)
            a.text(0.03, 0.92, f"$y_{{99}} = {y99:.1f}$", transform=a.transAxes,
                   fontsize=9)
            if r == 0:
                a.set_title(f"$t - t_n = {h * cfg.DT:g}\\,\\tau_{{d1}}$")
            if c == 0:
                a.set_ylabel(f"{LAB[k]}\n" + r"$(y_{n+h}-y_{n+h-1})/\Delta t$",
                             fontsize=9)
            if r == 2:
                a.set_xlabel("$y_{n+h}$")
    paper_typography(fig, 0.76)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_cloud{cfg.TAG}.pdf"))
    plt.close(fig)
    print(f"figure -> {cfg.FIG_DIR}/ex3_cloud{cfg.TAG}.pdf; outside-frame fractions "
          f"{ {k: [round(x, 4) for x in v] for k, v in outside.items()} }")


if __name__ == "__main__":
    main()
