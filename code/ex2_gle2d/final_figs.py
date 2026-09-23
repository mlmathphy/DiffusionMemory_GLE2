"""The FOUR final Example-2 figures (user selection, 2026-09-09).

Plot-only: reads out/summary.json and the out/paper_gallery forecast
cache; never simulates, trains, or runs inference. Run
`python run_all.py paperfigs` once first if the cache is missing.

  final_figs/ex2_vacf.pdf         1 x 4 (xx, xy, yx, yy), lag <= 15
  final_figs/ex2_evolution1d.pdf  v_x-derived x-limits shared by both
                                  rows within each horizon column
  final_figs/ex2_evolution2d.pdf  blue-to-red colormap (RdYlBu_r)
  final_figs/ex2_msd.pdf          unchanged content

    python final_figs.py          # or: python run_all.py finalfigs
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import config as cfg
from memdiff.plotting import setup_style, paper_typography
from paper_figs import COL, LAB, ORDER

VACF_TAU_MAX = 15.0          # lag range in time units (was 30)
CMAP_2D = "RdYlBu_r"         # blue -> red

# The committed manuscript-figure location (synced by ../../sync_figs.sh);
# previously figs/final/, which left the committed final_figs/ copies
# unreachable by a rerun.
FIG_DIR = Path(__file__).resolve().parent / "final_figs"


def kde1(values, grid, bw):
    y = np.zeros(len(grid))
    for a in range(0, len(values), 512):
        y += np.exp(-0.5 * ((grid[:, None]
                             - values[None, a:a + 512]) / bw) ** 2
                    ).sum(1)
    return y / (len(values) * bw * np.sqrt(2 * np.pi))


def mass_levels(H):
    a = np.sort(H.ravel())[::-1]
    cum = np.cumsum(a) / a.sum()
    return np.unique([a[min(np.searchsorted(cum, p), len(a) - 1)]
                      for p in (.5, .95)])


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fractions = {"ex2_vacf.pdf": .98, "ex2_evolution1d.pdf": .92,
                 "ex2_evolution2d.pdf": .92, "ex2_msd.pdf": .85}
    paper_typography(fig, fractions[name])
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {FIG_DIR / name}")


def fig_vacf(s):
    n_keep = int(round(VACF_TAU_MAX / cfg.DT)) + 1
    exact = np.asarray(s["vacf_exact"])[:n_keep]
    t = np.arange(len(exact)) * s["dt"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), sharex=True,
                             layout="constrained")
    for n, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ax = axes[n]
        ax.plot(t, exact[:, i, j], color=COL["exact"], lw=2,
                label=LAB["exact"])
        for name in ("mem", "markov", "lstm"):
            if name in s["vacf_curves"]:
                C = np.asarray(s["vacf_curves"][name])[:n_keep]
                ax.plot(t[:len(C)], C[:, i, j], color=COL[name],
                        lw=1.2, alpha=.9, label=LAB[name])
        ax.set_title(f"({chr(97 + n)}) "
                     rf"$C_{{{'xy'[i]}{'xy'[j]}}}(\tau)$")
        ax.set_xlabel(r"lag $\tau$")
    axes[0].set_ylabel("velocity covariance")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
    save(fig, "ex2_vacf.pdf")


def load_gallery():
    cache = Path(cfg.OUT_DIR) / "paper_gallery"
    if not (cache / "metadata.json").exists():
        raise SystemExit("No forecast cache: run "
                         "`python run_all.py paperfigs` once first.")
    meta = json.loads((cache / "metadata.json").read_text())
    ens = {}
    for k in ORDER:
        with np.load(cache / f"{k}.npz") as d:
            if str(d["key"]) != meta["keys"][k]:
                raise ValueError(f"stale forecast cache: {k}; rerun "
                                 "paperfigs")
            ens[k] = d["V"]
    return meta, ens


def fig_evolution1d(meta, ens):
    mu = np.asarray(meta["exact_mean"])
    cov = np.asarray(meta["exact_cov"])
    sd = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    snaps, dt, n = meta["snaps"], meta["dt"], meta["n_ens"]
    fig, axes = plt.subplots(2, len(snaps),
                             figsize=(3 * len(snaps), 5.4),
                             layout="constrained")
    for c, h in enumerate(snaps):
        # x-limits from the v_x exact marginal, shared by BOTH rows
        xlo = mu[h - 1, 0] - 4.5 * sd[h - 1, 0]
        xhi = mu[h - 1, 0] + 4.5 * sd[h - 1, 0]
        grid = np.linspace(xlo, xhi, 240)
        for j in range(2):
            ax = axes[j, c]
            bw = max(ens["exact"][:, h - 1, j].std() * n ** (-1 / 5),
                     1e-8)
            for k in ORDER:
                ax.plot(grid, kde1(ens[k][:, h - 1, j], grid, bw),
                        color=COL[k], label=LAB[k],
                        alpha=.85 if k != "exact" else 1.0,
                        lw=2 if k == "exact" else 1.2)
            ax.set_xlim(xlo, xhi)
            ax.set_xlabel("$v_x$" if j == 0 else "$v_y$")
            if c == 0:
                ax.set_ylabel("density")
            if j == 0:
                ax.set_title(f"$h={h}$, $t={h * dt:g}$")
    fig.legend(*axes[0][-1].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
    save(fig, "ex2_evolution1d.pdf")


def fig_evolution2d(meta, ens):
    mu = np.asarray(meta["exact_mean"])
    cov = np.asarray(meta["exact_cov"])
    sd = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    snaps, dt, n = meta["snaps"], meta["dt"], meta["n_ens"]
    idx = np.asarray(snaps) - 1
    lo = (mu[idx] - 4.5 * sd[idx]).min(0)
    hi = (mu[idx] + 4.5 * sd[idx]).max(0)
    limlo, limhi = float(lo.min()), float(hi.max())
    edges = np.linspace(limlo, limhi, 121)
    centers = (edges[1:] + edges[:-1]) / 2
    dx = edges[1] - edges[0]
    fig, axes = plt.subplots(2, len(snaps),
                             figsize=(3 * len(snaps), 6.2),
                             sharex=True, sharey=True,
                             layout="constrained")
    for c, h in enumerate(snaps):
        ex = ens["exact"][:, h - 1]
        bw = np.sqrt(np.var(ex, axis=0).mean()) * n ** (-1 / 6)
        density = {}
        for k in ("exact", "mem"):
            pts = ens[k][:, h - 1]
            H = np.histogram2d(pts[:, 0], pts[:, 1],
                               bins=(edges, edges))[0]
            frac_out = 1 - H.sum() / n
            H = gaussian_filter(H, max(bw / dx, .6))
            density[k] = (H / (max(H.sum(), 1) * dx * dx), frac_out)
        vmax = max(d[0].max() for d in density.values())
        for r, k in enumerate(("exact", "mem")):
            ax = axes[r, c]
            H, frac_out = density[k]
            cf = ax.contourf(centers, centers,
                             (H / max(vmax, 1e-30)).T,
                             levels=np.linspace(0, 1, 11),
                             cmap=CMAP_2D)
            if H.sum() > 0:
                ax.contour(centers, centers, H.T,
                           levels=mass_levels(H), colors="0.15",
                           linewidths=.7)
            ax.set_aspect("equal")
            ax.set_xlim(limlo, limhi)
            ax.set_ylim(limlo, limhi)
            if r == 0:
                ax.set_title(f"$h={h}$, $t={h * dt:g}$")
            if r == 1:
                ax.set_xlabel("$v_x$")
            if c == 0:
                ax.set_ylabel(LAB[k] + "\n$v_y$")
            ax.text(.02, .02, f"{frac_out:.1%} outside",
                    transform=ax.transAxes, fontsize=7)
    fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=.8,
                 label="relative density\n(normalized per column)")
    save(fig, "ex2_evolution2d.pdf")


def fig_msd(meta, ens):
    dt = meta["dt"]
    tt = np.arange(1, meta["horizon"] + 1) * dt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8),
                             layout="constrained")
    for k in ORDER:
        V = ens[k]
        vx = ((V - np.asarray(meta["v0"])) ** 2).sum(2).mean(0)
        xx = (np.cumsum(V, axis=1) * dt) ** 2
        xx = xx.sum(2).mean(0)
        if k == "exact":
            vx, xx = meta["msd_v_exact"], meta["msd_x_exact"]
        axes[0].plot(tt, xx, color=COL[k], label=LAB[k])
        axes[1].plot(tt, vx, color=COL[k], label=LAB[k])
    axes[0].set(ylabel=r"$\mathbb{E}[|r_{n+h}-r_n|^2\mid H_n]$",
                title="(a) Sampled-displacement MSD")
    axes[1].set(ylabel=r"$\mathbb{E}[|v_{n+h}-v_n|^2\mid H_n]$",
                title="(b) Velocity change")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("forecast time")
        ax.legend(frameon=False)
    save(fig, "ex2_msd.pdf")


def main():
    setup_style()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    summary_path = Path(cfg.OUT_DIR) / "summary.json"
    if not summary_path.exists():
        raise SystemExit("out/summary.json missing — run the "
                         "completed-comparison machine.")
    s = json.loads(summary_path.read_text())
    fig_vacf(s)
    meta, ens = load_gallery()
    fig_evolution1d(meta, ens)
    fig_evolution2d(meta, ens)
    fig_msd(meta, ens)
    print(f"final set -> {FIG_DIR}")


if __name__ == "__main__":
    main()
