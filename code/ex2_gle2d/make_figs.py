"""Paper-style comparisons from summary.json only: no model inference.

Shared typography and solid transparent curves follow Example 1;
matched 50/80/95% in-frame density contours follow the SOL figures.
The cloud is ONE STEP at the first declared evaluation anchor, not a
stationary cloud or a multi-horizon forecast. All plot inputs are saved.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import config as cfg
from memdiff.plotting import setup_style, MODEL_ALPHA

COLORS = {"mem": "C0", "markov": "C2", "lstm": "C3", "exact_floor": "0.6", "exact": "k"}
LABELS = {"mem": "Memory + diffusion", "markov": "Memoryless diffusion",
          "lstm": "LSTM (full Gaussian)", "exact_floor": "Exact ensemble", "exact": "Kalman reference"}
ORDER = ("mem", "markov", "lstm", "exact_floor")


def make_figures(s, fig_dir=None):
    setup_style()
    out = Path(fig_dir or cfg.FIG_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        print(f"figure -> {path}")

    if "vacf_exact" not in s:
        raise SystemExit("Run the updated evaluation once to save figure inputs; no retraining is needed.")
    exact = np.asarray(s["vacf_exact"])
    t = np.arange(len(exact)) * s["dt"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for n, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ax = axes.flat[n]
        ax.plot(t, exact[:, i, j], "k-", lw=2, label="Exact")
        for name in ORDER:
            if name in s["vacf_curves"]:
                C = np.asarray(s["vacf_curves"][name])
                ax.plot(t[:len(C)], C[:, i, j], color=COLORS[name],
                        alpha=MODEL_ALPHA, label=LABELS[name])
        ax.set(title=f"({chr(97+n)}) " + rf"$C_{{{'xy'[i]}{'xy'[j]}}}(\tau)$",
               xlabel=r"lag $\tau$", ylabel="velocity covariance")
    axes[0, 0].legend(fontsize=8)
    save(fig, "ex2_vacf.pdf")

    st = s["stationary"]
    edges = np.asarray(st["edges"])
    x = (edges[1:] + edges[:-1]) / 2
    exact_pdf = np.exp(-x*x/(2*s["kbt"])) / np.sqrt(2*np.pi*s["kbt"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for j, ax in enumerate(axes):
        ax.plot(x, exact_pdf, "k-", lw=2, label="Gibbs reference")
        for name in ORDER:
            if name in st["models"]:
                ax.plot(x, st["models"][name]["density"][j], color=COLORS[name],
                        alpha=MODEL_ALPHA, label=LABELS[name])
        ax.set(xlabel=rf"$v_{'xy'[j]}$", ylabel="probability density",
               title=f"({chr(97+j)}) Stationary velocity")
    axes[0].legend(fontsize=8)
    save(fig, "ex2_stationary.pdf")

    cl = s["conditional_cloud"]
    names = [n for n in ("exact", "mem", "markov", "lstm") if n in cl["samples"]]
    mu, V = np.asarray(cl["reference_mean"]), np.asarray(cl["reference_cov"])
    sd = np.sqrt(np.diag(V))
    grids = [np.linspace(mu[j]-4*sd[j], mu[j]+4*sd[j], 65) for j in range(2)]
    centers = [(g[1:]+g[:-1])/2 for g in grids]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1*len(names), 3.1),
                             sharex=True, sharey=True, squeeze=False)
    for n, name in enumerate(names):
        ax = axes[0, n]
        draws = np.asarray(cl["samples"][name])
        H = np.histogram2d(draws[:, 0], draws[:, 1], bins=grids)[0]
        outside = 1 - H.sum()/len(draws)
        density = gaussian_filter(H, 1.0)
        if density.sum() > 0:
            sorted_d = np.sort(density.ravel())[::-1]
            mass = np.cumsum(sorted_d)/sorted_d.sum()
            levels = np.unique([sorted_d[min(np.searchsorted(mass, p), len(mass)-1)]
                                for p in (.5, .8, .95)])
            levels = levels[levels > 0]
            if len(levels):
                ax.contourf(*centers, density.T,
                            levels=np.r_[levels, density.max()+1e-8],
                            colors=[COLORS[name]], alpha=.25)
                ax.contour(*centers, density.T, levels=levels,
                           colors=[COLORS[name]], linewidths=1)
        ax.text(.03, .03, f"{outside:.1%} outside", transform=ax.transAxes, fontsize=8)
        ax.set(title=f"({chr(97+n)}) {LABELS[name]}", xlabel=r"$\Delta v_x$",
               xlim=(grids[0][0], grids[0][-1]), ylim=(grids[1][0], grids[1][-1]))
    axes[0, 0].set_ylabel(r"$\Delta v_y$")
    save(fig, "ex2_conditionals.pdf")

    names = [n for n in ORDER[:3] if n in s["common_reference"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    metrics = [("cov_dev", "(a) Stationary covariance", "max absolute error"),
               ("vacf_err", "(b) Velocity correlations", "max normalized error"),
               (None, "(c) One-step vs full history", r"median normalized moment $W_2^2$")]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        values = [s["closed_loop"][n][metric] if metric else
                  s["common_reference"][n]["median"] for n in names]
        ax.bar(np.arange(len(names)), values, color=[COLORS[n] for n in names], alpha=MODEL_ALPHA)
        ax.set_xticks(np.arange(len(names)), [LABELS[n].replace(" ", "\n", 1) for n in names], fontsize=8)
        if metric and "exact_floor" in s["closed_loop"]:
            ax.axhline(s["closed_loop"]["exact_floor"][metric], color="0.4", ls=":", label="Exact ensemble error")
            ax.legend(fontsize=7)
        ax.set(title=title, ylabel=ylabel)
    save(fig, "ex2_accuracy.pdf")
    return paths


if __name__ == "__main__":
    with open(Path(cfg.OUT_DIR) / "summary.json") as f:
        make_figures(json.load(f))
