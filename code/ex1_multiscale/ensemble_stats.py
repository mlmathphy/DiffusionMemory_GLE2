"""Ensemble statistics for Example 4: the conditional-forecast
manuscript figure (torch inference only; needs trained checkpoints).

TWO figures: figs/ex4_ensemble.pdf (manuscript, panels a-b) and
figs/ex4_w2.pdf (the W2 panel; its marked values are Table 1(a)):

(a, b) SINGLE RELEASE: many particles are released from ONE held-out
   state + prehistory, and the ensemble mean and standard deviation
   are tracked against the exact conditional law given the full
   observed history (Kalman oracle) at every horizon up to
   tau = 2 T_max. The release point is chosen from a held-out pool as
   the query with the largest slow-mode displacement (largest |oracle
   mean| at tau = T_max), i.e. the history for which memory matters
   most; the choice rule is stated and deterministic. The exact
   conditional given (x_n, m_n) is drawn as the labeled compression
   floor. These panels show the MECHANISM: the difference lies in the
   conditional mean, not the spread.

(c) DISTRIBUTION METRIC over TYPICAL histories: the squared
   Wasserstein-2 distance between each model's conditional ensemble
   and the exact full-history conditional, normalized by the exact
   conditional variance -- medians over the same query set as
   forecast_fans.py (identical seeds). W2 combines mean and spread
   errors in one number per horizon, and being an aggregate over
   held-out histories it shows the single release is not a favorable
   pick. Floors: the bank compression floor (closed form,
   Gaussian-Gaussian W2) and the finite-ensemble sampling floor
   (Monte-Carlo estimate, horizon-independent in normalized units).

Prints a '===== NUMBERS =====' block to copy back into the manuscript,
and writes out/ensemble_stats.json (which stores every plotted curve,
so the figure can be rebuilt without any regeneration).

    python ensemble_stats.py --smoke     # minutes-scale sanity pass
    python ensemble_stats.py
    python ensemble_stats.py plot        # figure from stored JSON only
                                         # (numpy + matplotlib, no torch)
"""

import json
import os
import sys

import numpy as np

import config as cfg

# display order / labels / colors, mirrored from evaluate.py so that
# `plot` mode needs neither torch nor the checkpoints
CASES = [("markov", "no memory"),
         ("window", "raw last-$k$ history"),
         ("window_long", f"raw last-{cfg.WINDOW_LONG_M} history"),
         ("lstm", "LSTM + Gaussian head"),
         ("mem", "memory bank (ours)")]
COLORS = {"markov": "tab:green", "window": "tab:orange",
          "window_long": "tab:brown", "lstm": "tab:purple",
          "mem": "tab:red"}

MARKS = (30.0, 100.0, 200.0, 400.0)          # W2 report horizons, t.u.
REL_MARKS = (2.0, 10.0, 30.0, 100.0, 200.0, 400.0)
W2_FLOOR_REPS = 64


def gen_ensembles(cases, X_test, ti, ni, n_ens, H, seed0, device,
                  keep_members=False):
    """Batched per-query ensembles for every case; returns
    name -> (mu, sd) arrays (n_q, H), plus the raw members (n_q, n_ens,
    H) when keep_members (needed for W2)."""
    from evaluate import rollout
    from memdiff.lstm import ensemble_rollout_multi
    n_q = len(ti)
    out = {}
    for name, pl in cases.items():
        mu = np.empty((n_q, H))
        sd = np.empty((n_q, H))
        mem = np.empty((n_q, n_ens, H)) if keep_members else None
        for g0 in range(0, n_q, cfg.FAN_Q_BATCH):
            g1 = min(g0 + cfg.FAN_Q_BATCH, n_q)
            seed_g = seed0 + g0
            if pl["kind"] == "lstm":
                pre = np.stack([X_test[ti[q], ni[q] - cfg.LSTM_WARM:
                                       ni[q] + 1] for q in range(g0, g1)])
                E = ensemble_rollout_multi(pl["net"], pl["meta"], pre,
                                           n_ens, H, seed_g, device)
            else:
                x0 = np.repeat(X_test[ti[g0:g1], ni[g0:g1]], n_ens)
                m0 = np.repeat(pl["m"][ti[g0:g1], ni[g0:g1]], n_ens,
                               axis=0)
                E = rollout(pl["net"], pl["ckpt"]["norm"],
                            pl["ckpt"]["rates"], pl["ckpt"]["kappa_s"],
                            x0, m0, H, seed_g, device,
                            kind=pl["kind"]).reshape(g1 - g0, n_ens, H)
            mu[g0:g1] = E.mean(1)
            sd[g0:g1] = E.std(1)
            if keep_members:
                mem[g0:g1] = E
        out[name] = {"mu": mu, "sd": sd, "E": mem}
        print(f"  {name}: {n_q} x {n_ens} ensembles done")
    return out


def bank_conditional(cases, X_test, ti, ni, H):
    """Exact conditional given (x_n, m_n): the compression floor."""
    import multiscale_exact as exact
    if "mem" not in cases:
        return None
    rates = np.asarray(cases["mem"]["ckpt"]["rates"], float)
    A, Vh = exact.cond_horizon_given_memory(rates, H)
    v0 = np.concatenate([X_test[ti, ni][:, None],
                         cases["mem"]["m"][ti, ni]], axis=1)
    return {"mu": v0 @ A.T, "sd": np.sqrt(Vh)}


def make_figure(summary, tag=""):
    """Rebuild figs/ex4_ensemble.pdf from the (stored) summary dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, MODEL_ALPHA, paper_typography
    setup_style()
    os.makedirs(cfg.FIG_DIR, exist_ok=True)

    rc = summary.get("release_curves")
    w2 = summary.get("w2")
    if rc is None or w2 is None:
        raise SystemExit("stored JSON has no curves (older run); rerun "
                         "the full stage once: python ensemble_stats.py")
    tau = np.asarray(rc["tau"])
    mu_or = np.asarray(rc["oracle_mu"])
    sd_or = np.asarray(rc["oracle_sd"])
    sig = float(summary["stationary_sigma"])

    # Two output files, matching the manuscript: ex4_ensemble.pdf holds
    # the single-release mean/spread panels (a, b); the W2 panel goes to
    # the separate ex4_w2.pdf (not included in the paper; its marked
    # values are Table 1(a)).
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].fill_between(tau, mu_or - 2 * sd_or, mu_or + 2 * sd_or,
                       color="k", alpha=0.10)
    ax[0].plot(tau, mu_or, "k-", lw=2.0, label="exact | full history")
    if rc.get("bank_mu") is not None:
        ax[0].plot(tau, rc["bank_mu"], ":", color="0.4", lw=1.5,
                   label="exact | $(x_n, m_n)$")
    for name, label in CASES:
        if name not in rc["models"]:
            continue
        ax[0].plot(tau, rc["models"][name]["mu"], "-",
                   color=COLORS[name], alpha=MODEL_ALPHA, label=label)
    ax[0].set_xscale("log")
    ax[0].set(xlabel=r"time $\tau$ after release",
              ylabel="ensemble mean",
              title="(a) conditional mean (shading: exact $\\pm 2\\sigma$)")
    ax[0].legend(fontsize=6)

    ax[1].plot(tau, sd_or, "k-", lw=2.0, label="exact | full history")
    if rc.get("bank_sd") is not None:
        ax[1].plot(tau, rc["bank_sd"], ":", color="0.4", lw=1.5,
                   label="exact | $(x_n, m_n)$")
    for name, label in CASES:
        if name not in rc["models"]:
            continue
        ax[1].plot(tau, rc["models"][name]["sd"], "-",
                   color=COLORS[name], alpha=MODEL_ALPHA, label=label)
    ax[1].axhline(sig, color="k", ls="--", lw=0.8,
                  label=r"stationary $\sigma$")
    ax[1].set_xscale("log")
    ax[1].set(xlabel=r"time $\tau$ after release",
              ylabel="ensemble standard deviation",
              title="(b) conditional spread")
    ax[1].legend(fontsize=6)

    fig.tight_layout()
    out = f"{cfg.FIG_DIR}/ex4_ensemble{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out}")

    fig, axw = plt.subplots(figsize=(4.6, 3.4))
    tau_w = np.asarray(w2["tau"])
    for name, label in CASES:
        if name not in w2["curves"]:
            continue
        axw.loglog(tau_w, np.maximum(np.asarray(w2["curves"][name]),
                                     1e-8),
                   "-", color=COLORS[name], alpha=MODEL_ALPHA,
                   label=label)
    if w2.get("floor_curve") is not None:
        axw.loglog(tau_w, np.maximum(np.asarray(w2["floor_curve"]),
                                     1e-8),
                   ":", color="0.4", lw=1.5,
                   label="compression floor (closed form)")
    axw.axhline(w2["sampling_floor"], color="k", ls="--", lw=0.8,
                label="sampling floor "
                      f"($N={summary['config']['n_ens']}$)")
    axw.set(xlabel=r"forecast horizon $\tau$",
            ylabel=r"$W_2^2 \, / \, \sigma_{\mathrm{exact}}^2(\tau)$",
            title=f"median over {summary['config']['n_query']} "
                  "held-out histories")
    axw.legend(fontsize=6)
    paper_typography(fig, 0.9)
    fig.tight_layout()
    out = f"{cfg.FIG_DIR}/ex4_w2{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out}")


def main(smoke=False):
    import torch
    import multiscale_exact as exact
    from evaluate import load_cases
    from scipy.special import ndtri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = ""
    n_rel, pool = cfg.ENS_N_RELEASE, cfg.ENS_POOL
    if smoke:
        cfg.FAN_N_QUERY, cfg.FAN_N_ENS, cfg.FAN_Q_BATCH = 2, 32, 2
        n_rel, pool = 200, 16
        tag = "_smoke"
        print("SMOKE TEST: pool 16, release 200, 2 queries x 32 members")
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]
    H = int(round(2.0 * max(cfg.FORCING_TIMES) / cfg.DT))
    if smoke:
        H = min(H, 100)
    tauH = np.arange(1, H + 1) * cfg.DT

    cases, lo = load_cases(X_test, device)
    if not cases:
        raise SystemExit("no trained checkpoints under out/; train first")
    lo = max(lo, cfg.FAN_HIST + 1)
    hi = X_test.shape[1] - 1
    sig = float(np.sqrt(exact.acf_x(1)[0]))
    summary = {"config": {"n_release": n_rel, "pool": pool,
                          "n_query": cfg.FAN_N_QUERY,
                          "n_ens": cfg.FAN_N_ENS,
                          "horizon_tu": float(H * cfg.DT)},
               "stationary_sigma": sig}

    # ================= part 1: single-release mean/std =================
    rng = np.random.default_rng(cfg.SEED_EVAL + 1500)
    tp = rng.integers(0, X_test.shape[0], pool)
    np_ = rng.integers(lo, hi, pool)
    hist = np.stack([X_test[a, b - cfg.FAN_HIST:b + 1]
                     for a, b in zip(tp, np_)])
    mu_pool, kal_sd = exact.kalman_forecast(hist, H)
    h_ref = min(int(round(max(cfg.FORCING_TIMES) / cfg.DT)), H) - 1
    q = int(np.argmax(np.abs(mu_pool[:, h_ref])))
    ti1, ni1 = tp[q:q + 1], np_[q:q + 1]
    x0 = float(X_test[ti1[0], ni1[0]])
    print(f"release point (largest |oracle mean| at tau = "
          f"{(h_ref + 1) * cfg.DT:g} among {pool} held-out queries): "
          f"test traj {ti1[0]}, step {ni1[0]}, x0 = {x0:.4f} "
          f"({x0 / sig:.2f} sigma), oracle mean there "
          f"{mu_pool[q, h_ref]:.4f}")
    kal_mu1 = mu_pool[q]

    rel = gen_ensembles(cases, X_test, ti1, ni1, n_rel, H,
                        cfg.SEED_EVAL + 1600, device)
    bank1 = bank_conditional(cases, X_test, ti1, ni1, H)

    summary["release"] = {"traj": int(ti1[0]), "step": int(ni1[0]),
                          "x0": x0, "x0_sigma": x0 / sig}
    summary["release_curves"] = {
        "tau": tauH.tolist(),
        "oracle_mu": kal_mu1.tolist(), "oracle_sd": kal_sd.tolist(),
        "bank_mu": (None if bank1 is None
                    else bank1["mu"][0].tolist()),
        "bank_sd": (None if bank1 is None
                    else bank1["sd"].tolist()),
        "models": {name: {"mu": rel[name]["mu"][0].tolist(),
                          "sd": rel[name]["sd"][0].tolist()}
                   for name in rel}}
    rel_rows = {}
    for t in REL_MARKS:
        h = min(int(round(t / cfg.DT)), H) - 1
        row = {"oracle": {"mu": float(kal_mu1[h]),
                          "sd": float(kal_sd[h])}}
        if bank1 is not None:
            row["bank_conditional"] = {"mu": float(bank1["mu"][0, h]),
                                       "sd": float(bank1["sd"][h])}
        for name in rel:
            row[name] = {
                "mu": float(rel[name]["mu"][0, h]),
                "sd": float(rel[name]["sd"][0, h]),
                "mu_err_norm": float(abs(rel[name]["mu"][0, h]
                                         - kal_mu1[h]) / kal_sd[h]),
                "sd_ratio": float(rel[name]["sd"][0, h] / kal_sd[h])}
        rel_rows[f"tau={t:g}"] = row
    summary["release_marks"] = rel_rows

    # ================= part 2: W2(tau) metric ==========================
    rng = np.random.default_rng(cfg.SEED_EVAL + 1300)   # = forecast_fans
    ti = rng.integers(0, X_test.shape[0], cfg.FAN_N_QUERY)
    ni = rng.integers(lo, X_test.shape[1], cfg.FAN_N_QUERY)
    hist = np.stack([X_test[a, b - cfg.FAN_HIST:b + 1]
                     for a, b in zip(ti, ni)])
    kal_mu, kal_sd2 = exact.kalman_forecast(hist, H)
    _, Vinf = exact.v_full()
    relv = abs(kal_sd2[0] ** 2 - Vinf) / Vinf
    print(f"filter check: one-step forecast var {kal_sd2[0] ** 2:.6f} "
          f"vs V_full {Vinf:.6f}  (rel diff {relv:.1e})")
    if relv > 1e-3:
        raise FloatingPointError("Kalman variance disagrees with V_full")

    n_ens = cfg.FAN_N_ENS
    z = ndtri((np.arange(n_ens) + 0.5) / n_ens)
    ens = gen_ensembles(cases, X_test, ti, ni, n_ens, H,
                        cfg.SEED_EVAL + 1400, device, keep_members=True)

    w2 = {}
    for name in ens:
        w2_q = np.empty((cfg.FAN_N_QUERY, H))
        for qi in range(cfg.FAN_N_QUERY):       # per query: (n_ens, H)
            Es = np.sort(ens[name]["E"][qi], axis=0)
            t = kal_mu[qi][None, :] + kal_sd2[None, :] * z[:, None]
            w2_q[qi] = ((Es - t) ** 2).mean(0) / kal_sd2 ** 2
        w2[name] = np.median(w2_q, axis=0)
        ens[name]["E"] = None                       # release memory

    bank2 = bank_conditional(cases, X_test, ti, ni, H)
    w2_floor = None
    if bank2 is not None:                    # Gaussian-Gaussian W2^2
        w2_floor = np.median(((bank2["mu"] - kal_mu) ** 2
                              + (bank2["sd"][None, :]
                                 - kal_sd2[None, :]) ** 2)
                             / kal_sd2[None, :] ** 2, axis=0)
    rngf = np.random.default_rng(cfg.SEED_EVAL + 1700)
    samp = [np.mean((np.sort(rngf.standard_normal(n_ens)) - z) ** 2)
            for _ in range(W2_FLOOR_REPS)]
    w2_samp = float(np.median(samp))
    print(f"finite-ensemble W2 sampling floor (normalized, "
          f"N = {n_ens}): {w2_samp:.5f}")

    w2_marks = {}
    for name, curve in list(w2.items()) + (
            [("bank_conditional", w2_floor)] if w2_floor is not None
            else []):
        w2_marks[name] = {}
        for t in MARKS:
            h = min(int(round(t / cfg.DT)), H) - 1
            w2_marks[name][f"tau={t:g}"] = float(curve[h])
    summary["w2"] = {"marks": w2_marks, "sampling_floor": w2_samp,
                     "curves": {name: c.tolist()
                                for name, c in w2.items()},
                     "floor_curve": (None if w2_floor is None
                                     else w2_floor.tolist()),
                     "tau": tauH.tolist()}

    make_figure(summary, tag)
    with open(os.path.join(cfg.OUT_DIR, f"ensemble_stats{tag}.json"),
              "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {cfg.OUT_DIR}/ensemble_stats{tag}.json")

    # ======================== NUMBERS block ============================
    print("\n===== ensemble_stats NUMBERS (copy this block back) =====")
    print(f"release: traj {ti1[0]}, step {ni1[0]}, x0 {x0:.4f} "
          f"({x0 / sig:.2f} sigma)")
    hdr = "tau      " + "  ".join(f"{t:>6g}" for t in REL_MARKS)
    print("single release, |mean err| / sigma_exact:")
    print(hdr)
    for name in rel:
        vals = [rel_rows[f"tau={t:g}"][name]["mu_err_norm"]
                for t in REL_MARKS]
        print(f"  {name:12s}" + "  ".join(f"{v:6.3f}" for v in vals))
    print("single release, sd ratio (model / exact):")
    print(hdr)
    for name in rel:
        vals = [rel_rows[f"tau={t:g}"][name]["sd_ratio"]
                for t in REL_MARKS]
        print(f"  {name:12s}" + "  ".join(f"{v:6.3f}" for v in vals))
    print(f"W2^2 / sigma^2 medians over {cfg.FAN_N_QUERY} queries "
          f"(sampling floor {w2_samp:.5f}):")
    print("tau      " + "  ".join(f"{t:>8g}" for t in MARKS))
    order = [n for n, _ in CASES if n in w2] + (
        ["bank_conditional"] if w2_floor is not None else [])
    for name in order:
        vals = [w2_marks[name][f"tau={t:g}"] for t in MARKS]
        print(f"  {name:16s}" + "  ".join(f"{v:8.4f}" for v in vals))
    print("=========================================================")


if __name__ == "__main__":
    if "plot" in sys.argv[1:]:
        p = os.path.join(cfg.OUT_DIR, "ensemble_stats.json")
        if not os.path.exists(p):
            raise SystemExit(f"no {p}; run the full stage first")
        with open(p) as f:
            make_figure(json.load(f))
    else:
        main(smoke="--smoke" in sys.argv[1:])
