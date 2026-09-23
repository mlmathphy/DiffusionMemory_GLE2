"""The manuscript figures for Example 4. Nothing else.

  ex4_memory.pdf        one row x three panels, everything k means:
                        (a) ideal closed-loop ACF error vs conditioning
                        dimension, window AR(M) vs EMA bank at the
                        SELECTED band (star on the curve); (b) the
                        predictive spectrum of the selected bank;
                        (c) the worst-horizon loss of the rank-p
                        projection (closed form; numpy)
  (rank_analysis.py remains the standalone diagnostic with the
   hidden-variable reconstruction traces; its 3-panel figure
   ex4_rank.pdf is NOT part of the manuscript set)
  ex4_temporal.pdf      one row x three columns: ACF over the decades,
                        MSD, increment ACF -- exact + all trained cases
                        (torch; needs out/*/model.pt; not currently in
                        the manuscript)
  ex4_conditionals.pdf  conditional distributions at three instants
                        spanning the timescales (torch; not currently
                        in the manuscript)

  The remaining manuscript figures are ex4_compare.pdf (evaluate.py)
  and ex4_ensemble.pdf (ensemble_stats.py; three panels -- the W2
  metric is its panel (c)).

    python make_figs.py           # torch figures are skipped with a
                                  # message if checkpoints are absent
"""

import json
import os

import numpy as np

import config as cfg
import multiscale_exact as exact
from memdiff.features import acf_estimate, rates_from_band
from rank_analysis import selected_bank

TAUS = (2.0, 10.0, 30.0)      # all simulations until tau = 30


def fig_memory(plt, MODEL_ALPHA, band, k_sel):
    """The combined 'what k means' manuscript figure, one row:

    (a) ideal closed-loop ACF error vs conditioning dimension -- raw
        window AR(M) against the EMA bank AT THE SELECTED BAND, so the
        selected-k star lies ON the bank curve by construction;
    (b) Predictive spectrum (rank <= n_hidden);
    (c) worst-horizon loss of the rank-p projection.
    """
    from rank_analysis import spectrum_loss
    n_show = cfg.EVAL_MAX_LAG          # shoulder rule, tau <= 30
    C = exact.acf_x(n_show)
    dims_w, errs_w = [], []
    for M in (3, 5, 9, 17, 33, 65, 129):   # M >= 150 is exact here
        Ce = exact.ar_ideal_acf(M, n_show, C=exact.acf_x(max(n_show, M + 1)))
        dims_w.append(M)
        errs_w.append(np.max(np.abs(Ce - C)) / C[0])
    ks = sorted({2, 3, 4, 5, 8, 10} | {k_sel})
    dims_b, errs_b = [], []
    for kk in ks:                      # selected band throughout: the
        rr = rates_from_band(band, kk)  # star is a point of this curve
        dims_b.append(kk + 1)
        errs_b.append(np.max(np.abs(exact.bank_ideal_acf(rr, n_show) - C))
                      / C[0])
    e_sel = errs_b[ks.index(k_sel)]

    sv, eps_p, n_hid = spectrum_loss()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    ax[0].loglog(dims_w, errs_w, "o-", color="tab:orange",
                 alpha=MODEL_ALPHA, label="raw AR($M$)")
    ax[0].loglog(dims_b, errs_b, "s-", color="tab:red", alpha=MODEL_ALPHA,
                 label="EMA bank")
    ax[0].plot([k_sel + 1], [e_sel], "r*", ms=14,
               label=f"selected $k={k_sel}$")
    ax[0].set(xlabel="conditioning dimension",
              ylabel="ideal closed-loop\nACF error",
              title="(a) Representation error")
    ax[0].legend(fontsize=7)

    FLOOR = 1e-9
    j = np.arange(1, len(sv) + 1)
    ax[1].semilogy(j, np.maximum(sv, FLOOR), "ko-")
    ax[1].axhline(FLOOR, color="k", ls=":", lw=1)
    ax[1].axvline(n_hid + 0.5, color="tab:red", ls="--", lw=1)
    ax[1].text(n_hid + 0.42, sv[0] * 0.3, f"rank $\\leq {n_hid}$",
               fontsize=8, color="tab:red", ha="right")
    ax[1].set(xlabel="direction $j$", ylabel="singular value",
              title="(b) Predictive spectrum")
    ax[1].set_xticks(j)

    ax[2].semilogy(j, np.maximum(eps_p, FLOOR), "ko-")
    ax[2].axhline(FLOOR, color="k", ls=":", lw=1)
    ax[2].set(xlabel="kept dimensions $p$",
              ylabel=r"worst-horizon loss $\epsilon_p$",
              title="(c) Projection loss")
    ax[2].set_xticks(j)

    from memdiff.plotting import paper_typography
    paper_typography(fig, 0.92)
    fig.tight_layout()
    fig.savefig(f"{cfg.FIG_DIR}/ex4_memory.pdf")
    plt.close(fig)
    print(f"figure -> {cfg.FIG_DIR}/ex4_memory.pdf "
          f"(bank curve at the selected band; star on curve, "
          f"err {e_sel:.4f})")


def fig_temporal(plt, MODEL_ALPHA, evaluate, device):
    """One row x three columns over all available trained cases."""
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]
    n_lag = cfg.EVAL_MAX_LAG
    C_ex = exact.acf_x(n_lag)
    tau = np.arange(n_lag + 1) * cfg.DT
    RD_ex = exact.incr_acov(C_ex)

    R = evaluate.rollouts(X_test, device, seed=0)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))

    ax[0].semilogx(tau[1:], (C_ex / C_ex[0])[1:], "k-", lw=2.2,
                   label="exact")
    msd_ex = 2.0 * (C_ex[0] - C_ex)
    ax[1].loglog(tau[1:], msd_ex[1:], "k-", lw=2.2)
    n_d = 30
    taud = np.arange(n_d + 1) * cfg.DT
    ax[2].plot(taud, RD_ex[:n_d + 1] / RD_ex[0], "k-", lw=2.2)

    for name, label in evaluate.CASES:
        if name not in R:
            continue
        X = R[name]
        colr = evaluate.COLORS[name]
        acov = acf_estimate(X, n_lag)
        ax[0].semilogx(tau[1:], (acov / acov[0])[1:], "-", color=colr,
                       alpha=MODEL_ALPHA, label=label)
        ax[1].loglog(tau[1:], (2.0 * (acov[0] - acov))[1:], "-",
                     color=colr, alpha=MODEL_ALPHA)
        rd = acf_estimate(np.diff(X, axis=1), n_d)
        ax[2].plot(taud, rd / RD_ex[0], "-", color=colr,
                   alpha=MODEL_ALPHA)
    ax[0].set(xlabel=r"lag time $\tau$", ylabel=r"$C(\tau)/C(0)$",
              title="(a) autocovariance")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel=r"lag time $\tau$",
              ylabel=r"$\mathbb{E}[(X_{t+\tau}-X_t)^2]$",
              title="(b) mean-squared velocity change")
    ax[2].set(xlabel=r"lag time $\tau$", ylabel=r"$R_D(\tau)/R_D(0)$",
              title="(c) increment autocovariance")
    fig.tight_layout()
    fig.savefig(f"{cfg.FIG_DIR}/ex4_temporal.pdf")
    plt.close(fig)
    print(f"figure -> {cfg.FIG_DIR}/ex4_temporal.pdf")


def fig_conditionals(plt, MODEL_ALPHA, evaluate, device):
    """Conditional distributions at three instants spanning the decades,
    released from one held-out state + prehistory."""
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]
    net_m, ckpt_m = evaluate.load_variant("mem", device)
    rates = np.asarray(ckpt_m["rates"], float)
    m_all, n_burn = evaluate.memory_state(X_test, ckpt_m, "ema")
    sig = np.sqrt(exact.acf_x(1)[0])
    lo = max(n_burn, cfg.LSTM_WARM) + 1
    flat_x = X_test[:, lo:-1].reshape(-1)
    i_rel = int(np.argmin(np.abs(flat_x - 1.5 * sig)))
    it, jt = divmod(i_rel, X_test.shape[1] - 1 - lo)
    jt += lo
    x0 = X_test[it, jt]
    print(f"release: x0 = {x0:.4f} ({x0 / sig:.2f} sigma)")

    n_ens = 4000
    H = int(max(TAUS) / cfg.DT)
    A, Vh = exact.cond_horizon_given_memory(rates, H)
    v0 = np.concatenate([[x0], m_all[it, jt]])

    ens = {}
    for name, _ in evaluate.CASES:
        if name == "lstm":
            if evaluate.load_lstm(device)[0] is None:
                continue
            net, meta = evaluate.load_lstm(device)
            pre = X_test[it, jt - cfg.LSTM_WARM:jt + 1]
            from memdiff.lstm import ensemble_rollout
            ens[name] = ensemble_rollout(net, meta, pre, n_ens, H,
                                         cfg.SEED_EVAL + 600, device)
            continue
        if not os.path.exists(os.path.join(cfg.OUT_DIR, name, "model.pt")):
            continue
        net, ckpt = evaluate.load_variant(name, device)
        kind = "window" if name.startswith("window") else "ema"
        mm_all, _ = evaluate.memory_state(X_test, ckpt, kind)
        m0 = np.tile(mm_all[it, jt], (n_ens, 1))
        ens[name] = evaluate.rollout(net, ckpt["norm"], ckpt["rates"],
                                     ckpt["kappa_s"],
                                     np.full(n_ens, x0), m0, H,
                                     cfg.SEED_EVAL + 600, device,
                                     kind=kind)

    def kde(samples, grid, bw=None):
        s = np.asarray(samples, float)
        bw = bw or 1.06 * s.std() * s.size ** (-0.2)
        d = (grid[:, None] - s[None, :]) / bw
        return np.exp(-0.5 * d * d).sum(1) / (s.size * bw
                                              * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(1, len(TAUS), figsize=(10.5, 3.2))
    for j, tau_j in enumerate(TAUS):
        h = int(tau_j / cfg.DT)
        mu, sd = A[h - 1] @ v0, np.sqrt(Vh[h - 1])
        grid = np.linspace(mu - 4.5 * sd, mu + 4.5 * sd, 400)
        ax[j].plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2)
                   / (sd * np.sqrt(2 * np.pi)), "k-", lw=2.2,
                   label="exact conditional")
        for name, label in evaluate.CASES:
            if name not in ens:
                continue
            ax[j].plot(grid, kde(ens[name][:, h - 1], grid), "-",
                       color=evaluate.COLORS[name], alpha=MODEL_ALPHA,
                       label=label)
        ax[j].set(xlabel="$x$", title=f"$\\tau = {tau_j:g}$")
    ax[0].set(ylabel="conditional density")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{cfg.FIG_DIR}/ex4_conditionals.pdf")
    plt.close(fig)
    print(f"figure -> {cfg.FIG_DIR}/ex4_conditionals.pdf")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, MODEL_ALPHA
    setup_style()
    os.makedirs(cfg.FIG_DIR, exist_ok=True)

    band, k_sel, src = selected_bank()
    print(f"selected bank: k = {k_sel}, band = {np.round(band, 5)} [{src}]")
    fig_memory(plt, MODEL_ALPHA, band, k_sel)
    import sys
    if "--plot" in sys.argv:
        return

    try:
        import evaluate
    except ModuleNotFoundError as e:
        print(f"({e}; skipping the two trained-model figures)")
        return
    if not os.path.exists(os.path.join(cfg.OUT_DIR, "mem", "model.pt")):
        print("(no trained checkpoints under out/; skipping the two "
              "trained-model figures)")
        return
    device = "cuda" if evaluate.torch.cuda.is_available() else "cpu"
    fig_temporal(plt, MODEL_ALPHA, evaluate, device)
    fig_conditionals(plt, MODEL_ALPHA, evaluate, device)


if __name__ == "__main__":
    main()
