"""Stage-0 diagnostic: exact full-history conditional forecast fans.

The reference for EVERY model is the exact conditional law of X_{n+h}
given the observed history,

    p(X_{n+h} | X_{n-W}, ..., X_n),      W = FAN_HIST,

computed by an exact Kalman filter on the linear-Gaussian embedding
(the observation is the X component itself, noiseless). With W = 5
slow correlation times this is a numerically CONVERGED approximation
to infinite-history conditioning -- the startup check verifies that
its one-step forecast variance matches the converged full-history
predictive variance V_full. It is a common full-history ORACLE target
against which each model's compressed history representation is
evaluated; the models themselves condition on model-specific windows
(the EMA bank accumulated from the trajectory start, the LSTM warmed
on LSTM_WARM observed steps), each long relative to the slowest mode.
The exact conditional given the SELECTED BANK, (x_n, m_n), is shown as
a second, clearly labeled curve: the gap between the two references is
the memory-COMPRESSION floor of the bank, not a learning error.

Why this test: the closed-loop statistics of evaluate.py estimate the
slow tail from finite rollouts (noisy), whereas here every query has a
per-query exact mean and standard deviation at every horizon up to
tau = 2 T_max, so the comparison is paired and noise-free on the
REFERENCE side. (The model side keeps the Monte-Carlo error of a
finite ensemble: each model's sampling floor on the normalized
mean-RMSE is sigma_model(tau) / (sigma_exact(tau) sqrt(N)), i.e. the
panel-(c) spread ratio divided by sqrt(N). The figure draws the
NOMINAL floor for a calibrated model, 1/sqrt(FAN_N_ENS) ~ 0.032 --
accurate wherever the spread ratio is near one; values near the floor
are indistinguishable from it.)
A model whose training never supervised memory beyond its context (the
reset-state LSTM at LSTM_SEQ steps) shows up as conditional means
relaxing toward zero too early at slow-mode horizons, even when its
stationary statistics look competitive. The Gaussian output head is
correctly specified here, so any gap is attributable to MEMORY, not to
the output distribution.

Reported per case, as functions of the horizon:
  * conditional-mean RMSE over queries, in units of the exact
    conditional standard deviation;
  * median ratio of the ensemble standard deviation to the exact one.

Outputs: figs/ex4_fans.pdf and out/forecast_fans.json.
Needs trained checkpoints (torch, INFERENCE only -- zero retraining).
Generation is batched FAN_Q_BATCH queries at a time (all members of a
batch advance in one flattened net evaluation per step).

    python forecast_fans.py
"""

import json
import os

import numpy as np
import torch

import config as cfg
import multiscale_exact as exact
from evaluate import CASES, COLORS, load_cases, rollout
from memdiff.lstm import ensemble_rollout_multi

MARKS = (30.0, 100.0, 200.0, 400.0)     # report horizons, time units


def main(smoke=False):
    """smoke=True: minimal pass (2 queries, 32 members, short horizon,
    outputs suffixed _smoke) to catch CUDA/shape problems first."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = ""
    if smoke:
        cfg.FAN_N_QUERY, cfg.FAN_N_ENS, cfg.FAN_Q_BATCH = 2, 32, 2
        tag = "_smoke"
        print("SMOKE TEST: 2 queries, 32 members, horizon 100 steps")
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

    rng = np.random.default_rng(cfg.SEED_EVAL + 1300)
    ti = rng.integers(0, X_test.shape[0], cfg.FAN_N_QUERY)
    ni = rng.integers(lo, X_test.shape[1], cfg.FAN_N_QUERY)

    # ---- exact full-history reference (fair for every case) ------------
    hist = np.stack([X_test[a, b - cfg.FAN_HIST:b + 1]
                     for a, b in zip(ti, ni)])
    kal_mu, kal_sd = exact.kalman_forecast(hist, H)
    # filter sanity: the steady-state one-step forecast variance must
    # equal the converged full-history predictive variance
    _, Vinf = exact.v_full()
    rel = abs(kal_sd[0] ** 2 - Vinf) / Vinf
    print(f"filter check: one-step forecast var {kal_sd[0] ** 2:.6f} vs "
          f"V_full {Vinf:.6f}  (rel diff {rel:.1e})")
    if rel > 1e-3:
        raise FloatingPointError(
            "Kalman one-step variance disagrees with V_full; the "
            "prehistory window FAN_HIST may be too short")

    # ---- bank-conditional reference (compression floor, labeled) -------
    bank = None
    if "mem" in cases:
        rates = np.asarray(cases["mem"]["ckpt"]["rates"], float)
        A, Vh = exact.cond_horizon_given_memory(rates, H)
        v0 = np.concatenate([X_test[ti, ni][:, None],
                             cases["mem"]["m"][ti, ni]], axis=1)
        bank = {"mu": v0 @ A.T, "sd": np.sqrt(Vh)}

    # ---- model ensembles (batched: FAN_Q_BATCH queries per flattened
    # generation, so each time step is ONE net evaluation over
    # FAN_Q_BATCH * FAN_N_ENS members instead of one rollout per query)
    n_ens = cfg.FAN_N_ENS
    ens = {}
    for name, pl in cases.items():
        mu = np.empty((cfg.FAN_N_QUERY, H))
        sd = np.empty((cfg.FAN_N_QUERY, H))
        for g0 in range(0, cfg.FAN_N_QUERY, cfg.FAN_Q_BATCH):
            g1 = min(g0 + cfg.FAN_Q_BATCH, cfg.FAN_N_QUERY)
            seed_g = cfg.SEED_EVAL + 1400 + g0
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
        ens[name] = {"mu": mu, "sd": sd}
        print(f"  {name}: {cfg.FAN_N_QUERY} x {n_ens} ensembles done")

    # ---- metrics --------------------------------------------------------
    def curves(mu, sd):
        rmse = np.sqrt(np.mean(((mu - kal_mu) / kal_sd[None, :]) ** 2,
                               axis=0))
        ratio = np.median(np.broadcast_to(sd, kal_mu.shape)
                          / kal_sd[None, :], axis=0)
        return rmse, ratio

    mc_floor = 1.0 / np.sqrt(cfg.FAN_N_ENS)
    summary = {"config": {"n_query": cfg.FAN_N_QUERY,
                          "n_ens": cfg.FAN_N_ENS, "hist": cfg.FAN_HIST,
                          "q_batch": cfg.FAN_Q_BATCH,
                          "horizon_tu": float(H * cfg.DT)},
               "filter_onestep_var_reldiff": float(rel),
               # nominal floor for a CALIBRATED model (spread ratio ~ 1);
               # each model's exact floor is its sd_ratio / sqrt(n_ens).
               # mean-RMSE values near it are indistinguishable from the
               # finite-ensemble resolution, not a model error
               "ensemble_sampling_floor_nominal": float(mc_floor)}
    results = {}
    if bank is not None:
        results["bank_conditional"] = curves(bank["mu"], bank["sd"])
    for name in ens:
        results[name] = curves(ens[name]["mu"], ens[name]["sd"])
    for name, (rmse, ratio) in results.items():
        summary[name] = {}
        for t in MARKS:
            h = min(int(round(t / cfg.DT)), H) - 1
            summary[name][f"tau={t:g}"] = {
                "mean_rmse": float(rmse[h]), "sd_ratio": float(ratio[h])}
        line = "  ".join(f"tau={t:g}: rmse {summary[name][f'tau={t:g}']['mean_rmse']:.3f}"
                         f" ratio {summary[name][f'tau={t:g}']['sd_ratio']:.3f}"
                         for t in MARKS)
        print(f"  {name:16s} {line}")

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, MODEL_ALPHA
    setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    # (a) the query with the most slow-mode content at tau = T_max
    h_ref = min(int(round(max(cfg.FORCING_TIMES) / cfg.DT)), H) - 1
    q0 = int(np.argmax(np.abs(kal_mu[:, h_ref])))
    ax[0].fill_between(tauH, kal_mu[q0] - 2 * kal_sd,
                       kal_mu[q0] + 2 * kal_sd, color="k", alpha=0.12,
                       label=r"exact | full history, $\pm 2\sigma$")
    ax[0].plot(tauH, kal_mu[q0], "k-", lw=2.0)
    if bank is not None:
        ax[0].plot(tauH, bank["mu"][q0], ":", color="0.4", lw=1.4,
                   label="exact | bank (compression floor)")
    for name, label in CASES:
        if name not in ens:
            continue
        ax[0].plot(tauH, ens[name]["mu"][q0], "-", color=COLORS[name],
                   alpha=MODEL_ALPHA, label=label)
    ax[0].set(xlabel=r"forecast horizon $\tau$", ylabel="$x$",
              title="(a) one prehistory, conditional mean")
    ax[0].legend(fontsize=6)

    # (b) conditional-mean RMSE / (c) spread ratio, over queries
    for name, label in CASES:
        if name not in results:
            continue
        rmse, ratio = results[name]
        ax[1].semilogx(tauH, rmse, "-", color=COLORS[name],
                       alpha=MODEL_ALPHA, label=label)
        ax[2].semilogx(tauH, ratio, "-", color=COLORS[name],
                       alpha=MODEL_ALPHA, label=label)
    if "bank_conditional" in results:
        rmse, ratio = results["bank_conditional"]
        ax[1].semilogx(tauH, rmse, ":", color="0.4", lw=1.4,
                       label="compression floor")
        ax[2].semilogx(tauH, ratio, ":", color="0.4", lw=1.4)
    # nominal floor for a CALIBRATED model; the exact per-model floor is
    # the panel-(c) spread ratio divided by sqrt(N)
    ax[1].axhline(mc_floor, color="k", ls="--", lw=0.8,
                  label=r"nominal sampling floor $1/\sqrt{N}$")
    ax[1].set(xlabel=r"forecast horizon $\tau$",
              ylabel=r"mean RMSE $/\ \sigma_{\mathrm{exact}}$",
              title="(b) conditional-mean error")
    ax[1].legend(fontsize=6)
    ax[2].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[2].set(xlabel=r"forecast horizon $\tau$",
              ylabel=r"sd ratio (model / exact)",
              title="(c) conditional-spread calibration")

    fig.tight_layout()
    out = f"{cfg.FIG_DIR}/ex4_fans{tag}.pdf"
    fig.savefig(out)
    summary["curves"] = {name: {"tau": tauH.tolist(),
                                "mean_rmse": r.tolist(),
                                "sd_ratio": s.tolist()}
                         for name, (r, s) in results.items()}
    with open(os.path.join(cfg.OUT_DIR, f"forecast_fans{tag}.json"),
              "w") as f:
        json.dump(summary, f, indent=2)
    print(f"figure -> {out} ; "
          f"summary -> {cfg.OUT_DIR}/forecast_fans{tag}.json")


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv[1:])
