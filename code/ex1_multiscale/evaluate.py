"""Evaluation for Example 4: seed-banded closed-loop comparison.

Cases (whichever checkpoints exist under out/):
  markov, window (last k states, dim k+1), window_long (dim
  WINDOW_LONG_M + 1), lstm, mem (ours) -- plus CLOSED-FORM references:
  the ideal bank chain, the equal-dimension ideal window AR(k+1), the
  ideal long window AR(WINDOW_LONG_M + 1), and the exact ACF/MSD.

All cases share the same test snapshots and starting times per seed
(paired initialization), drawn to satisfy the largest warm-up
requirement among them, so no case gets easier initial conditions.

Stage-0 instrumentation (zero RETRAINING -- rollouts are regenerated,
but fingerprint-cached under out/cache/ so re-evaluation is cheap):
  * all N_SEED rollout seeds are actually generated and every metric is
    reported as median [min, max] over seeds (previously the docstring
    promised seed bands while only seed 0 was run);
  * statistics extend to tau = 2 T_max (EVAL_LAG_FULL = 400 t.u.), so
    the slowest 200 t.u. mode is actually tested; the tau <= 30
    shoulder max-error is kept for the window/bank comparison;
  * E_slow: integrated slow-tail autocovariance error over
    tau in SLOW_BAND x T_max (integral, not max: stable in the noisy
    tail), reported both raw and shape-normalized;
  * per-query exact forecast fans live in forecast_fans.py.

One figure (figs/ex4_compare.pdf) and out/summary.json.

    python evaluate.py
"""

import json
import os
import sys

import numpy as np

import config as cfg
import multiscale_exact as exact
from conditioning import build_window_tuples          # noqa: F401
from memdiff.cache import cached, fingerprint, state_fingerprint
from memdiff.features import acf_estimate, ema_features, msd

# Plot-only regeneration uses NumPy caches and never loads checkpoints.
if "--plot" not in sys.argv:
    import torch
    from memdiff.lstm import LSTMGaussian
    from memdiff.lstm import rollout as lstm_rollout
    from train import FlowMapNet

CASES = [("markov", "no memory"),
         ("window", "raw last-$k$ history"),
         ("window_long", f"raw last-{cfg.WINDOW_LONG_M} history"),
         ("lstm", "LSTM + Gaussian head"),
         ("mem", "memory bank (ours)")]
# short display names for tick labels (never expose checkpoint names
# in a manuscript figure)
SHORT = {"markov": "no memory", "window": "raw window",
         "window_long": "long window", "lstm": "LSTM",
         "mem": "memory bank"}
COLORS = {"markov": "tab:green", "window": "tab:orange",
          "window_long": "tab:brown", "lstm": "tab:purple",
          "mem": "tab:red"}


# ---------------------------------------------------------------- loading

def load_variant(name, device):
    ckpt = torch.load(os.path.join(cfg.OUT_DIR, name, "model.pt"),
                      map_location=device, weights_only=False)
    net = FlowMapNet(1 + ckpt["k"], 1, cfg.HIDDEN, cfg.N_LAYERS).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net, ckpt


def load_lstm(device):
    p = os.path.join(cfg.OUT_DIR, "lstm", "model.pt")
    if not os.path.exists(p):
        return None, None
    ck = torch.load(p, map_location=device, weights_only=False)
    meta = ck["meta"]
    net = LSTMGaussian(1, meta["hidden"], meta["layers"],
                       meta.get("n_mix", 1)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, meta


def net_sample(net, norm, C, z, device):
    X = np.hstack([C, z]).astype(np.float32)
    Xn = torch.tensor((X - norm["x_mu"]) / norm["x_sd"], device=device)
    with torch.no_grad():
        Y = net(Xn).cpu().numpy()
    return Y * norm["y_sd"] + norm["y_mu"]


def memory_state(X, ckpt, kind):
    """(m, n_burn): EMA features, or the raw lag stack for windows."""
    rates = ckpt["rates"]
    if kind != "window":
        return ema_features(X, rates, cfg.DT, cfg.BURN_TOL)
    k = len(rates)
    n_traj, L1 = X.shape
    m = np.zeros((n_traj, L1, k))
    for l in range(k):
        m[:, l + 1:, l] = X[:, :L1 - (l + 1)]
    return m, k


def rollout(net, norm, rates, kappa_s, x0, m0, n_steps, seed, device,
            kind="ema"):
    rng = np.random.default_rng(seed)
    rho = np.exp(-np.asarray(rates) * cfg.DT)
    k = len(rho)
    x = x0.copy()
    m = m0.copy()
    X = np.empty((len(x0), n_steps))
    for n in range(n_steps):
        z = rng.standard_normal((len(x0), 1))
        C = np.concatenate([x[:, None], m], axis=1)
        s = net_sample(net, norm, C, z, device)
        d = s[:, 0] / kappa_s
        x_old = x
        x = x + d
        if kind == "window":
            m = np.concatenate([x_old[:, None], m[:, :-1]], axis=1)
        elif k:
            m = rho[None, :] * m + d[:, None]
        X[:, n] = x
    return X


def load_cases(X_test, device):
    """Load every available case ONCE (models, memory states, warm-ups).

    Returns (cases, lo): cases maps name -> payload dict; lo is the
    smallest admissible starting index, satisfying the largest warm-up
    requirement among the available cases (paired initialization).
    """
    cases, warm = {}, []
    for name, _ in CASES:
        if name == "lstm":
            net, meta = load_lstm(device)
            if net is None:
                continue
            cases[name] = {"kind": "lstm", "net": net, "meta": meta}
            warm.append(cfg.LSTM_WARM)
            continue
        if not os.path.exists(os.path.join(cfg.OUT_DIR, name, "model.pt")):
            continue
        net, ckpt = load_variant(name, device)
        kind = "window" if name.startswith("window") else "ema"
        m_test, n_burn = memory_state(X_test, ckpt, kind)
        cases[name] = {"kind": kind, "net": net, "ckpt": ckpt, "m": m_test}
        warm.append(n_burn)
    lo = (max(warm) + 1) if warm else None
    return cases, lo


DISCARD = 500          # rollout transient discarded before any statistic


def provenance(cases):
    """Best-effort reproducibility record for summary.json: selection,
    label-stage diagnostics, parameter counts, seeds, recorded wall
    times, and the one-step accuracy record when it exists.  Everything
    is read from artifacts already on disk -- nothing is recomputed."""
    prov = {"seeds": {"data": cfg.SEED_DATA, "labels": cfg.SEED_LABELS,
                      "train": cfg.SEED_TRAIN, "eval": cfg.SEED_EVAL,
                      "lstm": cfg.SEED_LSTM},
            "n_labels": cfg.N_LABELS}
    for name, pl in cases.items():
        if pl["kind"] == "lstm":
            meta = {k: v for k, v in pl["meta"].items() if k != "x_sd"}
            meta["n_params"] = int(sum(p.numel()
                                       for p in pl["net"].parameters()))
            meta.setdefault("train_seed", cfg.SEED_LSTM)
            prov["lstm"] = meta
            continue
        ck = pl["ckpt"]
        rec = {"k": int(ck["k"]),
               "band": (None if ck.get("band") is None
                        else [float(b) for b in ck["band"]]),
               "kappa_s": float(ck["kappa_s"]),
               "sampler": ck.get("sampler"),
               "n_params": int(sum(v.numel()
                                   for v in ck["state"].values()))}
        for key in ("wall_labels_s", "wall_distill_s"):
            if key in ck:                # recorded by newer pipeline runs
                rec[key] = float(ck[key])
        lab = os.path.join(cfg.OUT_DIR, name, "labels.npz")
        if os.path.exists(lab):
            d = np.load(lab)
            if "ess" in d:
                rec["label_ess"] = {
                    "median": float(np.median(d["ess"])),
                    "p05": float(np.percentile(d["ess"], 5))}
            if "radius" in d:
                rec["label_radius_median"] = float(np.median(d["radius"]))
        prov[name] = rec
    p1 = os.path.join(cfg.OUT_DIR, "onestep.json")
    if os.path.exists(p1):
        with open(p1) as f:
            prov["onestep"] = json.load(f)
    else:
        prov["onestep"] = "pending: python run_all.py onestep"
    return prov


def rollouts(X_test, device, seed, cases=None, lo=None, use_cache=True):
    """Closed-loop trajectories for one seed, PAIRED across cases, and
    fingerprint-cached. The key hashes everything the rollout depends
    on: the trained weights, the ACTUAL initial states and memory
    states (which cover the data set and the burn-in), the
    normalization, rates and kappa_s (or the LSTM meta and warm-up),
    the model-noise seed, and the transient discard -- so a regenerated
    data set or retrained checkpoint recomputes instead of serving a
    stale rollout."""
    if cases is None:
        cases, lo = load_cases(X_test, device)
    if not cases:
        return {}
    rng = np.random.default_rng(cfg.SEED_EVAL + 900 + seed)
    ti = rng.integers(0, X_test.shape[0], cfg.N_GEN_TRAJ)
    ni = rng.integers(lo, X_test.shape[1] - 1, cfg.N_GEN_TRAJ)
    noise_seed = cfg.SEED_EVAL + 700 + seed

    out = {}
    for name, pl in cases.items():
        if pl["kind"] == "lstm":
            state = pl["net"].state_dict()
            pre = np.stack([X_test[a, b - cfg.LSTM_WARM:b + 1]
                            for a, b in zip(ti, ni)])

            def compute(pl=pl, pre=pre):
                return lstm_rollout(pl["net"], pl["meta"], pre, cfg.L_GEN,
                                    noise_seed, device)[:, DISCARD:]
            deps = (pre, pl["meta"], cfg.LSTM_WARM)
        else:
            state = pl["ckpt"]["state"]
            x0, m0 = X_test[ti, ni], pl["m"][ti, ni]

            def compute(pl=pl, x0=x0, m0=m0):
                return rollout(pl["net"], pl["ckpt"]["norm"],
                               pl["ckpt"]["rates"], pl["ckpt"]["kappa_s"],
                               x0, m0, cfg.L_GEN, noise_seed, device,
                               kind=pl["kind"])[:, DISCARD:]
            deps = (x0, m0, pl["ckpt"]["norm"],
                    np.asarray(pl["ckpt"]["rates"], float),
                    float(pl["ckpt"]["kappa_s"]), pl["kind"])
        # cfg.DT enters deployment through rho = exp(-rates * DT); the
        # device is included because CPU and GPU runs are seeded alike
        # yet not bit-identical
        key = fingerprint(state_fingerprint(state), noise_seed,
                          cfg.L_GEN, DISCARD, cfg.DT, str(device),
                          ti, ni, deps)
        path = os.path.join(cfg.OUT_DIR, "cache", f"roll_{name}_s{seed}.npz")
        out[name] = cached(path, key, compute) if use_cache else compute()
    return out


# ------------------------------------------------------------------ main

def main(smoke=False):
    """smoke=True: minimal pass (1 seed, 4 short rollouts, cache off,
    outputs suffixed _smoke) to catch CUDA/shape/device problems before
    the full run. Restore nothing afterwards -- config is untouched on
    disk and the real outputs are not overwritten."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = ""
    if smoke:
        cfg.N_SEED, cfg.L_GEN, cfg.N_GEN_TRAJ = 1, 3000, 4
        tag = "_smoke"
        print("SMOKE TEST: N_SEED=1, L_GEN=3000, N_GEN_TRAJ=4, no cache")
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]

    n_sh = cfg.EVAL_MAX_LAG          # shoulder max-error: tau <= 30
    n_full = cfg.EVAL_LAG_FULL       # long statistics: tau <= 2 T_max
    Tmax = max(cfg.FORCING_TIMES)
    l_lo = int(round(cfg.SLOW_BAND[0] * Tmax / cfg.DT))
    l_hi = min(int(round(cfg.SLOW_BAND[1] * Tmax / cfg.DT)), n_full)
    C_ex = exact.acf_x(n_full)
    tau = np.arange(n_full + 1) * cfg.DT
    msd_ex = 2.0 * (C_ex[0] - C_ex)
    keep = msd_ex > 0
    den_slow = float(np.sum(np.abs(C_ex[l_lo:l_hi + 1])))

    def e_slow(acov):
        """Integrated slow-tail autocovariance error (codeX metric)."""
        return float(np.sum(np.abs(acov[l_lo:l_hi + 1]
                                   - C_ex[l_lo:l_hi + 1])) / den_slow)

    def stats(X):
        acov = acf_estimate(X, n_full)
        m = msd(X[:, :, None], n_full)
        met = {
            "var_relerr": float(abs(X.var() - C_ex[0]) / C_ex[0]),
            "acf_shape_maxerr": float(np.max(np.abs(
                acov[1:n_sh + 1] / acov[0]
                - C_ex[1:n_sh + 1] / C_ex[0]))),
            "E_slow": e_slow(acov),
            # shape-normalized variant: removes the variance error so it
            # measures the slow correlation SHAPE alone
            "E_slow_shape": e_slow(acov / acov[0] * C_ex[0]),
            "msd_maxrelerr": float(np.max(np.abs(m[keep] - msd_ex[keep])
                                          / msd_ex[keep])),
        }
        return met, acov

    # closed-form references at the selected bank
    mem_ckpt = torch.load(os.path.join(cfg.OUT_DIR, "mem", "model.pt"),
                          map_location="cpu", weights_only=False)
    rates = np.asarray(mem_ckpt["rates"], float)
    k_sel = len(rates)
    C_bank = exact.bank_ideal_acf(rates, n_full)
    # the learned window at the bank's dimension conditions on k+1 states,
    # so its ideal reference is the population AR(k+1); likewise the long
    # window (WINDOW_LONG_M lags) corresponds to AR(WINDOW_LONG_M + 1)
    C_arw = exact.ar_ideal_acf(k_sel + 1, n_full,
                               C=exact.acf_x(max(n_full, k_sel + 2)))
    Ml = cfg.WINDOW_LONG_M + 1
    C_arl = exact.ar_ideal_acf(Ml, n_full, C=exact.acf_x(max(n_full, Ml + 1)))
    ideal = {}
    for ref_name, Ci in (("bank", C_bank), ("window_same_dim", C_arw),
                         ("window_long", C_arl)):
        ideal[ref_name] = {
            "acf_maxerr_shoulder": float(np.max(np.abs(
                Ci[:n_sh + 1] - C_ex[:n_sh + 1])) / C_ex[0]),
            "E_slow": e_slow(Ci),
        }
    # measured floor under the SAME protocol as the models: N_SEED ideal
    # rollouts, median + band -- a single seed could distort the floor
    floor_rows = []
    for seed in range(cfg.N_SEED):
        Xi = exact.ideal_rollout(rates, cfg.N_GEN_TRAJ, cfg.L_GEN,
                                 seed=seed)[:, DISCARD:]
        floor_rows.append(stats(Xi)[0])
    floor = {k: {"median": float(np.median([r[k] for r in floor_rows])),
                 "band": [float(np.min([r[k] for r in floor_rows])),
                          float(np.max([r[k] for r in floor_rows]))]}
             for k in floor_rows[0]}
    print(f"closed form at k = {k_sel}: ideal bank E_slow "
          f"{ideal['bank']['E_slow']:.3f}, ideal window at the same "
          f"dimension (AR({k_sel + 1})) {ideal['window_same_dim']['E_slow']:.3f}, "
          f"ideal long window (AR({Ml})) {ideal['window_long']['E_slow']:.3f}")
    print(f"measured ideal floor ({cfg.N_SEED} seeds): shoulder "
          f"{floor['acf_shape_maxerr']['median']:.3f}, E_slow "
          f"{floor['E_slow']['median']:.3f} "
          f"[{floor['E_slow']['band'][0]:.3f}, "
          f"{floor['E_slow']['band'][1]:.3f}] "
          f"(finite-sample ACF noise included)")

    cases, lo = load_cases(X_test, device)
    per_seed, acovs = {}, {}
    for seed in range(cfg.N_SEED):
        print(f"rolling out every available case, seed {seed} ...")
        R = rollouts(X_test, device, seed, cases=cases, lo=lo,
                     use_cache=not smoke)
        for name, X in R.items():
            met, acov = stats(X)
            per_seed.setdefault(name, []).append(met)
            acovs.setdefault(name, []).append(acov)

    summary = {"ideal_closed_form": ideal, "metric_floor_ideal": floor,
               "k_selected": int(k_sel),
               "eval": {"n_seed": cfg.N_SEED, "lag_full": n_full,
                        "shoulder_lag": n_sh,
                        "slow_band_tu": [l_lo * cfg.DT, l_hi * cfg.DT],
                        "band_type":
                            "rollout seeds, single trained checkpoint"},
               "provenance": provenance(cases)}
    for name, _ in CASES:
        if name not in per_seed:
            continue
        rows = per_seed[name]
        summary[name] = {k: {
            "median": float(np.median([r[k] for r in rows])),
            "band": [float(np.min([r[k] for r in rows])),
                     float(np.max([r[k] for r in rows]))]}
            for k in rows[0]}
        s = summary[name]

        def fmt(k):
            return (f"{s[k]['median']:.4f} "
                    f"[{s[k]['band'][0]:.4f}, {s[k]['band'][1]:.4f}]")
        print(f"  {name:12s} E_slow {fmt('E_slow')}   "
              f"shoulder {fmt('acf_shape_maxerr')}   "
              f"MSD {fmt('msd_maxrelerr')}   var {fmt('var_relerr')}")

    # ------------------------------------------------------------- figure
    out = plot_comparison(summary, acovs, C_ex, C_bank, tau, l_lo, l_hi, msd_ex, floor, ideal, tag)
    with open(os.path.join(cfg.OUT_DIR, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"figure -> {out} ; summary -> {cfg.OUT_DIR}/summary{tag}.json")


def plot_comparison(summary, acovs, C_ex, C_bank, tau, l_lo, l_hi,
                    msd_ex, floor, ideal, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, MODEL_ALPHA, BAND_ALPHA
    setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    ax[0].axvspan(l_lo * cfg.DT, l_hi * cfg.DT, color="0.93", zorder=0)
    ax[0].semilogx(tau[1:], (C_ex / C_ex[0])[1:], "k-", lw=2.0,
                   label="exact", zorder=5)
    ax[0].semilogx(tau[1:], (C_bank / C_ex[0])[1:], "--", color="0.45",
                   lw=1.2, label="ideal bank (closed form)", zorder=4)
    for name, label in CASES:
        if name not in acovs:
            continue
        A = np.array(acovs[name])
        sh = A / A[:, :1]
        ax[0].semilogx(tau[1:], sh.mean(0)[1:], "-", color=COLORS[name],
                       alpha=MODEL_ALPHA, label=label)
        ax[0].fill_between(tau[1:], sh.min(0)[1:], sh.max(0)[1:],
                           color=COLORS[name], alpha=BAND_ALPHA)
    ax[0].set(xlabel=r"lag time $\tau$", ylabel="normalized autocovariance",
              title="(a) autocovariance to $2\\,T_{\\max}$")
    ax[0].legend(fontsize=7)

    ax[1].loglog(tau[1:], msd_ex[1:], "k-", lw=2.0, label="exact", zorder=5)
    for name, _ in CASES:
        if name not in acovs:
            continue
        A = np.array(acovs[name]).mean(0)
        ax[1].loglog(tau[1:], (2.0 * (A[0] - A))[1:], "-",
                     color=COLORS[name], alpha=MODEL_ALPHA)
    ax[1].set(xlabel=r"lag time $\tau$",
              ylabel=r"$\mathbb{E}[(X_{t+\tau}-X_t)^2]$",
              title="(b) mean-squared velocity change")

    names = [n for n, _ in CASES if n in summary]
    med = [summary[n]["E_slow"]["median"] for n in names]
    err = np.array([[summary[n]["E_slow"]["median"]
                     - summary[n]["E_slow"]["band"][0] for n in names],
                    [summary[n]["E_slow"]["band"][1]
                     - summary[n]["E_slow"]["median"] for n in names]])
    ax[2].bar(range(len(names)), med, yerr=err, capsize=3,
              color=[COLORS[n] for n in names], alpha=MODEL_ALPHA)
    ax[2].axhspan(floor["E_slow"]["band"][0], floor["E_slow"]["band"][1],
                  color="0.85", zorder=0)
    ax[2].axhline(floor["E_slow"]["median"], color="k", ls="--", lw=1,
                  label="measured ideal floor (seed band)")
    ax[2].axhline(ideal["bank"]["E_slow"], color="0.45", ls=":", lw=1,
                  label="ideal bank (closed form)")
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(names)))
    ax[2].set_xticklabels([SHORT.get(n, n) for n in names],
                          rotation=20, fontsize=8)
    ax[2].set(ylabel=r"$E_{\mathrm{slow}}$",
              title="(c) slow-tail error (median, seed band)")
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    out = f"{cfg.FIG_DIR}/ex4_compare{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_saved():
    """Rebuild the manuscript comparison from saved metrics and rollouts."""
    from pathlib import Path
    with open(os.path.join(cfg.OUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(cfg.OUT_DIR, "mem", "selection.json")) as f:
        sel = json.load(f)
    from memdiff.features import rates_from_band
    rates = rates_from_band(sel["band"], sel["k"])
    C_ex = exact.acf_x(cfg.EVAL_LAG_FULL)
    C_bank = exact.bank_ideal_acf(rates, cfg.EVAL_LAG_FULL)
    acovs = {}
    for name, _ in CASES:
        if name not in summary:
            continue
        paths = sorted((Path(cfg.OUT_DIR) / "cache").glob(f"roll_{name}_s*.npz"))
        if len(paths) != cfg.N_SEED:
            raise SystemExit(f"Expected {cfg.N_SEED} saved rollouts for {name}, found {len(paths)}")
        acovs[name] = [acf_estimate(np.load(p)["value"], cfg.EVAL_LAG_FULL) for p in paths]
    tau = np.arange(cfg.EVAL_LAG_FULL + 1) * cfg.DT
    lo, hi = [int(round(v * max(cfg.FORCING_TIMES) / cfg.DT)) for v in cfg.SLOW_BAND]
    plot_comparison(summary, acovs, C_ex, C_bank, tau, lo, hi,
                    2 * (C_ex[0] - C_ex), summary["metric_floor_ideal"],
                    summary["ideal_closed_form"])


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        plot_saved()
    else:
        main(smoke="--smoke" in sys.argv[1:])
