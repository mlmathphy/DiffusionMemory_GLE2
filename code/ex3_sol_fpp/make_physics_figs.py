"""Physics-facing post-processing figures for Example 3 (numpy only).

Reads ONLY out/data.npz (held-out records) and the cached closed-loop
rollouts written by evaluate.py (out/cache/roll_ours_s*.npz and
roll_lstm<seed>_s*.npz, LSTM seed = config.LSTM_SEEDS[0]). No retraining,
no particle filter, no additional benchmark. Fairness rules (codeX):
predeclared trajectory/segment indices; COMMON EXPOSURE -- all three
sources are truncated to the same number of trajectories and time samples
before any statistic (the extra generated samples are used only with
`--all-samples`, a sensitivity run whose outputs carry the suffix
_allsamples); identical axes, bins, thresholds and color ranges;
trajectory-level block bootstrap; hidden pulse components never shown;
one LSTM instance.

Burst events: each contiguous excursion above mu + BURST_SIGMA sd is one
event (peak = its maximum, duration = excursion length); excursions whose
peaks are closer than MIN_SEP_T are merged into one event. Waiting times,
durations and amplitudes therefore refer to the same uniquely defined
events. The conditional waveform window is +-HALF_W_T = 20 tau_d1 (>= 2
tau_d2); a rise/decay half-excess time that does not cross inside the
window is reported as CENSORED at the window, not as a value.

  figs/ex3_bursts.pdf      (a) conditionally averaged burst waveform with
                           bootstrap band, (b) peak-amplitude exceedance
                           P(Y_peak > q) with the amplitude axis limited to
                           AMP_AXIS_MAX for readability (the per-event q99
                           beyond it is reported in out/physics_metrics.json
                           only; the manuscript table's extreme-event entry
                           is a time-sample fraction, not a peak statistic;
                           a merged event's peak is the largest member peak)
  out/physics_metrics.json burst rate, duty cycle, mean waiting time + CV,
                           conditional rise/decay times (with censoring
                           flags), peak-amplitude quantiles, event-envelope
                           duration, extreme-event probability -- each with
                           a 95% trajectory-bootstrap CI
  figs/ex3_phase.pdf       stochastic phase portrait: log joint density of
                           (y_n, dy_{n+1}/dt), global color scale,
                           conditional mean and 10-90% bands (in-range
                           points only), burst threshold

    python make_physics_figs.py [--smoke] [--all-samples] [--plot] [--bursts-only]

Use --plot --bursts-only to regenerate the manuscript burst figure from
cached trajectories without writing metrics or the phase-portrait figure.
"""

import glob
import json
import os
import sys

import numpy as np

import config as cfg
import model as md
from conditioning import ema_features

# ---------------- predeclared choices (fixed before any figure is made) --
N_BOOT = 200                 # trajectory-level bootstrap replicates
MIN_SEP_T = 2.0              # excursions with peaks closer than this merge
HALF_W_T = 20.0              # half window of the conditional average (t.u.)
EXTREME_SIGMA = 5.0          # extreme-event threshold mu + 5 sd
AMP_AXIS_MAX = 12.5          # panel (b) amplitude axis (tail in the metrics)
COL = {"truth": "k", "ours": "C3", "lstm": "C0"}
LAB = {"truth": "held-out truth", "ours": "memory-conditioned diffusion",
       "lstm": "LSTM + Gaussian head"}
NAMES = ("truth", "ours", "lstm")


# ----------------------------------------------------------------- loading
def load_sources(all_samples):
    data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
    X_test = data["X_test"]
    seed = cfg.LSTM_SEEDS[0]
    out = {}
    for name, pat in (("ours", "roll_ours_s*.npz"),
                      ("lstm", f"roll_lstm{seed}_s*.npz")):
        arrs = []
        for p in sorted(glob.glob(os.path.join(cfg.OUT_DIR, "cache", pat))):
            with np.load(p, allow_pickle=False) as z:
                arrs.append(z["value"][:, cfg.DISCARD:])
        if not arrs:
            sys.exit(f"no cached rollouts for {name} (run evaluate.py first)")
        out[name] = np.concatenate(arrs, 0)
    rates = None
    try:
        with open(os.path.join(cfg.OUT_DIR, "mem", "selection.json")) as f:
            rates = np.array(json.load(f)["rates"])
    except Exception:
        pass
    n_burn = ema_features(X_test, rates)[1] if rates is not None else 0
    lo = max(n_burn, cfg.LSTM_WARM, cfg.PF_WARM) + 1
    out["truth"] = X_test[:, lo:]
    # common exposure (predeclared): same number of trajectories and samples
    n_common = min(v.shape[0] for v in out.values())
    t_common = min(v.shape[1] for v in out.values())
    exposure = dict(n_common=int(n_common), t_common=int(t_common),
                    raw={k: list(v.shape) for k, v in out.items()},
                    all_samples=bool(all_samples))
    if not all_samples:
        out = {k: v[:n_common, :t_common] for k, v in out.items()}
    return out, exposure


# ------------------------------------------------------------ burst tools
def excursion_events(x, thr, min_sep):
    """Contiguous excursions above thr -> events (start, end, peak index,
    peak value); excursions whose peaks are closer than min_sep samples are
    merged (peak = larger maximum, span = union)."""
    above = x > thr
    if not above.any():
        return []
    d = np.diff(above.astype(int))
    starts = list(np.nonzero(d == 1)[0] + 1)
    ends = list(np.nonzero(d == -1)[0])
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(x) - 1]
    ev = []
    for s, e in zip(starts, ends):
        i = s + int(np.argmax(x[s:e + 1]))
        ev.append([s, e, i, float(x[i])])
    # merge chains of nearby excursions: closeness is tested against the
    # MOST RECENT original peak (so a chain 0, 3, 6 with min_sep 4 merges
    # fully), while the event's representative peak stays the largest; the
    # merged span (first start .. last end) is the EVENT-ENVELOPE duration,
    # which includes short below-threshold gaps between merged excursions
    merged = [ev[0]]
    last_peak = ev[0][2]
    for s, e, i, v in ev[1:]:
        ps, pe, pi, pv = merged[-1]
        if i - last_peak < min_sep:
            merged[-1] = [ps, e, i if v > pv else pi, max(v, pv)]
        else:
            merged.append([s, e, i, v])
        last_peak = i
    return merged


def burst_events(X, thr, thr_x, min_sep, half_w):
    """Per trajectory: peak indices, amplitudes, waiting times, durations,
    waveforms (peaks with a full +-half_w window), duty cycle, extreme prob."""
    out = []
    for x in X:
        ev = excursion_events(x, thr, min_sep)
        pk = np.array([e[2] for e in ev], int)
        amp = np.array([e[3] for e in ev])
        dur = np.array([(e[1] - e[0] + 1) * cfg.DT for e in ev])
        ok = (pk >= half_w) & (pk < len(x) - half_w)
        wave = np.array([x[i - half_w:i + half_w + 1] for i in pk[ok]]) \
            if ok.any() else np.zeros((0, 2 * half_w + 1))
        out.append(dict(pk=pk, amp=amp, wait=np.diff(pk) * cfg.DT, dur=dur,
                        wave=wave, n=len(x), above=float((x > thr).mean()),
                        extreme=float((x > thr_x).mean())))
    return out


def half_times(wave, base):
    """Half-excess rise/decay times of the averaged waveform; each is
    (time, censored) with censored=True when the level is not crossed
    inside the window."""
    c = len(wave) // 2
    half = base + 0.5 * (wave[c] - base)
    r = c
    while r > 0 and wave[r - 1] > half:
        r -= 1
    d = c
    while d < len(wave) - 1 and wave[d + 1] > half:
        d += 1
    return ((c - r) * cfg.DT, r == 0), ((d - c) * cfg.DT, d == len(wave) - 1)


def metrics_from_events(ev, idx, base):
    sub = [ev[i] for i in idx]
    cat = lambda k: (np.concatenate([e[k] for e in sub])          # noqa: E731
                     if sub else np.array([]))
    amp, wait, dur = cat("amp"), cat("wait"), cat("dur")
    T = sum(e["n"] for e in sub) * cfg.DT
    waves = [e["wave"] for e in sub if len(e["wave"])]
    m = dict(burst_rate=len(amp) / T,
             duty_cycle=float(np.mean([e["above"] for e in sub])),
             wait_mean=float(wait.mean()) if len(wait) else np.nan,
             wait_cv=float(wait.std() / wait.mean()) if len(wait) else np.nan,
             amp_q50=float(np.quantile(amp, 0.5)) if len(amp) else np.nan,
             amp_q90=float(np.quantile(amp, 0.9)) if len(amp) else np.nan,
             amp_q99=float(np.quantile(amp, 0.99)) if len(amp) else np.nan,
             dur_mean=float(dur.mean()) if len(dur) else np.nan,
             extreme_prob=float(np.mean([e["extreme"] for e in sub])),
             rise_time=np.nan, decay_time=np.nan, rise_censored=False,
             decay_censored=False)
    if waves:
        (rt, rc), (dt_, dc) = half_times(np.concatenate(waves, 0).mean(0), base)
        m.update(rise_time=rt, decay_time=dt_, rise_censored=bool(rc),
                 decay_censored=bool(dc))
    return m


def bootstrap(ev, base, rng):
    n = len(ev)
    est = metrics_from_events(ev, np.arange(n), base)
    num = [k for k, v in est.items() if not isinstance(v, bool)]
    boots = {k: [] for k in num}
    for _ in range(N_BOOT):
        mb = metrics_from_events(ev, rng.integers(0, n, n), base)
        for k in num:
            boots[k].append(mb[k])
    out = {k: dict(value=est[k], ci=[float(np.nanpercentile(boots[k], 2.5)),
                                     float(np.nanpercentile(boots[k], 97.5))])
           for k in num}
    out["rise_censored"], out["decay_censored"] = est["rise_censored"], \
        est["decay_censored"]
    return out


def hdr_levels(H, masses=(0.5, 0.8, 0.95)):
    """Highest-density thresholds of a 2-D histogram enclosing the given
    probability masses; unique and increasing (for contour)."""
    v = np.sort(H.ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return np.array([])
    c = np.cumsum(v) / tot
    lv = [v[min(int(np.searchsorted(c, p)), len(v) - 1)] for p in masses]
    lv = np.unique([x for x in lv if x > 0])
    return np.sort(lv)


# ------------------------------------------------------------------ main
def main():
    if "--smoke" in sys.argv:
        cfg.apply_smoke()
    all_samples = "--all-samples" in sys.argv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from memdiff.plotting import setup_style, paper_typography
    setup_style()
    P = md.params()
    mu, sd = P["mean"], np.sqrt(P["var"])
    thr, thr_x = mu + cfg.BURST_SIGMA * sd, mu + EXTREME_SIGMA * sd
    src, exposure = load_sources(all_samples)
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    tag = cfg.TAG + ("_allsamples" if all_samples else "")
    rng = np.random.default_rng(cfg.SEED_EVAL + 123)
    half_w = int(round(HALF_W_T / cfg.DT))
    min_sep = int(round(MIN_SEP_T / cfg.DT))
    print(f"common exposure: {exposure['n_common']} trajectories x "
          f"{exposure['t_common']} samples (raw {exposure['raw']}; all_samples="
          f"{all_samples})")

    # ================= 1. burst physics + metrics ========================
    ev = {n: burst_events(src[n], thr, thr_x, min_sep, half_w) for n in NAMES}
    metrics = {n: bootstrap(ev[n], mu, rng) for n in NAMES}
    for n in NAMES:
        metrics[n]["n_trajectories"] = int(len(src[n]))
        metrics[n]["n_bursts"] = int(sum(len(e["amp"]) for e in ev[n]))
    metrics["definition"] = dict(threshold=float(thr), extreme=float(thr_x),
                                 mean=float(mu), sd=float(sd),
                                 min_sep_t=MIN_SEP_T, half_window_t=HALF_W_T,
                                 events="contiguous excursions above the "
                                 "threshold; chains of excursions whose "
                                 "consecutive peaks are closer than min_sep_t "
                                 "are merged (peak = largest maximum); duration "
                                 "= event-envelope duration (first start to "
                                 "last end, including short below-threshold "
                                 "gaps between merged excursions)")
    metrics["exposure"] = exposure
    if "--plot" not in sys.argv:
        with open(os.path.join(cfg.OUT_DIR, f"physics_metrics{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=1)

    # main figure: (a) waveform, (b) amplitude exceedance (user selection);
    # the waiting-time and level-exceedance panels go to ex3_bursts_extra
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9))
    tw = np.arange(-half_w, half_w + 1) * cfg.DT
    amp_all = np.concatenate([np.concatenate([e["amp"] for e in ev[n]])
                              for n in NAMES])
    amp_max = {n: float(np.concatenate([e["amp"] for e in ev[n]]).max())
               for n in NAMES}
    for n in NAMES:
        waves = np.concatenate([e["wave"] for e in ev[n] if len(e["wave"])], 0)
        bands = []
        for _ in range(N_BOOT // 4):
            idx = rng.integers(0, len(ev[n]), len(ev[n]))
            w = [ev[n][i]["wave"] for i in idx if len(ev[n][i]["wave"])]
            if w:
                bands.append(np.concatenate(w, 0).mean(0))
        lo_b, hi_b = np.percentile(bands, [2.5, 97.5], axis=0)
        ax[0].plot(tw, waves.mean(0), color=COL[n],
                   label=f"{LAB[n]} ({len(waves)} bursts)")
        ax[0].fill_between(tw, lo_b, hi_b, color=COL[n], alpha=0.15, lw=0)
        amp = np.concatenate([e["amp"] for e in ev[n]])
        q = np.sort(amp)
        ax[1].semilogy(q, 1.0 - (np.arange(len(q)) + 0.5) / len(q),
                       color=COL[n], label=LAB[n])
    ax[0].axhline(thr, color="0.5", lw=0.6, ls=":")
    ax[0].set_xlabel(r"$(t - t_{peak})/\tau_{d1}$")
    ax[0].set_ylabel("conditionally averaged $y$")
    ax[0].set_title("(a) burst waveform")
    ax[0].legend(fontsize=7)
    ax[1].set_xlim(thr, AMP_AXIS_MAX)
    ax[1].set_ylim(5e-4, 1.2)
    ax[1].set_xlabel("peak amplitude $q$")
    ax[1].set_ylabel("$P(Y_{peak} > q)$")
    ax[1].set_title("(b) amplitude exceedance")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_bursts{tag}.pdf"))
    plt.close(fig)
    if "--bursts-only" in sys.argv:
        print(f"figure -> {cfg.FIG_DIR}/ex3_bursts{tag}.pdf")
        return

    # ================= 2. stochastic phase portrait ======================
    ye = np.linspace(0, mu + 6 * sd, 71)
    vlim = 3.0 * sd / cfg.DT
    ve = np.linspace(-0.6 * vlim, vlim, 81)
    Hp, cond = {}, {}
    yc = 0.5 * (ye[1:] + ye[:-1])
    for n in NAMES:
        X = src[n]
        y0 = X[:, :-1].ravel()
        v = (np.diff(X, axis=1) / cfg.DT).ravel()
        Hp[n] = np.histogram2d(y0, v, bins=[ye, ve], density=True)[0]
        inr = (y0 >= ye[0]) & (y0 < ye[-1])         # in-range points only
        ib = np.digitize(y0[inr], ye) - 1
        vv = v[inr]
        rows = []
        for k in range(len(yc)):
            sel = vv[ib == k]
            rows.append([sel.mean(), np.quantile(sel, 0.1), np.quantile(sel, 0.9)]
                        if len(sel) > 50 else [np.nan] * 3)
        cond[n] = np.array(rows)
    pos = np.concatenate([h[h > 0].ravel() for h in Hp.values()])
    vmin, vmax = float(pos.min()), float(pos.max())
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for c, n in enumerate(NAMES):
        ax[c].pcolormesh(ye, ve, Hp[n].T, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap="viridis", shading="flat")
        ax[c].plot(yc, cond[n][:, 0], "w", lw=1.3)
        ax[c].plot(yc, cond[n][:, 1], "w", lw=0.8, ls="--")
        ax[c].plot(yc, cond[n][:, 2], "w", lw=0.8, ls="--")
        ax[c].axvline(thr, color="r", lw=0.8, ls=":")
        ax[c].set_title(LAB[n].replace("memory-conditioned ", "memory-conditioned\n"))
        ax[c].set_xlabel("$y_n$")
    ax[0].set_ylabel(r"$\Delta y_{n+1}/\Delta t$")
    paper_typography(fig, 0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_phase{tag}.pdf"))
    plt.close(fig)

    # ---------------- console summary of the physics metrics -------------
    keys = ["burst_rate", "duty_cycle", "wait_mean", "wait_cv", "rise_time",
            "decay_time", "amp_q50", "amp_q90", "amp_q99", "dur_mean",
            "extreme_prob"]
    print("physics metrics (value [95% bootstrap CI]; 'c' = censored at the window):")
    print("  " + "metric".ljust(14) + "".join(LAB[n].ljust(40) for n in NAMES))
    for k in keys:
        row = k.ljust(14)
        for n in NAMES:
            m = metrics[n][k]
            cen = (k == "rise_time" and metrics[n]["rise_censored"]) or \
                  (k == "decay_time" and metrics[n]["decay_censored"])
            row += (f"{m['value']:.4g}{'c' if cen else ''} "
                    f"[{m['ci'][0]:.3g}, {m['ci'][1]:.3g}]").ljust(40)
        print("  " + row)
    print(f"figures -> {cfg.FIG_DIR}/ex3_bursts{tag}.pdf, ex3_phase{tag}.pdf; "
          f"metrics -> {cfg.OUT_DIR}/physics_metrics{tag}.json")


if __name__ == "__main__":
    main()
