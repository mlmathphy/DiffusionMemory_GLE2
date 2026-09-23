"""LSTM context-length robustness sweep (Stage 0.5; torch, workstation).

Addresses the one anticipated objection to the Stage-0 result -- "would
a longer-context LSTM recover the slow mode?" -- on the SAME data set,
with no new physics and no new metric. For every combination of

    LSTM_SEQ  in LSTM_SWEEP_SEQ      (training subsequence length)
    seed      in LSTM_SWEEP_SEEDS    (training seeds, for robustness)

an LSTM with the identical Gaussian head is trained with burn-in
LSTM_SWEEP_BURN_FRAC * SEQ and batch scaled as LSTM_SWEEP_TOKENS / SEQ:
configurations are APPROXIMATELY matched in processed tokens per update
and share the same update cap; actual compute is reported through wall
time and realized update counts. At deployment every swept LSTM gets
the SAME warm-up of LSTM_SWEEP_WARM observed steps (= the longest
training context), so the longest-context model receives its full
context at initialization and starting conditions stay paired across
the sweep.

Each checkpoint is scored with the same two Stage-0 instruments:

  * closed-loop: E_slow, E_slow_shape (the headline: variance error
    removed), and var_relerr over the N_SEED rollout seeds -- so a
    variance change cannot masquerade as memory recovery or loss;
  * oracle forecast fans (identical queries and Kalman reference as
    forecast_fans.py): conditional-mean RMSE AND the conditional-spread
    ratio at the report horizons -- a longer context that improves the
    mean while miscalibrating the spread is visible, not hidden.

Interpretation is predeclared and symmetric: if longer contexts recover
the slow mode, the conclusion is a COST-SCALING advantage (training
cost grows with the physical memory horizon; the bank keeps a small
fixed recursive state); if they do not, the advantage persists across
the tested context lengths, seeds, and budgets -- other capacity or
optimization choices remain possible and are not ruled out.

Resumability: an existing checkpoint is reused only after verifying
that its stored seq / burn / batch / training seed / architecture match
the current protocol; a mismatch aborts with instructions instead of
silently reusing an incompatible model.

Outputs: out/lstm_sweep/seq<S>_s<seed>/model.pt,
out/lstm_sweep/sweep.json, figs/ex4_lstm_sweep.pdf. Rollouts are
fingerprint-cached under out/lstm_sweep/cache/.

    python sweep_lstm.py            # train missing configs, then score all
    python sweep_lstm.py train      # training only
    python sweep_lstm.py eval       # score existing checkpoints only
    python sweep_lstm.py plot       # refresh the figure from sweep.json
                                    # (no rollouts, no fans)
"""

import json
import os
import sys
import time

import numpy as np
import torch

import config as cfg
import multiscale_exact as exact
from memdiff.cache import cached, fingerprint, state_fingerprint
from memdiff.features import acf_estimate
from memdiff.lstm import (LSTMGaussian, ensemble_rollout_multi,
                          train_lstm)
from memdiff.lstm import rollout as lstm_rollout

SWEEP_DIR = os.path.join(cfg.OUT_DIR, "lstm_sweep")
MARKS = (30.0, 100.0, 200.0, 400.0)
DISCARD = 500                       # same rollout transient as evaluate.py


def overrides(seq, seed):
    return {"LSTM_SEQ": int(seq),
            "LSTM_BURN": int(round(cfg.LSTM_SWEEP_BURN_FRAC * seq)),
            "SEED_LSTM": int(seed),
            "LSTM_BATCH": max(4, int(round(cfg.LSTM_SWEEP_TOKENS / seq)))}


def ckpt_path(seq, seed):
    return os.path.join(SWEEP_DIR, f"seq{seq}_s{seed}", "model.pt")


def check_compatible(meta, seq, seed):
    """(ok, detail): does a stored checkpoint match the current protocol?"""
    if "epochs_run" not in meta:
        return False, ("no cost record (epochs_run) -- trained with a "
                       "stale memdiff/lstm.py")
    want = overrides(seq, seed)
    have = {k: meta.get(k) for k in want}
    arch_want = {"hidden": cfg.LSTM_HIDDEN, "layers": cfg.LSTM_LAYERS,
                 "n_mix": cfg.LSTM_MIX}
    arch_have = {k: meta.get(k) for k in arch_want}
    ok = (have == want and arch_have == arch_want)
    return ok, f"stored {have} / {arch_have}, protocol {want} / {arch_want}"


# ---------------------------------------------------------------- training

def train_all(device):
    X_train = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_train"]
    for seq in cfg.LSTM_SWEEP_SEQ:
        for seed in cfg.LSTM_SWEEP_SEEDS:
            p = ckpt_path(seq, seed)
            if os.path.exists(p):
                meta = torch.load(p, map_location="cpu",
                                  weights_only=False)["meta"]
                ok, detail = check_compatible(meta, seq, seed)
                if not ok:
                    raise SystemExit(
                        f"checkpoint {p} does not match the current sweep "
                        f"protocol ({detail}); move or delete it before "
                        f"re-running")
                print(f"  seq={seq} seed={seed}: compatible checkpoint "
                      f"exists, skipping")
                continue
            over = overrides(seq, seed)
            for k, v in over.items():
                setattr(cfg, k, v)
            print(f"== LSTM sweep: seq={seq} burn={cfg.LSTM_BURN} "
                  f"batch={cfg.LSTM_BATCH} seed={seed} (device {device}) ==")
            t0 = time.time()
            net, meta = train_lstm(X_train, cfg, device)
            wall = time.time() - t0
            meta.update(over, wall_s=float(wall))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({"state": net.state_dict(), "meta": meta,
                        "n_parameters": sum(q.numel()
                                            for q in net.parameters())}, p)
            tokens = meta["epochs_run"] * meta["batch"] * seq
            print(f"  saved -> {p}  ({wall:.0f} s, {meta['epochs_run']} "
                  f"updates, {tokens / 1e6:.1f}M tokens, "
                  f"val NLL {meta['val_nll']:.5f})")


# -------------------------------------------------------------- evaluation

def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    net = LSTMGaussian(1, meta["hidden"], meta["layers"],
                       meta.get("n_mix", 1)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, meta


def evaluate_all(device):
    X_test = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))["X_test"]
    n_full = cfg.EVAL_LAG_FULL
    Tmax = max(cfg.FORCING_TIMES)
    l_lo = int(round(cfg.SLOW_BAND[0] * Tmax / cfg.DT))
    l_hi = min(int(round(cfg.SLOW_BAND[1] * Tmax / cfg.DT)), n_full)
    C_ex = exact.acf_x(n_full)
    den_slow = float(np.sum(np.abs(C_ex[l_lo:l_hi + 1])))

    def e_slow(acov):
        return float(np.sum(np.abs(acov[l_lo:l_hi + 1]
                                   - C_ex[l_lo:l_hi + 1])) / den_slow)

    def cl_stats(X):
        """Closed-loop metrics of one rollout array (Stage-0 definitions)."""
        acov = acf_estimate(X, n_full)
        return {"E_slow": e_slow(acov),
                "E_slow_shape": e_slow(acov / acov[0] * C_ex[0]),
                "var_relerr": float(abs(X.var() - C_ex[0]) / C_ex[0])}

    warm = cfg.LSTM_SWEEP_WARM       # common warm-up = longest context
    lo = warm + 1

    # oracle reference: identical queries to forecast_fans.py (their
    # admissible range is bounded by FAN_HIST = 5000, so the larger
    # sweep warm-up does not change the draw)
    H = int(round(2.0 * Tmax / cfg.DT))
    rngq = np.random.default_rng(cfg.SEED_EVAL + 1300)
    lo_f = max(lo, cfg.FAN_HIST + 1)
    ti = rngq.integers(0, X_test.shape[0], cfg.FAN_N_QUERY)
    ni = rngq.integers(lo_f, X_test.shape[1], cfg.FAN_N_QUERY)
    hist = np.stack([X_test[a, b - cfg.FAN_HIST:b + 1]
                     for a, b in zip(ti, ni)])
    kal_mu, kal_sd = exact.kalman_forecast(hist, H)

    def marks_of(arr):
        return {f"tau={t:g}":
                float(arr[min(int(round(t / cfg.DT)), H) - 1])
                for t in MARKS}

    rows = []
    for seq in cfg.LSTM_SWEEP_SEQ:
        for seed in cfg.LSTM_SWEEP_SEEDS:
            p = ckpt_path(seq, seed)
            if not os.path.exists(p):
                print(f"  seq={seq} seed={seed}: no checkpoint, skipping")
                continue
            net, meta = load_net(p, device)
            ok, detail = check_compatible(meta, seq, seed)
            if not ok:
                print(f"  seq={seq} seed={seed}: INCOMPATIBLE checkpoint "
                      f"({detail}), skipping")
                continue
            sfp = state_fingerprint(net.state_dict())

            # closed loop: N_SEED rollout seeds, cached, sweep-paired
            # starting indices (drawn once per rollout seed at lo)
            per_seed = []
            for rs in range(cfg.N_SEED):
                rng = np.random.default_rng(cfg.SEED_EVAL + 900 + rs)
                ti_r = rng.integers(0, X_test.shape[0], cfg.N_GEN_TRAJ)
                ni_r = rng.integers(lo, X_test.shape[1] - 1,
                                    cfg.N_GEN_TRAJ)
                pre = np.stack([X_test[a, b - warm:b + 1]
                                for a, b in zip(ti_r, ni_r)])

                def compute(net=net, meta=meta, pre=pre, rs=rs):
                    return lstm_rollout(net, meta, pre, cfg.L_GEN,
                                        cfg.SEED_EVAL + 700 + rs,
                                        device)[:, DISCARD:]
                key = fingerprint(sfp, rs, cfg.L_GEN, DISCARD, cfg.DT,
                                  str(device), ti_r, ni_r, pre, warm)
                X = cached(os.path.join(SWEEP_DIR, "cache",
                                        f"roll_seq{seq}_s{seed}_r{rs}.npz"),
                           key, compute)
                per_seed.append(cl_stats(X))
            cl = {k: {"median": float(np.median([r[k] for r in per_seed])),
                      "band": [float(np.min([r[k] for r in per_seed])),
                               float(np.max([r[k] for r in per_seed]))]}
                  for k in per_seed[0]}

            # oracle fans: mean RMSE and spread ratio (both, as in Stage 0)
            mu = np.empty((cfg.FAN_N_QUERY, H))
            sd = np.empty((cfg.FAN_N_QUERY, H))
            for g0 in range(0, cfg.FAN_N_QUERY, cfg.FAN_Q_BATCH):
                g1 = min(g0 + cfg.FAN_Q_BATCH, cfg.FAN_N_QUERY)
                preq = np.stack([X_test[ti[q], ni[q] - warm:ni[q] + 1]
                                 for q in range(g0, g1)])
                E = ensemble_rollout_multi(net, meta, preq, cfg.FAN_N_ENS,
                                           H, cfg.SEED_EVAL + 1400 + g0,
                                           device)
                mu[g0:g1] = E.mean(1)
                sd[g0:g1] = E.std(1)
            rmse = np.sqrt(np.mean(((mu - kal_mu)
                                    / kal_sd[None, :]) ** 2, axis=0))
            ratio = np.median(sd / kal_sd[None, :], axis=0)

            tokens = meta["epochs_run"] * meta["batch"] * seq
            rows.append({
                "seq": int(seq), "seed": int(seed),
                "val_nll": float(meta["val_nll"]),
                "updates": int(meta["epochs_run"]),
                "tokens": int(tokens),
                "wall_s": float(meta.get("wall_s", float("nan"))),
                "warm": int(warm),
                "closed_loop": cl,
                "fan_rmse": marks_of(rmse),
                "fan_sd_ratio": marks_of(ratio),
            })
            r = rows[-1]
            print(f"  seq={seq:5d} seed={seed}: E_slow_shape "
                  f"{cl['E_slow_shape']['median']:.3f} "
                  f"[{cl['E_slow_shape']['band'][0]:.3f}, "
                  f"{cl['E_slow_shape']['band'][1]:.3f}]  var "
                  f"{cl['var_relerr']['median']:.3f}   fan tau=100 rmse "
                  f"{r['fan_rmse']['tau=100']:.3f} ratio "
                  f"{r['fan_sd_ratio']['tau=100']:.3f}   "
                  f"({r['wall_s']:.0f} s, {tokens / 1e6:.0f}M tokens)")
    return rows


def references():
    """Stage-0 mem/floor numbers for the figure, if the JSONs exist."""
    refs = {}
    p = os.path.join(cfg.OUT_DIR, "summary.json")
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        refs["mem_E_slow_shape"] = s.get("mem", {}).get("E_slow_shape")
        refs["floor_E_slow_shape"] = \
            s.get("metric_floor_ideal", {}).get("E_slow_shape")
    p = os.path.join(cfg.OUT_DIR, "forecast_fans.json")
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        refs["mem_fan"] = s.get("mem")
        refs["compression_fan"] = s.get("bank_conditional")
        refs["sampling_floor"] = s.get("ensemble_sampling_floor_nominal")
    return refs


def main(mode="all"):
    # fail fast on a stale shared library BEFORE spending training time:
    # the sweep needs the cost-accounting keys train_lstm was given
    # alongside this script (epochs_run / seq / burn / batch in meta)
    import inspect
    if "epochs_run" not in inspect.getsource(train_lstm):
        raise SystemExit(
            "stale code/memdiff/lstm.py: train_lstm does not record "
            "epochs_run. Sync the updated memdiff/ folder to this "
            "machine and re-run.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    if mode == "plot":
        # rebuild the figure from the stored sweep.json -- no rollouts,
        # no fans, no torch work beyond imports
        p = os.path.join(SWEEP_DIR, "sweep.json")
        if not os.path.exists(p):
            raise SystemExit(f"no {p}; run the eval mode first")
        with open(p) as f:
            stored = json.load(f)
        make_figure(stored["rows"], stored["references"])
        return
    if mode in ("all", "train"):
        train_all(device)
    if mode not in ("all", "eval"):
        return
    rows = evaluate_all(device)
    if not rows:
        raise SystemExit("no sweep checkpoints found; run the train mode")
    refs = references()
    out = {"config": {"seqs": list(cfg.LSTM_SWEEP_SEQ),
                      "seeds": list(cfg.LSTM_SWEEP_SEEDS),
                      "burn_frac": cfg.LSTM_SWEEP_BURN_FRAC,
                      "tokens_per_update": cfg.LSTM_SWEEP_TOKENS,
                      "warm": cfg.LSTM_SWEEP_WARM},
           "references": refs, "rows": rows}
    with open(os.path.join(SWEEP_DIR, "sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    make_figure(rows, refs)


def make_figure(rows, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from memdiff.plotting import setup_style, MODEL_ALPHA
    setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    seqs = sorted({r["seq"] for r in rows})

    def per_seq(key):
        return [[key(r) for r in rows if r["seq"] == s] for s in seqs]

    panels = (
        (ax[0], per_seq(lambda r: r["closed_loop"]["E_slow_shape"]["median"]),
         r"$E_{\mathrm{slow}}$ (shape)", "(a) slow-tail shape error"),
        (ax[1], per_seq(lambda r: r["fan_rmse"]["tau=100"]),
         r"fan mean RMSE at $\tau=100$", "(b) forecast-mean error"),
        (ax[2], per_seq(lambda r: r["fan_sd_ratio"]["tau=100"]),
         r"fan sd ratio at $\tau=100$", "(c) forecast-spread calibration"),
    )
    for a, vals, ylab, title in panels:
        for s, v in zip(seqs, vals):
            a.plot([s] * len(v), v, "o", color="tab:purple",
                   alpha=MODEL_ALPHA)
        a.plot(seqs, [float(np.median(v)) for v in vals], "-",
               color="tab:purple", alpha=MODEL_ALPHA,
               label="LSTM (median of seeds)")
        a.set_xscale("log")
        a.set(xlabel="training subsequence length (steps)", ylabel=ylab,
              title=title)
        # only the three tested lengths on the x axis: log minor ticks
        # would collide with these labels around 2000-4000
        a.set_xticks(seqs)
        a.set_xticklabels([str(s) for s in seqs])
        a.xaxis.set_minor_locator(NullLocator())
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    if refs.get("mem_E_slow_shape"):
        ax[0].axhline(refs["mem_E_slow_shape"]["median"], color="tab:red",
                      ls="-", lw=1.2, alpha=MODEL_ALPHA,
                      label="memory bank (ours)")
    if refs.get("floor_E_slow_shape"):
        ax[0].axhline(refs["floor_E_slow_shape"]["median"], color="k",
                      ls="--", lw=1, label="measured ideal floor")
    if refs.get("mem_fan"):
        ax[1].axhline(refs["mem_fan"]["tau=100"]["mean_rmse"],
                      color="tab:red", ls="-", lw=1.2, alpha=MODEL_ALPHA,
                      label="memory bank (ours)")
    if refs.get("sampling_floor"):
        ax[1].axhline(refs["sampling_floor"], color="k", ls="--", lw=0.8,
                      label=r"nominal sampling floor $1/\sqrt{N}$")
    ax[2].axhline(1.0, color="k", ls="--", lw=0.8)
    for a in ax[:2]:
        a.legend(fontsize=7)
    for s in seqs:
        walls = [r["wall_s"] for r in rows if r["seq"] == s
                 if np.isfinite(r["wall_s"])]
        if walls:
            w = float(np.median(walls))
            lab = f"{w:.0f} s" if w < 90 else f"{w / 60:.0f} min"
            ax[0].annotate(lab, (s, ax[0].get_ylim()[0] * 1.3),
                           fontsize=7, ha="center", color="0.35")
    fig.tight_layout()
    figp = f"{cfg.FIG_DIR}/ex4_lstm_sweep.pdf"
    fig.savefig(figp)
    print(f"figure -> {figp} ; summary -> {SWEEP_DIR}/sweep.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
