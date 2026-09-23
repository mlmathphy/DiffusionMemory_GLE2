"""One-step conditional accuracy for Example 4 (torch inference only).

Each deployed case is measured against its OWN exact one-step target, at
a common set of held-out query instants:

  mem     exact Gaussian given (x_n, m_n)            [cond_given_memory]
  markov  exact one-step projection given x_n        [cond_given_memory([])]
  window  exact Gaussian given the k+1 raw states    [window_onestep]
  lstm    full-history oracle (Kalman on a FAN_HIST prehistory, the same
          numerically converged reference as forecast_fans.py); the LSTM
          head is Gaussian, so the comparison is analytic

For mem / markov / window BOTH the training-free sampler (the teacher,
rebuilt from the training tuples at the stored settings) and the
distilled network are measured; a step-doubling pass on a query subset
(paired latents, identical seed) checks the reverse-ODE resolution.

Per query: |mean error| / target sd, sd ratio, squared Wasserstein-2
distance normalized by the target variance. Medians + 90th percentiles,
plus the sampler's ESS / neighbor-radius diagnostics, go to
out/onestep.json.

    python onestep_accuracy.py --smoke     # 8 queries, seconds
    python onestep_accuracy.py
"""

import json
import os
import sys

import numpy as np
import torch

import config as cfg
import multiscale_exact as exact
from conditioning import build_tuples, build_window_tuples
from memdiff.lstm import one_step_moments
from memdiff.sampler import TrainingFreeSampler

import evaluate  # load_variant / load_lstm / memory_state / net_sample

N_DOUBLE = 16          # step-doubling subset (paired latents)


def w2sq_vs_gauss(samples, mu, var):
    """Squared 1D Wasserstein-2 distance to N(mu, var), quantile form."""
    from scipy.special import ndtri
    s = np.sort(np.asarray(samples, float).reshape(-1))
    q = (np.arange(len(s)) + 0.5) / len(s)
    g = mu + np.sqrt(var) * ndtri(q)
    return float(np.mean((s - g) ** 2))


def window_onestep(M):
    """Exact conditional of D_{n+1} given (x_n, ..., x_{n-M+1}).

    Returns (beta, V) in the layout of build_window_tuples: the i-th
    conditioning entry is x_{n-i}, so Cov(x_{n-i}, D_{n+1}) =
    C(i+1) - C(i).
    """
    C = exact.acf_x(M + 1)
    idx = np.arange(M)
    G = C[np.abs(idx[:, None] - idx[None, :])]
    cov = C[1:M + 1] - C[:M]
    beta = np.linalg.solve(G, cov)
    RD0 = 2.0 * (C[0] - C[1])
    return beta, float(RD0 - cov @ beta)


def summarize(rows, keys):
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = {"median": float(np.median(vals)),
                        "p90": float(np.percentile(vals, 90))}
    return out


def main(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = ""
    n_query, ns, n_double = cfg.N_QUERY, cfg.NS_COND, N_DOUBLE
    if smoke:
        n_query, ns, n_double = 8, 400, 2
        tag = "_smoke"
        print("SMOKE TEST: 8 queries, 400 samples, doubling on 2")

    data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
    X_train, X_test = data["X_train"], data["X_test"]

    # common held-out query instants; the slack covers every warm-up
    # requirement (FAN_HIST dominates: the LSTM's oracle target needs a
    # converged full-history prehistory)
    lo = cfg.FAN_HIST + 1
    hi = X_test.shape[1] - 2                    # X_{n+1} must exist
    rng = np.random.default_rng(cfg.SEED_EVAL + 300)
    ti = rng.integers(0, X_test.shape[0], n_query)
    ni = rng.integers(lo, hi, n_query)

    results = {"n_query": int(n_query), "ns_cond": int(ns),
               "device": str(device),
               "note": ("each case vs its OWN exact conditional target; "
                        "lstm vs the converged full-history Kalman "
                        "oracle (FAN_HIST prehistory), analytic "
                        "Gaussian-vs-Gaussian metrics")}

    # ------------------------------------------- sampler-based variants
    for name in ("mem", "markov", "window"):
        path = os.path.join(cfg.OUT_DIR, name, "model.pt")
        if not os.path.exists(path):
            print(f"[{name}] no checkpoint; skipped")
            continue
        net, ckpt = evaluate.load_variant(name, device)
        kappa_s = float(ckpt["kappa_s"])
        rates = np.asarray(ckpt["rates"], float)
        kind = "window" if name.startswith("window") else "ema"

        if kind == "window":
            k = len(rates)
            beta, V = window_onestep(k + 1)
            tr = build_window_tuples(X_train, k, kappa_s=kappa_s)
        else:
            beta, V = exact.cond_given_memory(rates)
            tr = build_tuples(X_train, rates, kappa_s=kappa_s)

        m_test, _ = evaluate.memory_state(X_test, ckpt, kind)
        v_q = np.concatenate([X_test[ti, ni][:, None], m_test[ti, ni]],
                             axis=1)
        mu_q = v_q @ beta * kappa_s             # scaled-displacement units
        v_s = V * kappa_s ** 2

        sp = ckpt["sampler"]
        smp = TrainingFreeSampler(tr["C"], tr["S"], device=device,
                                  j_neighbors=sp["J"], nu=sp["nu"],
                                  eps=sp["eps"], n_ode=sp["n_ode"],
                                  label_batch=cfg.LABEL_BATCH)
        print(f"[{name}] dim {v_q.shape[1]}, exact V = {V:.6f}, "
              f"{n_query} queries x {ns} samples ...")

        rows = []
        for i in range(n_query):
            c = v_q[i]
            s_tf, diag = smp.sample_conditional(c, ns,
                                                seed=cfg.SEED_EVAL + i)
            z = rng.standard_normal((ns, 1))
            s_nn = evaluate.net_sample(net, ckpt["norm"],
                                       np.tile(c, (ns, 1)), z,
                                       device)[:, 0]
            row = {"ess": float(diag["ess"]),
                   "radius": float(diag["radius"])}
            for pre, s in (("tf", s_tf[:, 0]), ("nn", s_nn)):
                row[f"{pre}_mean_err"] = float(
                    abs(s.mean() - mu_q[i]) / np.sqrt(v_s))
                row[f"{pre}_std_ratio"] = float(s.std() / np.sqrt(v_s))
                row[f"{pre}_w2sq"] = w2sq_vs_gauss(s, mu_q[i], v_s) / v_s
            if i < n_double:
                # identical seed -> identical latents: paired refinement
                s2, _ = smp.sample_conditional(c, ns,
                                               n_ode=2 * smp.n_ode,
                                               seed=cfg.SEED_EVAL + i)
                row["tf_double_shift"] = float(
                    np.mean((s2[:, 0] - s_tf[:, 0]) ** 2)) / v_s
            rows.append(row)

        res = summarize(rows, ("tf_mean_err", "tf_std_ratio", "tf_w2sq",
                               "nn_mean_err", "nn_std_ratio", "nn_w2sq",
                               "tf_double_shift"))
        res["ess_median"] = float(np.median([r["ess"] for r in rows]))
        res["ess_p05"] = float(np.percentile([r["ess"] for r in rows], 5))
        res["radius_median"] = float(np.median([r["radius"]
                                                for r in rows]))
        res["V_exact"] = float(V)
        results[name] = res
        print(f"  tf: mean_err {res['tf_mean_err']['median']:.4f}  "
              f"sd_ratio {res['tf_std_ratio']['median']:.4f}  "
              f"W2^2/V {res['tf_w2sq']['median']:.5f}   "
              f"nn: W2^2/V {res['nn_w2sq']['median']:.5f}   "
              f"ESS {res['ess_median']:.0f}/{res['ess_p05']:.0f}")

    # ------------------------------------------------------------- LSTM
    net_l, meta = evaluate.load_lstm(device)
    if net_l is None:
        print("[lstm] no checkpoint; skipped")
    else:
        x_n = X_test[ti, ni]
        pre_or = np.stack([X_test[a, b - cfg.FAN_HIST:b + 1]
                           for a, b in zip(ti, ni)])
        mu_or, sd_or = exact.kalman_forecast(pre_or, 1)
        mu_or, sd_or = mu_or[:, 0], float(sd_or[0])
        pre_l = np.stack([X_test[a, b - cfg.LSTM_WARM:b + 1]
                          for a, b in zip(ti, ni)])
        mean_d, sd_d = one_step_moments(net_l, meta, pre_l, device)
        mu_l = x_n + mean_d
        rows = []
        for i in range(n_query):
            rows.append({
                "mean_err": float(abs(mu_l[i] - mu_or[i]) / sd_or),
                "std_ratio": float(sd_d[i] / sd_or),
                # Gaussian-Gaussian squared W2, closed form
                "w2sq": float(((mu_l[i] - mu_or[i]) ** 2
                               + (sd_d[i] - sd_or) ** 2) / sd_or ** 2),
            })
        res = summarize(rows, ("mean_err", "std_ratio", "w2sq"))
        res["note"] = ("vs the full-history oracle: the LSTM conditions "
                       "on its whole prehistory, so the oracle is its "
                       "proper target; deployment warm-up LSTM_WARM = "
                       f"{cfg.LSTM_WARM} steps")
        results["lstm"] = res
        print(f"[lstm] vs oracle: mean_err "
              f"{res['mean_err']['median']:.4f}  sd_ratio "
              f"{res['std_ratio']['median']:.4f}  W2^2/V "
              f"{res['w2sq']['median']:.5f}")

    out = os.path.join(cfg.OUT_DIR, f"onestep{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
