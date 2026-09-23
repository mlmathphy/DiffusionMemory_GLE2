"""Evaluation for Example 3: memory-conditioned diffusion (distilled flow
map) vs an LSTM with a Gaussian head, on identical data (torch; workstation).

One-step (Example-2 style): at held-out anchors (quiet / active /
unstratified histories of the test records) every model's draws of the
next increment are compared with the OBSERVABLE-HISTORY reference, the
guided particle filter's predictive law p(y_{n+1} | y_{0:n}) (pfilter.py;
draws are fingerprint-cached under out/cache/; a 2N-particle replicate on
PF_CONV_TRAJ trajectories reports the p90 convergence), by the excess
error E_norm = [W2^2(model, PF) - W2^2(PF', PF)] / Var(PF). Reported:
ours (distilled net), the LSTM for every training seed, and two
diagnostics that are NOT benchmark models -- the raw training-free sampler
(distillation error) and the Gaussian with the PF predictive moments (the
ideal observable Gaussian head).

Closed loop: rollouts from paired held-out prehistories (N_SEED_ROLL noise
seeds) -> ACF error vs the closed-form ACF (max over lags <= N_OUT_ACF and
integrated slow tail over SLOW_TAIL_T), stationary mean/variance/skewness/
flatness vs closed form, burst probability P(y > mean + BURST_SIGMA sd) vs
the held-out data, stability (escapes, variance drift), rollout speed.
Cost: parameter counts, recorded wall times, LSTM tokens.

    python evaluate.py [--smoke]   -> out/summary.json, figs/ex3_*.pdf
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import skew as sk_skew, kurtosis as sk_kurt

import config as cfg
import model as md
from pfilter import ParticleFilter
from conditioning import ema_features, project
from memdiff.cache import cached, fingerprint, state_fingerprint
from memdiff.features import acf_estimate


if "--plot" not in sys.argv:
    import torch
    from memdiff.lstm import LSTMGaussian, ensemble_rollout_multi
    from memdiff.lstm import rollout as lstm_rollout
    from memdiff.sampler import TrainingFreeSampler
    from train import FlowMapNet


# ---------------------------------------------------------------- helpers
def w2_1d(A, B):
    # Exact squared 2-Wasserstein distance between two equal-size empirical
    # distributions: mean squared difference of order statistics. (The
    # previous interpolated-quantile form was biased away from zero even
    # for identical samples.)
    A = np.sort(np.asarray(A, float))
    B = np.sort(np.asarray(B, float))
    if A.shape != B.shape:
        raise ValueError("w2_1d requires equal-size samples")
    out = float(np.mean((A - B) ** 2))
    if not np.isfinite(out):
        raise FloatingPointError("non-finite W2")
    return out


def load_mem(device):
    ck = torch.load(os.path.join(cfg.OUT_DIR, "mem", "model.pt"),
                    map_location=device, weights_only=False)
    sel = ck["selection"]
    d_c = 1 + (sel["proj"]["p"] if sel.get("reduce") and sel.get("proj")
               else ck["k"])
    net = FlowMapNet(d_c, 1, cfg.HIDDEN, cfg.N_LAYERS).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def load_lstms(device):
    out = {}
    for seed in cfg.LSTM_SEEDS:
        p = os.path.join(cfg.OUT_DIR, f"lstm_s{seed}", "model.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location=device, weights_only=False)
        meta = ck["meta"]
        net = LSTMGaussian(1, meta["hidden"], meta["layers"],
                           meta.get("n_mix", 1)).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        out[int(seed)] = dict(net=net, meta=meta, n_par=ck["n_parameters"])
    return out


def net_sample(net, norm, C, z, device):
    X = np.hstack([C, z]).astype(np.float32)
    Xn = torch.tensor((X - norm["x_mu"]) / norm["x_sd"], device=device)
    with torch.no_grad():
        Y = net(Xn).cpu().numpy()
    return Y * norm["y_sd"] + norm["y_mu"]


def cond_from(x, m, sel):
    x = np.asarray(x, float).reshape(-1)
    if sel.get("reduce") and sel.get("proj"):
        return np.hstack([x[:, None], project(x, m, sel["proj"])])
    return np.hstack([x[:, None], m])


def rollout_mem(net, ck, x0, m0, n_steps, seed, device):
    """Autoregressive rollout of the distilled flow map with the bank
    recursion m <- rho m + dy (and the projection online)."""
    rng = np.random.default_rng(seed)
    rho = np.exp(-np.asarray(ck["rates"]) * cfg.DT)
    x, m = x0.copy(), m0.copy()
    X = np.empty((len(x0), n_steps))
    for n in range(n_steps):
        z = rng.standard_normal((len(x0), 1))
        s = net_sample(net, ck["norm"], cond_from(x, m, ck["selection"]), z,
                       device)
        d = s[:, 0] / ck["kappa_s"]
        x = x + d
        m = rho[None, :] * m + d[:, None]
        X[:, n] = x
    return X


def select_anchors(X_test, lo, rng):
    """(traj, t, stratum) anchors: quiet (y below its median), active (top
    20%), unstratified; from N_ANCH_TRAJ_EVAL test trajectories, t >= lo."""
    traj = rng.choice(X_test.shape[0], cfg.N_ANCH_TRAJ_EVAL, replace=False)
    T = np.arange(lo, X_test.shape[1] - 1)
    ii, tt = np.meshgrid(traj, T, indexing="ij")
    ii, tt = ii.ravel(), tt.ravel()
    y = X_test[ii, tt]
    qlo, qhi = np.quantile(y, [cfg.Q_QUIET, cfg.Q_ACTIVE])
    pick = lambda mask, n: rng.choice(np.nonzero(mask)[0], n, replace=False)  # noqa
    q = pick(y < qlo, cfg.N_ANCH_EVAL)
    a = pick(y > qhi, cfg.N_ANCH_EVAL)
    u = pick(np.ones_like(y, bool), cfg.N_UNSTRAT_EVAL)
    idx = np.concatenate([q, a, u])
    strata = np.array(["quiet"] * len(q) + ["active"] * len(a)
                      + ["unstrat"] * len(u))
    return ii[idx], tt[idx], strata


def pf_reference(X_test, ti, ni, n_part, n_s, seed, P, tag):
    """Guided-PF predictive draws at the anchors, fingerprint-cached.
    Returns d1, d2 (n_a, n_s), mu, var, ess_near_min (n_a,)."""
    key = fingerprint(X_test[np.unique(ti)], ti, ni, n_part, n_s, seed,
                      cfg.TAU_D, cfg.GAMMA, cfg.TAU_R, cfg.EPS_NOISE, cfg.DT,
                      cfg.BURN_T, "guided-v2.1")

    def compute():
        rng = np.random.default_rng(seed)
        out = np.empty((len(ti), 2 * n_s + 3))
        by = {}
        for a, (i, t) in enumerate(zip(ti, ni)):
            by.setdefault(int(i), []).append((int(t), a))
        for i, lst in by.items():
            pf = ParticleFilter(P, n_part, rng, guided=True)
            for t, a in sorted(lst):
                pf.run(X_test[i], t)
                d1, d2 = pf.predictive(n_s), pf.predictive(n_s)
                mu, var = pf.moments()
                out[a, :n_s], out[a, n_s:2 * n_s] = d1, d2
                out[a, 2 * n_s:] = (mu, var, float(pf.ess_near(t).min()))
        return out

    arr = cached(os.path.join(cfg.OUT_DIR, "cache", f"pf_ref_{tag}.npz"),
                 key, compute)
    return (arr[:, :n_s], arr[:, n_s:2 * n_s], arr[:, 2 * n_s],
            arr[:, 2 * n_s + 1], arr[:, 2 * n_s + 2])


# ------------------------------------------------------------------ main
def main():
    if "--smoke" in sys.argv:
        cfg.apply_smoke()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P = md.params()
    data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
    X_train, X_test = data["X_train"], data["X_test"]
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    print(f"== Example 3 evaluation (device {device}, out {cfg.OUT_DIR}) ==")

    net, ck = load_mem(device)
    sel = ck["selection"]
    rates, kappa_s = np.asarray(ck["rates"]), ck["kappa_s"]
    lstms = load_lstms(device)
    if not lstms:
        sys.exit("no LSTM checkpoints found (run train_lstm.py first)")
    print(f"  ours: k={len(rates)}, cond dim {1 + (sel['proj']['p'] if sel.get('reduce') else len(rates))}"
          f", flow-map params {sum(p.numel() for p in net.parameters())}; "
          f"LSTM seeds {sorted(lstms)} (params {next(iter(lstms.values()))['n_par']})")

    # the raw training-free sampler (teacher) -- distillation diagnostic
    m_tr, n_burn = ema_features(X_train, rates)
    C_tr = cond_from(X_train[:, n_burn:-1].reshape(-1),
                     m_tr[:, n_burn:-1].reshape(-1, len(rates)), sel)
    S_tr = (np.diff(X_train, axis=1)[:, n_burn:] * kappa_s).reshape(-1, 1)
    teacher = TrainingFreeSampler(C_tr, S_tr, cfg.J_NEIGHBORS, cfg.NU,
                                  cfg.EPS_METRIC, cfg.N_ODE,
                                  label_batch=cfg.LABEL_BATCH, device=device)

    # memory states on the test records; common admissible start
    m_te, n_burn_te = ema_features(X_test, rates)
    lo = max(n_burn_te, cfg.LSTM_WARM, cfg.PF_WARM) + 1
    rng = np.random.default_rng(cfg.SEED_EVAL)

    # ======================= one-step vs particle-filter reference =======
    ti, ni, strata = select_anchors(X_test, lo, rng)
    n_s = cfg.N_SAMP_EVAL
    counts = {str(k): int(v) for k, v in
              zip(*np.unique(strata, return_counts=True))}
    print(f"  anchors: {len(ti)} ({counts}); guided PF reference "
          f"(N={cfg.PF_N_EVAL}) ...", flush=True)
    t0 = time.perf_counter()
    d1, d2, pf_mu, pf_var, pf_essmin = pf_reference(
        X_test, ti, ni, cfg.PF_N_EVAL, n_s, cfg.SEED_EVAL + 11, P, "main")
    print(f"    reference ready ({time.perf_counter() - t0:.0f} s); PF "
          f"near-anchor weight-ESS min p10 {np.percentile(pf_essmin, 10):.0f}")
    # convergence replicate (2N) on a subset of trajectories
    sub_traj = np.unique(ti)[:cfg.PF_CONV_TRAJ]
    sub = np.isin(ti, sub_traj)
    c1, _, _, _, _ = pf_reference(X_test, ti[sub], ni[sub], 2 * cfg.PF_N_EVAL,
                                  n_s, cfg.SEED_EVAL + 12, P, "conv2N")
    conv = []
    for a_sub, a in enumerate(np.nonzero(sub)[0]):
        v = d1[a].var()
        conv.append((w2_1d(c1[a_sub], d1[a]) - w2_1d(d2[a], d1[a])) / v)
    conv = np.array(conv)
    print(f"    PF convergence (2N vs N) W2/var median {np.median(conv):.4f}, "
          f"p90 {np.percentile(conv, 90):.4f}")

    rows = []
    t0 = time.perf_counter()
    for a, (i, t, st) in enumerate(zip(ti, ni, strata)):
        y_t = X_test[i, t]
        ref1, ref2 = d1[a] - y_t, d2[a] - y_t
        floor = w2_1d(ref2, ref1)
        var = float(ref1.var())
        E = lambda x: (w2_1d(x, ref1) - floor) / var            # noqa: E731
        c = cond_from(np.array([y_t]), m_te[i, t][None, :], sel)
        z = rng.standard_normal((n_s, 1))
        ours = net_sample(net, ck["norm"], np.repeat(c, n_s, 0), z, device)[:, 0] / kappa_s
        raw, dg = teacher.sample_conditional(c[0], n_s, seed=int(rng.integers(1 << 31)))
        raw = raw[:, 0] / kappa_s
        row = dict(real=int(i), t=int(t), stratum=st, y=float(y_t),
                   floor=floor, var_pf=var, floor_over_var=floor / var,
                   ess=float(dg["ess"]), E_ours=E(ours), E_raw=E(raw),
                   E_gauss_pf=E((pf_mu[a] - y_t) + np.sqrt(pf_var[a])
                                * rng.standard_normal(n_s)),
                   ours_over_floor=(E(ours) * var + floor) / floor)
        pre = X_test[i, t - cfg.LSTM_WARM:t + 1][None, :]
        for seed, L in lstms.items():
            pos = ensemble_rollout_multi(L["net"], L["meta"], pre, n_s, 1,
                                         int(rng.integers(1 << 31)), device)
            row[f"E_lstm_{seed}"] = E(pos[0, :, 0] - y_t)
        rows.append(row)
        if (a + 1) % 32 == 0:
            print(f"    {a + 1} anchors ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    arr = lambda k: np.array([r[k] for r in rows])                # noqa: E731
    st = arr("stratum")
    med = lambda v, m=None: float(np.median(v if m is None else v[m]))  # noqa
    masks = {"all": np.ones(len(rows), bool), "quiet": st == "quiet",
             "active": st == "active", "unstrat": st == "unstrat",
             "stratified": st != "unstrat"}
    onestep = dict(n_anchors=len(rows), pf_conv_med=float(np.median(conv)),
                   pf_conv_p90=float(np.percentile(conv, 90)),
                   pf_ess_near_min_p10=float(np.percentile(pf_essmin, 10)),
                   label_ess_med=med(arr("ess")), label_ess_p05=float(
                       np.percentile(arr("ess"), 5)),
                   floor_over_var=med(arr("floor_over_var")), by_stratum={})
    for name, mk in masks.items():
        d = dict(E_ours=med(arr("E_ours"), mk), E_raw=med(arr("E_raw"), mk),
                 E_gauss_pf=med(arr("E_gauss_pf"), mk),
                 E_lstm={s: med(arr(f"E_lstm_{s}"), mk) for s in lstms})
        d["lstm_over_ours"] = {s: d["E_lstm"][s] / max(d["E_ours"], 1e-9)
                               for s in lstms}
        d["lstm_over_ours_median_seed"] = float(np.median(list(
            d["lstm_over_ours"].values())))
        d["gauss_pf_over_ours"] = d["E_gauss_pf"] / max(d["E_ours"], 1e-9)
        onestep["by_stratum"][name] = d
    onestep["anchors"] = rows
    s_all = onestep["by_stratum"]["stratified"]
    print(f"  ONE-STEP excess errors (stratified anchors, median): ours "
          f"{s_all['E_ours']:.4f}, raw sampler {s_all['E_raw']:.4f}, "
          f"Gaussian[PF] {s_all['E_gauss_pf']:.4f}, LSTM "
          f"{ {s: round(v, 4) for s, v in s_all['E_lstm'].items()} }")
    print(f"    LSTM/ours per seed { {s: round(v, 2) for s, v in s_all['lstm_over_ours'].items()} }"
          f" (median {s_all['lstm_over_ours_median_seed']:.2f}); quiet "
          f"{onestep['by_stratum']['quiet']['lstm_over_ours_median_seed']:.2f}, "
          f"active {onestep['by_stratum']['active']['lstm_over_ours_median_seed']:.2f}, "
          f"unstratified {onestep['by_stratum']['unstrat']['lstm_over_ours_median_seed']:.2f}")

    # ============================ closed-loop rollouts ===================
    C_cf = md.acf_sampled(cfg.N_OUT_ACF, P=P)
    cu = md.cumulants(P)
    thr = P["mean"] + cfg.BURST_SIGMA * np.sqrt(P["var"])
    burst_data = float((X_test > thr).mean())

    def stats(X):
        X = X[:, cfg.DISCARD:]
        ac = acf_estimate(X, cfg.N_OUT_ACF)
        ea, es = md.acf_errors(ac, C_cf, cfg.slow_tail_lags())
        mu, sd = X.mean(), X.std()
        half = X.shape[1] // 2
        return dict(acf_max=ea, acf_slow=es, mean=float(mu), var=float(sd ** 2),
                    mean_relerr=float(abs(mu - P["mean"]) / P["mean"]),
                    var_relerr=float(abs(sd ** 2 - P["var"]) / P["var"]),
                    skew=float(sk_skew(X.ravel())),
                    flat=float(sk_kurt(X.ravel(), fisher=False)),
                    burst_prob=float((X > thr).mean()),
                    escape_frac=float((np.abs(X - P["mean"])
                                       > 20 * np.sqrt(P["var"])).any(1).mean()),
                    var_drift=float(X[:, half:].var() / max(X[:, :half].var(), 1e-12)))

    # one trained benchmark instance (config.LSTM_SEEDS): its seed is the
    # one shown in the closed-loop figures and traces
    if len(lstms) != 1:
        raise RuntimeError(f"exactly one LSTM instance is compared "
                           f"(LSTM_SEEDS = {cfg.LSTM_SEEDS}); found {sorted(lstms)}")
    rep_seed = next(iter(lstms))
    print(f"  LSTM instance: seed {rep_seed} (val NLL "
          f"{lstms[rep_seed]['meta']['val_nll']:.4f})")
    # complete deployment state in the rollout fingerprints (codeX): weights,
    # normalization, kappa_s, rates, projection, DT, discard, LSTM meta/warm
    dep_ours = dict(state=state_fingerprint(ck["state"]), norm=ck["norm"],
                    kappa_s=kappa_s, rates=rates, dt=cfg.DT,
                    proj=(sel.get("proj") or {}), reduce=bool(sel.get("reduce")))
    roll = {"ours": [], **{f"lstm_{s}": [] for s in lstms}}
    speed = {"ours": [], **{f"lstm_{s}": [] for s in lstms}}   # s per 1000 steps
    for sd in range(cfg.N_SEED_ROLL):
        r2 = np.random.default_rng(cfg.SEED_EVAL + 900 + sd)
        ti_r = r2.integers(0, X_test.shape[0], cfg.N_GEN_TRAJ)
        ni_r = r2.integers(lo, X_test.shape[1] - 1, cfg.N_GEN_TRAJ)
        noise_seed = cfg.SEED_EVAL + 700 + sd
        x0 = X_test[ti_r, ni_r]
        m0 = m_te[ti_r, ni_r]
        key = fingerprint(dep_ours, x0, m0, noise_seed, cfg.L_GEN, cfg.DISCARD)
        t0 = time.perf_counter()
        computed = [False]

        def _ours():
            computed[0] = True
            return rollout_mem(net, ck, x0, m0, cfg.L_GEN, noise_seed, device)
        Xo = cached(os.path.join(cfg.OUT_DIR, "cache", f"roll_ours_s{sd}.npz"),
                    key, _ours)
        if computed[0]:
            speed["ours"].append((time.perf_counter() - t0) / cfg.L_GEN * 1000)
        roll["ours"].append(stats(Xo))
        pre = np.stack([X_test[a, b - cfg.LSTM_WARM:b + 1]
                        for a, b in zip(ti_r, ni_r)])
        Xl_rep = None
        for s, L in lstms.items():
            key = fingerprint(state_fingerprint(L["net"].state_dict()),
                              L["meta"], cfg.LSTM_WARM, pre, noise_seed,
                              cfg.L_GEN, cfg.DISCARD, cfg.DT)
            t0 = time.perf_counter()
            computed = [False]

            def _lstm(L=L):
                computed[0] = True
                return lstm_rollout(L["net"], L["meta"], pre, cfg.L_GEN,
                                    noise_seed, device)
            Xl = cached(os.path.join(cfg.OUT_DIR, "cache",
                                     f"roll_lstm{s}_s{sd}.npz"), key, _lstm)
            if computed[0]:
                speed[f"lstm_{s}"].append((time.perf_counter() - t0)
                                          / cfg.L_GEN * 1000)
            roll[f"lstm_{s}"].append(stats(Xl))
            if s == rep_seed:
                Xl_rep = Xl
        if sd == 0:
            Xl = Xl_rep
            trace = dict(rep_seed=int(rep_seed),
                         ours=Xo[:3, cfg.DISCARD:cfg.DISCARD + 400].tolist(),
                         lstm=Xl[:3, cfg.DISCARD:cfg.DISCARD + 400].tolist(),
                         data=X_test[:3, lo:lo + 400].tolist(),
                         Xo_hist=np.histogram(Xo[:, cfg.DISCARD:], bins=120,
                                              range=(-2, 25), density=True)[0].tolist(),
                         Xl_hist=np.histogram(Xl[:, cfg.DISCARD:], bins=120,
                                              range=(-2, 25), density=True)[0].tolist(),
                         Xd_hist=np.histogram(X_test, bins=120, range=(-2, 25),
                                              density=True)[0].tolist(),
                         acf_ours=acf_estimate(Xo[:, cfg.DISCARD:], cfg.N_OUT_ACF).tolist(),
                         acf_lstm=acf_estimate(Xl[:, cfg.DISCARD:], cfg.N_OUT_ACF).tolist())

    # rollout speed: a dedicated short timing run (independent of cache hits),
    # same batch of prehistories, N_TIME steps; reported as s per 1000 steps
    N_TIME = 200 if not cfg.TAG else 50
    t0 = time.perf_counter()
    rollout_mem(net, ck, x0, m0, N_TIME, 12345, device)
    speed["ours"].append((time.perf_counter() - t0) / N_TIME * 1000)
    for s, L in lstms.items():
        t0 = time.perf_counter()
        lstm_rollout(L["net"], L["meta"], pre, N_TIME, 12345, device)
        speed[f"lstm_{s}"].append((time.perf_counter() - t0) / N_TIME * 1000)
    speed_note = (f"timed on {cfg.N_GEN_TRAJ} trajectories x {N_TIME} steps "
                  f"(device {device})")

    def band(lst, key):
        v = np.array([d[key] for d in lst])
        return dict(median=float(np.median(v)), min=float(v.min()),
                    max=float(v.max()))
    keys = ["acf_max", "acf_slow", "mean_relerr", "var_relerr", "skew", "flat",
            "burst_prob", "escape_frac", "var_drift"]
    closed = {name: {k: band(lst, k) for k in keys} for name, lst in roll.items()}
    closed["reference"] = dict(skew=cu["skew"], flat=cu["flat"],
                               burst_prob_data=burst_data, mean=P["mean"],
                               var=P["var"])
    print("  CLOSED LOOP (median over rollout seeds): "
          + "; ".join(f"{n}: ACF max {c['acf_max']['median']:.3f}, slow "
                      f"{c['acf_slow']['median']:.3f}, skew {c['skew']['median']:.2f}"
                      f", flat {c['flat']['median']:.2f}, burst "
                      f"{c['burst_prob']['median']:.4f}, escapes "
                      f"{c['escape_frac']['median']:.3f}"
                      for n, c in closed.items() if n != "reference")
          + f" | reference skew {cu['skew']:.2f}, flat {cu['flat']:.2f}, burst "
          f"(data) {burst_data:.4f}")

    # ================================= cost ===============================
    cost = dict(ours=dict(params=int(sum(p.numel() for p in net.parameters())),
                          wall_labels_s=ck.get("wall_labels_s"),
                          wall_distill_s=ck.get("wall_distill_s"),
                          n_labels=cfg.N_LABELS),
                lstm={s: dict(params=int(L["n_par"]),
                              wall_train_s=L["meta"].get("wall_train_s"),
                              tokens=L["meta"].get("tokens"),
                              epochs=L["meta"].get("epochs_run"))
                      for s, L in lstms.items()},
                rollout_speed_s_per_1000_steps={
                    k: (dict(median=float(np.median(v)), n_timed=len(v))
                        if v else None) for k, v in speed.items()},
                rollout_speed_note=speed_note,
                lstm_seed=int(rep_seed))

    summary = dict(regime=dict(TAU_D=cfg.TAU_D, GAMMA=cfg.GAMMA, TAU_R=cfg.TAU_R,
                               EPS_NOISE=cfg.EPS_NOISE, DT=cfg.DT),
                   selection=dict(k=int(len(rates)), rates=rates.tolist(),
                                  band=ck.get("band"), kappa_s=float(kappa_s),
                                  p=(sel["proj"]["p"] if sel.get("proj") else None),
                                  spectrum=(sel["proj"]["spectrum"]
                                            if sel.get("proj") else None),
                                  acf_err_chosen=sel.get("acf_err_chosen")),
                   seeds=dict(data=cfg.SEED_DATA, labels=cfg.SEED_LABELS,
                              train=cfg.SEED_TRAIN, lstm=list(lstms),
                              eval=cfg.SEED_EVAL),
                   sizes=dict(N_TRAJ=cfg.N_TRAJ, L_TRAJ=cfg.L_TRAJ,
                              N_TEST=cfg.N_TEST, N_SAMP=n_s,
                              PF_N=cfg.PF_N_EVAL, N_GEN_TRAJ=cfg.N_GEN_TRAJ,
                              L_GEN=cfg.L_GEN, N_SEED_ROLL=cfg.N_SEED_ROLL),
                   onestep=onestep, closed_loop=closed, cost=cost,
                   trace=trace)
    path = os.path.join(cfg.OUT_DIR, "summary.json")
    with open(path, "w") as f:
        json.dump(_jsonable(summary), f, indent=1)
    print(f"  summary -> {path}")
    figures(summary, rows, masks, lstms, P)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------ figures
def figures(S, rows, masks, lstms, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style
    setup_style()
    arr = lambda k: np.array([r[k] for r in rows])                # noqa: E731
    seeds = sorted(lstms)
    # (1) one-step excess errors by stratum: the two learned models only
    # (ours vs the Gaussian-head LSTM, all training seeds)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, name in zip(ax, ("quiet", "active", "unstrat")):
        mk = masks[name]
        data = [arr("E_ours")[mk]] + [arr(f"E_lstm_{s}")[mk] for s in seeds]
        a.boxplot(data)
        a.set_xticks(range(1, len(data) + 1))
        a.set_xticklabels(["memory-conditioned\ndiffusion"]
                          + ["LSTM +\nGaussian head"] * len(seeds), fontsize=8)
        a.set_yscale("symlog", linthresh=0.01)
        a.set_ylabel("excess error $E_{norm}$")
        a.set_title(f"{name} histories")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_onestep{cfg.TAG}.pdf"))
    plt.close(fig)
    # (1b) appendix diagnostics: raw sampler (distillation error) and the
    # Gaussian with the PF moments (ideal observable Gaussian head)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, name in zip(ax, ("quiet", "active", "unstrat")):
        mk = masks[name]
        data = [arr("E_ours")[mk], arr("E_raw")[mk], arr("E_gauss_pf")[mk]]
        a.boxplot(data)
        a.set_xticks(range(1, 4))
        a.set_xticklabels(["distilled\nflow map", "raw training-free\nsampler",
                           "Gaussian with\nPF moments"], fontsize=8)
        a.set_yscale("symlog", linthresh=0.01)
        a.set_ylabel("excess error $E_{norm}$")
        a.set_title(f"{name} histories")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_diagnostics{cfg.TAG}.pdf"))
    plt.close(fig)
    plot_rollout(S)


def plot_rollout(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style, paper_typography
    setup_style()
    P = md.params()
    # (2) closed loop: ACF and stationary PDF (user/codeX: two panels)
    T = S["trace"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.5))
    C_cf = md.acf_sampled(cfg.N_OUT_ACF, P=P)
    lags = np.arange(cfg.N_OUT_ACF + 1) * cfg.DT
    ax[0].plot(lags, C_cf / C_cf[0], "k", label="closed form")
    ax[0].plot(lags, np.array(T["acf_ours"]) / C_cf[0], "C3", alpha=0.8, label="ours")
    ax[0].plot(lags, np.array(T["acf_lstm"]) / C_cf[0], "C0", alpha=0.8,
               label="LSTM + Gaussian head")
    ax[0].set_xlabel(r"$\tau/\tau_{d1}$")
    ax[0].set_ylabel("normalized ACF")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) autocovariance in rollout")
    edges = np.linspace(-2, 25, 121)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ax[1].semilogy(mid, T["Xd_hist"], "k", label="data")
    ax[1].semilogy(mid, T["Xo_hist"], "C3", alpha=0.8, label="ours")
    ax[1].semilogy(mid, T["Xl_hist"], "C0", alpha=0.8, label="LSTM")
    ax[1].set_xlabel("$y$")
    ax[1].set_ylim(1e-5, 1)
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) stationary PDF")
    paper_typography(fig, 0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"ex3_rollout{cfg.TAG}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        with open(os.path.join(cfg.OUT_DIR, "summary.json")) as f:
            plot_rollout(json.load(f))
    else:
        main()
