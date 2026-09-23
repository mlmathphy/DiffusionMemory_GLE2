"""SOL filtered-Poisson candidate pre-vet (numpy only; no training).
Protocol v2 (codeX rulings 2026-08-22): observable-history oracle by
particle filter; excess-error metrics; corrected gates; v2.1 adds the
reference-reliability checks (guided proposal, replicate filters at N and
2N + bootstrap 2N, p90 convergence, PIT calibration, ESS quantiles near
anchors, unstratified PF-vs-Markov predictive-variance comparison).

Reference for every conditional comparison: the GUIDED PARTICLE-FILTER
predictive law p(y_{n+1} | y_{0:n}) (pfilter.py), NOT the hidden-state
law (kept as a diagnostic: the observable conditional is the posterior
mixture over hidden states).  Excess errors
    E_norm(model) = [W2^2(model, PF draw) - W2^2(PF draw', PF draw)]
                    / Var(PF draw)
remove the Monte-Carlo floor and the N_SAMP dependence; raw floor ratios
are reported as diagnostics only.

Gates (config.GATES; the candidate passes only if every gate passes):
  G1 memory representation, CLOSED FORM (ex4 precedent): at k* (smallest
     k with linear one-step sufficiency eps_k <= EPS_K_MAX) the ideal bank
     (dim k*+1) beats the ideal raw window AR(k*+1) (same dim) in
     closed-loop ACF error by >= BANK_VS_AR_MIN.
  G2 generator advantage at MATCHED OBSERVABLE INFORMATION: median
     E_norm(Gaussian with the PF predictive moments = the ideal observable
     Gaussian head) / median E_norm(sampler) >= GAUSS_OVER_SAMPLER_MIN
     (per-anchor ratio median reported too).
  G3 sampler feasibility: median E_norm(sampler) <= E_NORM_MAX; PF
     convergence W2^2(PF_N, PF_2N)/Var <= PF_CONV_MAX (median over the
     convergence anchors); label-stage ESS median/p05.
  G4 physics / exactness per test record: skewness, flatness, atom in the
     declared ranges; simulator vs closed form (variance, ACF lags <= 50)
     within N_SE standard errors across independent trajectories.
  G5 robustness: G2/G3 on every test record's own anchors, G4 on every
     record.

Usage: python prevet.py [--full]  ->  out/prevet_<mode>.json,
                                      figs/prevet_<mode>.pdf
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import skew as sk_skew, kurtosis as sk_kurt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import config as cfg                                     # noqa: E402
import model as md                                       # noqa: E402
from pfilter import ParticleFilter                       # noqa: E402
from memdiff.features import (ema_features, rates_from_band,  # noqa: E402
                              acf_estimate)
from memdiff.sampler_np import TrainingFreeSamplerNP       # noqa: E402


# ===================================================================
# 1-D helpers
# ===================================================================
def w2_1d(A, B):
    """Exact 1-D W2^2 between two samples (quantile matching)."""
    A = np.sort(np.asarray(A, float))
    n = len(A)
    q = (np.arange(n) + 0.5) / n
    out = float(np.mean((A - np.quantile(B, q)) ** 2))
    if not np.isfinite(out):
        raise FloatingPointError("non-finite W2")
    return out


def gmm1d_fit(x, K, rng, iters=200, tol=1e-8, restarts=2):
    x = np.asarray(x, float)
    n = len(x)
    reg = 1e-6 * x.var() + 1e-12
    best = None
    for _ in range(restarts):
        mu = rng.choice(x, K, replace=False)
        var = np.full(K, x.var())
        pi = np.full(K, 1.0 / K)
        ll_old = -np.inf
        for _it in range(iters):
            logp = (-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
                    - 0.5 * np.log(2 * np.pi * var[None, :])
                    + np.log(pi[None, :] + 1e-300))
            mx = logp.max(1, keepdims=True)
            pj = np.exp(logp - mx)
            norm = pj.sum(1, keepdims=True)
            gam = pj / norm
            ll = float(np.sum(np.log(norm[:, 0]) + mx[:, 0]))
            Nk = gam.sum(0) + 1e-300
            pi = Nk / n
            mu = (gam * x[:, None]).sum(0) / Nk
            var = (gam * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk + reg
            if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best[3]:
            best = (pi.copy(), mu.copy(), var.copy(), ll)
    return best


def gmm1d_sample(pi, mu, var, n, rng):
    ks = rng.choice(len(pi), size=n, p=pi / pi.sum())
    return mu[ks] + np.sqrt(var[ks]) * rng.standard_normal(n)


# ===================================================================
# tuples / linear tools
# ===================================================================
def tuples(Y, ST, rates, H, W_max):
    """At time t: F0 = [1, y_t, dy_t], bank m_t (k), target S = y_{t+1}-y_t,
    futures S_h, window of W_max past increments, hidden state at t,
    (trajectory, time) index."""
    D = np.diff(Y, axis=1)
    m, n_burn = ema_features(Y, rates, cfg.DT, cfg.BURN_TOL)
    L = Y.shape[1] - 1
    t0 = max(n_burn, W_max, 1)
    ts = np.arange(t0, L - H + 1)
    n = Y.shape[0]
    y = Y[:, ts]
    F0 = np.stack([np.ones_like(y), y, D[:, ts - 1]], -1).reshape(-1, 3)
    M = m[:, ts].reshape(-1, m.shape[2])
    S = D[:, ts].reshape(-1, 1)
    FUT = np.stack([D[:, ts + h - 1] for h in range(1, H + 1)], 2) \
        .reshape(-1, H, 1)
    WIN = np.stack([D[:, ts - 1 - j] for j in range(W_max)], 2) \
        .reshape(-1, W_max)
    HID = ST[:, ts].reshape(-1, ST.shape[2])
    TI = np.stack([np.repeat(np.arange(n), len(ts)), np.tile(ts, n)], 1)
    return dict(F0=F0, M=M, S=S, FUT=FUT, WIN=WIN, HID=HID, TI=TI,
                Y=y.reshape(-1), t0=t0)


def heldout_resid(Ftr, Ytr, Fte, Yte):
    Wls, *_ = np.linalg.lstsq(Ftr, Ytr, rcond=None)
    res = Yte - Fte @ Wls
    return float(np.sum(res.var(0) / np.maximum(Yte.var(0), 1e-15)))


def predictive_rank(F0, M, FUT, energy_rule):
    mu_f, mu_m = F0.mean(0), M.mean(0)
    Gam, *_ = np.linalg.lstsq(F0 - mu_f, M - mu_m, rcond=None)
    Mt = (M - mu_m) - (F0 - mu_f) @ Gam
    ev, U = np.linalg.eigh(np.cov(Mt.T))
    ev = np.maximum(ev, 1e-12 * ev.max())
    Wm = U @ np.diag(ev ** -0.5) @ U.T
    Uw = Mt @ Wm
    blocks = []
    for h in range(FUT.shape[1]):
        d = FUT[:, h, 0]
        fut = np.stack([d, d ** 2], 1)
        B, *_ = np.linalg.lstsq(F0 - mu_f, fut - fut.mean(0), rcond=None)
        fr = fut - fut.mean(0) - (F0 - mu_f) @ B
        fr = fr / np.maximum(fr.std(0), 1e-12)
        blocks.append(Uw.T @ fr / len(fr))
    Gs = np.hstack(blocks)
    Ug, sg, _ = np.linalg.svd(Gs, full_matrices=False)
    energy = np.cumsum(sg ** 2) / np.sum(sg ** 2)
    p = max(1, min(int(np.searchsorted(energy, energy_rule) + 1), len(sg)))
    return dict(spectrum=sg.tolist(), energy=energy.tolist(), p=p,
                mu_f=mu_f, mu_m=mu_m, Gam=Gam, Pmap=Ug[:, :p].T @ Wm)


def cond_vec(T, rank):
    Mt = (T["M"] - rank["mu_m"]) - (T["F0"] - rank["mu_f"]) @ rank["Gam"]
    return np.hstack([T["F0"][:, 1:2], Mt @ rank["Pmap"].T])


def wres(S, w, n, rng):
    return S[rng.choice(len(S), size=n, replace=True, p=w / w.sum())]


# ===================================================================
# per-anchor evaluation against the particle-filter predictive law
# ===================================================================
def anchor_eval(c, y_t, hid, pf_d1, pf_d2, pf_mu, pf_var, sampler, n_s, rng,
                P, k_mix):
    """All draws are INCREMENTS y_{t+1} - y_t. Returns excess errors
    E_norm and diagnostics."""
    d1, d2 = pf_d1 - y_t, pf_d2 - y_t
    floor = w2_1d(d2, d1)
    var = float(d1.var())
    E = lambda x: (w2_1d(x, d1) - floor) / var                # noqa: E731
    out = dict(floor=floor, var_pf=var, floor_over_var=floor / var)
    # sampler (training pool, shared neighbors) and local resampling
    idx_tr, w_tr, ess_tr, _ = sampler.neighbors(c)
    idx_tr, w_tr = idx_tr[0], w_tr[0]
    out["ess"] = float(ess_tr[0])
    S = sampler.S_data[idx_tr, 0]
    s_mod = sampler.sample_at(c, n_s, rng, idx=idx_tr, w=w_tr)[:, 0]
    out["E_sampler"] = E(s_mod)
    out["E_resample"] = E(wres(S, w_tr, n_s, rng))
    # ideal observable Gaussian head: PF predictive moments
    out["E_gauss_pf"] = E((pf_mu - y_t) + np.sqrt(pf_var)
                          * rng.standard_normal(n_s))
    # Gaussian with the kNN-matched moments (what a kNN Gaussian head gets)
    mu_k = np.average(S, weights=w_tr)
    var_k = np.average((S - mu_k) ** 2, weights=w_tr)
    out["E_gauss_knn"] = E(mu_k + np.sqrt(var_k) * rng.standard_normal(n_s))
    # observable hardness: mixture fitted on an independent PF draw
    pi, mu_m, var_m, _ = gmm1d_fit(d2, k_mix, rng, iters=100, restarts=1)
    out["E_kmix_pf"] = E(gmm1d_sample(pi, mu_m, var_m, n_s, rng))
    # hidden-state law (diagnostic: how far the posterior mixture is from it)
    out["E_hidden"] = E(md.oracle_sample(hid, n_s, rng, P) - y_t)
    # raw floor ratios (diagnostics only)
    out["sampler_over_floor"] = (out["E_sampler"] * var + floor) / floor
    out["gauss_pf_over_floor"] = (out["E_gauss_pf"] * var + floor) / floor
    # kNN Markov (y only) conditional variance: the nonlinear Markov
    # reference for the unstratified variance comparison
    out["_sampler_draw"] = s_mod
    return out


def select_anchors(TE, SZ, rng):
    """N_ANCH quiet + N_ANCH active + N_UNSTRAT unstratified anchors per
    record, drawn from N_ANCH_TRAJ trajectories at times >= PF_WARM
    (seeded, no post-hoc)."""
    traj = rng.choice(np.unique(TE["TI"][:, 0]), SZ["N_ANCH_TRAJ"],
                      replace=False)
    ok = np.isin(TE["TI"][:, 0], traj) & (TE["TI"][:, 1] >= SZ["PF_WARM"])
    yq = TE["Y"]
    qlo, qhi = np.quantile(yq, [cfg.Q_QUIET, cfg.Q_ACTIVE])
    quiet = np.nonzero(ok & (yq < qlo))[0]
    active = np.nonzero(ok & (yq > qhi))[0]
    a = rng.choice(quiet, SZ["N_ANCH"], replace=False)
    b = rng.choice(active, SZ["N_ANCH"], replace=False)
    u = rng.choice(np.nonzero(ok)[0], SZ["N_UNSTRAT"], replace=False)
    return traj, np.concatenate([a, b, u]), \
        np.array(["quiet"] * len(a) + ["active"] * len(b)
                 + ["unstrat"] * len(u))


def pf_draws(Y, TE, anchors, n_part, n_s, seed, P, guided=True, warm=0,
             traj_subset=None):
    """Run one particle filter per trajectory that carries anchors; at each
    anchor time t return two independent predictive draws of y_{t+1}, the
    predictive moments, and the weight-ESS statistics near the anchor.
    Also returns per-trajectory PIT values (calibration) and the
    weight-ESS quantiles over the whole run."""
    out, pits, ess_all = {}, [], []
    rng = np.random.default_rng(seed)
    by_traj = {}
    for j in anchors:
        i, t = TE["TI"][j]
        if traj_subset is not None and int(i) not in traj_subset:
            continue
        by_traj.setdefault(int(i), []).append((int(t), int(j)))
    for i, lst in by_traj.items():
        pf = ParticleFilter(P, n_part, rng, guided=guided, pit_from=warm)
        for t, j in sorted(lst):
            pf.run(Y[i], t)
            d1, d2 = pf.predictive(n_s), pf.predictive(n_s)
            # large draws for the replicate-convergence statistics (their
            # Monte-Carlo noise must be well below the 0.02 gate)
            c1, c2 = pf.predictive(cfg.N_CONV_DRAW), pf.predictive(cfg.N_CONV_DRAW)
            mu, var = pf.moments()
            near = pf.ess_near(t)
            out[j] = dict(d1=d1, d2=d2, c1=c1, c2=c2, mu=mu, var=var,
                          ess_near_min=float(near.min()),
                          ess_near_med=float(np.median(near)))
        pits.extend(p for _, p in pf.pit)
        ess_all.extend(pf.ess)
    return out, np.array(pits), np.array(ess_all)


# ===================================================================
def run(mode):
    SZ, G = cfg.SIZES[mode], cfg.GATES
    P = md.params()
    rng = np.random.default_rng(cfg.SEED_BASE)
    t_all = time.time()
    R = dict(mode=mode, sizes=SZ, regime=dict(TAU_D=cfg.TAU_D, GAMMA=cfg.GAMMA,
                                             TAU_R=cfg.TAU_R,
                                             EPS_NOISE=cfg.EPS_NOISE,
                                             DT=cfg.DT),
             params={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in P.items()})
    print("=" * 72)
    print(f"SOL filtered-Poisson pre-vet v2 (PF oracle)  mode = {mode}")
    print("=" * 72)
    cu = md.cumulants(P)
    atom = md.atom_prob()
    print(f"  closed form: mean {P['mean']:.3f}, var {P['var']:.3f}, skew "
          f"{cu['skew']:.3f}, flat {cu['flat']:.3f}, atom/step {atom:.3f}; "
          f"amplitudes {np.round(P['amp'], 3).tolist()}, rates "
          f"{np.round(P['nu'], 3).tolist()}, sigma_n {P['sig_n']:.3f}")
    R["closed_form"] = dict(skew=cu["skew"], flat=cu["flat"], atom=atom)

    # ---------------- G1: closed-form design table -------------------
    C_sig = md.acf_cont(np.arange(0, 4000) * cfg.DT, P)
    tau_sig = float(np.argmax(C_sig / C_sig[0] < cfg.ACF_SIG_THRESH) * cfg.DT)
    lam_min, lam_max = 1.0 / tau_sig, 1.0 / cfg.DT
    band = (lam_min, lam_max)
    L_need = md.tail_length([lam_min], cfg.DT, cfg.TAIL_TOL) + cfg.N_OUT_ACF + 10
    C = md.acf_sampled(L_need, P=P)
    V_full = md.cond_given_window(C, 400)
    V_full2 = md.cond_given_window(C, 800)
    V_mk = md.cond_given_window(C, 1)
    design = []
    for k in cfg.K_LIST:
        rates = rates_from_band(band, k)
        _, V = md.cond_given_memory(C, rates, cfg.DT, cfg.TAIL_TOL)
        eps = (V - V_full) / (V_mk - V_full)
        Cb = md.bank_ideal_acf(C, rates, cfg.DT, cfg.TAIL_TOL, cfg.N_OUT_ACF)
        Ca = md.ar_ideal_acf(C, k + 1, cfg.N_OUT_ACF)
        eb = md.acf_errors(Cb, C, cfg.slow_tail_lags())
        ea = md.acf_errors(Ca, C, cfg.slow_tail_lags())
        design.append(dict(k=k, dim=k + 1, eps_k=float(eps), bank_max=eb[0],
                           bank_slow=eb[1], ar_max=ea[0], ar_slow=ea[1],
                           rates=rates.tolist()))
    ks_ok = [d for d in design if d["eps_k"] <= G["EPS_K_MAX"]]
    dstar = ks_ok[0] if ks_ok else min(design, key=lambda d: d["eps_k"])
    k_star = dstar["k"]
    ratio_max = dstar["ar_max"] / max(dstar["bank_max"], 1e-300)
    ratio_slow = dstar["ar_slow"] / max(dstar["bank_slow"], 1e-300)
    g1 = dict(band=band, tau_sig=tau_sig, V_full=V_full, V_full_800=V_full2,
              V_markov=V_mk, design=design, k_star=k_star,
              ar_over_bank_max=ratio_max, ar_over_bank_slow=ratio_slow,
              markov_one_step_excess=(V_mk - V_full) / V_full)
    g1["pass"] = bool(dstar["eps_k"] <= G["EPS_K_MAX"]
                      and ratio_max >= G["BANK_VS_AR_MIN"])
    print(f"  G1 closed form: band [{lam_min:.4f}, {lam_max:.1f}] (slow end "
          f"{tau_sig:.1f} t.u.); V_markov/V_full - 1 = "
          f"{g1['markov_one_step_excess']:.4f}")
    for d in design:
        print(f"    k={d['k']} dim={d['dim']}: eps_k {d['eps_k']:.4f} | bank "
              f"{d['bank_max']:.4f} (slow {d['bank_slow']:.3f}) vs AR({d['dim']})"
              f" {d['ar_max']:.4f} (slow {d['ar_slow']:.3f})")
    print(f"    k* = {k_star}: AR/bank {ratio_max:.1f}x (max), {ratio_slow:.1f}x"
          f" (slow tail) -> G1 {'PASS' if g1['pass'] else 'FAIL'}")
    R["G1"] = g1
    rates = rates_from_band(band, k_star)

    # ---------------- data ------------------------------------------
    print("  simulating pools ...", flush=True)
    s0 = cfg.SEED_BASE
    Ytr, STtr, _ = md.simulate(SZ["N_TRAJ_TR"], SZ["L_REC"], s0 + 1, P=P)
    # (v1's independent kNN reference pool is no longer used: the
    # particle filter is the observable-history reference)
    tests =[md.simulate(SZ["N_TRAJ_TEST"], SZ["L_REC"], s0 + 10 + r, P=P)
             for r in range(SZ["N_TEST_REC"])]
    Cemp = acf_estimate(Ytr, SZ["L_REC"] // 2)
    sig = np.nonzero(Cemp[1:] / Cemp[0] > cfg.ACF_SIG_THRESH)[0]
    R["data_band"] = dict(slow_end_tu=float((sig[-1] + 1) * cfg.DT)
                          if len(sig) else None, lam_max=lam_max)

    # ---------------- G4: exactness/physics per record (SE-scaled) ----
    phys = []
    for r, (Y, ST, NA) in enumerate(tests):
        n = Y.shape[0]
        v_traj = Y.var(axis=1)
        se_var = float(v_traj.std() / np.sqrt(n))
        var_err_se = float(abs(Y.var() - P["var"]) / max(se_var, 1e-300))
        acf_traj = np.stack([acf_estimate(Y[i:i + 1], 50) for i in range(n)])
        acf_mean = acf_traj.mean(0)
        acf_se = acf_traj.std(0) / np.sqrt(n)
        acf_err_se = float(np.max(np.abs(acf_mean - C[:51])
                                  / np.maximum(acf_se, 1e-300)))
        ph = dict(mean=float(Y.mean()), var=float(Y.var()), se_var=se_var,
                  var_err_se=var_err_se, var_relerr=float(
                      abs(Y.var() - P["var"]) / P["var"]),
                  acf_err_se=acf_err_se,
                  acf_err_abs=float(np.max(np.abs(acf_mean - C[:51])) / C[0]),
                  skew=float(sk_skew(Y.ravel())),
                  flat=float(sk_kurt(Y.ravel(), fisher=False)),
                  atom=float((NA.sum(2) == 0).mean()))
        ph["pass"] = bool(
            G["SKEW_RANGE"][0] <= ph["skew"] <= G["SKEW_RANGE"][1]
            and G["FLAT_RANGE"][0] <= ph["flat"] <= G["FLAT_RANGE"][1]
            and G["ATOM_RANGE"][0] <= ph["atom"] <= G["ATOM_RANGE"][1]
            and ph["var_err_se"] <= G["N_SE"] and ph["acf_err_se"] <= G["N_SE"])
        print(f"  G4 record {r}: var err {ph['var_err_se']:.1f} SE (rel "
              f"{ph['var_relerr']:.3f}), ACF err {ph['acf_err_se']:.1f} SE "
              f"(abs {ph['acf_err_abs']:.3f}), skew {ph['skew']:.2f}, flat "
              f"{ph['flat']:.2f}, atom {ph['atom']:.3f} -> "
              f"{'PASS' if ph['pass'] else 'FAIL'}")
        phys.append(ph)
    R["G4"] = phys

    # ---------------- tuples, one-step report, rank -----------------
    kmax = max(cfg.K_LIST)
    TR = tuples(Ytr, STtr, rates, cfg.H_RANK, kmax)
    TEs = [tuples(Y, ST, rates, cfg.H_RANK, kmax) for Y, ST, _ in tests]
    gains = []
    for TE in TEs:
        r0 = heldout_resid(TR["F0"], TR["S"], TE["F0"], TE["S"])
        Fb, Fbe = np.hstack([TR["F0"], TR["M"]]), np.hstack([TE["F0"], TE["M"]])
        Fw = np.hstack([TR["F0"], TR["WIN"][:, :k_star]])
        Fwe = np.hstack([TE["F0"], TE["WIN"][:, :k_star]])
        Fo, Foe = np.hstack([TR["F0"], TR["HID"]]), np.hstack([TE["F0"], TE["HID"]])
        gains.append(dict(
            bank=1 - heldout_resid(Fb, TR["S"], Fbe, TE["S"]) / r0,
            window=1 - heldout_resid(Fw, TR["S"], Fwe, TE["S"]) / r0,
            hidden=1 - heldout_resid(Fo, TR["S"], Foe, TE["S"]) / r0))
    R["one_step_linear_gains"] = gains
    print("  one-step linear OLS gains (report only): "
          + "; ".join(f"bank {g['bank']:.3f} / window {g['window']:.3f} / "
                      f"hidden {g['hidden']:.3f}" for g in gains))
    rank = predictive_rank(TR["F0"], TR["M"], TR["FUT"], cfg.ENERGY_RULE)
    R["rank"] = dict(spectrum=rank["spectrum"], energy=rank["energy"],
                     p=rank["p"], k_star=k_star)
    print(f"  predictive spectrum {np.round(rank['spectrum'], 3).tolist()} "
          f"-> p = {rank['p']}")

    # ---------------- sampler ---------------------------------------
    C_tr = cond_vec(TR, rank)
    sampler = TrainingFreeSamplerNP(C_tr, TR["S"], cfg.J_NEIGHBORS, cfg.NU,
                                    cfg.EPS_METRIC, SZ["N_ODE"])
    if not (np.isfinite(sampler.Lmap).all() and np.isfinite(sampler.Chat).all()):
        raise FloatingPointError("non-finite sampler metric")
    R["pool_sizes"] = dict(train=int(len(C_tr)), cond_dim=int(C_tr.shape[1]))
    print(f"  sampler pool {len(C_tr)}, cond dim {C_tr.shape[1]}", flush=True)

    # ---------------- anchors + particle filters --------------------
    # kNN Markov (y only) reference for the unstratified variance check
    sp_markov = TrainingFreeSamplerNP(TR["F0"][:, 1:2], TR["S"], cfg.J_NEIGHBORS,
                                      cfg.NU, cfg.EPS_METRIC, 10)
    rows, rep = [], []
    pits_all, ess_runs = [], []
    t0 = time.time()
    for rid, (TE, (Y, ST, _)) in enumerate(zip(TEs, tests)):
        traj, anchors, strata = select_anchors(TE, SZ, rng)
        C_te = cond_vec(TE, rank)
        pfd, pits, ess_run = pf_draws(Y, TE, anchors, SZ["PF_N"], SZ["N_SAMP"],
                                      cfg.SEED_BASE + 100 + rid, P, guided=True,
                                      warm=SZ["PF_WARM"])
        pits_all.append(pits)
        ess_runs.append(ess_run)
        print(f"    record {rid}: guided PF on {len(traj)} trajectories, "
              f"weight-ESS p05 {np.percentile(ess_run, 5):.0f} / median "
              f"{np.median(ess_run):.0f} / min {ess_run.min():.0f}; PIT mean "
              f"{pits.mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        for j, st in zip(anchors, strata):
            d = pfd[j]
            o = anchor_eval(C_te[j], TE["Y"][j], TE["HID"][j], d["d1"], d["d2"],
                            d["mu"], d["var"], sampler, SZ["N_SAMP"], rng, P,
                            cfg.K_MIX)
            _, hv = md.oracle_moments(TE["HID"][j], P)
            im, wm, _, _ = sp_markov.neighbors(TE["F0"][j, 1:2])
            Sm, wm = sp_markov.S_data[im[0], 0], wm[0]
            mum = np.average(Sm, weights=wm)
            o.update(real=rid, stratum=st, y=float(TE["Y"][j]),
                     pf_ess_near_min=d["ess_near_min"],
                     pf_ess_near_med=d["ess_near_med"],
                     var_pf=d["var"], var_hidden=float(hv[0]),
                     var_markov_knn=float(np.average((Sm - mum) ** 2,
                                                     weights=wm)))
            rows.append(o)
        # replicate filters on a subset of trajectories: guided N (other
        # seed), guided 2N, bootstrap 2N -- compared at those anchors
        sub_traj = set(int(x) for x in traj[:SZ["N_CONV_TRAJ"]])
        reps = {}
        for name, n_part, guided, sd in (("guided_N2", SZ["PF_N"], True, 300),
                                         ("guided_2N", 2 * SZ["PF_N"], True, 400),
                                         ("boot_2N", 2 * SZ["PF_N"], False, 500)):
            reps[name], _, _ = pf_draws(Y, TE, anchors, n_part, SZ["N_SAMP"],
                                        cfg.SEED_BASE + sd + rid, P,
                                        guided=guided, traj_subset=sub_traj)
        for j, st in zip(anchors, strata):
            if j not in reps["guided_2N"]:
                continue
            base = pfd[j]
            v = base["c1"].var()
            fl = w2_1d(base["c2"], base["c1"])
            r0 = next(r for r in rows if r["real"] == rid
                      and r["y"] == float(TE["Y"][j]) and r["stratum"] == st)
            ent = dict(real=rid, stratum=st, y=float(TE["Y"][j]))
            for name, rd in reps.items():
                d = rd[j]
                ent[name] = dict(
                    w2_over_var=(w2_1d(d["c1"], base["c1"]) - fl) / v,
                    dmean_over_sd=(d["mu"] - base["mu"]) / np.sqrt(base["var"]),
                    var_ratio=d["var"] / base["var"])
                # G2 ratio recomputed against this replicate reference
                dd1, dd2 = d["d1"] - TE["Y"][j], d["d2"] - TE["Y"][j]
                flr = w2_1d(dd2, dd1)
                vr = dd1.var()
                Es_r = (w2_1d(r0["_sampler_draw"], dd1) - flr) / vr
                # the ideal observable Gaussian head of THIS replicate:
                # its own predictive moments (codeX correction)
                gm, gv = d["mu"] - TE["Y"][j], d["var"]
                Eg_r = (w2_1d(gm + np.sqrt(gv) * rng.standard_normal(
                    SZ["N_SAMP"]), dd1) - flr) / vr
                ent[name]["E_sampler"], ent[name]["E_gauss_pf"] = Es_r, Eg_r
            rep.append(ent)
    for r in rows:                       # drop the stored draws
        r.pop("_sampler_draw")
    print(f"    {len(rows)} anchors ({len(rep)} with replicate filters) in "
          f"{time.time() - t0:.0f} s")
    arr = lambda key: np.array([r[key] for r in rows])         # noqa: E731
    strat = arr("stratum")
    act, uns = strat == "active", strat == "unstrat"
    qa = ~uns                                   # quiet + active (stratified)
    rid = arr("real")
    med = lambda v, m=None: float(np.median(v if m is None else v[m]))  # noqa
    Es, Eg, Egk, Er, Ek, Eh = (arr("E_sampler"), arr("E_gauss_pf"),
                               arr("E_gauss_knn"), arr("E_resample"),
                               arr("E_kmix_pf"), arr("E_hidden"))
    ess = arr("ess")
    ratio_pa = Eg / np.maximum(Es, 0.01)          # per-anchor (clipped)
    # reference reliability
    pits = np.concatenate(pits_all)
    pit_ks = float(np.max(np.abs(np.sort(pits) - (np.arange(len(pits)) + 0.5)
                                 / len(pits))))
    ess_run = np.concatenate(ess_runs)
    repstat = {}
    for name in ("guided_N2", "guided_2N", "boot_2N"):
        w = np.array([e[name]["w2_over_var"] for e in rep])
        dm = np.array([e[name]["dmean_over_sd"] for e in rep])
        vr = np.array([e[name]["var_ratio"] for e in rep])
        es = np.array([e[name]["E_sampler"] for e in rep])
        eg = np.array([e[name]["E_gauss_pf"] for e in rep])
        repstat[name] = dict(w2_over_var_med=med(w), w2_over_var_p90=float(
            np.percentile(w, 90)), dmean_over_sd_p90=float(np.percentile(
                np.abs(dm), 90)), var_ratio_p10=float(np.percentile(vr, 10)),
            var_ratio_p90=float(np.percentile(vr, 90)),
            E_sampler_med=med(es), E_gauss_pf_med=med(eg),
            gauss_over_sampler=med(eg) / max(med(es), 1e-9))
    # unstratified variance comparison, paired per anchor
    dv = arr("var_pf")[uns] - arr("var_markov_knn")[uns]
    unstrat = dict(n=int(uns.sum()),
                   var_pf_mean=float(arr("var_pf")[uns].mean()),
                   var_markov_knn_mean=float(arr("var_markov_knn")[uns].mean()),
                   var_hidden_mean=float(arr("var_hidden")[uns].mean()),
                   diff_mean=float(dv.mean()),
                   diff_se=float(dv.std(ddof=1) / np.sqrt(len(dv))),
                   markov_var_closed_form=V_mk, full_linear_var=V_full)
    unstrat["pass"] = bool(unstrat["diff_mean"] <= G["N_SE_PF"] * unstrat["diff_se"])
    g23 = dict(
        n_anchors=len(rows), n_stratified=int(qa.sum()),
        reference=dict(pf_weight_ess_p05=float(np.percentile(ess_run, 5)),
                       pf_weight_ess_p10=float(np.percentile(ess_run, 10)),
                       pf_weight_ess_med=float(np.median(ess_run)),
                       pf_weight_ess_min=float(ess_run.min()),
                       pf_ess_near_anchor_min_p10=float(np.percentile(
                           arr("pf_ess_near_min"), 10)),
                       pf_ess_near_anchor_med=med(arr("pf_ess_near_med")),
                       pit_mean=float(pits.mean()), pit_ks=pit_ks,
                       pit_n=int(len(pits)), replicates=repstat,
                       unstratified=unstrat),
        E_sampler=med(Es, qa), E_sampler_quiet=med(Es, ~act & qa),
        E_sampler_active=med(Es, act), E_sampler_unstrat=med(Es, uns),
        E_resample=med(Er, qa), E_gauss_pf=med(Eg, qa),
        E_gauss_pf_quiet=med(Eg, ~act & qa), E_gauss_pf_active=med(Eg, act),
        E_gauss_pf_unstrat=med(Eg, uns), E_gauss_knn=med(Egk, qa),
        E_kmix_pf=med(Ek, qa), E_hidden=med(Eh, qa),
        gauss_over_sampler=med(Eg, qa) / max(med(Es, qa), 1e-9),
        gauss_over_sampler_quiet=med(Eg, ~act & qa) / max(med(Es, ~act & qa), 1e-9),
        gauss_over_sampler_active=med(Eg, act) / max(med(Es, act), 1e-9),
        gauss_over_sampler_unstrat=med(Eg, uns) / max(med(Es, uns), 1e-9),
        gauss_over_sampler_per_anchor_med=med(ratio_pa, qa),
        sampler_over_floor=med(arr("sampler_over_floor"), qa),
        gauss_pf_over_floor=med(arr("gauss_pf_over_floor"), qa),
        floor_over_var=med(arr("floor_over_var"), qa),
        ess_med=med(ess), ess_p05=float(np.percentile(ess, 5)),
        per_real=[dict(E_sampler=med(Es, (rid == r) & qa),
                       gauss_over_sampler=med(Eg, (rid == r) & qa)
                       / max(med(Es, (rid == r) & qa), 1e-9),
                       ess_med=med(ess, rid == r),
                       ess_p05=float(np.percentile(ess[rid == r], 5)))
                  for r in range(len(TEs))],
        anchors=rows, replicate_anchors=rep)
    ref = g23["reference"]
    g23["G2_pass"] = bool(g23["gauss_over_sampler"] >= G["GAUSS_OVER_SAMPLER_MIN"])
    g23["ref_pass"] = bool(
        ref["replicates"]["guided_2N"]["w2_over_var_p90"] <= G["PF_CONV_P90_MAX"]
        and abs(ref["pit_mean"] - 0.5) <= G["PIT_MEAN_TOL"]
        and ref["pit_ks"] <= G["PIT_KS_MAX"] and unstrat["pass"])
    g23["G3_pass"] = bool(g23["E_sampler"] <= G["E_NORM_MAX"] and g23["ref_pass"]
                          and g23["ess_med"] >= G["ESS_MED_MIN"]
                          and g23["ess_p05"] >= G["ESS_P05_MIN"])
    g23["G2_pass_per_real"] = [bool(d["gauss_over_sampler"]
                                    >= G["GAUSS_OVER_SAMPLER_MIN"])
                               for d in g23["per_real"]]
    g23["G3_pass_per_real"] = [bool(d["E_sampler"] <= G["E_NORM_MAX"]
                                    and d["ess_med"] >= G["ESS_MED_MIN"]
                                    and d["ess_p05"] >= G["ESS_P05_MIN"])
                               for d in g23["per_real"]]
    R["G23"] = g23
    print(f"  reference: guided PF weight-ESS p05/p10/med/min "
          f"{ref['pf_weight_ess_p05']:.0f}/{ref['pf_weight_ess_p10']:.0f}/"
          f"{ref['pf_weight_ess_med']:.0f}/{ref['pf_weight_ess_min']:.0f}; near "
          f"anchors min-p10 {ref['pf_ess_near_anchor_min_p10']:.0f}, med "
          f"{ref['pf_ess_near_anchor_med']:.0f}; PIT mean {ref['pit_mean']:.3f} KS "
          f"{ref['pit_ks']:.3f} (n={ref['pit_n']})")
    for name, s in ref["replicates"].items():
        print(f"    replicate {name:10s}: W2/var med {s['w2_over_var_med']:.4f} "
              f"p90 {s['w2_over_var_p90']:.4f}; |dmean|/sd p90 "
              f"{s['dmean_over_sd_p90']:.3f}; var ratio p10-p90 "
              f"[{s['var_ratio_p10']:.3f}, {s['var_ratio_p90']:.3f}]; G2 ratio "
              f"with this reference {s['gauss_over_sampler']:.2f} "
              f"(E_s {s['E_sampler_med']:.3f}, E_g {s['E_gauss_pf_med']:.3f})")
    print(f"  unstratified anchors (n={unstrat['n']}): E[Var|history] (PF) "
          f"{unstrat['var_pf_mean']:.3f} vs E[Var|Y_n] (kNN Markov) "
          f"{unstrat['var_markov_knn_mean']:.3f} (paired diff "
          f"{unstrat['diff_mean']:+.4f} +- {unstrat['diff_se']:.4f}); hidden-state "
          f"{unstrat['var_hidden_mean']:.3f}; closed-form Markov "
          f"{unstrat['markov_var_closed_form']:.3f}, best linear full history "
          f"{unstrat['full_linear_var']:.3f} -> {'PASS' if unstrat['pass'] else 'FAIL'}")
    print(f"  excess errors E_norm (median, stratified anchors): sampler "
          f"{g23['E_sampler']:.3f} (quiet {g23['E_sampler_quiet']:.3f}, active "
          f"{g23['E_sampler_active']:.3f}, unstratified {g23['E_sampler_unstrat']:.3f})"
          f"; resample {g23['E_resample']:.3f}; Gaussian[PF moments] "
          f"{g23['E_gauss_pf']:.3f} (unstratified {g23['E_gauss_pf_unstrat']:.3f}); "
          f"Gaussian[kNN moments] {g23['E_gauss_knn']:.3f}; K{cfg.K_MIX}[PF] "
          f"{g23['E_kmix_pf']:.3f}; hidden-state law {g23['E_hidden']:.3f}")
    print(f"  G2 Gaussian/sampler excess ratio {g23['gauss_over_sampler']:.2f} "
          f"(quiet {g23['gauss_over_sampler_quiet']:.2f}, active "
          f"{g23['gauss_over_sampler_active']:.2f}, unstratified "
          f"{g23['gauss_over_sampler_unstrat']:.2f}; per-anchor median "
          f"{g23['gauss_over_sampler_per_anchor_med']:.2f}; per record "
          f"{[round(d['gauss_over_sampler'], 2) for d in g23['per_real']]}) -> "
          f"{'PASS' if g23['G2_pass'] else 'FAIL'}")
    print(f"  G3 E_sampler {g23['E_sampler']:.3f} (per record "
          f"{[round(d['E_sampler'], 3) for d in g23['per_real']]}); reference "
          f"{'OK' if g23['ref_pass'] else 'NOT RELIABLE'}; label ESS med "
          f"{g23['ess_med']:.0f} p05 {g23['ess_p05']:.0f}; raw ratios: sampler "
          f"{g23['sampler_over_floor']:.2f}x floor, Gaussian[PF] "
          f"{g23['gauss_pf_over_floor']:.2f}x -> {'PASS' if g23['G3_pass'] else 'FAIL'}")

    gates = dict(G1=g1["pass"], G2=g23["G2_pass"], G3=g23["G3_pass"],
                 G4=all(p["pass"] for p in phys),
                 G5=(all(p["pass"] for p in phys) and all(g23["G2_pass_per_real"])
                     and all(g23["G3_pass_per_real"])))
    R["gates"] = gates
    R["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    R["runtime_s"] = time.time() - t_all
    print(f"  GATES {gates} -> {R['verdict']}   ({R['runtime_s']:.0f} s)")
    aux = dict(rows=rows, act=act, qa=qa, Es=Es, Eg=Eg, Egk=Egk, Er=Er, Ek=Ek,
               Eh=Eh, ess=ess, P=P, tests=tests, TEs=TEs, rng=rng, Ytr=Ytr,
               sampler=sampler, rank=rank, pits=pits)
    return R, aux


# ===================================================================
def figure(R, aux, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from memdiff.plotting import setup_style
    setup_style()
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    ax = ax.ravel()
    P, G = aux["P"], cfg.GATES
    d = R["G1"]["design"]
    ks = [x["k"] for x in d]
    a = ax[0]
    a.semilogy(ks, [x["bank_max"] for x in d], "o-", label="ideal bank, max")
    a.semilogy(ks, [x["ar_max"] for x in d], "s-", label="ideal AR($k$+1), max")
    a.semilogy(ks, [x["bank_slow"] for x in d], "o--", label="bank, slow tail")
    a.semilogy(ks, [x["ar_slow"] for x in d], "s--", label="AR, slow tail")
    a.set_xlabel("$k$ (conditioning dim $k+1$)")
    a.set_ylabel("closed-loop ACF error")
    a.set_title("(a) memory representation (closed form)")
    a.legend(fontsize=7)
    a = ax[1]
    a.semilogy(ks, [x["eps_k"] for x in d], "o-")
    a.axhline(G["EPS_K_MAX"], color="r", ls="--", lw=1)
    a.set_xlabel("$k$")
    a.set_ylabel(r"one-step linear sufficiency $\epsilon_k$")
    a.set_title(f"(b) $\\epsilon_k$; $k^*={R['G1']['k_star']}$, $p={R['rank']['p']}$")
    ins = a.inset_axes([0.5, 0.5, 0.45, 0.45])
    ins.semilogy(np.arange(1, len(R["rank"]["spectrum"]) + 1),
                 R["rank"]["spectrum"], "o-", ms=3)
    ins.set_title("predictive spectrum", fontsize=7)
    a = ax[2]
    Y, ST, _ = aux["tests"][0]
    n_show = int(100 / cfg.DT)
    t = np.arange(n_show) * cfg.DT
    a.plot(t, Y[0, :n_show], "k", lw=1, label="$y$ (observed)")
    for j in range(len(P["a"])):
        a.plot(t, P["c"][j] * (ST[0, :n_show, 2 * j] - ST[0, :n_show, 2 * j + 1]),
               lw=1, alpha=0.8, label=f"hidden $\\Phi_{j+1}$ ($\\tau_d$={P['a'][j]:g})")
    a.set_xlabel("$t/\\tau_{d1}$")
    a.set_title("(c) signal and hidden populations")
    a.legend(fontsize=7)
    # (d) one-step laws at one active anchor: PF predictive vs hidden vs
    # Gaussian[PF] vs sampler
    a = ax[3]
    rows = aux["rows"]
    rng = aux["rng"]
    cand = [r for r in rows if r["stratum"] == "active"]
    r0 = cand[len(cand) // 2]
    TE = aux["TEs"][r0["real"]]
    Yr = aux["tests"][r0["real"]][0]
    j = int(np.argmin(np.abs(TE["Y"] - r0["y"])))
    i_tr, t_an = TE["TI"][j]
    pf = ParticleFilter(P, 8000, rng, guided=True)
    pf.run(Yr[i_tr], int(t_an))
    dpf = pf.predictive(20000) - TE["Y"][j]
    mu, var = pf.moments()
    dh = md.oracle_sample(TE["HID"][j], 20000, rng, P) - TE["Y"][j]
    c = cond_vec({k: TE[k][j:j + 1] for k in ("F0", "M")}, aux["rank"])[0]
    ds = aux["sampler"].sample_at(c, 4000, rng)[:, 0]
    xs = np.linspace(min(dpf.min(), dh.min()), max(dpf.max(), dh.max()), 400)
    a.hist(dpf, bins=120, density=True, color="C0", alpha=0.4,
           label="observable law (PF)")
    a.hist(dh, bins=120, density=True, histtype="step", color="C3", lw=1,
           label="hidden-state law")
    a.hist(ds, bins=80, density=True, histtype="step", color="C2", lw=1.2,
           label="raw sampler")
    a.plot(xs, np.exp(-0.5 * (xs - (mu - TE["Y"][j])) ** 2 / var)
           / np.sqrt(2 * np.pi * var), "k", lw=1.2, label="Gaussian[PF moments]")
    a.set_xlabel(r"$\Delta y$")
    a.set_title("(d) one-step law at an active anchor")
    a.legend(fontsize=7)
    a = ax[4]
    act, qa = aux["act"], aux["qa"]
    qu = ~act & qa
    data = [aux["Es"][qu], aux["Eg"][qu], aux["Es"][act], aux["Eg"][act]]
    a.boxplot(data)
    a.set_xticks(range(1, 5))
    a.set_xticklabels(["sampler\nquiet", "Gauss[PF]\nquiet", "sampler\nactive",
                       "Gauss[PF]\nactive"])
    a.axhline(G["E_NORM_MAX"], color="r", ls="--", lw=1)
    a.set_yscale("symlog", linthresh=0.01)
    a.set_ylabel("excess error $E_{norm}$")
    a.set_title("(e) sampler vs ideal Gaussian head")
    a = ax[5]
    a.boxplot([aux["Er"], aux["Egk"], aux["Ek"], aux["Eh"]])
    a.set_xticks(range(1, 5))
    a.set_xticklabels(["resample", "Gauss[kNN]", f"K{cfg.K_MIX}[PF]", "hidden\nlaw"])
    a.set_yscale("symlog", linthresh=0.01)
    a.set_ylabel("excess error $E_{norm}$")
    a.set_title("(f) diagnostics")
    a = ax[6]
    a.hist(aux["pits"], bins=20, range=(0, 1), density=True, color="C0",
           alpha=0.7)
    a.axhline(1.0, color="k", lw=0.8)
    a.set_xlabel("PIT of realized $y_{t+1}$ under the PF predictive law")
    a.set_title(f"(g) PF calibration (n={len(aux['pits'])})")
    ins = a.inset_axes([0.55, 0.55, 0.42, 0.4])
    ins.boxplot([aux["ess"][qu], aux["ess"][act]])
    ins.set_xticks([1, 2])
    ins.set_xticklabels(["quiet", "active"], fontsize=7)
    ins.set_yscale("log")
    ins.set_title("label ESS", fontsize=7)
    a = ax[7]
    a.hist(aux["Ytr"].ravel(), bins=150, density=True, color="0.4")
    a.set_yscale("log")
    a.set_xlabel("$y$")
    a.set_title(f"(h) stationary PDF: skew {R['G4'][0]['skew']:.2f} "
                f"(cf {R['closed_form']['skew']:.2f}), flat "
                f"{R['G4'][0]['flat']:.2f} (cf {R['closed_form']['flat']:.2f})",
                fontsize=9)
    txt = "  ".join(f"{k}:{'P' if v else 'F'}" for k, v in R["gates"].items())
    fig.suptitle(f"SOL filtered-Poisson pre-vet v2 ({R['mode']}) -- {txt} -> "
                 f"{R['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    mode = "full" if args.full else "quick"
    np.seterr(all="ignore")      # macOS BLAS subnormal flags; finiteness
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)   # asserted
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    R, aux = run(mode)
    figure(R, aux, os.path.join(HERE, "figs", f"prevet_{mode}.pdf"))
    with open(os.path.join(HERE, "out", f"prevet_{mode}.json"), "w") as f:
        json.dump(_jsonable(dict(gates=cfg.GATES, result=R)), f, indent=1)
    print(f"report: out/prevet_{mode}.json, figs/prevet_{mode}.pdf")


if __name__ == "__main__":
    main()
