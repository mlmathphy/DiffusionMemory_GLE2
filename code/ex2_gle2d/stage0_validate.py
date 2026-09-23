"""Stage 0: estimated-coordinate sampling + rollout validation (numpy).

Runs BEFORE any training and ABORTS the pipeline (nonzero exit) if a
predeclared check fails (codeX: verify estimated-coordinate sampling
and rollout against exact references before committing to training).

Checks (thresholds in config.py, declared before any run):
  0a  q_hat vs oracle on held-out data       (reported, no threshold)
  0b  one-step sampling at 48 anchors        moment W2 + projected W1,
                                             the v2 gate-D criteria
  0c  sampler-in-the-loop rollout            full bank recursion,
      (kNN sampler autoregressive, no net)   compression at query
                                             time: Cov(v), VACF
                                             matrix, sampled-
                                             displacement diffusion
                                             tensor vs exact, with a
                                             same-size EXACT ensemble
                                             as the noise floor

Approx cost: one-step ~5-8 min + rollout ~10-20 min, numpy only.
"""

import json
import os
import sys

import numpy as np

import config as cfg
import exact_refs
from conditioning import (bank_features, build_tuples, fit_compression,
                          qhat, raw_tuples)
from generate_data import simulate
from memdiff.sampler_np import TrainingFreeSamplerNP
from scout import gauss_w2sq, proj_w1_stat, psd_inv_sqrt, psd_sqrt, \
    jsonable


def vacf_matrix(V, max_lag):
    """Cov(v_{t+m}, v_t) matrix estimate, (max_lag+1, 2, 2).

    V: (n_traj, L, 2), assumed mean-zero up to sampling error (the
    stationary mean is removed with the global mean per component).
    """
    V = np.asarray(V, float)
    n, L, _ = V.shape
    max_lag = min(max_lag, L - 1)
    Vc = V - V.reshape(-1, 2).mean(axis=0)
    nfft = int(2 ** np.ceil(np.log2(2 * L)))
    F = np.fft.rfft(Vc, n=nfft, axis=1)
    C = np.empty((max_lag + 1, 2, 2))
    for a in range(2):
        for b in range(2):
            cc = np.fft.irfft((F[:, :, a] * np.conj(F[:, :, b])
                               ).mean(axis=0), n=nfft)[:max_lag + 1]
            C[:, a, b] = cc / (L - np.arange(max_lag + 1))
    return C


def diff_sampled(C):
    """Sampled-displacement diffusion tensor from a VACF estimate."""
    tail = C[1:]
    return 0.5 * cfg.DT * (C[0] + tail.sum(axis=0)
                           + tail.transpose(0, 2, 1).sum(axis=0))


def rollout_sampler(sampler, coef, rates, kappa_s, v0, m0, n_steps,
                    seed):
    """kNN-sampler autoregressive rollout: full bank recursion,
    compression at query time. v0 (n,2), m0 (n,2k) -> (n, n_steps, 2).
    """
    rng = np.random.default_rng(seed)
    rho = np.repeat(np.exp(-np.asarray(rates) * cfg.DT), 2)
    v, m = v0.copy(), m0.copy()
    out = np.empty((len(v0), n_steps, 2))
    for t in range(n_steps):
        Z = np.concatenate([v, m], axis=1)
        s, _, _ = sampler.sample_labels(
            qhat(Z, coef), rng=rng)
        dv = s / kappa_s
        v = v + dv
        m = rho[None, :] * m + np.tile(dv, len(rates))
        out[:, t] = v
        if (t + 1) % 500 == 0:
            print(f"      rollout step {t + 1}/{n_steps}")
    return out


def floor_only():
    """Standalone cheap reference-estimator check (codeX): the exact
    same-size ensemble ONLY — no data.npz, no sampler, ~1 min. Tells
    us whether the planned stage-0 rollout validation can resolve the
    declared tolerances before anything expensive is spent."""
    refs = exact_refs.references()
    Sv, m_acf = refs["Sv"], refs["m_acf"]
    scale = float(np.max(np.abs(Sv[0])))
    m_trunc = min(max(m_acf, int(round(10 * cfg.TAU2 / cfg.DT))),
                  cfg.S0_N_ROLL - 1)
    m_acf_cmp = min(m_acf, m_trunc)
    D_ref = diff_sampled(Sv[:m_trunc + 1])
    V = simulate(cfg.S0_N_ENS, cfg.S0_N_ROLL,
                 cfg.SEED_STAGE0 + 2000, refs)[:, 1:]
    C = vacf_matrix(V, m_trunc)
    floor = dict(
        cov_dev=float(np.max(np.abs(np.cov(V.reshape(-1, 2).T)
                                    - cfg.KBT * np.eye(2)))),
        vacf_err=float(max(np.max(np.abs(C[m] - Sv[m]))
                           for m in range(m_acf_cmp + 1))) / scale,
        diff_err=float(np.linalg.norm(diff_sampled(C) - D_ref)
                       / np.linalg.norm(D_ref)))
    tols = dict(cov_dev=cfg.S0_COV_TOL, vacf_err=cfg.S0_VACF_TOL,
                diff_err=cfg.S0_DIFF_TOL)
    resolvable = {k: bool(floor[k] <= cfg.S0_RESOLVE_FRAC * t)
                  for k, t in tols.items()}
    out = dict(floor=floor, tolerances=tols,
               resolve_frac=cfg.S0_RESOLVE_FRAC,
               resolvable=resolvable, n_ens=cfg.S0_N_ENS,
               n_steps=cfg.S0_N_ROLL, m_trunc=m_trunc,
               deterministic_truncation_rel_err=float(
                   np.linalg.norm(D_ref - refs["D_exact"])
                   / np.linalg.norm(refs["D_exact"])))
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUT_DIR, "stage0_floor.json"),
              "w") as f:
        json.dump(jsonable(out), f, indent=1)
    print("floor:", floor)
    print("resolvable:", resolvable)
    print("RESOLVABLE: full stage 0 can judge the declared "
          "tolerances" if all(resolvable.values()) else
          "NOT RESOLVABLE: stage 0 would report INCONCLUSIVE; agree "
          "on a larger stage-0 ensemble before running it")
    print(f"record: {cfg.OUT_DIR}/stage0_floor.json")


def cutoff_scan():
    """Diffusion-estimator lag-cutoff analysis (codeX): exact
    calculations and exact simulated trajectories ONLY, ~1 min.

    For each cutoff M: the DETERMINISTIC truncation error of the
    truncated reference vs the exact D (is the check still measuring
    the diffusion tensor?), and the statistical floor of the
    same-budget estimator over S0_FLOOR_REPS independent exact
    ensembles (rms + max — a Monte-Carlo floor, not a single draw).
    Verdict: the smallest M that is both MEANINGFUL (truncation
    <= S0_TRUNC_MEANINGFUL) and RESOLVABLE (rms floor <=
    S0_RESOLVE_FRAC x S0_DIFF_TOL); if none exists at this budget,
    the recorded recommendation is to defer diffusion to the final
    evaluation and narrow stage 0 — a protocol amendment for the
    three-way loop, not something this script enacts.
    """
    refs = exact_refs.references()
    Sv, m_acf = refs["Sv"], refs["m_acf"]
    scale = float(np.max(np.abs(Sv[0])))
    cutoffs = [m for m in cfg.S0_CUTOFF_LIST
               if m <= cfg.S0_N_ROLL - 1]
    m_max = max(cutoffs)
    m_acf_cmp = min(m_acf, m_max)
    D_refs = {M: diff_sampled(Sv[:M + 1]) for M in cutoffs}
    trunc = {M: float(np.linalg.norm(D_refs[M] - refs["D_exact"])
                      / np.linalg.norm(refs["D_exact"]))
             for M in cutoffs}
    errs = {M: [] for M in cutoffs}
    cov_fl, vacf_fl = [], []
    for r in range(cfg.S0_FLOOR_REPS):
        V = simulate(cfg.S0_N_ENS, cfg.S0_N_ROLL,
                     cfg.SEED_STAGE0 + 2000 + 100 * r, refs)[:, 1:]
        C = vacf_matrix(V, m_max)
        cov_fl.append(float(np.max(np.abs(
            np.cov(V.reshape(-1, 2).T) - cfg.KBT * np.eye(2)))))
        vacf_fl.append(float(max(np.max(np.abs(C[m] - Sv[m]))
                                 for m in range(m_acf_cmp + 1)))
                       / scale)
        for M in cutoffs:
            Dh = diff_sampled(C[:M + 1])
            errs[M].append(float(np.linalg.norm(Dh - D_refs[M])
                                 / np.linalg.norm(D_refs[M])))
    table, chosen = {}, None
    lim = cfg.S0_RESOLVE_FRAC * cfg.S0_DIFF_TOL
    print(f"{'M':>5}{'trunc_err':>11}{'floor_rms':>11}"
          f"{'floor_max':>11}{'meaningful':>12}{'resolvable':>12}")
    for M in cutoffs:
        e = np.asarray(errs[M])
        rms, mx = float(np.sqrt(np.mean(e ** 2))), float(e.max())
        meaningful = bool(trunc[M] <= cfg.S0_TRUNC_MEANINGFUL)
        resolvable = bool(rms <= lim)
        table[M] = dict(trunc_err=trunc[M], floor_rms=rms,
                        floor_max=mx, floor_all=errs[M],
                        meaningful=meaningful, resolvable=resolvable)
        if meaningful and resolvable and chosen is None:
            chosen = M
        print(f"{M:>5}{trunc[M]:>11.4f}{rms:>11.4f}{mx:>11.4f}"
              f"{str(meaningful):>12}{str(resolvable):>12}")
    out = dict(cutoffs=table, chosen=chosen,
               reps=cfg.S0_FLOOR_REPS,
               n_ens=cfg.S0_N_ENS, n_steps=cfg.S0_N_ROLL,
               trunc_meaningful=cfg.S0_TRUNC_MEANINGFUL,
               resolve_limit=lim,
               cov_floor_rms=float(np.sqrt(np.mean(
                   np.asarray(cov_fl) ** 2))),
               vacf_floor_rms=float(np.sqrt(np.mean(
                   np.asarray(vacf_fl) ** 2))),
               recommendation=(
                   f"freeze diffusion cutoff M={chosen}"
                   if chosen is not None else
                   "no cutoff is both meaningful and resolvable at "
                   "this budget: defer diffusion to the final "
                   "evaluation and narrow stage 0 to one-step + "
                   "covariance + VACF (protocol amendment to agree)"))
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUT_DIR, "diffusion_cutoff_scan.json"),
              "w") as f:
        json.dump(jsonable(out), f, indent=1)
    print(f"cov floor rms {out['cov_floor_rms']:.4f}  "
          f"vacf floor rms {out['vacf_floor_rms']:.4f}")
    print("verdict:", out["recommendation"])
    print(f"record: {cfg.OUT_DIR}/diffusion_cutoff_scan.json")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    refs = exact_refs.references()
    rates, band = refs["rates"], refs["band"]
    Gamma, Vb1 = refs["Gamma"], refs["Vb1"]
    data_path = os.path.join(cfg.OUT_DIR, "data.npz")
    data = np.load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    rep = {"band": list(band), "rates": list(map(float, rates)),
           "smoke_mechanical_only": bool(cfg.SMOKE), "pass": False,
           "status": "running"}
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    def save():
        with open(os.path.join(cfg.OUT_DIR, "stage0.json"), "w") as f:
            json.dump(jsonable(rep), f, indent=1)

    # ---- 0a: compression map vs oracle, held out --------------------
    coef, nb, stride = fit_compression(X_train, rates)
    Zt, Dt, _ = raw_tuples(X_test[:1], rates)
    q_or = Zt @ Gamma.T - Zt[:, :2]
    err = qhat(Zt, coef) - q_or
    rep["qhat_check"] = dict(
        rms_error_over_innov_std=float(
            np.sqrt(np.mean(err ** 2) / (np.trace(Vb1) / 2.0))),
        rms_error_over_q_rms=float(np.sqrt(np.mean(err ** 2)
                                           / np.mean(q_or ** 2))),
        n_holdout=int(len(Zt)))
    # binding block: what a later training run must match exactly
    # before it may rely on this validation (codeX finding #4)
    rep["binding"] = dict(
        data_sha256=_sha256(data_path),
        coef=coef.tolist(),
        rates=list(map(float, rates)),
        sampler=dict(J=cfg.J_NEIGHBORS, nu=cfg.NU, eps=cfg.EPS_METRIC,
                     n_ode=cfg.N_ODE, n_ref=cfg.N_REF),
        smoke=bool(cfg.SMOKE))
    print("0a q_hat vs oracle:", rep["qhat_check"])
    save()

    # ---- 0b: one-step sampling at anchors ---------------------------
    tup = build_tuples(X_train, rates, "mem", coef=coef)
    kappa_s = tup["kappa_s"]
    sampler = TrainingFreeSamplerNP(tup["C"], tup["S"],
                                    cfg.J_NEIGHBORS, cfg.NU,
                                    cfg.EPS_METRIC, cfg.N_ODE)
    ma, nb_t = bank_features(X_test[:1], rates)
    nb_t = max(nb_t, 1)
    t_anchor = np.linspace(nb_t, X_test.shape[1] - 2,
                           cfg.S0_ANCHORS).astype(int)
    Winv, sqV = psd_inv_sqrt(Vb1), psd_sqrt(Vb1)
    tr_ref = float(np.trace(Vb1))
    w2n, proj, proj_cal, ess = [], [], [], []
    print("0b one-step sampling at anchors ...")
    for i, t in enumerate(t_anchor):
        zeta = np.concatenate([X_test[0, t], ma[0, t]])
        mu = Gamma @ zeta - X_test[0, t]
        q = qhat(zeta[None, :], coef)[0]
        samp, diag = sampler.sample_conditional(
            q[None, :], cfg.S0_SAMPLES, seed=cfg.SEED_STAGE0 + i)
        dv = samp / kappa_s
        w2n.append(gauss_w2sq(mu, Vb1, dv.mean(axis=0),
                              np.cov(dv.T)) / tr_ref)
        proj.append(proj_w1_stat((dv - mu) @ Winv))
        g = np.random.default_rng(
            cfg.SEED_STAGE0 + 500 + i).standard_normal(
            (cfg.S0_SAMPLES, 2))
        proj_cal.append(proj_w1_stat(g))
        ess.append(diag["ess"])
        if (i + 1) % 16 == 0:
            print(f"      anchor {i + 1}/{cfg.S0_ANCHORS}")
    onestep = dict(w2_moment_median=float(np.median(w2n)),
                   w2_moment_p90=float(np.quantile(w2n, 0.9)),
                   proj_w1_median=float(np.median(proj)),
                   proj_w1_calib_median=float(np.median(proj_cal)),
                   ess_median=float(np.median(ess)),
                   ess_p10=float(np.quantile(ess, 0.1)))
    ok_b = (onestep["w2_moment_median"] <= cfg.S0_W2_MOMENT
            and onestep["proj_w1_median"] <= cfg.S0_PROJ_FACTOR
            * onestep["proj_w1_calib_median"])
    onestep["passes"] = bool(ok_b)
    rep["onestep"] = onestep
    print("0b:", onestep)
    save()
    if not ok_b and not cfg.SMOKE:
        # stop BEFORE the expensive rollout (codeX finding #3)
        rep["status"] = "stopped_after_onestep_failure"
        save()
        sys.exit("stage 0 FAILED at the one-step check; rollout "
                 "skipped. Inspect out/stage0.json.")

    # ---- 0c-pre: gating protocol (amendment, 2026-09-08, codeX +
    # user): stage 0 gates on ONE-STEP + COVARIANCE + VACF only.
    # Diffusion accuracy is EXPLICITLY UNRESOLVED at the stage-0
    # budget (measured floor 0.173 vs the 0.075 resolvability limit,
    # out/stage0_floor.json): it is computed and REPORTED with a
    # truncation-matched reference, never gated here. The declared
    # cov/VACF tolerances are unchanged.
    m_acf = refs["m_acf"]
    Sv = refs["Sv"]
    scale = float(np.max(np.abs(Sv[0])))
    m_acf_cmp = min(m_acf, cfg.S0_N_ROLL - 1)
    m_trunc = min(max(cfg.S0_CUTOFF_LIST), cfg.S0_N_ROLL - 1)
    scan_path = os.path.join(cfg.OUT_DIR,
                             "diffusion_cutoff_scan.json")
    scan = None
    if os.path.exists(scan_path):
        with open(scan_path) as f:
            scan = json.load(f)
        if scan.get("chosen") is not None:
            m_trunc = int(scan["chosen"])   # frozen by the scan
    D_ref_trunc = diff_sampled(Sv[:m_trunc + 1])
    rep["diffusion_protocol"] = dict(
        gated=False, reported_only=True, m_trunc=m_trunc,
        scan_used=bool(scan is not None),
        deterministic_rel_err=float(
            np.linalg.norm(D_ref_trunc - refs["D_exact"])
            / np.linalg.norm(refs["D_exact"])))

    def closed_loop(V):
        C = vacf_matrix(V, max(m_trunc, m_acf_cmp))
        vacf_err = float(max(np.max(np.abs(C[m] - Sv[m]))
                             for m in range(m_acf_cmp + 1))) / scale
        D = diff_sampled(C[:m_trunc + 1])
        d_err = float(np.linalg.norm(D - D_ref_trunc)
                      / np.linalg.norm(D_ref_trunc))
        cov_dev = float(np.max(np.abs(
            np.cov(V.reshape(-1, 2).T) - cfg.KBT * np.eye(2))))
        return dict(cov_dev=cov_dev, vacf_err=vacf_err,
                    diff_err=d_err, D=D.tolist())

    tols = dict(cov_dev=cfg.S0_COV_TOL, vacf_err=cfg.S0_VACF_TOL)
    n_ens = cfg.S0_N_ENS
    if scan is not None:
        floor = dict(cov_dev=scan["cov_floor_rms"],
                     vacf_err=scan["vacf_floor_rms"])
        floor_src = "scan_8_replica_rms"
    else:
        ex0 = simulate(n_ens, cfg.S0_N_ROLL, cfg.SEED_STAGE0 + 2000,
                       refs)[:, 1:]
        f0 = closed_loop(ex0)
        floor = dict(cov_dev=f0["cov_dev"], vacf_err=f0["vacf_err"])
        floor_src = "single_exact_ensemble"
    resolvable = {k: bool(floor[k] <= cfg.S0_RESOLVE_FRAC * t)
                  for k, t in tols.items()}
    rep["rollout_floor"] = dict(source=floor_src, **floor)
    rep["resolvability"] = dict(resolve_frac=cfg.S0_RESOLVE_FRAC,
                                **resolvable)
    print("0c-pre floors:", rep["rollout_floor"])
    print("0c-pre resolvable:", resolvable)
    print("0c-pre diffusion protocol:", rep["diffusion_protocol"])
    save()
    if not all(resolvable.values()) and not cfg.SMOKE:
        rep["status"] = "inconclusive_reference_noise"
        rep["inconclusive"] = True
        save()
        sys.exit("stage 0 INCONCLUSIVE: the reference floors cannot "
                 "resolve the declared cov/VACF tolerances "
                 f"({[k for k, v in resolvable.items() if not v]}); "
                 "sampler rollout skipped, training not authorized.")

    # ---- 0c: sampler-in-the-loop rollout ----------------------------
    print("0c rollout (kNN sampler autoregressive) ...")
    banks = [bank_features(X_test[j:j + 1], rates)[0][0]
             for j in range(min(len(X_test), n_ens))]
    t0 = X_test.shape[1] // 2
    idxs = [j % len(banks) for j in range(n_ens)]
    v0 = np.stack([X_test[j, t0 + 37 * j2]
                   for j2, j in enumerate(idxs)])
    m0 = np.stack([banks[j][t0 + 37 * j2]
                   for j2, j in enumerate(idxs)])
    roll = rollout_sampler(sampler, coef, rates, kappa_s, v0, m0,
                           cfg.S0_N_ROLL, cfg.SEED_STAGE0 + 1000)
    cl = closed_loop(roll)
    # DECLARED tolerances, never loosened; resolvability was already
    # established above, so a strict comparison is meaningful here
    ok_c = all(cl[k] <= t for k, t in tols.items())
    rep["rollout"] = dict(sampler=cl, exact_same_size_floor=floor,
                          tolerances=tols, n_ens=n_ens,
                          n_steps=cfg.S0_N_ROLL, passes=bool(ok_c))
    print("0c sampler:", cl)

    if cfg.SMOKE:
        # mechanical exercise only: record everything, pass so later
        # smoke stages can run, and say so loudly in the record
        rep["pass"] = True
        rep["status"] = "smoke_mechanical_only"
    else:
        rep["pass"] = bool(ok_b and ok_c)
        rep["status"] = "complete"
    save()
    print(f"\nstage 0 {'PASS' if rep['pass'] else 'FAIL'}"
          f"{' (SMOKE: mechanical only)' if cfg.SMOKE else ''} -> "
          f"{cfg.OUT_DIR}/stage0.json")
    if not rep["pass"]:
        sys.exit("stage 0 FAILED: training stages are not authorized; "
                 "inspect out/stage0.json before any protocol change.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-only", action="store_true",
                    help="cheap exact-ensemble resolvability check "
                         "only (~1 min, no data/sampler)")
    ap.add_argument("--cutoff-scan", action="store_true",
                    help="diffusion-estimator lag-cutoff analysis, "
                         "exact ensembles only (~1 min)")
    args = ap.parse_args()
    if args.cutoff_scan:
        cutoff_scan()
    elif args.floor_only:
        floor_only()
    else:
        main()
