"""Closed-loop + one-step evaluation for ex2_gle2d (torch inference).

Closed loop (per model: mem, markov, lstm; exact same-size ensemble as
the estimation-noise floor): stationary covariance, the VACF MATRIX
(all entries), the measured principal-axis rotation curve (masked
where unreliable), and the sampled-displacement diffusion tensor —
all against the exact references of exact_refs.py.

One step (N_QUERY anchors from held-out data): mem and markov sampled
from their distilled nets (moment W2 + projected W1, the stage-0
criteria). DISCLOSURE (codeX): the mem reference is the exact
conditional given the FULL bank, so the measured mem error INCLUDES
the estimated-compression error — a stricter test than "its own
information set", stated as such in the record. The LSTM's
full-covariance Gaussian head is compared to the exact full-history
(Kalman) conditional.

Outputs: out/summary.json, figs/ex2_vacf.pdf. Rollout semantics
throughout: the FULL memory bank is updated every step; compression
is applied at query time only.
"""

import json
import hashlib
from pathlib import Path
import os

import numpy as np
import torch

import config as cfg
import exact_refs
from conditioning import bank_features, qhat
from generate_data import simulate
from memdiff.distill import FlowMapNet
from scout import gauss_w2sq, proj_w1_stat, psd_inv_sqrt, jsonable
from stage0_validate import diff_sampled, vacf_matrix
from train_lstm import LSTMGaussianFull, predict_step_2d, rollout_2d

WARM = cfg.EVAL_WARM     # prehistory steps for rollout/anchor warm-ups


def load_net(name, device):
    ck = torch.load(os.path.join(cfg.OUT_DIR, name, "model.pt"),
                    map_location=device, weights_only=False)
    net = FlowMapNet(2, 2, cfg.HIDDEN, cfg.N_LAYERS).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


@torch.no_grad()
def net_sample(net, ck, C, z, device):
    """Distilled flow map: conditions C (n,2) + latents z -> dv."""
    X = np.hstack([C, z]).astype(np.float32)
    Xn = torch.tensor((X - ck["norm"]["x_mu"]) / ck["norm"]["x_sd"],
                      device=device)
    s = net(Xn).cpu().numpy() * ck["norm"]["y_sd"] + ck["norm"]["y_mu"]
    return s / ck["kappa_s"]


def net_rollout(net, ck, variant, coef, rates, v0, m0, n_steps, seed,
                device):
    rng = np.random.default_rng(seed)
    rho = np.repeat(np.exp(-np.asarray(rates) * cfg.DT), 2) \
        if len(rates) else None
    v, m = v0.copy(), m0.copy()
    out = np.empty((len(v0), n_steps, 2))
    for t in range(n_steps):
        if variant == "mem":
            C = qhat(np.concatenate([v, m], axis=1), coef)
        else:
            C = v
        dv = net_sample(net, ck, C,
                        rng.standard_normal((len(v0), 2)), device)
        v = v + dv
        if rho is not None:
            m = rho[None, :] * m + np.tile(dv, len(rates))
        out[:, t] = v
    return out


def kalman_filter_means(refs, X):
    """Exact-observation filter along one trajectory (L+1, 2):
    E[v_{n+1} | v_{0..n}] for every n (converges to the steady state
    long before the anchors are used)."""
    Ad, Qd = refs["Ad"], refs["Qd"]
    P = np.diag([0.0, 0.0, refs["Sigma"][2, 2], refs["Sigma"][3, 3]])
    mfilt = np.zeros(4)
    mfilt[:2] = X[0]
    means = np.zeros((len(X), 2))
    for n in range(len(X) - 1):
        mp = Ad @ mfilt
        Pp = Ad @ P @ Ad.T + Qd
        K = np.linalg.solve(Pp[:2, :2], Pp[:2, :]).T
        mfilt = mp + K @ (X[n + 1] - mp[:2])
        P = Pp - K @ Pp[:2, :]
        means[n + 1] = (Ad @ mfilt)[:2]
    return means           # means[n] = E[v_{n+1} | v_{<=n}]


def cached_rollout(name, device, compute):
    """Reuse only rollouts from identical inputs, code, and settings."""
    digest = hashlib.sha256()
    paths = [Path(cfg.OUT_DIR) / "data.npz",
             Path(cfg.OUT_DIR) / "compression.npz"]
    if name != "exact_floor":
        paths.append(Path(cfg.OUT_DIR) / name / "model.pt")
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += sorted((Path(__file__).parent.parent / "memdiff").glob("*.py"))
    for path in paths:
        digest.update(str(path.name).encode())
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    digest.update(str((device, torch.__version__, np.__version__, cfg.SMOKE,
                       cfg.N_GEN_TRAJ, cfg.L_GEN, cfg.SEED_EVAL)).encode())
    path = Path(cfg.OUT_DIR) / "cache" / f"rollout_{name}.npz"
    key = digest.hexdigest()
    if path.exists():
        with np.load(path) as saved:
            if str(saved["key"]) == key:
                print(f"reuse cached rollout {name}")
                return saved["V"]
    V = compute()
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, key=key, V=V)
    os.replace(temp, path)
    return V


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    refs = exact_refs.references()
    rates = refs["rates"]
    Sv, m_acf = refs["Sv"], refs["m_acf"]
    scale = float(np.max(np.abs(Sv[0])))
    data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
    X_test = data["X_test"]
    comp = np.load(os.path.join(cfg.OUT_DIR, "compression.npz"))
    coef = comp["coef"]

    # evaluate whichever models exist (amendment: memory model first,
    # baselines deferred until it works)
    nets = {v: load_net(v, device) for v in ("mem", "markov")
            if os.path.exists(os.path.join(cfg.OUT_DIR, v,
                                           "model.pt"))}
    if "mem" not in nets:
        raise SystemExit("no mem model found — run the train stage "
                         "first")
    have_lstm = os.path.exists(os.path.join(cfg.OUT_DIR, "lstm",
                                            "model.pt"))
    if have_lstm:
        lck = torch.load(os.path.join(cfg.OUT_DIR, "lstm",
                                      "model.pt"),
                         map_location=device, weights_only=False)
        lnet = LSTMGaussianFull(4, cfg.LSTM_HIDDEN,
                                cfg.LSTM_LAYERS).to(device)
        lnet.load_state_dict(lck["state"])
        lnet.eval()
    print(f"models: {sorted(nets)}"
          + (" + lstm" if have_lstm else "  (baselines deferred)"))

    # ------------------------------------------------ closed loop
    banks = [bank_features(X_test[j:j + 1], rates)[0][0]
             for j in range(len(X_test))]
    n_ens, L = cfg.N_GEN_TRAJ, X_test.shape[1]
    picks = [(j % len(X_test), L // 3 + 41 * j) for j in range(n_ens)]
    v0 = np.stack([X_test[a, b] for a, b in picks])
    m0 = np.stack([banks[a][b] for a, b in picks])
    X_pre = np.stack([X_test[a, b - WARM:b + 1] for a, b in picks])

    rolls = {}
    for v in nets:
        net, ck = nets[v]
        r = np.array([]) if v == "markov" else rates
        m00 = np.zeros((n_ens, 0)) if v == "markov" else m0
        print(f"rollout {v} ...")
        rolls[v] = cached_rollout(v, device, lambda: net_rollout(
            net, ck, v, coef, r, v0, m00, cfg.L_GEN, cfg.SEED_EVAL, device))
    if have_lstm:
        print("rollout lstm ...")
        rolls["lstm"] = cached_rollout("lstm", device, lambda: rollout_2d(
            lnet, lck["meta"], X_pre, cfg.L_GEN, cfg.SEED_EVAL, device))
    rolls["exact_floor"] = cached_rollout("exact_floor", device, lambda: simulate(
        n_ens, cfg.L_GEN, cfg.SEED_EVAL + 999, refs)[:, 1:])

    # truncation matched to the lags the rollouts can estimate
    max_lag = min(max(cfg.EVAL_MAX_LAG,
                      int(round(10 * cfg.TAU2 / cfg.DT))),
                  cfg.L_GEN - 1)
    m_acf_cmp = min(m_acf, max_lag)
    D_ref_trunc = diff_sampled(Sv[:max_lag + 1])
    closed, curves = {}, {}
    for name, V in rolls.items():
        C = vacf_matrix(V, max_lag)
        closed[name] = dict(
            cov_dev=float(np.max(np.abs(
                np.cov(V.reshape(-1, 2).T) - cfg.KBT * np.eye(2)))),
            vacf_err=float(max(np.max(np.abs(C[m] - Sv[m]))
                               for m in range(m_acf_cmp + 1))) / scale,
            diff_err=float(np.linalg.norm(diff_sampled(C)
                                          - D_ref_trunc)
                           / np.linalg.norm(D_ref_trunc)),
            D=diff_sampled(C).tolist())
        curves[name] = C
        print(name, {k: v for k, v in closed[name].items()
                     if k != "D"})

    # ------------------------------------------------ one step
    print("one-step anchors ...")
    ma = banks[0]
    nb = max(bank_features(X_test[:1], rates)[1], WARM + 1)
    t_anchor = np.linspace(nb, L - 2, cfg.N_QUERY).astype(int)
    Gamma, Vb1 = refs["Gamma"], refs["Vb1"]
    Phi, V_mk = exact_refs.markov_coeffs(Sv)
    Winv_b, Winv_m = psd_inv_sqrt(Vb1), psd_inv_sqrt(V_mk)
    kal_means = kalman_filter_means(refs, X_test[0])
    V_k1 = refs["V_kal1"]
    rng = np.random.default_rng(cfg.SEED_EVAL + 1)
    ones = {v: dict(w2=[], proj=[]) for v in nets}
    lstm_mean_err2, cal = [], []
    common = {v: [] for v in nets}
    cloud = {"trajectory": 0, "time_index": int(t_anchor[0]),
             "dt": cfg.DT, "samples": {}}
    stationary = {}
    edges = np.linspace(-4 * np.sqrt(cfg.KBT), 4 * np.sqrt(cfg.KBT), 81)
    for name, V in rolls.items():
        flat = V.reshape(-1, 2)
        stationary[name] = dict(
            density=[(np.histogram(flat[:, j], edges)[0] /
                      (len(flat) * np.diff(edges))).tolist() for j in range(2)],
            outside=[float(np.mean(np.abs(flat[:, j]) > edges[-1])) for j in range(2)])
    for i, t in enumerate(t_anchor):
        vhere = X_test[0, t]
        zeta = np.concatenate([vhere, ma[t]])
        exact = {"mem": (Gamma @ zeta - vhere, Vb1, Winv_b),
                 "markov": (Phi @ vhere - vhere, V_mk, Winv_m)}
        for v in nets:
            net, ck = nets[v]
            C = (qhat(zeta[None, :], coef) if v == "mem"
                 else vhere[None, :])
            dv = net_sample(net, ck, np.repeat(C, cfg.NS_COND, axis=0),
                            rng.standard_normal((cfg.NS_COND, 2)),
                            device)
            mu, Vc, Wi = exact[v]
            ones[v]["w2"].append(
                gauss_w2sq(mu, Vc, dv.mean(axis=0), np.cov(dv.T))
                / float(np.trace(Vc)))
            ones[v]["proj"].append(proj_w1_stat((dv - mu) @ Wi))
            mu_full = kal_means[t] - vhere
            common[v].append(gauss_w2sq(mu_full, V_k1, dv.mean(0),
                                       np.cov(dv.T)) / np.trace(V_k1))
            if i == 0:
                cloud["samples"][v] = dv.tolist()
        cal.append(proj_w1_stat(
            np.random.default_rng(cfg.SEED_EVAL + 500 + i)
            .standard_normal((cfg.NS_COND, 2))))
    onestep = {
        v: dict(w2_moment_median=float(np.median(ones[v]["w2"])),
                proj_w1_median=float(np.median(ones[v]["proj"])),
                proj_w1_calib_median=float(np.median(cal)))
        for v in nets}
    onestep["mem_reference_note"] = (
        "mem is compared to the exact FULL-BANK conditional; the "
        "measured error includes estimated-compression error")
    if have_lstm:
        # LSTM one-step vs its information set (full history /
        # Kalman); full-covariance head -> matrix comparison
        X_pre_a = np.stack([X_test[0, t - WARM:t + 1]
                            for t in t_anchor])
        lmean, lcov = predict_step_2d(lnet, lck["meta"], X_pre_a,
                                      device)
        mu_kal = kal_means[t_anchor] - X_test[0, t_anchor]
        lstm_mean_err2 = np.sum((lmean - mu_kal) ** 2, axis=1)
        lstm_cov_rel = [float(np.linalg.norm(lcov[i] - V_k1)
                              / np.linalg.norm(V_k1))
                        for i in range(len(t_anchor))]
        common["lstm"] = [gauss_w2sq(mu_kal[i], V_k1, lmean[i], lcov[i])
                          / np.trace(V_k1) for i in range(len(t_anchor))]
        cloud["samples"]["lstm"] = (lmean[0] + np.random.default_rng(
            cfg.SEED_EVAL + 700).standard_normal((cfg.NS_COND, 2))
            @ np.linalg.cholesky(lcov[0]).T).tolist()
        onestep["lstm_vs_kalman"] = dict(
            mean_rmse_over_innov=float(
                np.sqrt(np.mean(lstm_mean_err2) / np.trace(V_k1))),
            cov_frob_rel_median=float(np.median(lstm_cov_rel)))
    print("one-step:", onestep)

    # Save plot inputs; figure-only runs never regenerate trajectories.
    rot_exact = exact_refs.axis_rotation(Sv, cfg.EVAL_MAX_LAG)
    mu0 = kal_means[t_anchor[0]] - X_test[0, t_anchor[0]]
    cloud["reference_mean"] = mu0.tolist()
    cloud["reference_cov"] = V_k1.tolist()
    cloud["samples"]["exact"] = (mu0 + np.random.default_rng(
        cfg.SEED_EVAL + 701).standard_normal((cfg.NS_COND, 2))
        @ np.linalg.cholesky(V_k1).T).tolist()
    summary = dict(closed_loop=closed, onestep=onestep,
                   dt=cfg.DT, kbt=cfg.KBT,
                   common_reference_note="All models vs full-history Kalman; network scores use empirical moments, LSTM uses analytic head moments. Single trained checkpoint per model.",
                   common_reference={k: dict(median=float(np.median(v)),
                       per_anchor=list(map(float, v))) for k, v in common.items()},
                   conditional_cloud=cloud,
                   stationary=dict(edges=edges.tolist(), models=stationary),
                   vacf_exact=Sv[:cfg.EVAL_MAX_LAG + 1].tolist(),
                   axis_rotation_exact=rot_exact,
                   D_exact=refs["D_exact"].tolist(),
                   D_ref_trunc=D_ref_trunc.tolist(),
                   vacf_max_lag=max_lag,
                   n_ens=n_ens, L_gen=cfg.L_GEN,
                   vacf_curves={k: v[:cfg.EVAL_MAX_LAG + 1].tolist()
                                for k, v in curves.items()})
    with open(os.path.join(cfg.OUT_DIR, "summary.json"), "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    print(f"summary -> {cfg.OUT_DIR}/summary.json")
    print("common Kalman-reference moment W2:",
          {k: v["median"] for k, v in summary["common_reference"].items()})
    from make_figs import make_figures
    make_figures(summary)


if __name__ == "__main__":
    main()
