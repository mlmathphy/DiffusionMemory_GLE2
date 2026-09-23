"""Recurrent baseline: an LSTM with a Gaussian output head (torch).

The standard machine-learning answer to a non-Markovian series: encode the
observed history in a recurrent state and emit a parametric predictive
distribution for the next increment.  It is trained end to end by
maximum likelihood on the same trajectories the diffusion pipeline uses,
NOT distilled from the training-free sampler -- distilling it would make
it a student of our method rather than an alternative to it.

Contrast with the method of the paper:
  * the memory here is a trained recurrent state, not a fixed filter bank;
  * the conditional law is assumed Gaussian (or a Gaussian mixture),
    whereas the training-free sampler makes no distributional assumption;
  * training is generative (likelihood), so its optimization error cannot
    be separated from the closure error.

Input at step n is the pair (x_n, d_n) with d_n = x_n - x_{n-1}, both in
scaled units, so the baseline sees strictly more than either the
state-only or the bank conditioning.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMGaussian(nn.Module):
    """LSTM encoder + Gaussian (or mixture) head for the next increment."""

    def __init__(self, d_state=1, hidden=128, layers=1, n_mix=1,
                 d_in=None):
        super().__init__()
        self.d_state = d_state
        self.n_mix = n_mix
        # default input = (state, increment) pairs; homogeneous
        # (translation-invariant) settings pass d_in = d_state to feed
        # increments alone
        self.d_in = 2 * d_state if d_in is None else d_in
        self.lstm = nn.LSTM(self.d_in, hidden, layers, batch_first=True)
        # per mixture component: weight logit, mean, log sd
        self.head = nn.Linear(hidden, n_mix * (1 + 2 * d_state))

    def forward(self, u, state=None):
        """u: (B, T, 2 d) inputs -> (logits, mu, log_sd) and the new state."""
        h, state = self.lstm(u, state)
        p = self.head(h)
        d, m = self.d_state, self.n_mix
        logit = p[..., :m]
        mu = p[..., m:m + m * d].reshape(*p.shape[:-1], m, d)
        log_sd = p[..., m + m * d:].reshape(*p.shape[:-1], m, d)
        return logit, mu, log_sd.clamp(-8.0, 4.0), state

    def nll(self, u, target):
        """Negative log-likelihood of target (B, T, d) under the head."""
        logit, mu, log_sd = self.forward(u)[:3]
        z = (target.unsqueeze(-2) - mu) / log_sd.exp()
        logp = (-0.5 * (z ** 2) - log_sd).sum(-1) - 0.5 * np.log(2 * np.pi) * self.d_state
        return -(torch.logsumexp(logit.log_softmax(-1) + logp, dim=-1))

    @torch.no_grad()
    def sample_step(self, u, state, generator=None):
        """One step: u (B, 1, 2d) -> sampled increment (B, d), new state."""
        logit, mu, log_sd, state = self.forward(u, state)
        logit, mu, log_sd = logit[:, -1], mu[:, -1], log_sd[:, -1]
        if self.n_mix == 1:
            k = torch.zeros(mu.shape[0], dtype=torch.long, device=mu.device)
        else:
            k = torch.multinomial(logit.softmax(-1), 1,
                                  generator=generator).squeeze(-1)
        idx = torch.arange(mu.shape[0], device=mu.device)
        eps = torch.randn(mu.shape[0], self.d_state, device=mu.device,
                          generator=generator)
        return mu[idx, k] + log_sd[idx, k].exp() * eps, state

    @torch.no_grad()
    def predict_step(self, u, state):
        """Mean and standard deviation of the next increment (n_mix = 1)."""
        logit, mu, log_sd, state = self.forward(u, state)
        w = logit[:, -1].softmax(-1).unsqueeze(-1)
        mean = (w * mu[:, -1]).sum(-2)
        var = (w * (log_sd[:, -1].exp() ** 2 + mu[:, -1] ** 2)).sum(-2) - mean ** 2
        return mean, var.clamp_min(1e-12).sqrt(), state


# ------------------------------------------------------------------ data

def make_inputs(X, x_sd, kappa_s):
    """Trajectories (n, L+1) -> scaled inputs (n, L-1, 2) and targets.

    Input at step j is (x_j, d_j); the target is d_{j+1}, all scaled:
    positions by x_sd, increments by kappa_s (the pipeline's convention).
    """
    X = np.asarray(X, float)
    D = np.diff(X, axis=1)                       # d_j = X_j - X_{j-1}
    u = np.stack([X[:, 1:-1] / x_sd, D[:, :-1] * kappa_s], axis=-1)
    y = (D[:, 1:] * kappa_s)[..., None]
    return u.astype(np.float32), y.astype(np.float32)


# -------------------------------------------------------------- training

def train_lstm(X_train, cfg, device, verbose=True):
    """Fit LSTMGaussian by maximum likelihood; returns (net, meta)."""
    torch.manual_seed(getattr(cfg, "SEED_LSTM", 5))
    x_sd = float(np.asarray(X_train, float).std())
    kappa_s = float(1.0 / np.diff(np.asarray(X_train, float), axis=1).std())
    u, y = make_inputs(X_train, x_sd, kappa_s)

    seq = getattr(cfg, "LSTM_SEQ", 200)
    burn = getattr(cfg, "LSTM_BURN", 50)
    n_traj, T, _ = u.shape
    n_val = max(1, n_traj // 10)
    tr, va = slice(n_val, None), slice(0, n_val)

    net = LSTMGaussian(1, getattr(cfg, "LSTM_HIDDEN", 128),
                       getattr(cfg, "LSTM_LAYERS", 1),
                       getattr(cfg, "LSTM_MIX", 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=getattr(cfg, "LSTM_LR", 1e-3))
    rng = np.random.default_rng(getattr(cfg, "SEED_LSTM", 5))

    def batch(sl, n_batch):
        i = rng.integers(0, u[sl].shape[0], n_batch)
        j = rng.integers(0, T - seq, n_batch)
        ub = np.stack([u[sl][a, b:b + seq] for a, b in zip(i, j)])
        yb = np.stack([y[sl][a, b:b + seq] for a, b in zip(i, j)])
        return (torch.tensor(ub, device=device),
                torch.tensor(yb, device=device))

    n_epochs = getattr(cfg, "LSTM_EPOCHS", 400)
    n_batch = getattr(cfg, "LSTM_BATCH", 64)
    best, best_state, patience = np.inf, None, 0
    for ep in range(n_epochs):
        net.train()
        ub, yb = batch(tr, n_batch)
        loss = net.nll(ub, yb)[:, burn:].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if ep % 10 == 0 or ep == n_epochs - 1:
            net.eval()
            with torch.no_grad():
                uv, yv = batch(va, n_batch)
                val = float(net.nll(uv, yv)[:, burn:].mean())
            if verbose and ep % 50 == 0:
                print(f"    epoch {ep:5d}  val NLL {val:.5f}")
            if val < best - 1e-5:
                best, patience = val, 0
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= getattr(cfg, "LSTM_PATIENCE", 20):
                    if verbose:
                        print(f"    early stop at epoch {ep} "
                              f"(best val NLL {best:.5f})")
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"x_sd": x_sd, "kappa_s": kappa_s, "val_nll": best,
                 "hidden": getattr(cfg, "LSTM_HIDDEN", 128),
                 "layers": getattr(cfg, "LSTM_LAYERS", 1),
                 "n_mix": getattr(cfg, "LSTM_MIX", 1),
                 "train_seed": getattr(cfg, "SEED_LSTM", 5),
                 # cost accounting (one update per epoch in this loop):
                 # processed tokens = epochs_run * batch * seq
                 "epochs_run": ep + 1, "seq": seq, "burn": burn,
                 "batch": n_batch}


# ------------------------------------------------------------- deployment

@torch.no_grad()
def warm_state(net, meta, X_pre, device):
    """Build the recurrent state from an observed prehistory (n, W+1)."""
    u, _ = make_inputs(np.concatenate(
        [X_pre, X_pre[:, -1:]], axis=1), meta["x_sd"], meta["kappa_s"])
    u = torch.tensor(u[:, :-1], device=device)
    state = None
    if u.shape[1]:
        state = net.forward(u, None)[3]
    x = torch.tensor(X_pre[:, -1:], dtype=torch.float32, device=device)
    d = torch.tensor(X_pre[:, -1:] - X_pre[:, -2:-1], dtype=torch.float32,
                     device=device)
    return state, x, d


@torch.no_grad()
def rollout(net, meta, X_pre, n_steps, seed, device):
    """Autoregressive rollout from an observed prehistory -> (n, n_steps)."""
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    state, x, d = warm_state(net, meta, np.asarray(X_pre, float), device)
    out = torch.empty(x.shape[0], n_steps, device=device)
    for n in range(n_steps):
        u = torch.stack([x[:, 0] / meta["x_sd"], d[:, 0] * meta["kappa_s"]],
                        dim=-1)[:, None, :]
        s, state = net.sample_step(u, state, generator=g)
        d = s / meta["kappa_s"]
        x = x + d
        out[:, n] = x[:, 0]
    return out.cpu().numpy()


@torch.no_grad()
def ensemble_rollout(net, meta, x_pre, n_ens, n_steps, seed, device):
    """Ensemble from ONE prehistory: warm the recurrent state once on a
    single copy, replicate it across the ensemble, then generate.
    (Warming n_ens identical copies allocates the full (n_ens, W, hidden)
    output tensor and is an out-of-memory hazard for long prehistories.)
    """
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    state, x, d = warm_state(net, meta,
                             np.asarray(x_pre, float)[None, :], device)
    h, c = state
    state = (h.expand(-1, n_ens, -1).contiguous(),
             c.expand(-1, n_ens, -1).contiguous())
    x = x.expand(n_ens, -1).contiguous()
    d = d.expand(n_ens, -1).contiguous()
    out = torch.empty(n_ens, n_steps, device=device)
    for n in range(n_steps):
        u = torch.stack([x[:, 0] / meta["x_sd"], d[:, 0] * meta["kappa_s"]],
                        dim=-1)[:, None, :]
        s, state = net.sample_step(u, state, generator=g)
        d = s / meta["kappa_s"]
        x = x + d
        out[:, n] = x[:, 0]
    return out.cpu().numpy()


@torch.no_grad()
def ensemble_rollout_multi(net, meta, X_pre, n_ens, n_steps, seed, device):
    """Ensembles from SEVERAL prehistories in one flattened batch.

    X_pre: (n_q, W+1). The recurrent state is warmed once per
    prehistory, each warmed state is replicated n_ens times, and all
    n_q * n_ens members generate together -- one batched step per time
    step instead of one rollout per query. Returns (n_q, n_ens, n_steps).
    """
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    X_pre = np.asarray(X_pre, float)
    n_q = X_pre.shape[0]
    state, x, d = warm_state(net, meta, X_pre, device)
    h, c = state
    state = (h.repeat_interleave(n_ens, dim=1).contiguous(),
             c.repeat_interleave(n_ens, dim=1).contiguous())
    x = x.repeat_interleave(n_ens, dim=0).contiguous()
    d = d.repeat_interleave(n_ens, dim=0).contiguous()
    out = torch.empty(n_q * n_ens, n_steps, device=device)
    for n in range(n_steps):
        u = torch.stack([x[:, 0] / meta["x_sd"], d[:, 0] * meta["kappa_s"]],
                        dim=-1)[:, None, :]
        s, state = net.sample_step(u, state, generator=g)
        d = s / meta["kappa_s"]
        x = x + d
        out[:, n] = x[:, 0]
    return out.cpu().numpy().reshape(n_q, n_ens, n_steps)


@torch.no_grad()
def one_step_moments(net, meta, X_pre, device):
    """Predictive mean and sd of the next increment given a prehistory."""
    state, x, d = warm_state(net, meta, np.asarray(X_pre, float), device)
    u = torch.stack([x[:, 0] / meta["x_sd"], d[:, 0] * meta["kappa_s"]],
                    dim=-1)[:, None, :]
    mean, sd, _ = net.predict_step(u, state)
    return (mean[:, 0].cpu().numpy() / meta["kappa_s"],
            sd[:, 0].cpu().numpy() / meta["kappa_s"])


# --------------- homogeneous (increment-only) variant, any dimension ---

def make_inputs_incr(X, kappa_s):
    """Trajectories (n, L+1, d) -> scaled increment inputs and targets.

    The homogeneous (translation-invariant) counterpart of make_inputs:
    the input at step j is the scaled increment d_j alone -- absolute
    position never enters, matching the pipeline's homogeneous branch.
    Input u: (n, L-1, d) = d_1..d_{L-1}; target y: (n, L-1, d) =
    d_2..d_L.
    """
    X = np.asarray(X, float)
    D = np.diff(X, axis=1) * kappa_s             # (n, L, d)
    return D[:, :-1].astype(np.float32), D[:, 1:].astype(np.float32)


def train_lstm_incr(X_train, cfg, device, verbose=True):
    """Fit an increment-only LSTMGaussian (any state dimension) by
    maximum likelihood; returns (net, meta). Config knobs mirror
    train_lstm (LSTM_HIDDEN/LAYERS/MIX/SEQ/BURN/EPOCHS/BATCH/LR/
    PATIENCE, SEED_LSTM)."""
    torch.manual_seed(getattr(cfg, "SEED_LSTM", 5))
    X_train = np.asarray(X_train, float)
    d_state = X_train.shape[-1]
    kappa_s = float(1.0 / np.diff(X_train, axis=1).std())
    u, y = make_inputs_incr(X_train, kappa_s)

    seq = getattr(cfg, "LSTM_SEQ", 200)
    burn = getattr(cfg, "LSTM_BURN", 50)
    n_traj, T = u.shape[0], u.shape[1]
    n_val = max(1, n_traj // 10)
    tr, va = slice(n_val, None), slice(0, n_val)

    net = LSTMGaussian(d_state, getattr(cfg, "LSTM_HIDDEN", 128),
                       getattr(cfg, "LSTM_LAYERS", 1),
                       getattr(cfg, "LSTM_MIX", 1),
                       d_in=d_state).to(device)
    opt = torch.optim.Adam(net.parameters(),
                           lr=getattr(cfg, "LSTM_LR", 1e-3))
    rng = np.random.default_rng(getattr(cfg, "SEED_LSTM", 5))

    def batch(sl, n_batch):
        i = rng.integers(0, u[sl].shape[0], n_batch)
        j = rng.integers(0, T - seq, n_batch)
        ub = np.stack([u[sl][a, b:b + seq] for a, b in zip(i, j)])
        yb = np.stack([y[sl][a, b:b + seq] for a, b in zip(i, j)])
        return (torch.tensor(ub, device=device),
                torch.tensor(yb, device=device))

    n_epochs = getattr(cfg, "LSTM_EPOCHS", 400)
    n_batch = getattr(cfg, "LSTM_BATCH", 64)
    best, best_state, patience = np.inf, None, 0
    for ep in range(n_epochs):
        net.train()
        ub, yb = batch(tr, n_batch)
        loss = net.nll(ub, yb)[:, burn:].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if ep % 10 == 0 or ep == n_epochs - 1:
            net.eval()
            with torch.no_grad():
                uv, yv = batch(va, n_batch)
                val = float(net.nll(uv, yv)[:, burn:].mean())
            if verbose and ep % 50 == 0:
                print(f"    epoch {ep:5d}  val NLL {val:.5f}")
            if val < best - 1e-5:
                best, patience = val, 0
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= getattr(cfg, "LSTM_PATIENCE", 20):
                    if verbose:
                        print(f"    early stop at epoch {ep} "
                              f"(best val NLL {best:.5f})")
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"kappa_s": kappa_s, "val_nll": best,
                 "d_state": d_state, "d_in": d_state,
                 "hidden": getattr(cfg, "LSTM_HIDDEN", 128),
                 "layers": getattr(cfg, "LSTM_LAYERS", 1),
                 "n_mix": getattr(cfg, "LSTM_MIX", 1),
                 "train_seed": getattr(cfg, "SEED_LSTM", 5),
                 "epochs_run": ep + 1, "seq": seq, "burn": burn,
                 "batch": n_batch}


@torch.no_grad()
def warm_state_incr(net, meta, X_pre, device, chunk=2048):
    """Recurrent state from observed prehistories (n, W+1, d) fed as
    scaled increments; returns (state, x, d) with x the last position.

    Warming runs in chunks of `chunk` prehistories: a single call over
    the full batch materializes the (n, W, hidden) sequence output and
    its cuDNN workspace, which for tens of thousands of prehistories
    exceeds GPU memory.
    """
    X_pre = np.asarray(X_pre, float)
    D = np.diff(X_pre, axis=1) * meta["kappa_s"]
    state = None
    if D.shape[1] > 1:
        hs, cs = [], []
        for i0 in range(0, D.shape[0], chunk):
            u = torch.tensor(D[i0:i0 + chunk, :-1], dtype=torch.float32,
                             device=device)
            h, c = net.forward(u, None)[3]
            hs.append(h); cs.append(c)
        state = (torch.cat(hs, dim=1), torch.cat(cs, dim=1))
    x = torch.tensor(X_pre[:, -1], dtype=torch.float32, device=device)
    d = torch.tensor(X_pre[:, -1] - X_pre[:, -2], dtype=torch.float32,
                     device=device)
    return state, x, d


@torch.no_grad()
def rollout_incr(net, meta, X_pre, n_steps, seed, device):
    """Autoregressive rollout from observed prehistories (n, W+1, d)
    -> positions (n, n_steps, d)."""
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    state, x, d = warm_state_incr(net, meta, X_pre, device)
    out = torch.empty(x.shape[0], n_steps, x.shape[1], device=device)
    for n in range(n_steps):
        u = (d * meta["kappa_s"])[:, None, :]
        s, state = net.sample_step(u, state, generator=g)
        d = s / meta["kappa_s"]
        x = x + d
        out[:, n] = x
    return out.cpu().numpy()


@torch.no_grad()
def sample_step_incr(net, meta, X_pre, seed, device):
    """ONE sampled next increment given observed prehistories
    (n, W+1, d) -> (n, d)."""
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    state, _, d = warm_state_incr(net, meta, X_pre, device)
    u = (d * meta["kappa_s"])[:, None, :]
    s, _ = net.sample_step(u, state, generator=g)
    return (s / meta["kappa_s"]).cpu().numpy()


def train_lstm_incr_two_phase(X_train, cfg, device, verbose=True):
    """Fair, deterministic two-phase protocol (codeX, telegraph
    example): validation must not be a freshly sampled batch and the
    binding model must see the full training budget.

    Phase 1: fixed trajectory split (first n//10 trajectories held
    out), a FIXED predeclared set of validation windows (seed
    cfg.SEED_LSTM_VAL, cfg.LSTM_VAL_WINDOWS windows) evaluated
    identically at every check; early stopping selects only the
    NUMBER OF UPDATES.  Validation never consumes the training RNG.
    Phase 2: reinitialize (same torch seed) and retrain for exactly
    that many updates on ALL trajectories; the refit checkpoint is
    the binding benchmark.
    """
    X_train = np.asarray(X_train, float)
    d_state = X_train.shape[-1]
    kappa_s = float(1.0 / np.diff(X_train, axis=1).std())
    u, y = make_inputs_incr(X_train, kappa_s)
    seq = getattr(cfg, "LSTM_SEQ", 200)
    burn = getattr(cfg, "LSTM_BURN", 50)
    n_traj, T = u.shape[0], u.shape[1]
    n_val = max(1, n_traj // 10)
    n_batch = getattr(cfg, "LSTM_BATCH", 64)
    n_epochs = getattr(cfg, "LSTM_EPOCHS", 400)

    rngv = np.random.default_rng(getattr(cfg, "SEED_LSTM_VAL", 1234))
    n_win = getattr(cfg, "LSTM_VAL_WINDOWS", 256)
    iv = rngv.integers(0, n_val, n_win)
    jv = rngv.integers(0, T - seq, n_win)
    uv = torch.tensor(np.stack([u[a, b:b + seq]
                                for a, b in zip(iv, jv)]), device=device)
    yv = torch.tensor(np.stack([y[a, b:b + seq]
                                for a, b in zip(iv, jv)]), device=device)

    def _run(sl, max_updates, track_val):
        torch.manual_seed(getattr(cfg, "SEED_LSTM", 5))
        net = LSTMGaussian(d_state, getattr(cfg, "LSTM_HIDDEN", 128),
                           getattr(cfg, "LSTM_LAYERS", 1),
                           getattr(cfg, "LSTM_MIX", 1),
                           d_in=d_state).to(device)
        opt = torch.optim.Adam(net.parameters(),
                               lr=getattr(cfg, "LSTM_LR", 1e-3))
        rng = np.random.default_rng(getattr(cfg, "SEED_LSTM", 5))
        us, ys = u[sl], y[sl]
        best, best_updates, patience = np.inf, 0, 0
        ep = -1
        for ep in range(max_updates):
            net.train()
            i = rng.integers(0, us.shape[0], n_batch)
            j = rng.integers(0, T - seq, n_batch)
            ub = torch.tensor(np.stack([us[a, b:b + seq]
                                        for a, b in zip(i, j)]),
                              device=device)
            yb = torch.tensor(np.stack([ys[a, b:b + seq]
                                        for a, b in zip(i, j)]),
                              device=device)
            loss = net.nll(ub, yb)[:, burn:].mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if track_val and (ep % 10 == 0 or ep == max_updates - 1):
                net.eval()
                with torch.no_grad():
                    val = float(net.nll(uv, yv)[:, burn:].mean())
                if verbose and ep % 200 == 0:
                    print(f"    update {ep:5d}  fixed val NLL {val:.5f}")
                if val < best - 1e-5:
                    best, best_updates, patience = val, ep + 1, 0
                else:
                    patience += 1
                    if patience >= getattr(cfg, "LSTM_PATIENCE", 20):
                        if verbose:
                            print(f"    early stop at update {ep} "
                                  f"(best {best:.5f} at "
                                  f"{best_updates})")
                        break
        net.eval()
        return net, best, best_updates, ep + 1

    _n1, best_val, best_updates, ran = _run(slice(n_val, None),
                                            n_epochs, True)
    if verbose:
        print(f"  phase 1: {ran} updates ran, selected "
              f"{best_updates} (fixed val NLL {best_val:.5f}); "
              f"phase 2: refit on all {n_traj} trajectories")
    net, _b, _u2, _r2 = _run(slice(None), max(best_updates, 1), False)
    return net, {"kappa_s": kappa_s, "val_nll": best_val,
                 "d_state": d_state, "d_in": d_state,
                 "hidden": getattr(cfg, "LSTM_HIDDEN", 128),
                 "layers": getattr(cfg, "LSTM_LAYERS", 1),
                 "n_mix": getattr(cfg, "LSTM_MIX", 1),
                 "train_seed": getattr(cfg, "SEED_LSTM", 5),
                 "val_seed": getattr(cfg, "SEED_LSTM_VAL", 1234),
                 "epochs_run": best_updates, "phase1_updates": ran,
                 "protocol": "two_phase_fixed_val",
                 "seq": seq, "burn": burn, "batch": n_batch}
