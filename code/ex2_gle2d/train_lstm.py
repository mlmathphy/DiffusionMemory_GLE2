"""LSTM baseline for ex2_gle2d (torch): state + increment inputs,
FULL-covariance Gaussian head.

The conditional law depends on v itself (instantaneous friction), so
the increments-only variant would be misspecified by construction;
inputs are (v_j / v_sd, dv_j * kappa_s), d_in = 4, target
dv_{j+1} * kappa_s. The exact 2D conditional covariance has an
off-diagonal entry, so the shared diagonal-covariance head would NOT
be correctly specified (codeX finding #2); the head here
parameterizes the full covariance through its Cholesky factor, which
IS correctly specified for this linear-Gaussian model.

    python train_lstm.py          # saves out/lstm/model.pt
"""

import os

import numpy as np
import torch
import torch.nn as nn

import config as cfg


class LSTMGaussianFull(nn.Module):
    """LSTM encoder + full-covariance 2D Gaussian head (Cholesky:
    L = [[e^a, 0], [l21, e^b]], cov = L L^T)."""

    def __init__(self, d_in=4, hidden=128, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 5)   # mu(2), a, b, l21

    def forward(self, u, state=None):
        h, state = self.lstm(u, state)
        p = self.head(h)
        return (p[..., :2], p[..., 2:4].clamp(-8.0, 4.0), p[..., 4],
                state)

    def nll(self, u, target):
        mu, log_d, l21, _ = self.forward(u)
        r = target - mu
        z1 = r[..., 0] / log_d[..., 0].exp()
        z2 = (r[..., 1] - l21 * z1) / log_d[..., 1].exp()
        return (0.5 * (z1 ** 2 + z2 ** 2) + log_d.sum(-1)
                + float(np.log(2 * np.pi)))

    @torch.no_grad()
    def sample_step(self, u, state, generator=None):
        mu, log_d, l21, state = self.forward(u, state)
        mu, log_d, l21 = mu[:, -1], log_d[:, -1], l21[:, -1]
        eps = torch.randn(mu.shape[0], 2, device=mu.device,
                          generator=generator)
        s1 = log_d[:, 0].exp() * eps[:, 0]
        s2 = l21 * eps[:, 0] + log_d[:, 1].exp() * eps[:, 1]
        return mu + torch.stack([s1, s2], dim=-1), state

    @torch.no_grad()
    def predict_step(self, u, state):
        mu, log_d, l21, state = self.forward(u, state)
        mu, log_d, l21 = mu[:, -1], log_d[:, -1], l21[:, -1]
        a, b = log_d[:, 0].exp(), log_d[:, 1].exp()
        C = torch.stack(
            [torch.stack([a * a, a * l21], -1),
             torch.stack([a * l21, l21 ** 2 + b * b], -1)], -2)
        return mu, C, state


def make_inputs_2d(X, x_sd, kappa_s):
    """(n, L+1, 2) -> inputs (n, L-1, 4) and targets (n, L-1, 2)."""
    X = np.asarray(X, float)
    D = np.diff(X, axis=1)
    u = np.concatenate([X[:, 1:-1] / x_sd, D[:, :-1] * kappa_s],
                       axis=-1)
    y = D[:, 1:] * kappa_s
    return u.astype(np.float32), y.astype(np.float32)


def train_lstm_2d(X_train, device, verbose=True):
    torch.manual_seed(cfg.SEED_LSTM)
    X_train = np.asarray(X_train, float)
    x_sd = float(X_train.std())
    kappa_s = float(1.0 / np.diff(X_train, axis=1).std())
    u, y = make_inputs_2d(X_train, x_sd, kappa_s)

    seq, burn = cfg.LSTM_SEQ, cfg.LSTM_BURN
    n_traj, T = u.shape[0], u.shape[1]
    n_val = max(1, n_traj // 10)
    tr, va = slice(n_val, None), slice(0, n_val)

    net = LSTMGaussianFull(4, cfg.LSTM_HIDDEN,
                           cfg.LSTM_LAYERS).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LSTM_LR)
    rng = np.random.default_rng(cfg.SEED_LSTM)

    def batch(sl, n_batch):
        i = rng.integers(0, u[sl].shape[0], n_batch)
        j = rng.integers(0, T - seq, n_batch)
        ub = np.stack([u[sl][a, b:b + seq] for a, b in zip(i, j)])
        yb = np.stack([y[sl][a, b:b + seq] for a, b in zip(i, j)])
        return (torch.tensor(ub, device=device),
                torch.tensor(yb, device=device))

    best, best_state, patience = np.inf, None, 0
    for ep in range(cfg.LSTM_EPOCHS):
        net.train()
        ub, yb = batch(tr, cfg.LSTM_BATCH)
        loss = net.nll(ub, yb)[:, burn:].mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if ep % 10 == 0 or ep == cfg.LSTM_EPOCHS - 1:
            net.eval()
            with torch.no_grad():
                uv, yv = batch(va, cfg.LSTM_BATCH)
                val = float(net.nll(uv, yv)[:, burn:].mean())
            if verbose and ep % 100 == 0:
                print(f"    epoch {ep:5d}  val NLL {val:.5f}")
            if val < best - 1e-5:
                best, patience = val, 0
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.LSTM_PATIENCE:
                    if verbose:
                        print(f"    early stop at epoch {ep} "
                              f"(best {best:.5f})")
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"x_sd": x_sd, "kappa_s": kappa_s, "val_nll": best,
                 "epochs_run": ep + 1, "seq": seq, "burn": burn}


@torch.no_grad()
def warm_state_2d(net, meta, X_pre, device, chunk=2048):
    """Recurrent state from prehistories (n, W+1, 2); returns
    (state, v, dv)."""
    X_pre = np.asarray(X_pre, float)
    u = np.concatenate([X_pre[:, 1:-1] / meta["x_sd"],
                        np.diff(X_pre, axis=1)[:, :-1]
                        * meta["kappa_s"]], axis=-1).astype(np.float32)
    state = None
    if u.shape[1] > 0:
        hs, cs = [], []
        for i0 in range(0, u.shape[0], chunk):
            ub = torch.tensor(u[i0:i0 + chunk], device=device)
            h, c = net.forward(ub, None)[3]
            hs.append(h)
            cs.append(c)
        state = (torch.cat(hs, dim=1), torch.cat(cs, dim=1))
    v = torch.tensor(X_pre[:, -1], dtype=torch.float32, device=device)
    dv = torch.tensor(X_pre[:, -1] - X_pre[:, -2], dtype=torch.float32,
                      device=device)
    return state, v, dv


@torch.no_grad()
def rollout_2d(net, meta, X_pre, n_steps, seed, device):
    """Autoregressive rollout -> velocities (n, n_steps, 2)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    state, v, dv = warm_state_2d(net, meta, X_pre, device)
    out = torch.empty(v.shape[0], n_steps, 2, device=device)
    for n in range(n_steps):
        u = torch.cat([v / meta["x_sd"], dv * meta["kappa_s"]],
                      dim=-1)[:, None, :]
        s, state = net.sample_step(u, state, generator=g)
        dv = s / meta["kappa_s"]
        v = v + dv
        out[:, n] = v
    return out.cpu().numpy()


@torch.no_grad()
def predict_step_2d(net, meta, X_pre, device):
    """Full-covariance head: mean and covariance of the next
    increment given prehistories, in physical units."""
    state, v, dv = warm_state_2d(net, meta, X_pre, device)
    u = torch.cat([v / meta["x_sd"], dv * meta["kappa_s"]],
                  dim=-1)[:, None, :]
    mean, C, _ = net.predict_step(u, state)
    return (mean.cpu().numpy() / meta["kappa_s"],
            C.cpu().numpy() / meta["kappa_s"] ** 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
    net, meta = train_lstm_2d(data["X_train"], device)
    vdir = os.path.join(cfg.OUT_DIR, "lstm")
    os.makedirs(vdir, exist_ok=True)
    torch.save({"state": net.state_dict(), "meta": meta},
               os.path.join(vdir, "model.pt"))
    print(f"saved -> {vdir}/model.pt  (val NLL {meta['val_nll']:.5f})")


if __name__ == "__main__":
    main()
