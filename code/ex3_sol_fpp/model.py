"""Multi-timescale filtered-Poisson process: exact simulation, closed-form
statistics, exact one-step law given the hidden state, and the
ACF-based linear closed-loop references (ex4 machinery, ACF as input).

Per population j (decay a_j = TAU_D[j], rise b = TAU_R, rate nu_j,
exponential amplitudes with mean amp_j) the pulse train is the linear
filter of shot noise with two hidden states
    X_dj(t) = sum_k A_k exp(-(t - t_k)/a_j),
    X_rj(t) = sum_k A_k exp(-(t - t_k)/b),
    Phi_j   = c_j (X_dj - X_rj),   c_j = a_j / (a_j - b),
so the exact step over dt is: decay both states, add the exact
contributions of the Poisson arrivals inside the step (uniform arrival
times, exponential amplitudes).  Observed: y = sum_j Phi_j + sigma_n xi.
"""

import numpy as np
from scipy import integrate

import config as cfg


# =================================================================== params
def params():
    a = np.asarray(cfg.TAU_D, float)
    g = np.asarray(cfg.GAMMA, float)
    b = float(cfg.TAU_R)
    nu = g / a
    c = a / (a - b)
    # Campbell: Var_j = nu_j <A^2> c_j^2 S_j, <A^2> = 2 amp_j^2,
    # S_j = int_0^inf (e^{-s/a} - e^{-s/b})^2 ds = a/2 - 2ab/(a+b) + b/2
    S = a / 2.0 - 2.0 * a * b / (a + b) + b / 2.0
    share = np.asarray(cfg.VAR_SHARE, float)
    amp = np.empty(len(a))
    amp[0] = cfg.A1
    base = nu[0] * 2 * amp[0] ** 2 * c[0] ** 2 * S[0] / share[0]
    for j in range(1, len(a)):
        amp[j] = np.sqrt(base * share[j] / (nu[j] * 2 * c[j] ** 2 * S[j]))
    var_sig = float(np.sum(nu * 2 * amp ** 2 * c ** 2 * S))
    sig_n = float(np.sqrt(cfg.EPS_NOISE / (1.0 - cfg.EPS_NOISE) * var_sig))
    mean = float(np.sum(nu * amp * a))        # int psi_j = a_j
    return dict(a=a, b=b, nu=nu, c=c, amp=amp, S=S, var_sig=var_sig,
                sig_n=sig_n, mean=mean, var=var_sig + sig_n ** 2)


def atom_prob(dt=None):
    """Probability of no arrival (any population) within one step."""
    P = params()
    dt = cfg.DT if dt is None else dt
    return float(np.exp(-P["nu"].sum() * dt))


# ================================================= closed-form statistics
def pulse(j, theta, P=None):
    P = params() if P is None else P
    theta = np.asarray(theta, float)
    return np.where(theta >= 0, P["c"][j] * (np.exp(-theta / P["a"][j])
                                             - np.exp(-theta / P["b"])), 0.0)


def cumulants(P=None):
    """Stationary cumulants kappa_2..kappa_4 of y (Campbell: kappa_n =
    nu <A^n> int psi^n, <A^n> = n! amp^n; the noise adds to kappa_2)."""
    P = params() if P is None else P
    k2 = P["sig_n"] ** 2
    k3 = k4 = 0.0
    for j in range(len(P["a"])):
        up = 60.0 * P["a"][j]
        I2 = P["nu"][j] * 2 * P["amp"][j] ** 2 * P["c"][j] ** 2 * P["S"][j]
        I3 = integrate.quad(lambda s: pulse(j, s, P) ** 3, 0, up, limit=400)[0]
        I4 = integrate.quad(lambda s: pulse(j, s, P) ** 4, 0, up, limit=400)[0]
        k2 += I2
        k3 += P["nu"][j] * 6 * P["amp"][j] ** 3 * I3
        k4 += P["nu"][j] * 24 * P["amp"][j] ** 4 * I4
    skew = k3 / k2 ** 1.5
    flat = k4 / k2 ** 2 + 3.0
    return dict(k2=k2, k3=k3, k4=k4, skew=float(skew), flat=float(flat))


def acf_cont(tau, P=None):
    """Autocovariance C(tau) of the signal part (tau >= 0), closed form:
    C_j(tau) = nu_j <A^2> c_j^2 [e^{-tau/a}(a/2 - ab/(a+b))
                                 + e^{-tau/b}(b/2 - ab/(a+b))]."""
    P = params() if P is None else P
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    b = P["b"]
    for j in range(len(P["a"])):
        a = P["a"][j]
        pref = P["nu"][j] * 2 * P["amp"][j] ** 2 * P["c"][j] ** 2
        out = out + pref * (np.exp(-tau / a) * (a / 2 - a * b / (a + b))
                            + np.exp(-tau / b) * (b / 2 - a * b / (a + b)))
    return out


def acf_sampled(n_lags, dt=None, P=None):
    """C(m) = Cov(y_n, y_{n+m}), m = 0..n_lags, incl. the noise at lag 0."""
    P = params() if P is None else P
    dt = cfg.DT if dt is None else dt
    C = acf_cont(np.arange(n_lags + 1) * dt, P)
    C[0] += P["sig_n"] ** 2
    return C


# ==================================================== exact simulation
def signal(state, P=None):
    """Noise-free observable sum_j c_j (X_dj - X_rj) for states (n, 2 npop)."""
    P = params() if P is None else P
    state = np.atleast_2d(state)
    return sum(P["c"][j] * (state[:, 2 * j] - state[:, 2 * j + 1])
               for j in range(len(P["a"])))


def step(state, rng, P, dt=None, n_arr_out=None):
    """One exact step for n trajectories. state (n, 2*npop): [X_d, X_r]
    per population. Returns (new_state, y_new)."""
    dt = cfg.DT if dt is None else dt
    n = state.shape[0]
    new = np.empty_like(state)
    y = rng.normal(0.0, P["sig_n"], n)
    b = P["b"]
    for j in range(len(P["a"])):
        a = P["a"][j]
        Xd = state[:, 2 * j] * np.exp(-dt / a)
        Xr = state[:, 2 * j + 1] * np.exp(-dt / b)
        N = rng.poisson(P["nu"][j] * dt, n)
        if n_arr_out is not None:
            n_arr_out[:, j] = N
        for i in range(int(N.max()) if n else 0):
            m = N > i
            nm = int(m.sum())
            s = rng.uniform(0.0, dt, nm)          # arrival times in step
            A = rng.exponential(P["amp"][j], nm)
            Xd[m] += A * np.exp(-(dt - s) / a)
            Xr[m] += A * np.exp(-(dt - s) / b)
        new[:, 2 * j], new[:, 2 * j + 1] = Xd, Xr
        y += P["c"][j] * (Xd - Xr)
    return new, y


def simulate(n_traj, n_steps, seed, burn_t=None, P=None):
    """Stationary records: Y (n, n_steps+1), hidden states STATE
    (n, n_steps+1, 2 npop), arrivals NARR (n, n_steps, npop) per step."""
    P = params() if P is None else P
    rng = np.random.default_rng(seed)
    burn = int(round((cfg.BURN_T if burn_t is None else burn_t) / cfg.DT))
    npop = len(P["a"])
    st = np.zeros((n_traj, 2 * npop))
    for _ in range(burn):
        st, _y = step(st, rng, P)
    Y = np.empty((n_traj, n_steps + 1))
    ST = np.empty((n_traj, n_steps + 1, 2 * npop))
    NARR = np.empty((n_traj, n_steps, npop), dtype=np.int32)
    # y at the first record time (noise fresh)
    Y[:, 0] = rng.normal(0.0, P["sig_n"], n_traj) + sum(
        P["c"][j] * (st[:, 2 * j] - st[:, 2 * j + 1]) for j in range(npop))
    ST[:, 0] = st
    for t in range(1, n_steps + 1):
        st, y = step(st, rng, P, n_arr_out=NARR[:, t - 1])
        Y[:, t], ST[:, t] = y, st
    return Y, ST, NARR


# ===================================== exact one-step law given hidden state
def oracle_sample(state_row, n_s, rng, P=None):
    """n_s exact draws of y_{n+1} given the hidden state at time n."""
    P = params() if P is None else P
    st = np.repeat(np.asarray(state_row, float)[None, :], n_s, 0)
    _, y = step(st, rng, P)
    return y


def oracle_moments(state, P=None, dt=None):
    """Exact E[y_{n+1} | state], Var[y_{n+1} | state] (Campbell on the step)."""
    P = params() if P is None else P
    dt = cfg.DT if dt is None else dt
    state = np.atleast_2d(state)
    b = P["b"]
    mean = np.zeros(len(state))
    var = np.full(len(state), P["sig_n"] ** 2)
    for j in range(len(P["a"])):
        a, c = P["a"][j], P["c"][j]
        mean += c * (state[:, 2 * j] * np.exp(-dt / a)
                     - state[:, 2 * j + 1] * np.exp(-dt / b))
        # fresh arrivals: int_0^dt psi(u) du and int_0^dt psi(u)^2 du
        I1 = c * (a * (1 - np.exp(-dt / a)) - b * (1 - np.exp(-dt / b)))
        I2 = c ** 2 * (a / 2 * (1 - np.exp(-2 * dt / a))
                       - 2 * a * b / (a + b) * (1 - np.exp(-dt * (a + b) / (a * b)))
                       + b / 2 * (1 - np.exp(-2 * dt / b)))
        mean += P["nu"][j] * P["amp"][j] * I1
        var += P["nu"][j] * 2 * P["amp"][j] ** 2 * I2
    return mean, var


# =========================== linear closed-loop references (ACF-based)
def tail_length(rates, dt, tol):
    rho_max = np.exp(-min(rates) * dt)
    return int(np.ceil(np.log(tol) / np.log(rho_max)))


def incr_acov(C):
    L = len(C) - 2
    h = np.arange(L + 1)
    return 2.0 * C[h] - C[np.abs(h - 1)] - C[h + 1]


def memory_joint(C, rates, dt, tol):
    """Cov of v = (y_n, m_n) for the stationary EMA bank of increments
    (ex4 `_memory_joint`, with the ACF supplied). C must cover L+3 lags."""
    rates = np.asarray(rates, float)
    k = len(rates)
    L = tail_length(rates, dt, tol) if k else 4
    if len(C) < L + 3:
        raise ValueError("ACF too short for the geometric tail")
    RD = incr_acov(C)
    S = np.empty((1 + k, 1 + k))
    S[0, 0] = C[0]
    rho = W = None
    if k:
        rho = np.exp(-rates * dt)
        l = np.arange(L + 1)
        W = rho[:, None] ** l[None, :]
        S[0, 1:] = S[1:, 0] = W @ (C[:L + 1] - C[1:L + 2])
        denom = 1.0 - rho[:, None] * rho[None, :]
        h = np.arange(1, L + 1)
        pos = rho[:, None, None] ** h[None, None, :]
        neg = rho[None, :, None] ** h[None, None, :]
        cross = np.tensordot(pos * np.ones((1, k, 1)), RD[1:L + 1],
                             axes=([2], [0])) \
            + np.tensordot(neg * np.ones((k, 1, 1)), RD[1:L + 1],
                           axes=([2], [0]))
        S[1:, 1:] = (RD[0] + cross) / denom
    if not np.isfinite(S).all():
        raise FloatingPointError("non-finite memory covariance")
    return dict(C=C, RD=RD, rho=rho, W=W, L=L, S=S, k=k)


def cond_given_memory(C, rates, dt, tol):
    """Best LINEAR one-step predictor of D_{n+1} given (y_n, m_n):
    (beta, V)."""
    J = memory_joint(C, rates, dt, tol)
    RD, W, L, k = J["RD"], J["W"], J["L"], J["k"]
    c = np.empty(1 + k)
    c[0] = C[1] - C[0]
    if k:
        c[1:] = W @ RD[1:L + 2]
    beta = np.linalg.solve(J["S"], c)
    return beta, float(RD[0] - c @ beta)


def cond_given_window(C, M):
    """Linear one-step predictive variance given the last M states."""
    idx = np.arange(M)
    G = C[np.abs(idx[:, None] - idx[None, :])]
    cvec = C[1:M + 1]
    coef = np.linalg.solve(G, cvec)
    return float(C[0] - cvec @ coef)


def ar_ideal_acf(C, M, n_out):
    """Stationary ACF of the population AR(M): the ideal raw-window model."""
    if M >= n_out:
        return C[:n_out + 1].copy()
    idx = np.arange(M)
    G = C[np.abs(idx[:, None] - idx[None, :])]
    a = np.linalg.solve(G, C[1:M + 1])
    Ce = np.empty(n_out + 1)
    Ce[:M + 1] = C[:M + 1]
    for m in range(M + 1, n_out + 1):
        Ce[m] = a @ Ce[m - 1:m - M - 1:-1]
    return Ce


def bank_ideal_acf(C, rates, dt, tol, n_out):
    """Stationary ACF of the ideal linear bank chain y' = y + beta.(y, m) + e."""
    rates = np.asarray(rates, float)
    k = len(rates)
    rho = np.exp(-rates * dt)
    beta, V = cond_given_memory(C, rates, dt, tol)
    T = np.zeros((1 + k, 1 + k))
    T[0] = beta
    T[0, 0] += 1.0
    for i in range(k):
        T[1 + i] = beta
        T[1 + i, 1 + i] += rho[i]
    Qc = V * np.ones((1 + k, 1 + k))
    P = np.zeros_like(Qc)
    for _ in range(500000):
        Pn = T @ P @ T.T + Qc
        if np.max(np.abs(Pn - P)) < 1e-14 * max(1.0, np.max(np.abs(Pn))):
            P = Pn
            break
        P = Pn
    Ce = np.empty(n_out + 1)
    Ce[0] = P[0, 0]
    v = P[:, 0].copy()
    for m in range(1, n_out + 1):
        v = T @ v
        Ce[m] = v[0]
    return Ce


def acf_errors(Ce, C, slow):
    """Max normalized error over all lags and integrated slow-tail error
    over lags slow[0]..slow[1] (relative to the true integrated |C|)."""
    n = min(len(Ce), len(C)) - 1
    e_max = float(np.max(np.abs(Ce[:n + 1] - C[:n + 1])) / C[0])
    s0, s1 = slow[0], min(slow[1], n)
    e_slow = float(np.sum(np.abs(Ce[s0:s1 + 1] - C[s0:s1 + 1]))
                   / max(np.sum(np.abs(C[s0:s1 + 1])), 1e-300))
    return e_max, e_slow
