"""Example 3 candidate: intermittent scrape-off-layer (SOL) fluctuations as
a multi-timescale filtered-Poisson process -- PRE-VET configuration.

STATUS (2026-08-22): CANDIDATE, PRE-VET STAGE, numpy only, no training.
Ruling that created it (codeX): the E x B zonal candidate is rejected;
"move to a carefully designed multi-timescale filtered-Poisson SOL
pre-vet; do not yet build its Torch training pipeline", with the
requirements: >= 2 physically motivated pulse relaxation times;
asymmetric rise/decay; an observed scalar that does not reveal the
separate pulse components; Poisson arrivals producing a no-arrival atom
plus a continuous burst tail; parameters fixed from dimensionless SOL
intermittency quantities, not selected to defeat the LSTM; the same
bank-vs-window and Gaussian-vs-mixture gates; the physical law specified
INDEPENDENTLY of the method's EMA bank (bank selected afterwards from
observed trajectories).

Physical model (stochastic pulse / filtered-Poisson process, the standard
reduced model of SOL intermittency: Garcia PRL 2012; Theodorsen, Garcia,
Rypdal 2016-2018; Militello & Omotani 2016).  The observed signal is

    y(t) = sum_j Phi_j(t) + sigma_n xi(t),      j = pulse populations,
    Phi_j(t) = sum_k A_{jk} psi_j(t - t_{jk}),
    psi_j(theta) = c_j [exp(-theta/tau_dj) - exp(-theta/tau_r)],  theta >= 0,

i.e. each pulse rises on the time scale tau_r and decays on tau_dj
(c_j = tau_dj/(tau_dj - tau_r) normalizes the pulse integral to tau_dj);
arrivals t_{jk} are Poisson with rate nu_j = gamma_j / tau_dj, amplitudes
A_{jk} ~ Exp(mean a_j); xi is white observation noise.  Two populations
with distinct relaxation times: fast filament bursts (tau_d1 = 1, the
time unit) and rarer, larger, long-lived structures (tau_d2 = 10).  Only
the SUM is observed; the component amplitudes are hidden.  Each
population is a linear filter of shot noise, so the hidden state is
(X_dj, X_rj) per population (exact simulation, exact one-step law given
the hidden state, closed-form ACF), but the observation is neither
Markov nor Gaussian: the one-step law is a no-arrival atom (smeared by
the noise) plus a continuous burst tail.

Dimensionless regime (FROZEN 2026-08-22 per codeX ruling; wording
qualifications recorded verbatim):
  intermittency parameter gamma_1 = tau_d1/tau_w1 = 2   (SOL: ~1-3),
  gamma_2 = 1 for the slow population (rarer events),
  pulse asymmetry tau_r/tau_d1 = 0.1 -- PHYSICALLY GROUNDED (two-sided
    exponential pulses with asymmetry ~0.1 are reported in the SOL
    literature, e.g. Garcia et al. 2018),
  variance balance: the slow population carries the same variance as the
  fast one (a_2 fixed by Campbell's theorem, see model.py),
  observation-noise variance ratio eps = 0.01 of Var(y) and sampling
    DT = 0.5 tau_d1 (= 5 tau_r: the pulse rise is not resolved at the
    flow-map step) -- these two are OBSERVATION / COARSE-GRAINING CHOICES
    selected during the declared design scan (design_scan.py,
    out/design_scan.json), NOT plasma constants inferred from experiments.
    Pilot values were tau_r = 0.15, eps = 0.02, DT = 0.2.
Separation tau_d2/tau_d1 = 10 is moderate on purpose (codeX: the slow
scale is a declared physical regime, not placed beyond an LSTM context).

Reference for the observable conditional: a particle filter on the
4-dimensional hidden state (pfilter.py) gives p(y_{n+1} | y_{0:n}); the
hidden-state oracle is kept only as a diagnostic (codeX: the observable
conditional is the posterior mixture over hidden states, so a sampler
that reproduces it legitimately differs from the hidden-state law).

Conditioning of the memory model: c_n = [y_n, m_n (k) or z_n (p)],
m_n the paper's EMA bank of increments Delta y, rates log-spaced in the
data-selected band; the bank is selected AFTER the physics is fixed and
plays no role in the law.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

# ---------------- physics (time unit = tau_d1) ----------------------
TAU_D = (1.0, 10.0)       # decay times of the pulse populations
GAMMA = (2.0, 1.0)        # intermittency parameters gamma_j = tau_dj/tau_wj
TAU_R = 0.1               # common rise time (frozen; pilot 0.15)
A1 = 1.0                  # mean amplitude of the fast population (units)
VAR_SHARE = (0.5, 0.5)    # variance shares -> a_2 from Campbell's theorem
EPS_NOISE = 0.01          # observation-noise variance / Var(y) (frozen; pilot 0.02)
DT = 0.5                  # sampling step (frozen; pilot 0.2)
BURN_T = 120.0            # discarded initial transient (12 tau_d2)

# ---------------- pre-vet sizes --------------------------------------
# anchors: N_ANCH per stratum per test record, drawn from N_ANCH_TRAJ
# trajectories of the record at times >= PF_WARM (filter converged);
# the particle filter runs along those trajectories with PF_N particles,
# and the convergence check repeats PF_CONV_ANCH anchors with 2 PF_N.
# Reference validation (codeX, round 162): the main filter is GUIDED with
# PF_N particles; on N_CONV_TRAJ trajectories per record three replicate
# filters (guided PF_N with another seed, guided 2 PF_N, bootstrap 2 PF_N)
# are run and compared at that trajectory's anchors; N_UNSTRAT unstratified
# anchors per record (random times >= PF_WARM on the same trajectories)
# serve the PF-vs-Markov predictive-variance comparison; PIT calibration
# is collected at every step >= PF_WARM of the main filters.
SIZES = {
    "quick": dict(N_TRAJ_TR=160, N_TRAJ_TEST=100, L_REC=2000, N_TEST_REC=3,
                  N_ANCH=32, N_ANCH_TRAJ=8, N_UNSTRAT=48, N_CONV_TRAJ=3,
                  N_SAMP=512, N_ODE=200, PF_N=4000, PF_WARM=200),
    "full":  dict(N_TRAJ_TR=400, N_TRAJ_TEST=200, L_REC=4000, N_TEST_REC=3,
                  N_ANCH=64, N_ANCH_TRAJ=16, N_UNSTRAT=96, N_CONV_TRAJ=5,
                  N_SAMP=1024, N_ODE=400, PF_N=8000, PF_WARM=200),
}
N_CONV_DRAW = 4096        # predictive draws for replicate-convergence W2
SEED_BASE = 20260823

# ---------------- memory bank / rank -------------------------------
K_LIST = (2, 3, 4, 6, 8)
ACF_SIG_THRESH = 0.02     # state-ACF significance -> slow end (ex4 rule)
BURN_TOL = 0.05
TAIL_TOL = 1e-10          # geometric-sum truncation in the closed forms
H_RANK = 6
ENERGY_RULE = 0.99
N_OUT_ACF = 200           # closed-loop ACF horizon (lags) for the design
SLOW_TAIL_T = (10.0, 40.0)   # integrated slow-tail range in PHYSICAL time
                             # (tau_d2 .. 4 tau_d2); converted to lags with DT


def slow_tail_lags(dt=None):
    d = DT if dt is None else dt
    return (int(round(SLOW_TAIL_T[0] / d)), int(round(SLOW_TAIL_T[1] / d)))

# ---------------- sampler (paper defaults) -------------------------
J_NEIGHBORS = 1024
NU = 0.2
EPS_METRIC = 0.3
K_MIX = 8

# ---------------- anchors: strata by state quantile -----------------
Q_QUIET = 0.5             # quiet: y below the median
Q_ACTIVE = 0.8            # active: y above the 80th percentile

# =====================================================================
# TORCH STAGES (workstation): the two-model comparison ratified by codeX
# (2026-08-22) -- memory-conditioned diffusion + distilled flow map vs an
# LSTM with a Gaussian head, on identical trajectories; the guided
# particle filter is the evaluation reference only; the closed-form G1
# result is supporting evidence, not a trained baseline.
# =====================================================================
OUT_DIR = "out"
FIG_DIR = "figs"

# ---- data (exact simulation, model.simulate) ----
N_TRAJ = 400              # training trajectories
L_TRAJ = 4000             # steps per trajectory (2000 t.u. = 200 tau_d2)
N_TEST = 120              # held-out trajectories (anchors, prehistories)
SEED_DATA = 20260824

# ---- bank selection (Algorithm 1, stage A/B; ex4 conventions) ----
K_BANK = None             # None -> data-driven selection
K_MAX = 8
N_RATE_DICT = 40
N_TRAJ_SELECT = 40
LMAX_FACTORS = (1.0, 2.0, 3.0)   # candidate lam_max = factor / DT
ACF_MAX_LAG = 400         # lags (200 t.u.) for the state-ACF slow end
SEL_ACF_TOL = 0.05        # stage-B closed-loop tolerance
TIE_TOL = 0.02            # stage A forwards near-tied bands (ex4 amendment;
                          # the one-step trap flattens stage-A residuals)
REDUCE = True             # deploy predictive coordinates z = P m (App. A.3)
KAPPA_S = None            # None -> 1/std(increments)

# ---- training-free sampler / labels / distillation (paper defaults) ----
N_ODE = 400
N_LABELS = 200000
LABEL_BATCH = 4096
SEED_LABELS = 1
HIDDEN = 128
N_LAYERS = 3
N_EPOCHS = 3000
BATCH_TRAIN = 4096
LR = 1e-3
PATIENCE = 50
SEED_TRAIN = 2

# ---- LSTM + Gaussian head (memdiff.lstm.train_lstm; inputs (y, dy)) ----
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
LSTM_MIX = 1              # single Gaussian head (the tested baseline)
LSTM_SEQ = 1000           # 500 t.u. = 50 tau_d2 per training subsequence
LSTM_BURN = 200
LSTM_EPOCHS = 4000
LSTM_BATCH = 32
LSTM_LR = 1e-3
LSTM_PATIENCE = 30
LSTM_WARM = 1000          # deployment warm-up (observed prehistory steps)
LSTM_SEEDS = (5,)         # ONE trained benchmark instance (codeX/user:
                          # Example-2-style comparison, one model per side;
                          # seeds 6, 7 may exist on disk as archive only and
                          # are never loaded by evaluate.py)
SEED_LSTM = 5             # set per seed by train_lstm.py

# ---- evaluation ----
SEED_EVAL = 3
N_ANCH_EVAL = 64          # per stratum (quiet / active) per test set
N_UNSTRAT_EVAL = 64
N_ANCH_TRAJ_EVAL = 16     # test trajectories carrying anchors
PF_N_EVAL = 8000          # guided particle filter size (reference)
PF_WARM = 200             # anchors only after this many assimilated steps
PF_CONV_TRAJ = 4          # trajectories re-run with 2 PF_N (convergence)
N_SAMP_EVAL = 1024        # draws per anchor and model
N_SEED_ROLL = 4           # rollout noise seeds
N_GEN_TRAJ = 64           # prehistories per rollout seed (paired)
L_GEN = 8000              # rollout steps (4000 t.u.)
DISCARD = 400             # transient discarded before rollout statistics
BURST_SIGMA = 2.5         # burst threshold: y > mean + BURST_SIGMA sd

# ---- smoke overrides (mechanical check; outputs carry the _smoke tag) ----
SMOKE = dict(N_TRAJ=24, L_TRAJ=1200, N_TEST=12, N_TRAJ_SELECT=12, N_LABELS=4000,
             N_EPOCHS=30, PATIENCE=5, LSTM_EPOCHS=60, LSTM_PATIENCE=5,
             LSTM_SEQ=200, LSTM_BURN=40, LSTM_WARM=150, LSTM_SEEDS=(5,),
             N_ANCH_EVAL=6, N_UNSTRAT_EVAL=6, N_ANCH_TRAJ_EVAL=3, PF_N_EVAL=1000,
             PF_WARM=100, PF_CONV_TRAJ=1, N_SAMP_EVAL=256, N_SEED_ROLL=1,
             N_GEN_TRAJ=8,
             L_GEN=600, DISCARD=100, K_MAX=4, ACF_MAX_LAG=200,
             # the stage-B NO-GO gate stays fail-closed for the full run; at
             # smoke size the data ACF itself is noisy at the few-% level, so
             # the gate is disabled there (smoke verdicts are meaningless)
             SEL_ACF_TOL=1.0)
TAG = ""                  # "" or "_smoke" (figure/summary suffix)


def apply_smoke():
    """Activate the smoke sizes (call when '--smoke' is on the command
    line, BEFORE the stage reads any size). Smoke outputs live under
    out/smoke/ and figs/smoke/ so they never touch the full-run artifacts."""
    import os as _o
    g = globals()
    g.update(SMOKE)
    g["TAG"] = "_smoke"
    g["OUT_DIR"] = _o.path.join("out", "smoke")
    g["FIG_DIR"] = _o.path.join("figs", "smoke")


# ---------------- predeclared gates (corrected per codeX ruling) -----
# Excess errors: E_norm(model) = [W2^2(model, PF draw) - W2^2(PF draw',
# PF draw)] / Var(PF draw), against the particle-filter observable-
# history oracle (raw floor ratios are kept as diagnostics only; they
# depend on N_SAMP).
GATES = dict(
    # G1 memory representation (closed form, ex4 precedent): ideal bank at
    # k* vs ideal raw window AR(k*+1) at EQUAL conditioning dimension
    EPS_K_MAX=0.05, BANK_VS_AR_MIN=2.0,
    # G2 generator advantage at matched observable information: excess
    # error of the Gaussian with the PF predictive moments (the ideal
    # observable Gaussian head) / excess error of the sampler, median
    GAUSS_OVER_SAMPLER_MIN=2.0,
    # G3 sampler feasibility: absolute excess error vs the PF oracle;
    # reference reliability (codeX round 162): 90th percentile over the
    # replicate anchors of W2^2(guided 2N, guided N)/Var <= PF_CONV_MAX
    # (bootstrap-2N and same-N replicate reported), predictive calibration
    # |mean PIT - 0.5| <= PIT_MEAN_TOL and KS <= PIT_KS_MAX, unstratified
    # E[Var(Y'|history)] <= E[Var(Y'|Y_n)] (kNN Markov, same anchors) +
    # N_SE_PF paired standard errors; label-stage ESS
    E_NORM_MAX=0.05, PF_CONV_P90_MAX=0.02, PIT_MEAN_TOL=0.02, PIT_KS_MAX=0.05,
    N_SE_PF=2.0, ESS_MED_MIN=200.0, ESS_P05_MIN=20.0,
    # G4 physics / exactness: simulator vs closed form within N_SE standard
    # errors (across independent trajectories) -- revised after the pilot
    # (fixed 2% tolerance missed on one record by sampling noise; the
    # SE-scaled form is sample-size aware)
    SKEW_RANGE=(0.5, 3.0), FLAT_RANGE=(3.5, 15.0), ATOM_RANGE=(0.2, 0.9),
    N_SE=4.0,
)
