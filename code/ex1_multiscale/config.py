"""Example 4: 1D GLE with memory spanning three decades of timescales.

Model (stationary form, prehistory to -infinity):

    dX/dt = -GAMMA X - C_K \\int_{-inf}^t e^{-LAM_K (t-s)} X_s ds + sum_j F_j
    dF_j  = -F_j/T_j dt + SIG_j sqrt(2/T_j) dW_j,   T_j in FORCING_TIMES

Exact (2 + n_f)-dimensional Markovian embedding Y = (X, Z, F_1..F_nf):
linear-Gaussian, so every reference below is closed form, exactly as in
Example 1.

Purpose (designed in closed form BEFORE any training): the forcing
correlation times span 0.5 .. 200 time units (2.5 .. 1000 steps), so

  * a raw history window at the bank's conditioning dimension (k + 1
    consecutive states, ideal reference AR(k+1)) passes ONE-STEP
    selection (residual within ~1% of V_full) yet fails in CLOSED LOOP:
    ideal AR(7) errs by ~29% of C(0) on the ACF, because seven samples
    spanning 1.2 time units cannot separate the slow modes;
  * the EMA bank at the SAME dimension errs by ~4-5%: its slow features
    integrate over unbounded horizons;
  * a window matching the bank's accuracy needs dimension well past 65
    (ideal AR(65) still errs ~11%);
  * the LSTM's training subsequences cover ONE slow correlation time
    (1000 steps), a demanding long-memory test for recurrent training;
    the Gaussian head stays correctly specified, so output-distribution
    mismatch is removed as a cause -- the example shows an empirical
    advantage, it does not isolate memory as the only possible cause.

All stages read parameters from here only. Pre-vet with

    python multiscale_exact.py     (numpy only; prints the design tables)
"""

# make the shared package importable when running from this folder
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

# ---------------- physical model ----------------
GAMMA = 0.5                # instantaneous drag
C_K   = 2.0                # kernel amplitude   K(t) = C_K exp(-LAM_K t)
LAM_K = 1.0                # kernel decay rate
FORCING_TIMES = (0.5, 4.0, 30.0, 200.0)   # OU correlation times
FORCING_SIGS  = (0.6, 0.5, 0.45, 0.45)   # stationary std per component

# ---------------- observation -------------------
DT       = 0.2      # time step (slowest mode = 1000 steps)
N_TRAJ   = 100      # training trajectories
L_TRAJ   = 20000    # steps per trajectory (4000 t.u. = 20 slow times)
N_TEST   = 20       # held-out trajectories
SEED_DATA = 0

# ---------------- memory model -------------------
# Band and k are data-driven as in Example 1, with ONE change: the slow
# end of the candidate grid comes from the significance range of the
# STATE autocovariance, not the increment autocovariance. Slow modes are
# invisible in increments ((DT/T)^2 amplitude) but plain in the state
# ACF; the closed-loop stage-B criterion then decides what is kept.
K_BANK   = None     # None -> data-driven selection
K_MAX    = 10
N_RATE_DICT = 40
N_TRAJ_SELECT = 20
LMAX_FACTORS = (1.0, 2.0, 3.0)   # candidate lam_max = factor / DT
ACF_SIG_THRESH = 0.05   # state-ACF significance level (slow end of grid)
SEL_ACF_TOL = 0.05  # stage B tolerance; forces the slow band
TIE_TOL     = 0.02  # Ex4 amendment: stage A forwards ALL bands whose
                    # held-out one-step residual is within this relative
                    # tie of the best; stage B decides among them by the
                    # closed-loop criterion. Without it the one-step
                    # criterion (blind to slow memory) discards every
                    # slow band before stage B sees one.
K_SWEEP  = [0, 1, 2, 3, 4, 5, 6, 8, 10]
TAIL_TOL = 1e-10    # geometric tail cutoff of feature covariances
BURN_TOL = 3e-2     # EMA burn-in: rho_max^n > BURN_TOL discarded.
                    # Slowest candidate rate 1/200 -> n_burn ~ 3500 of
                    # 20000 steps; a tighter tolerance would discard
                    # most of every trajectory.

# ---------------- window baselines ----------------
# 'window'      : raw history of k retained lags (k+1 states, the
#                 bank's conditioning dimension; ideal ref AR(k+1))
# 'window_long' : raw window at WINDOW_LONG_M lags (dimension 1 + M).
#                 OPTIONAL and expensive: ~2e6 conditions in 65
#                 dimensions defeat KD-tree pruning, so label generation
#                 may be slow and low-ESS. The closed-form AR(65)
#                 reference in evaluate.py carries the same message;
#                 train this variant only if a pilot shows acceptable
#                 runtime (`python train.py window_long`).
WINDOW_LONG_M = 64

# ---------------- LSTM baseline -------------------
# Fairness for long memory: training subsequences of one slow
# correlation time (1000 steps) and a long warm-up at deployment.
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 1
LSTM_MIX      = 1   # single Gaussian head -- correctly specified here
                    # (the exact conditionals are Gaussian); the paper
                    # must say "LSTM with a Gaussian head", NOT LSTM-GMM
LSTM_SEQ      = 1000
LSTM_BURN     = 200 # in any context-length sweep, scale this WITH
                    # LSTM_SEQ (~20-25% of it), not fixed: a fixed 200
                    # leaves too few supervised steps at short SEQ
LSTM_EPOCHS   = 4000
LSTM_BATCH    = 32
LSTM_LR       = 1e-3
LSTM_PATIENCE = 30
LSTM_WARM     = 3000
SEED_LSTM     = 5

# ------- LSTM context-length robustness sweep (sweep_lstm.py) -------
# Answers the anticipated reviewer question "would a longer-context
# LSTM recover the slow mode?" on the SAME data -- no new physics.
# Tokens per update are held ~constant by scaling the batch as 1/SEQ,
# so configurations match in compute per update and in the update cap.
LSTM_SWEEP_SEQ       = (1000, 2500, 5000)
LSTM_SWEEP_SEEDS     = (5, 6, 7)     # training seeds (robustness)
LSTM_SWEEP_BURN_FRAC = 0.25          # burn-in as a fraction of SEQ
LSTM_SWEEP_TOKENS    = 32000         # ~matched tokens per update
                                     # (batch scaled as 1/SEQ; actual
                                     # compute is reported via wall
                                     # time and realized updates)
LSTM_SWEEP_WARM      = 5000          # common deployment warm-up for
                                     # EVERY swept LSTM = the longest
                                     # training context, so the
                                     # longest-context model gets its
                                     # full context at initialization
                                     # and starting conditions stay
                                     # paired across the sweep. (Fan
                                     # queries are unchanged: FAN_HIST
                                     # = 5000 already bounds them.)

# ---------------- training-free sampler ----------
J_NEIGHBORS = 1024
NU          = 0.2
EPS_METRIC  = 0.3
N_ODE       = 400
KAPPA_S     = None

# ---------------- labels + distillation ----------
N_LABELS    = 200000
LABEL_BATCH = 4096
SEED_LABELS = 1
HIDDEN      = 128
N_LAYERS    = 3
N_EPOCHS    = 3000
BATCH_TRAIN = 4096
LR          = 1e-3
PATIENCE    = 50
SEED_TRAIN  = 2

# ---------------- pass thresholds ----------------
PASS_EPS_K     = 0.02   # eps at the selected (band, k)
PASS_ACF_IDEAL = 0.07   # ideal-bank closed-loop ACF error (selected
                        # k lands at ~0.04-0.06; window at same dim ~0.3)
PASS_GAP       = 2.0    # ideal window(M=k) err / ideal bank(k) err

# ---------------- evaluation ---------------------
N_QUERY     = 256
NS_COND     = 4000
N_SEED      = 8
L_GEN       = 20000
N_GEN_TRAJ  = 50
ACF_MAX_LAG = 5000               # selection-internal lag budget
EVAL_MAX_LAG = 150               # "shoulder" max-error statistics:
                                 # tau <= 30 (kept -- the window/bank
                                 # representation gap peaks here)
EVAL_LAG_FULL = 2000             # long statistics: tau <= 400 = 2 T_max.
                                 # Stage-0 correction: tau <= 30 never
                                 # tested the slowest (200 t.u.) mode --
                                 # exactly where the reset-state LSTM
                                 # should be weakest. Seed bands absorb
                                 # the extra tail-estimation noise, and
                                 # E_slow below integrates rather than
                                 # maximizes over the noisy tail.
SLOW_BAND = (0.25, 1.0)          # E_slow integration window, as
                                 # fractions of T_max = max(FORCING_TIMES)

# ----------- ensemble statistics (ensemble_stats.py) ---------------
ENS_N_RELEASE = 4000  # members of the single-release ensemble
ENS_POOL      = 256   # held-out queries scanned for the release point
                      # (the one with the largest |oracle mean| at
                      # tau = T_max -- the history where memory matters
                      # most; deterministic given SEED_EVAL)

# ------------- exact forecast fans (forecast_fans.py) --------------
# Per-query exact references: conditional mean/sd of X_{n+h} given the
# FULL observed history (Kalman on the embedding), h up to 2 T_max.
FAN_N_QUERY = 32     # held-out prehistories
FAN_N_ENS   = 1000   # ensemble members per query and case; the
                     # ensemble-mean estimate carries an irreducible
                     # normalized MC error ~ 1/sqrt(N) ~ 0.032, drawn
                     # as the "ensemble sampling floor" in the figure
FAN_HIST    = 5000   # prehistory steps fed to the exact filter: a
                     # numerically CONVERGED approximation to
                     # infinite-history conditioning (1000 t.u. = 5
                     # slow correlation times; the startup check
                     # verifies the one-step variance against V_full)
FAN_Q_BATCH = 8      # queries generated per flattened batch
                     # (8 x 1000 members x 2000 steps ~ 130 MB float64)
HIST_WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEED_EVAL   = 3

OUT_DIR  = "out"
FIG_DIR  = "figs"
