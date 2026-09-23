"""ex2_gle2d pipeline configuration (single source of truth).

Frozen regime = scout grid INDEX 9, fixed by the v2 selection rule and
the Stage-A/B evidence chain (SCOUT.md, SCOUT_V3.md): the sampler is
accurate on this model's bank-conditioned law when conditioned on the
2D data-estimated predictive coordinate; dim-12 conditioning is not
deployed anywhere in this pipeline.

Stage 0 (numpy, no torch) validates estimated-coordinate sampling AND
sampler-in-the-loop rollout against exact references and ABORTS the
run before any training if its predeclared checks fail.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

# ---------------- physical model (frozen regime: scout index 9) -----
F_MEM = 0.9            # tr G_mem / (tr Gamma0 + tr G_mem)
TAU2 = 5.0             # slow kernel mode timescale
THETA = 60.0           # angle between u1 and u2 (degrees)
DT = 0.2               # observation interval
KBT = 1.0

# ---------------- conditioning (frozen from the scout) --------------
K_BANK = 5             # v2-selected k at index 9 (conditioning dim 12
                       # BEFORE compression; the deployed coordinate is
                       # the 2D OLS increment prediction)
BAND_FROM_EXACT = True  # band from the exact state-ACF rule (scout)
ACF_SIG = 0.01
BAND_MAX_LAG = 2000
BURN_TOL = 1e-3

# ---------------- data ----------------------------------------------
N_TRAJ = 32            # training trajectories
N_TEST = 8             # held-out trajectories (stage 0 + evaluation)
L_TRAJ = 25000         # steps per trajectory (5000 t.u. = 1000 tau2)
SEED_DATA = 10

# ---------------- training-free sampler -----------------------------
J_NEIGHBORS = 1024
NU = 0.2
EPS_METRIC = 0.3
N_ODE = 400
N_REF = 200000         # reference-tuple pool (strided from training
                       # tuples; the Stage-A evidence level)
LABEL_BATCH = 4096

# ---------------- labels + distillation -----------------------------
N_LABELS = 180000      # generated samples for distillation; MUST be
                       # strictly below every variant's reference-pool
                       # row count (striding to N_REF can land a few
                       # rows short, e.g. 193,648 for 'mem'), because
                       # the shared pipeline draws label queries
                       # without replacement (codeX finding #1)
SEED_LABELS = 1
HIDDEN = 128
N_LAYERS = 3
N_EPOCHS = 3000
BATCH_TRAIN = 4096
LR = 1e-3
PATIENCE = 50
SEED_TRAIN = 2

# ---------------- LSTM baseline (state + increment inputs) ----------
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
LSTM_MIX = 1           # Gaussian head: correctly specified here
LSTM_SEQ = 500         # 100 t.u. = 20 tau2 per training subsequence
LSTM_BURN = 100
LSTM_EPOCHS = 2000
LSTM_BATCH = 64
LSTM_LR = 1e-3
LSTM_PATIENCE = 40
SEED_LSTM = 5

# ---------------- stage 0: predeclared validation thresholds --------
# One-step (v2 gate-D criteria, verbatim):
S0_ANCHORS = 48
S0_SAMPLES = 1024
S0_W2_MOMENT = 0.05        # median normalized Gaussian-moment W2
S0_PROJ_FACTOR = 2.0       # median projected W1 <= factor x calib
# Sampler-in-the-loop rollout (kNN sampler autoregressive, no net):
S0_N_ENS = 32
S0_N_ROLL = 2000           # steps (400 t.u. = 80 tau2)
S0_COV_TOL = 0.10          # |Cov(v) - I|_max
S0_VACF_TOL = 0.10         # max-entry VACF error to lag 3 tau2 / DT,
                           # normalized by max entry of Sv(0); the
                           # same-size EXACT ensemble's error is
                           # reported alongside as the noise floor
S0_DIFF_TOL = 0.15         # sampled-displacement diffusion tensor,
                           # relative Frobenius, against the
                           # TRUNCATION-MATCHED exact reference (the
                           # deterministic truncation error is
                           # computed and reported separately)
S0_CUTOFF_LIST = (25, 50, 75, 125, 250)
                           # diffusion-estimator lag cutoffs scanned
                           # by `stage0_validate.py --cutoff-scan`
                           # (exact ensembles only, ~1 min)
S0_FLOOR_REPS = 8          # independent exact-ensemble replicas in
                           # the scan: a Monte-Carlo floor estimate,
                           # not a single draw
S0_TRUNC_MEANINGFUL = 0.05 # a cutoff is only MEANINGFUL if its
                           # deterministic truncation error vs the
                           # exact D is <= this (otherwise the check
                           # no longer measures the diffusion tensor)
S0_RESOLVE_FRAC = 0.5      # resolvability requirement (codeX): a
                           # rollout check is judged ONLY if the
                           # same-size exact-ensemble floor is <=
                           # this fraction of its tolerance. The
                           # declared tolerances are never loosened;
                           # if any check is unresolvable, stage 0
                           # reports INCONCLUSIVE and pauses the
                           # chain BEFORE the sampler rollout — no
                           # automatic training on unresolved
                           # reference noise.
SEED_STAGE0 = 30

# ---------------- evaluation ----------------------------------------
N_GEN_TRAJ = 64
L_GEN = 10000          # bounded budget: 64 x 10000 network steps per
                       # model — cheaper than sampler rollouts (small
                       # MLP/LSTM forward per step), a few minutes
                       # each, and it halves the diffusion-report
                       # noise floor relative to 5000
EVAL_MAX_LAG = 150         # VACF matrix comparison range (6 tau2)
EVAL_WARM = 1000           # prehistory steps for rollout/anchor warm
N_QUERY = 256              # one-step anchors
NS_COND = 1024             # draws per anchor
SEED_EVAL = 40

OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "out")
FIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "figs")

# ---------------- smoke mode (EX2_SMOKE=1; `python run_all.py smoke`)
# Tiny end-to-end MECHANICAL check (codeX): exercises data, stage-0
# code paths, label generation, a few optimizer updates, checkpoint
# loading, and rollout in ISOLATED outputs (out_smoke/, figs_smoke/).
# Its stage-0 thresholds are NOT scientific at these sizes: stage 0
# records pass mechanically (marked smoke_mechanical_only) so later
# stages can be exercised; a smoke run never authorizes real training.
SMOKE = bool(_os.environ.get("EX2_SMOKE"))
if SMOKE:
    N_TRAJ, N_TEST, L_TRAJ = 2, 2, 3000
    N_REF, N_LABELS = 20000, 4000
    J_NEIGHBORS, N_ODE = 256, 50
    S0_ANCHORS, S0_SAMPLES = 8, 256
    S0_N_ENS, S0_N_ROLL = 8, 200
    N_EPOCHS, BATCH_TRAIN, PATIENCE = 30, 1024, 10
    LSTM_EPOCHS, LSTM_SEQ, LSTM_BURN, LSTM_PATIENCE = 30, 200, 50, 10
    N_GEN_TRAJ, L_GEN, EVAL_MAX_LAG, EVAL_WARM = 8, 500, 50, 500
    N_QUERY, NS_COND = 16, 256
    OUT_DIR = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "out_smoke")
    FIG_DIR = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "figs_smoke")
