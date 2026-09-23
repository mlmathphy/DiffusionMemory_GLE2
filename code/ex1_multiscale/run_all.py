"""Stage runner for Example 4 (multiscale GLE).

    python run_all.py prevet      # numpy: design tables + GO gate
    python run_all.py data        # numpy: exact trajectories
    python run_all.py train       # torch: mem, markov, window, window_long
    python run_all.py lstm        # torch: recurrent baseline
    python run_all.py eval        # torch: N_SEED-banded comparison + figure
                                  # (rollouts fingerprint-cached in out/cache/)
    python run_all.py fans        # torch inference: exact full-history
                                  # forecast fans (Stage-0 diagnostic)
    python run_all.py onestep     # torch inference: one-step conditional
                                  # accuracy vs the exact targets
    python run_all.py stats       # conditional forecasts: mean/std in
                                  # ex4_ensemble.pdf, plus the separate
                                  # ex4_w2.pdf diagnostic
    python run_all.py rank        # numpy: predictive-rank analysis
                                  # (figs/ex4_rank.pdf)
    python run_all.py figs        # manuscript figure set (make_figs.py;
                                  # torch figures skipped if checkpoints
                                  # are absent)
    python run_all.py all         # everything, in order

Not in `all`: the optional Stage-0.5 LSTM context-length sweep
(`python run_all.py sweep`).

The pre-vet gate is enforced: multiscale_exact.py exits nonzero on
NO-GO, so `all` aborts before any GPU stage.
"""

import config                            # noqa: F401 (sys.path bootstrap)
from memdiff.runner import run_stages

STAGES = {
    "prevet": ["python", "multiscale_exact.py"],
    "data": ["python", "generate_data.py"],
    "train": ["python", "train.py"],
    "lstm": ["python", "train_lstm.py"],
    "eval": ["python", "evaluate.py"],
    "fans": ["python", "forecast_fans.py"],
    "onestep": ["python", "onestep_accuracy.py"],
    "stats": ["python", "ensemble_stats.py"],
    "rank": ["python", "rank_analysis.py"],
    "figs": ["python", "make_figs.py"],
    # optional Stage-0.5 robustness study, not part of `all`:
    # LSTM context-length sweep on the same data (sweep_lstm.py)
    "sweep": ["python", "sweep_lstm.py"],
}
# onestep precedes eval so that summary.json's provenance block can
# absorb the one-step record in a single `all` pass
ORDER = ["prevet", "data", "train", "lstm", "onestep", "eval", "fans",
         "stats", "rank", "figs"]

if __name__ == "__main__":
    run_stages(STAGES, ORDER)
