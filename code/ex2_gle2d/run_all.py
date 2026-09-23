"""Stage runner for ex2_gle2d.

    python run_all.py smoke      # tiny end-to-end MECHANICAL check
                                 # (~2-4 min total, isolated
                                 # out_smoke/ + figs_smoke/): data,
                                 # stage-0 code paths, label
                                 # generation, a few optimizer
                                 # updates, checkpoint loading,
                                 # rollout. Run this FIRST; it never
                                 # authorizes real training.
    python run_all.py            # all stages in order (full)
    python run_all.py <stage>    # one stage

Order (a nonzero exit — e.g. the stage-0 NO-GO — aborts the rest, so
training never runs on a failed validation):

    prevet   exact-reference printout of the frozen regime  (seconds)
    data     exact trajectories -> out/data.npz             (~1-2 min)
    stage0   estimated-coordinate sampling + rollout vs
             exact references; ABORTS on failure            (~15-25 min)
    train    labels + distillation, variants mem markov     (GPU, ~tens
                                                             of minutes)
    lstm     state+increment LSTM baseline                  (GPU)
    eval     closed-loop + one-step evaluation, figure      (~minutes)
"""

import sys

import config  # noqa: F401  (puts the repo's code/ on sys.path)
from memdiff.runner import run_stages

PY = sys.executable

STAGES = {
    "paperfigs": [PY, "paper_figs.py"],  # cached short showcase; no training
    "paperplot": [PY, "paper_figs.py", "--plot"],  # plot-only redraw
    "finalfigs": [PY, "final_figs.py"],  # the FOUR final paper figures
                                         # (plot-only; needs the
                                         # paperfigs cache + summary)
    "compare": [PY, "compare.py"],   # missing baselines + eval; preserves mem
    "figs": [PY, "make_figs.py"],    # saved summary only; no inference
    "prevet": [PY, "exact_refs.py"],
    "data": [PY, "generate_data.py"],
    "scan": [PY, "stage0_validate.py", "--cutoff-scan"],   # optional
    "stage0": [PY, "stage0_validate.py"],
    "train": [PY, "train.py", "mem"],       # MEMORY MODEL ONLY first
    "markov": [PY, "train.py", "markov"],   # baseline, deferred
    "lstm": [PY, "train_lstm.py"],          # baseline, deferred
    "eval": [PY, "evaluate.py"],
}
# Amendment (2026-09-08, codeX + user): train and evaluate the memory
# model FIRST; baselines are added only after it works. Run them later
# with `python run_all.py markov`, `python run_all.py lstm`, then
# `python run_all.py eval` again (eval includes whichever models
# exist).
ORDER = ["prevet", "data", "stage0", "train", "eval"]

if __name__ == "__main__":
    import os

    argv = sys.argv[1:]
    if argv and argv[0] == "smoke":
        os.environ["EX2_SMOKE"] = "1"   # inherited by stage processes
        print("SMOKE MODE: reduced settings, isolated out_smoke/; "
              "mechanical check only")
        argv = ["all"]
    run_stages(STAGES, ORDER, argv=argv)
