"""Stage runner for Example 3 (two-model comparison).

    python run_all.py smoke      # mechanical check: every stage with --smoke
                                 # (writes only *_smoke outputs)
    python run_all.py            # data -> train -> lstm -> eval (full)
    python run_all.py <stage>    # one stage: data | train | lstm | eval
    python run_all.py physics    # post-processing physics figures (numpy)
    python run_all.py figs       # conditional-forecast figure (torch + PF)

The numpy pre-vet (prevet.py) and the design scan are separate and were
run before this stack was built.
"""

import os
import sys

# the shared package lives in the sibling folder ../memdiff (repo layout
# code/memdiff next to code/ex3_sol_fpp); make it importable first
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))
from memdiff.runner import run_stages          # noqa: E402

PY = sys.executable
STAGES = {
    "data": [PY, "generate_data.py"],
    "train": [PY, "train.py"],
    "lstm": [PY, "train_lstm.py"],
    "eval": [PY, "evaluate.py"],
    # figure stages are NOT part of the default pipeline (codeX): run them
    # explicitly -- `python run_all.py physics` (numpy post-processing) and
    # `python run_all.py figs` (forecast figure: torch + a PF forecast)
    "physics": [PY, "make_physics_figs.py"],
    "figs": [PY, "make_figs.py"],
}
ORDER = ["data", "train", "lstm", "eval"]

if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        for s in ORDER:
            print(f"\n##### smoke stage: {s} #####")
            run_stages({s: STAGES[s] + ["--smoke"]}, [s], argv=[s])
    else:
        run_stages(STAGES, ORDER)
