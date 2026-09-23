"""Stage runner shared by every example (no torch: usable anywhere).

Kept out of pipeline.py so that the numpy-only stages -- and the
inference runner's checkpoint guard -- work on a machine without a
torch installation.
"""

import subprocess
import sys


def run_stages(stages, order, argv=None):
    """Shared run_all driver: run one named stage, or all in order.

    stages: {name: command list}; order: stage names for 'all'.
    Each stage is a subprocess with check=True, so a nonzero exit (e.g.
    a pre-vet NO-GO) aborts the sequence before later stages run.
    """
    argv = sys.argv[1:] if argv is None else argv
    arg = argv[0] if argv else "all"
    for s in (order if arg == "all" else [arg]):
        print(f"\n##### stage: {s} #####")
        subprocess.run(stages[s], check=True)
