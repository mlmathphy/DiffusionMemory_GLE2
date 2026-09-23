"""Train missing baselines, evaluate, and plot; preserve the memory model."""

import subprocess
import sys
from pathlib import Path

import config as cfg


def main():
    root = Path(__file__).resolve().parent
    out = Path(cfg.OUT_DIR)
    for name in ("data.npz", "compression.npz", "stage0.json", "mem/model.pt"):
        if not (out / name).exists():
            raise SystemExit(f"Missing {out / name}. Run on the workstation with the completed memory run.")
    for name, command in (("markov", ["train.py", "markov"]),
                          ("lstm", ["train_lstm.py"])):
        if (out / name / "model.pt").exists():
            print(f"Keep existing {name} checkpoint", flush=True)
        else:
            subprocess.run([sys.executable, *command], cwd=root, check=True)
    subprocess.run([sys.executable, "evaluate.py"], cwd=root, check=True)


if __name__ == "__main__":
    main()
