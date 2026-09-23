"""BaseExample: the shared label-generation + distillation pipeline.

Each numerical example subclasses BaseExample and provides only what is
example-specific:

  latent_dim            -- dimension of the latent z (= state dimension)
  variants              -- variant names, default ('mem', 'markov')
  prepare_conditioning  -- build the conditioning tuples for a variant
                           (selection, gates, baseline definition)

Everything else -- fitting the training-free sampler, generating the
labeled triples with diagnostics, saving labels.npz, distilling the
flow-map network, and saving model.pt -- is the identical template of
Algorithm 1 and lives here.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from memdiff.runner import run_stages    # noqa: F401 (re-export)
from memdiff.distill import distill
from memdiff.sampler import TrainingFreeSampler


class BaseExample:

    variants = ("mem", "markov")
    latent_dim = None            # subclasses must set (d of the state)

    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------ subclass hooks

    def prepare_conditioning(self, name, X_train):
        """Return (tup, band, rates, selection) for variant `name`.

        tup is the dict of build_tuples: C (N, d_c), S (N, d_s),
        kappa_s, n_burn, ... Selection gates that must abort the run
        should sys.exit here, before any label generation.
        """
        raise NotImplementedError

    # ------------------------------------------------ shared template

    def make_sampler(self, C, S, device=None, **overrides):
        """TrainingFreeSampler with defaults filled from the config."""
        cfg = self.cfg
        kw = dict(j_neighbors=cfg.J_NEIGHBORS, nu=cfg.NU,
                  eps=cfg.EPS_METRIC, n_ode=cfg.N_ODE,
                  label_batch=cfg.LABEL_BATCH)
        kw.update(overrides)
        return TrainingFreeSampler(C, S, device=device, **kw)

    def train_variant(self, name, device):
        cfg = self.cfg
        print(f"== variant '{name}' ==")
        data = np.load(os.path.join(cfg.OUT_DIR, "data.npz"))
        X_train = data["X_train"]

        tup, band, rates, selection = self.prepare_conditioning(name,
                                                                X_train)
        print(f"  conditioning dim {tup['C'].shape[1]}, tuples "
              f"{tup['C'].shape[0]} (burn-in {tup['n_burn']} steps), "
              f"kappa_s {tup['kappa_s']:.4f}")

        smp = self.make_sampler(tup["C"], tup["S"], device=device)
        rng = np.random.default_rng(cfg.SEED_LABELS)
        q_idx = rng.choice(tup["C"].shape[0], size=cfg.N_LABELS,
                           replace=False)
        C_q = tup["C"][q_idx]
        z_q = rng.standard_normal((cfg.N_LABELS, self.latent_dim))
        print(f"  generating {cfg.N_LABELS} labels "
              f"(J = {smp.J}, nu = {smp.nu}, n_ODE = {smp.n_ode}) ...")
        t0 = time.perf_counter()
        s_q, _, diag = smp.sample_labels(C_q, z=z_q)
        wall_labels = time.perf_counter() - t0
        print(f"  diagnostics: ESS median {np.median(diag['ess']):.0f} "
              f"(5% {np.percentile(diag['ess'], 5):.0f}), "
              f"radius median {np.median(diag['radius']):.3f}")

        vdir = os.path.join(cfg.OUT_DIR, name)
        os.makedirs(vdir, exist_ok=True)
        np.savez_compressed(os.path.join(vdir, "labels.npz"),
                            C=C_q, z=z_q, s=s_q,
                            ess=diag["ess"], radius=diag["radius"])

        t0 = time.perf_counter()
        net, norm = distill(C_q, z_q, s_q, device, cfg)
        wall_distill = time.perf_counter() - t0
        torch.save({
            "state": net.state_dict(), "norm": norm,
            "kappa_s": tup["kappa_s"], "rates": rates, "band": band,
            "k": len(rates), "selection": selection,
            "sampler": {"J": smp.J, "nu": smp.nu, "eps": smp.eps,
                        "n_ode": smp.n_ode},
            # cost accounting, read into summary.json by evaluate.py
            "wall_labels_s": wall_labels, "wall_distill_s": wall_distill,
        }, os.path.join(vdir, "model.pt"))
        # the same selection in a torch-free form, so numpy-only analysis
        # scripts can read the exact selected bank without loading model.pt
        with open(os.path.join(vdir, "selection.json"), "w") as f:
            json.dump({"k": int(len(rates)),
                       "band": None if band is None else [float(b) for b in band],
                       "rates": [float(r) for r in np.asarray(rates).ravel()],
                       "kappa_s": float(tup["kappa_s"])},
                      f, indent=2)
        print(f"  saved -> {vdir}/model.pt, {vdir}/selection.json")

    def main(self, argv=None):
        """CLI: train all variants, or the ones named on the command line."""
        argv = sys.argv[1:] if argv is None else argv
        device = "cuda" if torch.cuda.is_available() else "cpu"
        names = argv if argv else list(self.variants)
        for name in names:
            self.train_variant(name, device)
