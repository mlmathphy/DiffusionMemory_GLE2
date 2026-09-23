"""Labels + distillation for ex2_gle2d (torch; workstation).

Variants through the shared memdiff pipeline:
  'mem'     c = q_hat, the 2D data-estimated predictive coordinate
            (OLS compression of the (v, m) tuple; SCOUT_V3.md Stage B)
  'markov'  c = v, the memoryless baseline (2D)

    python train.py            # both variants
    python train.py mem        # one variant

Run order: run_all.py enforces stage 0 (stage0_validate.py) BEFORE
this script; do not run training on a failed stage 0.
"""

import os
import sys

import numpy as np

import config as cfg
from conditioning import build_tuples, fit_compression
from exact_refs import band_rates
from memdiff.pipeline import BaseExample

DEFAULT_VARIANTS = ("mem", "markov")


class GLE2DExample(BaseExample):

    variants = DEFAULT_VARIANTS
    latent_dim = 2
    _verified = None

    def _verify_stage0(self, X_train):
        """Bind training to the ACTUAL validated configuration (codeX
        finding #4): stage0.json must record pass on the same data
        file, compression coefficients, rates, and sampler settings
        this training run is about to use."""
        if self._verified is not None:
            return self._verified
        import json
        from stage0_validate import _sha256
        p = os.path.join(cfg.OUT_DIR, "stage0.json")
        if not os.path.exists(p):
            sys.exit("NO-GO: stage0.json missing — run stage 0 first.")
        with open(p) as f:
            s0 = json.load(f)
        if not s0.get("pass", False):
            sys.exit("NO-GO: stage 0 did not pass.")
        b = s0.get("binding")
        if b is None:
            sys.exit("NO-GO: stage0.json has no binding block; rerun "
                     "stage 0.")
        if bool(b.get("smoke")) != bool(cfg.SMOKE):
            sys.exit("NO-GO: stage-0 record and this run disagree on "
                     "smoke mode; a smoke validation cannot authorize "
                     "real training (or vice versa).")
        if b["data_sha256"] != _sha256(os.path.join(cfg.OUT_DIR,
                                                    "data.npz")):
            sys.exit("NO-GO: data.npz changed since stage 0 ran.")
        cur = dict(J=cfg.J_NEIGHBORS, nu=cfg.NU, eps=cfg.EPS_METRIC,
                   n_ode=cfg.N_ODE, n_ref=cfg.N_REF)
        if b["sampler"] != cur:
            sys.exit(f"NO-GO: sampler settings changed since stage 0 "
                     f"({b['sampler']} vs {cur}).")
        band, rates = band_rates()
        if not np.allclose(b["rates"], rates, rtol=0, atol=1e-12):
            sys.exit("NO-GO: rates changed since stage 0.")
        coef, nb, stride = fit_compression(X_train, rates)
        if not np.allclose(np.asarray(b["coef"]), coef,
                           rtol=0, atol=1e-10):
            sys.exit("NO-GO: refit compression coefficients differ "
                     "from the validated ones.")
        self._verified = (band, rates, coef, stride)
        return self._verified

    def prepare_conditioning(self, name, X_train):
        band, rates, coef, stride = self._verify_stage0(X_train)
        if name == "mem":
            tup = build_tuples(X_train, rates, "mem", coef=coef)
            print(f"  mem: k={cfg.K_BANK} bank compressed to 2D q_hat "
                  f"(OLS, {stride}-strided pool; stage-0-bound)")
            return tup, band, rates, {"k": cfg.K_BANK,
                                      "band": list(band),
                                      "compression": "ols_qhat_2d"}
        if name == "markov":
            tup = build_tuples(X_train, np.array([]), "markov")
            return tup, None, np.array([]), None
        raise ValueError(name)


if __name__ == "__main__":
    argv = sys.argv[1:] if len(sys.argv) > 1 else list(DEFAULT_VARIANTS)
    GLE2DExample(cfg).main(argv)
