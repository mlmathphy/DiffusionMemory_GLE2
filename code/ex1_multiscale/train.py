"""Labels + distillation for Example 4 (torch; run on the workstation).

Variants (all through the identical pipeline of memdiff.pipeline):
  'mem'         c = (x_n, m_n), band and k data-driven
  'markov'      the memoryless baseline, c = x_n
  'window'      raw history: k retained lags = k+1 states, the
                bank's conditioning dimension (ideal reference AR(k+1))
  'window_long' raw window of WINDOW_LONG_M lags (dim 1 + M)

    python train.py                    # default: mem, markov, window
    python train.py window_long        # the expensive long window (opt-in)
    python train.py mem window         # any subset
"""

import sys

import numpy as np

import config as cfg
from memdiff.distill import FlowMapNet          # noqa: F401 (re-export)
from memdiff.pipeline import BaseExample
from conditioning import (rates_from_band, build_tuples,
                             build_window_tuples,
                             select_rates_by_prediction)

VARIANTS = {"mem": None, "markov": 0, "window": "lags",
            "window_long": "lags_long"}
# window_long is expensive (65-dim kNN over ~2e6 points): opt-in only
DEFAULT_VARIANTS = ("mem", "markov", "window")


class MultiscaleExample(BaseExample):

    variants = tuple(VARIANTS)
    latent_dim = 1
    _selection_cache = None

    def _select(self, X_train):
        if self._selection_cache is None:
            sel = select_rates_by_prediction(X_train)
            if sel["acf_err_chosen"] > cfg.SEL_ACF_TOL:
                sys.exit(
                    f"NO-GO: stage B fell back on the training data "
                    f"(closed-loop ACF error {sel['acf_err_chosen']:.4f} "
                    f"> SEL_ACF_TOL = {cfg.SEL_ACF_TOL}).")
            self._selection_cache = sel
        return self._selection_cache

    def prepare_conditioning(self, name, X_train):
        kind = VARIANTS[name]
        if kind == 0:               # memoryless
            tup = build_tuples(X_train, np.array([]), kappa_s=cfg.KAPPA_S)
            return tup, None, np.array([]), None
        if kind == "lags":          # raw window at the bank's dimension
            k_sel = cfg.K_BANK or self._select(X_train)["k"]
            print(f"  raw history window: {k_sel} retained lags = "
                  f"{1 + k_sel} states (dim {1 + k_sel}, matching 'mem'; "
                  f"ideal reference AR({1 + k_sel}))")
            tup = build_window_tuples(X_train, k_sel, kappa_s=cfg.KAPPA_S)
            return tup, None, tup["rates"], None
        if kind == "lags_long":     # long raw window
            M = cfg.WINDOW_LONG_M
            print(f"  LONG history window: last {M} states (dim {1 + M})")
            tup = build_window_tuples(X_train, M, kappa_s=cfg.KAPPA_S)
            return tup, None, tup["rates"], None
        sel = self._select(X_train)                     # 'mem'
        k, rates = ((sel["k"], sel["rates"]) if cfg.K_BANK is None else
                    (cfg.K_BANK, rates_from_band(sel["band"], cfg.K_BANK)))
        print(f"  k={k}, band {sel['band']}, rates {np.round(rates, 4)}")
        tup = build_tuples(X_train, rates, kappa_s=cfg.KAPPA_S)
        return tup, sel["band"], rates, sel


if __name__ == "__main__":
    import sys as _s
    _argv = _s.argv[1:] if len(_s.argv) > 1 else list(DEFAULT_VARIANTS)
    MultiscaleExample(cfg).main(_argv)
