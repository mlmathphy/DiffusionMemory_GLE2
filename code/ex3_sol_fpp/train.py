"""Labels + distillation for Example 3 (torch; workstation).

One learned variant, 'mem': c_n = (y_n, z_n) with the data-selected EMA
bank and the predictive-rank coordinates (config.REDUCE), through the
identical pipeline of memdiff.pipeline (training-free sampler -> labels
-> distilled flow-map network).

    python train.py [--smoke]        -> out/mem/model.pt  (smoke: out/smoke/)
"""

import sys

import numpy as np

import config as cfg
from memdiff.distill import FlowMapNet          # noqa: F401 (re-export)
from memdiff.pipeline import BaseExample
from conditioning import (rates_from_band, build_tuples,
                             select_rates_by_prediction)


class SOLExample(BaseExample):

    variants = ("mem",)
    latent_dim = 1

    def prepare_conditioning(self, name, X_train):
        assert name.startswith("mem")
        if cfg.K_BANK is None:
            sel = select_rates_by_prediction(X_train)
            if sel["acf_err_chosen"] > cfg.SEL_ACF_TOL:
                sys.exit(f"NO-GO: stage B fell back (closed-loop ACF error "
                         f"{sel['acf_err_chosen']:.4f} > {cfg.SEL_ACF_TOL}).")
            k, rates = sel["k"], sel["rates"]
        else:
            sel = select_rates_by_prediction(X_train)
            k, rates = cfg.K_BANK, rates_from_band(sel["band"], cfg.K_BANK)
        print(f"  k={k}, band {sel['band']}, rates {np.round(rates, 4)}")
        tup = build_tuples(X_train, rates, kappa_s=cfg.KAPPA_S)
        if tup["proj"] is not None:
            spec = np.round(tup["proj"]["spectrum"], 3).tolist()
            print(f"  predictive spectrum {spec} -> p = {tup['proj']['p']} "
                  f"(conditioning dim {tup['C'].shape[1]})")
        sel = dict(sel)
        sel["proj"] = tup["proj"]
        sel["reduce"] = bool(cfg.REDUCE)
        return tup, sel["band"], rates, sel


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        cfg.apply_smoke()                 # OUT_DIR -> out/smoke
    SOLExample(cfg).main(["mem"])
