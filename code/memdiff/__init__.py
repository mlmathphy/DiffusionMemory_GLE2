"""Shared core of the numerical examples.

Everything here is example-agnostic and config-free: parameters are
passed in explicitly (usually from the example's config module via the
BaseExample pipeline class). Layout:

  features.py -- numpy primitives: ACF/MSD estimators, rate-band
                 construction, and the EMA memory bank (any state dim)

  sampler.py  -- the training-free conditional diffusion sampler (torch)
  sampler_np.py -- its NumPy mirror for torch-free design scans/pre-vets
  distill.py  -- the flow-map MLP and its supervised distillation (torch)
  pipeline.py -- BaseExample: the label-generation + distillation
                 template each example subclasses
"""

# Keep the example folder importable no matter which module is imported
# first: scripts run from inside code/<example>/ and rely on code/ being
# on sys.path (config.py does this too, but import order must not matter).
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
