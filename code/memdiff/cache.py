"""Fingerprinted array cache for expensive deterministic stages.

Rollouts are deterministic given (checkpoint, initial condition, seed),
so recomputing them on every evaluate.py rerun is wasted work --- but a
cache keyed only by a filename would silently serve stale arrays after
retraining, which is far worse than the wasted time. Every entry
therefore stores a fingerprint of everything it depends on and is
recomputed whenever that fingerprint changes.

    X = cached(path, fingerprint_parts, compute_fn)
"""

import hashlib
import os

import numpy as np


def fingerprint(*parts):
    """Stable hex digest of arrays, numbers, strings, and dicts."""
    h = hashlib.sha1()

    def feed(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                feed(v)
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        elif hasattr(x, "detach"):          # torch tensor
            h.update(x.detach().cpu().numpy().tobytes())
        else:
            h.update(repr(x).encode())

    feed(parts)
    return h.hexdigest()


def state_fingerprint(state_dict):
    """Digest of a torch state_dict (identifies the trained checkpoint)."""
    return fingerprint({k: v for k, v in state_dict.items()})


def cached(path, key, compute, verbose=True):
    """Return compute() , memoized in `path` under fingerprint `key`.

    The stored file keeps the key alongside the array; a mismatch (new
    checkpoint, changed seed or horizon) triggers recomputation, so a
    stale cache can never be served silently.
    """
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    if verbose:
                        print(f"    cache hit  {os.path.basename(path)}")
                    return z["value"]
                if verbose:
                    print(f"    cache stale {os.path.basename(path)}"
                          " (fingerprint changed) -- recomputing")
        except Exception:                    # corrupt/partial file
            if verbose:
                print(f"    cache unreadable {os.path.basename(path)}"
                      " -- recomputing")
    value = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"                  # atomic: no partial caches
    np.savez(tmp, key=np.array(key), value=value)
    os.replace(tmp, path)
    return value
