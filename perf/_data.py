"""
Shared synthetic data generators for the perf scripts.

These build the *arrays*; each script compresses/writes them however it
likes.  Keeping the data models here means a read benchmark and the
matching write benchmark exercise byte-identical data, so the two never
drift apart when a model is tuned.
"""

from __future__ import annotations

import numpy as np

# healpy's UNSEEN sentinel: most healsparse pixels carry it, which (with
# the run structure below) is what makes these maps compress so well.
SENTINEL = -1.6375e30


def healsparse_array(n, run_len, cov, quant, seed=0):
    """
    1-D f8 healsparse-like map: a sequence of fixed-length runs of one
    repeated value.  Most runs are the SENTINEL (uncovered sky); a
    fraction ``cov`` carry a real value quantized to ``quant`` decimals
    so values recur across runs.  ``run_len`` sets the per-tile decode
    work and is the knob tuned so timing ratios match the real file.
    """
    rng = np.random.default_rng(seed)
    nruns = (n + run_len - 1) // run_len
    is_real = rng.random(nruns) < cov
    vals = np.full(nruns, SENTINEL, dtype="f8")
    nreal = int(is_real.sum())
    if nreal:
        vals[is_real] = np.round(rng.standard_normal(nreal), quant)
    return np.repeat(vals, run_len)[:n]


def des_array(rows, cols, zero_frac, seed=0):
    """
    2-D f4 noise with a fraction ``zero_frac`` of exact-zero (masked)
    pixels; pixel [0, 0] is always zeroed as a known check point for the
    dither2 zero-preservation guarantee.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((rows, cols), dtype=np.float32)
    if zero_frac > 0:
        mask = rng.random((rows, cols), dtype=np.float32) < zero_frac
        data[mask] = 0.0
    data[0, 0] = 0.0
    return data
