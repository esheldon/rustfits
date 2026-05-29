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


# A deliberately type-exhaustive BINTABLE schema (>=30 columns): every
# major scalar type, f4/f8 fixed sub-arrays (1-D and 2-D, <=6 per dim),
# a couple of VLA columns (string + f4), and extra scalars to pad past
# 30.  Shared so the uncompressed and compressed table benchmarks use
# the same schema.
_SCHEMA = [
    ("c_b1", "?"),
    ("c_u1", "u1"),
    ("c_u2", "u2"),
    ("c_u4", "u4"),
    ("c_u8", "u8"),
    ("c_i2", "i2"),
    ("c_i4", "i4"),
    ("c_i8", "i8"),
    ("c_f4", "f4"),
    ("c_f8", "f8"),
    ("c_c8", "c8"),
    ("c_c16", "c16"),
    ("c_str", "S16"),
    ("c_ustr", "U16"),
    ("a_f4_3", "f4", (3,)),
    ("a_f4_6", "f4", (6,)),
    ("a_f8_4", "f8", (4,)),
    ("a_f4_2x3", "f4", (2, 3)),
    ("a_f4_6x6", "f4", (6, 6)),
    ("a_f8_3x3", "f8", (3, 3)),
    ("a_f8_5x2", "f8", (5, 2)),
    ("x_f4_0", "f4"),
    ("x_f8_0", "f8"),
    ("x_i4_0", "i4"),
    ("x_i8_0", "i8"),
    ("x_f4_1", "f4"),
    ("x_f8_1", "f8"),
    ("x_i2_0", "i2"),
    ("x_u4_0", "u4"),
    ("x_f8_2", "f8"),
    ("x_f4_2", "f4"),
    ("v_str", "O"),
    ("v_ustr", "O"),
    ("v_f4", "O"),
]
VAR_DTYPES = {"v_str": "S", "v_ustr": "U", "v_f4": "f4"}


def catalog_dtype():
    """The structured dtype for the type-exhaustive catalog schema."""
    fields = []
    for col in _SCHEMA:
        if len(col) == 3:
            fields.append((col[0], col[1], col[2]))
        else:
            fields.append((col[0], col[1]))
    return np.dtype(fields)


def _fill_vla(data, name, rng, lo, hi, kind):
    n = len(data)
    lens = rng.integers(lo, hi, size=n)
    off = np.concatenate(([0], np.cumsum(lens)))
    total = int(off[-1])
    out = np.empty(n, dtype=object)
    if kind in ("s", "u"):
        pool = rng.integers(97, 123, size=total, dtype=np.uint8).tobytes()
        for i in range(n):
            cell = pool[off[i] : off[i + 1]]
            out[i] = cell.decode() if kind == "u" else cell
    else:
        pool = rng.standard_normal(total).astype("f4")
        for i in range(n):
            out[i] = pool[off[i] : off[i + 1]]
    data[name] = out


def catalog_arrays(nrows, seed=0):
    """
    Build a ``nrows``-row structured array for the catalog schema plus
    the ``var_dtypes`` sidecar for the two VLA columns.  Integer/float
    content is random (irrelevant to read speed); VLA cells are
    variable-length (strings 5-20 bytes, f4 arrays 1-10 elements).
    """
    rng = np.random.default_rng(seed)
    dt = catalog_dtype()
    data = np.empty(nrows, dtype=dt)
    for name in dt.names:
        if name in VAR_DTYPES:
            continue
        base = dt[name].base
        shape = (nrows,) + dt[name].shape
        k = base.kind
        if k == "b":
            data[name] = rng.integers(0, 2, size=shape).astype(bool)
        elif k in ("i", "u"):
            data[name] = rng.integers(0, 1000, size=shape).astype(base)
        elif k == "f":
            data[name] = rng.standard_normal(shape).astype(base)
        elif k == "c":
            re = rng.standard_normal(shape)
            im = rng.standard_normal(shape)
            data[name] = (re + 1j * im).astype(base)
        elif k == "S":
            w = base.itemsize
            sb = rng.integers(65, 91, size=(nrows, w), dtype=np.uint8)
            data[name] = np.ascontiguousarray(sb).view(f"S{w}").reshape(nrows)
        elif k == "U":
            w = base.itemsize // 4
            sb = rng.integers(65, 91, size=(nrows, w), dtype=np.uint8)
            s = np.ascontiguousarray(sb).view(f"S{w}").reshape(nrows)
            data[name] = np.char.decode(s)
    _fill_vla(data, "v_str", rng, 5, 21, "s")
    _fill_vla(data, "v_ustr", rng, 5, 21, "u")
    _fill_vla(data, "v_f4", rng, 1, 11, "f")
    return data, dict(VAR_DTYPES)
