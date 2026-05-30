#!/usr/bin/env python
"""
Uncompressed 2-D image WRITE throughput: rustfits vs fitsio.

The uncompressed analog of the DES compressed write -- no quantize, just
byteswap + write raw f4 bytes.  Input built ONCE (untimed); each timed
iteration overwrites a scratch file; no fsync; warmup primes.
Correctness gate: each tool's write round-trips bit-exact.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-image-write-2d.py
    python perf/perf-image-write-2d.py --rows 10000 --cols 10000

Scratch file goes to CWD as perf-tmp-* and is removed on exit.
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _harness as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 4000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 4000))
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    rows, cols = args.rows, args.cols
    data = np.random.default_rng(0).standard_normal(
        (rows, cols), dtype=np.float32
    )

    # Fresh fname per iter: see h.fresh_path docstring.
    def rf_write():
        fname = h.fresh_path("wimg2d-rf")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(data)

    def fi_write():
        fname = h.fresh_path("wimg2d-fi")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(data)

    with h.scratch():
        gate_rf = h.fresh_path("wimg2d-gate-rf")
        gate_fi = h.fresh_path("wimg2d-gate-fi")
        with rustfits.FITS(gate_rf, "w+") as f:
            f.write_image(data)
        with rustfits.FITS(gate_rf) as f:
            np.testing.assert_array_equal(f[0].read(), data)
        with fitsio.FITS(gate_fi, "rw", clobber=True) as f:
            f.write(data)
        with fitsio.FITS(gate_fi) as f:
            np.testing.assert_array_equal(f[0].read(), data)

        raw = rows * cols * 4
        result = h.bench(
            f"write raw ({raw / 1e6:,.0f} MB f4)",
            rf_write,
            run_fitsio=fi_write,
            nbytes=raw,
            repeat=args.repeat,
        )
        h.print_env()
        h.report(f"Uncompressed 2-D image WRITE (f4, {rows}x{cols})", [result])


if __name__ == "__main__":
    main()
