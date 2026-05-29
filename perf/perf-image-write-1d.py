#!/usr/bin/env python
"""
Uncompressed 1-D image WRITE throughput: rustfits vs fitsio.

The uncompressed analog of the healsparse compressed write -- no gzip
level, no quantize, just byteswap + write raw f8 bytes.  Input built
ONCE (untimed); each timed iteration overwrites a scratch file; no
fsync; warmup primes.  Correctness gate: each tool's write round-trips
bit-exact.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-image-write-1d.py
    python perf/perf-image-write-1d.py --n 134217728

Scratch file goes to CWD as perf-tmp-* and is removed on exit.
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _harness as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=h.env_int("PERF_N", 64_000_000))
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    n = args.n
    fname = h.path("wimg1d.fits")
    # Content is irrelevant for a raw write; use plain noise.
    data = np.random.default_rng(0).standard_normal(n)

    def rf_write():
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(data)

    def fi_write():
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(data)

    with h.scratch():
        rf_write()
        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[0].read(), data)
        fi_write()
        with fitsio.FITS(fname) as f:
            np.testing.assert_array_equal(f[0].read(), data)

        result = h.bench(
            f"write raw ({n * 8 / 1e6:,.0f} MB f8)",
            rf_write,
            run_fitsio=fi_write,
            nbytes=n * 8,
            repeat=args.repeat,
        )
        h.print_env()
        h.report(f"Uncompressed 1-D image WRITE (f8, {n:,})", [result])


if __name__ == "__main__":
    main()
