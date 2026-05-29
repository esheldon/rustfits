#!/usr/bin/env python
"""
Uncompressed 1-D image read throughput: rustfits vs fitsio.

The uncompressed analog of the healsparse compressed-read benchmark.
There are no tiles, so there's no "chunk = tile" regime, and the data
content is irrelevant to read speed (it's raw bytes) -- we just read a
plain 1-D f8 image in chunks.  Three regimes:

* chunk 1000 (partial) -- many small sub-reads; per-call-overhead bound.
* chunk 50000 (whole)  -- the whole array in mid-size chunks.
* whole .read()        -- one full read; bulk byteswap+copy throughput.

Fresh open per timed iteration; the harness warmup primes the FS cache
so timing measures the read path, not cold disk I/O.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-image-read-1d.py
    python perf/perf-image-read-1d.py --n 134217728

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
    ap.add_argument("--small-chunk", type=int, default=1000)
    ap.add_argument("--mid-chunk", type=int, default=50000)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    n = args.n
    fname = h.path("img1d.fits")

    with h.scratch():
        # Content is irrelevant for an uncompressed read; use plain noise.
        data = np.random.default_rng(0).standard_normal(n)
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(data)

        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            np.testing.assert_array_equal(rf[0][:1000], fi[0][:1000])

        def chunks(mod, ranges):
            def run():
                with mod.FITS(fname) as f:
                    hdu = f[0]
                    for lo, hi in ranges:
                        hdu[lo:hi]

            return run

        def whole(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[0].read()

            return run

        cover = min(n, 8_000_000)
        small = [
            (lo, min(lo + args.small_chunk, cover))
            for lo in range(0, cover, args.small_chunk)
        ]
        mid = [
            (lo, min(lo + args.mid_chunk, n))
            for lo in range(0, n, args.mid_chunk)
        ]

        results = [
            h.bench(
                f"chunk {args.small_chunk} (partial, {cover // 1_000_000}M)",
                chunks(rustfits, small),
                run_fitsio=chunks(fitsio, small),
                nbytes=cover * 8,
                repeat=args.repeat,
            ),
            h.bench(
                f"chunk {args.mid_chunk} (whole)",
                chunks(rustfits, mid),
                run_fitsio=chunks(fitsio, mid),
                nbytes=n * 8,
                repeat=args.repeat,
            ),
            h.bench(
                "whole .read()",
                whole(rustfits),
                run_fitsio=whole(fitsio),
                nbytes=n * 8,
                repeat=args.repeat,
            ),
        ]

        h.print_env()
        h.report(f"Uncompressed 1-D image read (f8, {n:,})", results)


if __name__ == "__main__":
    main()
