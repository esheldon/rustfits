#!/usr/bin/env python
"""
Uncompressed 2-D image read throughput: rustfits vs fitsio.

The uncompressed analog of the DES compressed-read benchmark, minus the
compression-specific parts (no tiles, no quantize/dither, no tile cache).
Two regimes:

* stamps 32x32 (random) -- ~1000 randomly-positioned postage stamps, the
                           object-cutout access pattern.
* whole .read()         -- one full read; bulk throughput.

Data content is irrelevant to an uncompressed read; we use plain f4
noise.  Fresh open per timed iteration; warmup primes the FS cache.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-image-read-2d.py
    python perf/perf-image-read-2d.py --rows 10000 --cols 10000

Scratch file goes to CWD as perf-tmp-* and is removed on exit.
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _harness as h
import _read2d as r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 4000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 4000))
    ap.add_argument("--stamp", type=int, default=32)
    ap.add_argument("--stamps", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    rows, cols, box = args.rows, args.cols, args.stamp
    fname = h.path("img2d.fits")

    with h.scratch():
        data = np.random.default_rng(0).standard_normal(
            (rows, cols), dtype=np.float32
        )
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(data)

        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            np.testing.assert_array_equal(rf[0][:200, :200], fi[0][:200, :200])

        def read_stamps(mod, positions):
            def run():
                with mod.FITS(fname) as f:
                    hdu = f[0]
                    for r, c in positions:
                        hdu[r : r + box, c : c + box]

            return run

        def read_whole(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[0].read()

            return run

        positions = r2.stamp_positions(rows, cols, box, args.stamps)
        full_bytes = rows * cols * 4

        results = [
            h.bench(
                f"stamps {box}x{box} random (x{len(positions)})",
                read_stamps(rustfits, positions),
                run_fitsio=read_stamps(fitsio, positions),
                nbytes=len(positions) * box * box * 4,
                repeat=args.repeat,
            ),
            h.bench(
                "whole .read()",
                read_whole(rustfits),
                run_fitsio=read_whole(fitsio),
                nbytes=full_bytes,
                repeat=args.repeat,
            ),
        ]

        h.print_env()
        h.report(f"Uncompressed 2-D image read (f4, {rows}x{cols})", results)


if __name__ == "__main__":
    main()
