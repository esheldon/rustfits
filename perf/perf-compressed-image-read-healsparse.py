#!/usr/bin/env python
"""
Compressed 1-D image chunked-read throughput: rustfits vs fitsio.

Modeled on a real healsparse-style map: a 1-D float64 GZIP_2 image with
1,048,576-element tiles (8 MB/tile) built from long runs of repeated,
quantized values -- mostly a sentinel (uncovered sky) with covered runs
carrying real values.  The run length is tuned so the rustfits-vs-fitsio
timing RATIOS reproduce the real file (~42x small / ~3.3x big), which
reflects the actual decode work better than matching the disk
compression ratio does.  We generate a smaller version by default to
keep the write fast; point --file at a real .fits.fz to measure that
instead (it reproduces the same headline ratios).

Three read regimes (per the workload that matters):

* chunk = 1000      -- sub-tile, read only PART of the array.  Decode is
                       cheap and cache-warm, so this is per-call-overhead
                       bound: where rustfits's meta cache wins biggest.
* chunk = 50000     -- sub-tile, read the WHOLE array so fixed overheads
                       wash out.
* chunk = tile size -- one tile decoded per read across the WHOLE array:
                       pure decoder throughput (the "decoder floor").

REQUIRES a release build: ``maturin develop --release``.  A debug build
reads ~7x slower and will report rustfits as the loser (see CLAUDE.md).

Run::

    python perf/perf-compressed-image-read-healsparse.py
    python perf/perf-compressed-image-read-healsparse.py --ntiles 128
    python perf/perf-compressed-image-read-healsparse.py \\
        --file /path/to/test.fits.fz --repeat 2

Generated scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).  A --file you pass in is never touched.
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _compread as cr
import _harness as h

# healpy's UNSEEN sentinel: most healsparse pixels carry it, which is
# what makes these maps compress so well.
SENTINEL = -1.6375e30


def generate(fname, ntiles, tile, run_len, cov, quant):
    """
    Write a 1-D f8 GZIP_2 image of ``ntiles`` tiles that mimics a real
    healsparse map: the array is a sequence of fixed-length runs of one
    repeated value.  Most runs are the SENTINEL (uncovered sky); a
    fraction ``cov`` of runs carry a real value, quantized to ``quant``
    decimals so the same values recur across many runs.  ``run_len`` is
    the main knob: it sets how much decode work each tile costs, and is
    tuned (default 32) so the rustfits-vs-fitsio timing ratios match the
    real file rather than to hit a target compression ratio.  Returns n.
    """
    n = ntiles * tile
    rng = np.random.default_rng(0)
    nruns = (n + run_len - 1) // run_len
    is_real = rng.random(nruns) < cov
    vals = np.full(nruns, SENTINEL, dtype="f8")
    nreal = int(is_real.sum())
    if nreal:
        vals[is_real] = np.round(rng.standard_normal(nreal), quant)
    data = np.repeat(vals, run_len)[:n]
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Gzip2(tile_shape=(tile,), heap_format="Q"),
        )
    return n


def main():
    ap = argparse.ArgumentParser()
    cr.add_common_args(ap)
    ap.add_argument(
        "--run-len", type=int, default=32, help="length of each value run"
    )
    ap.add_argument(
        "--cov", type=float, default=0.5, help="fraction of real runs"
    )
    ap.add_argument(
        "--quant", type=int, default=1, help="decimals to round real values"
    )
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    def gen(fname, ntiles, tile):
        return generate(
            fname, ntiles, tile, args.run_len, args.cov, args.quant
        )

    with h.scratch():
        fname, n, tile = cr.resolve_source(args, gen)
        cr.run_read_benchmark(
            fname,
            n,
            tile,
            title="Compressed 1-D image read (GZIP_2 f8, healsparse-like)",
            small_chunk=args.small_chunk,
            mid_chunk=args.mid_chunk,
            repeat=args.repeat,
        )


if __name__ == "__main__":
    main()
