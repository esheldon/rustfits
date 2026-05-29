#!/usr/bin/env python
"""
Compressed 1-D image chunked-read throughput: rustfits vs fitsio.

Modeled on a real healsparse-style map: a 1-D float64 GZIP_2 image with
1,048,576-element tiles (8 MB/tile) built from long runs of repeated,
quantized values -- mostly a sentinel (uncovered sky) with covered runs
carrying real values.  The run length is tuned so the rustfits-vs-fitsio
timing RATIOS reproduce the real file (~42x small / ~3.5x big), which
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

import _harness as h

# healpy's UNSEEN sentinel: most healsparse pixels carry it, which is
# what makes these maps compress so well.
SENTINEL = -1.6375e30
DEFAULT_TILE = 1048576


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


def whole_ranges(n, chunk):
    return [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]


def partial_ranges(n, chunk, cover):
    cover = min(n, cover)
    return [(lo, min(lo + chunk, cover)) for lo in range(0, cover, chunk)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        help="read this existing .fits.fz instead of generating one",
    )
    ap.add_argument("--ntiles", type=int, default=h.env_int("PERF_NTILES", 64))
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument(
        "--run-len", type=int, default=32, help="length of each value run"
    )
    ap.add_argument(
        "--cov", type=float, default=0.5, help="fraction of real runs"
    )
    ap.add_argument(
        "--quant", type=int, default=1, help="decimals to round real values"
    )
    ap.add_argument(
        "--small-chunk", type=int, default=1000, help="sub-tile, partial"
    )
    ap.add_argument(
        "--mid-chunk", type=int, default=50000, help="sub-tile, whole"
    )
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    with h.scratch():
        if args.file:
            fname = args.file
            with rustfits.FITS(fname) as f:
                hdu = f[1]
                n = int(hdu.shape[0])
                tile = int(hdu.compression.tile_shape[0])
            print(f"file: {fname} (n={n:,}, tile={tile:,})")
        else:
            fname = h.path("comp1d.fits.fz")
            tile = args.tile
            n = generate(
                fname, args.ntiles, tile, args.run_len, args.cov, args.quant
            )
            import os

            ratio = (n * 8) / os.path.getsize(fname)
            print(
                f"generated: n={n:,} ({n * 8 / 1e6:,.0f} MB raw), "
                f"tile={tile:,}, {os.path.getsize(fname) / 1e6:,.0f} MB "
                f"on disk ({ratio:.1f}x)"
            )

        # Correctness gate across a tile boundary before any timing.
        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            lo, hi = tile - 500, tile + 500
            np.testing.assert_array_equal(rf[1][lo:hi], fi[1][lo:hi])

        # We OPEN A FRESH HANDLE inside every timed iteration so each
        # tool starts with an empty tile cache and genuinely decodes.
        # Timing repeated reads of one open handle would instead measure
        # fitsio's unbounded "cache every tile forever" hits against
        # rustfits's bounded LRU re-decoding -- backwards from the real
        # workload, which reads each tile once.  The harness's warmup
        # pass primes the OS page (FS) cache first, so the timed passes
        # read the compressed bytes from RAM and we measure decode-code
        # speed, not the (I/O-bound, near-identical) cold first read.
        def reader(mod, ranges):
            def run():
                with mod.FITS(fname) as f:
                    hdu = f[1]
                    for lo, hi in ranges:
                        hdu[lo:hi]

            return run

        # cover ~8 tiles for the small partial regime
        small_cover = min(n, 8 * tile)
        regimes = [
            (
                f"chunk {args.small_chunk} (partial, {small_cover // tile} "
                f"tiles)",
                partial_ranges(n, args.small_chunk, small_cover),
            ),
            (
                f"chunk {args.mid_chunk} (whole array)",
                whole_ranges(n, args.mid_chunk),
            ),
            (
                f"chunk {tile} = tile (whole array)",
                whole_ranges(n, tile),
            ),
        ]

        results = []
        for label, ranges in regimes:
            nbytes = sum(hi - lo for lo, hi in ranges) * 8
            results.append(
                h.bench(
                    label,
                    reader(rustfits, ranges),
                    run_fitsio=reader(fitsio, ranges),
                    nbytes=nbytes,
                    repeat=args.repeat,
                )
            )

        h.print_env()
        h.report("Compressed 1-D image read (GZIP_2 f8)", results)


if __name__ == "__main__":
    main()
