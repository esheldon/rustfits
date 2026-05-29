#!/usr/bin/env python
"""
2-D lossy compressed-image read throughput: rustfits vs fitsio (DES-like).

Modeled on Dark Energy Survey image compression: a 2-D float32 image
tile-compressed with RICE_1 and lossy quantization (~q=16, "four bits to
the noise", ~5x compression) using SUBTRACTIVE_DITHER_2 so exact-zero
(masked) pixels are preserved bit-for-bit.  Default is 10000x10000 f4
noise with a small fraction of masked zeros.

This complements the 1-D lossless healsparse benchmark: 2-D access, a
lossy codec, and a square tiling so cutouts touch a 2-D block of tiles.

Three read regimes:

* cutout NxN (partial)  -- many small 2-D boxes at cycling positions;
                           per-call-overhead + few-tile bound.
* row strips (whole)    -- the whole image in horizontal strips, a common
                           streaming/processing pattern; decode-bound.
* whole image (.read()) -- one full decode; pure decoder throughput.

Quantized decode is deterministic, so rustfits and fitsio return the same
values off the same file -- the correctness gate checks this, plus that a
masked zero survives exactly (the dither2 contract).

REQUIRES a release build: ``maturin develop --release`` (a debug build
reports rustfits as the loser; see CLAUDE.md).  Methodology matches the
healsparse script: a fresh handle is opened inside every timed iteration
(so fitsio's forever-cache can't masquerade as decode speed), and the
warmup pass primes the FS cache so timing measures decode, not disk.

NOTE: fpack's own RICE default is row-by-row tiling; DES's exact tile
dims aren't pinned here.  --tile sets a square tile (default 100); the
load-bearing DES-like settings are the quantization level (--q) and the
zero-preserving dither2.

Run::

    python perf/perf-compressed-image-read-des.py
    python perf/perf-compressed-image-read-des.py --rows 4000 --cols 4000
    python perf/perf-compressed-image-read-des.py --file /path/to/img.fz

Generated scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rustfits

import _harness as h


def generate(fname, rows, cols, tile, q, seed, zero_frac):
    """
    Write a 2-D f4 RICE+dither2 lossy image of random noise with a
    fraction ``zero_frac`` of exact-zero (masked) pixels.  Pixel [0, 0]
    is always zeroed as a known point for the zero-preservation check.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((rows, cols), dtype=np.float32)
    if zero_frac > 0:
        mask = rng.random((rows, cols), dtype=np.float32) < zero_frac
        data[mask] = 0.0
    data[0, 0] = 0.0
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Rice1(tile_shape=(tile, tile)),
            quantize=rustfits.Quantize(level=q, method="dither2", seed=seed),
        )


def cutout_positions(rows, cols, box, count):
    """
    Cycle ``count`` top-left corners for box x box cutouts, marching
    across the image so repeated reads hit a spread of tiles.
    """
    rspan = max(1, rows - box)
    cspan = max(1, cols - box)
    return [((i * box) % rspan, (i * box * 7) % cspan) for i in range(count)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 10000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 10000))
    ap.add_argument("--tile", type=int, default=100, help="square tile dim")
    ap.add_argument("--q", type=float, default=16.0, help="quantize level")
    ap.add_argument("--seed", type=int, default=1, help="dither seed")
    ap.add_argument("--zero-frac", type=float, default=0.05)
    ap.add_argument("--cutout", type=int, default=256)
    ap.add_argument("--cutout-count", type=int, default=200)
    ap.add_argument("--strip", type=int, default=1000, help="rows per strip")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--file", help="read this existing .fz instead of generating one"
    )
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    with h.scratch():
        if args.file:
            fname = args.file
            with rustfits.FITS(fname) as f:
                rows, cols = (int(x) for x in f[1].shape[:2])
                tile = int(f[1].compression.tile_shape[0])
            print(f"file: {fname} ({rows}x{cols}, tile {tile})")
        else:
            fname = h.path("comp2d.fits.fz")
            rows, cols, tile = args.rows, args.cols, args.tile
            generate(
                fname, rows, cols, tile, args.q, args.seed, args.zero_frac
            )
            raw = rows * cols * 4
            ratio = raw / os.path.getsize(fname)
            print(
                f"generated: {rows}x{cols} f4 ({raw / 1e6:,.0f} MB raw), "
                f"tile {tile}x{tile}, q={args.q}, dither2, "
                f"{os.path.getsize(fname) / 1e6:,.0f} MB ({ratio:.1f}x)"
            )

        box = min(args.cutout, rows, cols)

        # Correctness gate: both readers agree off the same file, and a
        # masked zero survives exactly (the dither2 guarantee).
        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            np.testing.assert_array_equal(rf[1][:box, :box], fi[1][:box, :box])
            if not args.file:
                assert rf[1][0, 0] == 0.0, "masked zero not preserved"

        def read_cutouts(mod, positions, box):
            def run():
                with mod.FITS(fname) as f:
                    hdu = f[1]
                    for r, c in positions:
                        hdu[r : r + box, c : c + box]

            return run

        def read_strips(mod, ranges):
            def run():
                with mod.FITS(fname) as f:
                    hdu = f[1]
                    for lo, hi in ranges:
                        hdu[lo:hi, :]

            return run

        def read_whole(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read()

            return run

        positions = cutout_positions(rows, cols, box, args.cutout_count)
        strips = [
            (lo, min(lo + args.strip, rows))
            for lo in range(0, rows, args.strip)
        ]
        full_bytes = rows * cols * 4

        results = [
            h.bench(
                f"cutout {box}x{box} (x{len(positions)})",
                read_cutouts(rustfits, positions, box),
                run_fitsio=read_cutouts(fitsio, positions, box),
                nbytes=len(positions) * box * box * 4,
                repeat=args.repeat,
            ),
            h.bench(
                f"row strips ({args.strip} rows, whole)",
                read_strips(rustfits, strips),
                run_fitsio=read_strips(fitsio, strips),
                nbytes=full_bytes,
                repeat=args.repeat,
            ),
            h.bench(
                "whole image (.read())",
                read_whole(rustfits),
                run_fitsio=read_whole(fitsio),
                nbytes=full_bytes,
                repeat=args.repeat,
            ),
        ]

        h.print_env()
        h.report(
            f"2-D lossy compressed read (RICE q={args.q} dither2, f4)",
            results,
        )


if __name__ == "__main__":
    main()
