#!/usr/bin/env python
"""
2-D lossy compressed-image read throughput: rustfits vs fitsio (DES-like).

Modeled on Dark Energy Survey image compression: a 2-D float32 image
tile-compressed with RICE_1 and lossy quantization (~q=16, "four bits to
the noise", ~5x compression) using SUBTRACTIVE_DITHER_2 so exact-zero
(masked) pixels are preserved bit-for-bit.  Default is 4000x4000 f4
noise with a small fraction of masked zeros -- big enough for a clean,
decisive result; scale up with --rows/--cols if you want to revisit.

This complements the 1-D lossless healsparse benchmark: 2-D access, a
lossy codec, and a square tiling so stamps touch a 2-D block of tiles.

Two read regimes:

* stamps 32x32 (random) -- ~1000 randomly-positioned postage stamps, the
                           real DES access pattern (cutouts around detected
                           objects).  rustfits's tile cache is sized to hold
                           the whole image so each tile decodes once for
                           BOTH tools (fitsio's cache is effectively
                           unbounded) -- a cache-neutral, decode-bound
                           comparison, not a cache-size one.
* whole file, in order  -- walk every tile-row band in storage order, each
                           tile decoded once; bounded memory, so it scales
                           to GB files (unlike a one-shot .read()).

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
    python perf/perf-compressed-image-read-des.py --rows 10000 --cols 10000
    python perf/perf-compressed-image-read-des.py --file /path/to/img.fz

Generated scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rustfits

import _data
import _harness as h
import _read2d as r2


def generate(fname, rows, cols, tile, q, seed, zero_frac):
    """
    Write a 2-D f4 RICE+dither2 lossy image of noise with a fraction
    ``zero_frac`` of exact-zero (masked) pixels (see _data.des_array).
    """
    data = _data.des_array(rows, cols, zero_frac)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Rice1(tile_shape=(tile, tile)),
            quantize=rustfits.Quantize(level=q, method="dither2", seed=seed),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 4000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 4000))
    ap.add_argument("--tile", type=int, default=100, help="square tile dim")
    ap.add_argument("--q", type=float, default=16.0, help="quantize level")
    ap.add_argument("--seed", type=int, default=1, help="dither seed")
    ap.add_argument("--zero-frac", type=float, default=0.05)
    ap.add_argument("--stamp", type=int, default=32, help="stamp size NxN")
    ap.add_argument("--stamps", type=int, default=1000, help="num stamps")
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

        box = min(args.stamp, rows, cols)

        # Correctness gate: both readers agree off the same file, and a
        # masked zero survives exactly (the dither2 guarantee).
        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            np.testing.assert_array_equal(rf[1][:box, :box], fi[1][:box, :box])
            if not args.file:
                assert rf[1][0, 0] == 0.0, "masked zero not preserved"

        # rustfits's tile cache is sized to hold the whole image so the
        # stamp regime is cache-neutral (each tile decodes once for both
        # tools); see _read2d.
        cache = r2.cache_bytes(rows, cols)
        positions = r2.stamp_positions(rows, cols, box, args.stamps)
        bands = r2.bands_for(rows, tile)
        full_bytes = rows * cols * 4

        results = [
            h.bench(
                f"stamps {box}x{box} random (x{len(positions)})",
                r2.read_stamps(rustfits, fname, positions, box, cache),
                run_fitsio=r2.read_stamps(
                    fitsio, fname, positions, box, cache
                ),
                nbytes=len(positions) * box * box * 4,
                repeat=args.repeat,
            ),
            h.bench(
                f"whole file, tiles in order ({len(bands)} bands)",
                r2.read_band_walk(rustfits, fname, bands),
                run_fitsio=r2.read_band_walk(fitsio, fname, bands),
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
