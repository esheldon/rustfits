#!/usr/bin/env python
"""
Isolation sweep for the 2-D lossy-RICE read slowdown.

The DES-like benchmark (perf-compressed-image-read-des.py) found rustfits
slower than fitsio on 2-D RICE+quantized reads -- the opposite of the 1-D
GZIP_2 result.  That workload changed four things at once vs the 1-D test:
codec (GZIP_2 -> RICE), lossy dequant, dimensionality, and tile size.
This sweep holds the 2-D shape fixed (4000x4000) and varies ONE factor per
case, measuring BOTH access regimes (random 32x32 stamps and a whole-file
tile-order band walk) so the gap can be decomposed:

  gzip2 f4 unquantized      -- 2-D GZIP baseline (should be rustfits-fast,
                               like the 1-D test)
  rice  i4 lossless         -- RICE decode alone (no dequant, no GZIP)
  gzip2 f4 quantized        -- dequant cost on top of GZIP
  rice  f4 quantized (DES)  -- the slow case
  rice  f4 quantized t=1000 -- same, larger tiles (per-tile overhead)

Decomposition:
  (rice-quant vs gzip2-quant)      -> RICE-vs-GZIP decode
  (gzip2-quant vs gzip2-unquant)   -> dequant cost
  (rice-quant t=100 vs t=1000)     -> small-tile / per-tile overhead
  (stamps vs whole within a case)  -> is the gap access-pattern-dependent

Same methodology as the other perf scripts: release build, fresh open per
timed iteration, warmup primes the FS cache.  Stamps are cache-neutral
(rustfits's cache is sized to hold the image; see _read2d).

Run::

    python perf/perf-compressed-2d-isolation.py
    python perf/perf-compressed-2d-isolation.py --rows 2000 --cols 2000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rustfits

import _harness as h
import _read2d as r2


def _noise_f4(rows, cols):
    return np.random.default_rng(0).standard_normal(
        (rows, cols), dtype=np.float32
    )


def gen_gzip2_unquant(fname, rows, cols, tile):
    data = _noise_f4(rows, cols)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(data, compress=rustfits.Gzip2(tile_shape=(tile, tile)))


def gen_rice_int(fname, rows, cols, tile):
    data = (_noise_f4(rows, cols) * 16).astype("i4")
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(data, compress=rustfits.Rice1(tile_shape=(tile, tile)))


def gen_gzip2_quant(fname, rows, cols, tile):
    data = _noise_f4(rows, cols)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Gzip2(tile_shape=(tile, tile)),
            quantize=rustfits.Quantize(level=16, method="dither2", seed=1),
        )


def gen_rice_quant(fname, rows, cols, tile):
    data = _noise_f4(rows, cols)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Rice1(tile_shape=(tile, tile)),
            quantize=rustfits.Quantize(level=16, method="dither2", seed=1),
        )


CASES = [
    ("gzip2 f4 unquant", gen_gzip2_unquant, 100),
    ("rice i4 lossless", gen_rice_int, 100),
    ("gzip2 f4 quant", gen_gzip2_quant, 100),
    ("rice f4 quant (DES)", gen_rice_quant, 100),
    ("rice f4 quant t=1000", gen_rice_quant, 1000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 4000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 4000))
    ap.add_argument("--stamp", type=int, default=32)
    ap.add_argument("--stamps", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    rows, cols = args.rows, args.cols
    box = args.stamp
    fname = h.path("iso2d.fits.fz")
    cache = r2.cache_bytes(rows, cols)
    positions = r2.stamp_positions(rows, cols, box, args.stamps)
    stamp_bytes = len(positions) * box * box * 4
    full_bytes = rows * cols * 4

    with h.scratch():
        results = []
        for label, gen, tile in CASES:
            gen(fname, rows, cols, tile)
            ratio = full_bytes / os.path.getsize(fname)
            with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
                np.testing.assert_array_equal(
                    rf[1][:200, :200], fi[1][:200, :200]
                )
            bands = r2.bands_for(rows, tile)
            results.append(
                h.bench(
                    f"{label} ({ratio:.1f}x) [stamps]",
                    r2.read_stamps(rustfits, fname, positions, box, cache),
                    run_fitsio=r2.read_stamps(
                        fitsio, fname, positions, box, cache
                    ),
                    nbytes=stamp_bytes,
                    repeat=args.repeat,
                )
            )
            results.append(
                h.bench(
                    f"{label} [whole]",
                    r2.read_band_walk(rustfits, fname, bands),
                    run_fitsio=r2.read_band_walk(fitsio, fname, bands),
                    nbytes=full_bytes,
                    repeat=args.repeat,
                )
            )
            os.remove(fname)

        h.print_env()
        h.report(
            f"2-D codec isolation ({rows}x{cols}): stamps + whole",
            results,
        )


if __name__ == "__main__":
    main()
