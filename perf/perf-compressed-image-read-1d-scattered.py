#!/usr/bin/env python
"""
Compressed 1-D image SCATTERED-access read: rustfits vs fitsio
across two tile-cache regimes.

Different from the existing ``perf-compressed-image-read-*.py``
benches: they sweep CHUNK SIZE (one-pass walk in storage order),
this one runs N=1000 RANDOM windows against the same file under
two different cache configurations -- the two regimes a real
caller actually has to choose between.

Setup: 64-tile (1 MiB / tile, 8 MB / tile) 1-D ``f8`` GZIP_2
image of healsparse-like run-structured data (~537 MB raw,
~15 MB compressed).  Read 1000 random windows of 1000 rows each;
with random sampling across 64 tiles, every tile is touched
~16x on average -- so cache misses dominate when the cache is
small, and cache hits dominate when it's large.

The two regimes:

* **default cache**.  rustfits at the shipped default
  (32 MB / 4 tiles); fitsio at its own default (unbounded;
  caches every tile it has ever decoded forever).  Even on
  this small file rustfits's bounded cache thrashes -- it
  holds 4 of 64 tiles so every random read evicts something
  recently used -- and the workload becomes decode-bound.
  fitsio's unbounded cache implicitly covers the full file,
  so it stops decoding after the first ~64 unique-tile reads
  and looks up the rest from RAM.  FITSIO WINS THIS ROW.

* **large cache**.  rustfits sized to hold all 64 unique
  tiles (~528 MB), matching fitsio's unbounded coverage on
  this file.  Both backends now decode each tile once per
  iteration; the residual gap is per-read lookup-and-slice
  cost, where rustfits's leaner Python boundary wins.

.. important::

   The fitsio default looks great here because the file is
   small enough that "cache every tile forever" fits in RAM.
   **For a multi-GB compressed image, fitsio's unbounded cache
   will OOM** -- there is no knob to bound it.  rustfits's
   bounded LRU degrades gracefully: above the cap, the per-
   tile decode cost replaces the cache hit, but the process
   keeps running.  This bench does NOT induce the OOM (it's
   too dependent on the machine's available RAM to reproduce
   reliably); the trade-off is documented here so users pick
   the cache size for their actual workload, not the small-
   file default.

Practical takeaway:

* If the access pattern is scattered AND the file fits in
  RAM AND you're not worried about RSS, fitsio's unbounded
  cache and rustfits with a comparably large
  ``set_tile_cache_size`` are both fast (rustfits a bit
  faster on per-read cost).
* If the file is too big to fit in RAM, you can't use the
  unbounded-cache pattern at all.  rustfits's bounded cache
  + decode cost per miss is the only option; tune
  ``set_tile_cache_size`` against the locality of YOUR
  access pattern (cluster of nearby reads -> bigger cache
  helps a lot; uniform-random reads on a huge file -> cache
  helps modestly, decode cost dominates).

Methodology mirrors the other compressed-read benches:

* Release build required (``maturin develop --release``).
* Fresh open per timed iteration: each iter starts with an
  empty tile cache so the steady-state hit rate is what we
  measure.  Within one iter, both tools populate their cache
  during the first ~64 reads, then the rest are hits (large-
  cache regime) or continued misses (default-cache regime).
* Warmup primes the FS cache so timed iters read the
  compressed bytes from RAM (decoder speed, not disk).
* Median of ``--repeat`` (default 5) runs.

Run::

    python perf/perf-compressed-image-read-1d-scattered.py
    python perf/perf-compressed-image-read-1d-scattered.py --ntiles 32
    python perf/perf-compressed-image-read-1d-scattered.py \\
        --n-chunks 5000 --chunk 100

Scratch files go to CWD as perf-tmp-* and are removed on exit
(``PERF_KEEP=1`` keeps them); ``--file`` points at an existing
``.fits.fz``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rustfits

import _data
import _harness as h


# rustfits's shipped default tile cache size.  Kept in sync with
# `DEFAULT_TILE_CACHE_BYTES` in src/zimage/tile_io.rs (32 MiB).
RUSTFITS_DEFAULT_CACHE = 32 << 20


def generate(fname, ntiles, tile, run_len, cov, quant):
    """Same healsparse-like array as the chunked-read bench."""
    n = ntiles * tile
    data = _data.healsparse_array(n, run_len, cov, quant)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Gzip2(tile_shape=(tile,), heap_format="Q"),
        )
    return n


def reader(mod, fname, ranges, cache_bytes=None):
    """
    Return a callable that opens fname, optionally sets rustfits's
    tile cache size, then issues every (lo, hi) in ``ranges`` as a
    single ``hdu[lo:hi]`` slice.

    ``cache_bytes`` only meaningful for rustfits.  fitsio's tile
    cache is unbounded and not user-configurable.
    """

    def run():
        with mod.FITS(fname) as f:
            hdu = f[1]
            if mod is rustfits and cache_bytes is not None:
                hdu.set_tile_cache_size(cache_bytes)
            for lo, hi in ranges:
                hdu[lo:hi]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        help="read this existing .fits.fz instead of generating one",
    )
    ap.add_argument("--ntiles", type=int, default=h.env_int("PERF_NTILES", 64))
    ap.add_argument(
        "--tile", type=int, default=1048576, help="elements per tile"
    )
    ap.add_argument(
        "--n-chunks",
        type=int,
        default=1000,
        help="random windows per iteration",
    )
    ap.add_argument(
        "--chunk", type=int, default=1000, help="window size (rows)"
    )
    ap.add_argument("--run-len", type=int, default=32)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--quant", type=int, default=1)
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
            fname = h.path("comp1dscatter.fits.fz")
            tile = args.tile
            n = generate(
                fname,
                args.ntiles,
                tile,
                args.run_len,
                args.cov,
                args.quant,
            )
            disk = os.path.getsize(fname)
            ratio = (n * 8) / disk
            print(
                f"generated: n={n:,} ({n * 8 / 1e6:,.0f} MB raw), "
                f"tile={tile:,}, {disk / 1e6:,.0f} MB on disk "
                f"({ratio:.1f}x)"
            )

        # Build the seeded random read plan.  Windows may overlap;
        # that's representative of scattered access.
        rng = np.random.default_rng(0)
        starts = rng.integers(0, n - args.chunk, size=args.n_chunks).tolist()
        ranges = [(int(s), int(s) + args.chunk) for s in starts]

        # Correctness gate: a handful of slices agree on both
        # backends before any timing.
        with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
            for lo, hi in ranges[:5]:
                np.testing.assert_array_equal(rf[1][lo:hi], fi[1][lo:hi])

        # Count unique tiles touched so the "large cache" sizing is
        # workload-fitted, not arbitrary.
        tile_set = set()
        for lo, hi in ranges:
            for t in range(lo // tile, (hi - 1) // tile + 1):
                tile_set.add(t)
        n_unique_tiles = len(tile_set)

        nbytes_per_iter = args.n_chunks * args.chunk * 8
        tile_bytes = tile * 8  # one tile of f8
        # 16 MB margin so an LRU eviction race doesn't drop a tile
        # we still need.
        large_cache = n_unique_tiles * tile_bytes + (16 << 20)

        print(
            f"\nN_chunks={args.n_chunks:,}, chunk={args.chunk:,}, "
            f"tile={tile:,}, unique tiles touched={n_unique_tiles} "
            f"of {args.ntiles}"
        )
        print(
            f"rustfits default cache = {RUSTFITS_DEFAULT_CACHE >> 20} MB "
            f"({RUSTFITS_DEFAULT_CACHE // tile_bytes} tiles)"
        )
        print(
            f"large cache for this workload = {large_cache >> 20} MB "
            f"({n_unique_tiles} tiles + 16 MB margin)"
        )
        print(
            "fitsio cache is unbounded — always behaves like the "
            "large-cache regime as long as it doesn't OOM (see the "
            "script docstring for the OOM caveat on big files)"
        )
        h.print_env()

        results = [
            h.bench(
                f"default cache (rf={RUSTFITS_DEFAULT_CACHE >> 20} MB, "
                "fi=unbounded)",
                reader(
                    rustfits,
                    fname,
                    ranges,
                    cache_bytes=RUSTFITS_DEFAULT_CACHE,
                ),
                run_fitsio=reader(fitsio, fname, ranges),
                nbytes=nbytes_per_iter,
                repeat=args.repeat,
                note=f"rf cache holds "
                f"{RUSTFITS_DEFAULT_CACHE // tile_bytes}/"
                f"{n_unique_tiles} tiles -- thrashes",
            ),
            h.bench(
                f"large cache (rf={large_cache >> 20} MB, fi=unbounded)",
                reader(
                    rustfits,
                    fname,
                    ranges,
                    cache_bytes=large_cache,
                ),
                run_fitsio=reader(fitsio, fname, ranges),
                nbytes=nbytes_per_iter,
                repeat=args.repeat,
                note="both backends cache all touched tiles",
            ),
        ]

        h.report(
            "Compressed 1-D image scattered read (GZIP_2 f8, "
            "healsparse-like) — two cache regimes",
            results,
        )


if __name__ == "__main__":
    main()
