"""
Shared 2-D compressed-image read regimes for the perf scripts.

Two access patterns, used by both the DES benchmark and the codec
isolation sweep:

* stamps     -- many randomly-positioned NxN postage stamps (the real
                object-cutout pattern).  rustfits's tile cache is sized to
                hold the whole image so each tile decodes once for BOTH
                tools (fitsio caches forever); the comparison is then
                decode-bound, not cache-size-bound.
* band walk  -- every tile-row band read in storage order, each tile
                decoded once, bounded memory (scales to GB files, unlike a
                one-shot .read()).

Each reader OPENS A FRESH HANDLE per call so the harness's warmup primes
the FS cache while the timed passes start from an empty tile cache -- the
same methodology as the 1-D scripts.
"""

from __future__ import annotations

import numpy as np
import rustfits


def stamp_positions(rows, cols, box, count):
    """
    ``count`` random (seeded) top-left corners for box x box stamps.
    """
    rng = np.random.default_rng(1)
    rs = rng.integers(0, rows - box + 1, size=count)
    cs = rng.integers(0, cols - box + 1, size=count)
    return list(zip(rs.tolist(), cs.tolist()))


def cache_bytes(rows, cols, itemsize=4):
    """
    A tile-cache size that holds the whole image plus margin, so the
    stamp regime stays cache-neutral.
    """
    return rows * cols * itemsize + (16 << 20)


def bands_for(rows, tile):
    """
    Tile-row band ranges covering the image top to bottom.
    """
    return [(i, min(i + tile, rows)) for i in range(0, rows, tile)]


def read_stamps(mod, fname, positions, box, cache):
    def run():
        with mod.FITS(fname) as f:
            hdu = f[1]
            if mod is rustfits:
                hdu.set_tile_cache_size(cache)
            for r, c in positions:
                hdu[r : r + box, c : c + box]

    return run


def read_band_walk(mod, fname, bands):
    def run():
        with mod.FITS(fname) as f:
            hdu = f[1]
            for lo, hi in bands:
                hdu[lo:hi, :]

    return run
