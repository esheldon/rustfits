#!/usr/bin/env python
"""
Compressed 1-D image chunked-read throughput on RANDOM data: the
incompressible-data counterpart to the healsparse benchmark.

Same shape, same three regimes, same methodology as
``perf-compressed-image-read-healsparse.py`` -- only the data differs.
Here the array is pure ``standard_normal`` f8 with no runs and no
sentinel, so it barely compresses (~1x).  This is the comparison case:
with no run structure for the decoder to exploit, GZIP_2 stores the
bytes nearly raw and the decode-bound regimes collapse toward a memcpy,
so the rustfits-vs-fitsio gap should narrow sharply relative to the
healsparse run.  Reading the two side by side shows how much of
rustfits's win comes from faster inflate on real, structured data.

NOTE: random data does not compress, so the generated scratch file is
~the raw size (8 bytes x n).  Lower --ntiles if disk space is tight.

REQUIRES a release build: ``maturin develop --release`` (a debug build
reports rustfits as the loser; see CLAUDE.md).

Run::

    python perf/perf-compressed-image-read-random.py
    python perf/perf-compressed-image-read-random.py --ntiles 32

Generated scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _compread as cr
import _harness as h


def generate(fname, ntiles, tile):
    """
    Write a 1-D f8 GZIP_2 image of ``ntiles`` tiles of pure random
    normal values (no runs, no sentinel) -- essentially incompressible.
    Returns n.
    """
    n = ntiles * tile
    rng = np.random.default_rng(0)
    data = rng.standard_normal(n)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(
            data,
            compress=rustfits.Gzip2(tile_shape=(tile,), heap_format="Q"),
        )
    return n


def main():
    ap = argparse.ArgumentParser()
    cr.add_common_args(ap)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    with h.scratch():
        fname, n, tile = cr.resolve_source(args, generate)
        cr.run_read_benchmark(
            fname,
            n,
            tile,
            title="Compressed 1-D image read (GZIP_2 f8, random)",
            small_chunk=args.small_chunk,
            mid_chunk=args.mid_chunk,
            repeat=args.repeat,
        )


if __name__ == "__main__":
    main()
