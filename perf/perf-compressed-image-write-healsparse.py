#!/usr/bin/env python
"""
Compressed 1-D image WRITE (encode) throughput: rustfits vs fitsio.

The write side of the healsparse read benchmark: time write_image of the
same run-structured 1-D f8 array, GZIP_2-compressed, rustfits vs fitsio.

The input array is built ONCE (untimed); each timed iteration overwrites
a scratch file.  We do NOT fsync -- the compressed output is small so
encode dominates, and buffered writes isolate encode-code speed from disk
flush.  A correctness gate writes once with each tool and reads it back
bit-exact (GZIP is lossless) so a broken/empty write can't post a fast
number.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-compressed-image-write-healsparse.py
    python perf/perf-compressed-image-write-healsparse.py --ntiles 128

Scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _data
import _harness as h

DEFAULT_TILE = 1048576


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntiles", type=int, default=h.env_int("PERF_NTILES", 64))
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--run-len", type=int, default=32)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--quant", type=int, default=1)
    ap.add_argument(
        "--level",
        type=int,
        default=1,
        help="gzip level; cfitsio hardcodes 1 (Z_BEST_SPEED) for tiles",
    )
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    tile, level = args.tile, args.level
    n = args.ntiles * tile
    data = _data.healsparse_array(n, args.run_len, args.cov, args.quant)

    def rf_write():
        # Match cfitsio's gzip level (Z_BEST_SPEED=1) for a fair
        # encode-SPEED comparison: rustfits defaults to level 6, which
        # compresses ~18% smaller but does more work, so leaving it would
        # measure compression effort, not encoder speed.
        fname = h.fresh_path("wcomp1d-rf")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(
                data,
                compress=rustfits.Gzip2(
                    tile_shape=(tile,), heap_format="Q", level=level
                ),
            )

    def fi_write():
        # qlevel=0 forces LOSSLESS GZIP_2: fitsio quantizes float data by
        # default, which would make this an unfair lossy-vs-lossless
        # comparison (rustfits's Gzip2 without a Quantize is lossless, as
        # is the healsparse read benchmark and the real file).  cfitsio's
        # gzip level is fixed at Z_BEST_SPEED=1 (zcompress.c), matched above.
        fname = h.fresh_path("wcomp1d-fi")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(data, compress="GZIP_2", tile_dims=(tile,), qlevel=0)

    with h.scratch():
        # Correctness gate: each tool's write round-trips bit-exact.
        gate_rf = h.fresh_path("wcomp1d-gate-rf")
        gate_fi = h.fresh_path("wcomp1d-gate-fi")
        with rustfits.FITS(gate_rf, "w+") as f:
            f.write_image(
                data,
                compress=rustfits.Gzip2(
                    tile_shape=(tile,), heap_format="Q", level=level
                ),
            )
        with rustfits.FITS(gate_rf) as f:
            np.testing.assert_array_equal(f[1].read(), data)
        with fitsio.FITS(gate_fi, "rw", clobber=True) as f:
            f.write(data, compress="GZIP_2", tile_dims=(tile,), qlevel=0)
        with fitsio.FITS(gate_fi) as f:
            np.testing.assert_array_equal(f[1].read(), data)

        result = h.bench(
            f"write GZIP_2 level={level} ({n * 8 / 1e6:,.0f} MB f8)",
            rf_write,
            run_fitsio=fi_write,
            nbytes=n * 8,
            repeat=args.repeat,
        )
        h.print_env()
        h.report("Compressed 1-D image WRITE (GZIP_2 f8)", [result])


if __name__ == "__main__":
    main()
