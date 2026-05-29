#!/usr/bin/env python
"""
Compressed BINTABLE (ZTABLE) read: rustfits self-comparison.

No cross-tool comparison here: fitsio's Python API does not decompress
ZTABLE (it reads the raw compressed structure and returns wrong values),
and astropy has no compressed-table read.  cfitsio CAN read ZTABLE (the
`funpack` CLI works); only rustfits exposes it transparently from Python.
So this measures the READ cost of decompression: rustfits reading a
ZTABLE vs rustfits reading the equivalent UNCOMPRESSED table, same
32-column catalog, across four regimes.

Complex columns (c8/c16) are excluded -- rustfits's ZTABLE codecs
mishandle them (real/imag swap on c8, 16-byte elements unsupported); see
https://github.com/esheldon/rustfits/issues/8 .

Regimes: whole / column subset / row slice / scattered rows.  Fresh open
per timed iteration; warmup primes the FS cache.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-compressed-read.py
    python perf/perf-table-compressed-read.py --nrows 2000000

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rustfits

import _data
import _harness as h

SUBSET = ["c_f8", "c_i4", "c_f4"]
EXCLUDE = ("c_c8", "c_c16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nrows", type=int, default=h.env_int("PERF_NROWS", 500_000)
    )
    ap.add_argument("--scatter", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    nrows = args.nrows
    fz = h.path("ztable.fits.fz")
    uc = h.path("utable.fits")

    with h.scratch():
        data, vd = _data.catalog_arrays(nrows, exclude=EXCLUDE)
        with rustfits.FITS(fz, "w+") as f:
            f.write_table(data, var_dtypes=vd, compress=True)
        with rustfits.FITS(uc, "w+") as f:
            f.write_table(data, var_dtypes=vd)

        # Gate: ZTABLE read == uncompressed read == original.
        with rustfits.FITS(fz) as f:
            _data.compare_catalog(f[1].read(), data, vd)
        with rustfits.FITS(uc) as f:
            _data.compare_catalog(f[1].read(), data, vd)

        zsize, usize = os.path.getsize(fz), os.path.getsize(uc)
        itemsize = data.dtype.itemsize
        print(
            f"table: {nrows:,} rows x {len(data.dtype.names)} cols, "
            f"row={itemsize} B; uncompressed {usize / 1e6:,.0f} MB, "
            f"ZTABLE {zsize / 1e6:,.0f} MB ({usize / zsize:.1f}x smaller)"
        )

        lo, hi = nrows // 4, nrows // 4 + nrows // 2
        rng = np.random.default_rng(1)
        scatter = rng.integers(0, nrows, size=args.scatter)
        sub_bytes = sum(data.dtype[c].itemsize for c in SUBSET)

        def whole(fn):
            with rustfits.FITS(fn) as f:
                f[1].read()

        def cols(fn):
            with rustfits.FITS(fn) as f:
                f[1].read(columns=SUBSET)

        def rowslice(fn):
            with rustfits.FITS(fn) as f:
                f[1][lo:hi]

        def scattered(fn):
            with rustfits.FITS(fn) as f:
                f[1].read(rows=scatter)

        regimes = [
            ("whole table", whole, nrows * itemsize),
            (f"column subset ({len(SUBSET)} cols)", cols, nrows * sub_bytes),
            (f"row slice [{lo}:{hi}]", rowslice, (hi - lo) * itemsize),
            (
                f"scattered rows (x{args.scatter})",
                scattered,
                args.scatter * itemsize,
            ),
        ]

        rows = []
        for label, fn, nbytes in regimes:
            zt = h.timeit(lambda fn=fn: fn(fz), repeat=args.repeat)
            un = h.timeit(lambda fn=fn: fn(uc), repeat=args.repeat)
            rows.append((label, zt, un, nbytes))

        h.print_env()
        cols_spec = [
            ("regime", 28, "l"),
            ("ZTABLE", 12, "r"),
            ("uncompressed", 13, "r"),
            ("ZT/uncomp", 11, "r"),
            ("ZTABLE rate", 13, "r"),
        ]
        title = f"Compressed BINTABLE read (ZTABLE self, {nrows:,} rows)"
        print()
        print(title)
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
        print(header)
        print("-" * len(header))
        for label, zt, un, nbytes in rows:
            ratio = zt.median / un.median
            cells = [
                (label, 28, "l"),
                (h.fmt_time(zt.median), 12, "r"),
                (h.fmt_time(un.median), 13, "r"),
                (f"{ratio:.2f}x", 11, "r"),
                (h.fmt_rate(nbytes, zt.median), 13, "r"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))


if __name__ == "__main__":
    main()
