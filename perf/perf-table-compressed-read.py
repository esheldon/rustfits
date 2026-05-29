#!/usr/bin/env python
"""
Compressed BINTABLE (ZTABLE) read: rustfits self-comparison.

No cross-tool comparison here: fitsio's Python API does not decompress
ZTABLE (it reads the raw compressed structure and returns wrong values),
and astropy has no compressed-table read.  cfitsio CAN read ZTABLE (the
`funpack` CLI works); only rustfits exposes it transparently from Python.
So this measures the READ cost of decompression: rustfits reading a
ZTABLE vs rustfits reading the equivalent UNCOMPRESSED table, same
34-column catalog, across four regimes.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nrows", type=int, default=h.env_int("PERF_NROWS", 500_000)
    )
    ap.add_argument("--scatter", type=int, default=2000)
    ap.add_argument(
        "--ztilelen",
        type=int,
        default=0,
        help="rows per compression tile (0 = cfitsio default ~17k); "
        "smaller helps scattered reads, hurts compression",
    )
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    nrows = args.nrows
    ztilelen = args.ztilelen or None
    fz = h.path("ztable.fits.fz")
    uc = h.path("utable.fits")

    with h.scratch():
        data, vd = _data.catalog_arrays(nrows)
        with rustfits.FITS(fz, "w+") as f:
            f.write_table(
                data, var_dtypes=vd, compress=True, ztilelen=ztilelen
            )
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
            f"row={itemsize} B, ztilelen={ztilelen or 'default'}; "
            f"uncompressed {usize / 1e6:,.0f} MB, ZTABLE "
            f"{zsize / 1e6:,.0f} MB ({usize / zsize:.1f}x smaller)"
        )

        lo, hi = nrows // 4, nrows // 4 + nrows // 2
        rng = np.random.default_rng(1)
        scatter = rng.integers(0, nrows, size=args.scatter)
        sub_bytes = sum(data.dtype[c].itemsize for c in SUBSET)

        # ZTABLE reads size the per-(tile,col) cache to hold the whole
        # table so each tile decompresses once -- cache-neutral, matching
        # the compressed-image stamp test.  Without this, scattered
        # random-row reads thrash the default 32 MiB cache and the result
        # measures eviction, not decode.  (No-op for the plain
        # uncompressed table, which has no tile cache.)
        cache = nrows * itemsize + (16 << 20)

        def read(fn, op):
            with rustfits.FITS(fn) as f:
                hdu = f[1]
                if fn == fz:
                    hdu.set_tile_cache_size(cache)
                op(hdu)

        regimes = [
            ("whole table", lambda x: x.read(), nrows * itemsize),
            (
                f"column subset ({len(SUBSET)} cols)",
                lambda x: x.read(columns=SUBSET),
                nrows * sub_bytes,
            ),
            (
                f"row slice [{lo}:{hi}]",
                lambda x: x[lo:hi],
                (hi - lo) * itemsize,
            ),
            (
                f"scattered rows (x{args.scatter})",
                lambda x: x.read(rows=scatter),
                args.scatter * itemsize,
            ),
        ]

        rows = []
        for label, op, nbytes in regimes:
            zt = h.timeit(lambda op=op: read(fz, op), repeat=args.repeat)
            un = h.timeit(lambda op=op: read(uc, op), repeat=args.repeat)
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
