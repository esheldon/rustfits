#!/usr/bin/env python
"""
Compressed BINTABLE (ZTABLE) write: rustfits self-comparison.

No cross-tool comparison: fitsio/astropy have no high-level ZTABLE
writer (only the `fpack -table` CLI), so this measures the WRITE cost of
compression -- rustfits writing a ZTABLE vs rustfits writing the
equivalent UNCOMPRESSED table, same 34-column catalog.

Input built ONCE (untimed); each timed iteration overwrites a scratch
file; no fsync; warmup primes.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-compressed-write.py
    python perf/perf-table-compressed-write.py --nrows 2000000

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import os

import rustfits

import _data
import _harness as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nrows", type=int, default=h.env_int("PERF_NROWS", 500_000)
    )
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    nrows = args.nrows
    fz = h.path("ztable.fits.fz")
    uc = h.path("utable.fits")
    data, vd = _data.catalog_arrays(nrows)

    def write_z():
        with rustfits.FITS(fz, "w+") as f:
            f.write_table(data, var_dtypes=vd, compress=True)

    def write_u():
        with rustfits.FITS(uc, "w+") as f:
            f.write_table(data, var_dtypes=vd)

    with h.scratch():
        # Gate: both writes round-trip.
        write_z()
        with rustfits.FITS(fz) as f:
            _data.compare_catalog(f[1].read(), data, vd)
        write_u()
        with rustfits.FITS(uc) as f:
            _data.compare_catalog(f[1].read(), data, vd)

        zsize, usize = os.path.getsize(fz), os.path.getsize(uc)
        itemsize = data.dtype.itemsize
        print(
            f"table: {nrows:,} rows x {len(data.dtype.names)} cols, "
            f"row={itemsize} B; uncompressed {usize / 1e6:,.0f} MB, "
            f"ZTABLE {zsize / 1e6:,.0f} MB ({usize / zsize:.1f}x smaller)"
        )

        zt = h.timeit(write_z, repeat=args.repeat)
        un = h.timeit(write_u, repeat=args.repeat)

        h.print_env()
        cols_spec = [
            ("operation", 28, "l"),
            ("ZTABLE", 12, "r"),
            ("uncompressed", 13, "r"),
            ("ZT/uncomp", 11, "r"),
            ("ZTABLE rate", 13, "r"),
        ]
        title = f"Compressed BINTABLE write (ZTABLE self, {nrows:,} rows)"
        print()
        print(title)
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
        print(header)
        print("-" * len(header))
        ratio = zt.median / un.median
        cells = [
            ("write catalog", 28, "l"),
            (h.fmt_time(zt.median), 12, "r"),
            (h.fmt_time(un.median), 13, "r"),
            (f"{ratio:.2f}x", 11, "r"),
            (h.fmt_rate(nrows * itemsize, zt.median), 13, "r"),
        ]
        print("  ".join(h._cell(x, w, a) for x, w, a in cells))


if __name__ == "__main__":
    main()
