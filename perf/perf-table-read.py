#!/usr/bin/env python
"""
Uncompressed BINTABLE read throughput: rustfits vs fitsio.

A deliberately type-exhaustive catalog (see _data.catalog_arrays): every
major scalar type, f4/f8 fixed sub-arrays (1-D and 2-D), and two VLA
columns (string + f4).  Four read regimes:

* whole table   -- hdu.read() (all rows, all columns); bulk.
* column subset -- read a few scalar columns; the projection case, where
                   strided per-column read efficiency is tested.
* row slice     -- hdu[lo:hi]; a contiguous row range.
* scattered     -- read(rows=[random ids]); object-lookup pattern.

Fresh open per timed iteration; the harness warmup primes the FS cache.
Content is irrelevant to read speed.  Row width ~540 B, so ~2M rows is
~1 GB; default is smaller for iteration (try --nrows 2000000 for ~1 GB).

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-read.py
    python perf/perf-table-read.py --nrows 2000000

Scratch file goes to CWD as perf-tmp-* and is removed on exit.
"""

from __future__ import annotations

import argparse

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
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    nrows = args.nrows
    fname = h.path("table.fits")

    with h.scratch():
        data, vd = _data.catalog_arrays(nrows)
        with rustfits.FITS(fname, "w+") as f:
            f.write_table(data, var_dtypes=vd)

        import os

        itemsize = data.dtype.itemsize
        print(
            f"table: {nrows:,} rows x {len(data.dtype.names)} cols, "
            f"row={itemsize} B, {os.path.getsize(fname) / 1e6:,.0f} MB on disk"
        )

        # Correctness gate: both readers agree with the original.  fitsio
        # needs vstorage="object" to return true VLA cells (its default,
        # "fixed", pads each cell to the column's max length).
        with rustfits.FITS(fname) as f:
            _data.compare_catalog(f[1].read(), data, vd)
        with fitsio.FITS(fname) as f:
            _data.compare_catalog(f[1].read(vstorage="object"), data, vd)

        lo, hi = nrows // 4, nrows // 4 + nrows // 2
        rng = np.random.default_rng(1)
        scatter = rng.integers(0, nrows, size=args.scatter)
        sub_bytes = sum(data.dtype[c].itemsize for c in SUBSET)

        # fitsio reads VLA as object arrays only with vstorage="object"
        # (matching rustfits); its default pads to max length, which is
        # different work.  rustfits.read() takes no such kwarg.
        def vskw(mod):
            return {"vstorage": "object"} if mod is fitsio else {}

        def whole(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read(**vskw(mod))

            return run

        def cols(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read(columns=SUBSET, **vskw(mod))

            return run

        def rowslice(mod):
            # native contiguous slice on both (fitsio returns fixed-padded
            # VLA here; dwarfed by the fixed-column bulk).
            def run():
                with mod.FITS(fname) as f:
                    f[1][lo:hi]

            return run

        def scattered(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read(rows=scatter, **vskw(mod))

            return run

        results = [
            h.bench(
                "whole table",
                whole(rustfits),
                run_fitsio=whole(fitsio),
                nbytes=nrows * itemsize,
                repeat=args.repeat,
            ),
            h.bench(
                f"column subset ({len(SUBSET)} cols)",
                cols(rustfits),
                run_fitsio=cols(fitsio),
                nbytes=nrows * sub_bytes,
                repeat=args.repeat,
            ),
            h.bench(
                f"row slice [{lo}:{hi}]",
                rowslice(rustfits),
                run_fitsio=rowslice(fitsio),
                nbytes=(hi - lo) * itemsize,
                repeat=args.repeat,
            ),
            h.bench(
                f"scattered rows (x{args.scatter})",
                scattered(rustfits),
                run_fitsio=scattered(fitsio),
                nbytes=args.scatter * itemsize,
                repeat=args.repeat,
            ),
        ]

        h.print_env()
        h.report(f"Uncompressed BINTABLE read ({nrows:,} rows)", results)


if __name__ == "__main__":
    main()
