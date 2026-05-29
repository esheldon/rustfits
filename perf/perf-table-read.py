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


def check(read, orig, vd):
    """
    Compare a table read to the original, tolerating rustfits reading
    strings as str vs the original bytes.
    """
    n = len(orig)
    for name in orig.dtype.names:
        if name in vd:
            for i in (0, n // 2, n - 1):
                ov, rv = orig[name][i], read[name][i]
                if vd[name] in ("S", "U"):
                    ov = ov.decode() if isinstance(ov, bytes) else ov
                    rv = rv.decode() if isinstance(rv, bytes) else rv
                    assert ov == rv, (name, i)
                else:
                    assert np.array_equal(rv, ov), (name, i)
        elif orig[name].dtype.kind in ("S", "U"):
            o = orig[name]
            r = read[name]
            o = np.char.encode(o) if o.dtype.kind == "U" else o
            r = np.char.encode(r) if r.dtype.kind == "U" else r
            np.testing.assert_array_equal(r, o)
        else:
            np.testing.assert_array_equal(read[name], orig[name])


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

        # Correctness gate: both readers agree with the original.
        with rustfits.FITS(fname) as f:
            check(f[1].read(), data, vd)
        with fitsio.FITS(fname) as f:
            check(f[1].read(), data, vd)

        lo, hi = nrows // 4, nrows // 4 + nrows // 2
        rng = np.random.default_rng(1)
        scatter = rng.integers(0, nrows, size=args.scatter)
        sub_bytes = sum(data.dtype[c].itemsize for c in SUBSET)

        def whole(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read()

            return run

        def cols(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read(columns=SUBSET)

            return run

        def rowslice(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1][lo:hi]

            return run

        def scattered(mod):
            def run():
                with mod.FITS(fname) as f:
                    f[1].read(rows=scatter)

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
