#!/usr/bin/env python
"""
Uncompressed BINTABLE write throughput: rustfits vs fitsio.

Writes the type-exhaustive catalog (see _data.catalog_arrays: every
scalar type, f4/f8 fixed sub-arrays, S+U fixed and VLA strings, an f4
VLA).  fitsio writes the VLA object columns as true variable-length
columns (1PA/1PE), same as rustfits, so the comparison is fair; the
correctness gate reads each back (fitsio with vstorage="object") and
checks it round-trips.

Input built ONCE (untimed); each timed iteration overwrites a scratch
file; no fsync; warmup primes.  Row width ~612 B, so ~1.75M rows is
~1 GB; default is smaller for iteration.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-write.py
    python perf/perf-table-write.py --nrows 1750000

Scratch file goes to CWD as perf-tmp-* and is removed on exit.
"""

from __future__ import annotations

import argparse

import rustfits

import _data
import _harness as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nrows", type=int, default=h.env_int("PERF_NROWS", 500_000)
    )
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    nrows = args.nrows
    data, vd = _data.catalog_arrays(nrows)

    def rf_write():
        fname = h.fresh_path("wtable-rf")
        with rustfits.FITS(fname, "w+") as f:
            f.write_table(data, var_dtypes=vd)

    def fi_write():
        fname = h.fresh_path("wtable-fi")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(data)

    with h.scratch():
        # Correctness gate: each tool's write round-trips.
        gate_rf = h.fresh_path("wtable-gate-rf")
        gate_fi = h.fresh_path("wtable-gate-fi")
        with rustfits.FITS(gate_rf, "w+") as f:
            f.write_table(data, var_dtypes=vd)
        with rustfits.FITS(gate_rf) as f:
            _data.compare_catalog(f[1].read(), data, vd)
        with fitsio.FITS(gate_fi, "rw", clobber=True) as f:
            f.write(data)
        with fitsio.FITS(gate_fi) as f:
            _data.compare_catalog(f[1].read(vstorage="object"), data, vd)

        import os

        itemsize = data.dtype.itemsize
        print(
            f"table: {nrows:,} rows x {len(data.dtype.names)} cols, "
            f"row={itemsize} B, "
            f"{os.path.getsize(gate_rf) / 1e6:,.0f} MB on disk"
        )

        result = h.bench(
            "write catalog",
            rf_write,
            run_fitsio=fi_write,
            nbytes=nrows * itemsize,
            repeat=args.repeat,
        )
        h.print_env()
        h.report(f"Uncompressed BINTABLE write ({nrows:,} rows)", [result])


if __name__ == "__main__":
    main()
