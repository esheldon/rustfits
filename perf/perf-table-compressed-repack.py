#!/usr/bin/env python
"""
ZTABLE compressed-table repack: time + peak RSS at large heap.

``CompressedTableHDU.repack()`` is documented (in CLAUDE.md, "Heap
repack" section) as streaming + staging: bounded at ``~1 MiB chunk
+ descriptor table + move-plan vector``, no heap-in-RAM allocation.
This bench measures the actual peak RSS at 10 MB, 100 MB, and 1 GB
PCOUNT to verify the bound.

This is the only one of the three repack benches that should
demonstrate constant-RAM scaling.  ``perf-table-repack.py`` and
``perf-compressed-image-repack.py`` both show RSS scaling linearly
with heap size (whole-heap-into-RAM implementations).  If this one
also scales, the streaming claim in CLAUDE.md is wrong.

Setup per size:

* Single wide ``f4`` column; ``write_table(data, compress=True)``
  encodes the per-column slabs into ZTABLE tiles.
* ``hdu[:] = data2`` overwrites every row, decoding + re-encoding
  every tile.  Old tile blobs become orphans; PCOUNT roughly
  doubles.
* Repack subprocess: open, ``hdu.repack()``, capture wall time and
  ``VmHWM`` from ``/proc/self/status`` (NOT ``getrusage`` —
  ``ru_maxrss`` is inherited across ``fork+exec`` and would
  silently report the build subprocess's peak).

Rustfits-self only — no library outside rustfits writes ZTABLE,
and fitsio's Python API cannot decompress / repack one.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-compressed-repack.py
    python perf/perf-table-compressed-repack.py --sizes 10,100

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import rustfits

import _harness as h


# Wide-f4 schema: one column of M f4 elements per row.  The bench
# tunes N or M per target size to hit the desired post-orphan
# PCOUNT.  Single column keeps the schema cost (descriptors,
# header) negligible relative to the heap; the heap IS the column
# data.
def schema_for(target_mb, nrows=10_000):
    """
    Pick a per-row f4 count so a noise-data ``write_table`` +
    ``setitem(:)`` produces ~target_mb of post-orphan PCOUNT.
    GZIP_1 on f4 noise compresses ~85%, so target / (2 × 0.85) =
    raw MB; / (4 bytes) = total floats; / nrows = per-row floats.
    """
    raw_mb = target_mb / (2 * 0.85)
    total_floats = int(raw_mb * 1024 * 1024 / 4)
    m = max(100, total_floats // nrows)
    # Round to 100 for tidy numbers.
    m = (m // 100) * 100
    return nrows, m


def build_fixture(fname, nrows, m):
    """
    Write a wide-f4 ZTABLE + setitem(:) once to orphan every
    initial tile blob.  Final PCOUNT ≈ 2 × initial heap.
    """
    dtype = np.dtype([("wide", "f4", m)])
    rng = np.random.default_rng(0)
    data = np.empty(nrows, dtype=dtype)
    data["wide"] = rng.standard_normal((nrows, m)).astype("f4")
    with rustfits.FITS(fname, "w+") as f:
        f.write_table(data, compress=True)
        data2 = np.empty(nrows, dtype=dtype)
        data2["wide"] = rng.standard_normal((nrows, m)).astype("f4")
        f[1][:] = data2
    del data, data2


def repack_worker(fname):
    """
    Fresh-subprocess open + time ``hdu.repack()`` + VmHWM read.
    """
    with rustfits.FITS(fname, "r+") as f:
        hdu = f[1]
        t0 = time.perf_counter()
        hdu.repack()
        t = time.perf_counter() - t0
    rss_kb = h.vm_hwm_kb()
    print(f"{t:.6f} {rss_kb}")


def run_one_size(target_mb, args):
    """
    Build a fresh fixture per repeat (build + repack both mutate),
    spawn the repack worker, collect (time, rss).  Returns
    (median_time_s, max_rss_kb, nrows, m, pcount).
    """
    nrows, m = schema_for(target_mb)
    times, rss, pcounts = [], [], []
    for _ in range(args.repeat):
        fname = h.fresh_path(f"ztrepack-{target_mb}mb")
        build_fixture(fname, nrows, m)
        with rustfits.FITS(fname) as f:
            pcounts.append(int(f[1].header["PCOUNT"]))
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            fname,
        ]
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout
        ts, rk = out.split()
        times.append(float(ts))
        rss.append(int(rk))
        try:
            os.remove(fname)
        except OSError:
            pass
    times.sort()
    return (
        times[len(times) // 2],
        max(rss),
        nrows,
        m,
        max(pcounts),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument(
        "--sizes",
        default="10,100,1000",
        help="comma-separated target post-orphan PCOUNT in MB "
        "(default 10,100,1000)",
    )
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        repack_worker(args.worker)
        return

    sizes_mb = [int(x) for x in args.sizes.split(",")]
    title = (
        "ZTABLE compressed-table repack — single wide f4 column, "
        "setitem(:) orphans every initial tile blob"
    )
    print(title)
    h.print_env()
    print()

    cols_spec = [
        ("target", 10, "r"),
        ("schema", 20, "r"),
        ("PCOUNT", 12, "r"),
        ("repack t", 11, "r"),
        ("peak RSS", 11, "r"),
        ("RSS / PCOUNT", 14, "r"),
    ]
    header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
    print(header)
    print("-" * len(header))

    with h.scratch():
        for mb in sizes_mb:
            t, rss_kb, nrows, m, pcount = run_one_size(mb, args)
            rss_mb = rss_kb / 1024
            pcount_mb = pcount / 1e6
            ratio = (rss_kb * 1024) / pcount
            cells = [
                (f"{mb:,} MB", 10, "r"),
                (f"{nrows:,}×{m}f4", 20, "r"),
                (f"{pcount_mb:,.0f} MB", 12, "r"),
                (h.fmt_time(t), 11, "r"),
                (f"{rss_mb:,.0f} MB", 11, "r"),
                (f"{ratio:.3f}×", 14, "r"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))
            h.emit_record(
                {
                    "kind": "rss",
                    "suite": title,
                    "op": f"target {mb} MB ({nrows:,}×{m} f4)",
                    "build_s": t,
                    "peak_rss_mb": rss_mb,
                    "nbytes": pcount,
                }
            )


if __name__ == "__main__":
    main()
