#!/usr/bin/env python
"""
Uncompressed BINTABLE VLA repack: time + peak RSS at large heap.

``TableHDU.repack()`` rebuilds the heap with only bytes that live
descriptors point at, dropping orphans accumulated by ``__setitem__``
on VLA columns.  The current implementation (``repack_table_heap``
in ``src/hdu_table/write_vla.rs``) is whole-heap-into-RAM: it reads
the entire main table + the entire old heap under a single file
lock, walks rows × columns to copy live cells into a fresh Vec, then
writes the rebuilt main + heap back.  Peak RSS during repack is
therefore expected to scale linearly with current PCOUNT.

This bench measures the actual scaling at 10 MB, 100 MB, and 1 GB
heap sizes, so users have a number to budget against (and so any
future bounded-memory rewrite has a regression target).

Setup per size:

* Build a fixture with N=10 000 rows on a single VLA ``f4`` column,
  each cell carrying enough floats that the initial heap reaches
  the target size (10 MB → ~250 floats/cell, 1 GB → ~25 000
  floats/cell).
* Setitem every row to a single-float cell, so every initial-write
  cell becomes an orphan; PCOUNT ≈ target_size + tiny.
* Run repack in a fresh subprocess and capture wall time +
  ``VmHWM`` from ``/proc/self/status`` (NOT ``getrusage`` --
  ``ru_maxrss`` is accumulated in ``signal_struct`` and inherited
  across ``fork+exec`` on Linux, so a child of a heavy parent
  reports the parent's peak; the shared ``h.vm_hwm_kb`` helper is
  the immune path).  The build runs in the parent so the repack
  subprocess starts with a clean RSS.

Rustfits-self only (no fitsio comparison): fitsio's Python API has
no equivalent ``repack``.  The orphan-bytes problem only exists
because we ship ``__setitem__`` on VLA columns at all; fitsio
doesn't.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-repack.py
    python perf/perf-table-repack.py --sizes 10,100
    python perf/perf-table-repack.py --rows 1000

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


def floats_per_cell(target_bytes, nrows):
    """
    Pick a per-cell f4 count so the initial heap = ``nrows * cell *
    4`` reaches ``target_bytes``.
    """
    return max(1, target_bytes // (nrows * 4))


def build_fixture(fname, nrows, cell_floats):
    """
    Build a VLA ``f4``-inner table at the given size, then orphan
    every cell by setitem with a 1-element cell.  Final PCOUNT is
    ``nrows * cell_floats * 4`` (initial heap) + ``nrows * 4`` (new
    cells); all initial bytes are orphans, reclaimable by repack.

    Runs in its own subprocess (no need to measure here) so the
    repack timing subprocess starts with a clean RSS.
    """
    big = np.empty(nrows, dtype=object)
    for i in range(nrows):
        big[i] = np.full(cell_floats, float(i), dtype="f4")
    small = np.empty(nrows, dtype=object)
    for i in range(nrows):
        small[i] = np.array([float(i)], dtype="f4")
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            np.dtype([("v", "O")]),
            nrows=nrows,
            var_dtypes={"v": "f4"},
        )
        f[1]["v"] = big
        f[1]["v"] = small


def repack_worker(fname):
    """
    Open the prebuilt fixture, time ``hdu.repack()``, print
    ``<seconds> <peak_rss_kb>`` to stdout.  RSS comes from
    ``h.vm_hwm_kb`` (``/proc/self/status:VmHWM``) -- the kernel's
    per-task high-water, immune to the parent's history (see
    docstring above for the ``getrusage`` inheritance bug).
    """
    with rustfits.FITS(fname, "r+") as f:
        hdu = f[1]
        t0 = time.perf_counter()
        hdu.repack()
        t = time.perf_counter() - t0
    rss_kb = h.vm_hwm_kb()
    print(f"{t:.6f} {rss_kb}")


def run_one_size(target_mb, rows, args):
    """
    For one target heap size:
      1. Build a fixture once per repeat (build mutates the file,
         and repack mutates further; each repeat needs a fresh
         starting state).
      2. Run the repack subprocess, capture (time, rss).
    Returns (median_time_s, max_rss_kb, pcount_bytes, live_bytes).
    """
    target_bytes = target_mb * 1024 * 1024
    cell_floats = floats_per_cell(target_bytes, rows)
    live_bytes = rows * 4
    initial_heap = rows * cell_floats * 4
    pcount_after_orphan = initial_heap + live_bytes
    times, rss = [], []
    for _ in range(args.repeat):
        fname = h.fresh_path(f"trepack-{target_mb}mb")
        # In-process build is fine here; the repack worker measures
        # its own peak RSS in isolation.
        build_fixture(fname, rows, cell_floats)
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
        pcount_after_orphan,
        live_bytes,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument(
        "--sizes",
        default="10,100,1000",
        help="comma-separated target heap sizes in MB (default 10,100,1000)",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="number of VLA rows (default 10 000); cell size auto-"
        "scales to hit each target heap size",
    )
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        repack_worker(args.worker)
        return

    sizes_mb = [int(x) for x in args.sizes.split(",")]
    title = (
        f"BINTABLE VLA repack (orphan reclaim) — N={args.rows:,} rows, "
        f"cell size auto-scaled to hit target heap"
    )
    print(title)
    h.print_env()
    print()

    cols_spec = [
        ("target heap", 13, "r"),
        ("cells", 9, "r"),
        ("live bytes", 11, "r"),
        ("repack t", 11, "r"),
        ("peak RSS", 11, "r"),
        ("RSS / heap", 12, "r"),
    ]
    header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
    print(header)
    print("-" * len(header))

    with h.scratch():
        for mb in sizes_mb:
            t, rss_kb, pcount, live = run_one_size(mb, args.rows, args)
            rss_mb = rss_kb / 1024
            ratio = (rss_kb * 1024) / pcount
            cells = [
                (f"{mb:,} MB", 13, "r"),
                (f"{args.rows:,}", 9, "r"),
                (f"{live:,}", 11, "r"),
                (h.fmt_time(t), 11, "r"),
                (f"{rss_mb:,.0f} MB", 11, "r"),
                (f"{ratio:.2f}×", 12, "r"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))
            h.emit_record(
                {
                    "kind": "rss",
                    "suite": title,
                    "op": f"target {mb} MB",
                    "build_s": t,
                    "peak_rss_mb": rss_mb,
                    "nbytes": pcount,
                }
            )


if __name__ == "__main__":
    main()
