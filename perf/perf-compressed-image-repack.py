#!/usr/bin/env python
"""
ZIMAGE compressed-image repack: time + peak RSS at large heap.

``CompressedImageHDU.repack()`` rebuilds the heap with only the
compressed tile bytes that live descriptors reference, dropping
orphans left by previous ``__setitem__``/``extend`` calls.  The
current implementation (``repack_compressed_heap`` in
``src/hdu_image_compressed/repack.rs``) is whole-heap-into-RAM:
reads the entire main table + entire old heap under a single file
lock, walks rows to copy live tile blobs into a fresh Vec, writes
back, shrinks the file extent.

This bench measures peak RSS scaling at 10 MB, 100 MB, and 1 GB
PCOUNT.  Setup per size:

* Pick image dimensions so a GZIP_1 write produces ~target/2 bytes
  of compressed heap.
* ``hdu[:] = data`` on the full image to re-encode every tile,
  appending new bytes to the heap and orphaning the entire initial
  encoding.  Final PCOUNT ≈ 2 × initial.
* Run repack in a fresh subprocess and capture wall time +
  ``VmHWM`` from ``/proc/self/status`` (NOT ``getrusage`` —
  ``ru_maxrss`` is inherited across ``fork+exec`` on Linux and
  would silently report the build subprocess's peak).

Rustfits-self only (no fitsio comparison): fitsio's Python API has
no equivalent ``repack`` for compressed images.  The orphan
accumulation only exists because we ship ``__setitem__`` and
``extend`` on compressed images; fitsio doesn't.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-compressed-image-repack.py
    python perf/perf-compressed-image-repack.py --sizes 10,100

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


def image_shape_for(target_mb):
    """
    Pick a square image shape so a GZIP_1 f4 encode produces
    ~target_mb / 2 bytes of compressed heap (the setitem step
    doubles PCOUNT, so the post-orphan PCOUNT lands near
    target_mb).  Noise compresses ~80% with GZIP_1, so raw bytes
    ≈ target/2 / 0.8 ≈ target * 0.625.  At 4 bytes/pixel, pixel
    count ≈ target * 0.156 M.  Rounded to a square.
    """
    raw_bytes = int(target_mb * 1024 * 1024 * 0.625)
    n_pixels = raw_bytes // 4
    side = int(n_pixels**0.5)
    # Snap to a multiple of 100 for tidy numbers.
    side = (side // 100) * 100
    return (side, side)


def build_fixture(fname, shape, tile_shape):
    """
    Write a GZIP_1 compressed image + setitem(:) it once to
    orphan the entire initial heap.  Runs in this process; the
    repack worker subprocess starts fresh.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape, dtype=np.float32)
    with rustfits.FITS(fname, "w+") as f:
        f.create_image_hdu(
            "f4",
            shape,
            compress=rustfits.Gzip1(tile_shape=tile_shape),
        )
        f[1].write(data)
        # Overwrite EVERY tile with new (different) data so old
        # tile bytes become orphans.  Use a different random stream
        # so the new compressed bytes differ from the initial ones.
        data2 = rng.standard_normal(shape, dtype=np.float32)
        f[1][:, :] = data2
    del data, data2


def repack_worker(fname):
    """
    Open the prebuilt fixture, time ``hdu.repack()``, print
    ``<seconds> <peak_rss_kb>`` to stdout.  Uses VmHWM via
    ``h.vm_hwm_kb`` so the reported RSS is just this subprocess's
    high-water (immune to the parent's history).
    """
    with rustfits.FITS(fname, "r+") as f:
        hdu = f[1]
        t0 = time.perf_counter()
        hdu.repack()
        t = time.perf_counter() - t0
    rss_kb = h.vm_hwm_kb()
    print(f"{t:.6f} {rss_kb}")


def run_one_size(target_mb, tile, args):
    """
    Build a fixture once per repeat (build mutates; repack mutates
    further), spawn the repack worker, collect (time, rss).
    Returns (median_time_s, max_rss_kb, image_shape, pcount).
    """
    shape = image_shape_for(target_mb)
    times, rss, pcounts = [], [], []
    for _ in range(args.repeat):
        fname = h.fresh_path(f"crepack-{target_mb}mb")
        build_fixture(fname, shape, tile)
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
        shape,
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
    ap.add_argument(
        "--tile",
        default="100,100",
        help="tile shape rows,cols (default 100,100)",
    )
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        repack_worker(args.worker)
        return

    sizes_mb = [int(x) for x in args.sizes.split(",")]
    th, tw = (int(x) for x in args.tile.split(","))
    tile = (th, tw)
    title = (
        f"ZIMAGE compressed-image repack (GZIP_1, tile={th}×{tw}) — "
        f"f4 noise, setitem(:) orphans the entire initial heap"
    )
    print(title)
    h.print_env()
    print()

    cols_spec = [
        ("target", 10, "r"),
        ("image shape", 14, "r"),
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
            t, rss_kb, shape, pcount = run_one_size(mb, tile, args)
            rss_mb = rss_kb / 1024
            pcount_mb = pcount / 1e6
            ratio = (rss_kb * 1024) / pcount
            cells = [
                (f"{mb:,} MB", 10, "r"),
                (f"{shape[0]}×{shape[1]}", 14, "r"),
                (f"{pcount_mb:,.0f} MB", 12, "r"),
                (h.fmt_time(t), 11, "r"),
                (f"{rss_mb:,.0f} MB", 11, "r"),
                (f"{ratio:.2f}×", 14, "r"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))
            h.emit_record(
                {
                    "kind": "rss",
                    "suite": title,
                    "op": f"target {mb} MB ({shape[0]}×{shape[1]})",
                    "build_s": t,
                    "peak_rss_mb": rss_mb,
                    "nbytes": pcount,
                }
            )


if __name__ == "__main__":
    main()
