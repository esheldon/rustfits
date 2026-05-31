#!/usr/bin/env python
"""
Compressed 2-D image EXTEND: bounded-memory mosaic build.

The 2-D analog of ``perf-compressed-image-extend-healsparse.py``,
covering the mosaic / strip pattern: append per-detector frames
(or per-night strips) to a growing tile-compressed image.

fitsio cannot extend a compressed image (``FITSIO status = 107:
tried to move past end of file``), so this is a rustfits self-
comparison:

* ``fitsio write-once``   -- O(rows*cols) RAM, write whole image.
* ``rustfits write-once`` -- same shape, the reference.
* ``rustfits extend``     -- create empty + append C-row chunks,
                             at three chunk-vs-tile alignments:
                             below ZTILE rows (re-encodes the
                             partial trailing tile every call),
                             exact tile-row, and several tile rows.

The chunk-row sweep exposes the same kind of small-chunk re-encode
cost as the ZTABLE merge-tile append.  Default tile is
``(100, cols)`` (full-width strips, 100 rows tall) so the chunk-
size axis maps to "tile-rows per append".

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-compressed-image-extend-2d.py
    python perf/perf-compressed-image-extend-2d.py --rows 40000
    python perf/perf-compressed-image-extend-2d.py --chunks 50,100,1000

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

import _data
import _harness as h


def build_worker(mode, rows, cols, chunk_rows, tile_rows, level):
    """
    One build per ``mode``, timing only the encode/write calls;
    print ``<build_seconds> <peak_rss_kb>``.  Runs in a subprocess
    for a clean per-build ru_maxrss.  Per-chunk frames are
    generated on the fly for ``extend_rf`` so peak RAM stays at
    O(chunk_rows * cols) plus the FITS handle's own pages.
    """
    fname = h.path("cext2dbuild.fits.fz")
    tile_shape = (tile_rows, cols)
    cfg = dict(tile_shape=tile_shape, level=level)

    if mode == "extend_rf":
        t = 0.0
        done = 0
        i = 0
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("f4", (0, cols), compress=rustfits.Gzip2(**cfg))
            hdu = f[1]
            while done < rows:
                c = min(chunk_rows, rows - done)
                data = _data.des_array(c, cols, zero_frac=0.05, seed=i + 1)
                t0 = time.perf_counter()
                hdu.extend(data)
                t += time.perf_counter() - t0
                done += c
                i += 1
                del data
    else:
        data = _data.des_array(rows, cols, zero_frac=0.05, seed=0)
        t0 = time.perf_counter()
        if mode == "write_rf":
            with rustfits.FITS(fname, "w+") as f:
                f.write_image(data, compress=rustfits.Gzip2(**cfg))
        elif mode == "write_fi":
            import fitsio

            with fitsio.FITS(fname, "rw", clobber=True) as f:
                f.write(
                    data,
                    compress="GZIP_2",
                    tile_dims=tile_shape,
                    qlevel=0,
                )
        else:
            raise ValueError(f"unknown mode: {mode}")
        t = time.perf_counter() - t0

    rss_kb = h.vm_hwm_kb()
    try:
        os.remove(fname)
    except OSError:
        pass
    print(f"{t:.6f} {rss_kb}")


def run_build(mode, chunk_rows, args):
    """
    Subprocess the worker ``args.repeat`` times; return
    ``(median_time_s, max_rss_kb)``.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        mode,
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--chunk",
        str(chunk_rows),
        "--tile-rows",
        str(args.tile_rows),
        "--level",
        str(args.level),
    ]
    times, rss = [], []
    for _ in range(args.repeat):
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout
        ts, rk = out.split()
        times.append(float(ts))
        rss.append(int(rk))
    times.sort()
    return times[len(times) // 2], max(rss)


def gate(args):
    """
    Small in-process check that extend round-trips bit-exact
    (GZIP_2 is lossless), using sub-tile chunks so the
    partial-last-tile path runs.
    """
    frames = [
        _data.des_array(50, args.cols, zero_frac=0.05, seed=i + 1)
        for i in range(3)
    ]
    full = np.concatenate(frames)
    gf = h.path("cext2dgate.fits.fz")
    with rustfits.FITS(gf, "w+") as f:
        f.create_image_hdu(
            "f4",
            (0, args.cols),
            compress=rustfits.Gzip2(
                tile_shape=(args.tile_rows, args.cols),
                level=args.level,
            ),
        )
        for fr in frames:
            f[1].extend(fr)
    with rustfits.FITS(gf) as f:
        np.testing.assert_array_equal(f[1].read(), full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--cols", type=int, default=4_000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument(
        "--chunks",
        default="50,100,1000",
        help="comma-separated chunk row counts; default sweeps "
        "below tile-rows (partial-tile re-encode), exact tile, "
        "and multi-tile",
    )
    ap.add_argument(
        "--tile-rows",
        type=int,
        default=100,
        help="tile shape rows (tile cols == cols)",
    )
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        build_worker(
            args.worker,
            args.rows,
            args.cols,
            args.chunk,
            args.tile_rows,
            args.level,
        )
        return

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    with h.scratch():
        gate(args)

        rows = args.rows
        cols = args.cols
        nbytes = rows * cols * 4
        chunks = [int(c) for c in args.chunks.split(",")]
        print(
            f"build ({rows:,} x {cols:,}) f4 ({nbytes / 1e6:,.0f} MB raw), "
            f"GZIP_2 tile=({args.tile_rows}, {cols}) level={args.level}"
        )
        h.print_env()

        rows_out = [
            ("fitsio write-once", run_build("write_fi", 0, args)),
            ("rustfits write-once", run_build("write_rf", 0, args)),
        ]
        ref_t, ref_r = rows_out[1][1]
        for c in chunks:
            k_chunks = -(-rows // c)
            tile_ratio = c / args.tile_rows
            tag = (
                "sub-tile"
                if tile_ratio < 1.0
                else "exact tile"
                if tile_ratio == 1.0
                else f"{tile_ratio:g} tiles"
            )
            label = f"rustfits extend C={c} rows (K={k_chunks}, {tag})"
            rows_out.append((label, run_build("extend_rf", c, args)))

        cols_spec = [
            ("regime", 46, "l"),
            ("build", 11, "r"),
            ("peak RSS", 11, "r"),
            ("vs rf write-once", 26, "l"),
        ]
        print()
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
        print(header)
        print("-" * len(header))
        title = (
            f"Compressed 2-D image extend / RSS "
            f"({rows:,} x {cols:,} f4, GZIP_2 "
            f"tile=({args.tile_rows}, {cols}))"
        )
        for name, (t, r) in rows_out:
            if name == "rustfits write-once":
                note = "(ref)"
            elif name.startswith("rustfits"):
                rss_ratio = ref_r / r
                if rss_ratio >= 1.0:
                    rss_note = f"{rss_ratio:.1f}x less RAM"
                else:
                    rss_note = f"{1 / rss_ratio:.1f}x more RAM"
                note = f"{t / ref_t:.2f}x time, {rss_note}"
            else:
                note = ""
            cells = [
                (name, 46, "l"),
                (h.fmt_time(t), 11, "r"),
                (f"{r / 1024:,.0f} MB", 11, "r"),
                (note, 26, "l"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))
            h.emit_record(
                {
                    "kind": "rss",
                    "suite": title,
                    "op": name,
                    "build_s": t,
                    "peak_rss_mb": r / 1024,
                    "ref_build_s": ref_t,
                    "ref_peak_rss_mb": ref_r / 1024,
                    "nbytes": nbytes,
                }
            )


if __name__ == "__main__":
    main()
