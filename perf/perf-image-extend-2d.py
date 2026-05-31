#!/usr/bin/env python
"""
Uncompressed 2-D image EXTEND: bounded-memory mosaic build.

The 2-D counterpart to ``perf-image-extend-1d.py``.  The natural
2-D workflow is mosaic / strip building: append per-detector
frames (or per-night strips) to a growing image instead of
holding the whole mosaic in RAM and writing it once.  Both
rustfits' ``ImageHDU.extend`` and fitsio's ``f[0].write(strip,
start=(row, 0))`` grow the slowest-varying axis -- same
primitive, two APIs.

Four modes per build, two chunk sizes for the extend variants:

* ``write_fi`` -- fitsio write-once (whole array in RAM).
* ``write_rf`` -- rustfits write-once (whole array in RAM); the
  reference for the ``vs rf write-once`` column.
* ``extend_rf`` -- rustfits ``create_image_hdu(dtype, (0, cols))``
  + ``hdu.extend(chunk)`` per chunk.
* ``extend_fi`` -- fitsio ``f.write(data[:1])`` (seed one row,
  untimed) + ``f[0].write(chunk, start=(row, 0))`` per chunk.
  fitsio can't ``create_image_hdu(dims=(0, cols))`` -- the HDU
  doesn't materialize -- so we seed with one row.  Bias is 1
  row out of N, invisible.

Each build runs in its own subprocess for a clean per-build
``ru_maxrss``; data generation runs outside the timed region.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-image-extend-2d.py
    python perf/perf-image-extend-2d.py --rows 40000 --cols 4000
    python perf/perf-image-extend-2d.py --chunks 50,500,2000

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


def build_worker(mode, rows, cols, chunk_rows):
    """
    One build per ``mode``, timing only the write/extend calls;
    print ``<build_seconds> <peak_rss_kb>``.  Runs in a subprocess
    for a clean per-build ru_maxrss.  Per-chunk frames are
    generated on the fly for extend modes so peak RAM stays at
    O(chunk_rows * cols) plus the FITS handle's own pages.
    """
    fname = h.path("uext2dbuild.fits")
    if mode == "extend_rf":
        t = 0.0
        done = 0
        i = 0
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("f4", (0, cols))
            hdu = f[0]
            while done < rows:
                c = min(chunk_rows, rows - done)
                data = _data.des_array(c, cols, zero_frac=0.05, seed=i + 1)
                t0 = time.perf_counter()
                hdu.extend(data)
                t += time.perf_counter() - t0
                done += c
                i += 1
                del data
    elif mode == "extend_fi":
        import fitsio

        t = 0.0
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            # Seed one row -- fitsio can't create_image_hdu with
            # a zero-row axis (the HDU doesn't materialize).
            f.write(_data.des_array(1, cols, zero_frac=0.05, seed=0))
            hdu = f[0]
            done = 1
            i = 0
            while done < rows:
                c = min(chunk_rows, rows - done)
                data = _data.des_array(c, cols, zero_frac=0.05, seed=i + 1)
                t0 = time.perf_counter()
                hdu.write(data, start=(done, 0))
                t += time.perf_counter() - t0
                done += c
                i += 1
                del data
    else:
        data = _data.des_array(rows, cols, zero_frac=0.05, seed=0)
        t0 = time.perf_counter()
        if mode == "write_rf":
            with rustfits.FITS(fname, "w+") as f:
                f.write_image(data)
        elif mode == "write_fi":
            import fitsio

            with fitsio.FITS(fname, "rw", clobber=True) as f:
                f.write(data)
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
    Small in-process check that both tools' extend round-trip
    bit-exact against write-once.
    """
    frames = [
        _data.des_array(50, args.cols, zero_frac=0.05, seed=i + 1)
        for i in range(3)
    ]
    full = np.concatenate(frames)
    # rustfits
    gf = h.path("uext2dgate-rf.fits")
    with rustfits.FITS(gf, "w+") as f:
        f.create_image_hdu("f4", (0, args.cols))
        for fr in frames:
            f[0].extend(fr)
    with rustfits.FITS(gf) as f:
        np.testing.assert_array_equal(f[0].read(), full)
    # fitsio
    import fitsio

    gf = h.path("uext2dgate-fi.fits")
    with fitsio.FITS(gf, "rw", clobber=True) as f:
        f.write(frames[0][:1])
        f[0].write(frames[0][1:], start=(1, 0))
        f[0].write(frames[1], start=(frames[0].shape[0], 0))
        f[0].write(
            frames[2],
            start=(frames[0].shape[0] + frames[1].shape[0], 0),
        )
    with fitsio.FITS(gf) as f:
        np.testing.assert_array_equal(f[0].read(), full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--cols", type=int, default=4_000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--chunks", default="100,1000")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        build_worker(args.worker, args.rows, args.cols, args.chunk)
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
            f"build ({rows:,} x {cols:,}) f4 ({nbytes / 1e6:,.0f} MB), "
            f"uncompressed"
        )
        h.print_env()

        rows_out = [
            ("fitsio write-once", run_build("write_fi", 0, args)),
            ("rustfits write-once", run_build("write_rf", 0, args)),
        ]
        ref_t, ref_r = rows_out[1][1]
        for c in chunks:
            k_chunks = -(-rows // c)
            rows_out.append(
                (
                    f"rustfits extend C={c} rows (K={k_chunks})",
                    run_build("extend_rf", c, args),
                )
            )
            rows_out.append(
                (
                    f"fitsio extend C={c} rows (K={k_chunks})",
                    run_build("extend_fi", c, args),
                )
            )

        cols_spec = [
            ("regime", 38, "l"),
            ("build", 11, "r"),
            ("peak RSS", 11, "r"),
            ("vs rf write-once", 26, "l"),
        ]
        print()
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
        print(header)
        print("-" * len(header))
        title = f"Uncompressed 2-D image extend / RSS ({rows:,} x {cols:,} f4)"
        for name, (t, r) in rows_out:
            if name == "rustfits write-once":
                note = "(ref)"
            elif name.startswith("rustfits") or name.startswith("fitsio"):
                rss_ratio = ref_r / r
                if rss_ratio >= 1.0:
                    rss_note = f"{rss_ratio:.1f}x less RAM"
                else:
                    rss_note = f"{1 / rss_ratio:.1f}x more RAM"
                note = f"{t / ref_t:.2f}x time, {rss_note}"
            else:
                note = ""
            cells = [
                (name, 38, "l"),
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
