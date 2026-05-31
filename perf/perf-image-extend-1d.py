#!/usr/bin/env python
"""
Uncompressed 1-D image EXTEND: bounded-memory incremental build.

The uncompressed analog of the healsparse compressed-extend benchmark.
``ImageHDU.extend`` appends raw rows by growing the data section (a
``set_len`` + write at the end for a last HDU) -- there's no compressed
heap to relocate, so building via K appends is O(N) (linear), NOT the
~quadratic-in-K cost the compressed path has.  So we expect extend time
to be ~flat across chunk sizes here, with the same bounded-memory win.

fitsio can't append to an image either, so this is a rustfits self-
characterization: per-build wall time + peak ``ru_maxrss``, each build in
its own subprocess (this script re-invokes itself with ``--worker``).
Build time excludes synthetic data gen; no fsync.

REQUIRES a release build: ``maturin develop --release``.

Run (heavy: builds ~1 GB several times; lower --n to iterate)::

    python perf/perf-image-extend-1d.py
    python perf/perf-image-extend-1d.py --n 67108864

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


def build_worker(mode, n, chunk):
    """
    One build per ``mode``, timing only the write/extend calls; print
    ``<build_seconds> <peak_rss_kb>``.  Runs in a subprocess for a clean
    per-build ru_maxrss.
    """
    fname = h.path("uextbuild.fits")
    if mode == "extend_rf":
        t = 0.0
        done = 0
        i = 0
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("f8", (0,))
            hdu = f[0]
            while done < n:
                c = min(chunk, n - done)
                data = np.random.default_rng(i + 1).standard_normal(c)
                t0 = time.perf_counter()
                hdu.extend(data)
                t += time.perf_counter() - t0
                done += c
                i += 1
                del data
    else:
        data = np.random.default_rng(0).standard_normal(n)
        t0 = time.perf_counter()
        if mode == "write_rf":
            with rustfits.FITS(fname, "w+") as f:
                f.write_image(data)
        else:  # write_fi
            import fitsio

            with fitsio.FITS(fname, "rw", clobber=True) as f:
                f.write(data)
        t = time.perf_counter() - t0

    rss_kb = h.vm_hwm_kb()
    try:
        os.remove(fname)
    except OSError:
        pass
    print(f"{t:.6f} {rss_kb}")


def run_build(mode, chunk, args):
    """
    Subprocess the worker ``args.repeat`` times; return
    ``(median_time_s, max_rss_kb)``.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        mode,
        "--n",
        str(args.n),
        "--chunk",
        str(chunk),
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
    """Small in-process check that extend round-trips bit-exact."""
    chunks = [
        np.random.default_rng(i + 1).standard_normal(100_000) for i in range(3)
    ]
    gf = h.path("uextgate.fits")
    with rustfits.FITS(gf, "w+") as f:
        f.create_image_hdu("f8", (0,))
        for ch in chunks:
            f[0].extend(ch)
    with rustfits.FITS(gf) as f:
        np.testing.assert_array_equal(f[0].read(), np.concatenate(chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--n", type=int, default=128 * 1048576)
    ap.add_argument("--chunk", type=int, default=1048576)
    ap.add_argument("--chunks", default="1048576,8388608")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        build_worker(args.worker, args.n, args.chunk)
        return

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    with h.scratch():
        gate(args)

        n = args.n
        chunks = [int(c) for c in args.chunks.split(",")]
        print(f"build N={n:,} f8 ({n * 8 / 1e6:,.0f} MB), uncompressed")
        h.print_env()

        rows = [
            ("fitsio write-once", run_build("write_fi", 0, args)),
            ("rustfits write-once", run_build("write_rf", 0, args)),
        ]
        ref_t, ref_r = rows[1][1]
        for c in chunks:
            label = f"rustfits extend C={c:,} (K={-(-n // c)})"
            rows.append((label, run_build("extend_rf", c, args)))

        cols = [
            ("regime", 34, "l"),
            ("build", 11, "r"),
            ("peak RSS", 11, "r"),
            ("vs rf write-once", 26, "l"),
        ]
        print()
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols)
        print(header)
        print("-" * len(header))
        title = f"Uncompressed 1-D image extend / RSS (N={n:,} f8)"
        for name, (t, r) in rows:
            if name.startswith("rustfits extend"):
                note = f"{t / ref_t:.2f}x time, {ref_r / r:.1f}x less RAM"
            elif name.startswith("rustfits write"):
                note = "(ref)"
            else:
                note = ""
            cells = [
                (name, 34, "l"),
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
                    "nbytes": n * 8,
                }
            )


if __name__ == "__main__":
    main()
