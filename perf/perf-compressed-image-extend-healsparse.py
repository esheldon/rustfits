#!/usr/bin/env python
"""
Compressed 1-D image EXTEND: bounded-memory incremental build.

healsparse today holds the whole map in RAM and writes it once (via
fitsio); it does not append.  rustfits's ``CompressedImageHDU.extend``
enables what that approach can't: build the same map with BOUNDED memory
by appending chunks, never holding the whole array.  So the win here is
peak MEMORY, with build TIME as the "and it's not a speed regression"
check -- not a faster-than-fitsio claim (fitsio can't append).

Regimes, all building the same N-element 1-D GZIP_2 healsparse map:

* fitsio write-once    -- the current approach: O(N) RAM, write once.
* rustfits write-once  -- same shape, O(N) RAM (the reference).
* rustfits extend C=.. -- create empty + append chunks of C: O(C) RAM.
                          A couple of chunk sizes show the time/memory
                          tradeoff (extend relocates the growing
                          compressed heap each call, so fewer/larger
                          chunks build faster).

Peak RSS can't be measured in-process (``ru_maxrss`` is a process-
lifetime high-water mark), so each build runs in its own SUBPROCESS that
reports build-time + ``ru_maxrss``; this script re-invokes itself with
``--worker``.  Build time excludes synthetic data generation (times only
the rustfits/fitsio calls); RSS includes the ~150 MB interpreter+libs
baseline equally across regimes.  GZIP level is matched to cfitsio's
Z_BEST_SPEED (1).  No fsync.

REQUIRES a release build: ``maturin develop --release``.

Run (heavy: builds ~1 GB several times; lower --n to iterate)::

    python perf/perf-compressed-image-extend-healsparse.py
    python perf/perf-compressed-image-extend-healsparse.py --n 67108864

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time

import numpy as np
import rustfits

import _data
import _harness as h

TILE = 1048576


def build_worker(mode, n, chunk, tile, level, run_len, cov, quant):
    """
    Do ONE build per ``mode``, timing only the encode/write, then print
    ``<build_seconds> <peak_rss_kb>``.  Runs in a subprocess so
    ``ru_maxrss`` is a clean per-build high-water mark.
    """
    fname = h.path("extbuild.fits.fz")
    cfg = dict(tile_shape=(tile,), heap_format="Q", level=level)

    if mode == "extend_rf":
        # Generate each chunk on the fly so peak RAM stays at O(chunk).
        t = 0.0
        done = 0
        i = 0
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("f8", (0,), compress=rustfits.Gzip2(**cfg))
            hdu = f[1]
            while done < n:
                c = min(chunk, n - done)
                data = _data.healsparse_array(
                    c, run_len, cov, quant, seed=i + 1
                )
                t0 = time.perf_counter()
                hdu.extend(data)
                t += time.perf_counter() - t0
                done += c
                i += 1
                del data
    else:
        data = _data.healsparse_array(n, run_len, cov, quant, seed=0)
        t0 = time.perf_counter()
        if mode == "write_rf":
            with rustfits.FITS(fname, "w+") as f:
                f.write_image(data, compress=rustfits.Gzip2(**cfg))
        else:  # write_fi
            import fitsio

            with fitsio.FITS(fname, "rw", clobber=True) as f:
                f.write(data, compress="GZIP_2", tile_dims=(tile,), qlevel=0)
        t = time.perf_counter() - t0

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
        "--tile",
        str(args.tile),
        "--level",
        str(args.level),
        "--run-len",
        str(args.run_len),
        "--cov",
        str(args.cov),
        "--quant",
        str(args.quant),
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
    Small in-process check that extend round-trips bit-exact (GZIP is
    lossless), using sub-tile chunks so the partial-last-tile path runs.
    """
    chunks = [
        _data.healsparse_array(
            100_000, args.run_len, args.cov, args.quant, seed=i + 1
        )
        for i in range(3)
    ]
    gf = h.path("extgate.fits.fz")
    with rustfits.FITS(gf, "w+") as f:
        f.create_image_hdu(
            "f8",
            (0,),
            compress=rustfits.Gzip2(
                tile_shape=(args.tile,), heap_format="Q", level=args.level
            ),
        )
        for ch in chunks:
            f[1].extend(ch)
    with rustfits.FITS(gf) as f:
        np.testing.assert_array_equal(f[1].read(), np.concatenate(chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--n", type=int, default=128 * TILE)
    ap.add_argument("--chunk", type=int, default=TILE)
    ap.add_argument("--chunks", default="1048576,8388608")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--run-len", type=int, default=32)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--quant", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        build_worker(
            args.worker,
            args.n,
            args.chunk,
            args.tile,
            args.level,
            args.run_len,
            args.cov,
            args.quant,
        )
        return

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    with h.scratch():
        gate(args)

        n = args.n
        chunks = [int(c) for c in args.chunks.split(",")]
        print(
            f"build N={n:,} f8 ({n * 8 / 1e6:,.0f} MB raw), "
            f"tile={args.tile:,}, gzip level={args.level}"
        )
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


if __name__ == "__main__":
    main()
