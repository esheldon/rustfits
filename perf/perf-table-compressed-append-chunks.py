#!/usr/bin/env python
"""
ZTABLE small-chunk append cost: characterize the re-encode tax.

When ``chunk_rows < ZTILELEN``, every ``hdu.append(chunk)`` on a
compressed table decompresses the current partial trailing tile,
merges the new rows, and re-encodes the merged tile back to the
heap.  The cost per append scales with the CURRENT size of the
trailing tile, not the new chunk -- so a streaming pipeline that
appends 100-row chunks into a 10 000-row tile pays the re-encode
of a tile that grows linearly with each append.

``perf-table-append.py`` already shows this is real (14× slower
than write-once on a smoke run at chunk << ZTILELEN).  This bench
characterizes the cost surface as a function of
``chunk_rows / ZTILELEN``, with realistic and magnified regimes,
plus an uncompressed BINTABLE baseline (no re-encode tax there)
so the per-tile re-encode share is isolated from generic per-call
overhead.

Two regimes:

* **Realistic streaming**: ``N=100 000`` rows, ``ZTILELEN=10 000``
  (≈ cfitsio's default for a small catalog), chunk sweep
  ``100 / 1000 / 5000 / 10000 / 20000``.  Maps to "per-frame
  source catalog harvested into one ZTABLE" — chunk = sources per
  frame, ZTILELEN = several frames worth.

* **Magnified**: ``N=10 000``, ``ZTILELEN=100``, chunk sweep
  ``1 / 10 / 50 / 100 / 500``.  Same shape but with a small
  tile size so the per-tile re-encode dominates and the cost
  curve is unambiguous.

For each regime: write-once is the floor (one encode per tile);
uncompressed append (same N, chunks) is a per-call-overhead floor
(no encode at all); ZTABLE append rows show how the re-encode tax
scales with ``chunk_rows / ZTILELEN``.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-compressed-append-chunks.py
    python perf/perf-table-compressed-append-chunks.py --realistic-only

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import rustfits

import _harness as h


# Tiny 4-column schema: keeps schema cost negligible so per-tile
# encode dominates the cost curve.  All float so compression is
# real (random data, ~85% compressed by GZIP_1) and no column
# kind dominates the per-tile work.
DTYPE = np.dtype([("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4")])


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    data = np.empty(n, dtype=DTYPE)
    for name in DTYPE.names:
        data[name] = rng.standard_normal(n).astype("f4")
    return data


def time_build(mode, n, chunk, ztilelen, compress):
    """
    Build one fixture, return wall seconds.  GC paused during the
    timed window.  ``mode`` is ``"write_once"`` or ``"append"``.
    """
    data = make_data(n)
    fname = h.fresh_path(f"ztappend-{mode}-{chunk}-{ztilelen}")
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        if mode == "write_once":
            with rustfits.FITS(fname, "w+") as f:
                kw = dict(compress=True, ztilelen=ztilelen) if compress else {}
                f.write_table(data, **kw)
        elif mode == "append":
            with rustfits.FITS(fname, "w+") as f:
                kw = dict(compress=True, ztilelen=ztilelen) if compress else {}
                f.create_table_hdu(DTYPE, nrows=0, **kw)
                hdu = f[1]
                done = 0
                while done < n:
                    c = min(chunk, n - done)
                    hdu.append(data[done : done + c])
                    done += c
        else:
            raise ValueError(mode)
        return time.perf_counter() - t0
    finally:
        gc.enable()


def median_of(fn, repeat):
    samples = sorted(fn() for _ in range(repeat))
    return samples[len(samples) // 2]


def run_regime(label, n, ztilelen, chunks, repeat):
    """
    For one (n, ztilelen) regime, time write-once + append @ each
    chunk size for both ZTABLE (compressed) and uncompressed
    BINTABLE.  Print a single table that lets the reader read off
    the re-encode tax (ZTABLE append) vs the per-call-overhead
    floor (uncompressed append) vs the write-once reference.
    """
    title = f"{label} — N={n:,} rows, ZTILELEN={ztilelen}"
    print()
    print(title)
    print("-" * len(title))

    # Reference: write-once for each storage mode.
    ref = {}
    for compress, key in [(True, "ZTABLE"), (False, "uncomp")]:
        t = median_of(
            lambda c=compress: time_build("write_once", n, 0, ztilelen, c),
            repeat,
        )
        ref[key] = t

    cols_spec = [
        ("storage", 10, "l"),
        ("op", 18, "l"),
        ("chunk", 8, "r"),
        ("chunks", 8, "r"),
        ("chunk/tile", 11, "r"),
        ("total t", 11, "r"),
        ("vs write-once", 16, "r"),
        ("per-append", 12, "r"),
    ]
    header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
    print(header)
    print("-" * len(header))

    # Build display rows: write-once references then per-chunk
    # appends, with ZTABLE and uncompressed paired so the eye sees
    # the re-encode tax in the ratio column.
    for compress, key in [(True, "ZTABLE"), (False, "uncomp")]:
        cells = [
            (key, 10, "l"),
            ("write_once", 18, "l"),
            ("-", 8, "r"),
            ("-", 8, "r"),
            ("-", 11, "r"),
            (h.fmt_time(ref[key]), 11, "r"),
            ("1.00× (ref)", 16, "r"),
            ("-", 12, "r"),
        ]
        print("  ".join(h._cell(x, w, a) for x, w, a in cells))

    for chunk in chunks:
        n_appends = -(-n // chunk)  # ceil
        chunk_ratio = chunk / ztilelen
        for compress, key in [(True, "ZTABLE"), (False, "uncomp")]:
            t = median_of(
                lambda c=compress, ch=chunk: time_build(
                    "append", n, ch, ztilelen, c
                ),
                repeat,
            )
            ratio = t / ref[key]
            per_app_ms = t / n_appends * 1000
            cells = [
                (key, 10, "l"),
                (f"append C={chunk}", 18, "l"),
                (f"{chunk:,}", 8, "r"),
                (f"{n_appends:,}", 8, "r"),
                (f"{chunk_ratio:.2g}", 11, "r"),
                (h.fmt_time(t), 11, "r"),
                (f"{ratio:.2f}×", 16, "r"),
                (f"{per_app_ms:.2f} ms", 12, "r"),
            ]
            print("  ".join(h._cell(x, w, a) for x, w, a in cells))
            h.emit_record(
                {
                    "kind": "self_comparison",
                    "suite": title,
                    "op": f"{key} append C={chunk}",
                    "storage": key,
                    "chunk": chunk,
                    "n_appends": n_appends,
                    "chunk_per_tile": chunk_ratio,
                    "total_s": t,
                    "ref_writeonce_s": ref[key],
                    "ratio_to_writeonce": ratio,
                }
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--realistic-only",
        action="store_true",
        help="skip the magnified regime",
    )
    ap.add_argument(
        "--magnified-only",
        action="store_true",
        help="skip the realistic regime",
    )
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    print("ZTABLE small-chunk append cost — re-encode tax characterization")
    h.print_env()

    with h.scratch():
        if not args.magnified_only:
            run_regime(
                "Realistic streaming",
                n=100_000,
                ztilelen=10_000,
                chunks=[100, 1000, 5000, 10_000, 20_000],
                repeat=args.repeat,
            )
        if not args.realistic_only:
            run_regime(
                "Magnified (small tile)",
                n=10_000,
                ztilelen=100,
                chunks=[1, 10, 50, 100, 500],
                repeat=args.repeat,
            )


if __name__ == "__main__":
    main()
