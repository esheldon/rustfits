#!/usr/bin/env python
"""
Table append: incremental catalog build, both uncompressed and ZTABLE,
fixed-only vs VLA-bearing.

Building a catalog by N/K calls to ``append(K rows)`` is the natural
pattern for streaming pipelines (per-frame source extraction, per-file
harvest, etc.).  This script measures wall + peak RSS of that pattern
vs the equivalent one-shot ``write_table(N rows)``.

The four variants matter for different reasons:

* **Uncompressed, fixed-only** — pure data-section grow; reference
  for "what does append cost when nothing else is moving."
* **Uncompressed, VLA-bearing** — each append relocates the existing
  heap forward to sit after the new main rows.  Still linear in N,
  but the per-chunk heap-walk is visible.
* **ZTABLE, fixed-only** — every append decompresses + merges into
  the partial last tile then re-encodes (default ZTILELEN ≈ 10 MB
  / row_width ≈ 16 k rows for this catalog), so small chunks pay a
  per-call re-encode cost on the trailing tile.
* **ZTABLE, VLA-bearing** — the merge-tile re-encode plus the
  dual-descriptor heap dance.  Most complex path on the write side.

fitsio appears as a write-once reference on the uncompressed variants
only (its Python API cannot write ZTABLE).  Each build runs in its
own subprocess for a clean per-build ``ru_maxrss``.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-table-append.py
    python perf/perf-table-append.py --nrows 500000
    python perf/perf-table-append.py --chunks 1000,10000,50000

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time

import rustfits

import _data
import _harness as h


VARIANTS = [
    # (key, label, compress, include_vla)
    ("u-fixed", "uncompressed, fixed-only", False, False),
    ("u-vla", "uncompressed, with VLA", False, True),
    ("z-fixed", "ZTABLE, fixed-only", True, False),
    ("z-vla", "ZTABLE, with VLA", True, True),
]

# VLA column names in the catalog schema; excluded for fixed-only.
VLA_NAMES = ("v_str", "v_ustr", "v_f4")


def _excludes(include_vla: bool):
    return () if include_vla else VLA_NAMES


def build_worker(mode, variant_key, n, chunk):
    """
    One build of (mode, variant); print ``<build_seconds> <peak_rss_kb>``.
    Runs in a subprocess for a clean per-build ru_maxrss.
    """
    var = next(v for v in VARIANTS if v[0] == variant_key)
    _, _, compressed, include_vla = var
    data, vd = _data.catalog_arrays(n, exclude=_excludes(include_vla))

    suffix = ".fz" if compressed else ""
    fname = h.fresh_path(f"tappend-{variant_key}-{mode}") + suffix

    if mode == "write_rf":
        t0 = time.perf_counter()
        with rustfits.FITS(fname, "w+") as f:
            kwargs = {"var_dtypes": vd} if vd else {}
            if compressed:
                kwargs["compress"] = True
            f.write_table(data, **kwargs)
        t = time.perf_counter() - t0
    elif mode == "write_fi":
        import fitsio

        t0 = time.perf_counter()
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(data)
        t = time.perf_counter() - t0
    elif mode == "append_rf":
        t = 0.0
        with rustfits.FITS(fname, "w+") as f:
            kwargs = {"var_dtypes": vd} if vd else {}
            if compressed:
                kwargs["compress"] = True
            f.create_table_hdu(data.dtype, nrows=0, **kwargs)
            hdu = f[-1]
            done = 0
            while done < n:
                c = min(chunk, n - done)
                t0 = time.perf_counter()
                hdu.append(data[done : done + c])
                t += time.perf_counter() - t0
                done += c
    else:
        raise ValueError(f"unknown mode: {mode}")

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        os.remove(fname)
    except OSError:
        pass
    print(f"{t:.6f} {rss_kb}")


def run_build(mode, variant_key, chunk, args):
    """
    Subprocess the worker ``args.repeat`` times; return
    ``(median_time_s, max_rss_kb)``.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        mode,
        "--variant",
        variant_key,
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
    """
    Quick in-process round-trip per variant (small N) so a regression
    in either path is caught before the long subprocess matrix runs.
    """
    ng = 1000
    for variant_key, _, compressed, include_vla in VARIANTS:
        data, vd = _data.catalog_arrays(ng, exclude=_excludes(include_vla))
        fname = h.fresh_path(f"tappend-gate-{variant_key}")
        if compressed:
            fname += ".fz"
        with rustfits.FITS(fname, "w+") as f:
            kwargs = {"var_dtypes": vd} if vd else {}
            if compressed:
                kwargs["compress"] = True
            f.create_table_hdu(data.dtype, nrows=0, **kwargs)
            f[-1].append(data[: ng // 2])
            f[-1].append(data[ng // 2 :])
        with rustfits.FITS(fname) as f:
            _data.compare_catalog(f[1].read(), data, vd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--variant")
    ap.add_argument("--n", type=int, default=h.env_int("PERF_NROWS", 100_000))
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--chunks", default="1000,10000")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        build_worker(args.worker, args.variant, args.n, args.chunk)
        return

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")

    with h.scratch():
        gate(args)

        n = args.n
        chunks = [int(c) for c in args.chunks.split(",")]
        # Row width depends on whether VLA fields are in the dtype --
        # the structured dtype includes 8 bytes (1PE-like descriptor
        # slot) per VLA field, so VLA variants are wider on disk than
        # fixed-only.
        dt_full = _data.catalog_dtype()
        dt_fix = _data.catalog_dtype(exclude=VLA_NAMES)
        print(
            f"catalog: N={n:,} rows, fixed-only row={dt_fix.itemsize} B, "
            f"with-VLA row={dt_full.itemsize} B"
        )
        h.print_env()

        cols = [
            ("regime", 38, "l"),
            ("build", 11, "r"),
            ("peak RSS", 11, "r"),
            ("vs rf write-once", 26, "l"),
        ]
        header = "  ".join(h._cell(c, w, a) for c, w, a in cols)

        for variant_key, label, compressed, include_vla in VARIANTS:
            row_width = (dt_full if include_vla else dt_fix).itemsize
            nbytes = n * row_width
            title = f"Table append ({label}, N={n:,})"
            print()
            print(title)
            print(header)
            print("-" * len(header))

            rows = []
            # fitsio write-once is meaningful only on uncompressed --
            # fitsio's Python API can't write ZTABLE.
            if not compressed:
                rows.append(
                    (
                        "fitsio write-once",
                        run_build("write_fi", variant_key, 0, args),
                    )
                )
            rows.append(
                (
                    "rustfits write-once",
                    run_build("write_rf", variant_key, 0, args),
                )
            )
            ref_t, ref_r = rows[-1][1]
            for c in chunks:
                k_chunks = -(-n // c)
                op_label = f"rustfits append C={c:,} (K={k_chunks})"
                rows.append(
                    (op_label, run_build("append_rf", variant_key, c, args))
                )

            for name, (t, r) in rows:
                if name.startswith("rustfits append"):
                    note = f"{t / ref_t:.2f}x time, {ref_r / r:.1f}x less RAM"
                elif name == "rustfits write-once":
                    note = "(ref)"
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
