#!/usr/bin/env python
"""
One-shot open-and-time of a pre-existing FITS file.

Companion to ``perf-fits-open-many-hdus.py``: build a fixture today
with that script using ``PERF_KEEP=1``, then come back later (after
the OS / GPFS / Lustre metadata cache has actually evicted) and run
this against the preserved fixture to measure the TRUE cold-cache
open cost.  The other script's ``--cold`` mode times a just-written
fixture, which is almost always still in the filesystem client
cache; this script targets the "archive file untouched for weeks"
case that motivates lazy-open mode discussions in the first place.

Defaults to timing rustfits only — the second tool to open the same
file would hit warm cache and produce a meaningless number.  To
compare both tools, either run this twice with two separately
preserved fixtures of the same shape, or use ``--tool both``
understanding that the second tool's number is warm-cache.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-fits-open-one.py /gpfs/some/old/many-hdu.fits
    python perf/perf-fits-open-one.py FILE1 FILE2 FILE3   # bulk
    python perf/perf-fits-open-one.py FILE --tool fitsio
    python perf/perf-fits-open-one.py FILE --tool both    # warm caveat
"""

from __future__ import annotations

import argparse
import time

import rustfits

try:
    import fitsio

    HAVE_FITSIO = True
except ImportError:
    HAVE_FITSIO = False


def time_rustfits(path: str) -> tuple[int, float]:
    """
    Open ``path`` with rustfits, return ``(n_hdus, elapsed_s)``.
    ``len(fits)`` is included in the timed region so that a future
    lazy mode that defers the walk would be measured correctly.
    """
    t0 = time.perf_counter()
    f = rustfits.FITS(path)
    n = len(f)
    elapsed = time.perf_counter() - t0
    f.close()
    return n, elapsed


def time_fitsio(path: str) -> tuple[int, float]:
    """
    Open ``path`` with fitsio, force the lazy HDU walk via
    ``update_hdu_list()``, return ``(n_hdus, elapsed_s)``.
    """
    if not HAVE_FITSIO:
        raise SystemExit("fitsio not installed")
    t0 = time.perf_counter()
    f = fitsio.FITS(path)
    f.update_hdu_list()
    n = len(f)
    elapsed = time.perf_counter() - t0
    f.close()
    return n, elapsed


def fmt_ms(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1e3:,.1f} ms"
    return f"{seconds:.3f} s"


def report(tool: str, path: str, n: int, elapsed: float) -> None:
    per_hdu_us = elapsed / n * 1e6 if n else 0.0
    print(
        f"{tool:9s} {path}: opened {n:,} HDUs in "
        f"{fmt_ms(elapsed)} ({per_hdu_us:.1f} us/HDU)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths",
        nargs="+",
        help="one or more existing FITS file paths to time",
    )
    ap.add_argument(
        "--tool",
        choices=("rustfits", "fitsio", "both"),
        default="rustfits",
        help="which tool to time (default: rustfits; 'both' opens "
        "rustfits first then fitsio -- the second is warm-cache)",
    )
    args = ap.parse_args()

    for path in args.paths:
        if args.tool in ("rustfits", "both"):
            n, t = time_rustfits(path)
            report("rustfits", path, n, t)
        if args.tool in ("fitsio", "both"):
            n, t = time_fitsio(path)
            report("fitsio", path, n, t)


if __name__ == "__main__":
    main()
