"""
Shared harness for the perf-*.py performance scripts.

These scripts are NOT pytest tests: pytest only collects ``test_*.py``
/ ``*_test.py`` by default, and the ``perf-`` filename prefix keeps
them out of collection.  They are meant to be run directly::

    python perf/perf-compressed-image-read.py

Each script writes scratch FITS files into the current directory (all
named ``perf-tmp-*``) and removes them on exit.  Set ``PERF_KEEP=1`` to
keep them for inspection -- the files are large by design.

This module is importable from the hyphen-named scripts because Python
puts the script's own directory on ``sys.path[0]`` when it runs a file
path, so ``import _harness`` resolves to this file.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import fitsio  # noqa: F401

    HAVE_FITSIO = True
except Exception:
    HAVE_FITSIO = False

try:
    import astropy.io.fits as _apfits  # noqa: F401

    HAVE_ASTROPY = True
except Exception:
    HAVE_ASTROPY = False

# Every scratch file we create in CWD starts with this, so cleanup is a
# simple glob sweep and stray leftovers are easy to spot.
PREFIX = "perf-tmp-"


def path(name: str) -> str:
    """
    Return a CWD-relative scratch path carrying the perf prefix.
    """
    return PREFIX + name


_fresh_counter: int = 0


def fresh_path(stem: str) -> str:
    """
    Return a fresh CWD-relative scratch path carrying the perf prefix.

    Each call returns a unique path (monotonically incremented
    counter); the ``scratch()`` cleanup catches them all.  Use this
    in a write-bench closure so each timed iteration writes a fresh
    file rather than overwriting the same one, which avoids a kernel
    page-cache artifact that penalizes overwrites of large files.
    """
    global _fresh_counter
    _fresh_counter += 1
    return f"{PREFIX}{stem}-{_fresh_counter}.fits"


@contextmanager
def scratch():
    """
    Remove every ``perf-tmp-*`` file in CWD on exit.

    Wrap a script body in this so generated fixtures are cleaned up
    even on error.  Honors ``PERF_KEEP=1`` to retain them.
    """
    try:
        yield
    finally:
        if not os.environ.get("PERF_KEEP"):
            for p in glob.glob(PREFIX + "*"):
                try:
                    os.remove(p)
                except OSError:
                    pass


def env_int(name: str, default: int) -> int:
    """
    Read an int from the environment, falling back to ``default``.
    """
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass
class Timing:
    """
    Timing samples in seconds for one operation, sorted ascending.
    """

    samples: list

    @property
    def best(self) -> float:
        return self.samples[0]

    @property
    def median(self) -> float:
        s = self.samples
        n = len(s)
        mid = n // 2
        if n % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples)

    def stat(self, which: str) -> float:
        return getattr(self, which)


def timeit(fn: Callable[[], object], *, repeat: int = 5, warmup: int = 1):
    """
    Time ``fn`` ``repeat`` times after ``warmup`` untimed calls.

    GC is paused around each timed call to cut jitter.  Returns a
    Timing whose samples are sorted ascending.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
        finally:
            gc.enable()
        samples.append(t1 - t0)
    samples.sort()
    return Timing(samples)


@dataclass
class Result:
    """
    One row of a perf table: the rustfits timing plus any comparisons.
    """

    op: str
    rustfits: Timing
    fitsio: Optional[Timing] = None
    astropy: Optional[Timing] = None
    nbytes: int = 0
    target_s: Optional[float] = None
    note: str = ""


def bench(
    op: str,
    run_rustfits: Callable[[], object],
    *,
    run_fitsio: Optional[Callable[[], object]] = None,
    run_astropy: Optional[Callable[[], object]] = None,
    nbytes: int = 0,
    repeat: int = 5,
    warmup: int = 1,
    target_s: Optional[float] = None,
    note: str = "",
) -> Result:
    """
    Time the rustfits callable plus any comparison callables.

    Correctness is the caller's job: check that the backends agree
    once, before calling ``bench``, so a fast-but-wrong path can never
    post a good number.
    """
    rf = timeit(run_rustfits, repeat=repeat, warmup=warmup)
    fi = (
        timeit(run_fitsio, repeat=repeat, warmup=warmup)
        if run_fitsio is not None
        else None
    )
    ap = (
        timeit(run_astropy, repeat=repeat, warmup=warmup)
        if run_astropy is not None
        else None
    )
    return Result(
        op,
        rf,
        fitsio=fi,
        astropy=ap,
        nbytes=nbytes,
        target_s=target_s,
        note=note,
    )


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_rate(nbytes: int, seconds: float) -> str:
    if not nbytes or seconds <= 0:
        return "-"
    return f"{nbytes / 1e6 / seconds:,.0f} MB/s"


def _cell(text: str, width: int, align: str) -> str:
    text = str(text)
    return text.ljust(width) if align == "l" else text.rjust(width)


def _speedup(result: Result, stat: str) -> str:
    if result.fitsio is None:
        return "-"
    rf = result.rustfits.stat(stat)
    fi = result.fitsio.stat(stat)
    if rf <= 0:
        return "-"
    ratio = fi / rf
    tag = "" if ratio >= 1.0 else "  SLOW"
    return f"{ratio:.2f}x{tag}"


def _note(result: Result, stat: str) -> str:
    if result.note:
        return result.note
    if result.target_s is not None:
        rf = result.rustfits.stat(stat)
        verdict = "OK" if rf <= result.target_s else "OVER"
        return f"target {fmt_time(result.target_s)}: {verdict}"
    return ""


def print_env() -> None:
    """
    Print backend versions so a result line is reproducible.
    """
    import numpy as np
    import rustfits

    parts = [
        f"rustfits {getattr(rustfits, '__version__', '?')}",
        f"numpy {np.__version__}",
    ]
    if HAVE_FITSIO:
        parts.append(f"fitsio {fitsio.__version__}")
    if HAVE_ASTROPY:
        import astropy

        parts.append(f"astropy {astropy.__version__}")
    print("env: " + ", ".join(parts))


def report(title: str, results, *, stat: str = "median") -> None:
    """
    Print a comparison table.  ``stat`` is one of median/best/mean.
    """
    show_fi = any(r.fitsio for r in results)
    show_ap = any(r.astropy for r in results)
    show_note = any(r.note or r.target_s is not None for r in results)

    cols = [("operation", 30, "l"), ("rustfits", 12, "r")]
    if show_fi:
        cols.append(("fitsio", 12, "r"))
    if show_ap:
        cols.append(("astropy", 12, "r"))
    if show_fi:
        cols.append(("vs fitsio", 12, "r"))
    cols.append(("throughput", 13, "r"))
    if show_note:
        cols.append(("note", 26, "l"))

    print()
    print(title + f"   (stat={stat})")
    header = "  ".join(_cell(h, w, a) for h, w, a in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        rf = r.rustfits.stat(stat)
        row = [(r.op, 30, "l"), (fmt_time(rf), 12, "r")]
        if show_fi:
            fi = fmt_time(r.fitsio.stat(stat)) if r.fitsio else "-"
            row.append((fi, 12, "r"))
        if show_ap:
            ap = fmt_time(r.astropy.stat(stat)) if r.astropy else "-"
            row.append((ap, 12, "r"))
        if show_fi:
            row.append((_speedup(r, stat), 12, "r"))
        row.append((fmt_rate(r.nbytes, rf), 13, "r"))
        if show_note:
            row.append((_note(r, stat), 26, "l"))
        print("  ".join(_cell(t, w, a) for t, w, a in row))

    for r in results:
        emit_record(
            {
                "kind": "cross_tool",
                "suite": title,
                "op": r.op,
                "stat": stat,
                "rustfits_s": r.rustfits.stat(stat),
                "fitsio_s": (r.fitsio.stat(stat) if r.fitsio else None),
                "astropy_s": (r.astropy.stat(stat) if r.astropy else None),
                "nbytes": r.nbytes,
                "target_s": r.target_s,
                "note": r.note,
            }
        )


def emit_record(record: dict) -> None:
    """
    Append a free-form record (as JSON line) to ``$PERF_JSON`` if set.

    Used both by ``report()`` automatically (one record per Result)
    and by scripts that bypass ``report()`` -- the RSS extend
    benches and the ZTABLE self-comparisons -- so a runner like
    ``perf-all.py`` can aggregate every script's results uniformly.

    Records SHOULD include a ``"kind"`` field categorizing the
    record (``"cross_tool"`` for vs-fitsio, ``"self_comparison"``
    for ZTABLE-style self-vs-self, ``"rss"`` for the RSS extend
    benches, etc.) so the runner can sort them into the right
    output table.  A ``"script"`` field is auto-filled from the
    ``PERF_SCRIPT`` env var when the runner is active, so callers
    don't need to repeat it.  No-op when ``PERF_JSON`` is not set,
    so scripts work standalone too.
    """
    out = os.environ.get("PERF_JSON")
    if not out:
        return
    if "script" not in record:
        script = os.environ.get("PERF_SCRIPT")
        if script:
            record = {"script": script, **record}
    with open(out, "a") as fh:
        fh.write(json.dumps(record) + "\n")
