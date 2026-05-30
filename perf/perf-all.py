#!/usr/bin/env python
"""
Run every perf-*.py script and produce two consolidated summary tables.

Sequential by design — running benches in parallel would skew the
measurements through CPU/memory/I/O contention.  Each child script
runs in its own subprocess with ``$PERF_JSON`` pointing at a shared
JSON-lines file the harness appends records to; this script collects
them at the end and formats:

* **Cross-tool table** — every ``"kind": "cross_tool"`` record (the
  vast majority — rustfits vs fitsio for image / table / compressed
  read+write).  Sorted by per-script grouping for readability.
* **Self-comparison / RSS table** — every ``"kind": "self_comparison"``
  (ZTABLE rustfits-compressed vs rustfits-uncompressed) and ``"rss"``
  record (extend-build wall + peak resident memory).

REQUIRES a release build: ``maturin develop --release``.  Each child
inherits the parent's working directory and writes scratch files into
CWD as ``perf-tmp-*`` (cleaned up by the child's own ``h.scratch()``).

Run::

    python perf/perf-all.py                  # everything (~5 min)
    python perf/perf-all.py --skip extend    # skip RSS extend scripts
    python perf/perf-all.py --only read      # only read benches
    python perf/perf-all.py --list           # list the scripts found

The full run takes a few minutes; pass ``--only <substr>`` to filter
when iterating.  Set ``PERF_KEEP=1`` to retain scratch files for
inspection.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def discover_scripts() -> list[str]:
    """Return the list of perf-*.py scripts, excluding this runner."""
    here = os.path.basename(__file__)
    scripts = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(SCRIPT_DIR, "perf-*.py"))
        if os.path.basename(p) != here
    )
    return scripts


def filter_scripts(
    scripts: list[str], only: list[str], skip: list[str]
) -> list[str]:
    out = []
    for s in scripts:
        if only and not any(o in s for o in only):
            continue
        if skip and any(k in s for k in skip):
            continue
        out.append(s)
    return out


def run_one(
    script: str, json_path: str, env_extra: dict, quiet: bool
) -> tuple[int, float]:
    """
    Run ``script`` in a subprocess with PERF_JSON set.  Returns
    (returncode, wall_seconds).  Prints the child's stdout/stderr
    unless quiet (in which case only stderr on failure).
    """
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)]
    env = os.environ.copy()
    env["PERF_JSON"] = json_path
    env["PERF_SCRIPT"] = script
    env.update(env_extra)
    t0 = time.perf_counter()
    if quiet:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
    else:
        proc = subprocess.run(cmd, env=env)
    return proc.returncode, time.perf_counter() - t0


def load_records(json_path: str) -> list[dict]:
    """Read JSON-lines from ``json_path``; return list of records."""
    out: list[dict] = []
    if not os.path.exists(json_path):
        return out
    with open(json_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------- formatting ----------


def fmt_time(s: float | None) -> str:
    if s is None:
        return "-"
    if s < 1e-3:
        return f"{s * 1e6:.1f} us"
    if s < 1.0:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def fmt_rate(nbytes: int | None, seconds: float | None) -> str:
    if not nbytes or not seconds or seconds <= 0:
        return "-"
    return f"{nbytes / 1e6 / seconds:,.0f} MB/s"


def print_cross_tool(records: list[dict]) -> None:
    rows = [r for r in records if r.get("kind") == "cross_tool"]
    if not rows:
        print("\n(no cross-tool records)")
        return

    cols = [
        ("script", 38, "l"),
        ("operation", 36, "l"),
        ("rustfits", 12, "r"),
        ("fitsio", 12, "r"),
        ("vs fitsio", 10, "r"),
        ("rustfits rate", 14, "r"),
    ]
    title = "=== Cross-tool comparisons (rustfits vs fitsio) ==="
    print()
    print(title)
    header = "  ".join(_cell(c, w, a) for c, w, a in cols)
    print(header)
    print("-" * len(header))

    # Sort by script then operation for grouping
    rows.sort(key=lambda r: (r.get("script", ""), r.get("op", "")))
    last_script: str | None = None
    n_wins = n_par = n_loss = 0
    for r in rows:
        script = _short_script(r.get("script", ""))
        if script != last_script and last_script is not None:
            # Blank separator between scripts (visual grouping)
            pass
        last_script = script
        rf = r.get("rustfits_s")
        fi = r.get("fitsio_s")
        ratio_str = "-"
        marker = ""
        if rf and fi and rf > 0:
            ratio = fi / rf
            if ratio >= 1.10:
                marker = "FAST"
                n_wins += 1
            elif ratio >= 0.95:
                marker = "~par"
                n_par += 1
            else:
                marker = "SLOW"
                n_loss += 1
            ratio_str = f"{ratio:.2f}x {marker}"
        cells = [
            (script, 38, "l"),
            (r.get("op", "")[:36], 36, "l"),
            (fmt_time(rf), 12, "r"),
            (fmt_time(fi), 12, "r"),
            (ratio_str, 10, "r"),
            (fmt_rate(r.get("nbytes", 0), rf), 14, "r"),
        ]
        print("  ".join(_cell(x, w, a) for x, w, a in cells))

    total = n_wins + n_par + n_loss
    if total:
        print()
        print(
            f"  {n_wins}/{total} faster (>=1.10x), "
            f"{n_par}/{total} ~par (0.95-1.10x), "
            f"{n_loss}/{total} slower (<0.95x)"
        )


def print_self_and_rss(records: list[dict]) -> None:
    self_rows = [r for r in records if r.get("kind") == "self_comparison"]
    rss_rows = [r for r in records if r.get("kind") == "rss"]
    if not self_rows and not rss_rows:
        print("\n(no internal/self-comparison records)")
        return

    if self_rows:
        cols = [
            ("script", 38, "l"),
            ("operation", 28, "l"),
            ("primary", 16, "l"),
            ("primary t", 12, "r"),
            ("secondary", 16, "l"),
            ("secondary t", 12, "r"),
            ("ratio", 10, "r"),
        ]
        title = "=== Self-comparisons (rustfits vs rustfits) ==="
        print()
        print(title)
        header = "  ".join(_cell(c, w, a) for c, w, a in cols)
        print(header)
        print("-" * len(header))
        self_rows.sort(key=lambda r: (r.get("suite", ""), r.get("op", "")))
        for r in self_rows:
            ps = r.get("primary_s")
            ss = r.get("secondary_s")
            ratio_str = "-"
            if ps and ss and ss > 0:
                ratio_str = f"{ps / ss:.2f}x"
            cells = [
                (
                    _short_script(r.get("script") or r.get("suite", "")),
                    38,
                    "l",
                ),
                (r.get("op", "")[:28], 28, "l"),
                (r.get("primary_label", "")[:16], 16, "l"),
                (fmt_time(ps), 12, "r"),
                (r.get("secondary_label", "")[:16], 16, "l"),
                (fmt_time(ss), 12, "r"),
                (ratio_str, 10, "r"),
            ]
            print("  ".join(_cell(x, w, a) for x, w, a in cells))

    if rss_rows:
        cols = [
            ("script", 50, "l"),
            ("regime", 38, "l"),
            ("build", 12, "r"),
            ("peak RSS", 12, "r"),
        ]
        title = "=== RSS / build (extend benches; rustfits self) ==="
        print()
        print(title)
        header = "  ".join(_cell(c, w, a) for c, w, a in cols)
        print(header)
        print("-" * len(header))
        rss_rows.sort(key=lambda r: (r.get("suite", ""), r.get("op", "")))
        last_suite = None
        for r in rss_rows:
            if r.get("suite") != last_suite:
                last_suite = r.get("suite")
            cells = [
                (_short_script(r.get("script") or last_suite or ""), 50, "l"),
                (r.get("op", "")[:38], 38, "l"),
                (fmt_time(r.get("build_s")), 12, "r"),
                (f"{r.get('peak_rss_mb', 0):,.0f} MB", 12, "r"),
            ]
            print("  ".join(_cell(x, w, a) for x, w, a in cells))


def _short_script(s: str) -> str:
    if s.endswith(".py"):
        s = s[:-3]
    if s.startswith("perf-"):
        s = s[5:]
    return s[:38]


def _cell(text: str, width: int, align: str) -> str:
    text = str(text)
    return text.ljust(width) if align == "l" else text.rjust(width)


# ---------- main ----------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only scripts whose name contains this substring "
        "(repeatable)",
    )
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        help="skip scripts whose name contains this substring (repeatable)",
    )
    ap.add_argument("--list", action="store_true", help="list scripts + exit")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress child stdout/stderr (still shown on failure)",
    )
    ap.add_argument(
        "--json-out",
        help="write the aggregated JSON-lines file to this path "
        "(default: tmp, deleted after)",
    )
    args = ap.parse_args()

    scripts = filter_scripts(discover_scripts(), args.only, args.skip)
    if args.list:
        for s in scripts:
            print(s)
        return 0
    if not scripts:
        print("no scripts match filter", file=sys.stderr)
        return 2

    json_path = args.json_out
    tmp_handle = None
    if not json_path:
        tmp_handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        )
        json_path = tmp_handle.name
        tmp_handle.close()
    # Start fresh
    open(json_path, "w").close()

    failures: list[tuple[str, int]] = []
    overall_start = time.perf_counter()
    for s in scripts:
        sys.stdout.write(f"[{s}] ... ")
        sys.stdout.flush()
        rc, wall = run_one(s, json_path, {}, args.quiet)
        sys.stdout.write(f"{wall:.1f}s")
        if rc != 0:
            sys.stdout.write(f"  (exit {rc} FAIL)")
            failures.append((s, rc))
        sys.stdout.write("\n")

    records = load_records(json_path)
    if tmp_handle:
        try:
            os.remove(json_path)
        except OSError:
            pass

    print()
    print(
        f"Ran {len(scripts)} script(s) in "
        f"{time.perf_counter() - overall_start:.1f}s, "
        f"collected {len(records)} record(s); "
        f"{len(failures)} failure(s)."
    )
    if failures:
        for name, rc in failures:
            print(f"  FAIL: {name} (exit {rc})")

    print_cross_tool(records)
    print_self_and_rss(records)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
