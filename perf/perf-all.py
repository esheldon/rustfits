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


# ---------- RST writer ----------


# Scripts that have a hand-written narrative section in the docs
# (performance.rst).  ``write_rst`` excludes their records from the
# auto-generated tables so the same numbers don't appear twice on
# the page.  Records still flow through the JSON-lines stream for
# stdout reporting and for any consumer that aggregates everything.
SCRIPTS_WITH_NARRATIVE: set[str] = {
    "perf-table-append.py",
}


# Per-script title + one-line subtitle for the docs tables.  The
# subtitle supplies the file-layout context (dtype, shape, size)
# that the per-row "operation" labels lack on their own.  Order in
# this dict drives display order in the generated RST.
SCRIPT_GROUPS: dict[str, tuple[str, str]] = {
    # Uncompressed images
    "perf-image-read-1d.py": (
        "Uncompressed 1-D image read",
        "1-D ``f8`` image, 64 M pixels (~512 MB on disk).",
    ),
    "perf-image-read-2d.py": (
        "Uncompressed 2-D image read",
        "2-D ``f4`` image, 4000×4000 pixels (64 MB on disk).",
    ),
    "perf-image-write-1d.py": (
        "Uncompressed 1-D image write",
        "1-D ``f8`` image, 64 M pixels (~512 MB on disk); fresh file "
        "per timed iteration.",
    ),
    "perf-image-write-2d.py": (
        "Uncompressed 2-D image write",
        "2-D ``f4`` image, 4000×4000 pixels (64 MB on disk); fresh "
        "file per timed iteration.",
    ),
    # Compressed images — 1-D
    "perf-compressed-image-read-healsparse.py": (
        "Compressed 1-D image read (GZIP_2, healsparse-like)",
        "1-D ``f8`` healsparse map with long quantized runs; "
        "~537 MB raw, ~26 MB compressed (GZIP_2, 1 MiB tiles).",
    ),
    "perf-compressed-image-read-random.py": (
        "Compressed 1-D image read (GZIP_2, random data)",
        "1-D ``f8`` array of pure noise (worst case for any codec); "
        "GZIP_2 with 1 MiB tiles, ~512 MB raw / ~485 MB compressed.",
    ),
    "perf-compressed-image-write-healsparse.py": (
        "Compressed 1-D image write (GZIP_2, healsparse-like)",
        "Same input as the healsparse read bench, lossless GZIP_2 "
        "encode (level matched to cfitsio's ``Z_BEST_SPEED=1``).",
    ),
    # Compressed images — 2-D
    "perf-compressed-image-read-des.py": (
        "Compressed 2-D image read (RICE_1 + dither2, DES-like)",
        "2-D ``f4`` image with ~5% masked zeros; RICE_1 + "
        "``Quantize(level=16, method='dither2')``, 100×100 tiles, "
        "4000×4000 pixels.  Stamps = 1000 random 32×32 windows; "
        "whole = all tiles in disk order (band walk).",
    ),
    "perf-compressed-2d-isolation.py": (
        "Compressed 2-D image read — codec isolation sweep",
        "Same shape as DES, but varying ONE factor per row (codec, "
        "quantization, tile size) so the gap from each contributing "
        "factor is visible.  Helps separate \"decode is slow\" "
        "from \"dequant is slow\".",
    ),
    "perf-compressed-image-write-des.py": (
        "Compressed 2-D image write (RICE_1 + dither2, DES-like)",
        "Same input/shape as the DES read bench; RICE_1 + dither2.",
    ),
    # Tables — uncompressed
    "perf-table-read.py": (
        "Uncompressed BINTABLE read",
        "Type-exhaustive 34-column catalog (every scalar type, "
        "1-D + 2-D ``f4``/``f8`` sub-arrays, ``S`` and ``U`` "
        "strings fixed + VLA, an ``f4`` VLA); 500 k rows, ~306 MB.",
    ),
    "perf-table-write.py": (
        "Uncompressed BINTABLE write",
        "Same catalog as the BINTABLE read bench; fresh file per "
        "timed iteration.",
    ),
    # Tables — compressed (ZTABLE; rustfits self-comparison)
    "perf-table-compressed-read.py": (
        "Compressed BINTABLE read (ZTABLE; rustfits self)",
        "Same catalog as the uncompressed BINTABLE bench.  fitsio's "
        "Python API does not decompress ZTABLE, so this is a rustfits "
        "self-comparison (compressed vs uncompressed timing).",
    ),
    "perf-table-compressed-write.py": (
        "Compressed BINTABLE write (ZTABLE; rustfits self)",
        "Same catalog.  Self-comparison (no other Python library "
        "writes ZTABLE).",
    ),
    # FITS container
    "perf-fits-open-many-hdus.py": (
        "Open file with many HDUs",
        "Synthetic files with N empty image HDUs (N=100/1000/10000). "
        "Two regimes per N: ``bare open`` (rustfits eager parse vs "
        "fitsio lazy noop -- different constructor contracts, "
        "rustfits looks slower) and ``open + walk all`` "
        "(apples-to-apples; both walk every HDU).  The note column "
        "carries per-HDU normalized time -- flat across N confirms "
        "linear scaling (cfitsio had a quadratic-on-open bug here).",
    ),
    # Extend (RSS)
    "perf-image-extend-1d.py": (
        "Uncompressed 1-D image build — bounded-memory extend vs write-once",
        "1 GB ``f8`` map; ``ImageHDU.extend`` appends chunks instead "
        "of holding the whole array in RAM.  fitsio cannot append; "
        "RSS comparison is rustfits-self plus fitsio's write-once "
        "as a reference.",
    ),
    "perf-table-append.py": (
        "Table append — incremental catalog build vs write-once",
        "Type-exhaustive catalog (same as the BINTABLE read/write "
        "benches); four variants pair {uncompressed, ZTABLE} × "
        "{fixed-only, with VLA}.  Builds N rows by N/K calls to "
        "``hdu.append(K rows)`` against the one-shot "
        "``f.write_table(N rows)`` reference; fitsio write-once on "
        "the uncompressed variants only (it has no ZTABLE writer). "
        "ZTABLE small-chunk rows pay a per-call partial-tile "
        "re-encode; VLA rows additionally exercise the heap "
        "relocate-forward (uncompressed) or dual-descriptor heap "
        "(ZTABLE).",
    ),
    "perf-compressed-image-extend-healsparse.py": (
        "Compressed 1-D image build — bounded-memory extend vs write-once",
        "Same shape as the compressed healsparse benches; "
        "``CompressedImageHDU.extend`` appends chunks.  fitsio cannot "
        "append compressed.",
    ),
}


def _bench_env_string() -> str:
    """One-line summary of the bench environment (versions + CPU + OS)."""
    import platform

    parts = [f"Python {platform.python_version()}"]
    try:
        import numpy

        parts.append(f"numpy {numpy.__version__}")
    except ImportError:
        pass
    try:
        import rustfits

        parts.append(f"rustfits {getattr(rustfits, '__version__', '?')}")
    except ImportError:
        pass
    try:
        import fitsio

        parts.append(f"fitsio {fitsio.__version__}")
    except ImportError:
        pass
    try:
        import astropy

        parts.append(f"astropy {astropy.__version__}")
    except ImportError:
        pass
    parts.append(f"{platform.machine()} / {platform.system()}")
    return ", ".join(parts)


def _rst_escape(s: str) -> str:
    """
    Escape pipe characters that would break list-table cell parsing.
    The bench labels don't usually contain markup, so we keep it
    minimal — only the few characters that confuse RST inline.
    """
    return s.replace("|", r"\|")


def _list_table(
    title: str, widths: list[int], headers: list[str], rows: list[list[str]]
) -> str:
    """
    Render a single RST ``list-table`` directive as a string.
    """
    lines = [
        f".. list-table:: {title}",
        f"   :widths: {' '.join(str(w) for w in widths)}",
        "   :header-rows: 1",
        "",
    ]
    for i, row in enumerate([headers, *rows]):
        prefix = "   * - "
        cont = "     - "
        for j, cell in enumerate(row):
            cell = _rst_escape(str(cell))
            lines.append((prefix if j == 0 else cont) + cell)
        # mark row boundary; the next * - begins a new row
        if i < len(rows):
            pass  # nothing extra; next iter writes its own "* -"
    lines.append("")
    return "\n".join(lines)


def _ratio_classify(ratio: float) -> tuple[str, str]:
    """
    Return (CSS role name, badge label) for a fitsio/rustfits ratio.

    Roles are inline RST class directives declared at the top of the
    generated file; the CSS in ``docs/_static/perf.css`` paints them
    green / amber / red.  When the surrounding renderer ignores
    roles (plain RST viewers, GitHub), the badge label still appears
    so the win/loss tag is visible in monochrome.
    """
    if ratio >= 1.10:
        return "perf-fast", "FAST"
    if ratio >= 0.95:
        return "perf-par", "~par"
    return "perf-slow", "SLOW"


def _colored_ratio(ratio: float) -> str:
    """RST role-wrapped, colorized ratio cell."""
    role, _tag = _ratio_classify(ratio)
    return f":{role}:`{ratio:.2f}×`"


def write_rst(
    records: list[dict],
    path: str,
    kinds: set[str] | None = None,
) -> None:
    """
    Write per-script summary tables to ``path`` as RST list-table
    directives.  The output is meant to be ``.. include::``-d from a
    Sphinx page; it has no top-level header so the including page can
    supply its own.

    ``kinds`` filters to scripts whose records include at least one
    of the named record kinds (``"cross_tool"``, ``"self_comparison"``,
    ``"rss"``).  ``None`` keeps everything.  The cross-tool tally at
    the bottom only appears when ``"cross_tool"`` is in ``kinds`` (or
    ``kinds`` is ``None``).

    Each script gets its own table with a descriptive title and a
    one-line subtitle (from ``SCRIPT_GROUPS``) supplying the file-
    layout context the per-row operation labels lack on their own.
    Three table flavors — cross_tool, self_comparison, rss — pick
    their columns from the record shape.  Cross-tool ratio cells are
    colorized via the ``:perf-fast:`` / ``:perf-par:`` / ``:perf-slow:``
    inline roles declared at the file head (CSS in perf.css).
    """
    import datetime

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    out: list[str] = []
    out.append(
        ".. -- this file is generated by perf/perf-all.py "
        "(--rst-out-xtool / --rst-out-self);"
    )
    out.append(".. -- do not edit by hand; re-run the script to refresh.")
    out.append("")
    # Inline-role declarations for the colorized ratio cells.  Each
    # role's name becomes the CSS class Sphinx puts on the rendered
    # span; perf.css supplies the actual color.
    out.append(".. role:: perf-fast")
    out.append(".. role:: perf-par")
    out.append(".. role:: perf-slow")
    out.append("")
    out.append(
        f"*Last generated {stamp}*.  Environment: {_bench_env_string()}."
    )
    out.append("")
    out.append(
        "These are point-in-time measurements.  The benchmarks under "
        "``perf/`` are the source of truth — re-run "
        "``python perf/perf-all.py`` for current numbers on your "
        "hardware."
    )
    out.append("")

    # Bucket records by script, preserving display order from
    # SCRIPT_GROUPS.  Unknown scripts (new benches not yet registered)
    # appear at the end in name order.
    by_script: dict[str, list[dict]] = {}
    for r in records:
        by_script.setdefault(r.get("script", ""), []).append(r)
    ordered_scripts = [s for s in SCRIPT_GROUPS if s in by_script]
    extras = sorted(s for s in by_script if s not in SCRIPT_GROUPS)
    ordered_scripts.extend(extras)
    if kinds is not None:
        ordered_scripts = [
            s
            for s in ordered_scripts
            if any(r.get("kind") in kinds for r in by_script[s])
        ]
    ordered_scripts = [
        s for s in ordered_scripts if s not in SCRIPTS_WITH_NARRATIVE
    ]

    n_wins = n_par = n_loss = 0
    for script in ordered_scripts:
        rows = by_script[script]
        title, subtitle = SCRIPT_GROUPS.get(
            script, (_short_script(script), "")
        )
        out.append(f"**{title}**")
        out.append("")
        if subtitle:
            out.append(subtitle)
            out.append("")
        # All records from a single script are the same kind today.
        kind = rows[0].get("kind", "cross_tool")
        if kind == "cross_tool":
            tbl_rows = []
            for r in sorted(rows, key=lambda r: r.get("op", "")):
                rf = r.get("rustfits_s")
                fi = r.get("fitsio_s")
                ratio_cell = ""
                if rf and fi and rf > 0:
                    ratio = fi / rf
                    ratio_cell = _colored_ratio(ratio)
                    tag = _ratio_classify(ratio)[1]
                    if tag == "FAST":
                        n_wins += 1
                    elif tag == "~par":
                        n_par += 1
                    else:
                        n_loss += 1
                tbl_rows.append(
                    [
                        r.get("op", ""),
                        fmt_time(rf),
                        fmt_time(fi),
                        ratio_cell,
                        fmt_rate(r.get("nbytes", 0), rf),
                    ]
                )
            out.append(
                _list_table(
                    f"{title} — vs fitsio",
                    [38, 12, 12, 14, 12],
                    [
                        "operation",
                        "rustfits",
                        "fitsio",
                        "vs fitsio",
                        "rustfits rate",
                    ],
                    tbl_rows,
                )
            )
        elif kind == "self_comparison":
            tbl_rows = []
            for r in sorted(rows, key=lambda r: r.get("op", "")):
                ps = r.get("primary_s")
                ss = r.get("secondary_s")
                ratio_cell = ""
                if ps and ss and ss > 0:
                    ratio_cell = f"{ps / ss:.2f}×"
                tbl_rows.append(
                    [
                        r.get("op", ""),
                        r.get("primary_label", ""),
                        fmt_time(ps),
                        r.get("secondary_label", ""),
                        fmt_time(ss),
                        ratio_cell,
                    ]
                )
            out.append(
                _list_table(
                    title,
                    [30, 12, 10, 12, 10, 10],
                    [
                        "operation",
                        "primary",
                        "primary t",
                        "secondary",
                        "secondary t",
                        "ratio",
                    ],
                    tbl_rows,
                )
            )
        elif kind == "rss":
            tbl_rows = []
            for r in sorted(rows, key=lambda r: r.get("op", "")):
                tbl_rows.append(
                    [
                        r.get("op", ""),
                        fmt_time(r.get("build_s")),
                        f"{r.get('peak_rss_mb', 0):,.0f} MB",
                    ]
                )
            out.append(
                _list_table(
                    f"{title} — build wall + peak RSS",
                    [40, 14, 14],
                    ["regime", "build", "peak RSS"],
                    tbl_rows,
                )
            )

    total = n_wins + n_par + n_loss
    if total and (kinds is None or "cross_tool" in kinds):
        out.append("Cross-tool tally")
        out.append("~~~~~~~~~~~~~~~~")
        out.append("")
        out.append(
            f"Across all cross-tool comparisons: "
            f":perf-fast:`{n_wins}/{total} faster` (≥ 1.10×), "
            f":perf-par:`{n_par}/{total} ≈ par` (0.95–1.10×), "
            f":perf-slow:`{n_loss}/{total} slower` (< 0.95×)."
        )
        out.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(out).rstrip() + "\n")


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
    ap.add_argument(
        "--rst-out-xtool",
        help="write cross-tool (rustfits vs fitsio) RST tables to "
        "this path (suitable for `.. include::` from a Sphinx page)",
    )
    ap.add_argument(
        "--rst-out-self",
        help="write self-comparison + RSS RST tables to this path",
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

    if args.rst_out_xtool:
        write_rst(records, args.rst_out_xtool, kinds={"cross_tool"})
        print(f"\nwrote cross-tool RST tables to {args.rst_out_xtool}")
    if args.rst_out_self:
        write_rst(records, args.rst_out_self, kinds={"self_comparison", "rss"})
        print(f"\nwrote self/RSS RST tables to {args.rst_out_self}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
