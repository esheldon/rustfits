#!/usr/bin/env python
"""
Open performance with many HDUs: rustfits vs fitsio.

cfitsio had a quadratic-on-open bug in this regime historically -- every
HDU's header parse re-walked from the file start, so opening a file with
N HDUs cost O(N^2) card reads.  This bench verifies rustfits stays
linear in N by building synthetic files with N empty image HDUs (N
sweeping across orders of magnitude) and timing fresh
``FITS(fname, "r")`` per iter.

Two regimes per N -- the two tools have different open contracts:

* **bare open** -- ``rustfits.FITS(path, "r")`` vs ``fitsio.FITS(path)``.
  rustfits is EAGER (``parse_hdus_from_file`` runs at construction);
  fitsio is LAZY (no HDU list built until first access).  This row
  surfaces the honest cost difference of the constructor.
* **open + walk all HDUs** -- the apples-to-apples comparison.  fitsio
  is forced to do the same work via ``fits.update_hdu_list()``; that
  call walks every HDU and is what ``fits[i]`` triggers internally.
  rustfits has already done the walk in its constructor, so its row
  is identical to bare open (same code path).  The fitsio row in this
  regime is where the historical quadratic bug would surface.

Per N the bench reports total wall time AND per-HDU normalized time
(``time_s / N``).  The per-HDU column should stay flat across N if
scaling is linear; growth proportional to N would indicate the
fitsio-style quadratic bug.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-fits-open-many-hdus.py
    python perf/perf-fits-open-many-hdus.py --n 100 1000 10000 50000

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rustfits

import _harness as h


def build_fixture(fname: str, n_hdus: int) -> None:
    """
    Create a FITS file with ``n_hdus`` minimal image HDUs.

    Each HDU has BITPIX=8 and a 1-pixel data section -- the smallest
    legal image HDU.  File size is roughly ``n_hdus * 2 * 2880`` bytes
    (one header block + one data block per HDU), so even 10 k HDUs is
    ~57 MB; the open cost the bench measures is parse work, not I/O.
    """
    tiny = np.zeros(1, dtype=np.uint8)
    with rustfits.FITS(fname, "w+") as f:
        for _ in range(n_hdus):
            f.write_image(tiny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="N values to test (default: 100 1000 10000)",
    )
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    n_values = sorted(set(args.n))

    with h.scratch():
        fixtures: dict[int, str] = {}
        for n in n_values:
            fname = h.path(f"many-hdus-{n}.fits")
            t0 = time.perf_counter()
            build_fixture(fname, n)
            t1 = time.perf_counter()
            print(f"built fixture N={n:,}: {t1 - t0:.2f}s")
            fixtures[n] = fname

        # Correctness gate: both tools count HDUs identically.  Touch
        # fitsio[0] here so its HDU list builds before len() (which on
        # a fresh handle would return 0 with the lazy contract).
        for n, fname in fixtures.items():
            with rustfits.FITS(fname) as rf, fitsio.FITS(fname) as fi:
                fi.update_hdu_list()
                assert len(rf) == n, f"rustfits saw {len(rf)} HDUs, want {n}"
                assert len(fi) == n, f"fitsio saw {len(fi)} HDUs, want {n}"

        def open_rf(fname):
            def run():
                with rustfits.FITS(fname):
                    pass

            return run

        def open_fi(fname):
            def run():
                with fitsio.FITS(fname):
                    pass

            return run

        def open_walk_rf(fname):
            # rustfits already walked at open; this just matches the
            # fitsio shape (no extra work needed on the rustfits side).
            def run():
                with rustfits.FITS(fname):
                    pass

            return run

        def open_walk_fi(fname):
            # update_hdu_list() forces the walk fitsio defers in its
            # constructor.  fits[1] would do the same thing.
            def run():
                with fitsio.FITS(fname) as f:
                    f.update_hdu_list()

            return run

        results = []
        for n in n_values:
            fname = fixtures[n]
            r1 = h.bench(
                f"bare open (N={n:,})",
                open_rf(fname),
                run_fitsio=open_fi(fname),
                repeat=args.repeat,
            )
            us_rf = r1.rustfits.median / n * 1e6
            us_fi = r1.fitsio.median / n * 1e6
            r1.note = f"rf {us_rf:.1f} us/HDU, fi {us_fi:.1f} us/HDU"
            results.append(r1)

            r2 = h.bench(
                f"open + walk all (N={n:,})",
                open_walk_rf(fname),
                run_fitsio=open_walk_fi(fname),
                repeat=args.repeat,
            )
            us_rf = r2.rustfits.median / n * 1e6
            us_fi = r2.fitsio.median / n * 1e6
            r2.note = f"rf {us_rf:.1f} us/HDU, fi {us_fi:.1f} us/HDU"
            results.append(r2)

        h.print_env()
        h.report("Open file with many HDUs", results)


if __name__ == "__main__":
    main()
