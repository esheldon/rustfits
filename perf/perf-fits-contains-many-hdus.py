#!/usr/bin/env python
"""
``key in fits`` / ``fits[key]`` performance with many HDUs.

Each EXTNAME lookup walks the HDU list until it finds a match (or
falls off the end), reading and parsing the EXTNAME card from each
HDU's header along the way.  This bench answers two questions:

1. Does ``__contains__`` stay linear in N as the HDU count grows
   (the contract — we walk and stop at first match, or walk all and
   return False on miss), or is there hidden quadratic overhead?

2. How does it compare to fitsio, which does the same try/except
   around ``__getitem__`` shape on its end (which dispatches into
   cfitsio's own EXTNAME search)?

Per N the bench measures with the FITS handle ALREADY OPEN (we're
timing the lookup, not the open):

* ``0 in fits``           -- int membership, expected O(1).
* ``(N-1) in fits``       -- same, last index.
* ``"HDU_0" in fits``     -- EXTNAME found at first HDU (early-exit
                              best case).
* ``"HDU_{N-1}" in fits`` -- EXTNAME found at last HDU (must walk all
                              -- the diagnostic row for linear scaling).
* ``"MISSING" in fits``   -- EXTNAME miss (walks all, returns False).
* ``fits["HDU_{N-1}"]``   -- __getitem__ worst case, for symmetry with
                              the __contains__ worst case.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-fits-contains-many-hdus.py
    python perf/perf-fits-contains-many-hdus.py --n 100 1000 10000 50000

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
    Create a FITS file with ``n_hdus`` minimal image HDUs, each with a
    unique EXTNAME ``HDU_<i>`` so we can test best-case (first), worst-
    case (last), and miss lookups.
    """
    tiny = np.zeros(1, dtype=np.uint8)
    with rustfits.FITS(fname, "w+") as f:
        # write_image creates an unnamed primary; write the extname'd
        # ones as extension HDUs.  Primary stays unnamed (i=0).
        f.write_image(tiny)
        for i in range(n_hdus - 1):
            f.create_image_hdu(dtype="u1", dims=[1], extname=f"HDU_{i}")


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
            fname = h.path(f"contains-{n}.fits")
            t0 = time.perf_counter()
            build_fixture(fname, n)
            t1 = time.perf_counter()
            print(f"built fixture N={n:,}: {t1 - t0:.2f}s")
            fixtures[n] = fname

        results = []
        for n in n_values:
            fname = fixtures[n]
            last_name = f"HDU_{n - 2}"  # last extension HDU (primary is 0)

            # Open both handles ONCE per N and keep them alive across
            # the timed closures.  Bench measures pure lookup cost,
            # no open overhead.  fi.update_hdu_list() forces fitsio's
            # lazy parse so the first lookup doesn't pay the walk.
            rf = rustfits.FITS(fname)
            fi = fitsio.FITS(fname)
            fi.update_hdu_list()
            try:
                # Correctness gate: both tools agree on the lookups
                # before we time anything.
                assert "HDU_0" in rf and "HDU_0" in fi
                assert last_name in rf and last_name in fi
                assert "MISSING" not in rf and "MISSING" not in fi

                # int membership (expected O(1) in both)
                results.append(
                    h.bench(
                        f"0 in fits (N={n:,})",
                        lambda: 0 in rf,
                        run_fitsio=lambda: 0 in fi,
                        repeat=args.repeat,
                    )
                )
                results.append(
                    h.bench(
                        f"(N-1) in fits (N={n:,})",
                        lambda: (n - 1) in rf,
                        run_fitsio=lambda: (n - 1) in fi,
                        repeat=args.repeat,
                    )
                )

                # EXTNAME found at FIRST HDU (early-exit best case)
                results.append(
                    h.bench(
                        f"'HDU_0' in fits (N={n:,})",
                        lambda: "HDU_0" in rf,
                        run_fitsio=lambda: "HDU_0" in fi,
                        repeat=args.repeat,
                    )
                )

                # EXTNAME found at LAST HDU (must walk all -- this is
                # where the dict-vs-walk gap shows up).  fitsio keeps a
                # name->index dict so it's O(1); rustfits currently
                # walks every HDU and re-parses its EXTNAME card.
                results.append(
                    h.bench(
                        f"'HDU_{n - 2}' in fits (N={n:,})",
                        lambda: last_name in rf,
                        run_fitsio=lambda: last_name in fi,
                        repeat=args.repeat,
                    )
                )

                # EXTNAME miss (walks all, returns False)
                results.append(
                    h.bench(
                        f"'MISSING' in fits (N={n:,})",
                        lambda: "MISSING" in rf,
                        run_fitsio=lambda: "MISSING" in fi,
                        repeat=args.repeat,
                    )
                )

                # fits[name] worst case (for symmetry with __contains__)
                results.append(
                    h.bench(
                        f"fits['HDU_{n - 2}'] (N={n:,})",
                        lambda: rf[last_name],
                        run_fitsio=lambda: fi[last_name],
                        repeat=args.repeat,
                    )
                )

            finally:
                rf.close()
                fi.close()

        h.print_env()
        h.report("`key in fits` / `fits[key]` with many HDUs", results)


if __name__ == "__main__":
    main()
