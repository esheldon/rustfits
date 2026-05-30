#!/usr/bin/env python
"""
Compressed 2-D image WRITE throughput: rustfits vs fitsio (DES-like).

The write side of the DES read benchmark: time write_image of the same
2-D f4 array, RICE_1 + quantize (q=16, SUBTRACTIVE_DITHER_2 so masked
zeros are preserved), rustfits vs fitsio.  This also measures RICE
*encode* -- the counterpart to the RICE *decode* slowdown the read
isolation found (see CLAUDE.md performance section).

The input array is built ONCE (untimed); each timed iteration overwrites
a scratch file.  We do NOT fsync -- encode dominates and buffered writes
isolate encode speed from disk flush.  A correctness gate reads each
tool's write back within the quantization tolerance, with the masked
zero exact.

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-compressed-image-write-des.py
    python perf/perf-compressed-image-write-des.py --rows 10000 --cols 10000

Scratch file goes to CWD as perf-tmp-* and is removed on exit
(PERF_KEEP=1 keeps it).
"""

from __future__ import annotations

import argparse

import numpy as np
import rustfits

import _data
import _harness as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=h.env_int("PERF_ROWS", 4000))
    ap.add_argument("--cols", type=int, default=h.env_int("PERF_COLS", 4000))
    ap.add_argument("--tile", type=int, default=100, help="square tile dim")
    ap.add_argument("--q", type=float, default=16.0, help="quantize level")
    ap.add_argument("--seed", type=int, default=1, help="dither seed")
    ap.add_argument("--zero-frac", type=float, default=0.05)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if not h.HAVE_FITSIO:
        raise SystemExit("fitsio not installed; this comparison needs it")
    import fitsio

    rows, cols, tile = args.rows, args.cols, args.tile
    q, seed = args.q, args.seed
    data = _data.des_array(rows, cols, args.zero_frac)

    def rf_write():
        fname = h.fresh_path("wcomp2d-rf")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(
                data,
                compress=rustfits.Rice1(tile_shape=(tile, tile)),
                quantize=rustfits.Quantize(
                    level=q, method="dither2", seed=seed
                ),
            )

    def fi_write():
        fname = h.fresh_path("wcomp2d-fi")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(
                data,
                compress="RICE",
                qlevel=q,
                qmethod="SUBTRACTIVE_DITHER_2",
                dither_seed=seed,
                tile_dims=(tile, tile),
            )

    with h.scratch():
        # Correctness gate: lossy round-trip within tolerance, zero exact.
        gate_rf = h.fresh_path("wcomp2d-gate-rf")
        gate_fi = h.fresh_path("wcomp2d-gate-fi")
        with rustfits.FITS(gate_rf, "w+") as f:
            f.write_image(
                data,
                compress=rustfits.Rice1(tile_shape=(tile, tile)),
                quantize=rustfits.Quantize(
                    level=q, method="dither2", seed=seed
                ),
            )
        with rustfits.FITS(gate_rf) as f:
            r = f[1].read()
        assert np.allclose(r, data, atol=0.1), "rustfits round-trip off"
        assert r[0, 0] == 0.0, "masked zero not preserved"
        with fitsio.FITS(gate_fi, "rw", clobber=True) as f:
            f.write(
                data,
                compress="RICE",
                qlevel=q,
                qmethod="SUBTRACTIVE_DITHER_2",
                dither_seed=seed,
                tile_dims=(tile, tile),
            )
        with fitsio.FITS(gate_fi) as f:
            assert np.allclose(f[1].read(), data, atol=0.1)

        raw = rows * cols * 4
        result = h.bench(
            f"write RICE q={q} dither2 ({raw / 1e6:,.0f} MB f4)",
            rf_write,
            run_fitsio=fi_write,
            nbytes=raw,
            repeat=args.repeat,
        )
        h.print_env()
        h.report(
            f"2-D lossy compressed WRITE (RICE q={q} dither2, f4)", [result]
        )


if __name__ == "__main__":
    main()
