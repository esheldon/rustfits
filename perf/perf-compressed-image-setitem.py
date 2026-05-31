#!/usr/bin/env python
"""
Compressed image __setitem__ cost: per-tile re-encode tax.

A ``hdu[selection] = value`` on a tile-compressed image decodes every
tile the selection touches, modifies it in numpy, and re-encodes /
appends to the heap.  A "patch a few pixels" workflow (interactive
masking, bad-pixel fix-up) therefore pays a full tile re-encode for
every tile the selection covers -- even a single-pixel write.

This bench measures the per-call cost across algorithms and selection
sizes, so users have a number to budget against.  Two axes:

* **algorithm**: GZIP_1, GZIP_2, RICE_1, HCOMPRESS_1, PLIO_1
  (integer i4 path), plus GZIP_1 unquantized-f4 and GZIP_1 quantized-f4
  (the quantize path re-uses each tile's existing bscale/bzero/seed,
  so it's a separate cost shape from the integer one).  An uncompressed
  i4 image is included as the "memcpy" floor.

* **selection**: four shapes against a fixed 8x8 grid of (32,32) tiles
  (image = (256,256)).
  - ``[64, 64]``         single pixel; touches 1 tile.
  - ``[64:96, 64:96]``   one full tile, aligned; touches 1 tile.
  - ``[60:68, 60:68]``   8x8 across a tile corner; touches 4 tiles.
  - ``[64:192, 64:192]`` 4x4 tile block, aligned; touches 16 tiles.

Rustfits-self only -- no fitsio cross-tool comparison.  fitsio's
``write(start=)`` API can patch compressed images, but repeated calls
across an algorithm sweep tickle a cfitsio memory-corruption bug
(``free(): invalid next size``, same shape as the macOS ffbinit issue
documented in CLAUDE.md) that aborts the Python process.  Subprocess
isolation would work around it but the per-call rustfits cost is the
useful number on its own -- this bench answers "what does each
algorithm cost me per touched tile" rather than a head-to-head.

For each (algo, selection) we open the fixture once, do 5
back-to-back setitem calls, and divide by 5 -- that amortizes
open/close across many calls and reflects "patch many pixels in a
loop", which is the realistic interactive-masking shape.  PCOUNT
grows monotonically across iters but per-call cost is constant
(setitem just appends new tile bytes to the heap; it doesn't repack).

Reported columns: per-call time, per-tile cost (the per-tile re-encode
rate, the number users should multiply by tiles-touched to estimate
their workload), and per-pixel cost (most interesting for the
single-pixel row, where it's the worst-case per-pixel rate).

REQUIRES a release build: ``maturin develop --release``.

Run::

    python perf/perf-compressed-image-setitem.py
    python perf/perf-compressed-image-setitem.py --image 512,512 --tile 64,64

Scratch files go to CWD as perf-tmp-* and are removed on exit.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import rustfits

import _harness as h


def _make_data(shape, dtype):
    rng = np.random.default_rng(0)
    if dtype == "i4":
        return rng.integers(0, 1_000_000, shape, dtype="i4")
    elif dtype == "f4":
        return rng.standard_normal(shape).astype("f4")
    else:
        raise ValueError(dtype)


def _build_fixture(fname, dtype, shape, compress, quantize):
    """
    Write a fresh fixture under ``compress`` / ``quantize`` (any of the
    config-object types, or ``None`` for uncompressed).  Returns the
    HDU index of the image (0 for uncompressed primary, 1 for
    compressed extension; uncompressed ``create_image_hdu`` on an
    empty file lands as primary, but compressed always lands as an
    extension because it needs a primary HDU to exist first).
    """
    data = _make_data(shape, dtype)
    with rustfits.FITS(fname, "w+") as f:
        kwargs = {}
        if compress is not None:
            kwargs["compress"] = compress
        if quantize is not None:
            kwargs["quantize"] = quantize
        f.create_image_hdu(dtype, shape, **kwargs)
        idx = 0 if compress is None else 1
        f[idx].write(data)
    return idx


def _selections(tile_shape):
    """
    (label, sel, touched-tile-count).
    """
    th, tw = tile_shape
    aligned = (slice(2 * th, 3 * th), slice(2 * tw, 3 * tw))
    spanning_lo = 2 * th - th // 8
    spanning_hi = 2 * th + th // 8
    spanning = (
        slice(spanning_lo, spanning_hi),
        slice(spanning_lo, spanning_hi),
    )
    block = (slice(2 * th, 6 * th), slice(2 * tw, 6 * tw))
    return [
        ("single pixel", (2 * th, 2 * tw), 1),
        ("1 tile aligned", aligned, 1),
        ("4 tiles spanning", spanning, 4),
        ("16 tiles aligned", block, 16),
    ]


def _value_for(sel, shape, dtype):
    if all(isinstance(s, int) for s in sel):
        return np.array(0, dtype=dtype).item()
    arr_shape = []
    for axis_s, axis_n in zip(sel, shape):
        if isinstance(axis_s, int):
            continue
        start, stop, step = axis_s.indices(axis_n)
        arr_shape.append(len(range(start, stop, step)))
    return np.zeros(tuple(arr_shape), dtype=dtype)


def _calibrate_calls(fname, idx, sel, value, target_s=0.1):
    """
    Pick an inner-loop ``calls`` count so a single timed window
    takes at least ``target_s`` seconds (default 100 ms).  At
    sub-10-µs per call, a small fixed ``calls`` count would let
    scheduler / cache / interrupt jitter dominate (we saw 55%
    swings run-to-run on the uncompressed 3 µs row at calls=5);
    scaling to ~100 ms per window puts every row well above the
    noise floor.

    Probes with 20 untimed calls to estimate per-call cost, then
    rounds up to the next multiple of 100.
    """
    with rustfits.FITS(fname, "r+") as f:
        hdu = f[idx]
        # Prime caches
        hdu[sel] = value
        t0 = time.perf_counter()
        for _ in range(20):
            hdu[sel] = value
        t1 = time.perf_counter()
    per_call = (t1 - t0) / 20
    if per_call <= 0:
        return 10_000
    raw = int(target_s / per_call) + 1
    # Round up to nearest 100, clamp to [50, 50_000]
    return max(50, min(50_000, ((raw + 99) // 100) * 100))


def _time_setitem(fname, idx, sel, value, *, repeats, warmup, calls):
    """
    Time the per-call cost of ``hdu[sel] = value``.  Runs
    ``warmup`` untimed calls, then ``calls`` timed calls per
    repeat, dividing by ``calls`` to get per-call.  GC paused
    during the timed window.  Returns median across ``repeats``.
    """
    samples = []
    for _ in range(repeats):
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[idx]
            for _ in range(warmup):
                hdu[sel] = value
            gc.collect()
            gc.disable()
            try:
                t0 = time.perf_counter()
                for _ in range(calls):
                    hdu[sel] = value
                t1 = time.perf_counter()
            finally:
                gc.enable()
        samples.append((t1 - t0) / calls)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default="256,256",
        help="image shape rows,cols (default 256,256)",
    )
    ap.add_argument(
        "--tile",
        default="32,32",
        help="tile shape rows,cols (default 32,32)",
    )
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--target-window",
        type=float,
        default=0.1,
        help="target seconds per timed window; calls/repeat auto-"
        "scales per (algo, sel) so the smallest per-call rows take "
        "this long (default 0.1 = 100 ms; lower = faster but noisier)",
    )
    args = ap.parse_args()

    rows, cols = (int(x) for x in args.image.split(","))
    th, tw = (int(x) for x in args.tile.split(","))
    shape = (rows, cols)
    tile_shape = (th, tw)
    sels = _selections(tile_shape)

    # (label, dtype, compress_spec, quantize_spec)
    algos = [
        ("uncompressed i4", "i4", None, None),
        ("GZIP_1 i4", "i4", rustfits.Gzip1(tile_shape=tile_shape), None),
        ("GZIP_2 i4", "i4", rustfits.Gzip2(tile_shape=tile_shape), None),
        ("RICE_1 i4", "i4", rustfits.Rice1(tile_shape=tile_shape), None),
        (
            "HCOMPRESS_1 i4",
            "i4",
            rustfits.Hcompress1(tile_shape=tile_shape),
            None,
        ),
        ("PLIO_1 i4", "i4", rustfits.Plio1(tile_shape=tile_shape), None),
        (
            "GZIP_1 f4 unquantized",
            "f4",
            rustfits.Gzip1(tile_shape=tile_shape),
            None,
        ),
        (
            "GZIP_1 f4 quantized",
            "f4",
            rustfits.Gzip1(tile_shape=tile_shape),
            rustfits.Quantize(level=4.0),
        ),
    ]

    title = (
        f"Compressed-image __setitem__ cost "
        f"(image={rows}x{cols}, tile={th}x{tw})"
    )
    print(title)
    h.print_env()
    print()

    cols_spec = [
        ("algorithm", 26, "l"),
        ("selection", 18, "l"),
        ("tiles", 7, "r"),
        ("calls", 8, "r"),
        ("per call", 12, "r"),
        ("per tile", 12, "r"),
        ("per pixel", 13, "r"),
    ]
    header = "  ".join(h._cell(c, w, a) for c, w, a in cols_spec)
    print(header)
    print("-" * len(header))

    with h.scratch():
        for algo_label, dtype, compress, quantize in algos:
            fname = h.fresh_path("csetitem")
            idx = _build_fixture(fname, dtype, shape, compress, quantize)
            for sel_label, sel, n_tiles in sels:
                value = _value_for(sel, shape, dtype)
                n_pixels = (
                    int(value.size) if isinstance(value, np.ndarray) else 1
                )
                calls = _calibrate_calls(
                    fname, idx, sel, value, target_s=args.target_window
                )
                t = _time_setitem(
                    fname,
                    idx,
                    sel,
                    value,
                    repeats=args.repeat,
                    warmup=args.warmup,
                    calls=calls,
                )
                per_tile = t / n_tiles
                per_pixel = t / n_pixels
                cells = [
                    (algo_label, 26, "l"),
                    (sel_label, 18, "l"),
                    (str(n_tiles), 7, "r"),
                    (str(calls), 8, "r"),
                    (h.fmt_time(t), 12, "r"),
                    (h.fmt_time(per_tile), 12, "r"),
                    (h.fmt_time(per_pixel), 13, "r"),
                ]
                print("  ".join(h._cell(x, w, a) for x, w, a in cells))
                h.emit_record(
                    {
                        "kind": "self_comparison",
                        "suite": title,
                        "op": f"{algo_label} / {sel_label}",
                        "algo": algo_label,
                        "selection": sel_label,
                        "n_tiles": n_tiles,
                        "n_pixels": n_pixels,
                        "calls_per_window": calls,
                        "per_call_s": t,
                        "per_tile_s": per_tile,
                        "per_pixel_s": per_pixel,
                    }
                )


if __name__ == "__main__":
    main()
