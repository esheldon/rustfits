"""
Float-image edge cases: subnormals + NaN + mixed-tile + lossless
round-trips.

Ported from fitsio's regression suite as part of cross-tool
oddball-case mining (real user pain on these inputs):

  - Subnormal-float round-trip (uncompressed + lossless gzip):
    fitsio/tests/test_image.py::test_image_subnormal_float32/64
  - NaN + subnormal pixels in one compressed tile:
    fitsio/tests/test_image_compression.py
      ::test_image_compression_nulls_patches_with_subnormal
  - DITHER_2 zero preservation across algorithms + dtypes:
    fitsio/tests/test_image_compression.py
      ::test_compress_preserve_zeros

The DITHER_2 + zero-preservation invariant is also exercised by
test_image_compressed_write_quantize.py::test_dither2_exact_zero_round_trips_exactly
on a different fixture; this file's case covers a wider matrix
(gzip_1 / gzip_2 / rice_1 x f4 / f8 x +/- NaN) on the same
seeded random fixture fitsio used.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import rustfits


# Smallest representable subnormal floats — these are the values
# a naive flush-to-zero implementation drops silently.
SUBNORMAL_F32 = 8.82818e-44
SUBNORMAL_F64 = 2.225073858507203e-309


# ---------------------------------------------------------------
# Test #3 — subnormal floats round-trip exactly
# ---------------------------------------------------------------
#
# Uncompressed and lossless-gzip paths must preserve the smallest
# representable float without flushing to zero.  A NaN appended to
# the array doesn't change the answer.  Bit-exact equality.


@pytest.mark.parametrize(
    "dtype,subnormal",
    [("f4", SUBNORMAL_F32), ("f8", SUBNORMAL_F64)],
)
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("compress", [None, "gzip_1"])
def test_subnormal_floats_round_trip(dtype, subnormal, with_nan, compress):
    vs = [subnormal] * 10
    if with_nan:
        vs.append(np.nan)
    arr = np.array(vs, dtype=dtype)

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            if compress is None:
                f.write_image(arr)
            else:
                # Lossless float path: explicit quantize=None
                # routes the float bytes straight through gzip
                # without quantization.
                f.write_image(arr, compress=compress, quantize=None)

            # Same-handle read sees what we just wrote.
            same_handle = f[-1].read()
        np.testing.assert_array_equal(same_handle, arr)

        # Post-reopen read sees the on-disk bytes.
        with rustfits.FITS(fname, "r") as f:
            reread = f[-1].read()
        np.testing.assert_array_equal(reread, arr)


# ---------------------------------------------------------------
# Test #4 — NaN + subnormal pixels mixed in one compressed tile
# ---------------------------------------------------------------
#
# Combines the two failure modes (NaN sentinel handling + subnormal
# preservation) on a per-tile basis.  Lossless paths must preserve
# both bit-exactly; lossy paths must preserve NaN and stay within
# 0.1 of the non-NaN values.
#
# Coverage matrix (round-trip cases):
#   gzip_1 + lossless (quantize=None) — full bit-exact
#   gzip_1 + lossy (Quantize(level=4)) — NaN preserved, others within tol
#   rice_1 + lossy (Quantize(level=4)) — same
#
# The fourth cell (rice_1 + quantize=None) is rejected by design;
# the dedicated rejection assertion follows this parametrized
# function below.


_TILE_MIXED_CASES = [
    ("gzip_1", 0),
    ("gzip_1", 4),
    ("rice_1", 4),
]


@pytest.mark.parametrize("compress,qlevel", _TILE_MIXED_CASES)
def test_nan_plus_subnormal_in_compressed_tile(compress, qlevel):
    ncols = 4
    rng = np.random.default_rng(seed=10)
    data = np.arange(ncols * 4, dtype="f4").reshape((4, ncols))
    data += rng.normal(scale=0.5, size=data.shape).astype("f4")
    # Row 1: NaN in col 0, subnormal in the rest.
    data[1, 0] = np.nan
    data[1, 1:] = SUBNORMAL_F32
    # Row 2: NaN in col 0, ordinary float (5.0) in the rest.
    data[2, 0] = np.nan
    data[2, 1:] = 5.0

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            if qlevel == 0:
                f.write_image(data, compress=compress, quantize=None)
            else:
                f.write_image(
                    data,
                    compress=compress,
                    quantize=rustfits.Quantize(
                        level=float(qlevel),
                        seed=10,
                    ),
                )
        with rustfits.FITS(fname, "r") as f:
            rdata = f[1].read()

    # NaN pixels survive in every case.
    assert np.isnan(rdata[1, 0])
    assert np.isnan(rdata[2, 0])

    if qlevel == 0:
        # Lossless: subnormals and ordinary floats round-trip exactly.
        np.testing.assert_array_equal(rdata[1, 1:], data[1, 1:])
        np.testing.assert_array_equal(rdata[2, 1:], data[2, 1:])
    else:
        # Lossy: 0.1 absolute tolerance on the non-NaN pixels.
        np.testing.assert_allclose(rdata[1, :], data[1, :], rtol=0, atol=0.1)
        np.testing.assert_allclose(rdata[2, :], data[2, :], rtol=0, atol=0.1)


def test_rice_with_quantize_none_rejected_via_write_image():
    """
    Companion to test_quantize_none_requires_gzip in
    test_image_compressed_write_quantize.py — that one exercises the
    create_image_hdu + Rice1 class path; this one exercises the
    write_image(data, compress="rice_1", quantize=None) path.  Both
    must reject with the same clear error pointing at the
    Gzip1/Gzip2 alternatives.

    Completes the (gzip_1, rice_1) x (qlevel=0, qlevel=4) matrix
    started above: three pass-through cases + this one explicit
    rejection.
    """
    arr = np.zeros((4, 4), dtype="f4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="unquantized float"):
                f.write_image(arr, compress="rice_1", quantize=None)


# ---------------------------------------------------------------
# Non-finite round-trip on a single small (1x5) tile
# ---------------------------------------------------------------
#
# Narrow happy-case pin: NaN + +Inf + -Inf all round-trip exactly
# in a 1x5 single-tile image across the supported compression
# modes.  Lossless paths (Gzip1/Gzip2 + quantize=None) are
# byte-faithful; the lossy paths (qlevel=4) happen to preserve
# +/-Inf in this fixture because the tile is so small that the
# bscale computation degenerates predictably — the noise estimator
# returns a degenerate value that decodes back to the original.
#
# Inf preservation does NOT generalize to realistic multi-tile
# lossy fixtures — see ``test_isolated_nonfinite_in_multi_tile_lossy``
# below (4 strict-xfail cases) for the bug where Inf inside a tile
# with finite neighbors corrupts the bscale and decodes neighbors
# to a constant.  Cross-link: GitHub issue #19 (filed 2026-06-01,
# `gh issue view 19` for the full diagnosis + fix-location pointer).
#
# Companion to
# fitsio/tests/test_util.py::test_nonfinite_as_cfitsio_floating_null_value
# (which tests fitsio's internal nonfinite-to-sentinel utility;
# this test exercises the user-facing round-trip directly).


_NONFINITE_CASES = [
    ("gzip_1", None),
    ("gzip_1", 4),
    ("gzip_2", None),
    ("gzip_2", 4),
    ("rice_1", 4),
    # rice_1 + None is rejected by design (covered above).
]


@pytest.mark.parametrize("compress,qlevel", _NONFINITE_CASES)
def test_nonfinite_round_trips_through_compressed_tile(compress, qlevel):
    data = np.array(
        [[1.0, 2.0, np.nan, np.inf, -np.inf]],
        dtype="f4",
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            if qlevel is None:
                f.write_image(data, compress=compress, quantize=None)
            else:
                f.write_image(
                    data,
                    compress=compress,
                    quantize=rustfits.Quantize(
                        level=float(qlevel),
                        seed=10,
                    ),
                )
        with rustfits.FITS(fname, "r") as f:
            r = f[1].read()
    assert np.isnan(r[0, 2]), "NaN should survive"
    assert r[0, 3] == np.inf, f"+Inf should survive (got {r[0, 3]})"
    assert r[0, 4] == -np.inf, f"-Inf should survive (got {r[0, 4]})"
    # Finite values round-trip too (lossy is within tolerance).
    if qlevel is None:
        assert r[0, 0] == 1.0
        assert r[0, 1] == 2.0
    else:
        np.testing.assert_allclose(r[0, 0], 1.0, rtol=0, atol=0.1)
        np.testing.assert_allclose(r[0, 1], 2.0, rtol=0, atol=0.1)


# ---------------------------------------------------------------
# Test #5 — DITHER_2 preserves exact zero pixels
# ---------------------------------------------------------------
#
# SUBTRACTIVE_DITHER_2 was added to the FITS spec specifically to
# preserve pixels whose stored value is exactly 0.0 through lossy
# quantization (the ZERO_VALUE_I32 sentinel mechanism).  Verifies
# the invariant across the algorithm + dtype + NaN matrix on a
# 5x20 seeded-random fixture; HCOMPRESS is excluded because the
# FITS spec disallows DITHER_2 with HCOMPRESS.


_DITHER2_ALGOS = ["gzip_1", "gzip_2", "rice_1"]
_DITHER2_DTYPES = ["f4", "f8"]


@pytest.mark.parametrize("compress", _DITHER2_ALGOS)
@pytest.mark.parametrize("dtype", _DITHER2_DTYPES)
@pytest.mark.parametrize("with_nan", [False, True])
def test_dither2_preserves_zeros(compress, dtype, with_nan):
    rng = np.random.default_rng(2020)
    data = rng.normal(size=5 * 20).reshape(5, 20).astype(dtype)
    # Two designated pixels forced to exact 0.0 — DITHER_2 must
    # preserve these regardless of the surrounding quantization
    # noise level.
    zero_inds = [(1, 3), (2, 9)]
    for r, c in zero_inds:
        data[r, c] = 0.0
    if with_nan:
        data[3, 15] = np.nan

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(
                data,
                compress=compress,
                quantize=rustfits.Quantize(level=16.0, method="dither2"),
            )
        with rustfits.FITS(fname, "r") as f:
            rdata = f[-1].read()

    for r, c in zero_inds:
        assert rdata[r, c] == 0.0, (
            f"zero pixel at ({r},{c}) lost under "
            f"compress={compress} dtype={dtype} with_nan={with_nan}"
        )
    if with_nan:
        assert np.isnan(rdata[3, 15])


# ---------------------------------------------------------------
# Sanity probe — every f4 / f8 across both modes should not flush
# any subnormal to zero ANYWHERE in the array.
# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,subnormal",
    [
        ("f4", SUBNORMAL_F32),
        ("f8", SUBNORMAL_F64),
    ],
)
def test_subnormal_array_never_flushes_to_zero(dtype, subnormal):
    """
    Defense against a regression where any float code path
    introduces an unintended flush-to-zero on subnormals (a
    common cause: passing -ffast-math / FTZ to a future codec).
    """
    arr = np.full(100, subnormal, dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(arr)
        with rustfits.FITS(fname, "r") as f:
            r = f[0].read() if f[0].has_data else f[-1].read()
    assert (r == subnormal).all()
    assert (r != 0).all()


# ---------------------------------------------------------------
# Isolated non-finite cell in a multi-tile lossy compressed image
# ---------------------------------------------------------------
#
# Sets exactly one cell (the [1, 1] position) of a 6x6 float image
# to NaN / +Inf / -Inf, then compresses with RICE_1 + qlevel=2 +
# tile_shape=(3, 3) so the non-finite cell sits inside one tile
# while three other tiles are well-formed integer ranges.  This is
# the realistic "one bad pixel in a mostly-good image" pattern
# (cosmic ray, saturated pixel, sensor defect).
#
# Spec contract: a non-finite input cell must read back non-finite
# AND surrounding cells in the same tile must round-trip within
# the lossy tolerance.  cfitsio (and fitsio) achieve this by
# replacing non-finite values with the null sentinel BEFORE
# computing the per-tile bscale, so non-finite values don't
# contaminate the quantization range.
#
# NaN case works in rustfits today — likely because NaN propagates
# through the noise estimator in a way that's already handled
# (NaN-aware reductions or explicit skip).  The Inf cases FAIL:
# the per-tile bscale calculation receives Inf and degenerates to
# Inf, which means every cell in that tile decodes to bzero
# (mid-range constant), corrupting the non-Inf neighbors.  Pinned
# as xfail so the limitation is discoverable and a future codec
# fix that pre-filters non-finite values before bscale will surface
# as xpass.
#
# Regression pin against fitsio/tests/test_image_compression.py
# ::test_image_compression_nulls.


@pytest.mark.parametrize(
    "dtype,nan_value",
    [
        ("f4", np.nan),
        ("f8", np.nan),
        pytest.param(
            "f4",
            np.inf,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "rustfits's quantizer doesn't pre-filter non-finite "
                    "values before computing per-tile bscale; +Inf in the "
                    "tile makes bscale degenerate, corrupting neighbors."
                ),
            ),
        ),
        pytest.param(
            "f8",
            np.inf,
            marks=pytest.mark.xfail(
                strict=True,
                reason="see f4 +Inf above",
            ),
        ),
        pytest.param(
            "f4",
            -np.inf,
            marks=pytest.mark.xfail(
                strict=True,
                reason="see f4 +Inf above",
            ),
        ),
        pytest.param(
            "f8",
            -np.inf,
            marks=pytest.mark.xfail(
                strict=True,
                reason="see f4 +Inf above",
            ),
        ),
    ],
)
def test_isolated_nonfinite_in_multi_tile_lossy(dtype, nan_value):
    data = np.arange(36).reshape((6, 6)).astype(dtype)
    data[1, 1] = nan_value

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(
                data,
                compress=rustfits.Rice1(tile_shape=(3, 3)),
                quantize=rustfits.Quantize(level=2.0, seed=10),
            )
        with rustfits.FITS(fname, "r") as f:
            r = f[1].read()

    # The marked cell stays non-finite (NaN or Inf — either is OK).
    assert not np.isfinite(r[1, 1]), (
        f"expected non-finite at [1, 1], got {r[1, 1]!r}"
    )
    # Every other cell round-trips within lossy tolerance.  Use a
    # mask so the non-finite [1, 1] doesn't poison the diff.
    mask = np.isfinite(data)
    np.testing.assert_allclose(r[mask], data[mask], rtol=0, atol=1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
