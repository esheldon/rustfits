"""
ZIMAGE quantized float compressed image writes.

Float images are encoded by quantizing each tile to i32 (per-tile
bscale/bzero from a noise estimate), then compressing the i32
stream with the chosen algorithm (Rice1/Gzip1/Gzip2/Hcompress1).
Tiles that can't be quantized fall back to GZIP-compressed raw
float bytes.

Coverage:
    - Schema validation: 4 columns (COMPRESSED_DATA + ZSCALE +
      ZZERO + GZIP_COMPRESSED_DATA), ZQUANTIZ + ZDITHER0 + ZBLANK
      cards
    - Round-trip across f4/f8, all three dither methods
      (NO_DITHER / SUBTRACTIVE_DITHER_1 / SUBTRACTIVE_DITHER_2),
      same-handle + post-reopen
    - DITHER_2 exact-zero preservation (ZERO_VALUE_I32 sentinel)
    - NaN preservation for ALL dither methods (NULL_VALUE_I32
      sentinel; cfitsio interop)
    - GZIP fallback on constant-tile / unquantizable input
    - fitsio cross-read agreement across (algorithm, method) matrix
    - Rejection paths: quantize= on integer dtype, quantize=
      without compress=
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


# On macOS, cfitsio's dequantization produces slightly different
# float results than on Linux — almost certainly compiler-codegen
# variance (FMA fusion, Apple libm vs glibc).  rustfits's Rust
# dequant bit-matches Linux cfitsio.  Linux remains strictly
# bit-exact below; macOS allows allclose at the documented level
# (rtol up to ~1.6e-5 on near-zero values, atol up to ~2.6e-9).
_MACOS_FP_RTOL = 1e-5
_MACOS_FP_ATOL = 1e-8
_IS_MACOS = sys.platform == "darwin"


def _smooth(shape, dtype="f4", seed=0):
    """
    Smooth-ish float data with enough variation that the per-tile
    noise estimate finds something to work with.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    x = np.arange(n) * 0.1
    base = np.sin(x) * 5.0 + np.cos(x * 0.3) * 2.0
    noise = rng.standard_normal(n) * 0.3
    return (base + noise).astype(dtype).reshape(shape)


# ---------------------- schema -------------------------------------


def test_create_emits_quantize_schema():
    """create_image_hdu(..., compress=Rice1, quantize=Quantize)
    emits a 4-column BINTABLE with ZQUANTIZ + ZDITHER0 cards."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (32, 48),
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=42,
                ),
            )
            hdr = f[1].header
            assert hdr["ZBITPIX"] == -32
            assert hdr["TFIELDS"] == 4
            assert hdr["TTYPE1"].strip() == "COMPRESSED_DATA"
            assert hdr["TTYPE2"].strip() == "ZSCALE"
            assert hdr["TTYPE3"].strip() == "ZZERO"
            assert hdr["TTYPE4"].strip() == "GZIP_COMPRESSED_DATA"
            assert hdr["TFORM2"].strip() == "1D"
            assert hdr["TFORM3"].strip() == "1D"
            assert hdr["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
            assert hdr["ZDITHER0"] == 42


# ---------------------- round-trip ---------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_round_trip_no_dither(dtype):
    """NO_DITHER round-trip: max error bounded by ~bscale (= noise/4)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), dtype=dtype, seed=11)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="no_dither",
                ),
            )
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        assert same.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(same, reopen)
        # Quantization step ≈ noise/4 ≈ 0.3/4 ≈ 0.075 for this
        # data; allow plenty of headroom.
        assert np.abs(same - data).max() < 1.0


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_round_trip_dither1(dtype):
    """SUBTRACTIVE_DITHER_1 round-trip; same precision target as
    NO_DITHER (dither just whitens the quantization noise)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), dtype=dtype, seed=22)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=99,
                ),
            )
            f[1].write(data)
            out = f[1].read()
        assert np.abs(out - data).max() < 1.5


def test_quantize_none_requires_gzip():
    """quantize=None on a float HDU without compress=Gzip1 or Gzip2
    is rejected with a clear error.  Other algorithms (Rice1,
    Hcompress1, Plio1) cannot round-trip raw float bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="unquantized float"):
                f.create_image_hdu(
                    "f4",
                    (20, 20),
                    compress=rustfits.Rice1(tile_shape=(20, 20)),
                )


# ---------------------- fitsio interop -----------------------------


def test_fitsio_read_matches_rustfits_read():
    """
    rustfits-written quantized-float file: cfitsio's reader (via
    fitsio) must return the same physical values rustfits would.
    Bit-exact because both decoders go through the same Park-Miller
    dither table and the same dequant formula.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), seed=44)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=12345,
                ),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            rust_out = f[1].read()
        with fitsio.FITS(fn) as f:
            fitsio_out = f[1].read()
        np.testing.assert_array_equal(rust_out, fitsio_out)


# ---------------------- GZIP fallback ------------------------------


def test_constant_tile_triggers_gzip_fallback():
    """
    A constant-value tile has zero noise, so quantize_float returns
    None and the encoder falls back to GZIP-compressing the raw
    float bytes into the GZIP_COMPRESSED_DATA column.  Round-trip
    must be exact (no quantization loss).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.full((16, 16), 3.14, dtype="f4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method="dither1"),
            )
            f[1].write(data)
            out = f[1].read()
        # Lossless fallback → exact round-trip.
        np.testing.assert_array_equal(out, data)
        # Heap is non-empty (the fallback bytes live there).
        with rustfits.FITS(fn, "r") as f:
            assert f[1].header["PCOUNT"] > 0


# ---------------------- rejections ---------------------------------


def test_quantize_on_integer_dtype_rejected():
    """quantize= with integer ZBITPIX raises (it's float-only)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="quantize"):
                f.create_image_hdu(
                    "i4",
                    (16, 16),
                    compress=rustfits.Rice1(tile_shape=(16, 16)),
                    quantize=rustfits.Quantize(),
                )


def test_quantize_without_compress_rejected():
    """quantize= without compress= raises (no compressed HDU to
    quantize against)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="quantize"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    quantize=rustfits.Quantize(),
                )


# ---------------------- DITHER_2 + NaN -----------------------------


def test_dither2_exact_zero_round_trips_exactly():
    """
    DITHER_2 reserves ZERO_VALUE_I32 for exact-zero pixels so they
    round-trip bit-exact (not through the dequant formula's noise).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(55)
        data = rng.standard_normal((32, 32)).astype("f4") * 5
        # Inject several exact-zero pixels.
        data[5, 5] = 0.0
        data[10, 20] = 0.0
        data[15, 0] = 0.0
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method="dither2"),
            )
            f[1].write(data)
            out = f[1].read()
        # Exact-zero pixels round-trip bit-exact under DITHER_2.
        assert out[5, 5] == 0.0
        assert out[10, 20] == 0.0
        assert out[15, 0] == 0.0


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
def test_nan_round_trips_for_all_methods(method):
    """
    NaN input pixels map to NULL_VALUE_I32 on encode and back to
    NaN on decode for all three dither methods (cfitsio's
    convention).  Non-NaN pixels round-trip within the usual
    quantization tolerance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(77)
        data = rng.standard_normal((32, 32)).astype("f4") * 5
        # Inject NaNs at deterministic locations.
        nan_positions = [(3, 5), (10, 20), (25, 0)]
        for r, c in nan_positions:
            data[r, c] = np.nan
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method=method),
            )
            f[1].write(data)
            out = f[1].read()
        # NaN positions preserved exactly.
        assert np.array_equal(np.isnan(out), np.isnan(data))
        # Non-NaN pixels within quantization tolerance (loose
        # bound — bscale ~ noise/4 for this data).
        valid = ~np.isnan(data)
        assert np.abs(out[valid] - data[valid]).max() < 3.0


def test_zblank_card_emitted_for_float():
    """
    Float HDUs carry ZBLANK = -2147483647 so fitsio / astropy
    readers see the quantized null sentinel cfitsio expects.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(),
            )
            assert f[1].header["ZBLANK"] == -2147483647


def test_zblank_absent_for_integer():
    """
    Integer compressed HDUs do not get a ZBLANK card from the
    Phase 8 wiring — quantization isn't in play, and the sentinel
    would be meaningless.  (The Phase 1 ZBLANK plumbing for
    integer images via the header is a separate, unrelated path.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
            )
            assert "ZBLANK" not in f[1].header


# ---------------------- fitsio cross-check matrix ------------------


# fitsio + cfitsio agree with rustfits on the same physical
# pixel values across the supported (algorithm, method) matrix.
# Byte-exact heap agreement is NOT asserted here — fitsio's
# noise2/3/5 estimator picks a slightly different bscale on some
# inputs due to f32 vs f64 precision in the per-row diff arrays.
# The dequantized output still matches bit-for-bit because the
# encoded i32 stream + the on-disk ZSCALE/ZZERO together fully
# specify the decoded values.


@pytest.mark.parametrize(
    "algorithm_name",
    ["Rice1", "Gzip1", "Gzip2", "Hcompress1"],
)
@pytest.mark.parametrize(
    "method",
    ["no_dither", "dither1", "dither2"],
)
def test_fitsio_read_agrees_across_matrix(algorithm_name, method):
    """rustfits-written → fitsio-decoded bit-exact across the
    (algorithm, dither method) matrix."""
    algo_cls = getattr(rustfits, algorithm_name)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), seed=88)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=algo_cls(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method=method,
                    seed=2024,
                ),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            rust_out = f[1].read()
        with fitsio.FITS(fn) as f:
            fitsio_out = f[1].read()
        if _IS_MACOS:
            np.testing.assert_allclose(
                rust_out,
                fitsio_out,
                rtol=_MACOS_FP_RTOL,
                atol=_MACOS_FP_ATOL,
            )
        else:
            np.testing.assert_array_equal(rust_out, fitsio_out)


def test_default_seed_emits_one():
    """Quantize(seed=0) → on-disk ZDITHER0=1 (cfitsio default)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(seed=0),
            )
            assert f[1].header["ZDITHER0"] == 1
