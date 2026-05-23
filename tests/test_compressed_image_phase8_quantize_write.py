"""
ZIMAGE Phase 8: quantized float compressed image writes (commit 2 —
NO_DITHER + DITHER_1 end-to-end smoke).

This is the first user-reachable Phase 8 capability: writing a
float image as a tile-compressed HDU.  Per-tile bscale/bzero are
chosen from a noise estimate; the float pixels are quantized to
i32, then compressed with the chosen algorithm (Rice1/Gzip1/...).

This commit covers the foundational round-trips:
    - NO_DITHER + DITHER_1 round-trip via same-handle and reopen
    - f4 and f8 dtypes
    - Schema: 4 columns (COMPRESSED_DATA + ZSCALE + ZZERO +
      GZIP_COMPRESSED_DATA), ZQUANTIZ + ZDITHER0 cards
    - Cross-read with fitsio (rustfits-written → fitsio-decoded
      must match rustfits-decoded byte-for-byte)
    - GZIP fallback fires on constant-tile input (quantize_float
      returns None when delta == 0)

Commit 3 adds the full dither matrix (DITHER_2 + NaN), byte-exact
heap comparison with fitsio, more rejection paths, and a refactor
of write_compressed_image_data into smaller helpers.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


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


def test_default_quantize_for_float():
    """quantize=None on a float HDU uses the default Quantize
    (level=4.0, method='dither1', seed=0)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((20, 20), seed=33)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(20, 20)),
            )
            f[1].write(data)
            assert f[1].header["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
            out = f[1].read()
        assert np.abs(out - data).max() < 1.5


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
