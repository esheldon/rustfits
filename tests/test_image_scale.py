"""
Tests for ImageHDU BSCALE/BZERO scaling on read.

Covers:
    - Trivial scaling (BSCALE=1, BZERO=0) is a no-op, returns
      BITPIX native dtype
    - Unsigned-int trick for BITPIX 16/32/64 (signed-int stored
      with BZERO=2^(n-1), read back as matching unsigned dtype)
    - Signed-byte trick (BITPIX=8, BZERO=-128, read back as i1)
    - General scaling (BSCALE != 1 or BZERO not the trick offset)
      promotes to f8
    - scale=False opt-out returns raw stored values
    - __getitem__ applies scaling (no opt-out, matches table
      convention)
    - Scalar return from all-int indexing is in the scaled dtype
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_image(tmpdir, dtype, data, bscale=None, bzero=None):
    """
    Create a one-HDU FITS file with the given stored data and
    optional BSCALE / BZERO header cards.  Returns the path.
    """
    fname = os.path.join(tmpdir, "t.fits")
    arr = np.asarray(data, dtype=dtype)
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=arr.shape)
        fits[0].write(arr)
        if bscale is not None:
            fits[0].header["BSCALE"] = bscale
        if bzero is not None:
            fits[0].header["BZERO"] = bzero
    return fname


# -------------------- trivial: no scaling cards --------------------


def test_no_scaling_cards_returns_bitpix_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3, 4], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, stored)


def test_trivial_bscale_bzero_is_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([5, 6, 7], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bscale=1.0, bzero=0.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got, stored)


# -------------------- unsigned-int trick --------------------


def test_unsigned_trick_i2_to_u2():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([-32768, -1, 0, 32767], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(
            got, np.array([0, 32767, 32768, 65535], dtype="u2")
        )


def test_unsigned_trick_i4_to_u4():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([-2147483648, -1, 0, 2147483647], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored, bzero=2147483648.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint32
        np.testing.assert_array_equal(
            got, np.array([0, 2147483647, 2147483648, 4294967295], dtype="u4")
        )


def test_unsigned_trick_i8_to_u8():
    with tempfile.TemporaryDirectory() as tmpdir:
        lo = np.int64(np.iinfo("i8").min)
        hi = np.int64(np.iinfo("i8").max)
        stored = np.array([lo, -1, 0, hi], dtype="i8")
        fname = _make_image(tmpdir, "i8", stored, bzero=9223372036854775808.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint64
        expected = np.array(
            [
                0,
                np.iinfo("u8").max // 2,
                np.uint64(2) ** 63,
                np.iinfo("u8").max,
            ],
            dtype="u8",
        )
        np.testing.assert_array_equal(got, expected)


def test_signed_byte_trick_u1_to_i1():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([0, 127, 128, 255], dtype="u1")
        fname = _make_image(tmpdir, "u1", stored, bzero=-128.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.int8
        np.testing.assert_array_equal(
            got, np.array([-128, -1, 0, 127], dtype="i1")
        )


# -------------------- general scaling --------------------


def test_general_scaling_returns_f8():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3, 4], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bscale=2.5, bzero=10.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.float64
        expected = stored.astype("f8") * 2.5 + 10.0
        np.testing.assert_array_equal(got, expected)


def test_general_scaling_bscale_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([10, 20, 30], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored, bscale=0.5)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, stored.astype("f8") * 0.5)


def test_general_scaling_bzero_only_non_trick():
    """
    BZERO that does NOT match the unsigned-trick offset goes
    through the General path → output is f8.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=7.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, stored.astype("f8") + 7.0)


def test_general_scaling_on_float_bitpix():
    """
    Float BITPIX with non-trivial BSCALE/BZERO also goes through
    the General path → f8 output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1.0, 2.0, 3.0], dtype="f4")
        fname = _make_image(tmpdir, "f4", stored, bscale=2.0, bzero=1.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.float64
        np.testing.assert_array_almost_equal(
            got, stored.astype("f8") * 2.0 + 1.0
        )


# -------------------- scale=False opt-out --------------------


def test_scale_false_returns_raw_unsigned_trick():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([-1, 0, 32767], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(scale=False)
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got, stored)


def test_scale_false_returns_raw_general():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored, bscale=2.5, bzero=10.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(scale=False)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, stored)


# -------------------- __getitem__ scaling --------------------


def test_getitem_slice_applies_scaling():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.arange(-3, 5, dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0][2:5]
        assert got.dtype == np.uint16
        expected = (stored[2:5].astype("i4") + 32768).astype("u2")
        np.testing.assert_array_equal(got, expected)


def test_getitem_all_int_returns_scaled_scalar():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([[-1, 0], [1, 100]], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0)
        with rustfits.FITS(fname, "r") as fits:
            val = fits[0][0, 0]
        assert isinstance(val, np.uint16)
        assert int(val) == 32767


def test_getitem_general_scaling_scalar():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([[1, 2], [3, 4]], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bscale=2.0, bzero=5.0)
        with rustfits.FITS(fname, "r") as fits:
            val = fits[0][1, 1]
        assert isinstance(val, np.float64)
        assert val == pytest.approx(13.0)


# -------------------- post-reopen consistency --------------------


@pytest.mark.parametrize(
    "bitpix_dtype,bzero",
    [
        ("i2", 32768.0),
        ("i4", 2147483648.0),
        ("u1", -128.0),
    ],
)
def test_unsigned_trick_post_reopen(bitpix_dtype, bzero):
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([0, 1, 2, 3], dtype=bitpix_dtype)
        fname = _make_image(tmpdir, bitpix_dtype, stored, bzero=bzero)
        # First handle (same as in _make_image) is already closed.
        # Reopen and verify the unsigned trick still applies cleanly
        # — i.e. the BZERO card was persisted and is parsed back
        # correctly.
        with rustfits.FITS(fname, "r") as fits:
            scaled = fits[0].read()
            raw = fits[0].read(scale=False)
        # Scaled dtype matches the trick output type.
        if bitpix_dtype == "u1":
            assert scaled.dtype == np.int8
        elif bitpix_dtype == "i2":
            assert scaled.dtype == np.uint16
        elif bitpix_dtype == "i4":
            assert scaled.dtype == np.uint32
        # Raw matches stored.
        np.testing.assert_array_equal(raw, stored)
