"""
Tests for ImageHDU.read(mask_blank=True).

Covers:
    - mask_blank=False (default) returns a plain ndarray, never a
      MaskedArray
    - mask_blank=True with BLANK in the header returns a MaskedArray
      with True at sentinel pixels (compared in stored, pre-scaling
      space per the FITS spec)
    - mask_blank=True with no BLANK in header returns a MaskedArray
      with all-False mask for consistent return type
    - mask_blank=True composes with the BSCALE/BZERO unsigned-int
      trick (mask aligns with the u2/u4/u8 output)
    - mask_blank=True + scale=False returns raw + mask
    - mask_blank=True on float BITPIX rejects with ValueError
    - Post-reopen consistency
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_image(tmpdir, dtype, data, bscale=None, bzero=None, blank=None):
    """
    Create a one-HDU FITS file with the given stored data and optional
    BSCALE / BZERO / BLANK header cards.  Returns the path.
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
        if blank is not None:
            fits[0].header["BLANK"] = blank
    return fname


# -------------------- default behavior --------------------


def test_mask_blank_default_returns_plain_ndarray():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, -99, 4], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, blank=-99)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert not isinstance(got, np.ma.MaskedArray)
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got, stored)


# -------------------- mask_blank=True with BLANK set --------------------


def test_mask_blank_marks_sentinel_pixels():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, -99, 3, -99, 5], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, blank=-99)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        expected_mask = np.array([False, True, False, True, False], dtype=bool)
        np.testing.assert_array_equal(np.ma.getmaskarray(got), expected_mask)
        np.testing.assert_array_equal(got.data, stored)


def test_mask_blank_2d_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([[1, -99, 3], [-99, 5, 6]], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored, blank=-99)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        expected_mask = np.array(
            [[False, True, False], [True, False, False]], dtype=bool
        )
        np.testing.assert_array_equal(np.ma.getmaskarray(got), expected_mask)


def test_mask_blank_no_sentinels_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3, 4], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, blank=-99)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        # No element is masked.  For a plain (non-structured) ndarray,
        # the mask is materialized when the array is constructed with
        # an explicit mask arg; we just check that no element is masked.
        assert not np.ma.getmaskarray(got).any()


# -------------------- BLANK absent from header --------------------


def test_mask_blank_with_no_blank_keyword_returns_nomask():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, 2, 3], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored)  # no BLANK
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        # The constructor with no mask uses nomask; for plain ndarray
        # data, mask identity-equals np.ma.nomask.
        assert got.mask is np.ma.nomask


# -------------------- composition with scaling --------------------


def test_mask_blank_with_unsigned_trick():
    """
    BLANK is compared in stored (i2) space; output is in scaled (u2)
    space.  The mask aligns by position.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # i2 stored: [-32768, -1, 0, 32767]
        # BLANK=-32768 marks the first pixel.
        # BZERO=32768 → scaled u2: [0, 32767, 32768, 65535]
        stored = np.array([-32768, -1, 0, 32767], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0, blank=-32768)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)  # scale=True default
        assert isinstance(got, np.ma.MaskedArray)
        assert got.data.dtype == np.uint16
        np.testing.assert_array_equal(
            got.data, np.array([0, 32767, 32768, 65535], dtype="u2")
        )
        np.testing.assert_array_equal(
            np.ma.getmaskarray(got),
            np.array([True, False, False, False], dtype=bool),
        )


def test_mask_blank_with_general_scaling():
    """
    BLANK is compared in stored (i2) space; output is f8 after general
    scaling.  The mask aligns by position.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, -99, 3, -99], dtype="i2")
        fname = _make_image(
            tmpdir, "i2", stored, bscale=2.0, bzero=1.0, blank=-99
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        assert got.data.dtype == np.float64
        expected_data = stored.astype("f8") * 2.0 + 1.0
        np.testing.assert_array_equal(got.data, expected_data)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(got),
            np.array([False, True, False, True], dtype=bool),
        )


def test_mask_blank_with_scale_false_returns_raw_plus_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, -99, 3, -99], dtype="i2")
        fname = _make_image(tmpdir, "i2", stored, bzero=32768.0, blank=-99)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(scale=False, mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        assert got.data.dtype == np.int16
        np.testing.assert_array_equal(got.data, stored)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(got),
            np.array([False, True, False, True], dtype=bool),
        )


# -------------------- float BITPIX rejection --------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_mask_blank_rejected_on_float_bitpix(dtype):
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1.0, 2.0, 3.0], dtype=dtype)
        fname = _make_image(tmpdir, dtype, stored)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="float BITPIX"):
                fits[0].read(mask_blank=True)


# -------------------- post-reopen consistency --------------------


def test_mask_blank_post_reopen():
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([1, -99, 3, -99, 5], dtype="i4")
        fname = _make_image(tmpdir, "i4", stored, blank=-99)
        # Reopen (separate handle); confirm mask still computes from
        # the persisted BLANK card.
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(got),
            np.array([False, True, False, True, False], dtype=bool),
        )


# -------------------- BLANK + non-integer dtype edge case --------------


def test_blank_value_out_of_range_for_dtype_no_match():
    """
    BLANK header value that can't be represented in the BITPIX dtype
    (e.g. 300 on a u1 image) doesn't match any pixel — harmless
    all-False mask.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stored = np.array([0, 100, 200, 255], dtype="u1")
        fname = _make_image(tmpdir, "u1", stored, blank=300)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read(mask_blank=True)
        assert isinstance(got, np.ma.MaskedArray)
        assert not np.ma.getmaskarray(got).any()
