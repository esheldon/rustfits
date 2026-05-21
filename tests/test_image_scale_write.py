"""
Tests for ImageHDU write-side BSCALE/BZERO support (the unsigned-int
trick).

Covers:
    - create_image_hdu accepts unsigned-int dtypes (u2/u4/u8) and
      signed-byte (i1); emits BITPIX=signed + BZERO=2^(n-1) in the
      header
    - write() / __setitem__ / extend() accept scaled-dtype ndarray
      input and reverse-transform on the fly to the stored dtype
    - Stored bytes on disk match the signed-int representation;
      reads recover the unsigned dtype via the existing read-side
      scaling
    - Symmetric round-trip: write u2/u4/u8/i1 → read returns the
      same unsigned dtype with the same values
    - The user's input ndarray is not modified by write
    - Mismatched dtype on write is rejected with a clear error
    - Writing in the BITPIX (stored) dtype still works as an
      opt-out path
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- create_image_hdu accepts unsigned dtypes -------


@pytest.mark.parametrize(
    "dtype,expected_bitpix,expected_bzero",
    [
        ("u2", 16, 32768),
        ("u4", 32, 2147483648),
        # u8 BZERO=2^63 is too big for header_dict's i64; we just
        # check BITPIX here and verify the round-trip elsewhere.
        ("u8", 64, None),
        ("i1", 8, -128),
    ],
)
def test_create_unsigned_emits_bitpix_and_bzero(
    dtype, expected_bitpix, expected_bzero
):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=(4,))
            assert fits[0].header["BITPIX"] == expected_bitpix
            assert fits[0].header["BSCALE"] == 1
            if expected_bzero is not None:
                assert fits[0].header["BZERO"] == expected_bzero


# -------------------- round-trip via write() ------------------------


def test_roundtrip_u2():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.array([0, 1, 32768, 65535], dtype="u2")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(got, original)


def test_roundtrip_u4():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.array([0, 1, 2147483648, 4294967295], dtype="u4")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u4", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint32
        np.testing.assert_array_equal(got, original)


def test_roundtrip_u8():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.array(
            [
                0,
                1,
                np.uint64(2) ** 63,
                np.iinfo("u8").max,
            ],
            dtype="u8",
        )
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u8", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint64
        np.testing.assert_array_equal(got, original)


def test_roundtrip_i1():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.array([-128, -1, 0, 127], dtype="i1")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i1", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.int8
        np.testing.assert_array_equal(got, original)


def test_roundtrip_2d_u2():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.arange(12, dtype="u2").reshape(3, 4) + 30000
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(got, original)


# -------------------- stored bytes are signed-int representation -----


def test_stored_bytes_are_signed_representation():
    """
    Confirm the on-disk bytes for an u2 write are the i2 (signed-int)
    storage form, so the read-side scaling needs to apply BZERO to
    recover the unsigned values.  Verifies by reading with scale=False.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.array([0, 32768, 65535], dtype="u2")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=original.shape)
            fits[0].write(original)
        with rustfits.FITS(fname, "r") as fits:
            raw = fits[0].read(scale=False)
        assert raw.dtype == np.int16
        # original 0 → stored -32768; 32768 → 0; 65535 → 32767
        np.testing.assert_array_equal(
            raw, np.array([-32768, 0, 32767], dtype="i2")
        )


# -------------------- __setitem__ accepts scaled dtype ---------------


def test_setitem_slice_with_scaled_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=(6,))
            # Whole-array initial fill (already covered by write tests).
            fits[0].write(np.zeros(6, dtype="u2"))
            # Slice write in u2 (the scaled dtype).
            fits[0][1:4] = np.array([100, 32768, 65535], dtype="u2")
            got = fits[0].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(
            got, np.array([0, 100, 32768, 65535, 0, 0], dtype="u2")
        )


# -------------------- extend() accepts scaled dtype ------------------


def test_extend_with_scaled_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=(3,))
            fits[0].write(np.array([10, 20, 30], dtype="u2"))
            # Append three more rows via extend (default appends after
            # the current data).
            fits[0].extend(np.array([40000, 50000, 60000], dtype="u2"))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[0].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(
            got,
            np.array([10, 20, 30, 40000, 50000, 60000], dtype="u2"),
        )


# -------------------- stored-dtype input still works -----------------


def test_write_stored_dtype_input_still_accepted():
    """
    A user who wants to bypass the reverse-transform can pass the
    BITPIX-native (signed) dtype directly.  This works because the
    fast path in normalize_input_dtype accepts BITPIX-native input
    unchanged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=(3,))
            # i2 input matches BITPIX storage directly.
            fits[0].write(np.array([-32768, 0, 32767], dtype="i2"))
        with rustfits.FITS(fname, "r") as fits:
            got_scaled = fits[0].read()
            got_raw = fits[0].read(scale=False)
        # Read with default scaling sees u2 values via the BZERO
        # offset.
        np.testing.assert_array_equal(
            got_scaled, np.array([0, 32768, 65535], dtype="u2")
        )
        # Raw read recovers the i2 we wrote.
        np.testing.assert_array_equal(
            got_raw, np.array([-32768, 0, 32767], dtype="i2")
        )


# -------------------- dtype mismatch rejected ------------------------


def test_dtype_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=(3,))
            # f4 doesn't match BITPIX=16 or scaled u2 → rejected.
            with pytest.raises(ValueError, match=r"BITPIX=16.*scaled 'u2'"):
                fits[0].write(np.array([1.0, 2.0, 3.0], dtype="f4"))


# -------------------- user input is not modified ---------------------


def test_write_does_not_modify_user_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        arr = np.array([0, 1, 32768, 65535], dtype="u2")
        arr_copy = arr.copy()
        ptr_before = arr.ctypes.data
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u2", dims=arr.shape)
            fits[0].write(arr)
        np.testing.assert_array_equal(arr, arr_copy)
        assert arr.ctypes.data == ptr_before
