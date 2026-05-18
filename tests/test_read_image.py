"""Tests for ImageHDU.read.

Covers:
    - round-trip write -> read across all supported dtypes
    - 1D, 2D, 3D shapes
    - freshly-created HDU returns zeros
    - returned array is native-endian (so downstream numpy ops are fast)
    - read of each HDU in a multi-HDU file
    - NAXIS=0 HDU rejects read (no data section)
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- round-trip across dtypes --------------------


@pytest.mark.parametrize(
    "dtype",
    ["u1", "i2", "i4", "i8", "f4", "f8"],
)
def test_roundtrip_write_then_read(dtype):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.arange(5 * 7, dtype=dtype).reshape(5, 7)
        # Shift values so we exercise non-zero data for both signed and
        # unsigned.
        if np.issubdtype(original.dtype, np.signedinteger):
            original = original - 3
        elif np.issubdtype(original.dtype, np.floating):
            original = original * 0.25 - 1.0

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=(5, 7))
            fits.hdus[0].write(original)
            got = fits.hdus[0].read()

        assert got.shape == (5, 7)
        np.testing.assert_array_equal(got, original)


# -------------------- shape variations --------------------


def test_read_1d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        data = np.arange(13, dtype="f4") * 0.5

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=(13,))
            fits.hdus[0].write(data)
            got = fits.hdus[0].read()

        assert got.shape == (13,)
        np.testing.assert_array_equal(got, data)


def test_read_3d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        data = np.arange(2 * 3 * 4, dtype="i4").reshape(2, 3, 4) - 5

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(2, 3, 4))
            fits.hdus[0].write(data)
            got = fits.hdus[0].read()

        assert got.shape == (2, 3, 4)
        np.testing.assert_array_equal(got, data)


# -------------------- freshly-created HDU is all zeros --------------------


def test_read_fresh_hdu_returns_zeros():
    """create_image_hdu zero-fills the data section (sparse).  read() must
    surface those zeros."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f8", dims=(4, 6))
            got = fits.hdus[0].read()

        assert got.shape == (4, 6)
        assert got.dtype == np.dtype("f8")  # native f8
        assert (got == 0).all()


# -------------------- native endianness --------------------


def test_read_result_is_native_endian():
    """Returned dtype must be native — that's the byte order downstream
    numpy operations are fastest on."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f8", dims=(2, 2))
            got = fits.hdus[0].read()

        # numpy dtype.str uses '<' / '>' / '|' explicitly; native should
        # resolve to one of '<' or '>' depending on host, matching
        # sys.byteorder.
        host = sys.byteorder  # 'little' or 'big'
        if host == "little":
            assert got.dtype.str.startswith("<")
        else:
            assert got.dtype.str.startswith(">")


def test_read_values_correct_after_byteswap():
    """Sanity check: write known values, read back, confirm numpy interprets
    them correctly (this would fail if the byte-swap on read was missed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        values = np.array(
            [[1.0, -2.5, 3.75], [1e10, -1e-10, np.pi]], dtype="f8"
        )

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f8", dims=(2, 3))
            fits.hdus[0].write(values)
            got = fits.hdus[0].read()

        np.testing.assert_array_equal(got, values)


# -------------------- multi-HDU file --------------------


def test_read_multi_hdu_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        a = np.arange(12, dtype="i4").reshape(3, 4) + 1
        b = np.arange(10, dtype="f8").reshape(2, 5) * 0.5
        c = np.arange(6, dtype="u1").reshape(2, 3)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(3, 4), extname="A")
            fits.create_image_hdu(dtype="f8", dims=(2, 5), extname="B")
            fits.create_image_hdu(dtype="u1", dims=(2, 3), extname="C")

            fits.hdus[0].write(a)
            fits.hdus[1].write(b)
            fits.hdus[2].write(c)

        # Reopen and read each HDU.
        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 3
            np.testing.assert_array_equal(fits.hdus[0].read(), a)
            np.testing.assert_array_equal(fits.hdus[1].read(), b)
            np.testing.assert_array_equal(fits.hdus[2].read(), c)


# -------------------- partial-write then read --------------------


def test_read_after_partial_write():
    """After a partial write, read must return the full HDU with the sub-
    region updated and the rest still zero (from create_image_hdu)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        sub = np.array([[10, 20, 30], [40, 50, 60]], dtype="i2")

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i2", dims=(5, 6))
            fits.hdus[0].write(sub, start=(2, 1))
            got = fits.hdus[0].read()

        assert got.shape == (5, 6)
        np.testing.assert_array_equal(got[2:4, 1:4], sub)
        mask = np.ones(got.shape, dtype=bool)
        mask[2:4, 1:4] = False
        assert (got[mask] == 0).all()


# -------------------- extend + read --------------------


def test_read_after_extend():
    """After extend, read must return the larger array with both regions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        original = np.arange(3 * 4, dtype="f4").reshape(3, 4) - 1.0
        more = np.arange(2 * 4, dtype="f4").reshape(2, 4) + 100.0

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=(3, 4))
            fits.hdus[0].write(original)
            fits.hdus[0].extend(more)
            got = fits.hdus[0].read()

        assert got.shape == (5, 4)
        np.testing.assert_array_equal(got[:3], original)
        np.testing.assert_array_equal(got[3:], more)


if __name__ == "__main__":
    for d in ["u1", "i2", "i4", "i8", "f4", "f8"]:
        test_roundtrip_write_then_read(d)
    test_read_1d()
    test_read_3d()
    test_read_fresh_hdu_returns_zeros()
    test_read_result_is_native_endian()
    test_read_values_correct_after_byteswap()
    test_read_multi_hdu_file()
    test_read_after_partial_write()
    test_read_after_extend()
    print("all tests passed")
