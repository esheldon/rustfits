"""
Tests for `image_hdu[key] = value` write via __setitem__.

Symmetric with __getitem__: anything `img[key]` reads, `img[key] = v`
writes.  RHS is either a scalar (broadcast) or a shape-matching
numpy ndarray.  Dtype must match BITPIX exactly.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _new_image(tmpdir, dtype, dims, fill=0):
    """
    Create a fresh FITS file with a zero-filled image HDU, return
    the path.  Caller reopens in r+ mode to write."""
    fname = os.path.join(tmpdir, "img.fits")
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=tuple(dims))
        if fill != 0:
            n = int(np.prod(dims))
            fits.hdus[0].write(np.full(n, fill, dtype=dtype).reshape(dims))
    return fname


# ---------------------------------------------------------------------------
# Scalar pixel writes
# ---------------------------------------------------------------------------


def test_setitem_scalar_pixel_2d():
    """`img[i, j] = 1` writes a single pixel; read confirms."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1, 2] = 42
            assert fits[0][1, 2] == 42
            # Other pixels untouched.
            assert fits[0][0, 0] == 0
            assert fits[0][2, 3] == 0


def test_setitem_scalar_pixel_1d():
    """`img[i] = 1` on a 1-D image."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "f4", (10,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][3] = 2.5
            assert fits[0][3] == pytest.approx(2.5)
            assert fits[0][0] == 0


def test_setitem_scalar_pixel_3d():
    """`img[i, j, k] = 1` on a 3-D image."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i2", (2, 3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1, 2, 3] = 7
            assert fits[0][1, 2, 3] == 7
            assert fits[0][0, 0, 0] == 0


def test_setitem_negative_indices():
    """Negative indices route through the same normalization as reads."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][-1, -1] = 99
            assert fits[0][2, 3] == 99


# ---------------------------------------------------------------------------
# ndarray writes (shape match)
# ---------------------------------------------------------------------------


def test_setitem_full_image_array():
    """`img[...] = arr` overwrites the entire image."""
    data = np.arange(20, dtype="f8").reshape(4, 5) * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "f8", (4, 5))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][...] = data
            got = fits[0].read()
            np.testing.assert_array_equal(got, data)


def test_setitem_full_image_via_colon():
    """`img[:] = arr` works on a 1-D image."""
    data = np.arange(10, dtype="i4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (10,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][:] = data
            np.testing.assert_array_equal(fits[0].read(), data)


def test_setitem_slice_block():
    """`img[a:b, c:d] = block` writes a rectangular sub-region."""
    block = np.array([[10, 20], [30, 40]], dtype="i4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (4, 5))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1:3, 2:4] = block
            got = fits[0].read()
            np.testing.assert_array_equal(got[1:3, 2:4], block)
            # Outside the block stays zero.
            assert got[0, 0] == 0
            assert got[3, 4] == 0


def test_setitem_int_slice_mix():
    """`img[i, :] = row` writes one row of a 2-D image."""
    row = np.array([1, 2, 3, 4, 5], dtype="f4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "f4", (3, 5))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1, :] = row
            got = fits[0].read()
            np.testing.assert_array_equal(got[1], row)
            assert got[0].tolist() == [0, 0, 0, 0, 0]


def test_setitem_stepped_slice():
    """`img[::2] = arr` writes every other row (Tier C strided write)."""
    rows = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (4, 3))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][::2] = rows
            got = fits[0].read()
            # Rows 0 and 2 written, rows 1 and 3 still zero.
            np.testing.assert_array_equal(got[0], [1, 2, 3])
            np.testing.assert_array_equal(got[2], [4, 5, 6])
            np.testing.assert_array_equal(got[1], [0, 0, 0])
            np.testing.assert_array_equal(got[3], [0, 0, 0])


def test_setitem_stepped_2d():
    """`img[::2, ::2] = arr` — both axes strided."""
    block = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (4, 5))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][::2, ::2] = block
            got = fits[0].read()
            # Pixels written: (0,0)=1, (0,2)=2, (0,4)=3,
            #                 (2,0)=4, (2,2)=5, (2,4)=6.
            assert got[0, 0] == 1 and got[0, 2] == 2 and got[0, 4] == 3
            assert got[2, 0] == 4 and got[2, 2] == 5 and got[2, 4] == 6
            # Untouched pixels stay zero.
            assert got[1, 1] == 0
            assert got[0, 1] == 0


# ---------------------------------------------------------------------------
# Scalar broadcast
# ---------------------------------------------------------------------------


def test_setitem_scalar_broadcast_full():
    """`img[:] = 7` broadcasts the scalar to every pixel."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][:] = 7
            got = fits[0].read()
            assert np.all(got == 7)


def test_setitem_scalar_broadcast_slice():
    """`img[1:3, :] = 5` broadcasts to just those rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (4, 3))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1:3, :] = 5
            got = fits[0].read()
            assert np.all(got[1:3] == 5)
            assert np.all(got[0] == 0)
            assert np.all(got[3] == 0)


def test_setitem_zero_d_ndarray_broadcasts():
    """
    A 0-d numpy ndarray RHS is treated as a scalar broadcast
    (matches numpy semantics; np.ndim == 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "f4", (3, 3))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][...] = np.float32(2.5)
            got = fits[0].read()
            assert np.all(got == 2.5)


def test_setitem_numpy_scalar_broadcast():
    """numpy scalars (np.int32, np.float64, ...) broadcast cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i2", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1, 2] = np.int16(123)
            assert fits[0][1, 2] == 123


# ---------------------------------------------------------------------------
# Round-trip: write then read confirms exact bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    ["u1", "i2", "i4", "i8", "f4", "f8"],
)
def test_setitem_roundtrip_all_dtypes(dtype):
    """
    For every supported BITPIX dtype, an array written via
    `__setitem__` reads back identically."""
    data = np.arange(15, dtype=dtype).reshape(3, 5)
    if np.issubdtype(data.dtype, np.signedinteger):
        data = data - 3
    elif np.issubdtype(data.dtype, np.floating):
        data = data * 0.25 - 1.0
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, dtype, (3, 5))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][...] = data
            np.testing.assert_array_equal(fits[0].read(), data)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_setitem_shape_mismatch_raises():
    """RHS shape that doesn't match the slice's output shape raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            bad = np.zeros((2, 3), dtype="i4")
            with pytest.raises(ValueError, match="shape"):
                fits[0][1:3, 0:4] = bad


def test_setitem_dtype_mismatch_raises():
    """RHS dtype that doesn't match BITPIX raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 3))
        with rustfits.FITS(fname, "r+") as fits:
            bad = np.zeros((3, 3), dtype="i2")  # i2 vs i4
            with pytest.raises(ValueError, match="dtype"):
                fits[0][...] = bad


def test_setitem_out_of_range_index_raises():
    """Out-of-range integer index raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises((IndexError, ValueError)):
                fits[0][3, 0] = 1


def test_setitem_scalar_overflow_raises():
    """
    A scalar broadcast value that's out of range for the dtype
    raises (extract::<u8>(300) → OverflowError; extract::<i16>(99999)
    → OverflowError)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "u1", (3, 3))
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises((OverflowError, ValueError)):
                fits[0][0, 0] = 300


def test_setitem_read_only_raises():
    """Writing to a file opened read-only raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (3, 3))
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises((PermissionError, OSError, IOError)):
                fits[0][0, 0] = 1


def test_setitem_empty_slice_is_noop():
    """An empty slice (e.g. img[5:5]) writes nothing and doesn't raise."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _new_image(tmp, "i4", (4, 4))
        with rustfits.FITS(fname, "r+") as fits:
            before = fits[0].read()
            fits[0][2:2] = np.zeros((0, 4), dtype="i4")
            after = fits[0].read()
            np.testing.assert_array_equal(before, after)
