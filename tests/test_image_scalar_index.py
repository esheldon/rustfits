"""Tests for `image_hdu[i, j, ...]` returning a numpy scalar when
every axis is integer-indexed.

Matches numpy semantics: each integer index reduces a dimension; if
ALL axes are integer-indexed, the result has zero dimensions and is
returned as a numpy scalar (e.g. `np.float32`, `np.int16`), not a
0-d ndarray.  Mixed slice + int still returns an ndarray.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_image(tmpdir, dtype, dims, fill=None):
    """Create a FITS file with one image HDU populated with arange()
    data so each pixel is easy to predict.  Returns the file path."""
    fname = os.path.join(tmpdir, "img.fits")
    n = int(np.prod(dims))
    arr = np.arange(n, dtype=dtype).reshape(dims)
    if fill is not None:
        arr = arr + fill
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=tuple(dims))
        fits.hdus[0].write(arr)
    return fname, arr


# ---------------------------------------------------------------------------
# Scalar return when all axes are integer-indexed
# ---------------------------------------------------------------------------


def test_2d_all_int_returns_scalar():
    """`hdu[i, j]` on a 2-D image returns a numpy scalar matching the
    BITPIX dtype."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f8", (3, 5))
        with rustfits.FITS(fname) as fits:
            val = fits[0][1, 2]
            assert isinstance(val, np.floating)
            assert not isinstance(val, np.ndarray)
            assert val == arr[1, 2]


def test_1d_int_returns_scalar():
    """`hdu[i]` on a 1-D image — every axis is int → scalar."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "i4", (10,))
        with rustfits.FITS(fname) as fits:
            val = fits[0][3]
            assert isinstance(val, np.integer)
            assert not isinstance(val, np.ndarray)
            assert val == arr[3]


def test_3d_all_int_returns_scalar():
    """`hdu[i, j, k]` on a 3-D image returns scalar."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "i2", (2, 3, 4))
        with rustfits.FITS(fname) as fits:
            val = fits[0][1, 2, 3]
            assert not isinstance(val, np.ndarray)
            assert val == arr[1, 2, 3]


# ---------------------------------------------------------------------------
# Scalar dtype matches BITPIX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,np_scalar_type",
    [
        ("u1", np.uint8),
        ("i2", np.int16),
        ("i4", np.int32),
        ("i8", np.int64),
        ("f4", np.float32),
        ("f8", np.float64),
    ],
)
def test_scalar_dtype_matches_bitpix(dtype, np_scalar_type):
    """The scalar returned for `hdu[i, j]` is the numpy type matching
    the image's BITPIX."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_image(tmp, dtype, (2, 2))
        with rustfits.FITS(fname) as fits:
            val = fits[0][0, 0]
            assert isinstance(val, np_scalar_type)


# ---------------------------------------------------------------------------
# Mixed slice + int still returns ndarray
# ---------------------------------------------------------------------------


def test_partial_indexing_returns_ndarray():
    """`hdu[i]` on a 2-D image reduces only the first axis → 1-D
    ndarray (the row), NOT a scalar."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f4", (3, 5))
        with rustfits.FITS(fname) as fits:
            row = fits[0][1]
            assert isinstance(row, np.ndarray)
            assert row.shape == (5,)
            np.testing.assert_array_equal(row, arr[1])


def test_slice_int_mix_returns_ndarray():
    """`hdu[1:3, 2]` mixes a slice and an int → 1-D ndarray."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f4", (5, 5))
        with rustfits.FITS(fname) as fits:
            sub = fits[0][1:3, 2]
            assert isinstance(sub, np.ndarray)
            assert sub.shape == (2,)
            np.testing.assert_array_equal(sub, arr[1:3, 2])


def test_length_one_slice_returns_ndarray():
    """`hdu[5:6, 6:7]` (all length-1 slices) returns a shape-(1,1)
    ndarray, NOT a scalar — slices preserve the axis even when their
    length is 1."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f4", (10, 10))
        with rustfits.FITS(fname) as fits:
            sub = fits[0][5:6, 6:7]
            assert isinstance(sub, np.ndarray)
            assert sub.shape == (1, 1)
            assert sub[0, 0] == arr[5, 6]


# ---------------------------------------------------------------------------
# Negative indexing + bounds checking
# ---------------------------------------------------------------------------


def test_negative_indices_scalar():
    """Negative indices on all axes return a scalar."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "i4", (3, 4))
        with rustfits.FITS(fname) as fits:
            val = fits[0][-1, -1]
            assert not isinstance(val, np.ndarray)
            assert val == arr[-1, -1]


def test_out_of_range_raises():
    """Out-of-range int raises (IndexError per the existing axis
    bounds check)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_image(tmp, "f4", (3, 4))
        with rustfits.FITS(fname) as fits:
            with pytest.raises((IndexError, ValueError)):
                _ = fits[0][3, 0]
            with pytest.raises((IndexError, ValueError)):
                _ = fits[0][0, -5]


# ---------------------------------------------------------------------------
# Ellipsis interaction
# ---------------------------------------------------------------------------


def test_ellipsis_plus_int_returns_ndarray():
    """`hdu[..., i]` on a 2-D image still leaves one axis → 1-D
    ndarray.  Not all axes are int."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f4", (3, 4))
        with rustfits.FITS(fname) as fits:
            col = fits[0][..., 2]
            assert isinstance(col, np.ndarray)
            assert col.shape == (3,)
            np.testing.assert_array_equal(col, arr[..., 2])


def test_ellipsis_only_returns_ndarray():
    """`hdu[...]` — no int axes — returns the full ndarray."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_image(tmp, "f4", (3, 4))
        with rustfits.FITS(fname) as fits:
            full = fits[0][...]
            assert isinstance(full, np.ndarray)
            assert full.shape == (3, 4)
            np.testing.assert_array_equal(full, arr)
