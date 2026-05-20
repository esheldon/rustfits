"""
Tests for ImageHDU.__getitem__: numpy-style sliced reads.

Covers integer indexing (with axis dropping), slice indexing (with and
without step), Ellipsis, multi-axis combinations, error cases, and a
3-D example exercising the full coalescing logic.

Each test writes a known arange-pattern image with rustfits and then
compares the sliced read against the equivalent numpy slice of the
reference array.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_test_image(fname, shape, dtype="f8"):
    """
    Create a FITS file with one image HDU filled with arange data,
    returning the in-memory reference array."""
    reference = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=list(shape))
        fits[0].write(reference)
    return reference


# ----------------------- whole-array round-trips ------------------------


def test_full_read_via_colon():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            assert np.array_equal(fits[0][:], reference)


def test_full_read_via_ellipsis():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            assert np.array_equal(fits[0][...], reference)


def test_full_read_via_two_colons():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            assert np.array_equal(fits[0][:, :], reference)


# --------------------------- integer indexing ----------------------------


def test_int_index_drops_axis():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            row = fits[0][5]
            assert row.shape == (20,)
            assert np.array_equal(row, reference[5])


def test_negative_int_index():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            assert np.array_equal(fits[0][-1], reference[-1])
            assert np.array_equal(fits[0][-3], reference[-3])


def test_int_on_fast_axis_drops_axis():
    """
    Int index on the *fast* numpy axis (== FITS NAXIS1).  Requires the
    engine to do one read per outer position rather than a single strip."""
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            col = fits[0][:, 7]
            assert col.shape == (10,)
            assert np.array_equal(col, reference[:, 7])


def test_two_int_indices_returns_scalar_shape():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            v = fits[0][3, 7]
            assert v.shape == ()
            assert v == reference[3, 7]


# --------------------------- contiguous slices ----------------------------


def test_slice_on_slow_axis():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][2:7]
            assert sub.shape == (5, 20)
            assert np.array_equal(sub, reference[2:7])


def test_slice_2d_subregion():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][2:7, 5:15]
            assert sub.shape == (5, 10)
            assert np.array_equal(sub, reference[2:7, 5:15])


# --------------------------- strided slices ----------------------------


def test_step_on_slow_axis():
    """
    Step on the *slow* axis: each output row reads a full contiguous
    strip; the outer iteration skips every other row."""
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][::2]
            assert sub.shape == (5, 20)
            assert np.array_equal(sub, reference[::2])


def test_step_on_fast_axis():
    """
    Step on the *fast* axis: strip coalescing stops at the fast axis,
    so each strided element is read individually."""
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][:, ::2]
            assert sub.shape == (10, 10)
            assert np.array_equal(sub, reference[:, ::2])


def test_step_on_both_axes():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][::3, ::4]
            assert sub.shape == reference[::3, ::4].shape
            assert np.array_equal(sub, reference[::3, ::4])


# --------------------------- ellipsis ----------------------------


def test_ellipsis_at_end():
    shape = (3, 5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][1, ...]
            assert sub.shape == (5, 7)
            assert np.array_equal(sub, reference[1, ...])


def test_ellipsis_at_start():
    shape = (3, 5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][..., 3]
            assert sub.shape == (3, 5)
            assert np.array_equal(sub, reference[..., 3])


# --------------------------- 3-D combinations ----------------------------


def test_3d_full_slice_with_step():
    shape = (4, 6, 8)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][1:4, ::2, 2:6]
            assert sub.shape == reference[1:4, ::2, 2:6].shape
            assert np.array_equal(sub, reference[1:4, ::2, 2:6])


def test_3d_int_plus_slice_combination():
    shape = (4, 6, 8)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][2, 1:5]
            assert sub.shape == reference[2, 1:5].shape
            assert np.array_equal(sub, reference[2, 1:5])


# --------------------------- dtype / endianness ----------------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8", "f4", "f8"])
def test_dtype_round_trip(dtype):
    shape = (5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, f"t-{dtype}.fits")
        reference = _make_test_image(fname, shape, dtype=dtype)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][1:4, 2:6]
            assert sub.dtype == np.dtype(dtype)
            assert np.array_equal(sub, reference[1:4, 2:6])


# ----------------------- empty / boundary slices ------------------------


def test_empty_slice_on_slow_axis():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            empty = fits[0][5:5]
            assert empty.shape == (0, 20)


def test_empty_slice_on_fast_axis():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            empty = fits[0][:, 7:7]
            assert empty.shape == (10, 0)


def test_slice_past_end_clamps():
    """
    Slice bounds past the end of the axis are clamped by Python's
    slice.indices, matching numpy behavior."""
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        reference = _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[0][5:1000]
            assert sub.shape == (5, 20)
            assert np.array_equal(sub, reference[5:1000])


# --------------------------- error cases ----------------------------


def test_int_out_of_bounds_raises():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(IndexError):
                _ = fits[0][1000]
            with pytest.raises(IndexError):
                _ = fits[0][-1000]


def test_negative_step_rejected():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                _ = fits[0][::-1]


def test_too_many_indices_rejected():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                _ = fits[0][1, 2, 3]


def test_double_ellipsis_rejected():
    shape = (3, 5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                _ = fits[0][..., 1, ...]


def test_unsupported_index_type_rejected():
    shape = (10, 20)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _make_test_image(fname, shape)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                _ = fits[0]["not-an-index"]


if __name__ == "__main__":
    # Quick smoke run without pytest.
    test_full_read_via_colon()
    test_int_index_drops_axis()
    test_slice_2d_subregion()
    test_step_on_fast_axis()
    test_ellipsis_at_end()
    test_3d_full_slice_with_step()
    test_dtype_round_trip("f8")
    test_empty_slice_on_slow_axis()
    test_negative_step_rejected()
    print("smoke tests passed")
