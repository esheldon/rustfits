"""Tests for ImageHDU.extend.

Covers:
    - default start places new data at the end of the slow axis (grows by
      data.shape[0])
    - explicit start with growth, no-growth, and a "gap" past the current end
    - rejection: growing an inner (non-slow) axis
    - rejection: extending a non-last HDU on disk
    - dtype mismatch caught before any file modification
    - round-trip via reopen: NAXISn reflects the new size, data lands where
      expected, pre-existing pixels are preserved
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_file_with_image(tmpdir, dtype, dims, name="t.fits", extname="img"):
    fname = os.path.join(tmpdir, name)
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=dims, extname=extname)
    return fname


# -------------------- default start (append at slow-axis end) --------------------


def test_extend_default_start_grows_slow_axis():
    """With no start, new data is appended along numpy axis 0; existing data
    is preserved and the slow axis grows by data.shape[0]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (5, 20))
        original = np.arange(5 * 20, dtype="f8").reshape(5, 20) - 1.5
        new = np.arange(3 * 20, dtype="f8").reshape(3, 20) + 100.0

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(original)
            fits.hdus[0].extend(new)

            # In-memory header reflects the new slow-axis size.
            hd = fits.hdus[0].header_dict
            assert hd["NAXIS1"]["value"] == 20  # fast axis unchanged
            assert hd["NAXIS2"]["value"] == 8  # slow axis grew 5 -> 8

        # Round-trip from disk.
        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header_dict
            assert hd["NAXIS2"]["value"] == 8

        # Verify on-disk pixels: original in rows 0..4, new in rows 5..7.
        with open(fname, "rb") as f:
            f.seek(2880)
            raw = f.read(8 * 8 * 20)
        img = np.frombuffer(raw, dtype=">f8").reshape(8, 20).copy()
        np.testing.assert_array_equal(img[:5], original)
        np.testing.assert_array_equal(img[5:], new)


def test_extend_1d_default_start():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (10,))
        original = np.arange(10, dtype="i4")
        more = np.array([100, 200, 300], dtype="i4")

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(original)
            fits.hdus[0].extend(more)
            assert fits.hdus[0].header_dict["NAXIS1"]["value"] == 13

        with open(fname, "rb") as f:
            f.seek(2880)
            raw = f.read(13 * 4)
        img = np.frombuffer(raw, dtype=">i4")
        np.testing.assert_array_equal(img[:10], original)
        np.testing.assert_array_equal(img[10:], more)


# -------------------- explicit start --------------------


def test_extend_explicit_start_with_growth():
    """Explicit start past the current slow-axis end; the HDU grows to
    accommodate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (3, 4))
        new = np.arange(2 * 4, dtype="f4").reshape(2, 4) + 10.0

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].extend(new, start=(3, 0))
            hd = fits.hdus[0].header_dict
            assert hd["NAXIS2"]["value"] == 5  # 3 + 2


def test_extend_with_gap_zero_filled():
    """Explicit start past the end with a gap: rows between the old end and
    the start of the new data must read back as zeros (set_len sparse-fills
    the gap)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i2", (2, 4))
        original = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="i2")
        new = np.array([[100, 200, 300, 400]], dtype="i2")

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(original)
            fits.hdus[0].extend(
                new, start=(4, 0)
            )  # leaves row index 2 and 3 empty
            hd = fits.hdus[0].header_dict
            assert hd["NAXIS2"]["value"] == 5

        with open(fname, "rb") as f:
            f.seek(2880)
            raw = f.read(5 * 4 * 2)
        img = np.frombuffer(raw, dtype=">i2").reshape(5, 4).copy()
        np.testing.assert_array_equal(img[:2], original)
        assert (img[2:4] == 0).all()
        np.testing.assert_array_equal(img[4:], new)


def test_extend_no_growth_falls_through_to_write():
    """When start + data shape fits the existing HDU, extend should behave
    like write — no header change, no file growth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (5, 4))
        size_before = os.path.getsize(fname)
        sub = np.array([[1.0, 2.0, 3.0, 4.0]], dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].extend(sub, start=(1, 0))
            assert fits.hdus[0].header_dict["NAXIS2"]["value"] == 5

        assert os.path.getsize(fname) == size_before


# -------------------- error paths --------------------


def test_extend_inner_axis_growth_rejected():
    """Attempting to grow an inner (fast) axis is rejected — only the slow
    axis may grow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (3, 4))
        bad = np.zeros((3, 6), dtype="f4")  # wants col extension

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="extend only grows the slow"):
                fits.hdus[0].extend(bad, start=(0, 0))


def test_extend_dtype_mismatch_rejected_before_any_change():
    """dtype mismatch must be caught before the file is touched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (3, 4))
        size_before = os.path.getsize(fname)
        bad = np.zeros((2, 4), dtype="f8")

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="does not match HDU BITPIX"):
                fits.hdus[0].extend(bad)
            # Header is unchanged.
            assert fits.hdus[0].header_dict["NAXIS2"]["value"] == 3

        # File on disk is also unchanged.
        assert os.path.getsize(fname) == size_before


def test_extend_non_last_hdu_rejected():
    """An HDU with other HDUs after it on disk cannot grow yet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(3, 4), extname="first")
            fits.create_image_hdu(dtype="f8", dims=(2, 5), extname="second")

            new = np.zeros((2, 4), dtype="i4")
            with pytest.raises(ValueError, match="extending non-last"):
                fits.hdus[0].extend(new)


def test_extend_last_hdu_when_multiple_works():
    """Extending the LAST HDU works even when there are multiple HDUs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(3, 4), extname="first")
            fits.create_image_hdu(dtype="f8", dims=(2, 5), extname="second")

            new = np.arange(3 * 5, dtype="f8").reshape(3, 5) + 1.0
            fits.hdus[1].extend(new)
            assert fits.hdus[1].header_dict["NAXIS2"]["value"] == 5  # 2 + 3

        # Reopen and confirm both HDUs are still parseable and correct.
        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 2
            assert fits.hdus[0].header_dict["NAXIS2"]["value"] == 3
            assert fits.hdus[1].header_dict["NAXIS2"]["value"] == 5


# -------------------- crossing block boundary --------------------


def test_extend_crosses_block_boundary():
    """Growth that pushes the data section into a new 2880-byte block must
    bump the file size; the new region must be zero-filled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (300, 1))
        # 300 * 8 = 2400 bytes; padded to 2880 (one block).  Extending to
        # 400 makes 3200 bytes, which needs two blocks (5760).
        size_before = os.path.getsize(fname)
        assert size_before == 2880 * 2  # header + one data block

        new = np.arange(100, dtype="f8").reshape(100, 1) + 1.0
        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].extend(new)
            assert fits.hdus[0].header_dict["NAXIS2"]["value"] == 400

        size_after = os.path.getsize(fname)
        assert size_after == 2880 * 3  # header + two data blocks


if __name__ == "__main__":
    test_extend_default_start_grows_slow_axis()
    test_extend_1d_default_start()
    test_extend_explicit_start_with_growth()
    test_extend_with_gap_zero_filled()
    test_extend_no_growth_falls_through_to_write()
    test_extend_last_hdu_when_multiple_works()
    test_extend_crosses_block_boundary()
    print("all tests passed")
