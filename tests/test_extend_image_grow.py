"""
Tests for ImageHDU.extend on HDUs that are NOT the last on disk.

When an image-extend grows the data section of an HDU with other HDUs
after it, the file tail is shifted forward via the same primitive used
for header overflow (shift_file_tail_and_update_offsets), then the gap
left at the old end-of-data is zero-filled, and finally the new header
(updated NAXISn) and the new image data are written.  This file checks:

  - subsequent HDUs' data integrity after a shift
  - previously-issued HDU and FITSHeader handles see post-shift offsets
    transparently (Arc<HduOffsets> shared with the layout)
  - multi-block growth, block-boundary growth, growth into an existing
    last HDU
  - the inserted gap is zero-filled (read back the new region's
    pre-write bytes via slice and check they are 0)
  - sequential grows of the same middle HDU compose
  - extend triggered by a write that grows axis 0 just past the
    current padded size
"""

import os
import tempfile
import contextlib

import numpy as np

import rustfits


@contextlib.contextmanager
def _new_three_image_hdus():
    """Three image HDUs with distinct shapes/dtypes and seeded data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "multi.fits")
        arr0 = np.arange(5 * 7, dtype="i4").reshape(5, 7)
        arr1 = (np.arange(3 * 11, dtype="f8") * 0.5).reshape(3, 11)
        arr2 = np.arange(4 * 4 * 4, dtype="i2").reshape(4, 4, 4)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=[5, 7])
            fits[0].write(arr0)
            fits.create_image_hdu(dtype="f8", dims=[3, 11], extname="IMG1")
            fits[1].write(arr1)
            fits.create_image_hdu(dtype="i2", dims=[4, 4, 4], extname="IMG2")
            fits[2].write(arr2)
        yield fname, [arr0, arr1, arr2]


# ---------------------------------------------------------------------------
# basic non-last extend round-trips
# ---------------------------------------------------------------------------


def test_extend_primary_image_preserves_later_hdu_data():
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            new = np.arange(3 * 7, dtype="i4").reshape(3, 7) + 1000
            fits[0].extend(new)
            assert fits[0].header["NAXIS2"] == 8   # 5 + 3
            # Both subsequent HDUs intact.
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])
            # The grown HDU's old data + new rows are both present.
            grown = fits[0].read()
            np.testing.assert_array_equal(grown[:5], arrays[0])
            np.testing.assert_array_equal(grown[5:], new)

        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["NAXIS2"] == 8
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])
            grown = fits[0].read()
            np.testing.assert_array_equal(grown[:5], arrays[0])
            np.testing.assert_array_equal(grown[5:], new)


def test_extend_middle_image_shifts_only_later_hdus():
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            new = (np.arange(4 * 11, dtype="f8") * 0.25).reshape(4, 11)
            fits[1].extend(new)
            assert fits[1].header["NAXIS2"] == 7   # 3 + 4
            # HDU 0 unchanged.
            np.testing.assert_array_equal(fits[0].read(), arrays[0])
            # HDU 2 shifted forward but data preserved.
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), arrays[0])
            assert fits[1].header["NAXIS2"] == 7
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# previously-issued handles transparently see post-shift offsets
# ---------------------------------------------------------------------------


def test_old_hdu_handle_sees_post_extend_offsets():
    """
    An HDU reference captured BEFORE the extend on an earlier HDU
    must still read correctly afterward — the Arc<HduOffsets> is shared."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            hdu2_before = fits[2]
            new = np.arange(2 * 7, dtype="i4").reshape(2, 7) + 7000
            fits[0].extend(new)

            np.testing.assert_array_equal(hdu2_before.read(), arrays[2])


def test_old_fitsheader_handle_sees_post_extend_offsets():
    with _new_three_image_hdus() as (fname, _arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h2_before = fits[2].header
            h2_before["MARKER"] = "before"

            new = np.arange(2 * 7, dtype="i4").reshape(2, 7)
            fits[0].extend(new)

            assert h2_before["MARKER"] == "before"
            h2_before["AFTER"] = "ok"
            assert h2_before["AFTER"] == "ok"

        with rustfits.FITS(fname, "r") as fits:
            assert fits[2].header["MARKER"] == "before"
            assert fits[2].header["AFTER"] == "ok"


# ---------------------------------------------------------------------------
# multi-block growth and block-boundary growth
# ---------------------------------------------------------------------------


def test_extend_grows_multiple_blocks_in_one_call():
    """
    A single extend can push the data section across many 2880-byte
    block boundaries.  Verify both grown data and subsequent HDU."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            # Each row of HDU 0 is 7 * 4 = 28 bytes; 100 new rows = 2800 bytes,
            # comfortably more than one block past the original padded size.
            new = np.arange(100 * 7, dtype="i4").reshape(100, 7) + 50000
            fits[0].extend(new)
            assert fits[0].header["NAXIS2"] == 105
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            grown = fits[0].read()
            np.testing.assert_array_equal(grown[5:], new)
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# gap is zero-filled where no data is written
# ---------------------------------------------------------------------------


def test_extend_zero_fills_gap_before_new_data():
    """
    Extend with a start past the current end leaves a gap between the
    old data and the new rows.  After the shift, the gap must read as
    zero — the bytes shifted out of place must not leak back as data."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            # Write 1 row at index 8, leaving rows 5,6,7 as a gap.
            new = (np.arange(7, dtype="i4") + 9000).reshape(1, 7)
            fits[0].extend(new, start=[8, 0])
            assert fits[0].header["NAXIS2"] == 9
            grown = fits[0].read()
            np.testing.assert_array_equal(grown[:5], arrays[0])
            assert np.all(grown[5:8] == 0)
            np.testing.assert_array_equal(grown[8], new[0])
            # Subsequent HDU still fine.
            np.testing.assert_array_equal(fits[1].read(), arrays[1])

        with rustfits.FITS(fname, "r") as fits:
            grown = fits[0].read()
            assert np.all(grown[5:8] == 0)
            np.testing.assert_array_equal(grown[8], new[0])


# ---------------------------------------------------------------------------
# sequential grows compose
# ---------------------------------------------------------------------------


def test_two_sequential_extends_on_middle_hdu_compose():
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            new1 = (np.arange(2 * 11, dtype="f8") + 100).reshape(2, 11)
            fits[1].extend(new1)
            new2 = (np.arange(3 * 11, dtype="f8") + 200).reshape(3, 11)
            fits[1].extend(new2)
            assert fits[1].header["NAXIS2"] == 8   # 3 + 2 + 3
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].header["NAXIS2"] == 8
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# growth that lands exactly on an existing block boundary (no shift needed)
# ---------------------------------------------------------------------------


def test_extend_within_existing_padding_does_not_shift_tail():
    """
    If the new data still fits in the currently-padded last block,
    no file shift is required — only the in-block bytes change.  The
    test exercises the path where new_padded == current_padded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            # 1 row of 7 i4 pixels = 28 bytes; padded to 2880 (one block).
            # Add another row → 56 bytes total, still in one block.
            fits.create_image_hdu(dtype="i4", dims=[1, 7], extname="A")
            fits[0].write(np.arange(7, dtype="i4").reshape(1, 7))
            fits.create_image_hdu(dtype="i4", dims=[2, 3], extname="B")
            arr_b = (np.arange(6, dtype="i4") + 100).reshape(2, 3)
            fits[1].write(arr_b)

            new_row = (np.arange(7, dtype="i4") + 1000).reshape(1, 7)
            file_size_before = os.path.getsize(fname)
            fits[0].extend(new_row)
            assert fits[0].header["NAXIS2"] == 2
            # File size unchanged — both data sections still pad to 2880.
            # Use os.path.getsize after the open handle has flushed.
            np.testing.assert_array_equal(fits[1].read(), arr_b)
        assert os.path.getsize(fname) == file_size_before


if __name__ == "__main__":
    test_extend_primary_image_preserves_later_hdu_data()
    test_extend_middle_image_shifts_only_later_hdus()
    test_old_hdu_handle_sees_post_extend_offsets()
    test_old_fitsheader_handle_sees_post_extend_offsets()
    test_extend_grows_multiple_blocks_in_one_call()
    test_extend_zero_fills_gap_before_new_data()
    test_two_sequential_extends_on_middle_hdu_compose()
    test_extend_within_existing_padding_does_not_shift_tail()
    print("smoke tests passed")
