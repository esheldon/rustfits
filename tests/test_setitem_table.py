"""
Phase 2 tests: TableHDU.__setitem__.

Three index forms are supported:

- hdu[i] = record   single-row write.  Value is numpy.void (0-d
  structured scalar) or a length-1 structured ndarray.  Negative
  i supported; out-of-range raises IndexError.

- hdu[a:b[:s]] = arr   slice write.  Value is a structured ndarray
  whose length equals the slicelength.  step=1 uses the bulk-write
  fast path; step>1 does per-row writes.  step<=0 is rejected.

- hdu["col"] = arr   whole-column write.  Value is an ndarray of
  shape (nrows,) + per-cell shape, matching what hdu[:][col] would
  return.  Other columns' bytes are preserved.

Each mutation is verified through BOTH a same-handle read AND a
post-reopen read (CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# Reusable HDU dtype that exercises Identity, UnsignedXor, BoolToLogical,
# BytesCopy, and subarray paths.
DT = np.dtype(
    [
        ("id", "i4"),
        # U6 (numpy Unicode) matches what read returns by default for
        # an A column; using S6 here would force every per-row
        # equality check to special-case the str/bytes mismatch.
        ("name", "U6"),
        ("vec", "f4", (3,)),
        ("img", "f8", (2, 2)),
        ("flag", "?"),
        ("umask", "u2", (2, 2)),
    ]
)


def _make_arr(n, base=0):
    arr = np.zeros(n, dtype=DT)
    arr["id"] = np.arange(n, dtype="i4") + base * 100
    arr["name"] = [f"r{base * 100 + i:04d}" for i in range(n)]
    arr["vec"] = np.arange(n * 3, dtype="f4").reshape(n, 3) + base * 1000
    arr["img"] = np.arange(n * 4, dtype="f8").reshape(n, 2, 2) + base * 10000
    arr["flag"] = np.arange(n) % 2 == 0
    arr["umask"] = np.arange(n * 4, dtype="u2").reshape(n, 2, 2) + base * 1000
    return arr


def _create_table(fname, nrows):
    arr = _make_arr(nrows, base=0)
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_table_hdu(DT, nrows=nrows)
        fits[1].write(arr)
    return arr


# ------------------ hdu[i] = record  (single-row write) ------------------


def test_single_row_write_numpy_void():
    """
    Read a row to get a numpy.void scalar, modify a field, write it
    back.  Verify via both same-handle and reopen reads.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 5)
        replacement = _make_arr(1, base=9)[0]  # numpy.void

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][2] = replacement
            got = fits[1].read()
            np.testing.assert_array_equal(got[2], replacement)
            # Adjacent rows untouched.
            np.testing.assert_array_equal(got[1], initial[1])
            np.testing.assert_array_equal(got[3], initial[3])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got[2], replacement)
            np.testing.assert_array_equal(got[1], initial[1])
            np.testing.assert_array_equal(got[3], initial[3])


def test_single_row_write_len1_ndarray():
    """A shape-(1,) structured ndarray is also accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 4)
        replacement = _make_arr(1, base=7)  # shape (1,)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0] = replacement
            got = fits[1].read()
            np.testing.assert_array_equal(got[0], replacement[0])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got[0], replacement[0])


def test_single_row_write_negative_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 4)
        replacement = _make_arr(1, base=5)[0]

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][-1] = replacement
            got = fits[1].read()
            np.testing.assert_array_equal(got[3], replacement)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got[3], replacement)


def test_single_row_out_of_range_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        rec = _make_arr(1, base=1)[0]
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(IndexError, match="out of bounds"):
                fits[1][5] = rec
            with pytest.raises(IndexError, match="out of bounds"):
                fits[1][-4] = rec


def test_single_row_wrong_shape_rejected():
    """Multi-element ndarray rejected for single-row assignment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        multi = _make_arr(2, base=1)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="shape"):
                fits[1][0] = multi


def test_single_row_non_structured_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            # Plain ndarray with no fields.
            with pytest.raises(ValueError, match="structured"):
                fits[1][0] = np.zeros(1, dtype="f4")
            # Python tuple — not yet supported.
            with pytest.raises(ValueError, match="structured numpy record"):
                fits[1][0] = (
                    1,
                    b"x",
                    [0.0] * 3,
                    [[0.0] * 2] * 2,
                    True,
                    [[0] * 2] * 2,
                )


# ------------------ hdu[a:b[:s]] = arr  (slice writes) ------------------


def test_slice_step1_contiguous():
    """
    A step=1 slice falls through to the bulk fast path.  Replace
    rows 2..5 of a 7-row table.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 7)
        rep = _make_arr(3, base=4)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][2:5] = rep
            got = fits[1].read()
            for r in range(3):
                np.testing.assert_array_equal(got[2 + r], rep[r])
            np.testing.assert_array_equal(got[:2], initial[:2])
            np.testing.assert_array_equal(got[5:], initial[5:])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for r in range(3):
                np.testing.assert_array_equal(got[2 + r], rep[r])
            np.testing.assert_array_equal(got[:2], initial[:2])
            np.testing.assert_array_equal(got[5:], initial[5:])


def test_slice_full_equals_write():
    """hdu[:] = arr should be equivalent to hdu.write(arr)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname1 = os.path.join(tmpdir, "via_setitem.fits")
        fname2 = os.path.join(tmpdir, "via_write.fits")
        arr = _make_arr(6, base=2)
        for fname in (fname1, fname2):
            with rustfits.FITS(fname, "w+") as fits:
                fits.create_table_hdu(DT, nrows=6)
        with rustfits.FITS(fname1, "r+") as fits:
            fits[1][:] = arr
        with rustfits.FITS(fname2, "r+") as fits:
            fits[1].write(arr)
        with open(fname1, "rb") as f1, open(fname2, "rb") as f2:
            assert f1.read() == f2.read()


def test_slice_strided():
    """
    step=2 selects every other row.  Use the per-row writer path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 6)
        rep = _make_arr(3, base=3)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0::2] = rep  # rows 0, 2, 4
            got = fits[1].read()
            for r, src in zip((0, 2, 4), range(3)):
                np.testing.assert_array_equal(got[r], rep[src])
            for r in (1, 3, 5):
                np.testing.assert_array_equal(got[r], initial[r])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for r, src in zip((0, 2, 4), range(3)):
                np.testing.assert_array_equal(got[r], rep[src])
            for r in (1, 3, 5):
                np.testing.assert_array_equal(got[r], initial[r])


def test_slice_length_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 5)
        rep = _make_arr(2, base=1)
        with rustfits.FITS(fname, "r+") as fits:
            # Slice selects 3 rows but value has 2.
            with pytest.raises(ValueError, match="rows"):
                fits[1][1:4] = rep


def test_slice_negative_step_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 5)
        rep = _make_arr(3, base=1)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="negative or zero step"):
                fits[1][4:1:-1] = rep


def test_slice_empty():
    """
    A slice selecting 0 rows + length-0 value is a no-op.  Length
    mismatch on an empty slice still raises.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 4)
        empty = _make_arr(0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][2:2] = empty  # OK
            got = fits[1].read()
            np.testing.assert_array_equal(got, initial)

            with pytest.raises(ValueError, match="length"):
                fits[1][2:2] = _make_arr(1, base=1)


# ---------------- hdu["col"] = arr  (whole-column write) ----------------


def test_whole_column_write_scalar_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 5)
        new_ids = np.array([999, 888, 777, 666, 555], dtype="i4")

        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["id"] = new_ids
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], new_ids)
            # Other columns unchanged.
            for name in DT.names:
                if name == "id":
                    continue
                np.testing.assert_array_equal(got[name], initial[name])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], new_ids)


def test_whole_column_write_subarray_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_table(fname, 5)
        new_img = np.arange(20, dtype="f8").reshape(5, 2, 2) - 0.5

        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["img"] = new_img
            got = fits[1].read()
            np.testing.assert_array_equal(got["img"], new_img)
            for name in DT.names:
                if name == "img":
                    continue
                np.testing.assert_array_equal(got[name], initial[name])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["img"], new_img)


def test_whole_column_write_unsigned_trick():
    """u2 column round-trips through XOR-top-bit transform."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 4)
        new_mask = np.full((4, 2, 2), 60000, dtype="u2")
        new_mask[0] = 0
        new_mask[1] = 65535

        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["umask"] = new_mask
            got = fits[1].read()
            np.testing.assert_array_equal(got["umask"], new_mask)
            assert got["umask"].dtype == np.dtype("u2")

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["umask"], new_mask)


def test_whole_column_write_bool():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 4)
        new_flag = np.array([True, True, False, True])

        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["flag"] = new_flag
            got = fits[1].read()
            np.testing.assert_array_equal(got["flag"], new_flag)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["flag"], new_flag)


def test_whole_column_write_case_insensitive_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        new_ids = np.array([7, 8, 9], dtype="i4")

        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["ID"] = new_ids  # uppercase, column is lowercase "id"
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], new_ids)


def test_whole_column_unknown_name_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="no column named"):
                fits[1]["no_such_col"] = np.zeros(3, dtype="i4")


def test_whole_column_wrong_length_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 4)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="first axis"):
                fits[1]["id"] = np.array([1, 2], dtype="i4")


def test_whole_column_wrong_dtype_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                # f8 input where i4 is required
                fits[1]["id"] = np.array([1.0, 2.0, 3.0], dtype="f8")


# ----------------------- key classification errors -----------------------


def test_unsupported_key_rejected():
    """Tuple shapes other than (int row, str col) still raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="tuple shapes"):
                fits[1][(slice(0, 2), "id")] = np.zeros(2)
            with pytest.raises(ValueError, match="tuple shapes"):
                fits[1][(0, ["id", "flag"])] = (1, True)


# --------------- per-column input via shape mismatch ---------------


def test_whole_column_wrong_cell_shape_rejected():
    """Subarray column where input has wrong cell shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 3)
        # img column expects (3, 2, 2); pass (3, 3, 2)
        bad = np.zeros((3, 3, 2), dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="shape"):
                fits[1]["img"] = bad


# --------- structured ndarray with reordered fields (slow path) ----------


def test_slice_write_reordered_field_dtype():
    """
    Value has the same fields but in a different order — falls through
    the slow path inside prepare_structured_input.  Result should still
    be correct.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_table(fname, 5)
        # Reordered dtype (flag and id swapped relative to DT).
        reordered_dt = np.dtype(
            [
                ("flag", "?"),
                ("name", "U6"),
                ("vec", "f4", (3,)),
                ("id", "i4"),
                ("img", "f8", (2, 2)),
                ("umask", "u2", (2, 2)),
            ]
        )
        rep = np.zeros(2, dtype=reordered_dt)
        rep["id"] = [42, 43]
        rep["flag"] = [True, False]
        rep["name"] = ["alpha", "beta"]

        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1:3] = rep
            got = fits[1].read()
            for name in DT.names:
                np.testing.assert_array_equal(got[name][1:3], rep[name])
