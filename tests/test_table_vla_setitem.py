"""
Tests for TableHDU.__setitem__ on tables with variable-length columns.

Forms supported:
- hdu[i] = record         — single row, all columns
- hdu[a:b] = arr          — slice (step=1 only), all columns
- hdu["vla_col"] = arr    — whole VLA column (Object dtype)

Heap model: new cells are appended at the end of the existing heap
(old cells become orphans; PCOUNT grows monotonically).  Matches the
compressed-image __setitem__ pattern; a future repack() can compact
the heap when a workload demands it.

Strided slice writes (step > 1) on VLA tables are rejected; mixed-
dtype tables with fixed columns still take the existing fixed-only
path for whole-column writes of fixed columns.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vla_table(tmpdir, dtype, nrows, var_dtypes, fill_rows=None):
    """
    Create a one-HDU VLA table and (optionally) bulk-write `fill_rows`
    so the heap has some pre-existing content for orphan-tracking
    tests.  Returns the path.
    """
    fname = os.path.join(tmpdir, "t.fits")
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dtype, nrows=nrows, var_dtypes=var_dtypes)
        if fill_rows is not None:
            f[1].write(fill_rows)
    return fname


def _single_vla_table(tmpdir, inner="f4", nrows=4):
    """A 3-row table with one VLA float column, pre-populated."""
    dt = np.dtype([("v", "O")])
    arr = np.zeros(nrows, dtype=dt)
    for i in range(nrows):
        arr["v"][i] = np.arange(i + 1, dtype=inner)
    return _make_vla_table(
        tmpdir,
        dt,
        nrows,
        var_dtypes={"v": inner},
        fill_rows=arr,
    ), arr


def _mixed_table(tmpdir, nrows=4):
    """Mixed fixed + VLA columns, pre-populated."""
    dt = np.dtype([("id", "i4"), ("lc", "O"), ("flux", "f8")])
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows)
    arr["flux"] = np.arange(nrows, dtype="f8") * 0.5
    for i in range(nrows):
        arr["lc"][i] = np.arange(i + 1, dtype="f4")
    return _make_vla_table(
        tmpdir,
        dt,
        nrows,
        var_dtypes={"lc": "f4"},
        fill_rows=arr,
    ), arr


# ---------------------------------------------------------------------------
# Single-row write
# ---------------------------------------------------------------------------


def test_single_row_smaller_cell():
    """hdu[i] = record where the new VLA cell is smaller than the old."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp)
        new = np.zeros(1, dtype=arr.dtype)
        new["v"][0] = np.array([7.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][2] = new[0]
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][2], [7.0])
            # Other rows untouched.
            np.testing.assert_array_equal(got["v"][0], arr["v"][0])
            np.testing.assert_array_equal(got["v"][1], arr["v"][1])
            np.testing.assert_array_equal(got["v"][3], arr["v"][3])
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][2], [7.0])


def test_single_row_larger_cell_grows_pcount():
    """
    Writing a larger cell appends to heap end; PCOUNT must grow by
    the new cell's byte size (old cell becomes orphan, NOT freed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, inner="f4", nrows=3)
        with rustfits.FITS(fname, "r") as fits:
            old_pcount = int(fits[1].header["PCOUNT"])
        new = np.zeros(1, dtype=arr.dtype)
        new["v"][0] = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype="f4")
        new_cell_bytes = 5 * 4
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1] = new[0]
        with rustfits.FITS(fname) as fits:
            assert int(fits[1].header["PCOUNT"]) == old_pcount + new_cell_bytes
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][1], new["v"][0])
            np.testing.assert_array_equal(got["v"][0], arr["v"][0])
            np.testing.assert_array_equal(got["v"][2], arr["v"][2])


def test_single_row_empty_cell():
    """Empty VLA cell on a single-row write."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp)
        new = np.zeros(1, dtype=arr.dtype)
        new["v"][0] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0] = new[0]
            got = fits[1].read()
            assert got["v"][0].shape == (0,)
            np.testing.assert_array_equal(got["v"][1], arr["v"][1])


def test_single_row_negative_index():
    """Negative index works the same as for fixed tables."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        new = np.zeros(1, dtype=arr.dtype)
        new["v"][0] = np.array([99.0, 100.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][-1] = new[0]
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][3], new["v"][0])


def test_single_row_mixed_columns():
    """
    Single-row write on a table with both fixed and VLA columns —
    fixed cells get the new bytes, VLA cell appends to heap.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _mixed_table(tmp)
        new = np.zeros(1, dtype=arr.dtype)
        new["id"][0] = 99
        new["flux"][0] = 42.5
        new["lc"][0] = np.array([1.0, 2.0, 3.0, 4.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1] = new[0]
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["id"][1] == 99
            assert got["flux"][1] == 42.5
            np.testing.assert_array_equal(got["lc"][1], new["lc"][0])
            # Other rows untouched.
            assert got["id"][0] == arr["id"][0]
            np.testing.assert_array_equal(got["lc"][2], arr["lc"][2])


def test_single_row_repeated_writes_orphan_heap_grows():
    """
    Successive __setitem__ on the same row should all succeed; PCOUNT
    grows monotonically (orphans accumulate); the final read is correct.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=3)
        with rustfits.FITS(fname, "r+") as fits:
            with rustfits.FITS(fname, "r") as fr:
                p0 = int(fr[1].header["PCOUNT"])
            new = np.zeros(1, dtype=arr.dtype)
            for k in range(3):
                new["v"][0] = np.arange(2 + k, dtype="f4") + 100.0
                fits[1][0] = new[0]
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][0], new["v"][0])
            assert int(fits[1].header["PCOUNT"]) > p0


# ---------------------------------------------------------------------------
# Slice write (step=1 only)
# ---------------------------------------------------------------------------


def test_slice_write_step1():
    """hdu[a:b] = arr overwrites a contiguous range of rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=5)
        new = np.zeros(2, dtype=arr.dtype)
        new["v"][0] = np.array([100.0, 101.0], dtype="f4")
        new["v"][1] = np.array([200.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1:3] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][1], new["v"][0])
            np.testing.assert_array_equal(got["v"][2], new["v"][1])
            np.testing.assert_array_equal(got["v"][0], arr["v"][0])
            np.testing.assert_array_equal(got["v"][3], arr["v"][3])
            np.testing.assert_array_equal(got["v"][4], arr["v"][4])


def test_slice_write_full_range():
    """hdu[:] = arr overwrites every row."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=3)
        new = np.zeros(3, dtype=arr.dtype)
        new["v"][0] = np.array([1.0], dtype="f4")
        new["v"][1] = np.array([2.0, 3.0], dtype="f4")
        new["v"][2] = np.array([4.0, 5.0, 6.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][:] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for i in range(3):
                np.testing.assert_array_equal(got["v"][i], new["v"][i])


def test_slice_write_stepped_round_trip():
    """Stepped slice writes on VLA tables go through write_vla_data_strided."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=6)
        new = np.zeros(3, dtype=arr.dtype)
        new["v"][0] = np.array([1.5, 2.5], dtype="f4")
        new["v"][1] = np.array([], dtype="f4")
        new["v"][2] = np.array([7.0, 8.0, 9.0, 10.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0:6:2] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        for out_k, in_k in enumerate([0, 2, 4]):
            np.testing.assert_array_equal(got["v"][in_k], new["v"][out_k])
        # Untouched rows (1, 3, 5) still have the originals.
        for i in [1, 3, 5]:
            np.testing.assert_array_equal(got["v"][i], arr["v"][i])


def test_slice_write_negative_step_rejected():
    """Negative-step slice writes still raise (parity with fixed-cols)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        new = np.zeros(2, dtype=arr.dtype)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="negative"):
                fits[1][3:0:-1] = new


def test_fancy_rows_write_round_trip():
    """Fancy-row writes on VLA tables go through the strided helper."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=5)
        new = np.zeros(3, dtype=arr.dtype)
        new["v"][0] = np.array([10.0], dtype="f4")
        new["v"][1] = np.array([20.0, 21.0, 22.0], dtype="f4")
        new["v"][2] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][[1, 3, 4]] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        for out_k, in_k in enumerate([1, 3, 4]):
            np.testing.assert_array_equal(got["v"][in_k], new["v"][out_k])
        # Untouched rows (0, 2) still have the originals.
        for i in [0, 2]:
            np.testing.assert_array_equal(got["v"][i], arr["v"][i])


def test_fancy_rows_duplicate_indices_last_wins():
    """Duplicate rows in the fancy list — last write wins (numpy semantics)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        new = np.zeros(3, dtype=arr.dtype)
        new["v"][0] = np.array([1.0], dtype="f4")
        new["v"][1] = np.array([2.0], dtype="f4")
        new["v"][2] = np.array([3.0, 4.0], dtype="f4")
        # Indices [2, 2, 1] → row 2 written twice; last write wins.
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][[2, 2, 1]] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        # Row 2 reflects new[1] (second write); row 1 reflects new[2].
        np.testing.assert_array_equal(got["v"][2], new["v"][1])
        np.testing.assert_array_equal(got["v"][1], new["v"][2])


def test_fancy_rows_negative_indices_wrap():
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=5)
        new = np.zeros(2, dtype=arr.dtype)
        new["v"][0] = np.array([1.0, 2.0], dtype="f4")
        new["v"][1] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][[-1, -2]] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got["v"][4], new["v"][0])
        np.testing.assert_array_equal(got["v"][3], new["v"][1])


def test_slice_write_empty_noop():
    """An empty slice with empty value is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=3)
        empty = np.zeros(0, dtype=arr.dtype)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][2:2] = empty
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for i in range(3):
                np.testing.assert_array_equal(got["v"][i], arr["v"][i])


def test_slice_write_length_mismatch_raises():
    """RHS length must match slice length."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        bad = np.zeros(3, dtype=arr.dtype)  # slice picks 2 rows
        bad["v"][0] = np.array([1.0], dtype="f4")
        bad["v"][1] = np.array([2.0], dtype="f4")
        bad["v"][2] = np.array([3.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="rows"):
                fits[1][1:3] = bad


# ---------------------------------------------------------------------------
# Whole-column write
# ---------------------------------------------------------------------------


def test_whole_vla_column_write():
    """hdu['vla'] = arr rewrites every row's VLA cell."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        new = np.zeros(4, dtype="O")
        new[0] = np.array([10.0, 11.0], dtype="f4")
        new[1] = np.array([], dtype="f4")
        new[2] = np.array([20.0], dtype="f4")
        new[3] = np.array([30.0, 31.0, 32.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["v"] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for i in range(4):
                np.testing.assert_array_equal(got["v"][i], new[i])


def test_whole_vla_column_pcount_orphans_old():
    """Old heap cells stay (orphaned); PCOUNT grows by new total."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=3, inner="i4")
        with rustfits.FITS(fname, "r") as fits:
            old_pcount = int(fits[1].header["PCOUNT"])
        new = np.zeros(3, dtype="O")
        new[0] = np.array([100, 200, 300], dtype="i4")
        new[1] = np.array([400], dtype="i4")
        new[2] = np.array([500, 600], dtype="i4")
        new_total_bytes = (3 + 1 + 2) * 4
        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["v"] = new
        with rustfits.FITS(fname) as fits:
            assert (
                int(fits[1].header["PCOUNT"]) == old_pcount + new_total_bytes
            )


def test_whole_vla_column_does_not_touch_fixed_columns():
    """
    Writing a VLA column on a mixed-column table leaves the fixed
    columns' bytes unchanged.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _mixed_table(tmp, nrows=4)
        new = np.zeros(4, dtype="O")
        for i in range(4):
            new[i] = np.array([i * 100.0, i * 200.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["lc"] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["flux"], arr["flux"])
            for i in range(4):
                np.testing.assert_array_equal(got["lc"][i], new[i])


def test_whole_vla_column_case_insensitive():
    """Column-name lookup is case-insensitive (matches fixed setitem)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=3)
        new = np.zeros(3, dtype="O")
        for i in range(3):
            new[i] = np.array([i * 1.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["V"] = new  # uppercase, column is "v"
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for i in range(3):
                np.testing.assert_array_equal(got["v"][i], new[i])


def test_whole_vla_column_wrong_length_raises():
    """Object ndarray of wrong length raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=4)
        bad = np.zeros(3, dtype="O")
        for i in range(3):
            bad[i] = np.array([1.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="shape"):
                fits[1]["v"] = bad


def test_whole_vla_column_wrong_inner_dtype_raises():
    """Inner dtype mismatch raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, inner="f4", nrows=3)
        bad = np.zeros(3, dtype="O")
        for i in range(3):
            bad[i] = np.array([1], dtype="i8")  # i8 cells in an f4 col
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="VLA cell dtype"):
                fits[1]["v"] = bad


# ---------------------------------------------------------------------------
# Cross-cutting: heap grows, file shifts for non-last HDU
# ---------------------------------------------------------------------------


def test_non_last_hdu_setitem_shifts_tail():
    """
    A VLA setitem that grows the heap on a non-last HDU must shift
    the following HDUs forward; previously-issued HDU handles still
    work (shared Arc<HduOffsets>).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "two.fits")
        dt = np.dtype([("v", "O")])
        nrows = 3
        first = np.zeros(nrows, dtype=dt)
        for i in range(nrows):
            first["v"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": "f4"})
            f[1].write(first)
            f.create_image_hdu("i4", (5,), extname="AFTER")
            f[2].write(np.arange(5, dtype="i4") + 1000)

        new = np.zeros(1, dtype=dt)
        new["v"][0] = np.arange(50, dtype="f4")  # much bigger cell
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1] = new[0]
            # Same-handle: second HDU still readable at shifted offset.
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][1], new["v"][0])


# ---------------------------------------------------------------------------
# Fixed-column writes on a VLA-bearing table still work (existing path)
# ---------------------------------------------------------------------------


def test_fixed_column_write_on_vla_table_still_fixed_only():
    """
    Writing to a fixed column on a table that also has VLA columns
    should NOT trigger the heap path — only the fixed column's bytes
    change, PCOUNT and the heap are untouched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _mixed_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r") as fits:
            p0 = int(fits[1].header["PCOUNT"])
        new_ids = np.array([100, 200, 300, 400], dtype="i4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1]["id"] = new_ids
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], new_ids)
            # VLA column unchanged.
            for i in range(4):
                np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])
            # PCOUNT unchanged.
            assert int(fits[1].header["PCOUNT"]) == p0


def test_stepped_slice_mixed_table_round_trip():
    """Stepped slice on a mixed fixed+VLA table; both round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _mixed_table(tmp, nrows=6)
        new = np.zeros(3, dtype=arr.dtype)
        new["id"] = [10, 20, 30]
        new["flux"] = [1.0, 2.0, 3.0]
        new["lc"][0] = np.array([0.5], dtype="f4")
        new["lc"][1] = np.array([], dtype="f4")
        new["lc"][2] = np.array([4.0, 5.0, 6.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0:6:2] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        for out_k, in_k in enumerate([0, 2, 4]):
            assert got["id"][in_k] == new["id"][out_k]
            assert got["flux"][in_k] == new["flux"][out_k]
            np.testing.assert_array_equal(got["lc"][in_k], new["lc"][out_k])
        # Untouched rows preserved.
        for i in [1, 3, 5]:
            assert got["id"][i] == arr["id"][i]
            assert got["flux"][i] == arr["flux"][i]
            np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])


def test_fancy_rows_mixed_table_round_trip():
    """Fancy rows on a mixed fixed+VLA table."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _mixed_table(tmp, nrows=5)
        new = np.zeros(2, dtype=arr.dtype)
        new["id"] = [77, 88]
        new["flux"] = [-1.0, -2.0]
        new["lc"][0] = np.array([100.0, 200.0], dtype="f4")
        new["lc"][1] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][[1, 4]] = new
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        for out_k, in_k in enumerate([1, 4]):
            assert got["id"][in_k] == new["id"][out_k]
            assert got["flux"][in_k] == new["flux"][out_k]
            np.testing.assert_array_equal(got["lc"][in_k], new["lc"][out_k])
        # Untouched rows preserved.
        for i in [0, 2, 3]:
            assert got["id"][i] == arr["id"][i]
            assert got["flux"][i] == arr["flux"][i]
            np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])


def test_stepped_slice_then_repack_reclaims_orphans():
    """Multiple stepped writes orphan old cells; repack reclaims."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _single_vla_table(tmp, nrows=6)
        with rustfits.FITS(fname, "r+") as fits:
            for k in range(3):
                new = np.zeros(3, dtype=arr.dtype)
                for r in range(3):
                    new["v"][r] = np.arange(k + 2, dtype="f4")
                fits[1][0:6:2] = new
            p_before = int(fits[1].header["PCOUNT"])
            fits[1].repack()
            p_after = int(fits[1].header["PCOUNT"])
        assert p_after < p_before
        # Last write wins.
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
        for i in [0, 2, 4]:
            np.testing.assert_array_equal(
                got["v"][i], np.arange(4, dtype="f4")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
