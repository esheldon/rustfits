"""
Tests for __setitem__ on TableHDU subset objects:
- SingleColumnSubset (from hdu["name"])
- ColumnSubset       (from hdu[["a", "b"]])

These complete the read-write surface so that anything `hdu[col][rows]`
reads, `hdu[col][rows] = v` can write.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_table(tmpdir, nrows=5):
    """A fixed table: id (i4), name (U6), flux (f8), img (3, 2)."""
    fname = os.path.join(tmpdir, "t.fits")
    dt = np.dtype(
        [("id", "i4"), ("name", "U6"), ("flux", "f8"), ("img", "f4", (3, 2))]
    )
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows)
        arr = np.zeros(nrows, dtype=dt)
        arr["id"] = np.arange(nrows)
        arr["name"] = [f"r{i:02d}" for i in range(nrows)]
        arr["flux"] = np.arange(nrows, dtype="f8") * 0.5
        for i in range(nrows):
            arr["img"][i] = np.arange(6, dtype="f4").reshape(3, 2) + i
        f[1].write(arr)
    return fname, dt


def _make_vla_table(tmpdir, nrows=4):
    """Mixed fixed + numeric VLA + string VLA."""
    fname = os.path.join(tmpdir, "v.fits")
    dt = np.dtype([("id", "i4"), ("lc", "O"), ("name", "O")])
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows)
    for i in range(nrows):
        arr["lc"][i] = np.arange(i + 1, dtype="f4")
        arr["name"][i] = f"row{i}"
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            dt,
            nrows=nrows,
            var_dtypes={"lc": "f4", "name": "S"},
        )
        f[1].write(arr)
    return fname, dt


# ---------------------------------------------------------------------------
# SingleColumnSubset.__setitem__ — hdu["name"][rows] = value
# ---------------------------------------------------------------------------


def test_single_col_subset_int_row():
    """hdu['id'][2] = 99 is the cell shortcut."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][2] = 99
        with rustfits.FITS(fname) as f:
            assert f[1].read()["id"][2] == 99


def test_single_col_subset_slice():
    """hdu['flux'][1:4] = arr writes a range of one column."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["flux"][1:4] = np.array([10.0, 20.0, 30.0])
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(
                got["flux"],
                [0.0, 10.0, 20.0, 30.0, 2.0],
            )
            # Other columns untouched.
            np.testing.assert_array_equal(got["id"], np.arange(5))


def test_single_col_subset_fancy_rows():
    """hdu['id'][[0, 2, 4]] = arr writes 3 non-contiguous cells."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][[0, 2, 4]] = np.array([100, 200, 300], dtype="i4")
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(
                got["id"],
                [100, 1, 200, 3, 300],
            )


def test_single_col_subset_negative_row():
    """Negative row resolves like numpy."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][-1] = 555
        with rustfits.FITS(fname) as f:
            assert f[1].read()["id"][4] == 555


def test_single_col_subset_full_slice():
    """hdu['name'][:] = arr is equivalent to whole-column write."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        new = np.array(["a", "b", "c", "d", "e"], dtype="U6")
        with rustfits.FITS(fname, "r+") as f:
            f[1]["name"][:] = new
        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[1].read()["name"], new)


def test_single_col_subset_subarray_column_slice():
    """Subarray column: each element of the value matches per-cell shape."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        patches = np.zeros((2, 3, 2), dtype="f4")
        patches[0] = np.full((3, 2), 7.0)
        patches[1] = np.full((3, 2), 8.0)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["img"][1:3] = patches
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["img"][1], patches[0])
            np.testing.assert_array_equal(got["img"][2], patches[1])


def test_single_col_subset_vla_numeric():
    """hdu['lc'][i] = ndarray works on numeric VLA columns."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_vla_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["lc"][1] = np.array([100.0, 101.0, 102.0], dtype="f4")
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["lc"][1], [100.0, 101.0, 102.0])


def test_single_col_subset_vla_string():
    """hdu['name'][i] = str works on string VLA columns."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_vla_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["name"][2] = "modified_long"
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][2] == "modified_long"
            assert got["name"][0] == "row0"
            assert got["name"][1] == "row1"


def test_single_col_subset_vla_slice():
    """hdu['lc'][i:j] = obj_array writes multiple VLA cells."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_vla_table(tmp)
        new = np.zeros(2, dtype="O")
        new[0] = np.array([1.0], dtype="f4")
        new[1] = np.array([2.0, 3.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1]["lc"][1:3] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["lc"][1], [1.0])
            np.testing.assert_array_equal(got["lc"][2], [2.0, 3.0])


def test_single_col_subset_length_mismatch_raises():
    """RHS length must match the row count."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        bad = np.array([1, 2, 3])  # slice picks 2
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="length"):
                f[1]["id"][1:3] = bad


def test_single_col_subset_empty_slice_noop():
    """Empty slice with any value is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][2:2] = np.array([], dtype="i4")
        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[1].read()["id"], np.arange(5))


# ---------------------------------------------------------------------------
# ColumnSubset.__setitem__ — hdu[["a", "b"]][rows] = value
# ---------------------------------------------------------------------------


def test_column_subset_int_row():
    """hdu[['id', 'flux']][1] = record writes one row of the subset."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        rec_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        rec = np.array([(77, 9.9)], dtype=rec_dt)[0]
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "flux"]][1] = rec
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["id"][1] == 77
            assert got["flux"][1] == 9.9
            # Other columns / rows untouched.
            assert got["id"][0] == 0
            assert got["name"][1] == "r01"


def test_column_subset_slice():
    """hdu[['id','flux']][1:4] = arr writes a slice across multiple cols."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        new = np.zeros(3, dtype=sub_dt)
        new["id"] = [10, 20, 30]
        new["flux"] = [1.1, 2.2, 3.3]
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "flux"]][1:4] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [0, 10, 20, 30, 4])
            np.testing.assert_array_equal(
                got["flux"],
                [0.0, 1.1, 2.2, 3.3, 2.0],
            )
            # name column untouched.
            assert got["name"][0] == "r00"


def test_column_subset_fancy_rows():
    """hdu[['id','flux']][[0, 2, 4]] = arr writes non-contiguous rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        new = np.zeros(3, dtype=sub_dt)
        new["id"] = [100, 200, 300]
        new["flux"] = [1.0, 2.0, 3.0]
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "flux"]][[0, 2, 4]] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [100, 1, 200, 3, 300])
            np.testing.assert_array_equal(
                got["flux"],
                [1.0, 0.5, 2.0, 1.5, 3.0],
            )


def test_column_subset_with_vla_column():
    """Column-subset slice write where one column is VLA."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_vla_table(tmp)
        sub_dt = np.dtype([("id", "i4"), ("lc", "O")])
        new = np.zeros(2, dtype=sub_dt)
        new["id"] = [99, 88]
        new["lc"][0] = np.array([7.0, 8.0], dtype="f4")
        new["lc"][1] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "lc"]][1:3] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [0, 99, 88, 3])
            np.testing.assert_array_equal(got["lc"][1], [7.0, 8.0])
            assert got["lc"][2].shape == (0,)
            # name VLA untouched.
            assert got["name"][1] == "row1"


def test_column_subset_missing_field_raises():
    """Value missing one of the subset's columns raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        only_id = np.zeros(3, dtype=np.dtype([("id", "i4")]))
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="missing field"):
                f[1][["id", "flux"]][0:3] = only_id


def test_column_subset_length_mismatch_raises():
    """Value length mismatch raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        bad = np.zeros(2, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="length"):
                f[1][["id", "flux"]][0:3] = bad


def test_column_subset_empty_rows_noop():
    """Empty row selector is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        empty = np.zeros(0, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "flux"]][2:2] = empty
        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[1].read()["id"], np.arange(5))


# ---------------------------------------------------------------------------
# Read/write symmetry: anything we can read via the subset, we can write
# ---------------------------------------------------------------------------


def test_round_trip_single_col_subset():
    """Read a slice of one column, modify, write back, re-read."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            got = f[1]["flux"][1:4]
            got = got * 2
            f[1]["flux"][1:4] = got
        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(
                f[1].read()["flux"],
                [0.0, 1.0, 2.0, 3.0, 2.0],
            )


def test_round_trip_column_subset():
    """Read a column subset over rows, modify, write back."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp)
        with rustfits.FITS(fname, "r+") as f:
            got = f[1][["id", "flux"]][1:4]
            got["id"] = got["id"] * 10
            got["flux"] = got["flux"] + 100
            f[1][["id", "flux"]][1:4] = got
        with rustfits.FITS(fname) as f:
            got2 = f[1].read()
            np.testing.assert_array_equal(got2["id"], [0, 10, 20, 30, 4])
            np.testing.assert_array_equal(
                got2["flux"],
                [0.0, 100.5, 101.0, 101.5, 2.0],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
