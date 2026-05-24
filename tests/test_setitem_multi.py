"""
Tests for the multi-column / fancy-row / single-cell forms of
TableHDU.__setitem__:

- hdu[[name1, name2]] = arr        # multi-column subset write
- hdu[[i, j, k]] = arr             # fancy-row write (non-contiguous)
- hdu[row, "name"] = value         # single-cell write

These complement the existing single-row / slice / whole-column forms.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(tmpdir, nrows, fill=True):
    """A 4-column fixed table: id (i4), name (U6), flux (f8), img (3,2,2)."""
    fname = os.path.join(tmpdir, "t.fits")
    dt = np.dtype(
        [("id", "i4"), ("name", "U6"), ("flux", "f8"), ("img", "f4", (3, 2))]
    )
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows)
        if fill and nrows > 0:
            arr = np.zeros(nrows, dtype=dt)
            arr["id"] = np.arange(nrows)
            arr["name"] = [f"row{i:02d}" for i in range(nrows)]
            arr["flux"] = np.arange(nrows, dtype="f8") * 0.5
            for i in range(nrows):
                arr["img"][i] = np.arange(6, dtype="f4").reshape(3, 2) + i
            f[1].write(arr)
    return fname, dt


# ---------------------------------------------------------------------------
# Fancy rows: hdu[[i, j, k]] = arr
# ---------------------------------------------------------------------------


def test_fancy_rows_basic():
    """hdu[[0, 2, 4]] = arr writes 3 non-contiguous rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, dt = _make_table(tmp, nrows=5)
        new = np.zeros(3, dtype=dt)
        new["id"] = [100, 200, 300]
        new["name"] = ["alpha", "beta", "gamma"]
        new["flux"] = [1.0, 2.0, 3.0]
        for i in range(3):
            new["img"][i] = np.full((3, 2), 99 + i, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][[0, 2, 4]] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["id"][0] == 100
            assert got["id"][2] == 200
            assert got["id"][4] == 300
            # Untouched rows.
            assert got["id"][1] == 1
            assert got["id"][3] == 3
            assert got["name"][0] == "alpha"
            assert got["name"][4] == "gamma"


def test_fancy_rows_negative_indices():
    """Negative indices in the row list resolve like numpy."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, dt = _make_table(tmp, nrows=5)
        new = np.zeros(2, dtype=dt)
        new["id"] = [-1, -2]
        new["name"] = ["last", "second_last"]
        new["flux"] = [10.0, 20.0]
        with rustfits.FITS(fname, "r+") as f:
            f[1][[-1, -2]] = new  # rows 4 and 3
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["id"][4] == -1
            assert got["id"][3] == -2


def test_fancy_rows_length_mismatch_raises():
    """RHS length must match the row list length."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, dt = _make_table(tmp, nrows=5)
        bad = np.zeros(2, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="length"):
                f[1][[0, 2, 4]] = bad


def test_fancy_rows_out_of_range_raises():
    """Out-of-range row index raises IndexError."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, dt = _make_table(tmp, nrows=3)
        new = np.zeros(1, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises((IndexError, ValueError)):
                f[1][[5]] = new


def test_fancy_rows_on_vla_table_rejected():
    """Fancy-row writes on tables with VLA columns are deferred."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "v.fits")
        dt = np.dtype([("v", "O")])
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=3, var_dtypes={"v": "f4"})
        new = np.zeros(2, dtype=dt)
        new["v"][0] = np.array([1.0], dtype="f4")
        new["v"][1] = np.array([2.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="VLA"):
                f[1][[0, 2]] = new


# ---------------------------------------------------------------------------
# Multi-column subset: hdu[["a", "b"]] = arr
# ---------------------------------------------------------------------------


def test_multi_columns_basic():
    """hdu[['id', 'flux']] = arr rewrites those two columns across all rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        new = np.zeros(4, dtype=sub_dt)
        new["id"] = [1000, 2000, 3000, 4000]
        new["flux"] = [0.1, 0.2, 0.3, 0.4]
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "flux"]] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], new["id"])
            np.testing.assert_array_equal(got["flux"], new["flux"])
            # Untouched columns preserve their original values.
            assert got["name"][0] == "row00"
            np.testing.assert_array_equal(
                got["img"][1],
                (np.arange(6, dtype="f4").reshape(3, 2) + 1),
            )


def test_multi_columns_case_insensitive():
    """Column names match case-insensitively."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        sub_dt = np.dtype([("ID", "i4"), ("FLUX", "f8")])
        new = np.zeros(3, dtype=sub_dt)
        new["ID"] = [10, 20, 30]
        new["FLUX"] = [9.0, 8.0, 7.0]
        with rustfits.FITS(fname, "r+") as f:
            f[1][["ID", "FLUX"]] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [10, 20, 30])
            np.testing.assert_array_equal(got["flux"], [9.0, 8.0, 7.0])


def test_multi_columns_includes_vla():
    """Multi-column write where the subset includes a VLA column."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "mv.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O"), ("flux", "f8")])
        arr = np.zeros(3, dtype=dt)
        arr["id"] = [1, 2, 3]
        arr["flux"] = [0.5, 1.5, 2.5]
        for i in range(3):
            arr["lc"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=3, var_dtypes={"lc": "f4"})
            f[1].write(arr)
        sub_dt = np.dtype([("id", "i4"), ("lc", "O")])
        new = np.zeros(3, dtype=sub_dt)
        new["id"] = [10, 20, 30]
        for i in range(3):
            new["lc"][i] = np.array([100.0 + i], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "lc"]] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [10, 20, 30])
            np.testing.assert_array_equal(got["flux"], [0.5, 1.5, 2.5])
            for i in range(3):
                np.testing.assert_array_equal(got["lc"][i], [100.0 + i])


def test_multi_columns_duplicate_raises():
    """Duplicate name in the column list raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        new = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="duplicate"):
                f[1][["id", "id"]] = new


def test_multi_columns_missing_field_raises():
    """Value struct dtype missing one of the named columns raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        only_id = np.zeros(3, dtype=np.dtype([("id", "i4")]))
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="missing field"):
                f[1][["id", "flux"]] = only_id


def test_multi_columns_wrong_length_raises():
    """Value of wrong length raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        sub_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        new = np.zeros(2, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="NAXIS2"):
                f[1][["id", "flux"]] = new


def test_multi_columns_unknown_name_raises():
    """Unknown column name raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        sub_dt = np.dtype([("id", "i4"), ("nope", "f8")])
        new = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="no column"):
                f[1][["id", "nope"]] = new


# ---------------------------------------------------------------------------
# Single cell: hdu[row, "name"] = value
# ---------------------------------------------------------------------------


def test_cell_scalar_int_column():
    """hdu[i, 'id'] = 42 writes a single cell in a scalar i4 column."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r+") as f:
            f[1][1, "id"] = 999
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["id"][1] == 999
            assert got["id"][0] == 0  # untouched


def test_cell_scalar_float_column():
    """hdu[i, 'flux'] = 3.14 writes a single cell."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r+") as f:
            f[1][2, "flux"] = 3.14
        with rustfits.FITS(fname) as f:
            assert f[1].read()["flux"][2] == 3.14


def test_cell_negative_row():
    """Negative row index resolves like numpy."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r+") as f:
            f[1][-1, "id"] = 555
        with rustfits.FITS(fname) as f:
            assert f[1].read()["id"][3] == 555


def test_cell_subarray_column():
    """
    Subarray column (img has shape (3, 2) per row): RHS must be an
    ndarray matching that shape.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        patch = np.full((3, 2), 99.0, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][1, "img"] = patch
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["img"][1], patch)
            # Other rows untouched.
            np.testing.assert_array_equal(
                got["img"][0],
                np.arange(6, dtype="f4").reshape(3, 2),
            )


def test_cell_case_insensitive_name():
    """Column name lookup is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        with rustfits.FITS(fname, "r+") as f:
            f[1][0, "ID"] = 777
        with rustfits.FITS(fname) as f:
            assert f[1].read()["id"][0] == 777


def test_cell_vla_column():
    """Single-cell write on a VLA column appends new bytes to the heap."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "v.fits")
        dt = np.dtype([("v", "O")])
        arr = np.zeros(3, dtype=dt)
        for i in range(3):
            arr["v"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=3, var_dtypes={"v": "f4"})
            f[1].write(arr)
            pre_pc = int(f[1].header["PCOUNT"])
        new_cell = np.array([100.0, 101.0, 102.0, 103.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][1, "v"] = new_cell
        with rustfits.FITS(fname) as f:
            assert int(f[1].header["PCOUNT"]) == pre_pc + 4 * 4
            got = f[1].read()
            np.testing.assert_array_equal(got["v"][1], new_cell)
            # Other VLA cells untouched.
            np.testing.assert_array_equal(got["v"][0], arr["v"][0])
            np.testing.assert_array_equal(got["v"][2], arr["v"][2])


def test_cell_vla_string_column():
    """Single-cell write on a string VLA column accepts a str."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "s.fits")
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "alpha"
        arr["name"][1] = "beta"
        arr["name"][2] = "gamma"
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=3, var_dtypes={"name": "S"})
            f[1].write(arr)
        with rustfits.FITS(fname, "r+") as f:
            f[1][1, "name"] = "modified_long"
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "alpha"
            assert got["name"][1] == "modified_long"
            assert got["name"][2] == "gamma"


def test_cell_unknown_column_raises():
    """Unknown column name raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="no column"):
                f[1][0, "nope"] = 1


def test_cell_out_of_range_row_raises():
    """Out-of-range row index raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises((IndexError, ValueError)):
                f[1][10, "id"] = 1


def test_cell_subarray_wrong_shape_raises():
    """Subarray cell with wrong-shape RHS raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        bad = np.zeros((2, 2), dtype="f4")  # img wants (3, 2)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises((ValueError, Exception)):
                f[1][0, "img"] = bad


# ---------------------------------------------------------------------------
# Rejection of unsupported key shapes
# ---------------------------------------------------------------------------


def test_tuple_with_slice_row_rejected():
    """(slice, str) tuple is deferred; raises clearly."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="tuple shapes"):
                f[1][1:3, "id"] = np.zeros(2)


def test_tuple_with_col_list_rejected():
    """(int, [str, str]) tuple is deferred."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=3)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="tuple shapes"):
                f[1][0, ["id", "flux"]] = (1, 1.0)


def test_mixed_int_str_iterable_rejected():
    """Iterable mixing int and str raises (ambiguous)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_table(tmp, nrows=4)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][[0, "id"]] = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
