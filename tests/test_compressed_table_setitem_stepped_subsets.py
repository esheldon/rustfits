"""
ZTABLE Phase 6c-2d — stepped slices + subset-object __setitem__
on compressed tables.

Three surface additions:

  - hdu[a:b:s] = arr            (stepped slice; row writes)
  - hdu["name"][rows] = value   (single-column subset write)
  - hdu[[c1, c2]][rows] = value (multi-column subset write)

The stepped slice extends 6c-2b row writes; the subset writes
extend 6c-2c column writes with a rows= constraint.  All three
forms route through the same `setitem_compressed_fixed_rows`
primitive — stepped slice expands to a fancy-row list; subsets
narrow the column selection AND the row selection at the same
time.

Targeting a VLA column raises NotImplementedError pointing at
6c-2e.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits


def _have_funpack():
    return shutil.which("funpack") is not None


def _dt_basic():
    return np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])


def _basic_data(start, stop, dt=None):
    if dt is None:
        dt = _dt_basic()
    n = stop - start
    arr = np.zeros(n, dtype=dt)
    arr["id"] = np.arange(start, stop, dtype="i4")
    arr["v"] = np.arange(start, stop, dtype="f8") * 0.25
    arr["c"] = np.arange(start, stop, dtype="f4") * -1.0
    return arr


def _make_table(fname, *, nrows, ztilelen):
    dt = _dt_basic()
    data = _basic_data(0, nrows, dt)
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows, compress=True, ztilelen=ztilelen)
        f[1].write(data)
    return data, dt


def _check_round_trip(fname, expected, dt):
    with rustfits.FITS(fname, "r") as f:
        out = f[1].read()
    for col in dt.names:
        np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Stepped-slice row writes
# ---------------------------------------------------------------------


def test_setitem_stepped_slice_within_one_tile():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        # 10 rows at positions 10, 20, 30, ..., 100 (all in tile 0).
        chunk = _basic_data(70_000, 70_010, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][10:110:10] = chunk
        expected = original.copy()
        expected[10:110:10] = chunk
        _check_round_trip(fname, expected, dt)


def test_setitem_stepped_slice_spans_multiple_tiles():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=800, ztilelen=200)
        # 8 rows at 50, 150, 250, ..., 750 — touches every tile.
        chunk = _basic_data(80_000, 80_008, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][50:800:100] = chunk
        expected = original.copy()
        expected[50:800:100] = chunk
        _check_round_trip(fname, expected, dt)


def test_setitem_stepped_slice_step_2_long():
    """Large stepped slice — every other row."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        chunk = _basic_data(50_000, 50_000 + 300, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][0:600:2] = chunk
        expected = original.copy()
        expected[0:600:2] = chunk
        _check_round_trip(fname, expected, dt)


def test_setitem_stepped_slice_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        chunk = _basic_data(0, 3, _dt_basic())
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][0:100:10] = chunk  # selects 10, value has 3


def test_setitem_stepped_slice_negative_step_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        chunk = _basic_data(0, 5, _dt_basic())
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1][100:50:-10] = chunk
            assert "negative" in str(ei.value).lower()


# ---------------------------------------------------------------------
# Single-column subset writes: hdu["name"][rows] = value
# ---------------------------------------------------------------------


def test_subset_single_col_cell_write_with_int_row():
    """hdu['v'][50] = 3.14 — single-cell shortcut through the subset."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][50] = 3.14
        expected = original.copy()
        expected["v"][50] = 3.14
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_slice_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        new_chunk = np.arange(50, dtype="f8") * -0.5 + 1000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][100:150] = new_chunk
        expected = original.copy()
        expected["v"][100:150] = new_chunk
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_stepped_slice_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        new_chunk = np.arange(10, dtype="i4") + 9_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][0:100:10] = new_chunk
        expected = original.copy()
        expected["id"][0:100:10] = new_chunk
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_fancy_rows_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        idx = [10, 50, 250, 350]
        new_chunk = np.arange(4, dtype="i4") + 8_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][idx] = new_chunk
        expected = original.copy()
        for k, i in enumerate(idx):
            expected["id"][i] = new_chunk[k]
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_full_slice_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        new_v = np.arange(300, dtype="f8") + 100.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][:] = new_v
        expected = original.copy()
        expected["v"][:] = new_v
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_negative_int_row():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][-1] = -777.0
        expected = original.copy()
        expected["v"][-1] = -777.0
        _check_round_trip(fname, expected, dt)


def test_subset_single_col_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1]["v"][0:10] = np.arange(5, dtype="f8")


def test_subset_single_col_out_of_range_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(IndexError):
                f[1]["v"][100] = 1.0
            with pytest.raises(IndexError):
                f[1]["v"][-101] = 1.0


def test_subset_single_col_subarray_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("vec", "f4", (3,))])
        nrows = 200
        data = np.zeros(nrows, dtype=dt)
        data["id"] = np.arange(nrows, dtype="i4")
        data["vec"] = np.arange(nrows * 3, dtype="f4").reshape(nrows, 3)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, compress=True, ztilelen=80)
            f[1].write(data)
        # Slice write — value has shape (50, 3).
        new_chunk = np.arange(150, dtype="f4").reshape(50, 3) - 5000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["vec"][50:100] = new_chunk
            # Single-cell write through subset: scalar broadcast.
            f[1]["vec"][10] = 9.5
        expected = data.copy()
        expected["vec"][50:100] = new_chunk
        expected["vec"][10] = [9.5, 9.5, 9.5]
        _check_round_trip(fname, expected, dt)


# ---------------------------------------------------------------------
# Multi-column subset writes: hdu[["a", "b"]][rows] = value
# ---------------------------------------------------------------------


def test_subset_multi_col_record_write_with_int_row():
    """hdu[['id', 'v']][50] = record."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8")])
        rec = np.zeros(1, dtype=sub_dt)
        rec["id"] = 42
        rec["v"] = -1.5
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]][50] = rec[0]
        expected = original.copy()
        expected["id"][50] = 42
        expected["v"][50] = -1.5
        _check_round_trip(fname, expected, dt)


def test_subset_multi_col_slice_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        sub_dt = np.dtype([("id", "i4"), ("c", "f4")])
        sub = np.zeros(50, dtype=sub_dt)
        sub["id"] = np.arange(50, dtype="i4") + 7_000
        sub["c"] = np.arange(50, dtype="f4") - 50.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "c"]][100:150] = sub
        expected = original.copy()
        expected["id"][100:150] = sub["id"]
        expected["c"][100:150] = sub["c"]
        _check_round_trip(fname, expected, dt)


def test_subset_multi_col_stepped_slice_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8")])
        sub = np.zeros(10, dtype=sub_dt)
        sub["id"] = np.arange(10, dtype="i4") + 5_000
        sub["v"] = np.arange(10, dtype="f8") * 0.5
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]][0:100:10] = sub
        expected = original.copy()
        expected["id"][0:100:10] = sub["id"]
        expected["v"][0:100:10] = sub["v"]
        _check_round_trip(fname, expected, dt)


def test_subset_multi_col_fancy_rows_write():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=150)
        idx = [10, 50, 250, 350]
        sub_dt = np.dtype([("v", "f8"), ("c", "f4")])
        sub = np.zeros(4, dtype=sub_dt)
        sub["v"] = np.arange(4, dtype="f8") + 1_000
        sub["c"] = -np.arange(4, dtype="f4") - 1_000
        with rustfits.FITS(fname, "r+") as f:
            f[1][["v", "c"]][idx] = sub
        expected = original.copy()
        for k, i in enumerate(idx):
            expected["v"][i] = sub["v"][k]
            expected["c"][i] = sub["c"][k]
        _check_round_trip(fname, expected, dt)


def test_subset_multi_col_tolerates_extras_in_value():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8"), ("extra", "i4")])
        sub = np.zeros(20, dtype=sub_dt)
        sub["id"] = np.arange(20, dtype="i4") - 1
        sub["v"] = np.arange(20, dtype="f8") * 0.1
        sub["extra"] = 999  # ignored
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]][10:30] = sub
        expected = original.copy()
        expected["id"][10:30] = sub["id"]
        expected["v"][10:30] = sub["v"]
        _check_round_trip(fname, expected, dt)


def test_subset_multi_col_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8")])
        sub = np.zeros(5, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][["id", "v"]][10:30] = sub  # need 20


def test_subset_multi_col_missing_field_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        sub_dt = np.dtype([("id", "i4")])  # missing "v"
        sub = np.zeros(50, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][["id", "v"]][0:50] = sub


def test_subset_multi_col_unknown_column_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        sub_dt = np.dtype([("nope", "i4")])
        sub = np.zeros(10, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1][["nope"]][0:10] = sub
            assert "nope" in str(ei.value)


def test_subset_multi_col_non_vla_subset_on_vla_table_works():
    """Multi-col subset of non-VLA columns on a VLA-bearing table."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O"), ("c", "f4")])
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=10,
                compress=True,
                ztilelen=5,
                var_dtypes={"v": "f4"},
            )
            data = np.zeros(10, dtype=dt)
            data["id"] = np.arange(10, dtype="i4")
            data["c"] = np.arange(10, dtype="f4")
            for i in range(10):
                data["v"][i] = np.arange(i + 1, dtype="f4")
            f[1].write(data)
        sub_dt = np.dtype([("id", "i4"), ("c", "f4")])
        sub = np.zeros(5, dtype=sub_dt)
        sub["id"] = np.arange(5, dtype="i4") + 100
        sub["c"] = np.arange(5, dtype="f4") - 50
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "c"]][0:5] = sub
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["id"][0:5], sub["id"])
        np.testing.assert_array_equal(out["c"][0:5], sub["c"])
        # VLA column unchanged.
        for i in range(10):
            np.testing.assert_array_equal(
                out["v"][i], np.arange(i + 1, dtype="f4")
            )


# ---------------------------------------------------------------------
# Cross-tool: funpack reads files mutated through subset writes
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack required for cross-tool verification",
)
def test_funpack_decompresses_stepped_and_subset_modified_file():
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        stepped = _basic_data(70_000, 70_006, dt)
        single_col_chunk = np.arange(20, dtype="f8") * -7.5
        sub_dt = np.dtype([("id", "i4"), ("c", "f4")])
        multi_chunk = np.zeros(10, dtype=sub_dt)
        multi_chunk["id"] = np.arange(10, dtype="i4") + 33_000
        multi_chunk["c"] = np.arange(10, dtype="f4") - 7.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][0:600:100] = stepped  # stepped row write
            f[1]["v"][100:120] = single_col_chunk  # subset single-col
            f[1][["id", "c"]][
                [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
            ] = multi_chunk
        expected = original.copy()
        expected[0:600:100] = stepped
        expected["v"][100:120] = single_col_chunk
        for k, i in enumerate(range(500, 510)):
            expected["id"][i] = multi_chunk["id"][k]
            expected["c"][i] = multi_chunk["c"][k]
        out_path = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out_path, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out_path, "r") as f:
            cfit = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(cfit[col], expected[col])


# ---------------------------------------------------------------------
# Same-handle vs reopen parity for all three new forms
# ---------------------------------------------------------------------


def test_subset_writes_same_handle_and_reopen_agree():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        stepped = _basic_data(0, 30, dt)
        single_col_chunk = np.arange(50, dtype="i4") + 4_000
        sub_dt = np.dtype([("v", "f8"), ("c", "f4")])
        multi = np.zeros(20, dtype=sub_dt)
        multi["v"] = np.arange(20, dtype="f8")
        multi["c"] = -np.arange(20, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][0:300:10] = stepped
            f[1]["id"][100:150] = single_col_chunk
            f[1][["v", "c"]][200:220] = multi
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            re = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(same[col], re[col])
        expected = original.copy()
        expected[0:300:10] = stepped
        expected["id"][100:150] = single_col_chunk
        expected["v"][200:220] = multi["v"]
        expected["c"][200:220] = multi["c"]
        for col in dt.names:
            np.testing.assert_array_equal(re[col], expected[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
