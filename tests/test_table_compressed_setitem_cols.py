"""
ZTABLE Phase 6c-2c — column / cell / multi-column __setitem__ on
compressed tables.

`CompressedTableHDU.__setitem__` for the three column-targeted forms:

  - hdu["col"] = arr           (whole-column; all tiles)
  - hdu["col"][r] = v          (single cell; one tile; subset form)
  - hdu[[c1, c2]] = arr        (multi-column subset; structured RHS)

Affected tiles are decoded for ONLY the selected columns, the rows'
bytes are overwritten via the shared per-cell transform, and the
slabs are re-encoded + appended to the heap end.  Non-selected
columns' descriptors stay unchanged.  Old blobs become orphans
(reclaimed by repack()).

Targeting a VLA column (whole-column / cell / multi-col that
includes a VLA name) raises NotImplementedError pointing at the
6c-2e phase.
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


def _make_table(fname, *, nrows, ztilelen, compress=True):
    dt = _dt_basic()
    data = _basic_data(0, nrows, dt)
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            dt, nrows=nrows, compress=compress, ztilelen=ztilelen
        )
        f[1].write(data)
    return data, dt


def _check_round_trip(fname, expected, dt):
    with rustfits.FITS(fname, "r") as f:
        out = f[1].read()
    for col in dt.names:
        np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Whole-column writes
# ---------------------------------------------------------------------


def test_setitem_whole_column_round_trip():
    """
    hdu['col'] = arr replaces a single column across all rows.  Other
    columns are untouched.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        new_v = np.arange(600, dtype="f8") * -2.5
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
        modified = original.copy()
        modified["v"] = new_v
        _check_round_trip(fname, modified, dt)


def test_setitem_whole_column_case_insensitive():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        new_id = np.arange(400, dtype="i4") + 50_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["ID"] = new_id  # uppercase variant
        modified = original.copy()
        modified["id"] = new_id
        _check_round_trip(fname, modified, dt)


def test_setitem_whole_column_unknown_column_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        arr = np.zeros(100, dtype="i4")
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1]["nope"] = arr
            assert "nope" in str(ei.value)


def test_setitem_whole_column_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1]["id"] = np.zeros(50, dtype="i4")  # wrong length


def test_setitem_whole_column_does_not_touch_other_columns():
    """
    PCOUNT grows only by the cost of the one rewritten column —
    other columns' descriptors stay pointing at the original blobs.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        new_id = np.arange(400, dtype="i4") + 100
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"] = new_id
        modified = original.copy()
        modified["id"] = new_id
        _check_round_trip(fname, modified, dt)


def test_setitem_whole_column_in_empty_table_noop():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_basic()
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=0, compress=True, ztilelen=1)
        # Nothing to write, nothing to read — no-op.
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"] = np.zeros(0, dtype="i4")


def test_setitem_whole_column_subarray():
    """Whole-column write for a subarray (TDIM) column."""
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
        new_vec = np.arange(nrows * 3, dtype="f4").reshape(nrows, 3) - 1000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["vec"] = new_vec
        modified = data.copy()
        modified["vec"] = new_vec
        _check_round_trip(fname, modified, dt)


def test_setitem_whole_column_repack_reclaims_orphans():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        new_v = np.arange(600, dtype="f8") + 0.5
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
            pcount_before = int(f[1].header["PCOUNT"])
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
        assert pcount_after < pcount_before
        modified = original.copy()
        modified["v"] = new_v
        _check_round_trip(fname, modified, dt)


# ---------------------------------------------------------------------
# Single-cell writes
# ---------------------------------------------------------------------


def test_setitem_cell_scalar_int():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][50] = 99_999
        modified = original.copy()
        modified["id"][50] = 99_999
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_scalar_float():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][50] = 12.5
            f[1]["c"][100] = -0.125
        modified = original.copy()
        modified["v"][50] = 12.5
        modified["c"][100] = -0.125
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_numpy_scalar():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][50] = np.int32(7)
        modified = original.copy()
        modified["id"][50] = 7
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_negative_row_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][-1] = 42
        modified = original.copy()
        modified["id"][-1] = 42
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_out_of_range_row_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(IndexError):
                f[1]["id"][100] = 5
            with pytest.raises(IndexError):
                f[1]["v"][-101] = 5.0


def test_setitem_cell_unknown_column_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1]["nope"][10] = 1
            assert "nope" in str(ei.value)


def test_setitem_cell_subarray_column_with_full_vector():
    """Cell write to a subarray column accepts a per-cell-shape ndarray."""
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
        new_vec = np.array([1.0, 2.0, 3.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1]["vec"][50] = new_vec
        modified = data.copy()
        modified["vec"][50] = new_vec
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_subarray_column_broadcasts_scalar():
    """A scalar RHS broadcasts across all elements of a subarray cell."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("vec", "f4", (3,))])
        nrows = 50
        data = np.zeros(nrows, dtype=dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, compress=True, ztilelen=25)
            f[1].write(data)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["vec"][10] = 5.0
        modified = data.copy()
        modified["vec"][10] = [5.0, 5.0, 5.0]
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_modifies_only_one_column():
    """
    Cell write changes ONE cell — same-handle + reopen reads agree
    on the new value AND on every other cell.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"][150] = -1
            same = f[1].read()
        modified = original.copy()
        modified["id"][150] = -1
        for col in dt.names:
            np.testing.assert_array_equal(same[col], modified[col])
        _check_round_trip(fname, modified, dt)


def test_setitem_cell_then_repack_reclaims_orphan():
    """
    Hammer the same cell N times; the heap grows monotonically.
    Repack shrinks it back.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        with rustfits.FITS(fname, "r+") as f:
            for k in range(5):
                f[1]["id"][100] = 5000 + k
            pcount_before = int(f[1].header["PCOUNT"])
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
        assert pcount_after < pcount_before
        modified = original.copy()
        modified["id"][100] = 5004  # last write wins
        _check_round_trip(fname, modified, dt)


# ---------------------------------------------------------------------
# Multi-column subset writes
# ---------------------------------------------------------------------


def test_table_setitem_multi_columns_subset_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8")])
        sub = np.zeros(400, dtype=sub_dt)
        sub["id"] = np.arange(400, dtype="i4") + 5000
        sub["v"] = np.arange(400, dtype="f8") * 7.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]] = sub
        modified = original.copy()
        modified["id"] = sub["id"]
        modified["v"] = sub["v"]
        _check_round_trip(fname, modified, dt)


def test_table_setitem_multi_columns_tolerates_extra_fields_in_value():
    """The RHS may carry extra fields; only the named ones are used."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=150)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8"), ("extra", "i4")])
        sub = np.zeros(300, dtype=sub_dt)
        sub["id"] = np.arange(300, dtype="i4") - 1
        sub["v"] = np.arange(300, dtype="f8") * 0.1
        sub["extra"] = 999  # ignored
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]] = sub
        modified = original.copy()
        modified["id"] = sub["id"]
        modified["v"] = sub["v"]
        _check_round_trip(fname, modified, dt)


def test_table_setitem_multi_columns_case_insensitive():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("ID", "i4"), ("V", "f8")])
        sub = np.zeros(200, dtype=sub_dt)
        sub["ID"] = np.arange(200, dtype="i4") + 11
        sub["V"] = np.arange(200, dtype="f8") * 0.5
        with rustfits.FITS(fname, "r+") as f:
            f[1][["ID", "V"]] = sub
        modified = original.copy()
        modified["id"] = sub["ID"]
        modified["v"] = sub["V"]
        _check_round_trip(fname, modified, dt)


def test_table_setitem_multi_columns_duplicate_name_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        sub_dt = np.dtype([("id", "i4")])
        sub = np.zeros(100, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1][["id", "id"]] = sub
            assert "duplicate" in str(ei.value).lower()


def test_table_setitem_multi_columns_missing_field_in_value_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        # Value missing "v" field.
        sub_dt = np.dtype([("id", "i4")])
        sub = np.zeros(100, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][["id", "v"]] = sub


def test_table_setitem_multi_columns_unknown_column_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        sub_dt = np.dtype([("nope", "i4")])
        sub = np.zeros(100, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError) as ei:
                f[1][["nope"]] = sub
            assert "nope" in str(ei.value)


def test_table_setitem_multi_columns_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("id", "i4"), ("v", "f8")])
        sub = np.zeros(50, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][["id", "v"]] = sub


# ---------------------------------------------------------------------
# Algorithm matrix
# ---------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2", "RICE_1"])
def test_setitem_column_round_trip_across_algorithms(algo):
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("a", "i4"), ("b", "i4"), ("c", "i4")])
        nrows = 400
        ztilelen = 150
        data = np.zeros(nrows, dtype=dt)
        data["a"] = np.arange(nrows, dtype="i4")
        data["b"] = np.arange(nrows, dtype="i4") * 2
        data["c"] = np.arange(nrows, dtype="i4") * 3
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                compress={"a": algo, "b": algo, "c": algo},
                ztilelen=ztilelen,
            )
            f[1].write(data)
        new_b = np.arange(nrows, dtype="i4") + 10_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["b"] = new_b
            f[1]["a"][250] = -1
        modified = data.copy()
        modified["b"] = new_b
        modified["a"][250] = -1
        _check_round_trip(fname, modified, dt)


# ---------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------


def test_setitem_whole_column_clears_cache():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[1]
            _ = hdu.read()
            assert hdu.tile_cache_used > 0
            hdu["id"] = np.arange(400, dtype="i4")
            assert hdu.tile_cache_used == 0


def test_setitem_cell_clears_cache():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[1]
            _ = hdu.read()
            assert hdu.tile_cache_used > 0
            hdu["id"][50] = 5
            assert hdu.tile_cache_used == 0


# ---------------------------------------------------------------------
# Non-last HDU preserves trailing HDU
# ---------------------------------------------------------------------


def test_setitem_whole_column_non_last_hdu_preserves_trailing():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_basic()
        base = _basic_data(0, 600, dt)
        trail = np.arange(40, dtype="i2")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=600, compress=True, ztilelen=400)
            f[1].write(base)
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)
        new_v = np.arange(600, dtype="f8") - 99.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
            np.testing.assert_array_equal(f[2].read(), trail)
        with rustfits.FITS(fname, "r") as f:
            np.testing.assert_array_equal(f[2].read(), trail)
            out = f[1].read()
        expected = base.copy()
        expected["v"] = new_v
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Cross-tool: funpack reads mutated files
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_col_cell_multi_modified_file():
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        new_v = np.arange(600, dtype="f8") + 0.5
        sub_dt = np.dtype([("id", "i4"), ("c", "f4")])
        sub = np.zeros(600, dtype=sub_dt)
        sub["id"] = np.arange(600, dtype="i4") + 1000
        sub["c"] = np.arange(600, dtype="f4") * -0.25
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
            f[1][["id", "c"]] = sub
            f[1]["v"][500] = 9999.0
        modified = original.copy()
        modified["v"] = new_v
        modified["id"] = sub["id"]
        modified["c"] = sub["c"]
        modified["v"][500] = 9999.0
        out_path = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out_path, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out_path, "r") as f:
            cfit = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(cfit[col], modified[col])


# ---------------------------------------------------------------------
# Positive non-VLA-on-VLA-table case (VLA-targeting tests live in
# tests/test_table_compressed_setitem_vla.py)
# ---------------------------------------------------------------------


def test_table_setitem_multi_columns_non_vla_subset_on_vla_table_works():
    """
    Multi-col write to NON-VLA columns of a table that ALSO has a
    VLA column should succeed (the VLA column is untouched).
    """
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
        sub = np.zeros(10, dtype=sub_dt)
        sub["id"] = np.arange(10, dtype="i4") + 100
        sub["c"] = np.arange(10, dtype="f4") - 50.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "c"]] = sub
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["id"], sub["id"])
        np.testing.assert_array_equal(out["c"], sub["c"])
        # VLA column unchanged.
        for i in range(10):
            np.testing.assert_array_equal(
                out["v"][i], np.arange(i + 1, dtype="f4")
            )


# ---------------------------------------------------------------------
# Same-handle vs reopen parity
# ---------------------------------------------------------------------


def test_setitem_col_cell_multi_same_handle_and_reopen_agree():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        new_id = np.arange(300, dtype="i4") + 7
        with rustfits.FITS(fname, "r+") as f:
            f[1]["id"] = new_id
            f[1]["v"][50] = 7.5
            sub_dt = np.dtype([("c", "f4")])
            sub = np.zeros(300, dtype=sub_dt)
            sub["c"] = np.arange(300, dtype="f4") * -3.0
            f[1][["c"]] = sub
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            re = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(same[col], re[col])
        modified = original.copy()
        modified["id"] = new_id
        modified["v"][50] = 7.5
        modified["c"] = sub["c"]
        for col in dt.names:
            np.testing.assert_array_equal(re[col], modified[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
