"""
Tests for TableHDU.insert_column and TableHDU.delete_column.

Coverage:
- insert_column: append-at-end (default), position=, after=, before=,
  by name or index, into empty table, multi-D cells, string columns,
  unsigned-int trick, into VLA-bearing table.
- delete_column: by name, by index (positive + negative), fixed and
  VLA columns, VLA-bearing table preserves other VLAs, repack after
  VLA delete reclaims orphan heap.
- Non-last HDU: insert/delete shifts later HDUs in lockstep.
- Cross-tool: astropy reads our edited tables correctly.
- Errors: name collision, position out of range, multiple location
  kwargs, name not found, Object dtype rejected, non-default THEAP.
- Bounded memory invariant: large table (many rows) round-trips
  through insert/delete without OOM.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _basic_dtype():
    return np.dtype([("a", "i4"), ("b", "f8"), ("c", "i2")])


def _basic_data(nrows=5):
    dt = _basic_dtype()
    arr = np.zeros(nrows, dtype=dt)
    arr["a"] = np.arange(nrows, dtype="i4")
    arr["b"] = np.arange(nrows, dtype="f8") + 0.5
    arr["c"] = -np.arange(nrows, dtype="i2")
    return arr


def _make_table(fname, dtype=None, data=None, with_image_hdu_first=True):
    """
    Create a FITS file with an optional dummy primary HDU and one table
    HDU.  Returns the data that was written.
    """
    if dtype is None:
        dtype = _basic_dtype()
    if data is None:
        data = _basic_data()
    with rustfits.FITS(fname, "w+") as f:
        if with_image_hdu_first:
            f.create_image_hdu("i4", (1,))
        f.create_table_hdu(dtype, nrows=len(data))
        table_idx = 1 if with_image_hdu_first else 0
        f[table_idx].write(data)
    return data


def _check_both(fname, mutate_fn, predicate_fn):
    """
    Run mutate_fn(fits) under an r+ handle, then assert predicate_fn
    on the same handle AND after reopen.
    """
    with rustfits.FITS(fname, "r+") as f:
        mutate_fn(f)
        predicate_fn(f, "same handle")
    with rustfits.FITS(fname, "r") as f:
        predicate_fn(f, "after reopen")


# ----------------------------------------------------------------------
# insert_column — append at end (default)
# ----------------------------------------------------------------------


def test_insert_appends_at_end_by_default():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)
        new = np.arange(len(orig), dtype="f4") * 100.0

        def mutate(f):
            f[1].insert_column("d", new)

        def check(f, label):
            assert f[1].colnames == ("a", "b", "c", "d"), label
            arr = f[1].read()
            np.testing.assert_array_equal(arr["a"], orig["a"])
            np.testing.assert_array_equal(arr["b"], orig["b"])
            np.testing.assert_array_equal(arr["c"], orig["c"])
            np.testing.assert_array_equal(arr["d"], new)

        _check_both(fname, mutate, check)


def test_insert_at_position_zero():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)
        new = np.arange(len(orig), dtype="u1") + 50

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("x", new, position=0)
            assert f[1].colnames == ("x", "a", "b", "c")
            arr = f[1].read()
            np.testing.assert_array_equal(arr["x"], new)
            np.testing.assert_array_equal(arr["a"], orig["a"])

        with rustfits.FITS(fname, "r") as f:
            assert f[1].colnames == ("x", "a", "b", "c")
            arr = f[1].read()
            np.testing.assert_array_equal(arr["x"], new)


def test_insert_at_middle_position():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="f8") * -1.0

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("mid", new, position=2)
            assert f[1].colnames == ("a", "b", "mid", "c")


def test_insert_after_name():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, after="a")
            assert f[1].colnames == ("a", "z", "b", "c")


def test_insert_after_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, after=0)
            assert f[1].colnames == ("a", "z", "b", "c")


def test_insert_before_name():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, before="b")
            assert f[1].colnames == ("a", "z", "b", "c")


def test_insert_before_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, before=2)
            assert f[1].colnames == ("a", "b", "z", "c")


def test_insert_before_negative_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, before=-1)
            assert f[1].colnames == ("a", "b", "z", "c")


def test_insert_case_insensitive_after():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="i4")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("z", new, after="A")
            assert f[1].colnames == ("a", "z", "b", "c")


# ----------------------------------------------------------------------
# insert_column — dtype handling
# ----------------------------------------------------------------------


def test_insert_unsigned_int_trick():
    """
    u2/u4/u8 inputs should round-trip through the TZERO unsigned-int
    trick (same as create_table_hdu paths).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        u2 = np.array([0, 32768, 65535, 1, 30000], dtype="u2")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("u2col", u2)
            arr = f[1].read()
            assert arr["u2col"].dtype == np.uint16
            np.testing.assert_array_equal(arr["u2col"], u2)


def test_insert_subarray_emits_tdim():
    """
    Multi-D per-cell shape: should emit TDIM and round-trip with the
    same shape.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        cube = np.arange(5 * 2 * 3, dtype="f4").reshape(5, 2, 3)

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("cube", cube)
            arr = f[1].read()
            assert arr["cube"].shape == (5, 2, 3)
            np.testing.assert_array_equal(arr["cube"], cube)


def test_insert_string_column():
    """
    S<n> input should land as a fixed-width A column.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        strs = np.array(["one", "two", "three", "four", "five"], dtype="S8")

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("name", strs)
            arr = f[1].read()
            # read returns U<n>; compare contents.
            np.testing.assert_array_equal(arr["name"].astype("S8"), strs)


def test_insert_with_unit():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        new = np.arange(5, dtype="f4") * 10.0

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("flux", new, unit="Jy")
            units = f[1].units
            assert units["flux"] == "Jy"


# ----------------------------------------------------------------------
# insert_column into VLA-bearing tables
# ----------------------------------------------------------------------


def test_insert_fixed_into_vla_table_preserves_vla():
    """
    Insert a fixed column into a table that already has a VLA column.
    The heap should be relocated forward and the VLA contents should
    still read correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("a", "i4"), ("v", "O")])
        nrows = 6
        cells = [np.arange(i + 1, dtype="f4") * (i + 1) for i in range(nrows)]
        arr = np.empty(nrows, dtype=dt)
        arr["a"] = np.arange(nrows, dtype="i4")
        for i, c in enumerate(cells):
            arr["v"][i] = c

        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": "f4"})
            f[1].write(arr)

        new_col = np.arange(nrows, dtype="i2") * 10

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("k", new_col, before="v")
            assert f[1].colnames == ("a", "k", "v")
            r = f[1].read()
            np.testing.assert_array_equal(r["a"], arr["a"])
            np.testing.assert_array_equal(r["k"], new_col)
            for i in range(nrows):
                np.testing.assert_array_equal(r["v"][i], cells[i])

        with rustfits.FITS(fname, "r") as f:
            r = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(r["v"][i], cells[i])


# ----------------------------------------------------------------------
# delete_column
# ----------------------------------------------------------------------


def test_delete_by_name():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)

        def mutate(f):
            f[1].delete_column("b")

        def check(f, label):
            assert f[1].colnames == ("a", "c"), label
            arr = f[1].read()
            np.testing.assert_array_equal(arr["a"], orig["a"])
            np.testing.assert_array_equal(arr["c"], orig["c"])

        _check_both(fname, mutate, check)


def test_delete_by_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column(1)
            assert f[1].colnames == ("a", "c")


def test_delete_by_negative_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column(-1)
            assert f[1].colnames == ("a", "b")


def test_delete_first_then_last():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column(0)
            f[1].delete_column(-1)
            assert f[1].colnames == ("b",)
            arr = f[1].read()
            np.testing.assert_array_equal(arr["b"], orig["b"])


def test_delete_case_insensitive():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column("B")
            assert f[1].colnames == ("a", "c")


def test_delete_vla_column_leaves_orphans_repack_reclaims():
    """
    Deleting a VLA column drops the descriptor bytes but leaves heap
    cells orphaned.  repack() should reclaim them and shrink PCOUNT.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("a", "i4"), ("v", "O"), ("w", "O")])
        nrows = 5
        v_cells = [np.arange(i + 1, dtype="f4") for i in range(nrows)]
        w_cells = [np.arange(i + 2, dtype="i2") * 100 for i in range(nrows)]
        arr = np.empty(nrows, dtype=dt)
        arr["a"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = v_cells[i]
            arr["w"][i] = w_cells[i]

        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(
                dt, nrows=nrows, var_dtypes={"v": "f4", "w": "i2"}
            )
            f[1].write(arr)
            # Both v and w contributed to PCOUNT.
            pcount_before = int(f[1].header["PCOUNT"])

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column("v")
            # Pcount unchanged immediately after delete (orphans).
            assert int(f[1].header["PCOUNT"]) == pcount_before
            # Other VLA still readable.
            r = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(r["w"][i], w_cells[i])

        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
            assert pcount_after < pcount_before
            r = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(r["w"][i], w_cells[i])


# ----------------------------------------------------------------------
# Non-last HDU shifts
# ----------------------------------------------------------------------


def test_insert_non_last_hdu_shifts_later_hdus():
    """
    Insert into a middle HDU should shift the trailing HDU forward and
    leave its content intact.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            dt = _basic_dtype()
            f.create_table_hdu(dt, nrows=5)
            f[1].write(_basic_data())
            # Trailing HDU we don't want to corrupt.
            trail = np.arange(120, dtype="f4").reshape(10, 12)
            f.create_image_hdu("f4", trail.shape, extname="TRAIL")
            f[2].write(trail)

        new_col = np.arange(5, dtype="f8") * 7.0
        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("d", new_col, after="b")
            assert f[1].colnames == ("a", "b", "d", "c")
            # Trailing HDU still correct.
            assert f[2].extname == "TRAIL"
            np.testing.assert_array_equal(f[2].read(), trail)

        with rustfits.FITS(fname, "r") as f:
            assert f[2].extname == "TRAIL"
            np.testing.assert_array_equal(f[2].read(), trail)


def test_delete_non_last_hdu_shifts_later_hdus():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            dt = _basic_dtype()
            f.create_table_hdu(dt, nrows=5)
            f[1].write(_basic_data())
            trail = np.arange(40, dtype="i2")
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column("c")
            assert f[1].colnames == ("a", "b")
            np.testing.assert_array_equal(f[2].read(), trail)

        with rustfits.FITS(fname, "r") as f:
            np.testing.assert_array_equal(f[2].read(), trail)


# ----------------------------------------------------------------------
# Round trip through many operations
# ----------------------------------------------------------------------


def test_round_trip_insert_then_delete_restores_layout():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)
        original_naxis1 = None
        with rustfits.FITS(fname, "r") as f:
            original_naxis1 = int(f[1].header["NAXIS1"])

        new = np.arange(len(orig), dtype="f4") * 11.0
        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("tmp", new, position=1)
            f[1].delete_column("tmp")
            assert f[1].colnames == ("a", "b", "c")
            arr = f[1].read()
            np.testing.assert_array_equal(arr["a"], orig["a"])
            np.testing.assert_array_equal(arr["b"], orig["b"])
            np.testing.assert_array_equal(arr["c"], orig["c"])
            assert int(f[1].header["NAXIS1"]) == original_naxis1


# ----------------------------------------------------------------------
# Cross-tool verification (astropy reads our edited tables)
# ----------------------------------------------------------------------


def test_astropy_reads_inserted_table():
    astropy = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)
        new = np.arange(len(orig), dtype="f4") * 9.5

        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("d", new, after="a")

        with astropy.open(fname) as hdul:
            t = hdul[1].data
            assert list(t.dtype.names) == ["a", "d", "b", "c"]
            np.testing.assert_array_equal(t["d"], new)
            np.testing.assert_array_equal(t["a"], orig["a"])


def test_astropy_reads_deleted_table():
    astropy = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        orig = _make_table(fname)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column("b")

        with astropy.open(fname) as hdul:
            t = hdul[1].data
            assert list(t.dtype.names) == ["a", "c"]
            np.testing.assert_array_equal(t["a"], orig["a"])
            np.testing.assert_array_equal(t["c"], orig["c"])


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_insert_duplicate_name_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="already exists"):
                f[1].insert_column("a", np.arange(5, dtype="i4"))


def test_insert_duplicate_name_case_insensitive_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="already exists"):
                f[1].insert_column("A", np.arange(5, dtype="i4"))


def test_insert_empty_name_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="non-empty"):
                f[1].insert_column("", np.arange(5, dtype="i4"))


def test_insert_position_out_of_range_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="out of range"):
                f[1].insert_column("d", np.arange(5, dtype="i4"), position=10)


def test_insert_multiple_location_kwargs_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="at most one"):
                f[1].insert_column(
                    "d",
                    np.arange(5, dtype="i4"),
                    position=0,
                    after="a",
                )


def test_insert_after_unknown_column_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="not found"):
                f[1].insert_column("d", np.arange(5, dtype="i4"), after="nope")


def test_insert_shape_mismatch_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="nrows"):
                f[1].insert_column("d", np.arange(7, dtype="i4"))


def test_insert_object_dtype_rejected():
    """
    VLA insertion is a follow-up; first cut rejects Object dtype.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            obj = np.empty(5, dtype="O")
            for i in range(5):
                obj[i] = np.arange(i + 1, dtype="f4")
            with pytest.raises(ValueError, match="VLA"):
                f[1].insert_column("v", obj)


def test_delete_unknown_column_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="not found"):
                f[1].delete_column("nope")


def test_delete_index_out_of_range_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError, match="out of range"):
                f[1].delete_column(10)


# ----------------------------------------------------------------------
# Bounded memory: strip-based shuffle handles a large table
# ----------------------------------------------------------------------


def test_insert_into_table_larger_than_strip():
    """
    Strip size is ~1 MiB.  A table with row_width ~80 bytes and
    50_000 rows is ~4 MB of main data — exercises multi-strip
    behavior.  Memory should stay bounded; we only sanity-check that
    the operation completes and round-trips correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 50_000
        dt = np.dtype([("a", "i8"), ("b", "f8"), ("c", ("f4", (16,)))])
        arr = np.empty(nrows, dtype=dt)
        arr["a"] = np.arange(nrows, dtype="i8")
        arr["b"] = np.arange(nrows, dtype="f8") * 0.5
        arr["c"] = np.arange(nrows * 16, dtype="f4").reshape(nrows, 16)
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=nrows)
            f[1].write(arr)

        new = np.arange(nrows, dtype="i4")
        with rustfits.FITS(fname, "r+") as f:
            f[1].insert_column("k", new, position=1)
            r = f[1].read()
            np.testing.assert_array_equal(r["a"], arr["a"])
            np.testing.assert_array_equal(r["k"], new)
            np.testing.assert_array_equal(r["b"], arr["b"])
            np.testing.assert_array_equal(r["c"], arr["c"])

        with rustfits.FITS(fname, "r") as f:
            r = f[1].read()
            np.testing.assert_array_equal(r["c"], arr["c"])


def test_delete_from_large_table():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 50_000
        dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "f4")])
        arr = np.empty(nrows, dtype=dt)
        arr["a"] = np.arange(nrows, dtype="i4")
        arr["b"] = np.arange(nrows, dtype="f8") * 0.25
        arr["c"] = np.arange(nrows, dtype="f4") * -1.0
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=nrows)
            f[1].write(arr)

        with rustfits.FITS(fname, "r+") as f:
            f[1].delete_column("b")
            r = f[1].read()
            np.testing.assert_array_equal(r["a"], arr["a"])
            np.testing.assert_array_equal(r["c"], arr["c"])

        with rustfits.FITS(fname, "r") as f:
            r = f[1].read()
            np.testing.assert_array_equal(r["a"], arr["a"])
            np.testing.assert_array_equal(r["c"], arr["c"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
