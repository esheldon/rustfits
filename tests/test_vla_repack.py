"""
Tests for TableHDU.repack() — rebuild the VLA heap with only live cells,
dropping orphans left behind by __setitem__.

When the heap shrinks, the on-disk file shrinks too:
- last HDU: set_len drops trailing block(s).
- non-last HDU: file tail shifts backward, later HDU offsets bump down.

Live-cell ordering after repack is row-major × VLA-column-order (the
same layout the bulk write path produces).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_simple_vla(tmpdir, inner="f4", nrows=4):
    dt = np.dtype([("v", "O")])
    arr = np.zeros(nrows, dtype=dt)
    for i in range(nrows):
        arr["v"][i] = np.arange(i + 1, dtype=inner)
    fname = os.path.join(tmpdir, "t.fits")
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": inner})
        f[1].write(arr)
    return fname, arr


def _filesize(fname):
    return os.path.getsize(fname)


# ---------------------------------------------------------------------------
# Repack reduces PCOUNT and recovers disk space
# ---------------------------------------------------------------------------


def test_repack_after_setitem_drops_orphans():
    """
    Write rows, then __setitem__ several rows so the heap grows past
    the live-cell total.  Repack should shrink PCOUNT back to the
    sum of live cell bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_simple_vla(tmp, nrows=4)
        with rustfits.FITS(fname, "r") as fits:
            base_pcount = int(fits[1].header["PCOUNT"])
        new = np.zeros(1, dtype=arr.dtype)
        with rustfits.FITS(fname, "r+") as fits:
            for r in range(4):
                new["v"][0] = np.array([100 + r] * 3, dtype="f4")
                fits[1][r] = new[0]
        # Each __setitem__ orphaned the original cell.
        with rustfits.FITS(fname, "r") as fits:
            mid_pcount = int(fits[1].header["PCOUNT"])
        assert mid_pcount > base_pcount  # orphans accumulated
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        with rustfits.FITS(fname, "r") as fits:
            final_pcount = int(fits[1].header["PCOUNT"])
            # All 4 rows now hold a 3-element f4 cell.
            assert final_pcount == 4 * 3 * 4
            got = fits[1].read()
            for r in range(4):
                np.testing.assert_array_equal(
                    got["v"][r],
                    [100 + r] * 3,
                )


def test_repack_shrinks_last_hdu_file():
    """
    Repack on a last-HDU table should reduce the on-disk file size
    when enough orphans accumulate to free a 2880-byte block.  We
    write the same row repeatedly so each call adds a 3200-byte
    orphan; after ~10 calls the savings cross multiple block
    boundaries and repack must shrink the file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_simple_vla(tmp, nrows=4)
        new = np.zeros(1, dtype=arr.dtype)
        with rustfits.FITS(fname, "r+") as fits:
            for _ in range(10):
                new["v"][0] = np.arange(800, dtype="f4")  # 3200 bytes
                fits[1][0] = new[0]  # same row each time → orphans
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        post_size = _filesize(fname)
        assert post_size < pre_size
        # Round-trip OK: row 0 has the last-written value, rows 1-3
        # still have the original increasing-length cells.
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(
                got["v"][0],
                np.arange(800, dtype="f4"),
            )
            for r in range(1, 4):
                np.testing.assert_array_equal(got["v"][r], arr["v"][r])


def test_repack_no_op_when_already_compact():
    """
    Right after a bulk write with no __setitem__, the heap is already
    compact.  repack() should be a no-op (PCOUNT unchanged, no file
    size change, contents preserved).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_simple_vla(tmp, nrows=4)
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r") as fits:
            pre_pcount = int(fits[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        with rustfits.FITS(fname, "r") as fits:
            assert int(fits[1].header["PCOUNT"]) == pre_pcount
            got = fits[1].read()
            for r in range(4):
                np.testing.assert_array_equal(got["v"][r], arr["v"][r])
        assert _filesize(fname) == pre_size


def test_repack_no_op_on_non_vla_table():
    """A table with no VLA columns is unaffected by repack()."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "f.fits")
        dt = np.dtype([("id", "i4"), ("x", "f8")])
        arr = np.zeros(3, dtype=dt)
        arr["id"] = [1, 2, 3]
        arr["x"] = [0.5, 1.5, 2.5]
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=3)
            f[1].write(arr)
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        assert _filesize(fname) == pre_size
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got, arr)


# ---------------------------------------------------------------------------
# Multiple VLA columns and mixed fixed/VLA
# ---------------------------------------------------------------------------


def test_repack_multiple_vla_columns():
    """Two VLA columns repack together; both round-trip correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        dt = np.dtype([("a", "O"), ("b", "O")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        for i in range(nrows):
            arr["a"][i] = np.arange(i + 1, dtype="i4")
            arr["b"][i] = np.arange(i + 2, dtype="f8")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"a": "i4", "b": "f8"},
            )
            f[1].write(arr)
        # Generate orphans on both columns.
        new = np.zeros(1, dtype=dt)
        with rustfits.FITS(fname, "r+") as fits:
            for r in range(nrows):
                new["a"][0] = np.array([10 * r], dtype="i4")
                new["b"][0] = np.array([100.0 + r], dtype="f8")
                fits[1][r] = new[0]
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for r in range(nrows):
                np.testing.assert_array_equal(got["a"][r], [10 * r])
                np.testing.assert_array_equal(got["b"][r], [100.0 + r])
            # All cells live → minimal PCOUNT.
            expected = nrows * (1 * 4 + 1 * 8)
            assert int(fits[1].header["PCOUNT"]) == expected


def test_repack_mixed_fixed_and_vla():
    """Mixed fixed + VLA columns: fixed cells untouched, heap compacted."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O"), ("flux", "f8")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        arr["id"] = np.arange(nrows)
        arr["flux"] = np.arange(nrows, dtype="f8") * 0.5
        for i in range(nrows):
            arr["lc"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"lc": "f4"})
            f[1].write(arr)
        new = np.zeros(1, dtype=dt)
        with rustfits.FITS(fname, "r+") as fits:
            new["id"][0] = 99
            new["flux"][0] = 42.0
            new["lc"][0] = np.array([7.0, 8.0], dtype="f4")
            fits[1][1] = new[0]
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            # Modified row.
            assert got["id"][1] == 99
            assert got["flux"][1] == 42.0
            np.testing.assert_array_equal(got["lc"][1], [7.0, 8.0])
            # Unmodified rows.
            assert got["id"][0] == 0
            assert got["id"][2] == 2
            np.testing.assert_array_equal(got["lc"][0], arr["lc"][0])
            np.testing.assert_array_equal(got["lc"][2], arr["lc"][2])
            # Compact PCOUNT after repack.
            expected = (1 + 2 + 3) * 4  # cells of sizes 1, 2, 3 f4
            assert int(fits[1].header["PCOUNT"]) == expected


# ---------------------------------------------------------------------------
# Non-last HDU shrink path
# ---------------------------------------------------------------------------


def test_repack_non_last_hdu_shifts_tail_backward():
    """
    Repack on a non-last HDU should shift the following HDUs backward
    via shift_file_tail_backward_and_update_offsets.  Previously-issued
    handles to later HDUs still work (shared Arc<HduOffsets>).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "two.fits")
        dt = np.dtype([("v", "O")])
        nrows = 4
        arr = np.zeros(nrows, dtype=dt)
        for i in range(nrows):
            arr["v"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": "f4"})
            f[1].write(arr)
            f.create_image_hdu("i4", (5,), extname="AFTER")
            f[2].write(np.arange(5, dtype="i4") + 1000)
        # Inflate the table's heap with orphans: same row written
        # repeatedly so each write leaves a 3200-byte orphan.
        new = np.zeros(1, dtype=dt)
        with rustfits.FITS(fname, "r+") as fits:
            for _ in range(10):
                new["v"][0] = np.arange(800, dtype="f4")
                fits[1][0] = new[0]
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
            # Same-handle: second HDU still reads at its shifted offset.
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
        assert _filesize(fname) < pre_size
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
            got = fits[1].read()
            np.testing.assert_array_equal(
                got["v"][0],
                np.arange(800, dtype="f4"),
            )
            for r in range(1, nrows):
                np.testing.assert_array_equal(got["v"][r], arr["v"][r])


# ---------------------------------------------------------------------------
# Empty/edge cases
# ---------------------------------------------------------------------------


def test_repack_all_empty_cells():
    """Heap full of empty cells repacks to PCOUNT=0."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        for i in range(nrows):
            arr["v"][i] = np.array([float(i)], dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": "f4"})
            f[1].write(arr)
        # Overwrite all with empty cells.
        new = np.zeros(1, dtype=dt)
        new["v"][0] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            for r in range(nrows):
                fits[1][r] = new[0]
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            assert int(fits[1].header["PCOUNT"]) == 0
            got = fits[1].read()
            for r in range(nrows):
                assert got["v"][r].shape == (0,)


def test_repack_then_setitem_then_repack():
    """
    Sequence of repack → setitem → repack stays consistent and
    eventually converges to compact PCOUNT.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, arr = _make_simple_vla(tmp, nrows=3)
        new = np.zeros(1, dtype=arr.dtype)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
            new["v"][0] = np.arange(7, dtype="f4")
            fits[1][1] = new[0]
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["v"][0], arr["v"][0])
            np.testing.assert_array_equal(
                got["v"][1], np.arange(7, dtype="f4")
            )
            np.testing.assert_array_equal(got["v"][2], arr["v"][2])
            expected = (1 + 7 + 3) * 4
            assert int(fits[1].header["PCOUNT"]) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
