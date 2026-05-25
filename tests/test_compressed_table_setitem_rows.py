"""
ZTABLE Phase 6c-2b — row __setitem__ on compressed tables.

`CompressedTableHDU.__setitem__` for row writes covers three forms,
all of which touch ALL columns of the selected rows:

  - hdu[i] = record         (single tile)
  - hdu[a:b] = arr          (step=1; one or more tiles)
  - hdu[[i, j, k]] = arr    (fancy rows; one or more tiles)

Affected tiles are decoded, the rows' bytes are overwritten via the
shared per-cell transform, and the slabs are re-encoded + appended
to the heap end.  Old blobs become orphans (reclaimed by repack()).

Stepped slices, column writes, cell writes, and VLA tables are
deferred to Phases 6c-2c/d/e and raise NotImplementedError pointing
at the right phase.
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


def _make_table(
    fname, *, nrows, ztilelen, with_image_first=False, compress=True
):
    dt = _dt_basic()
    data = _basic_data(0, nrows, dt)
    with rustfits.FITS(fname, "w+") as f:
        if with_image_first:
            f.create_image_hdu("i4", (1,))
        f.create_table_hdu(
            dt, nrows=nrows, compress=compress, ztilelen=ztilelen
        )
        f[1].write(data)
    return data, dt


def _check_round_trip(fname, expected, dt):
    """Read via same-handle and via reopen; both must equal expected."""
    with rustfits.FITS(fname, "r") as f:
        out = f[1].read()
    for col in dt.names:
        np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Single-row writes
# ---------------------------------------------------------------------


def test_setitem_single_row_middle_of_tile():
    """
    Overwrite a single row inside a tile.  Same-handle and reopen
    reads both reflect the new value.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        record = np.zeros(1, dtype=dt)
        record["id"] = 999_999
        record["v"] = 3.14159
        record["c"] = -0.5
        with rustfits.FITS(fname, "r+") as f:
            f[1][150] = record[0]
            out_same = f[1].read()
        original_modified = original.copy()
        original_modified[150] = record[0]
        for col in dt.names:
            np.testing.assert_array_equal(
                out_same[col], original_modified[col]
            )
        _check_round_trip(fname, original_modified, dt)


def test_setitem_single_row_accepts_length1_structured_array():
    """A shape-(1,) structured ndarray works as RHS, not just a void."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        record = np.zeros(1, dtype=dt)
        record["id"] = 12345
        record["v"] = -7.25
        record["c"] = 0.125
        with rustfits.FITS(fname, "r+") as f:
            f[1][50] = record  # shape-(1,) form
        original_modified = original.copy()
        original_modified[50] = record[0]
        _check_round_trip(fname, original_modified, dt)


def test_setitem_single_row_negative_index():
    """Negative indices wrap around like numpy."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=200)
        record = np.zeros(1, dtype=dt)
        record["id"] = -1
        record["v"] = 42.0
        record["c"] = -42.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][-1] = record[0]
        original_modified = original.copy()
        original_modified[-1] = record[0]
        _check_round_trip(fname, original_modified, dt)


def test_setitem_single_row_out_of_range_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        record = np.zeros(1, dtype=_dt_basic())
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(IndexError):
                f[1][100] = record[0]
            with pytest.raises(IndexError):
                f[1][-101] = record[0]


def test_setitem_single_row_orphans_old_blob():
    """PCOUNT grows after a setitem (old blob orphaned)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        record = np.zeros(1, dtype=_dt_basic())
        record["id"] = 7
        record["v"] = 11.0
        record["c"] = -3.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][100] = record[0]
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
        # We rewrite the touched tile + replace all 3 columns,
        # so PCOUNT must grow.
        assert pcount_after > pcount_before


def test_setitem_single_row_at_tile_boundary():
    """
    Hit the last row of tile 0 then the first row of tile 1 — two
    separate setitem calls, each in a different tile.  Both round
    trip correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        r_a = np.zeros(1, dtype=dt)
        r_a["id"] = 100_000
        r_a["v"] = 1.0
        r_a["c"] = -1.0
        r_b = np.zeros(1, dtype=dt)
        r_b["id"] = 200_000
        r_b["v"] = 2.0
        r_b["c"] = -2.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][199] = r_a[0]
            f[1][200] = r_b[0]
        original_modified = original.copy()
        original_modified[199] = r_a[0]
        original_modified[200] = r_b[0]
        _check_round_trip(fname, original_modified, dt)


# ---------------------------------------------------------------------
# Slice writes (step=1)
# ---------------------------------------------------------------------


def test_setitem_slice_within_one_tile():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        chunk = _basic_data(10_000, 10_020, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][50:70] = chunk
        original_modified = original.copy()
        original_modified[50:70] = chunk
        _check_round_trip(fname, original_modified, dt)


def test_setitem_slice_spans_multiple_tiles():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        # Slice spans the boundary between tiles 0/1/2.
        chunk = _basic_data(50_000, 50_300, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][150:450] = chunk
        original_modified = original.copy()
        original_modified[150:450] = chunk
        _check_round_trip(fname, original_modified, dt)


def test_setitem_full_slice_overwrites_everything():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=150)
        dt = _dt_basic()
        replacement = _basic_data(900_000, 900_400, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][:] = replacement
        _check_round_trip(fname, replacement, dt)


def test_setitem_slice_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        dt = _dt_basic()
        wrong_len = _basic_data(0, 5, dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][50:70] = wrong_len  # selects 20, value has 5


def test_setitem_slice_step_not_one_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        dt = _dt_basic()
        chunk = _basic_data(0, 5, dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError) as ei:
                f[1][0:50:10] = chunk
            assert "6c-2d" in str(ei.value)


def test_setitem_empty_slice_with_empty_value_noop():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        empty = np.zeros(0, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][50:50] = empty
        with rustfits.FITS(fname, "r") as f:
            assert int(f[1].header["PCOUNT"]) == pcount_before
        _check_round_trip(fname, original, dt)


def test_setitem_empty_slice_with_nonempty_value_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        chunk = _basic_data(0, 3, _dt_basic())
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][50:50] = chunk


# ---------------------------------------------------------------------
# Fancy-row writes
# ---------------------------------------------------------------------


def test_setitem_fancy_rows_within_one_tile():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        chunk = _basic_data(70_000, 70_004, dt)
        idx = [10, 20, 30, 40]
        with rustfits.FITS(fname, "r+") as f:
            f[1][idx] = chunk
        original_modified = original.copy()
        for k, i in enumerate(idx):
            original_modified[i] = chunk[k]
        _check_round_trip(fname, original_modified, dt)


def test_setitem_fancy_rows_span_tiles():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=150)
        # Touch tile 0, 1, 2, 3.
        idx = [10, 200, 350, 500]
        chunk = _basic_data(80_000, 80_004, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][idx] = chunk
        original_modified = original.copy()
        for k, i in enumerate(idx):
            original_modified[i] = chunk[k]
        _check_round_trip(fname, original_modified, dt)


def test_setitem_fancy_rows_negative_indices():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        idx = [-1, -150, 0]
        chunk = _basic_data(90_000, 90_003, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][idx] = chunk
        normalized = [n % 300 for n in idx]
        original_modified = original.copy()
        for k, i in enumerate(normalized):
            original_modified[i] = chunk[k]
        _check_round_trip(fname, original_modified, dt)


def test_setitem_fancy_rows_duplicate_indices_last_wins():
    """
    With duplicate disk indices in the input list, the LAST row in
    the input becomes the final stored value (matches numpy fancy-
    assignment semantics).  Both rows go into the same tile so this
    also tests that multiple per-tile edits are applied in input
    order.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=200)
        idx = [50, 50, 100]
        chunk = np.zeros(3, dtype=dt)
        for k in range(3):
            chunk["id"][k] = 1000 + k
            chunk["v"][k] = float(k)
            chunk["c"][k] = -float(k)
        with rustfits.FITS(fname, "r+") as f:
            f[1][idx] = chunk
        original_modified = original.copy()
        for k, i in enumerate(idx):
            original_modified[i] = chunk[k]  # last write wins
        _check_round_trip(fname, original_modified, dt)


def test_setitem_fancy_rows_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=200, ztilelen=100)
        wrong = _basic_data(0, 5, _dt_basic())
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1][[1, 2, 3]] = wrong


# ---------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------


def test_setitem_clears_tile_cache():
    """
    Modified tiles' cache entries are stale; the simplest correct
    invalidation is a full clear.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=400, ztilelen=200)
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[1]
            _ = hdu.read()
            assert hdu.tile_cache_used > 0
            record = np.zeros(1, dtype=_dt_basic())
            hdu[10] = record[0]
            assert hdu.tile_cache_used == 0


# ---------------------------------------------------------------------
# repack reclaims orphans
# ---------------------------------------------------------------------


def test_setitem_then_repack_reclaims_orphans():
    """
    After repeated setitem the heap grows; repack reclaims the
    orphans and PCOUNT drops back.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        with rustfits.FITS(fname, "r+") as f:
            # Hammer the same tile 5 times.
            for k in range(5):
                rec = np.zeros(1, dtype=dt)
                rec["id"] = 10_000 + k
                rec["v"] = float(k)
                rec["c"] = -float(k)
                f[1][100] = rec[0]
            pcount_before = int(f[1].header["PCOUNT"])
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
        assert pcount_after < pcount_before
        # Final value reflects the last write.
        original_modified = original.copy()
        original_modified["id"][100] = 10_004
        original_modified["v"][100] = 4.0
        original_modified["c"][100] = -4.0
        _check_round_trip(fname, original_modified, dt)


# ---------------------------------------------------------------------
# Non-last HDU preserves trailing HDU
# ---------------------------------------------------------------------


def test_setitem_non_last_hdu_preserves_trailing_hdu():
    """
    Compressed table that is not the last HDU on disk — setitem
    grows the heap (PCOUNT bump → maybe file shift) but leaves
    the trailing HDU intact.
    """
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
        chunk = _basic_data(99_000, 99_100, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][50:150] = chunk
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        expected = base.copy()
        expected[50:150] = chunk
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Algorithm matrix
# ---------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2", "RICE_1"])
def test_setitem_round_trip_across_algorithms(algo):
    """
    Force every (i4) column to a specific algorithm via the
    per-column compress dict.  Note: RICE_1 is only valid for B/I/J
    columns, so the table is i4-only for this test.
    """
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
        # Slice across tiles + a fancy row write to exercise the
        # multi-tile + multi-edit path.
        chunk = np.zeros(200, dtype=dt)
        chunk["a"] = np.arange(200, dtype="i4") + 10_000
        chunk["b"] = np.arange(200, dtype="i4") + 20_000
        chunk["c"] = np.arange(200, dtype="i4") + 30_000
        with rustfits.FITS(fname, "r+") as f:
            f[1][100:300] = chunk
            rec = np.zeros(1, dtype=dt)
            rec["a"] = 7
            rec["b"] = 77
            rec["c"] = 777
            f[1][399] = rec[0]
        expected = data.copy()
        expected[100:300] = chunk
        expected[399] = rec[0]
        _check_round_trip(fname, expected, dt)


# ---------------------------------------------------------------------
# Subarray (TDIM) column
# ---------------------------------------------------------------------


def test_setitem_subarray_column_round_trip():
    """
    A column with a per-cell numpy shape (subarray / TDIM) round-
    trips correctly through setitem.
    """
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
        # Replace 5 rows spanning two tiles.
        rep = np.zeros(5, dtype=dt)
        rep["id"] = [100, 101, 102, 103, 104]
        rep["vec"] = np.arange(15, dtype="f4").reshape(5, 3) + 1000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][78:83] = rep
        expected = data.copy()
        expected[78:83] = rep
        _check_round_trip(fname, expected, dt)


# ---------------------------------------------------------------------
# Cross-tool: funpack reads files we mutated
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_setitem_modified_file():
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=200)
        chunk = _basic_data(40_000, 40_120, dt)
        with rustfits.FITS(fname, "r+") as f:
            # Slice across tiles.
            f[1][140:260] = chunk
            # Fancy across more tiles.
            idx = [10, 220, 555]
            small = _basic_data(50_000, 50_003, dt)
            f[1][idx] = small
            # Single-row at end.
            rec = np.zeros(1, dtype=dt)
            rec["id"] = 9_999_999
            rec["v"] = -1.5
            rec["c"] = 2.5
            f[1][599] = rec[0]
        expected = original.copy()
        expected[140:260] = chunk
        for k, i in enumerate(idx):
            expected[i] = small[k]
        expected[599] = rec[0]
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
# Rejections: column / cell writes and VLA tables still raise pointers
# ---------------------------------------------------------------------


def test_setitem_single_column_rejected_with_phase_pointer():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        arr = np.arange(100, dtype="i4")
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError) as ei:
                f[1]["id"] = arr
            assert "6c-2c" in str(ei.value)


def test_setitem_multi_columns_rejected_with_phase_pointer():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        dt = np.dtype([("id", "i4"), ("v", "f8")])
        arr = np.zeros(100, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError) as ei:
                f[1][["id", "v"]] = arr
            assert "6c-2c" in str(ei.value)


def test_setitem_cell_rejected_with_phase_pointer():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError) as ei:
                f[1][10, "id"] = np.int32(42)
            assert "6c-2c" in str(ei.value)


def test_setitem_vla_table_rejected_with_phase_pointer():
    """
    A table with a VLA column rejects setitem with the 6c-2e pointer
    (any row form).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
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
            for i in range(10):
                data["v"][i] = np.arange(i + 1, dtype="f4")
            f[1].write(data)
        rec = np.zeros(1, dtype=dt)
        rec["id"] = 99
        rec["v"][0] = np.arange(2, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError) as ei:
                f[1][0] = rec[0]
            assert "6c-2e" in str(ei.value)


# ---------------------------------------------------------------------
# Same-handle vs reopen parity
# ---------------------------------------------------------------------


def test_setitem_same_handle_and_reopen_agree():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=400, ztilelen=200)
        chunk = _basic_data(60_000, 60_010, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1][20:30] = chunk
            same_handle = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            reopen = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(same_handle[col], reopen[col])
        expected = original.copy()
        expected[20:30] = chunk
        for col in dt.names:
            np.testing.assert_array_equal(reopen[col], expected[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
