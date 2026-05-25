"""
ZTABLE Phase 6c-2e — VLA __setitem__ on compressed tables.

Lifts the last remaining VLA rejection across every __setitem__
form on CompressedTableHDU.  For VLA-targeted writes the per-tile
helper:

  1. Reads + GZIP-decompresses the existing dual-descriptor blob.
  2. Encodes each edited cell with the per-column ZCTYPn algorithm
     (with uncompressed fallback when compressed >= uncompressed,
     matching cfitsio's table-compressor convention).
  3. Appends new cell bytes to the heap end (orphans old bytes).
  4. Updates the in-RAM blob's compressed-Q descriptor for the
     edited row(s) and assigns a fresh original_offset = current
     ZPCOUNT (orphans the old original-heap slot in funpack's
     reconstructed view).
  5. Re-GZIPs the blob, appends to heap end, updates the main
     descriptor in the descriptor table.

PCOUNT and ZPCOUNT both grow monotonically; repack() reclaims
orphans.  The original-side descriptor's new_offset placement means
funpack always sees per-cell offsets that don't collide, even when
nelements changes between an edit's old and new value.

Forms covered:
  - hdu[r, "vla_col"] = v       (single cell)
  - hdu["vla_col"] = arr        (whole column; Object ndarray)
  - hdu[i] = record             (single row touching a VLA col)
  - hdu[a:b] = arr              (slice row writes)
  - hdu[[i, j]] = arr           (fancy rows)
  - hdu[a:b:s] = arr            (stepped slice)
  - hdu[[c_fixed, c_vla]] = arr (mixed multi-column subset)
  - hdu["vla_col"][rows] = v    (single-col subset)
  - hdu[[c_fixed, c_vla]][rows] = v (multi-col subset)
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


def _dt_with_vla():
    return np.dtype([("id", "i4"), ("v", "O"), ("c", "f4")])


def _basic_data(start, stop, dt=None, vla_inner=("f4",)):
    if dt is None:
        dt = _dt_with_vla()
    n = stop - start
    arr = np.zeros(n, dtype=dt)
    arr["id"] = np.arange(start, stop, dtype="i4")
    arr["c"] = np.arange(start, stop, dtype="f4") * -1.0
    inner_kind = vla_inner[0]
    for i in range(n):
        # Cell length scales with row index for variety.
        cell_len = (start + i) % 7 + 1
        arr["v"][i] = np.arange(cell_len, dtype=inner_kind) + (start + i)
    return arr


def _make_table(fname, *, nrows, ztilelen, vla_inner="f4"):
    dt = _dt_with_vla()
    data = _basic_data(0, nrows, dt, vla_inner=(vla_inner,))
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            dt,
            nrows=nrows,
            compress=True,
            ztilelen=ztilelen,
            var_dtypes={"v": vla_inner},
        )
        f[1].write(data)
    return data, dt


def _read_and_compare(fname, expected, dt):
    """Both same-handle and reopen reads must match expected."""
    with rustfits.FITS(fname, "r") as f:
        out = f[1].read()
    np.testing.assert_array_equal(out["id"], expected["id"])
    np.testing.assert_array_equal(out["c"], expected["c"])
    assert len(out["v"]) == len(expected["v"])
    for i, _ in enumerate(expected["v"]):
        np.testing.assert_array_equal(out["v"][i], expected["v"][i])


# ---------------------------------------------------------------------
# Single-cell VLA write: hdu[r, "vla_col"] = v
# ---------------------------------------------------------------------


def test_setitem_vla_cell_same_length():
    """Replace a VLA cell with another cell of the SAME length."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        new_cell = np.arange(len(original["v"][50]), dtype="f4") + 1000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][50, "v"] = new_cell
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][50] = new_cell
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_cell_grows_in_size():
    """Replacing a cell with a LONGER cell — orphans old original-heap slot."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        new_cell = np.arange(20, dtype="f4") + 9_000.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][70, "v"] = new_cell
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][70] = new_cell
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_cell_shrinks_to_empty():
    """Replace a cell with an empty cell (nelements=0)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        with rustfits.FITS(fname, "r+") as f:
            f[1][10, "v"] = np.array([], dtype="f4")
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][10] = np.array([], dtype="f4")
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_cell_grows_zpcount():
    """Each cell write appends to ZPCOUNT (monotonically grows)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r") as f:
            zpcount_before = int(f[1].header["ZPCOUNT"])
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1][50, "v"] = np.arange(8, dtype="f4")
        with rustfits.FITS(fname, "r") as f:
            zpcount_after = int(f[1].header["ZPCOUNT"])
            pcount_after = int(f[1].header["PCOUNT"])
        # ZPCOUNT grew by exactly 8 * 4 = 32 bytes (the new cell).
        assert zpcount_after == zpcount_before + 8 * 4
        # PCOUNT grew (new compressed cell bytes + new gzipped blob).
        assert pcount_after > pcount_before


def test_setitem_vla_cell_negative_row_index():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        new_cell = np.arange(3, dtype="f4") + 77.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][-1, "v"] = new_cell
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][-1] = new_cell
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_cell_out_of_range_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=50, ztilelen=25)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(IndexError):
                f[1][50, "v"] = np.arange(2, dtype="f4")


def test_setitem_vla_cell_wrong_inner_dtype_raises():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=50, ztilelen=25)
        # Cell uses i4 but the column's inner is f4.
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises((ValueError, TypeError)):
                f[1][10, "v"] = np.arange(3, dtype="i4")


# ---------------------------------------------------------------------
# Whole-column VLA write: hdu["vla_col"] = arr
# ---------------------------------------------------------------------


def test_setitem_vla_whole_column_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        new_v = np.empty(200, dtype="O")
        for i in range(200):
            new_v[i] = np.arange((i % 5) + 1, dtype="f4") + 10_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
        expected = original.copy()
        expected["v"] = new_v
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_whole_column_other_cols_untouched():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        new_v = np.empty(100, dtype="O")
        for i in range(100):
            new_v[i] = np.array([i * 0.5], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"] = new_v
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        # id and c columns must equal the originals (descriptor and
        # bytes untouched).
        np.testing.assert_array_equal(out["id"], original["id"])
        np.testing.assert_array_equal(out["c"], original["c"])


def test_setitem_vla_whole_column_wrong_dtype_kind_raises():
    """Whole-col VLA write requires Object dtype, not f4."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_table(fname, nrows=100, ztilelen=50)
        arr = np.zeros(100, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(ValueError):
                f[1]["v"] = arr


# ---------------------------------------------------------------------
# Row writes touching a VLA column
# ---------------------------------------------------------------------


def test_setitem_vla_single_row_record():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        rec = np.zeros(1, dtype=dt)
        rec["id"] = 999
        rec["c"] = -9.5
        rec["v"][0] = np.arange(11, dtype="f4") + 5_000
        with rustfits.FITS(fname, "r+") as f:
            f[1][30] = rec[0]
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["id"][30] = 999
        expected["c"][30] = -9.5
        expected["v"][30] = rec["v"][0]
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_slice_rows():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        chunk = np.zeros(50, dtype=dt)
        chunk["id"] = np.arange(50, dtype="i4") + 7_000
        chunk["c"] = np.arange(50, dtype="f4") * -2.0
        for i in range(50):
            chunk["v"][i] = np.arange((i % 4) + 1, dtype="f4") + 33_000
        with rustfits.FITS(fname, "r+") as f:
            f[1][100:150] = chunk
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(range(100, 150)):
            expected["id"][i] = chunk["id"][k]
            expected["c"][i] = chunk["c"][k]
            expected["v"][i] = chunk["v"][k]
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_fancy_rows():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=300, ztilelen=100)
        idx = [10, 110, 210]  # one row per tile
        chunk = np.zeros(3, dtype=dt)
        chunk["id"] = [1, 2, 3]
        chunk["c"] = [-1.0, -2.0, -3.0]
        chunk["v"][0] = np.arange(2, dtype="f4")
        chunk["v"][1] = np.arange(4, dtype="f4") + 100
        chunk["v"][2] = np.array([], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][idx] = chunk
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(idx):
            expected["id"][i] = chunk["id"][k]
            expected["c"][i] = chunk["c"][k]
            expected["v"][i] = chunk["v"][k]
        _read_and_compare(fname, expected, dt)


def test_setitem_vla_stepped_slice():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        chunk = np.zeros(10, dtype=dt)
        chunk["id"] = np.arange(10, dtype="i4")
        chunk["c"] = np.arange(10, dtype="f4") * 0.5
        for i in range(10):
            chunk["v"][i] = np.arange((i % 3) + 1, dtype="f4") + 200
        with rustfits.FITS(fname, "r+") as f:
            f[1][0:200:20] = chunk
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(range(0, 200, 20)):
            expected["id"][i] = chunk["id"][k]
            expected["c"][i] = chunk["c"][k]
            expected["v"][i] = chunk["v"][k]
        _read_and_compare(fname, expected, dt)


# ---------------------------------------------------------------------
# Multi-column subset including a VLA
# ---------------------------------------------------------------------


def test_setitem_multi_col_mixed_fixed_and_vla():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("id", "i4"), ("v", "O")])
        sub = np.zeros(200, dtype=sub_dt)
        sub["id"] = np.arange(200, dtype="i4") + 4_000
        for i in range(200):
            sub["v"][i] = np.arange((i % 6) + 1, dtype="f4") + 50_000
        with rustfits.FITS(fname, "r+") as f:
            f[1][["id", "v"]] = sub
        expected = original.copy()
        expected["v"] = sub["v"].copy()
        expected["id"] = sub["id"]
        # c column unchanged
        _read_and_compare(fname, expected, dt)


def test_setitem_multi_col_vla_only():
    """[['v']] = ... with a VLA-only subset list."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        sub_dt = np.dtype([("v", "O")])
        sub = np.zeros(100, dtype=sub_dt)
        for i in range(100):
            sub["v"][i] = np.array([float(i)], dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1][["v"]] = sub
        expected = original.copy()
        expected["v"] = sub["v"].copy()
        _read_and_compare(fname, expected, dt)


# ---------------------------------------------------------------------
# Subset-object writes: hdu["vla_col"][rows] = value
# ---------------------------------------------------------------------


def test_subset_vla_cell_via_int_row():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        new_cell = np.arange(5, dtype="f4") + 4242.0
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][30] = new_cell
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][30] = new_cell
        _read_and_compare(fname, expected, dt)


def test_subset_vla_slice_rows():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        new_chunk = np.empty(50, dtype="O")
        for i in range(50):
            new_chunk[i] = np.arange((i % 4) + 1, dtype="f4") + 11_000
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][100:150] = new_chunk
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(range(100, 150)):
            expected["v"][i] = new_chunk[k]
        _read_and_compare(fname, expected, dt)


def test_subset_vla_fancy_rows():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        idx = [5, 25, 75]
        new_chunk = np.empty(3, dtype="O")
        new_chunk[0] = np.arange(2, dtype="f4")
        new_chunk[1] = np.array([], dtype="f4")
        new_chunk[2] = np.arange(8, dtype="f4") + 99
        with rustfits.FITS(fname, "r+") as f:
            f[1]["v"][idx] = new_chunk
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(idx):
            expected["v"][i] = new_chunk[k]
        _read_and_compare(fname, expected, dt)


def test_subset_multi_col_mixed_via_subset():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        sub_dt = np.dtype([("c", "f4"), ("v", "O")])
        sub = np.zeros(40, dtype=sub_dt)
        sub["c"] = np.arange(40, dtype="f4") * 0.25
        for i in range(40):
            sub["v"][i] = np.arange((i % 3) + 1, dtype="f4") - 50
        with rustfits.FITS(fname, "r+") as f:
            f[1][["c", "v"]][80:120] = sub
        expected = original.copy()
        expected["v"] = original["v"].copy()
        for k, i in enumerate(range(80, 120)):
            expected["c"][i] = sub["c"][k]
            expected["v"][i] = sub["v"][k]
        _read_and_compare(fname, expected, dt)


# ---------------------------------------------------------------------
# Repack reclaims VLA orphans after setitem
# ---------------------------------------------------------------------


def test_setitem_vla_then_repack_reclaims():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            for k in range(5):
                f[1][20, "v"] = np.arange(10, dtype="f4") + k
            pcount_before = int(f[1].header["PCOUNT"])
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
        assert pcount_after < pcount_before
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][20] = np.arange(10, dtype="f4") + 4  # last-wins
        _read_and_compare(fname, expected, dt)


# ---------------------------------------------------------------------
# Non-last HDU preserves trailing HDU
# ---------------------------------------------------------------------


def test_setitem_vla_non_last_hdu_preserves_trailing():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_with_vla()
        base = _basic_data(0, 100, dt)
        trail = np.arange(40, dtype="i2")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(
                dt,
                nrows=100,
                compress=True,
                ztilelen=50,
                var_dtypes={"v": "f4"},
            )
            f[1].write(base)
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)
        new_cell = np.arange(20, dtype="f4") + 88.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][50, "v"] = new_cell
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
            out = f[1].read()
        np.testing.assert_array_equal(out["v"][50], new_cell)


# ---------------------------------------------------------------------
# String VLA (PA) round-trip
# ---------------------------------------------------------------------


def test_setitem_vla_string_pa_cell():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("s", "O")])
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=10,
                compress=True,
                ztilelen=5,
                var_dtypes={"s": "S"},
            )
            data = np.zeros(10, dtype=dt)
            data["id"] = np.arange(10, dtype="i4")
            for i in range(10):
                data["s"][i] = f"orig_{i}"
            f[1].write(data)
        with rustfits.FITS(fname, "r+") as f:
            f[1][3, "s"] = "hello world"
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert out["s"][3] == "hello world"
        assert out["s"][2] == "orig_2"
        assert out["s"][4] == "orig_4"


# ---------------------------------------------------------------------
# Cross-tool: funpack reads VLA-mutated files
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack required for cross-tool verification",
)
def test_funpack_decompresses_vla_mutated_file():
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=200, ztilelen=100)
        new_cell = np.arange(15, dtype="f4") + 5000.0
        sub_dt = np.dtype([("id", "i4"), ("v", "O")])
        sub = np.zeros(20, dtype=sub_dt)
        sub["id"] = np.arange(20, dtype="i4") + 20_000
        for i in range(20):
            sub["v"][i] = np.arange((i % 5) + 1, dtype="f4") - 999
        with rustfits.FITS(fname, "r+") as f:
            f[1][50, "v"] = new_cell
            f[1][["id", "v"]][100:120] = sub
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][50] = new_cell
        for k, i in enumerate(range(100, 120)):
            expected["id"][i] = sub["id"][k]
            expected["v"][i] = sub["v"][k]
        out_path = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out_path, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out_path, "r") as f:
            cfit = f[1].read()
        np.testing.assert_array_equal(cfit["id"], expected["id"])
        np.testing.assert_array_equal(cfit["c"], expected["c"])
        for i in range(len(expected["v"])):
            n = len(expected["v"][i])
            # fitsio reads VLAs as max-width zero-padded; slice to n.
            np.testing.assert_array_equal(
                np.asarray(cfit["v"][i])[:n], expected["v"][i]
            )


# ---------------------------------------------------------------------
# Same-handle vs reopen parity
# ---------------------------------------------------------------------


def test_setitem_vla_same_handle_and_reopen_agree():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=150, ztilelen=50)
        new_cell = np.arange(7, dtype="f4") - 1.0
        with rustfits.FITS(fname, "r+") as f:
            f[1][70, "v"] = new_cell
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            re = f[1].read()
        np.testing.assert_array_equal(same["id"], re["id"])
        np.testing.assert_array_equal(same["c"], re["c"])
        for i in range(150):
            np.testing.assert_array_equal(same["v"][i], re["v"][i])
        expected = original.copy()
        expected["v"] = original["v"].copy()
        expected["v"][70] = new_cell
        np.testing.assert_array_equal(re["v"][70], expected["v"][70])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
