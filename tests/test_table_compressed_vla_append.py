"""
ZTABLE Phase 6c-1b — VLA append on compressed tables.

Lifts the VLA-append rejection.  For the merge tile, existing
per-cell compressed bytes are copied verbatim (no decode +
re-encode); merge-tile new rows go through the per-cell encode +
uncompressed-fallback path; new tiles use the Phase 6a per-tile
encoder.  Original-heap offsets in the dual-descriptor blob
continue from the current ZPCOUNT so funpack's reconstructed
file stays consistent.
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


def _vla_arr_equal(a, b):
    if len(a) != len(b):
        return False
    return all(np.array_equal(a[i], b[i]) for i in range(len(a)))


def _make_vla_data(start, stop, dt, inner="f4"):
    """Build a structured ndarray with one fixed and one VLA col."""
    n = stop - start
    arr = np.empty(n, dtype=dt)
    arr["id"] = np.arange(start, stop, dtype="i4")
    for i, gid in enumerate(range(start, stop)):
        arr["v"][i] = (np.arange(gid % 7, dtype=inner) * 0.25).astype(inner)
    return arr


# ---------------------------------------------------------------------
# Basic round-trip: append entirely inside a fresh tile
# ---------------------------------------------------------------------


def test_vla_append_into_fresh_tile():
    """
    Initial nrows == ztilelen (no partial last tile), append
    creates a fresh tile.  Merge path is skipped.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 400, dt)
        more = _make_vla_data(400, 550, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=400,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            reopened = f[1].read()
        combined = np.concatenate([base, more])
        for out in (same, reopened):
            np.testing.assert_array_equal(out["id"], combined["id"])
            assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Merge path: append rows that fit in the existing partial last tile
# ---------------------------------------------------------------------


def test_vla_append_merge_into_partial_last_tile():
    """
    Initial nrows (600) > ztilelen (400) → 2 tiles, last is
    partial (200 rows).  Append 100 rows that all merge into
    that partial tile (no new tile created).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 700, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            assert f[1].n_tiles == 2
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            reopened = f[1].read()
        combined = np.concatenate([base, more])
        for out in (same, reopened):
            np.testing.assert_array_equal(out["id"], combined["id"])
            assert _vla_arr_equal(out["v"], combined["v"])


def test_vla_append_merge_and_spill_to_new_tile():
    """
    Append rows that fill the partial last tile AND overflow into
    one (or more) fresh tiles.  Exercises both merge + new-tile
    VLA branches in one call.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        # 600 base → 2 tiles, last has 200 rows; +300 = 200 merge
        # + 100 spill into a 3rd tile.
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 900, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            assert f[1].n_tiles == 3
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            reopened = f[1].read()
        combined = np.concatenate([base, more])
        for out in (same, reopened):
            np.testing.assert_array_equal(out["id"], combined["id"])
            assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Mixed schema: append on a table with one fixed + one VLA col
# ---------------------------------------------------------------------


def test_vla_append_mixed_fixed_and_vla_columns():
    """
    Table has BOTH a fixed col (id) and a VLA col (v).  Append
    must dispatch per-col: fixed → existing path, VLA → the new
    VLA branch.  Verifies both branches play nicely in one call.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 850, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            same = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            reopened = f[1].read()
        combined = np.concatenate([base, more])
        for out in (same, reopened):
            np.testing.assert_array_equal(out["id"], combined["id"])
            assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Empty cells, multiple VLA columns
# ---------------------------------------------------------------------


def test_vla_append_with_empty_cells():
    """
    Some rows have nelements=0 (descriptor is (0, 0) or
    (0, current_offset)) — these have no per-cell heap bytes.
    Append must handle empty cells in both merge and new-tile.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = np.empty(250, dtype=dt)
        base["id"] = np.arange(250, dtype="i4")
        for i in range(250):
            n = 0 if i % 3 == 0 else (i % 5 + 1)
            base["v"][i] = np.arange(n, dtype="f4")
        more = np.empty(200, dtype=dt)
        more["id"] = np.arange(250, 450, dtype="i4")
        for i in range(200):
            n = 0 if i % 4 == 0 else (i % 3 + 1)
            more["v"][i] = np.arange(n, dtype="f4") * 2.0
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=250,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


def test_vla_append_multiple_vla_columns():
    """
    Two VLA columns + one fixed column.  plan_vla_heap_layout
    walks both VLA cols' cells per row in a single cursor; both
    must come back correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O"), ("w", "O")])
        nrows_base = 300

        def build(start, stop):
            n = stop - start
            arr = np.empty(n, dtype=dt)
            arr["id"] = np.arange(start, stop, dtype="i4")
            for i, gid in enumerate(range(start, stop)):
                arr["v"][i] = (np.arange(gid % 5, dtype="f4") + 0.1).astype(
                    "f4"
                )
                arr["w"][i] = np.arange(gid % 4, dtype="i2")
            return arr

        base = build(0, nrows_base)
        more = build(nrows_base, nrows_base + 200)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows_base,
                var_dtypes={"v": "f4", "w": "i2"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])
        assert _vla_arr_equal(out["w"], combined["w"])


# ---------------------------------------------------------------------
# ZPCOUNT bookkeeping: original-heap size grows monotonically
# ---------------------------------------------------------------------


def test_zpcount_grows_after_vla_append():
    """
    ZPCOUNT records the ORIGINAL (uncompressed) heap size.  After
    appending VLA rows the value must grow by exactly the bytes the
    new cells would occupy uncompressed.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        # Each row has nelements = (i % 7) f4 values; sum the bytes
        # for the base and the append, expect ZPCOUNT to grow by
        # exactly that.
        base_nrows = 250
        added_nrows = 200
        base = np.empty(base_nrows, dtype=dt)
        for i in range(base_nrows):
            base["v"][i] = np.arange(i % 7, dtype="f4")
        more = np.empty(added_nrows, dtype=dt)
        for i in range(added_nrows):
            more["v"][i] = np.arange(i % 5, dtype="f4")
        base_bytes = sum(len(base["v"][i]) for i in range(base_nrows)) * 4
        more_bytes = sum(len(more["v"][i]) for i in range(added_nrows)) * 4
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=base_nrows,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r") as f:
            zpcount_before = int(f[1].header["ZPCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            zpcount_after = int(f[1].header["ZPCOUNT"])
        assert zpcount_before == base_bytes
        assert zpcount_after == base_bytes + more_bytes


# ---------------------------------------------------------------------
# Multiple sequential appends — each accumulates ZPCOUNT + PCOUNT
# ---------------------------------------------------------------------


def test_vla_multiple_sequential_appends():
    """
    Three appends in a row.  Each must round-trip and accumulate
    rows + ZPCOUNT correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        chunks = [
            _make_vla_data(0, 200, dt),
            _make_vla_data(200, 350, dt),
            _make_vla_data(350, 500, dt),
            _make_vla_data(500, 750, dt),
        ]
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=200,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(chunks[0])
        for chunk in chunks[1:]:
            with rustfits.FITS(fname, "r+") as f:
                f[1].append(chunk)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate(chunks)
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Non-last HDU: append shifts trailing HDU forward
# ---------------------------------------------------------------------


def test_vla_append_non_last_hdu_preserves_trailing():
    """
    Compressed VLA table with another HDU after it; append must
    shift the trailing HDU forward and leave its bytes intact.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        trail = np.arange(40, dtype="i2")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)
        more = _make_vla_data(600, 900, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            np.testing.assert_array_equal(f[2].read(), trail)
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# String VLA ('PA') append
# ---------------------------------------------------------------------


def test_vla_append_string_pa_column():
    """
    'PA' (ASCII string VLA) inner type works end-to-end through
    append: existing rows preserved, new rows added correctly.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("s", "O")])
        base = np.empty(250, dtype=dt)
        for i in range(250):
            base["s"][i] = f"r{i:04d}"
        more = np.empty(200, dtype=dt)
        for i in range(200):
            more["s"][i] = f"new{i:03d}"
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=250,
                var_dtypes={"s": "S"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([base, more])
        for i in range(450):
            assert out["s"][i] == combined["s"][i]


# ---------------------------------------------------------------------
# Cross-tool: funpack on a VLA-appended compressed table
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_vla_appended_file():
    """
    Build a VLA-bearing compressed table, append some rows, run
    funpack, then verify the decompressed table's contents.  This
    is the strongest check that the dual-descriptor blob + the
    original-heap offsets are well-formed (funpack uses both;
    incorrect offsets would collide cells in the output heap).
    fitsio reads VLAs as max-width zero-padded arrays; we compare
    each cell's first `len(src_cell)` elements.
    """
    import fitsio
    import warnings

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 900, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        out_path = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out_path, fname],
            check=True,
            capture_output=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fitsio.FITS(out_path, "r") as f:
                cfit = f[1].read()
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(cfit["id"], combined["id"])
        for i in range(len(combined)):
            n = len(combined["v"][i])
            np.testing.assert_array_equal(
                np.asarray(cfit["v"][i])[:n],
                combined["v"][i],
                err_msg=f"row {i}",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
