"""
ZTABLE Phase 6c-2 prep — VLA `repack()` on compressed tables.

Lifts the VLA-rejection in `CompressedTableHDU.repack()`.
Streaming, staging-only path: per (tile, col), per-cell
compressed bytes get copied old → staging, dual-descriptor
blob gets re-gzipped with rewritten cvlastart values, blob
written to staging.  Then one big back-copy staging → heap,
file shrink, descriptor + PCOUNT rewrite.

Mixed-table (fixed + VLA cols) tables also take this path.
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
    n = stop - start
    arr = np.empty(n, dtype=dt)
    arr["id"] = np.arange(start, stop, dtype="i4")
    for i, gid in enumerate(range(start, stop)):
        arr["v"][i] = (np.arange(gid % 7, dtype=inner) * 0.25).astype(inner)
    return arr


# ---------------------------------------------------------------------
# Basic VLA repack semantics: PCOUNT shrinks after append orphans
# ---------------------------------------------------------------------


def test_table_vla_repack_reclaims_append_merge_orphans():
    """
    Append with merge orphans the old dual-descriptor blob + the
    old per-cell bytes for the merged tile.  Repack reclaims
    them; data round-trips unchanged.
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
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            out = f[1].read()
        assert pcount_after < pcount_before, (
            f"PCOUNT did not shrink: {pcount_before} -> {pcount_after}"
        )
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


def test_table_vla_repack_noop_when_compact():
    """
    Fresh compressed VLA table (no append → no orphans).  Repack
    re-gzips the dual-descriptor blobs but should leave PCOUNT
    unchanged (default gzip level is deterministic).  Data round-
    trips identically.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        src = _make_vla_data(0, 600, dt)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=600,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
            before_data = f[1].read()
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            after_data = f[1].read()
        assert pcount_after == pcount_before
        np.testing.assert_array_equal(after_data["id"], before_data["id"])
        assert _vla_arr_equal(after_data["v"], before_data["v"])


# ---------------------------------------------------------------------
# Multiple accumulating appends → one big repack
# ---------------------------------------------------------------------


def test_table_vla_repack_after_multiple_appends_with_merge():
    """
    Several VLA appends, each merging into the current partial
    last tile, accumulate orphans of both kinds (per-cell bytes
    + old dual-descriptor blobs).  Repack reclaims them all.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        chunks = [
            _make_vla_data(0, 500, dt),
            _make_vla_data(500, 600, dt),
            _make_vla_data(600, 700, dt),
            _make_vla_data(700, 850, dt),
        ]
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=500,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(chunks[0])
        for chunk in chunks[1:]:
            with rustfits.FITS(fname, "r+") as f:
                f[1].append(chunk)
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            out = f[1].read()
        assert pcount_after < pcount_before
        expected = np.concatenate(chunks)
        np.testing.assert_array_equal(out["id"], expected["id"])
        assert _vla_arr_equal(out["v"], expected["v"])


# ---------------------------------------------------------------------
# File-shrink: last vs non-last HDU
# ---------------------------------------------------------------------


def test_table_vla_repack_shrinks_last_hdu_file_size():
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
        size_before = os.path.getsize(fname)
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        size_after = os.path.getsize(fname)
        # Should be at most as big (and usually smaller; orphans
        # typically span at least one block).
        assert size_after <= size_before


def test_table_vla_repack_non_last_hdu_preserves_trailing():
    """
    Compressed VLA table is NOT the last HDU.  After repack, the
    trailing HDU's content must survive the back-copy + shrink +
    shift_file_tail_backward sequence.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 800, dt)
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
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            f[1].repack()
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Mixed schema: fixed + VLA cols in the same table
# ---------------------------------------------------------------------


def test_table_vla_repack_mixed_fixed_and_vla_columns():
    """
    Table has both fixed (id, big_fixed) and VLA (v) cols.  Repack
    must handle each (tile, col) correctly: fixed cols stream-copy
    their blobs, VLA cols re-encode their dual-descriptor blobs.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("big", ("f8", (3,))), ("v", "O")])

        def build(start, stop):
            n = stop - start
            arr = np.empty(n, dtype=dt)
            arr["id"] = np.arange(start, stop, dtype="i4")
            arr["big"] = np.arange(n * 3, dtype="f8").reshape(n, 3) * 0.5
            for i, gid in enumerate(range(start, stop)):
                arr["v"][i] = np.arange(gid % 6, dtype="f4") * 0.25
            return arr

        base = build(0, 600)
        more = build(600, 800)
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
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            out = f[1].read()
        assert pcount_after < pcount_before
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        np.testing.assert_array_equal(out["big"], combined["big"])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Multiple VLA columns
# ---------------------------------------------------------------------


def test_table_vla_repack_multiple_vla_columns():
    """
    Two VLA cols + one fixed.  Repack walks both VLA cols per
    (tile, col), each with its own dual-descriptor blob.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O"), ("w", "O")])

        def build(start, stop):
            n = stop - start
            arr = np.empty(n, dtype=dt)
            arr["id"] = np.arange(start, stop, dtype="i4")
            for i, gid in enumerate(range(start, stop)):
                arr["v"][i] = np.arange(gid % 5, dtype="f4") + 0.1
                arr["w"][i] = np.arange(gid % 4, dtype="i2")
            return arr

        base = build(0, 500)
        more = build(500, 700)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=500,
                var_dtypes={"v": "f4", "w": "i2"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(base)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([base, more])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])
        assert _vla_arr_equal(out["w"], combined["w"])


# ---------------------------------------------------------------------
# ZPCOUNT is invariant under repack
# ---------------------------------------------------------------------


def test_table_vla_repack_preserves_zpcount():
    """
    ZPCOUNT is the original (uncompressed) heap size — the sum
    of `vlalen * elem_size` over live cells.  Repack only
    reorganizes compressed bytes; nelements per cell is
    unchanged.  ZPCOUNT must NOT move.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        for i in range(nrows):
            src["v"][i] = np.arange(i % 7, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        # Append to create orphans.
        more = np.empty(100, dtype=dt)
        for i in range(100):
            more["v"][i] = np.arange(i % 4, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            zpcount_before = int(f[1].header["ZPCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            zpcount_after = int(f[1].header["ZPCOUNT"])
        assert zpcount_after == zpcount_before


# ---------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------


def test_table_vla_repack_clears_tile_cache():
    """
    Repack rewrites every descriptor, so prior cache entries are
    stale.  Repack must clear the cache.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        for i in range(nrows):
            src["v"][i] = np.arange(i % 5, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        more = np.empty(100, dtype=dt)
        for i in range(100):
            more["v"][i] = np.arange(i % 3, dtype="f4")
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[1]
            _ = hdu.read()  # warm the cache
            assert hdu.tile_cache_used > 0
            hdu.repack()
            assert hdu.tile_cache_used == 0


# ---------------------------------------------------------------------
# Empty cells survive repack
# ---------------------------------------------------------------------


def test_table_vla_repack_preserves_empty_cells():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        for i in range(nrows):
            n = 0 if i % 4 == 0 else (i % 5 + 1)
            src["v"][i] = np.arange(n, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        more = np.empty(100, dtype=dt)
        for i in range(100):
            n = 0 if i % 3 == 0 else (i % 4 + 1)
            more["v"][i] = np.arange(n, dtype="f4") * -1.0
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([src, more])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# Cross-tool: funpack reads a repacked VLA file
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_repacked_vla_file():
    """
    Strongest check that repack-rewritten dual-descriptor blobs
    + per-cell offsets are well-formed: funpack reconstructs the
    original BINTABLE byte-exactly (within the per-row nelements
    bound that fitsio's max-width-pad VLA read returns).
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
            f[1].repack()
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


# ---------------------------------------------------------------------
# Repack → append → repack: round trip after composition
# ---------------------------------------------------------------------


def test_table_vla_repack_then_append_then_repack():
    """
    Compose mutations: write, append, repack, append again,
    repack again.  Catches stale-state bugs between repack +
    subsequent append.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        chunk1 = _make_vla_data(600, 800, dt)
        chunk2 = _make_vla_data(800, 1000, dt)
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
            f[1].append(chunk1)
            f[1].repack()
            f[1].append(chunk2)
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([base, chunk1, chunk2])
        np.testing.assert_array_equal(out["id"], combined["id"])
        assert _vla_arr_equal(out["v"], combined["v"])


# ---------------------------------------------------------------------
# String VLA ('PA') survives repack
# ---------------------------------------------------------------------


def test_table_vla_repack_string_pa_column():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("s", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        for i in range(nrows):
            src["s"][i] = f"r{i:04d}"
        more = np.empty(100, dtype=dt)
        for i in range(100):
            more["s"][i] = f"new{i:03d}"
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"s": "S"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        combined = np.concatenate([src, more])
        for i in range(700):
            assert out["s"][i] == combined["s"][i]


# ---------------------------------------------------------------------
# Same-handle vs post-reopen verification
# ---------------------------------------------------------------------


def test_table_vla_repack_round_trip_same_handle_and_reopen():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        base = _make_vla_data(0, 600, dt)
        more = _make_vla_data(600, 750, dt)
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
            f[1].repack()
            same = f[1].read()
            same_pcount = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r") as f:
            reopened = f[1].read()
            reopened_pcount = int(f[1].header["PCOUNT"])
        assert same_pcount == reopened_pcount
        combined = np.concatenate([base, more])
        for out in (same, reopened):
            np.testing.assert_array_equal(out["id"], combined["id"])
            assert _vla_arr_equal(out["v"], combined["v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
