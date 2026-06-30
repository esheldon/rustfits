"""
ZTABLE Phase 6c-1 — repack() on compressed tables.

Reclaims heap orphans accumulated by append-with-merge.  Streaming
in-place fast path when the orphan pattern fits the post-merge
shape; staging fallback for arbitrary patterns (groundwork for
__setitem__ in Phase 6c-2).  Memory bounded by ~1 MiB plus the
descriptor table.

VLA columns on compressed tables are not yet supported on
repack() — the dual-descriptor heap layout needs an
indirection-aware rewrite.  Rejected with a clear error.
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


def _data(start, stop, dt):
    n = stop - start
    arr = np.zeros(n, dtype=dt)
    arr["id"] = np.arange(start, stop, dtype="i4")
    arr["v"] = np.arange(start, stop, dtype="f8") * 0.25
    arr["c"] = np.arange(start, stop, dtype="f4") * -1.0
    return arr


def _make_and_append(fname, *, initial_nrows, ztilelen, append_nrows):
    """Create a table + one append; return (combined_data, dt)."""
    dt = np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])
    base = _data(0, initial_nrows, dt)
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            dt, nrows=initial_nrows, compress=True, ztilelen=ztilelen
        )
        f[1].write(base)
    if append_nrows > 0:
        more = _data(initial_nrows, initial_nrows + append_nrows, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        combined = np.concatenate([base, more])
    else:
        combined = base
    return combined, dt


# ---------------------------------------------------------------------
# Basic repack semantics: PCOUNT shrinks, data preserved
# ---------------------------------------------------------------------


def test_repack_reclaims_merge_orphans():
    """
    Append-with-merge orphans the old last-tile blobs; repack
    reclaims them.  PCOUNT shrinks; data round-trips unchanged.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        combined, dt = _make_and_append(
            fname, initial_nrows=600, ztilelen=400, append_nrows=100
        )
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            assert pcount_after < pcount_before, (
                f"PCOUNT did not shrink: {pcount_before} -> {pcount_after}"
            )
            out = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(out[col], combined[col])


def test_repack_is_noop_when_compact():
    """
    A fresh compressed table (no append → no orphans) is already
    compact; repack is a no-op.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        combined, dt = _make_and_append(
            fname, initial_nrows=800, ztilelen=400, append_nrows=0
        )
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            assert pcount_after == pcount_before
            out = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(out[col], combined[col])


def test_repack_after_multiple_appends_with_merge():
    """
    Several appends, each merging into the current partial last
    tile, accumulate orphans.  Repack reclaims them all.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])
        chunks = [
            _data(0, 500, dt),
            _data(500, 600, dt),
            _data(600, 700, dt),
            _data(700, 850, dt),
        ]
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=500, compress=True, ztilelen=400)
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
            assert pcount_after < pcount_before
            out = f[1].read()
        expected = np.concatenate(chunks)
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# File-shrink behavior: data section block-aligned, file size drops
# ---------------------------------------------------------------------


def test_repack_shrinks_last_hdu_file_size():
    """
    On the last HDU, repack drops the file size by the reclaimed
    blocks (via set_len, since there's nothing after to shift).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _, _ = _make_and_append(
            fname, initial_nrows=600, ztilelen=400, append_nrows=100
        )
        size_before = os.path.getsize(fname)
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        size_after = os.path.getsize(fname)
        # File should be at most as big as before (and probably
        # smaller; merge orphans typically span at least one block).
        assert size_after <= size_before


def test_repack_non_last_hdu_preserves_trailing_hdu():
    """
    Trailing HDU survives repack on a compressed table that's not
    the last HDU on disk.  This exercises the shift_file_tail_backward
    path with the corrected delta computation.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])
        base = _data(0, 600, dt)
        trail = np.arange(40, dtype="i2")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=600, compress=True, ztilelen=400)
            f[1].write(base)
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)
        more = _data(600, 700, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
            f[1].repack()
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        expected = np.concatenate([base, more])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------


def test_repack_clears_tile_cache():
    """
    Repack rewrites descriptors, so prior cache entries are stale.
    Repack clears the cache.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _, _ = _make_and_append(
            fname, initial_nrows=600, ztilelen=400, append_nrows=100
        )
        with rustfits.FITS(fname, "r+") as f:
            hdu = f[1]
            # Warm the cache by reading.
            _ = hdu.read()
            assert hdu.tile_cache_used > 0
            hdu.repack()
            assert hdu.tile_cache_used == 0


# ---------------------------------------------------------------------
# Cross-tool: funpack reads repacked file
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_repacked_file():
    fitsio = pytest.importorskip("fitsio")

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        combined, dt = _make_and_append(
            fname, initial_nrows=800, ztilelen=300, append_nrows=500
        )
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        out = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out, "r") as f:
            cfit = f[1].read()
        for col in dt.names:
            np.testing.assert_array_equal(cfit[col], combined[col])


# ---------------------------------------------------------------------
# Same-handle vs post-reopen verification
# ---------------------------------------------------------------------


def test_repack_round_trip_same_handle_and_reopen():
    """
    Repack visible through the same FITS handle AND after close+
    reopen — covers the in-memory cards commit + on-disk
    persistence.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        combined, dt = _make_and_append(
            fname, initial_nrows=600, ztilelen=400, append_nrows=150
        )
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
            same_handle_out = f[1].read()
            same_handle_pcount = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r") as f:
            reopen_out = f[1].read()
            reopen_pcount = int(f[1].header["PCOUNT"])
        assert same_handle_pcount == reopen_pcount
        for col in dt.names:
            np.testing.assert_array_equal(same_handle_out[col], combined[col])
            np.testing.assert_array_equal(reopen_out[col], combined[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
