"""
ZTABLE append on compressed tables.

`CompressedTableHDU.append(data, *, names=None)` extends the table
with new rows.  The new rows merge into the existing partial last
tile up to ZTILELEN (decode + re-encode, old blobs orphaned), then
any remaining rows become fresh full / partial tiles.  Maintains
the FITS Tile Compression Convention's "all tiles same size
except the last" invariant so funpack reads back correctly.

VLA columns on compressed tables are not yet supported on append
(rejected at the pymethod with a clear NotImplementedError).
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


def _basic_data(start, stop, dtype):
    n = stop - start
    arr = np.zeros(n, dtype=dtype)
    arr["id"] = np.arange(start, stop, dtype="i4")
    arr["v"] = np.arange(start, stop, dtype="f8") * 0.25
    arr["c"] = np.arange(start, stop, dtype="f4") * -1.0
    return arr


def _make_table(fname, nrows, ztilelen, *, with_image_first=False):
    """Create a compressed table with `nrows` rows; return the written data."""
    dt = np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])
    data = _basic_data(0, nrows, dt)
    with rustfits.FITS(fname, "w+") as f:
        if with_image_first:
            f.create_image_hdu("i4", (1,))
        f.create_table_hdu(dt, nrows=nrows, compress=True, ztilelen=ztilelen)
        idx = 1 if with_image_first else 1  # primary auto-created
        f[idx].write(data)
    return data, dt


def _table_idx_in_file(f):
    """Find the CompressedTableHDU index — primary autocreate puts it at 1."""
    for i, hdu in enumerate(f):
        if isinstance(hdu, rustfits.CompressedTableHDU):
            return i
    raise AssertionError("no CompressedTableHDU found")


# ---------------------------------------------------------------------
# Append: round-trip semantics
# ---------------------------------------------------------------------


def test_append_into_partial_last_tile_no_new_tiles():
    """
    Append rows that fit entirely in the existing partial last tile.
    No new tiles created; merge step decodes + re-encodes the last
    tile only.  PCOUNT grows (old last-tile blobs become orphans).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        # 600 rows, ztilelen=400 -> tile 0 full (400), tile 1 partial (200).
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].n_tiles == 2
        # Append 100 -> merges into tile 1 (200 + 100 = 300, still partial).
        added = _basic_data(600, 700, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 700
            assert f[1].n_tiles == 2  # still 2 tiles
            out = f[1].read()
        expected = np.concatenate([original, added])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


def test_append_exactly_fills_last_tile():
    """
    Append rows that exactly fill the existing partial last tile.
    No new tiles; merged tile becomes full.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        # 600 rows, ztilelen=400 -> last tile has 200 rows; append exactly 200.
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        added = _basic_data(600, 800, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 800
            assert f[1].n_tiles == 2  # both tiles now full
            out = f[1].read()
        expected = np.concatenate([original, added])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


def test_append_overflows_into_new_tile():
    """
    Append more rows than fit in the partial last tile.  Merge fills
    the last tile to ZTILELEN, remaining rows go into a fresh tile.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        # 600 rows, ztilelen=400 -> last tile has 200 rows.
        # Append 300: 200 merge + 100 new tile.
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        added = _basic_data(600, 900, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 900
            assert f[1].n_tiles == 3
            out = f[1].read()
        expected = np.concatenate([original, added])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


def test_append_with_no_room_in_last_tile():
    """
    Last existing tile is already full (nrows % ZTILELEN == 0).  No
    merge — all appended rows become fresh tiles.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        # 800 rows, ztilelen=400 -> both tiles full.
        original, dt = _make_table(fname, nrows=800, ztilelen=400)
        added = _basic_data(800, 1300, dt)  # 500 rows -> 1 full + 1 partial
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 1300
            assert f[1].n_tiles == 4
            out = f[1].read()
        expected = np.concatenate([original, added])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


def test_multiple_appends_accumulate():
    """
    Three appends in sequence; each works against the previous state.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        a1 = _basic_data(600, 700, dt)
        a2 = _basic_data(700, 1500, dt)
        a3 = _basic_data(1500, 1700, dt)
        for chunk in (a1, a2, a3):
            with rustfits.FITS(fname, "r+") as f:
                f[1].append(chunk)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        expected = np.concatenate([original, a1, a2, a3])
        for col in dt.names:
            np.testing.assert_array_equal(out[col], expected[col])


# ---------------------------------------------------------------------
# Append: input forms (mirrors uncompressed TableHDU.append)
# ---------------------------------------------------------------------


def test_append_accepts_dict_input():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        added = {
            "id": np.arange(600, 800, dtype="i4"),
            "v": np.arange(600, 800, dtype="f8") * 0.25,
            "c": np.arange(600, 800, dtype="f4") * -1.0,
        }
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert out.shape == (800,)


def test_append_accepts_list_plus_names():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        ids = np.arange(600, 800, dtype="i4")
        vs = np.arange(600, 800, dtype="f8") * 0.25
        cs = np.arange(600, 800, dtype="f4") * -1.0
        with rustfits.FITS(fname, "r+") as f:
            f[1].append([ids, vs, cs], names=["id", "v", "c"])
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 800


def test_extend_alias():
    """`extend()` is the symmetric alias to `append()`."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        added = _basic_data(600, 700, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].extend(added)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 700


def test_append_zero_rows_is_noop():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        empty = np.empty(0, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(empty)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 600


# ---------------------------------------------------------------------
# Append: non-last-HDU case
# ---------------------------------------------------------------------


def test_append_non_last_hdu_preserves_trailing_hdu():
    """
    Append to a compressed table that's NOT the last HDU on disk.
    The trailing HDU's content must survive the data-section grow +
    heap relocation (block-aligned, so trailing HDU header stays
    block-aligned).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])
        src = _basic_data(0, 400, dt)
        trail = np.arange(120, dtype="i2").reshape(10, 12)
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_table_hdu(dt, nrows=400, compress=True, ztilelen=200)
            f[1].write(src)
            f.create_image_hdu("i2", trail.shape, extname="TRAIL")
            f[2].write(trail)
        added = _basic_data(400, 550, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"
        with rustfits.FITS(fname, "r") as f:
            assert f[1].nrows == 550
            np.testing.assert_array_equal(f[2].read(), trail)
            assert f[2].extname == "TRAIL"


# ---------------------------------------------------------------------
# Append: cross-tool funpack interop
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_appended_table():
    """
    cfitsio's funpack reads a file we created + appended to and
    reconstructs the full row sequence byte-exactly.
    """
    fitsio = pytest.importorskip("fitsio")

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=800, ztilelen=300)
        added = _basic_data(800, 1300, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        out = os.path.join(td, "unfz.fits")
        subprocess.run(
            ["funpack", "-O", out, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out, "r") as f:
            cfit = f[1].read()
        expected = np.concatenate([original, added])
        for col in dt.names:
            np.testing.assert_array_equal(cfit[col], expected[col])


# ---------------------------------------------------------------------
# Header bookkeeping: PCOUNT grows monotonically, ZNAXIS2 updates
# ---------------------------------------------------------------------


def test_pcount_grows_after_append_with_merge():
    """
    Merging into a partial tile orphans the old last-tile blobs;
    PCOUNT grows by the new blob sizes (NOT compacted in-place).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        with rustfits.FITS(fname, "r") as f:
            pcount_before = int(f[1].header["PCOUNT"])
        added = _basic_data(600, 700, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            pcount_after = int(f[1].header["PCOUNT"])
            assert pcount_after > pcount_before


def test_znaxis2_updated_after_append():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        original, dt = _make_table(fname, nrows=600, ztilelen=400)
        added = _basic_data(600, 850, dt)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(added)
        with rustfits.FITS(fname, "r") as f:
            assert int(f[1].header["ZNAXIS2"]) == 850
            assert int(f[1].header["NAXIS2"]) == 3  # ceil(850/400)


# ---------- streaming-create (nrows=0) ZTILELEN regressions ---------------


def test_explicit_ztilelen_preserved_when_nrows_zero():
    """
    Regression for the ZTILELEN-collapses-when-nrows-zero bug:
    ``create_table_hdu(nrows=0, compress=True, ztilelen=K)`` must
    record ``ZTILELEN=K`` on disk, NOT cap to 1.  The buggy logic
    capped the user's value by ``nrows.max(1)`` which collapsed to
    1 when nrows=0, forcing every subsequent appended row into its
    own tile (each independently gzipped — catastrophic).
    """
    dt = np.dtype([("a", "f4"), ("b", "f4")])
    for ztilelen in (100, 1000, 10_000):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "t.fits")
            with rustfits.FITS(fname, "w+") as f:
                f.create_table_hdu(
                    dt, nrows=0, compress=True, ztilelen=ztilelen
                )
                assert int(f[1].header["ZTILELEN"]) == ztilelen
            # post-reopen check
            with rustfits.FITS(fname, "r") as f:
                assert int(f[1].header["ZTILELEN"]) == ztilelen


def test_default_ztilelen_sensible_when_nrows_zero():
    """
    Regression: ``create_table_hdu(nrows=0, compress=True)`` (no
    explicit ztilelen) must default to the cfitsio-style ~10 MB
    cap, NOT 1.  Previously it returned 1 because
    ``default_ztilelen(nrows=0, ...)`` short-circuited.
    """
    dt = np.dtype([("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4")])
    # row_width = 16 bytes → cap = 10_000_000 / 16 = 625_000
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=0, compress=True)
            assert int(f[1].header["ZTILELEN"]) == 625_000


def test_streaming_create_then_append_uses_user_ztilelen():
    """
    End-to-end: ``create_table_hdu(nrows=0, ..., ztilelen=K)``
    followed by ``append(chunk)`` must produce ``ceil(N/K)`` tiles,
    not N tiles (one per row).  This is the user-visible symptom of
    the ZTILELEN-collapses bug.
    """
    dt = np.dtype([("x", "f4"), ("y", "f4")])
    N = 5000
    K = 1000
    data = np.empty(N, dtype=dt)
    data["x"] = np.arange(N, dtype="f4")
    data["y"] = np.arange(N, dtype="f4") * 2.0
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=0, compress=True, ztilelen=K)
            f[1].append(data)
            # Expected: 5 tiles (5000/1000), not 5000 (one per row).
            assert int(f[1].header["NAXIS2"]) == N // K
            assert int(f[1].header["ZTILELEN"]) == K
            assert int(f[1].header["ZNAXIS2"]) == N
        # round-trip data integrity
        with rustfits.FITS(fname, "r") as f:
            rt = f[1].read()
            np.testing.assert_array_equal(rt["x"], data["x"])
            np.testing.assert_array_equal(rt["y"], data["y"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
