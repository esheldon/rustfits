"""
ZTABLE slicing, __getitem__, column-subset reads, and
the per-(tile, column) decompressed-bytes cache.

Phase 3 builds on Phase 2 (whole-table read).  New surface:
  - read(rows=slice | iterable) row subset / slicing
  - __getitem__: int / slice / fancy / col-name / col-list
  - CompressedSingleColumnSubset, CompressedColumnSubset
  - Tile cache: set_tile_cache_size / tile_cache_used / clear_tile_cache
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


pytestmark = pytest.mark.skipif(
    shutil.which("fpack") is None,
    reason="fpack (cfitsio CLI) is required to build ZTABLE fixtures",
)


def _make_ztable_fixture(td, data, *, ztilelen=None):
    """
    Helper — write `data` to BINTABLE, fpack -table it, return .fz
    path.  Asserts the output HDU actually IS compressed (cfitsio's
    fits_compress_table copies the HDU verbatim when the data extent
    is under 5760 bytes — easy trap when shrinking a fixture for a
    test).
    """
    src = os.path.join(td, "src.fits")
    with fitsio.FITS(src, "rw", clobber=True) as f:
        f.write(data)
        if ztilelen is not None:
            f[-1].write_key("FZTILELN", ztilelen)
    subprocess.run(
        ["fpack", "-table", src],
        check=True,
        capture_output=True,
    )
    fz = src + ".fz"
    with rustfits.FITS(fz, "r") as f:
        assert isinstance(f[1], rustfits.CompressedTableHDU), (
            "fpack -table did not compress this fixture (table likely "
            "under the 5760-byte cfitsio threshold); use a larger nrows"
        )
    return src, fz


def _make_default_fixture(td, *, nrows=2000, ztilelen=None):
    """
    Standard fixture used by most tests below: a-i4, b-f8, c-f4.
    Default nrows=2000 keeps the data extent well above the 5760-byte
    cfitsio compression threshold (2000 rows * 16 bytes/row ~= 32 KB).
    """
    dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "f4")])
    arr = np.zeros(nrows, dtype=dt)
    arr["a"] = np.arange(nrows, dtype="i4")
    arr["b"] = np.arange(nrows, dtype="f8") * 0.25
    arr["c"] = np.arange(nrows, dtype="f4") * -1.0
    _, fz = _make_ztable_fixture(td, arr, ztilelen=ztilelen)
    return arr, fz


# ---------------------------------------------------------------------
# read(rows=slice / iterable / int subset)
# ---------------------------------------------------------------------


def test_read_rows_slice():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000, ztilelen=500)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=slice(100, 200))
        assert arr.shape == (100,)
        np.testing.assert_array_equal(arr["a"], src["a"][100:200])
        np.testing.assert_array_equal(arr["b"], src["b"][100:200])


def test_read_rows_slice_across_tiles():
    """
    Slice that spans several tiles: descriptor + heap reads for
    multiple tiles, output assembled in row order.
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000, ztilelen=300)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles == 7  # 2000 / 300 = 6.67 -> 7 tiles
            arr = hdu.read(rows=slice(250, 1050))  # spans tiles 0..3
        np.testing.assert_array_equal(arr["a"], src["a"][250:1050])


def test_read_rows_slice_stepped():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000, ztilelen=500)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=slice(0, 1000, 7))
        expected = src[slice(0, 1000, 7)]
        np.testing.assert_array_equal(arr["a"], expected["a"])
        np.testing.assert_array_equal(arr["b"], expected["b"])


def test_read_rows_list_preserves_order():
    """
    rows=[4, 1, 9, 2] should return rows in the user's requested
    order (not sorted by disk position).
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=200, ztilelen=50)
        wanted = [4, 1, 9, 2, 100, 199, 7]
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=wanted)
        assert arr.shape == (len(wanted),)
        for out_i, disk_i in enumerate(wanted):
            assert arr["a"][out_i] == src["a"][disk_i]
            assert arr["b"][out_i] == src["b"][disk_i]


def test_read_rows_negative_indices():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000, ztilelen=500)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=[-1, -10, -50])
        np.testing.assert_array_equal(
            arr["a"], np.array([src["a"][-1], src["a"][-10], src["a"][-50]])
        )


def test_read_rows_dedupes_first_occurrence_wins():
    """
    resolve_rows dedupes: the second occurrence of an index is dropped.
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=[3, 1, 3, 2])
        # 3 is deduped; result should be [3, 1, 2].
        assert arr.shape == (3,)
        np.testing.assert_array_equal(
            arr["a"], np.array([src["a"][3], src["a"][1], src["a"][2]])
        )


def test_read_rows_combined_with_columns():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500, ztilelen=100)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(rows=slice(50, 150), columns=["b"])
        assert arr.dtype.names == ("b",)
        np.testing.assert_array_equal(arr["b"], src["b"][50:150])


# ---------------------------------------------------------------------
# __getitem__ — int / slice / fancy / col-name / col-list
# ---------------------------------------------------------------------


def test_getitem_int_returns_scalar_record():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500, ztilelen=200)
        with rustfits.FITS(fz, "r") as f:
            rec = f[1][7]
        # numpy.void scalar — shape () with named fields
        assert rec["a"] == src["a"][7]
        assert rec["b"] == src["b"][7]


def test_getitem_negative_int():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500)
        with rustfits.FITS(fz, "r") as f:
            rec = f[1][-1]
        assert rec["a"] == src["a"][-1]


def test_getitem_slice():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500, ztilelen=125)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1][50:75]
        assert arr.shape == (25,)
        np.testing.assert_array_equal(arr["a"], src["a"][50:75])


def test_getitem_slice_stepped():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500, ztilelen=125)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1][0:200:5]
        np.testing.assert_array_equal(arr["a"], src["a"][0:200:5])


def test_getitem_fancy_rows_preserves_order():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=200, ztilelen=50)
        wanted = [4, 1, 9, 2]
        with rustfits.FITS(fz, "r") as f:
            arr = f[1][wanted]
        for out_i, disk_i in enumerate(wanted):
            assert arr["a"][out_i] == src["a"][disk_i]


def test_getitem_string_column_returns_subset_object():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            col = f[1]["a"]
            assert isinstance(col, rustfits.CompressedSingleColumnSubset)
            assert "TableColumn" in repr(col)


def test_getitem_list_of_strings_returns_subset_object():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            cs = f[1][["a", "b"]]
            assert isinstance(cs, rustfits.CompressedColumnSubset)
            assert "TableColumns" in repr(cs)


# ---------------------------------------------------------------------
# Column subset objects — chained reads
# ---------------------------------------------------------------------


def test_single_column_subset_chained_slice():
    """
    hdu["col"][i:j] returns the column's values as a plain
    (non-structured) ndarray of length j-i.
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=500, ztilelen=125)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1]["a"][100:200]
        assert arr.shape == (100,)
        assert arr.dtype == np.int32
        np.testing.assert_array_equal(arr, src["a"][100:200])


def test_single_column_subset_chained_fancy():
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000)
        wanted = [10, 50, 100, 5]
        with rustfits.FITS(fz, "r") as f:
            arr = f[1]["b"][wanted]
        np.testing.assert_array_equal(
            arr, np.array([src["b"][i] for i in wanted])
        )


def test_multi_column_subset_chained_slice():
    """
    hdu[["a", "b"]][i:j] returns a structured ndarray with those
    fields restricted to rows i:j.
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=300, ztilelen=100)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1][["a", "c"]][50:80]
        assert arr.dtype.names == ("a", "c")
        np.testing.assert_array_equal(arr["a"], src["a"][50:80])
        np.testing.assert_array_equal(arr["c"], src["c"][50:80])


def test_multi_column_subset_reorders():
    """
    The columns appear in the user-requested order, not file order.
    """
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1][["c", "a"]][:10]
        assert arr.dtype.names == ("c", "a")


# ---------------------------------------------------------------------
# Tile cache
# ---------------------------------------------------------------------


def test_cache_default_capacity():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            assert f[1].tile_cache_size == 32 * 1024 * 1024  # default 32 MiB
            assert f[1].tile_cache_used == 0


def test_cache_warms_on_read():
    """
    A whole-table read should populate the cache.
    """
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000, ztilelen=500)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.tile_cache_used == 0
            hdu.read()
            assert hdu.tile_cache_used > 0


def test_cache_repeat_read_serves_from_cache():
    """
    Second read with same (rows, columns) should not increase used
    bytes (everything already cached).
    """
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=1000, ztilelen=250)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            hdu.read()
            first = hdu.tile_cache_used
            hdu.read()
            assert hdu.tile_cache_used == first


def test_clear_tile_cache():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=1000)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            hdu.read()
            assert hdu.tile_cache_used > 0
            hdu.clear_tile_cache()
            assert hdu.tile_cache_used == 0
            # Capacity setting unchanged.
            assert hdu.tile_cache_size == 32 * 1024 * 1024


def test_set_tile_cache_size_zero_disables_caching():
    """
    Capacity 0 means no entries are kept — every read decompresses
    fresh.  used_bytes stays at 0.
    """
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=1000)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            hdu.set_tile_cache_size(0)
            hdu.read()
            assert hdu.tile_cache_used == 0
            # Reading still works.
            arr = hdu.read()
            assert arr.shape == (1000,)


def test_set_tile_cache_size_smaller_evicts():
    """
    Shrinking the capacity below current usage evicts LRU entries
    until used_bytes fits.
    """
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000, ztilelen=200)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            hdu.read()
            used_before = hdu.tile_cache_used
            assert used_before > 0
            tighter = used_before // 4
            hdu.set_tile_cache_size(tighter)
            assert hdu.tile_cache_used <= tighter


def test_cache_correctness_under_partial_reads():
    """
    A column-subset read followed by a full read should still give
    correct values — cache entries from the partial read don't
    corrupt subsequent decodes.
    """
    with tempfile.TemporaryDirectory() as td:
        src, fz = _make_default_fixture(td, nrows=2000, ztilelen=400)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            _ = hdu.read(columns=["a"])
            arr = hdu.read()
        np.testing.assert_array_equal(arr["a"], src["a"])
        np.testing.assert_array_equal(arr["b"], src["b"])
        np.testing.assert_array_equal(arr["c"], src["c"])


# ---------------------------------------------------------------------
# Range / type errors
# ---------------------------------------------------------------------


def test_getitem_out_of_range_int():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            with pytest.raises((IndexError, ValueError)):
                _ = f[1][5000]


def test_getitem_empty_iterable_rejected():
    with tempfile.TemporaryDirectory() as td:
        _, fz = _make_default_fixture(td, nrows=2000)
        with rustfits.FITS(fz, "r") as f:
            with pytest.raises(ValueError, match="empty"):
                _ = f[1][[]]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
