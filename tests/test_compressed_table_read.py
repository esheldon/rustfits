"""
Tests for ZTABLE whole-table read across GZIP_1 / GZIP_2 /
RICE_1.

Read-side correctness is verified against the UNCOMPRESSED source
file that fpack consumed.  Neither astropy nor fitsio currently
auto-decompresses ZTABLE files (astropy 7.2.0 shows raw 1QB
descriptors; fitsio likewise), so cross-tool verification means
comparing rustfits's decompressed output against the original
pre-fpack table.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------


def _have_fpack():
    return shutil.which("fpack") is not None


pytestmark = pytest.mark.skipif(
    not _have_fpack(),
    reason="fpack (cfitsio CLI) is required to build ZTABLE fixtures",
)


def _make_ztable_fixture(
    td,
    data,
    units=None,
    primary_data=None,
    *,
    ztilelen=None,
):
    """
    Write `data` to a BINTABLE via fitsio, then run `fpack -table`.
    Returns (uncompressed_path, compressed_path).  `ztilelen` (if
    given) sets fpack's per-tile row count via the FZTILELN header
    keyword on the source HDU — fpack has no CLI flag for it.
    """
    import fitsio

    src = os.path.join(td, "src.fits")
    with fitsio.FITS(src, "rw", clobber=True) as f:
        if primary_data is not None:
            f.write(primary_data)
        f.write(data, units=units)
        if ztilelen is not None:
            f[-1].write_key("FZTILELN", ztilelen)
    subprocess.run(
        ["fpack", "-table", src],
        check=True,
        capture_output=True,
    )
    fz = src + ".fz"
    assert os.path.exists(fz), f"fpack did not produce {fz}"
    return src, fz


# ---------------------------------------------------------------------
# Mixed-dtype round trip — exercises every Phase 2 algorithm at once
# ---------------------------------------------------------------------


def test_round_trip_mixed_dtypes():
    """
    The default fpack table fixture exercises all three algorithms
    in one go: integer columns go through RICE_1, float through
    GZIP_2, strings through GZIP_1.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype(
            [
                ("a", "i4"),
                ("b", "f8"),
                ("c", ("f4", (2, 3))),
                ("s", "S10"),
            ]
        )
        nrows = 10000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        src["b"] = np.arange(nrows, dtype="f8") * 0.25
        src["c"] = np.arange(nrows * 6, dtype="f4").reshape(nrows, 2, 3)
        src["s"] = [f"row{i:05d}".encode() for i in range(nrows)]

        _, fz = _make_ztable_fixture(
            td, src, units=["count", "Jy", "m", "char"]
        )
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read()
        assert arr.shape == (nrows,)
        np.testing.assert_array_equal(arr["a"], src["a"])
        np.testing.assert_array_equal(arr["b"], src["b"])
        np.testing.assert_array_equal(arr["c"], src["c"])
        # Strings come back as numpy U; compare as bytes via S10.
        np.testing.assert_array_equal(arr["s"].astype("S10"), src["s"])


# ---------------------------------------------------------------------
# Per-algorithm coverage
# ---------------------------------------------------------------------


# cfitsio's fits_compress_table picks per-dtype defaults:
#   B (u1)    -> GZIP_1
#   I (i2)    -> GZIP_2
#   J (i4)    -> RICE_1
# (See imcompress.c around line 8261.)
@pytest.mark.parametrize(
    "col_dtype,expected_algo",
    [("u1", "GZIP_1"), ("i2", "GZIP_2"), ("i4", "RICE_1")],
)
def test_integer_columns_default_algorithm_round_trip(
    col_dtype,
    expected_algo,
):
    """
    Round-trip integer columns through each algorithm cfitsio's
    table compressor picks by default for that dtype.  Covers the
    RICE_1 read path for J (the only letter cfitsio uses RICE for),
    plus GZIP_1 / GZIP_2 on integer columns.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("x", col_dtype)])
        nrows = 10000
        src = np.zeros(nrows, dtype=dt)
        if col_dtype.startswith("u"):
            src["x"] = np.arange(nrows, dtype=col_dtype) % 200
        else:
            src["x"] = np.arange(nrows, dtype=col_dtype)
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.compression["x"] == expected_algo
            arr = hdu.read()
        np.testing.assert_array_equal(arr["x"], src["x"])


@pytest.mark.parametrize("col_dtype", ["f4", "f8"])
def test_gzip_2_float_columns(col_dtype):
    """
    Floats default to GZIP_2 (byte-shuffle + gzip).
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("x", col_dtype)])
        nrows = 10000
        src = np.zeros(nrows, dtype=dt)
        src["x"] = np.arange(nrows, dtype=col_dtype) * 0.5
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.compression["x"] == "GZIP_2"
            arr = hdu.read()
        np.testing.assert_array_equal(arr["x"], src["x"])


def test_gzip_1_string_column():
    """
    String columns (A) default to GZIP_1.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("s", "S6")])
        nrows = 10000
        src = np.zeros(nrows, dtype=dt)
        src["s"] = [f"r{i:04d}".encode() for i in range(nrows)]
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.compression["s"] == "GZIP_1"
            arr = hdu.read()
        np.testing.assert_array_equal(arr["s"].astype("S6"), src["s"])


# ---------------------------------------------------------------------
# Multi-tile coverage — exercise the per-tile descriptor walk
# ---------------------------------------------------------------------


def test_multi_tile_read():
    """
    Force fpack to use a small ZTILELEN so the table is split into
    several tiles.  Each tile is decompressed independently;
    correctness should be unaffected.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4"), ("b", "f8")])
        nrows = 5000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        src["b"] = np.arange(nrows, dtype="f8") * 0.1
        # 750 rows per tile -> 7 tiles for 5000 rows.
        _, fz = _make_ztable_fixture(td, src, ztilelen=750)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles == 7
            assert hdu.ztile_rows == 750
            arr = hdu.read()
        np.testing.assert_array_equal(arr["a"], src["a"])
        np.testing.assert_array_equal(arr["b"], src["b"])


def test_last_tile_partial():
    """
    Row count not divisible by ZTILELEN: the last tile is shorter.
    Make sure the row-count computation per tile gets that right.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4")])
        nrows = 1000 + 17  # last tile has 17 rows when ztilelen=1000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        _, fz = _make_ztable_fixture(td, src, ztilelen=1000)
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles == 2
            arr = hdu.read()
        np.testing.assert_array_equal(arr["a"], src["a"])


# ---------------------------------------------------------------------
# Scaling (TZERO unsigned-int trick)
# ---------------------------------------------------------------------


def test_unsigned_int_trick_round_trip():
    """
    Unsigned u2 round-trip through fitsio's unsigned-trick TZERO
    plus ZTABLE compression should land back as u2 with values
    preserved.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("u", "u2")])
        nrows = 10000
        src = np.zeros(nrows, dtype=dt)
        src["u"] = (np.arange(nrows, dtype="u2") * 7) % 65530
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read()
        assert arr["u"].dtype == np.uint16
        np.testing.assert_array_equal(arr["u"], src["u"])


def test_scale_false_returns_stored_dtype():
    """
    scale=False on a u2 (TZERO=32768) column returns the raw signed
    i2 stored values rather than promoting to u2.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("u", "u2")])
        nrows = 5000
        src = np.zeros(nrows, dtype=dt)
        src["u"] = (np.arange(nrows, dtype="u2") * 7) % 65530
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(scale=False)
        assert arr["u"].dtype == np.int16


# ---------------------------------------------------------------------
# Column subset
# ---------------------------------------------------------------------


def test_columns_subset_decompresses_only_selected():
    """
    columns=['b'] should decompress only column b, skipping a's
    heap blob entirely.  Output dtype contains only the requested
    field; values match the source.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "i2")])
        nrows = 4000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        src["b"] = np.arange(nrows, dtype="f8") * 0.25
        src["c"] = -np.arange(nrows, dtype="i2")
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(columns=["b"])
        assert arr.dtype.names == ("b",)
        np.testing.assert_array_equal(arr["b"], src["b"])


def test_columns_subset_reorders():
    """
    columns=['c', 'a'] returns the columns in the user-requested
    order, regardless of file order.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "i2")])
        nrows = 4000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        src["b"] = np.arange(nrows, dtype="f8") * 0.25
        src["c"] = -np.arange(nrows, dtype="i2")
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(columns=["c", "a"])
        assert arr.dtype.names == ("c", "a")
        np.testing.assert_array_equal(arr["c"], src["c"])
        np.testing.assert_array_equal(arr["a"], src["a"])


def test_columns_subset_case_insensitive():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("Flux", "f4"), ("ID", "i4")])
        nrows = 2000
        src = np.zeros(nrows, dtype=dt)
        src["Flux"] = np.arange(nrows, dtype="f4") * 0.1
        src["ID"] = np.arange(nrows, dtype="i4")
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read(columns=["flux", "id"])
        # Names come back case-preserved from disk, even though the
        # lookup itself is case-insensitive.
        assert arr.dtype.names == ("Flux", "ID")


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_subarray_column_round_trip():
    """
    Multi-D per-cell shape (TDIM-bearing column) should survive
    compression + decompression with the same shape.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("box", ("f4", (3, 4)))])
        nrows = 4000
        src = np.zeros(nrows, dtype=dt)
        src["box"] = np.arange(nrows * 12, dtype="f4").reshape(nrows, 3, 4)
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            arr = f[1].read()
        assert arr["box"].shape == (nrows, 3, 4)
        np.testing.assert_array_equal(arr["box"], src["box"])


def test_compressed_table_after_other_hdus():
    """
    A ZTABLE that is not at index 1 should still be detected + read.
    fpack -table compresses BOTH the primary image and the table
    (when the primary has data), so the table ends up at index 2
    here.  Find the CompressedTableHDU dynamically.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4")])
        nrows = 3000
        src = np.zeros(nrows, dtype=dt)
        src["a"] = np.arange(nrows, dtype="i4")
        primary = np.arange(100, dtype="f4")
        _, fz = _make_ztable_fixture(td, src, primary_data=primary)
        with rustfits.FITS(fz, "r") as f:
            table_idx = next(
                i
                for i, hdu in enumerate(f)
                if isinstance(hdu, rustfits.CompressedTableHDU)
            )
            assert table_idx >= 2
            arr = f[table_idx].read()
        np.testing.assert_array_equal(arr["a"], src["a"])


# ---------------------------------------------------------------------
# Phase boundaries — mask_null= and VLA still rejected
# (rows= subset moved into Phase 3 — see test_compressed_table_read_slice.py)
# ---------------------------------------------------------------------


def test_mask_null_still_rejected():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("a", "i4")])
        src = np.zeros(2000, dtype=dt)
        src["a"] = np.arange(2000, dtype="i4")
        _, fz = _make_ztable_fixture(td, src)
        with rustfits.FITS(fz, "r") as f:
            with pytest.raises(NotImplementedError):
                f[1].read(mask_null=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
