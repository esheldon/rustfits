"""
ZTABLE Phase 5 — table-side compressed create + write.

create_table_hdu(dtype, nrows, compress=...) routes to the ZTABLE
write path when compress is non-None.  Accepted compress shapes:
  - True                     : cfitsio defaults per column
  - "GZIP_2" / Gzip2(...)    : same algorithm everywhere (validated)
  - {"col": algo, ...}       : per-column overrides + defaults
hdu.write(data) accepts the same three input forms as the
uncompressed TableHDU.write (structured ndarray / dict / list+names).

Round-trip is verified against the same data fed to write() — we
write, reopen, read, compare element-wise.  Cross-tool checks use
cfitsio's funpack to decompress and re-read with fitsio.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _have_funpack():
    return shutil.which("funpack") is not None


def _basic_data(nrows=2000):
    dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "f4")])
    arr = np.zeros(nrows, dtype=dt)
    arr["a"] = np.arange(nrows, dtype="i4")
    arr["b"] = np.arange(nrows, dtype="f8") * 0.25
    arr["c"] = np.arange(nrows, dtype="f4") * -1.0
    return arr


def _write_and_reread(fname, data, dtype, *, compress=True, ztilelen=None):
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(
            dtype,
            nrows=len(data),
            compress=compress,
            ztilelen=ztilelen,
        )
        f[1].write(data)
    with rustfits.FITS(fname, "r") as f:
        assert isinstance(f[1], rustfits.CompressedTableHDU)
        return f[1].read()


# ---------------------------------------------------------------------
# compress= argument shapes
# ---------------------------------------------------------------------


def test_compress_true_uses_per_dtype_defaults():
    """
    compress=True picks per-column defaults: i4 → RICE_1, f8 → GZIP_2,
    f4 → GZIP_2 (cfitsio's fits_compress_table per-letter defaults).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        out = _write_and_reread(fname, src, src.dtype, compress=True)
        with rustfits.FITS(fname, "r") as f:
            assert dict(f[1].compression) == {
                "a": "RICE_1",
                "b": "GZIP_2",
                "c": "GZIP_2",
            }
        for col in src.dtype.names:
            np.testing.assert_array_equal(out[col], src[col])


def test_compress_false_routes_to_uncompressed():
    """compress=False (or None / omitted) stays on the uncompressed path."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(src.dtype, nrows=len(src), compress=False)
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            assert isinstance(f[1], rustfits.TableHDU)
            assert not isinstance(f[1], rustfits.CompressedTableHDU)


def test_compress_string_alias_applies_everywhere():
    """compress="GZIP_1" picks GZIP_1 for every column."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        out = _write_and_reread(fname, src, src.dtype, compress="GZIP_1")
        with rustfits.FITS(fname, "r") as f:
            algos = dict(f[1].compression)
            assert all(v == "GZIP_1" for v in algos.values())
        for col in src.dtype.names:
            np.testing.assert_array_equal(out[col], src[col])


def test_compress_class_applies_everywhere():
    """compress=Gzip2() picks GZIP_2 for every column (where allowed)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        out = _write_and_reread(
            fname,
            src,
            src.dtype,
            compress=rustfits.Gzip2(),
        )
        with rustfits.FITS(fname, "r") as f:
            algos = dict(f[1].compression)
            assert all(v == "GZIP_2" for v in algos.values())
        for col in src.dtype.names:
            np.testing.assert_array_equal(out[col], src[col])


def test_compress_dict_per_column_override():
    """
    Per-column dict: named columns get the override, unspecified
    columns get cfitsio defaults.  Mix string and class values.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        out = _write_and_reread(
            fname,
            src,
            src.dtype,
            compress={"a": "GZIP_2", "c": rustfits.Gzip1()},
        )
        # 'a' overridden to GZIP_2 (default would be RICE_1).
        # 'b' uses default → GZIP_2.
        # 'c' overridden to GZIP_1 (default would be GZIP_2).
        with rustfits.FITS(fname, "r") as f:
            assert dict(f[1].compression) == {
                "a": "GZIP_2",
                "b": "GZIP_2",
                "c": "GZIP_1",
            }
        for col in src.dtype.names:
            np.testing.assert_array_equal(out[col], src[col])


def test_compress_dict_unknown_column_rejected():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="not match any column"):
                f.create_table_hdu(
                    src.dtype,
                    nrows=len(src),
                    compress={"nope": "GZIP_1"},
                )


def test_compress_invalid_algorithm_for_dtype_rejected():
    """
    Asking for RICE_1 on an f8 column raises with the allowed list.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="Allowed for this dtype"):
                f.create_table_hdu(
                    src.dtype,
                    nrows=len(src),
                    compress={"b": "RICE_1"},
                )


def test_compress_image_only_algorithm_rejected():
    """HCOMPRESS_1 and PLIO_1 are image-only — reject on a table."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="image-only"):
                f.create_table_hdu(
                    src.dtype,
                    nrows=len(src),
                    compress="HCOMPRESS_1",
                )


def test_ztilelen_requires_compress():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="ztilelen= requires"):
                f.create_table_hdu(
                    src.dtype,
                    nrows=len(src),
                    ztilelen=500,
                )


def test_compress_with_var_dtypes_rejected_phase6():
    """compress=True + var_dtypes={...} is Phase 6 territory."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("a", "i4"), ("v", "O")])
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(NotImplementedError, match="Phase 6"):
                f.create_table_hdu(
                    dt,
                    nrows=10,
                    compress=True,
                    var_dtypes={"v": "f4"},
                )


# ---------------------------------------------------------------------
# Input-form flexibility (mirrors uncompressed TableHDU.write)
# ---------------------------------------------------------------------


def test_write_accepts_structured_ndarray():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data()
        out = _write_and_reread(fname, src, src.dtype)
        for col in src.dtype.names:
            np.testing.assert_array_equal(out[col], src[col])


def test_write_accepts_dict_input():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("a", "i4"), ("b", "f8")])
        data = {
            "a": np.arange(nrows, dtype="i4"),
            "b": np.arange(nrows, dtype="f8") * 0.5,
        }
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, compress=True)
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["a"], data["a"])
        np.testing.assert_array_equal(out["b"], data["b"])


def test_write_accepts_list_plus_names():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("a", "i4"), ("b", "f8")])
        a = np.arange(nrows, dtype="i4")
        b = np.arange(nrows, dtype="f8") * 0.5
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, compress=True)
            f[1].write([a, b], names=["a", "b"])
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["a"], a)
        np.testing.assert_array_equal(out["b"], b)


# ---------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "col_dtype",
    ["u1", "i2", "i4", "i8", "f4", "f8", "?"],
)
def test_round_trip_per_dtype(col_dtype):
    """
    Each scalar dtype round-trips through compress=True (defaults).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("x", col_dtype)])
        src = np.zeros(nrows, dtype=dt)
        if col_dtype == "?":
            src["x"] = np.arange(nrows) % 3 == 0
        elif col_dtype.startswith("u"):
            src["x"] = np.arange(nrows, dtype=col_dtype) % 200
        else:
            src["x"] = np.arange(nrows, dtype=col_dtype)
        out = _write_and_reread(fname, src, dt)
        np.testing.assert_array_equal(out["x"], src["x"])


@pytest.mark.parametrize("col_dtype", ["u2", "u4"])
def test_unsigned_int_trick_round_trip(col_dtype):
    """
    Unsigned-int trick: u2/u4 in, on-disk i2/i4 + TZERO, back to u2/u4
    on read.  The per-cell UnsignedXor transform handles the
    bit-flip both ways.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("u", col_dtype)])
        max_val = 60000 if col_dtype == "u2" else 4_000_000_000
        src = np.zeros(nrows, dtype=dt)
        src["u"] = (np.arange(nrows, dtype=col_dtype) * 7) % max_val
        out = _write_and_reread(fname, src, dt)
        assert out["u"].dtype == np.dtype(col_dtype)
        np.testing.assert_array_equal(out["u"], src["u"])


def test_subarray_column_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("box", ("f4", (2, 3)))])
        src = np.empty(nrows, dtype=dt)
        src["box"] = np.arange(nrows * 6, dtype="f4").reshape(nrows, 2, 3)
        out = _write_and_reread(fname, src, dt)
        assert out["box"].shape == (nrows, 2, 3)
        np.testing.assert_array_equal(out["box"], src["box"])


def test_string_column_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        nrows = 2000
        dt = np.dtype([("name", "S8")])
        src = np.empty(nrows, dtype=dt)
        src["name"] = [f"r{i:05d}".encode() for i in range(nrows)]
        out = _write_and_reread(fname, src, dt)
        # Read comes back as numpy U; compare as bytes.
        np.testing.assert_array_equal(out["name"].astype("S8"), src["name"])


# ---------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------


def test_default_ztilelen_matches_cfitsio_10mb_rule():
    """
    Default ztilelen targets ~10 MB worth of rows (cfitsio's
    maxchunksize / row_width).  For a 16-byte-row table with 2000
    rows that's max(1, min(2000, 10_000_000 / 16)) = 2000.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data(nrows=2000)
        _write_and_reread(fname, src, src.dtype, compress=True)
        with rustfits.FITS(fname, "r") as f:
            # All rows in one tile for a 32 KB table.
            assert f[1].ztile_rows == 2000
            assert f[1].n_tiles == 1


def test_explicit_ztilelen_creates_multiple_tiles():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data(nrows=2000)
        _write_and_reread(
            fname,
            src,
            src.dtype,
            compress=True,
            ztilelen=500,
        )
        with rustfits.FITS(fname, "r") as f:
            assert f[1].ztile_rows == 500
            assert f[1].n_tiles == 4


# ---------------------------------------------------------------------
# Cross-tool round trip via funpack
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_our_file():
    """
    cfitsio's funpack reads a file we wrote and reconstructs the
    original BINTABLE bit-exactly.
    """
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        src = _basic_data(nrows=3000)
        _write_and_reread(fname, src, src.dtype, compress=True)
        out = os.path.join(td, "out.fits")
        subprocess.run(
            ["funpack", "-O", out, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out, "r") as f:
            decompressed = f[1].read()
        for col in src.dtype.names:
            np.testing.assert_array_equal(decompressed[col], src[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
