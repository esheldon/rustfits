"""
ZTABLE Phase 6a — VLA-column write (dual-descriptor heap).

Extends Phase 5 to handle variable-length columns.  Per VLA column
per tile, the writer produces:
  - A GZIP_1-compressed "dual-descriptor blob" containing the
    original (P/Q) descriptors plus the compressed-side (Q)
    descriptors that point at the per-cell compressed bytes.
  - One compressed blob per row holding the cell's bytes (with
    cfitsio's uncompressed-fallback when compression doesn't help).

Round-trip is verified two ways:
  - rustfits write → rustfits read (Phase 4 read path).
  - rustfits write → funpack decompress → fitsio read; bytes
    match the original up to each cell's true nelements (fitsio
    reads VLAs as max-width zero-padded fixed arrays, hence the
    slice).
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


# ---------------------------------------------------------------------
# Whole-table VLA write round trip
# ---------------------------------------------------------------------


def test_vla_round_trip_f4_inner():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["v"][i] = np.arange(i % 8, dtype="f4") * 0.5
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            assert isinstance(f[1], rustfits.CompressedTableHDU)
            out = f[1].read()
        np.testing.assert_array_equal(out["id"], src["id"])
        assert _vla_arr_equal(out["v"], src["v"])


@pytest.mark.parametrize(
    "inner_dtype",
    ["u1", "i2", "i4", "i8", "f4", "f8"],
)
def test_vla_round_trip_per_inner_dtype(inner_dtype):
    """
    Cover every fixed-width inner dtype across the per-column
    defaults (B → GZIP_1, I → GZIP_2, J → RICE_1, K/E/D → GZIP_2).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["v"][i] = np.arange(i % 6, dtype=inner_dtype)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": inner_dtype},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["v"], src["v"])


def test_vla_empty_cells_supported():
    """
    Some cells with zero elements: descriptors carry nelements=0
    and the cell is omitted from the per-row heap stream.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 800
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            if i % 5 == 0:
                src["v"][i] = np.empty(0, dtype="f4")
            else:
                src["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["v"], src["v"])
        for i in range(0, nrows, 5):
            assert len(out["v"][i]) == 0


def test_vla_multiple_columns():
    """
    Two VLA columns in the same table — each gets its own
    dual-descriptor blob per tile + its own per-cell compressed
    bytes.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("a", "O"), ("b", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["a"][i] = np.arange(i % 5 + 1, dtype="f4")
            src["b"][i] = np.arange(i % 3 + 1, dtype="i8") * 100
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"a": "f4", "b": "i8"},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["a"], src["a"])
        assert _vla_arr_equal(out["b"], src["b"])


def test_vla_mixed_with_fixed_columns():
    """Fixed + VLA columns side by side."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),
                ("flux", "f8"),
                ("v", "O"),
                ("mask", "i2"),
            ]
        )
        nrows = 800
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        src["flux"] = np.arange(nrows, dtype="f8") * 0.5
        src["mask"] = -np.arange(nrows, dtype="i2")
        for i in range(nrows):
            src["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["id"], src["id"])
        np.testing.assert_array_equal(out["flux"], src["flux"])
        np.testing.assert_array_equal(out["mask"], src["mask"])
        assert _vla_arr_equal(out["v"], src["v"])


def test_vla_multi_tile_write():
    """Force multi-tile via small ztilelen so each tile gets its own blob."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1500
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["v"][i] = np.arange(i % 6 + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4"},
                compress=True,
                ztilelen=400,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles == 4  # ceil(1500 / 400)
            out = hdu.read()
        assert _vla_arr_equal(out["v"], src["v"])


def test_vla_string_column():
    """PA (variable-length ASCII string) column."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("name", "O")])
        nrows = 500
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["name"][i] = f"row-{i:04d}"
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"name": "S"},
                compress=True,
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        for i in range(nrows):
            assert out["name"][i] == src["name"][i]


def test_vla_per_column_compression_override():
    """
    Pass a per-column dict to override the default algorithm for
    VLA columns.  RICE_1 is allowed on a VLA-of-i4.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 500
        src = np.empty(nrows, dtype=dt)
        for i in range(nrows):
            src["v"][i] = np.arange(i % 5, dtype="i4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "i4"},
                compress={"v": "GZIP_1"},
            )
            f[1].write(src)
        with rustfits.FITS(fname, "r") as f:
            assert dict(f[1].compression) == {"v": "GZIP_1"}
            out = f[1].read()
        assert _vla_arr_equal(out["v"], src["v"])


# ---------------------------------------------------------------------
# Cross-tool round trip via funpack
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack (cfitsio CLI) required for cross-tool verification",
)
def test_funpack_decompresses_vla_file():
    """
    cfitsio's funpack reads a VLA-bearing file we wrote and
    reconstructs the original BINTABLE.  fitsio reads VLA columns
    as max-width zero-padded fixed arrays, so we compare each cell
    to the first `len(src_cell)` elements of fitsio's output cell.
    """
    import fitsio
    import warnings

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O"), ("m", "O")])
        nrows = 600
        src = np.empty(nrows, dtype=dt)
        src["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            src["v"][i] = np.arange(i % 6, dtype="f4") * 0.5
            src["m"][i] = np.arange(i % 4 + 1, dtype="i2") * 100
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"v": "f4", "m": "i2"},
                compress=True,
            )
            f[1].write(src)

        out = os.path.join(td, "out.fits")
        subprocess.run(
            ["funpack", "-O", out, fname],
            check=True,
            capture_output=True,
        )
        with warnings.catch_warnings():
            # fitsio warns about missing TFORM maxlen on the
            # reconstructed VLA columns — informational only.
            warnings.simplefilter("ignore")
            with fitsio.FITS(out, "r") as f:
                cfit = f[1].read()

        np.testing.assert_array_equal(cfit["id"], src["id"])
        for col in ("v", "m"):
            for i in range(nrows):
                n = len(src[col][i])
                np.testing.assert_array_equal(
                    np.asarray(cfit[col][i])[:n],
                    src[col][i],
                    err_msg=f"col '{col}' row {i}",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
