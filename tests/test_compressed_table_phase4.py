"""
ZTABLE Phase 4 — VLA (variable-length array) column read.

ZTABLE encodes each VLA column per tile as a single GZIP_1-compressed
"dual-descriptor" blob: the original P/Q descriptors followed by the
compressed-side Q descriptors that point at per-cell compressed VLA
bytes elsewhere in the heap.  Per-cell decompression uses the
algorithm named in ZCTYPn — RICE_1 only for B/I/J inner types, GZIP_1
/ GZIP_2 for everything else.

Verification compares rustfits's decompressed output against the
uncompressed source file (also read via rustfits, since its VLA-read
path is independently tested and well-covered).  fitsio does not
auto-decompress ZTABLEs — it shows the raw 1QB-shaped heap bytes.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits


pytestmark = pytest.mark.skipif(
    shutil.which("fpack") is None,
    reason="fpack (cfitsio CLI) is required to build ZTABLE fixtures",
)


def _fpack(td, data, *, ztilelen=None):
    """Write `data` to BINTABLE via fitsio, fpack -table, return (src, fz)."""
    import fitsio

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
            "fixture is too small to trigger fpack compression"
        )
    return src, fz


def _vla_arr_equal(a, b):
    """Element-wise comparison for two Object-dtype VLA columns."""
    if len(a) != len(b):
        return False
    return all(np.array_equal(a[i], b[i]) for i in range(len(a)))


# ---------------------------------------------------------------------
# Whole-table VLA round trip
# ---------------------------------------------------------------------


def test_vla_round_trip_f4_inner():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 8, dtype="f4") * 0.5
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()

        np.testing.assert_array_equal(out["id"], ref["id"])
        assert _vla_arr_equal(out["v"], ref["v"])


@pytest.mark.parametrize(
    "inner_dtype",
    ["u1", "i2", "i4", "i8", "f4", "f8"],
)
def test_vla_round_trip_per_inner_dtype(inner_dtype):
    """
    Cover every fixed-width inner dtype across all algorithms fpack
    picks by default (RICE_1 for B/I/J, GZIP_2 for K/E/D, GZIP_1 for
    A handled separately).
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 6, dtype=inner_dtype)
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["v"], ref["v"])


def test_vla_empty_cells_supported():
    """
    Some cells with zero elements: the compressed blob still
    encodes a descriptor with nelements=0, and our reader returns
    an empty ndarray for them.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 800
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            if i % 5 == 0:
                arr["v"][i] = np.empty(0, dtype="f4")
            else:
                arr["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["v"], ref["v"])
        # Make sure empty cells round-trip as empty ndarrays
        for i in range(0, nrows, 5):
            assert len(out["v"][i]) == 0


def test_vla_multiple_columns():
    """
    Two VLA columns in the same table — each gets its own
    dual-descriptor blob per tile.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("a", "O"), ("b", "O")])
        nrows = 800
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["a"][i] = np.arange(i % 5 + 1, dtype="f4")
            arr["b"][i] = np.arange(i % 3 + 1, dtype="i8") * 100
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["a"], ref["a"])
        assert _vla_arr_equal(out["b"], ref["b"])


def test_vla_mixed_with_fixed_columns():
    """
    Fixed + VLA columns side by side.  Both code paths run in the
    same per-tile loop.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype(
            [
                ("id", "i4"),
                ("flux", "f8"),
                ("v", "O"),
                ("mask", "i2"),
            ]
        )
        nrows = 800
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        arr["flux"] = np.arange(nrows, dtype="f8") * 0.5
        arr["mask"] = -np.arange(nrows, dtype="i2")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["id"], ref["id"])
        np.testing.assert_array_equal(out["flux"], ref["flux"])
        np.testing.assert_array_equal(out["mask"], ref["mask"])
        assert _vla_arr_equal(out["v"], ref["v"])


# ---------------------------------------------------------------------
# Multi-tile + slicing on VLA tables
# ---------------------------------------------------------------------


def test_vla_multi_tile_read():
    """
    Force a small ZTILELEN so the VLA descriptors get split into
    several tiles' worth of dual-descriptor blobs.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1200
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 6 + 1, dtype="f4")
        src, fz = _fpack(td, arr, ztilelen=200)

        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles == 6
        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read()
        assert _vla_arr_equal(out["v"], ref["v"])


def test_vla_slice_across_tiles():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1200
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 6 + 1, dtype="f4")
        src, fz = _fpack(td, arr, ztilelen=200)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read(rows=slice(150, 550))
        np.testing.assert_array_equal(out["id"], ref["id"][150:550])
        assert _vla_arr_equal(out["v"], ref["v"][150:550])


def test_vla_fancy_rows_preserves_order():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 800
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 5 + 1, dtype="i4")
        src, fz = _fpack(td, arr, ztilelen=100)

        wanted = [400, 1, 250, 700, 50]
        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read(rows=wanted)
        for out_i, disk_i in enumerate(wanted):
            assert out["id"][out_i] == ref["id"][disk_i]
            np.testing.assert_array_equal(out["v"][out_i], ref["v"][disk_i])


def test_vla_single_row_via_getitem():
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 800
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 5 + 1, dtype="f4")
        src, fz = _fpack(td, arr, ztilelen=100)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            rec = f[1][333]
        assert rec["id"] == ref["id"][333]
        np.testing.assert_array_equal(rec["v"], ref["v"][333])


# ---------------------------------------------------------------------
# Column subset + VLA
# ---------------------------------------------------------------------


def test_vla_columns_subset():
    """
    columns=['v'] returns a structured array with one Object field
    of VLA cells.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            out = f[1].read(columns=["v"])
        assert out.dtype.names == ("v",)
        assert _vla_arr_equal(out["v"], ref["v"])


def test_vla_single_column_subset_chained():
    """
    hdu['v'][rows] returns a plain Object ndarray of just that
    column's cells.
    """
    with tempfile.TemporaryDirectory() as td:
        dt = np.dtype([("id", "i4"), ("v", "O")])
        nrows = 1000
        arr = np.empty(nrows, dtype=dt)
        arr["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            arr["v"][i] = np.arange(i % 4 + 1, dtype="f4")
        src, fz = _fpack(td, arr)

        with rustfits.FITS(src, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fz, "r") as f:
            col = f[1]["v"][:50]
        assert col.dtype == object
        assert len(col) == 50
        for i in range(50):
            np.testing.assert_array_equal(col[i], ref["v"][i])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
