"""
read() / write() methods on column-subset pyclasses.

Covers all four subset classes:
    - SingleColumnSubset (uncompressed, hdu["name"])
    - ColumnSubset (uncompressed, hdu[[names]])
    - CompressedSingleColumnSubset (hdu["name"] on a ZTABLE HDU)
    - CompressedColumnSubset (hdu[[names]] on a ZTABLE HDU)

For each:
    - read() with default kwargs returns the same shape as
      subset[:] (plain ndarray for single-col, structured for
      multi-col).
    - read(rows=...) / read(scale=False) / read(mask_null=True)
      forward kwargs correctly.
    - write(data) wholesale-writes the subset, equivalent to
      hdu[key] = data via the __setitem__ dispatch.
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import rustfits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_uncompressed(path):
    """Write a small mixed-dtype table for the uncompressed tests."""
    dt = np.dtype([("x", "f8"), ("y", "i4"), ("flag", "i2")])
    rows = np.array(
        [(1.0, 10, 0), (2.0, 20, 1), (3.0, 30, 0), (4.0, 40, 1)],
        dtype=dt,
    )
    with rustfits.FITS(path, "w+") as f:
        f.write_table(rows, extname="T")
    return rows


def _mk_compressed(path):
    """Write a ZTABLE with enough rows that ztilelen=2 gives 2 tiles."""
    dt = np.dtype([("x", "f8"), ("y", "i4"), ("flag", "i2")])
    rows = np.array(
        [
            (1.0, 10, 0),
            (2.0, 20, 1),
            (3.0, 30, 0),
            (4.0, 40, 1),
        ],
        dtype=dt,
    )
    with rustfits.FITS(path, "w+") as f:
        f.write_table(rows, extname="T", compress=True, ztilelen=2)
    return rows


# ---------------------------------------------------------------------------
# SingleColumnSubset (uncompressed)
# ---------------------------------------------------------------------------


def test_single_col_read_returns_plain_ndarray():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            arr = col.read()
            assert arr.dtype.names is None  # plain, not structured
            npt.assert_equal(arr, rows["x"])


def test_single_col_read_matches_getitem_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            npt.assert_equal(col.read(), col[:])


def test_single_col_read_with_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            npt.assert_equal(col.read(rows=slice(1, 3)), rows["x"][1:3])
            npt.assert_equal(col.read(rows=[3, 0, 2]), rows["x"][[3, 0, 2]])


def test_single_col_write_whole_column():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        new_x = np.array([100.0, 200.0, 300.0, 400.0], dtype="f8")
        expected_y = np.array([10, 20, 30, 40], dtype="i4")
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(new_x)
            # Same-handle verify.
            npt.assert_equal(f[1]["x"].read(), new_x)
            # Other columns untouched.
            npt.assert_equal(f[1]["y"].read(), expected_y)
        # Post-reopen verify.
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), new_x)


def test_single_col_write_with_unsigned_trick():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.fits")
        dt = np.dtype([("v", "u4")])
        rows = np.array([(1,), (2,), (3,)], dtype=dt)
        with rustfits.FITS(path, "w+") as f:
            f.write_table(rows, extname="T")
        new_v = np.array([4000000000, 5000, 0], dtype="u4")
        with rustfits.FITS(path, "r+") as f:
            f[1]["v"].write(new_v)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["v"].read(), new_v)


# ---------------------------------------------------------------------------
# ColumnSubset (uncompressed)
# ---------------------------------------------------------------------------


def test_multi_col_read_returns_structured():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            arr = sub.read()
            assert arr.dtype.names == ("x", "y")
            npt.assert_equal(arr["x"], rows["x"])
            npt.assert_equal(arr["y"], rows["y"])


def test_multi_col_read_matches_getitem_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            via_read = sub.read()
            via_slice = sub[:]
            assert via_read.dtype == via_slice.dtype
            npt.assert_equal(via_read, via_slice)


def test_multi_col_read_with_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "flag"]]
            r = sub.read(rows=slice(1, 3))
            npt.assert_equal(r["x"], rows["x"][1:3])
            npt.assert_equal(r["flag"], rows["flag"][1:3])


def test_multi_col_read_scale_false():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.fits")
        # Build a u4 column so the unsigned-int trick fires;
        # scale=False should return the stored (signed) view.
        dt = np.dtype([("v", "u4")])
        rows = np.array([(0,), (1,), (4000000000,)], dtype=dt)
        with rustfits.FITS(path, "w+") as f:
            f.write_table(rows, extname="T")
        with rustfits.FITS(path) as f:
            sub = f[1][["v"]]
            scaled = sub.read()
            unscaled = sub.read(scale=False)
            assert scaled["v"].dtype == np.dtype("u4")
            assert unscaled["v"].dtype == np.dtype("i4")


def test_multi_col_write_whole_subset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        sub_dt = np.dtype([("x", "f8"), ("y", "i4")])
        new = np.array(
            [(1.5, 11), (2.5, 22), (3.5, 33), (4.5, 44)], dtype=sub_dt
        )
        expected_flag = np.array([0, 1, 0, 1], dtype="i2")
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new)
            # Other column untouched.
            npt.assert_equal(f[1]["flag"].read(), expected_flag)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), new["x"])
            npt.assert_equal(f[1]["y"].read(), new["y"])


def test_multi_col_write_extra_fields_tolerated():
    """write() forwards through MultiColumns dispatch which tolerates
    extra fields in the source."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        rich_dt = np.dtype([("x", "f8"), ("y", "i4"), ("extra", "f4")])
        new = np.array(
            [
                (1.5, 11, 0.0),
                (2.5, 22, 0.0),
                (3.5, 33, 0.0),
                (4.5, 44, 0.0),
            ],
            dtype=rich_dt,
        )
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), new["x"])


# ---------------------------------------------------------------------------
# CompressedSingleColumnSubset
# ---------------------------------------------------------------------------


def test_compressed_single_col_read_returns_plain_ndarray():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            arr = col.read()
            assert arr.dtype.names is None
            npt.assert_equal(arr, rows["x"])


def test_compressed_single_col_read_matches_getitem():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["y"]
            npt.assert_equal(col.read(), col[:])


def test_compressed_single_col_read_with_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            arr = f[1]["x"].read(rows=[3, 1])
            npt.assert_equal(arr, rows["x"][[3, 1]])


def test_compressed_single_col_read_mask_null_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path) as f:
            with pytest.raises(NotImplementedError, match="TNULL masking"):
                f[1]["x"].read(mask_null=True)


def test_compressed_single_col_write_whole_column():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        new_x = np.array([100.0, 200.0, 300.0, 400.0], dtype="f8")
        expected_y = np.array([10, 20, 30, 40], dtype="i4")
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(new_x)
            npt.assert_equal(f[1]["x"].read(), new_x)
            # Cross-check via parent HDU read (forces tile decode).
            full = f[1].read()
            npt.assert_equal(full["y"], expected_y)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), new_x)


# ---------------------------------------------------------------------------
# CompressedColumnSubset
# ---------------------------------------------------------------------------


def test_compressed_multi_col_read_returns_structured():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "flag"]]
            arr = sub.read()
            assert arr.dtype.names == ("x", "flag")
            npt.assert_equal(arr["x"], rows["x"])
            npt.assert_equal(arr["flag"], rows["flag"])


def test_compressed_multi_col_read_matches_getitem():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            npt.assert_equal(sub.read(), sub[:])


def test_compressed_multi_col_read_with_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            r = sub.read(rows=slice(1, 3))
            npt.assert_equal(r["x"], rows["x"][1:3])


def test_compressed_multi_col_read_mask_null_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path) as f:
            with pytest.raises(NotImplementedError, match="TNULL masking"):
                f[1][["x", "y"]].read(mask_null=True)


def test_compressed_multi_col_write_whole_subset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        sub_dt = np.dtype([("x", "f8"), ("y", "i4")])
        new = np.array(
            [(1.5, 11), (2.5, 22), (3.5, 33), (4.5, 44)], dtype=sub_dt
        )
        expected_flag = np.array([0, 1, 0, 1], dtype="i2")
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new)
            full = f[1].read()
            npt.assert_equal(full["flag"], expected_flag)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), new["x"])
            npt.assert_equal(f[1]["y"].read(), new["y"])


# ---------------------------------------------------------------------------
# write(rows=...) row-restricted path — all four subset classes
# ---------------------------------------------------------------------------


def test_single_col_write_with_rows_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(
                np.array([99.0, 98.0], dtype="f8"), rows=slice(1, 3)
            )
        with rustfits.FITS(path) as f:
            npt.assert_equal(
                f[1]["x"].read(), np.array([1.0, 99.0, 98.0, 4.0])
            )


def test_single_col_write_with_rows_int():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(99.0, rows=2)
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), np.array([1.0, 2.0, 99.0, 4.0]))


def test_single_col_write_with_rows_fancy():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(np.array([7.0, 8.0], dtype="f8"), rows=[3, 0])
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), np.array([8.0, 2.0, 3.0, 7.0]))


def test_multi_col_write_with_rows_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        sub_dt = np.dtype([("x", "f8"), ("y", "i4")])
        new = np.array([(99.0, 99), (88.0, 88)], dtype=sub_dt)
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new, rows=slice(1, 3))
        with rustfits.FITS(path) as f:
            npt.assert_equal(
                f[1]["x"].read(), np.array([1.0, 99.0, 88.0, 4.0])
            )
            npt.assert_equal(f[1]["y"].read(), np.array([10, 99, 88, 40]))
            # Untouched.
            npt.assert_equal(
                f[1]["flag"].read(),
                np.array([0, 1, 0, 1], dtype="i2"),
            )


def test_multi_col_write_with_rows_fancy():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        sub_dt = np.dtype([("x", "f8"), ("y", "i4")])
        new = np.array([(99.0, 99), (88.0, 88)], dtype=sub_dt)
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new, rows=[3, 0])
        with rustfits.FITS(path) as f:
            npt.assert_equal(
                f[1]["x"].read(), np.array([88.0, 2.0, 3.0, 99.0])
            )


def test_compressed_single_col_write_with_rows_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(
                np.array([99.0, 98.0], dtype="f8"), rows=slice(1, 3)
            )
        with rustfits.FITS(path) as f:
            npt.assert_equal(
                f[1]["x"].read(), np.array([1.0, 99.0, 98.0, 4.0])
            )


def test_compressed_single_col_write_with_rows_fancy():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        with rustfits.FITS(path, "r+") as f:
            f[1]["x"].write(np.array([7.0, 8.0], dtype="f8"), rows=[3, 0])
        with rustfits.FITS(path) as f:
            npt.assert_equal(f[1]["x"].read(), np.array([8.0, 2.0, 3.0, 7.0]))


def test_compressed_multi_col_write_with_rows_slice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.fits")
        _mk_compressed(path)
        sub_dt = np.dtype([("x", "f8"), ("y", "i4")])
        new = np.array([(99.0, 99), (88.0, 88)], dtype=sub_dt)
        with rustfits.FITS(path, "r+") as f:
            f[1][["x", "y"]].write(new, rows=slice(1, 3))
        with rustfits.FITS(path) as f:
            npt.assert_equal(
                f[1]["x"].read(), np.array([1.0, 99.0, 88.0, 4.0])
            )
            npt.assert_equal(f[1]["y"].read(), np.array([10, 99, 88, 40]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
