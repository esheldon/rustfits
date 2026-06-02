"""
Scalar-int row indexing on column-subset pyclasses.

Before this was fixed, ``hdu["flam"][350]`` raised "rows= must be a
slice or an iterable of integers" because the subset's __getitem__
dispatched straight to the row-list resolver.  The fix centralizes
the wrap-int-as-1-element-list + strip-leading-axis dance in
``read_rows_maybe_scalar`` (src/hdu_table/read.rs), shared across
all 6 subset classes:

  - SingleColumnSubset (uncompressed)
  - ColumnSubset (uncompressed)
  - CompressedSingleColumnSubset
  - CompressedColumnSubset
  - AsciiSingleColumnSubset
  - AsciiColumnSubset

Contract:
  * ``subset[i]`` returns a SCALAR (0-d shape) for single-column
    subsets and a 0-d np.void record for multi-column subsets,
    paralleling ``arr[i]`` numpy semantics and ``hdu[i]`` (which
    already returned np.void).
  * Negative indices work.
  * ``subset[i:i+1]`` still returns a shape-(1,) ndarray — the
    scalar form is distinct from the 1-element-slice form, same as
    numpy.
  * Out-of-range int raises IndexError.
  * Bool is rejected (parallel to TableHDU.__getitem__).
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import rustfits


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _mk_uncompressed(path):
    dt = np.dtype([("x", "f8"), ("y", "i4"), ("name", "S6")])
    rows = np.array(
        [(1.0, 10, b"a"), (2.0, 20, b"b"), (3.0, 30, b"c"), (4.0, 40, b"d")],
        dtype=dt,
    )
    with rustfits.FITS(path, "w+") as f:
        f.write_table(rows, extname="T")
    return rows


def _mk_compressed(path):
    dt = np.dtype([("x", "f8"), ("y", "i4")])
    rows = np.array(
        [(1.0, 10), (2.0, 20), (3.0, 30), (4.0, 40), (5.0, 50)],
        dtype=dt,
    )
    with rustfits.FITS(path, "w+") as f:
        f.write_table(rows, extname="T", compress=True, ztilelen=2)
    return rows


def _mk_ascii(path):
    # ASCII table -- I5 + E15.7 columns
    dt = np.dtype([("id", "i8"), ("flux", "f4")])
    rows = np.array(
        [(10, 1.5), (20, 2.5), (30, 3.5), (40, 4.5)],
        dtype=dt,
    )
    with rustfits.FITS(path, "w+") as f:
        f.create_ascii_table_hdu(dt, nrows=len(rows), extname="T")
        f[1].write(rows)
    return rows


# ---------------------------------------------------------------------------
# SingleColumnSubset (uncompressed)
# ---------------------------------------------------------------------------


def test_single_col_scalar_index_positive():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            for i in range(len(rows)):
                v = col[i]
                # Scalar: 0-d shape, no axis to slice.
                assert np.ndim(v) == 0
                assert float(v) == float(rows["x"][i])


def test_single_col_scalar_index_negative():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            assert float(col[-1]) == float(rows["x"][-1])
            assert float(col[-len(rows)]) == float(rows["x"][0])


def test_single_col_scalar_vs_slice_shape():
    """Bare int returns scalar; 1-element slice returns shape-(1,)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            scalar = col[2]
            one_slice = col[2:3]
            assert np.ndim(scalar) == 0
            assert one_slice.shape == (1,)
            assert float(scalar) == float(one_slice[0])


def test_single_col_scalar_matches_parent_row():
    """col[i] == hdu[i][name] (same value via the two paths)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            hdu = f[1]
            for i in range(len(rows)):
                via_col = hdu["x"][i]
                via_row = hdu[i]["x"]
                assert float(via_col) == float(via_row)


def test_single_col_string_scalar():
    """A column appears with string dtype too — scalar must work."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["name"]
            # decoded as Python str
            assert col[1] == rows["name"][1].decode()


def test_single_col_out_of_range_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            with pytest.raises((IndexError, ValueError)):
                _ = col[len(rows)]
            with pytest.raises((IndexError, ValueError)):
                _ = col[-len(rows) - 1]


def test_single_col_bool_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            with pytest.raises(ValueError, match="bool"):
                _ = col[True]


# ---------------------------------------------------------------------------
# ColumnSubset (uncompressed)
# ---------------------------------------------------------------------------


def test_multi_col_scalar_returns_void_record():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            rec = sub[2]
            # 0-d structured record (np.void with named fields)
            assert isinstance(rec, np.void)
            assert rec.dtype.names == ("x", "y")
            assert float(rec["x"]) == float(rows["x"][2])
            assert int(rec["y"]) == int(rows["y"][2])


def test_multi_col_scalar_negative():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            assert float(sub[-1]["x"]) == float(rows["x"][-1])


def test_multi_col_scalar_vs_slice_shape():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            scalar = sub[1]
            one_slice = sub[1:2]
            assert isinstance(scalar, np.void)
            assert one_slice.shape == (1,)
            assert float(scalar["x"]) == float(one_slice["x"][0])


def test_multi_col_bool_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_uncompressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            with pytest.raises(ValueError, match="bool"):
                _ = sub[True]


# ---------------------------------------------------------------------------
# CompressedSingleColumnSubset / CompressedColumnSubset
# ---------------------------------------------------------------------------


def test_compressed_single_col_scalar():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            for i in range(len(rows)):
                v = col[i]
                assert np.ndim(v) == 0
                assert float(v) == float(rows["x"][i])
            # negative
            assert float(col[-1]) == float(rows["x"][-1])


def test_compressed_single_col_scalar_vs_slice_shape():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_compressed(path)
        with rustfits.FITS(path) as f:
            col = f[1]["x"]
            assert np.ndim(col[3]) == 0
            assert col[3:4].shape == (1,)


def test_compressed_multi_col_scalar_returns_void():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_compressed(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["x", "y"]]
            rec = sub[3]
            assert isinstance(rec, np.void)
            assert rec.dtype.names == ("x", "y")
            assert float(rec["x"]) == float(rows["x"][3])


# ---------------------------------------------------------------------------
# ASCII single & multi-col
# ---------------------------------------------------------------------------


def test_ascii_single_col_scalar():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_ascii(path)
        with rustfits.FITS(path) as f:
            col = f[1]["flux"]
            for i in range(len(rows)):
                v = col[i]
                assert np.ndim(v) == 0
                npt.assert_allclose(
                    float(v), float(rows["flux"][i]), rtol=1e-5
                )
            assert np.ndim(col[-1]) == 0


def test_ascii_single_col_scalar_vs_slice_shape():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_ascii(path)
        with rustfits.FITS(path) as f:
            col = f[1]["id"]
            assert np.ndim(col[0]) == 0
            assert col[0:1].shape == (1,)


def test_ascii_multi_col_scalar_returns_void():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        rows = _mk_ascii(path)
        with rustfits.FITS(path) as f:
            sub = f[1][["id", "flux"]]
            rec = sub[2]
            assert isinstance(rec, np.void)
            assert rec.dtype.names == ("id", "flux")
            assert int(rec["id"]) == int(rows["id"][2])


def test_ascii_single_col_bool_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fits")
        _mk_ascii(path)
        with rustfits.FITS(path) as f:
            col = f[1]["id"]
            with pytest.raises(ValueError, match="bool"):
                _ = col[True]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
