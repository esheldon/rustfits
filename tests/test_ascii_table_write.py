"""
ASCII-table bulk write — AsciiTableHDU.write.

Tests dtype-by-dtype round-trip (write -> read), input forms
(structured ndarray / dict / list+names), unsigned-int trick on u*
columns, and cross-tool verification (astropy + fitsio read what
rustfits writes).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _open_write(path, dtype, rows, *, units=None, formats=None, extname=None):
    """Create + write helper: returns the path."""
    nrows = len(rows) if hasattr(rows, "__len__") else 0
    with rustfits.FITS(path, "w+") as f:
        f.create_ascii_table_hdu(
            dtype,
            nrows=nrows,
            units=units,
            formats=formats,
            extname=extname,
        )
        f[1].write(rows)
    return path


# ---------------------------------------------------------------------------
# round-trip every supported dtype
# ---------------------------------------------------------------------------


def test_write_i8_round_trip():
    dtype = np.dtype([("X", "i8")])
    data = np.array([(1,), (-2,), (3,)], dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], [1, -2, 3])


def test_write_i4_promotes_to_i8_on_read():
    """Input i4 writes as I20; read returns i8."""
    dtype = np.dtype([("X", "i4")])
    data = np.array([(10,), (-20,), (30,)], dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            assert arr.dtype["X"] == np.dtype("i8")
            np.testing.assert_array_equal(arr["X"], [10, -20, 30])


def test_write_u4_unsigned_trick_round_trip():
    """u4 writes as I20 + TZERO=2^63; read returns u8 with the same values."""
    dtype = np.dtype([("X", "u4")])
    data = np.array([(0,), (42,), (4294967295,)], dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            assert f[1].header["TZERO1"] == 1 << 63
            arr = f[1].read()
            assert arr.dtype["X"] == np.dtype("u8")
            np.testing.assert_array_equal(
                arr["X"],
                np.array([0, 42, 4294967295], dtype=np.uint64),
            )


def test_write_u8_unsigned_trick_full_range():
    dtype = np.dtype([("X", "u8")])
    data = np.array(
        [(0,), (1,), ((1 << 64) - 1,), ((1 << 63),)],
        dtype=dtype,
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            expected = np.array(
                [0, 1, (1 << 64) - 1, 1 << 63],
                dtype=np.uint64,
            )
            np.testing.assert_array_equal(arr["X"], expected)


def test_write_f4_round_trip():
    dtype = np.dtype([("X", "f4")])
    data = np.array(
        [(1.5,), (-2.25,), (1.5e10,)],
        dtype=dtype,
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            # Read maps E15.7 (d=7) -> f4 per the d<=7 rule.
            assert arr.dtype["X"] == np.dtype("f4")
            np.testing.assert_allclose(
                arr["X"],
                [1.5, -2.25, 1.5e10],
                rtol=1e-6,
            )


def test_write_f8_round_trip_full_precision():
    dtype = np.dtype([("X", "f8")])
    data = np.array(
        [(1.23456789012345e10,), (-7.654321098765432e-15,), (0.0,)],
        dtype=dtype,
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            # D25.17 -> f8.
            assert arr.dtype["X"] == np.dtype("f8")
            np.testing.assert_allclose(
                arr["X"],
                [1.23456789012345e10, -7.654321098765432e-15, 0.0],
                rtol=1e-14,
            )


def test_write_S_string_round_trip():
    dtype = np.dtype([("NAME", "S8")])
    data = np.array(
        [("alice",), ("bob",), ("",), ("longname",)],
        dtype=dtype,
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            assert arr.dtype["NAME"] == np.dtype("U8")
            assert list(arr["NAME"]) == ["alice", "bob", "", "longname"]


def test_write_U_unicode_round_trip():
    dtype = np.dtype([("NAME", "U6")])
    data = np.array([("alpha",), ("beta",), ("gamma",)], dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            assert list(arr["NAME"]) == ["alpha", "beta", "gamma"]


def test_write_U_non_ascii_raises():
    """numpy U field containing non-ASCII codepoints raises on write."""
    dtype = np.dtype([("NAME", "U6")])
    data = np.array([("café",)], dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=1)
            with pytest.raises(ValueError, match="non-ASCII"):
                f[1].write(data)


# ---------------------------------------------------------------------------
# input forms
# ---------------------------------------------------------------------------


def test_write_dict_input():
    dtype = np.dtype([("X", "i4"), ("Y", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=3)
            f[1].write(
                {"X": np.array([1, 2, 3]), "Y": np.array([1.5, 2.5, 3.5])}
            )
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], [1, 2, 3])
            np.testing.assert_allclose(arr["Y"], [1.5, 2.5, 3.5], rtol=1e-5)


def test_write_list_names_input():
    dtype = np.dtype([("X", "i4"), ("Y", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=2)
            f[1].write(
                [np.array([7, 9]), np.array([0.5, -0.25])],
                names=["X", "Y"],
            )
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], [7, 9])


def test_write_dict_missing_column_raises():
    dtype = np.dtype([("X", "i4"), ("Y", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=2)
            with pytest.raises(ValueError, match="missing column 'Y'"):
                f[1].write({"X": np.array([1, 2])})


def test_write_length_mismatch_raises():
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=3)
            with pytest.raises(ValueError, match="expected 3 rows"):
                f[1].write(np.array([(1,), (2,)], dtype=dtype))


def test_write_overflow_raises_on_integer_too_wide():
    """Custom formats= I3 + value 9999 raises (digits exceed width)."""
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(
                dtype,
                nrows=1,
                formats={"X": "I3"},
            )
            with pytest.raises(ValueError, match="does not fit"):
                f[1].write(np.array([(9999,)], dtype=dtype))


# ---------------------------------------------------------------------------
# extend alias + zero-row write
# ---------------------------------------------------------------------------


def test_write_zero_rows():
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            # Writing 0 rows is a no-op.
            f[1].write(np.array([], dtype=dtype))
            assert f[1].nrows == 0


# ---------------------------------------------------------------------------
# cross-tool: astropy / fitsio can read what rustfits writes
# ---------------------------------------------------------------------------


def test_astropy_reads_rustfits_output():
    astropy_fits = pytest.importorskip("astropy.io.fits")
    dtype = np.dtype(
        [
            ("ID", "i8"),
            ("FLUX", "f4"),
            ("MJD", "f8"),
            ("NAME", "S6"),
        ]
    )
    data = np.zeros(3, dtype=dtype)
    data["ID"] = [1, 2, 3]
    data["FLUX"] = [1.5, -2.25, 0.0]
    data["MJD"] = [58000.123456789, 58001.987654321, 58002.5]
    data["NAME"] = ["alpha", "beta", "g"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with astropy_fits.open(fname) as hdul:
            assert isinstance(hdul[1], astropy_fits.TableHDU)
            t = hdul[1].data
            # astropy returns int32 for any Iw column (cfitsio's
            # FITS2NUMPY default).  Verify values not dtypes.
            np.testing.assert_array_equal(t["ID"], [1, 2, 3])
            np.testing.assert_allclose(
                t["FLUX"],
                [1.5, -2.25, 0.0],
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                t["MJD"],
                [58000.123456789, 58001.987654321, 58002.5],
                rtol=1e-12,
            )
            # astropy returns S-strings (bytes); decode for compare.
            assert [s.strip() for s in t["NAME"]] == ["alpha", "beta", "g"]


def test_fitsio_reads_rustfits_output():
    fitsio = pytest.importorskip("fitsio")
    dtype = np.dtype(
        [
            ("ID", "i8"),
            ("FLUX", "f4"),
            ("NAME", "S8"),
        ]
    )
    data = np.zeros(3, dtype=dtype)
    data["ID"] = [10, 20, 30]
    data["FLUX"] = [1.5, -2.25, 0.0]
    data["NAME"] = ["a", "bb", "ccc"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _open_write(os.path.join(tmp, "t.fits"), dtype, data)
        with fitsio.FITS(fname) as f:
            assert f[1].get_exttype() == "ASCII_TBL"
            t = f[1].read()
            np.testing.assert_array_equal(t["ID"], [10, 20, 30])
            np.testing.assert_allclose(
                t["FLUX"],
                [1.5, -2.25, 0.0],
                rtol=1e-5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
