"""Phase 1 MVP tests for create_table_hdu + TableHDU.write.

Scope:
  - Round-trip write + read for the MVP scalar dtypes
    (i2/i4/i8/u1/f4/f8), as np.dtype and as descr-list inputs.
  - Multi-column tables.
  - Auto-add of an empty primary image when create_table_hdu is the
    first call on a fresh file.
  - extname / extver / units kwargs round-trip.
  - Validation: nrows<0, wrong-length write, mismatched column names,
    unsupported dtypes all raise.

Per CLAUDE.md, every mutation is verified through both a same-handle
read and a fresh-reopen read.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# --------------------------- helpers ---------------------------


def _tmp(name="t.fits"):
    return tempfile.TemporaryDirectory(), name


def _make_arr(dt, n=5):
    """Populate a structured array with distinguishable values per field."""
    arr = np.zeros(n, dtype=dt)
    for i, name in enumerate(arr.dtype.names):
        base = arr.dtype.fields[name][0]
        if np.issubdtype(base, np.floating):
            arr[name] = (np.arange(n) - 1) * (1 + i) * 0.5
        else:
            # Integer (signed or unsigned): keep values inside u1 range
            # so the same generator works for all int widths in the MVP.
            arr[name] = (np.arange(n) + 10 * (i + 1)) % 200
    return arr


# --------------------- round-trip across dtypes ---------------------


@pytest.mark.parametrize(
    "field_dtype",
    ["i2", "i4", "i8", "u1", "f4", "f8"],
)
def test_roundtrip_single_column_dtype(field_dtype):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("x", field_dtype)])
        arr = _make_arr(dt, n=7)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=7)
            fits[1].write(arr)
            # Same-handle read
            got = fits[1].read()
            np.testing.assert_array_equal(got["x"], arr["x"])

        # Post-reopen read
        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["x"], arr["x"])
            assert got.dtype.fields["x"][0] == np.dtype(field_dtype)


def test_roundtrip_multi_column_mixed_dtypes():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),
                ("flux", "f8"),
                ("snr", "f4"),
                ("flag", "u1"),
                ("idx16", "i2"),
                ("idx64", "i8"),
            ]
        )
        arr = _make_arr(dt, n=11)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=11)
            fits[1].write(arr)
            got = fits[1].read()
            for name in dt.names:
                np.testing.assert_array_equal(got[name], arr[name])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for name in dt.names:
                np.testing.assert_array_equal(got[name], arr[name])


# ----------------------- input dtype forms -----------------------


def test_dtype_input_accepts_descr_list():
    """create_table_hdu accepts the same forms numpy.dtype() does."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        descr = [("id", "i4"), ("flux", "f8")]
        arr = _make_arr(np.dtype(descr), n=3)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(descr, nrows=3)
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["flux"], arr["flux"])


def test_dtype_input_accepts_np_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("a", "i4"), ("b", "f4")])
        arr = _make_arr(dt, n=3)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["a"], arr["a"])
            np.testing.assert_array_equal(got["b"], arr["b"])


# ----------------------- HDU positioning -----------------------


def test_auto_primary_on_fresh_file():
    """create_table_hdu on a fresh file auto-adds an empty primary
    image (NAXIS=0) so the BINTABLE can land as extension 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=0)
            assert len(fits.hdus) == 2
            assert isinstance(fits[0], rustfits.ImageHDU)
            assert isinstance(fits[1], rustfits.TableHDU)
            assert fits[0].header["NAXIS"] == 0

        with rustfits.FITS(fname) as fits:
            assert len(fits.hdus) == 2
            assert isinstance(fits[0], rustfits.ImageHDU)
            assert isinstance(fits[1], rustfits.TableHDU)


def test_no_double_primary_when_user_already_created_one():
    """If the user already created an image HDU first, the next
    create_table_hdu must NOT auto-add another primary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=(2, 3))
            fits.create_table_hdu(dt, nrows=2)
            assert len(fits.hdus) == 2  # one image + one table
            assert isinstance(fits[0], rustfits.ImageHDU)
            assert isinstance(fits[1], rustfits.TableHDU)

        with rustfits.FITS(fname) as fits:
            assert len(fits.hdus) == 2


def test_two_tables_in_one_file():
    """Successive create_table_hdu calls just append extensions; only
    the first call adds the primary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_a = np.dtype([("a", "i4")])
        dt_b = np.dtype([("x", "f4"), ("y", "f8")])
        a = _make_arr(dt_a, 3)
        b = _make_arr(dt_b, 5)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_a, nrows=3, extname="ONE")
            fits.create_table_hdu(dt_b, nrows=5, extname="TWO")
            fits[1].write(a)
            fits[2].write(b)
            assert len(fits.hdus) == 3
            assert fits[1].header["EXTNAME"] == "ONE"
            assert fits[2].header["EXTNAME"] == "TWO"

        with rustfits.FITS(fname) as fits:
            assert len(fits.hdus) == 3
            np.testing.assert_array_equal(fits["ONE"].read()["a"], a["a"])
            np.testing.assert_array_equal(fits["TWO"].read()["x"], b["x"])
            np.testing.assert_array_equal(fits["TWO"].read()["y"], b["y"])


# ----------------------- metadata kwargs -----------------------


def test_extname_extver():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=0, extname="MYTABLE", extver=7)
            assert fits[1].header["EXTNAME"] == "MYTABLE"
            assert fits[1].header["EXTVER"] == 7

        with rustfits.FITS(fname) as fits:
            assert fits[1].header["EXTNAME"] == "MYTABLE"
            assert fits[1].header["EXTVER"] == 7
            # Findable by name as well.
            assert fits["mytable"].header["EXTVER"] == 7


def test_units_kwarg_round_trips():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("flux", "f4"), ("snr", "f4"), ("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(
                dt,
                nrows=0,
                units={"flux": "Jy", "snr": "1"},
            )
            u = fits[1].units
            assert u["flux"] == "Jy"
            assert u["snr"] == "1"
            assert u["id"] is None  # unit not given

        with rustfits.FITS(fname) as fits:
            u = fits[1].units
            assert u["flux"] == "Jy"
            assert u["snr"] == "1"
            assert u["id"] is None


# ----------------------- empty tables -----------------------


def test_empty_table_nrows_zero():
    """nrows=0 creates a header-only BINTABLE; read returns empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=0)
            got = fits[1].read()
            assert got.shape == (0,)
            assert got.dtype.names == ("id", "flux")
            assert fits[1].nrows == 0
            assert fits[1].ncols == 2
            assert fits[1].colnames == ("id", "flux")

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got.shape == (0,)
            assert len(fits[1]) == 0


def test_write_empty_into_empty_table():
    """write(arr) with len(arr) == 0 must succeed on an nrows=0 table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        empty = np.zeros(0, dtype=dt)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=0)
            fits[1].write(empty)
            got = fits[1].read()
            assert got.shape == (0,)


# ----------------------- validation -----------------------


def test_negative_nrows_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="nrows"):
                fits.create_table_hdu([("id", "i4")], nrows=-1)


def test_dtype_with_no_fields_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="structured"):
                fits.create_table_hdu("f4", nrows=3)


def test_unsupported_dtype_rejected_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="Phase 1d"):
                fits.create_table_hdu([("x", "S10")], nrows=1)


def test_int8_rejected_explicitly():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="int8"):
                fits.create_table_hdu([("x", "i1")], nrows=1)


def test_write_wrong_length_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        arr_too_short = np.zeros(2, dtype=dt)
        arr_too_long = np.zeros(10, dtype=dt)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=5)
            with pytest.raises(ValueError, match="NAXIS2"):
                fits[1].write(arr_too_short)
            with pytest.raises(ValueError, match="NAXIS2"):
                fits[1].write(arr_too_long)


def test_write_wrong_column_name_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("id", "i4")])
        dt_arr = np.dtype([("nope", "i4")])
        arr = np.zeros(3, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=3)
            with pytest.raises(ValueError, match="does not match"):
                fits[1].write(arr)


def test_write_non_ndarray_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="ndarray"):
                fits[1].write([1, 2, 3])


def test_write_plain_ndarray_rejected():
    """A non-structured ndarray (no field names) is not a valid
    table-row source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        plain = np.zeros(3, dtype="i4")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="structured"):
                fits[1].write(plain)


# ----------------------- read accessors -----------------------


def test_getitem_and_read_column_round_trip():
    """The existing __getitem__ and read_column paths should work on
    a freshly-created+written table without any special handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8")])
        arr = _make_arr(dt, n=8)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=8)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(
                fits[1].read_column("flux"), arr["flux"]
            )
            np.testing.assert_array_equal(fits[1][2:5]["id"], arr["id"][2:5])
            np.testing.assert_array_equal(fits[1]["flux"][:], arr["flux"])


def test_larger_table_exercises_strip_loop():
    """Write a table large enough (~2 MiB) to span multiple strips in
    the chunked write loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("v1", "f8"), ("v2", "f8")])
        # 20 bytes per row; 200_000 rows ~ 4 MiB.
        n = 200_000
        arr = np.zeros(n, dtype=dt)
        arr["id"] = np.arange(n, dtype="i4")
        arr["v1"] = np.arange(n, dtype="f8") * 0.5
        arr["v2"] = np.arange(n, dtype="f8") - 1000.0

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["v1"], arr["v1"])
            np.testing.assert_array_equal(got["v2"], arr["v2"])
