"""
ASCII-table subset writes — AsciiSingleColumnSubset.__setitem__ /
.write and AsciiColumnSubset.__setitem__ / .write.

Tests the four-class symmetric write surface:

- hdu["col"][i] = scalar              cell write
- hdu["col"][a:b[:s]] = arr           slice write on one column
- hdu["col"][[i, j, k]] = arr         fancy-row write on one column
- hdu[["a", "b"]][i] = record         single-row column-subset
- hdu[["a", "b"]][a:b] = arr          slice column-subset
- hdu[["a", "b"]][[i, j, k]] = arr    fancy-row column-subset

Plus the wholesale + row-restricted forms of subset.write(data, rows=).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


DT = np.dtype(
    [
        ("ID", "i8"),
        ("FLUX", "f4"),
        ("MJD", "f8"),
        ("NAME", "S8"),
    ]
)


def _make_rows(n, base=0):
    arr = np.zeros(n, dtype=DT)
    arr["ID"] = np.arange(n, dtype="i8") + base * 1000
    arr["FLUX"] = (np.arange(n, dtype="f4") + base) * 0.5
    arr["MJD"] = 58000.0 + np.arange(n, dtype="f8") + base * 10
    arr["NAME"] = [f"r{base * 1000 + i:03d}".encode() for i in range(n)]
    return arr


def _create_table(fname, nrows):
    arr = _make_rows(nrows, base=0)
    with rustfits.FITS(fname, "w+") as f:
        f.create_ascii_table_hdu(DT, nrows=nrows)
        f[1].write(arr)
    return arr


def _both(fname, mutate, predicate):
    with rustfits.FITS(fname, "r+") as fits:
        mutate(fits)
        predicate(fits)
    with rustfits.FITS(fname, "r") as fits:
        predicate(fits)


# ---------------------------------------------------------------------------
# SingleColumnSubset: hdu["col"][rows] = ...
# ---------------------------------------------------------------------------


def test_single_column_cell_write_int_field():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)

        def mutate(fits):
            fits[1]["ID"][2] = 999

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][2] == 999
            assert arr["ID"][0] == 0  # untouched

        _both(fname, mutate, check)


def test_single_column_cell_write_float_field():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)

        def mutate(fits):
            fits[1]["FLUX"][1] = 3.14

        def check(fits):
            arr = fits[1].read()
            assert arr["FLUX"][1] == pytest.approx(3.14, rel=1e-6)

        _both(fname, mutate, check)


def test_single_column_cell_write_string_field():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)

        def mutate(fits):
            fits[1]["NAME"][1] = "hello"

        def check(fits):
            arr = fits[1].read()
            assert arr["NAME"][1] == "hello"

        _both(fname, mutate, check)


def test_single_column_cell_write_negative_index():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)

        def mutate(fits):
            fits[1]["ID"][-1] = 7777

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][3] == 7777

        _both(fname, mutate, check)


def test_single_column_slice_write():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 6)

        def mutate(fits):
            fits[1]["ID"][1:4] = np.array([100, 200, 300], dtype="i8")

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"][1:4], [100, 200, 300])
            assert arr["ID"][0] == 0
            assert arr["ID"][4] == 4

        _both(fname, mutate, check)


def test_single_column_slice_write_stepped():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 8)

        def mutate(fits):
            fits[1]["ID"][0:6:2] = np.array([10, 20, 30], dtype="i8")

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][0] == 10
            assert arr["ID"][2] == 20
            assert arr["ID"][4] == 30
            assert arr["ID"][1] == 1  # untouched

        _both(fname, mutate, check)


def test_single_column_fancy_write():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 6)

        def mutate(fits):
            fits[1]["ID"][[0, 3, 5]] = np.array([99, 88, 77], dtype="i8")

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][0] == 99
            assert arr["ID"][3] == 88
            assert arr["ID"][5] == 77
            assert arr["ID"][1] == 1  # untouched

        _both(fname, mutate, check)


def test_single_column_full_slice_write():
    """hdu["col"][:] = arr matches hdu["col"] = arr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)
        new_flux = np.array([5.0, 6.0, 7.0, 8.0], dtype="f4")

        def mutate(fits):
            fits[1]["FLUX"][:] = new_flux

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_allclose(arr["FLUX"], new_flux, rtol=1e-6)

        _both(fname, mutate, check)


def test_single_column_write_method_no_rows():
    """subset.write(data) with rows=None routes to whole-column write."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        new_ids = np.array([55, 66, 77], dtype="i8")

        def mutate(fits):
            fits[1]["ID"].write(new_ids)

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], new_ids)

        _both(fname, mutate, check)


def test_single_column_write_method_with_rows():
    """subset.write(data, rows=...) writes only those rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)

        def mutate(fits):
            fits[1]["ID"].write(np.array([100, 200], dtype="i8"), rows=[1, 3])

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][1] == 100
            assert arr["ID"][3] == 200
            assert arr["ID"][0] == 0  # untouched

        _both(fname, mutate, check)


def test_single_column_unsigned_trick_round_trip():
    """Subset write goes through the unsigned-trick reverse transform."""
    dtype = np.dtype([("X", "u4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=3)
            f[1].write(np.array([(1,), (2,), (3,)], dtype=dtype))
        new_x = np.array([(1 << 31), (1 << 32) - 1, 0], dtype="u4")

        def mutate(fits):
            fits[1]["X"][:] = new_x

        def check(fits):
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("u8")
            np.testing.assert_array_equal(arr["X"], new_x.astype("u8"))

        _both(fname, mutate, check)


# ---------------------------------------------------------------------------
# ColumnSubset: hdu[["a", "b"]][rows] = ...
# ---------------------------------------------------------------------------


def test_multi_column_subset_single_row_write():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rec = np.zeros(1, dtype=sub_dt)
        rec["ID"] = [99]
        rec["FLUX"] = [3.5]

        def mutate(fits):
            fits[1][["ID", "FLUX"]][2] = rec[0]

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][2] == 99
            assert arr["FLUX"][2] == pytest.approx(3.5, rel=1e-6)
            # MJD untouched
            assert arr["MJD"][2] == pytest.approx(58002.0)

        _both(fname, mutate, check)


def test_multi_column_subset_slice_write():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 6)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(3, dtype=sub_dt)
        rep["ID"] = [11, 22, 33]
        rep["FLUX"] = [1.1, 2.2, 3.3]

        def mutate(fits):
            fits[1][["ID", "FLUX"]][1:4] = rep

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"][1:4], [11, 22, 33])
            np.testing.assert_allclose(
                arr["FLUX"][1:4], [1.1, 2.2, 3.3], rtol=1e-6
            )
            # MJD column untouched
            assert arr["MJD"][1] == pytest.approx(58001.0)

        _both(fname, mutate, check)


def test_multi_column_subset_fancy_write():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(2, dtype=sub_dt)
        rep["ID"] = [70, 80]
        rep["FLUX"] = [7.0, 8.0]

        def mutate(fits):
            fits[1][["ID", "FLUX"]][[0, 4]] = rep

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][0] == 70
            assert arr["ID"][4] == 80
            assert arr["FLUX"][0] == pytest.approx(7.0, rel=1e-6)
            assert arr["FLUX"][4] == pytest.approx(8.0, rel=1e-6)
            # row 2 untouched
            assert arr["ID"][2] == 2

        _both(fname, mutate, check)


def test_multi_column_subset_full_slice_write():
    """hdu[["a","b"]][:] = arr matches hdu[["a","b"]] = arr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(3, dtype=sub_dt)
        rep["ID"] = [100, 200, 300]
        rep["FLUX"] = [1.0, 2.0, 3.0]

        def mutate(fits):
            fits[1][["ID", "FLUX"]][:] = rep

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], [100, 200, 300])

        _both(fname, mutate, check)


def test_multi_column_subset_write_method_no_rows():
    """subset.write(data) with rows=None routes to multi-column write."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(3, dtype=sub_dt)
        rep["ID"] = [9, 8, 7]
        rep["FLUX"] = [0.5, 0.6, 0.7]

        def mutate(fits):
            fits[1][["ID", "FLUX"]].write(rep)

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], [9, 8, 7])

        _both(fname, mutate, check)


def test_multi_column_subset_write_method_with_rows():
    """subset.write(data, rows=...) writes only those rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(2, dtype=sub_dt)
        rep["ID"] = [50, 60]
        rep["FLUX"] = [5.0, 6.0]

        def mutate(fits):
            fits[1][["ID", "FLUX"]].write(rep, rows=slice(1, 3))

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][1] == 50
            assert arr["ID"][2] == 60
            assert arr["ID"][0] == 0  # untouched

        _both(fname, mutate, check)


def test_multi_column_subset_missing_field_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8")])  # missing FLUX
        rep = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][["ID", "FLUX"]][:] = rep
