"""
ASCII-table __setitem__ — AsciiTableHDU.__setitem__.

Covers all 7 forms (mirror of the BINTABLE surface, minus VLA):

- hdu[i] = record            # single-row write
- hdu[a:b[:s]] = arr         # slice (step=1 contig; step>1 strided)
- hdu["col"] = arr           # whole-column write
- hdu[[i, j, k]] = arr       # fancy-row write
- hdu[["a", "b"]] = arr      # multi-column subset write

Plus per-letter coverage on I/F/E/D/A columns, unsigned-int trick,
negative indices, OOB rejection, width-overflow rejection, and the
same-handle + post-reopen verification pattern.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# Five-column schema exercising every supported ASCII TFORM letter.
DT = np.dtype(
    [
        ("ID", "i8"),  # -> I20
        ("FLUX", "f4"),  # -> E15.7
        ("MJD", "f8"),  # -> D25.17
        ("MASK", "u4"),  # -> I20 + TZERO=2^63 (unsigned-int trick)
        ("NAME", "S10"),  # -> A10
    ]
)


def _make_rows(n, base=0):
    arr = np.zeros(n, dtype=DT)
    arr["ID"] = np.arange(n, dtype="i8") + base * 1000
    arr["FLUX"] = (np.arange(n, dtype="f4") + base) * 1.5
    arr["MJD"] = 58000.0 + np.arange(n, dtype="f8") + base * 100
    arr["MASK"] = np.arange(n, dtype="u4") + base * 10
    arr["NAME"] = [f"r{base * 1000 + i:04d}".encode() for i in range(n)]
    return arr


def _create_table(fname, nrows):
    arr = _make_rows(nrows, base=0)
    with rustfits.FITS(fname, "w+") as f:
        f.create_ascii_table_hdu(DT, nrows=nrows)
        f[1].write(arr)
    return arr


def _both(fname, mutate, predicate):
    """Run mutate(fits) on r+, verify via same-handle then reopen reads."""
    with rustfits.FITS(fname, "r+") as fits:
        mutate(fits)
        predicate(fits)
    with rustfits.FITS(fname, "r") as fits:
        predicate(fits)


# ---------------------------------------------------------------------------
# Single-row writes: hdu[i] = record
# ---------------------------------------------------------------------------


def test_single_row_write_void_scalar():
    """Read a row to get np.void, modify a field, write back."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        rec = _make_rows(1, base=9)[0]  # np.void

        def mutate(fits):
            fits[1][2] = rec

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][2] == 9000
            assert arr["FLUX"][2] == pytest.approx(13.5, rel=1e-6)
            assert arr["MJD"][2] == pytest.approx(58000.0 + 9 * 100)
            assert arr["MASK"][2] == 90
            assert arr["NAME"][2] == "r9000"

        _both(fname, mutate, check)


def test_single_row_write_shape1_ndarray():
    """Shape-(1,) structured ndarray is also accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        rec = _make_rows(1, base=7)

        def mutate(fits):
            fits[1][1] = rec

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][1] == 7000

        _both(fname, mutate, check)


def test_single_row_negative_index():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)
        rec = _make_rows(1, base=5)[0]

        def mutate(fits):
            fits[1][-1] = rec

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][3] == 5000

        _both(fname, mutate, check)


def test_single_row_out_of_bounds_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        rec = _make_rows(1, base=0)[0]
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(IndexError):
                fits[1][3] = rec
            with pytest.raises(IndexError):
                fits[1][-4] = rec


def test_single_row_rejects_non_record():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 2)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][0] = 42  # bare int, not a record


# ---------------------------------------------------------------------------
# Slice writes: hdu[a:b[:s]] = arr
# ---------------------------------------------------------------------------


def test_slice_write_step_1():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 10)
        rep = _make_rows(4, base=3)

        def mutate(fits):
            fits[1][2:6] = rep

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"][2:6], rep["ID"])
            assert arr["ID"][0] == 0  # untouched
            assert arr["ID"][6] == 6  # untouched

        _both(fname, mutate, check)


def test_slice_write_step_gt_1():
    """Stepped slice routes through write_ascii_table_strided."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 10)
        rep = _make_rows(3, base=4)  # rows 0, 3, 6

        def mutate(fits):
            fits[1][0:9:3] = rep

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][0] == 4000
            assert arr["ID"][3] == 4001
            assert arr["ID"][6] == 4002
            # Untouched rows
            assert arr["ID"][1] == 1
            assert arr["ID"][2] == 2

        _both(fname, mutate, check)


def test_slice_write_step_zero_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises((ValueError, Exception)):
                fits[1][::0] = _make_rows(1)


def test_slice_write_negative_step_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][::-1] = _make_rows(5)


def test_slice_write_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][0:3] = _make_rows(2)


def test_slice_write_empty_slice_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        orig = _create_table(fname, 5)
        empty = np.array([], dtype=DT)

        def mutate(fits):
            fits[1][3:3] = empty

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], orig["ID"])

        _both(fname, mutate, check)


# ---------------------------------------------------------------------------
# Whole-column writes: hdu["col"] = arr
# ---------------------------------------------------------------------------


def test_whole_column_write_i8():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        new_ids = np.array([100, 200, 300, 400, 500], dtype="i8")

        def mutate(fits):
            fits[1]["ID"] = new_ids

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], new_ids)
            # Other columns untouched
            np.testing.assert_array_equal(
                arr["FLUX"], np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype="f4")
            )

        _both(fname, mutate, check)


def test_whole_column_write_f4_e_format():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)
        new_flux = np.array([1.5e-3, 2.5e2, -1.25e10, 0.0], dtype="f4")

        def mutate(fits):
            fits[1]["FLUX"] = new_flux

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_allclose(arr["FLUX"], new_flux, rtol=1e-6)

        _both(fname, mutate, check)


def test_whole_column_write_f8_d_format():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        new_mjd = np.array([59000.123456789, 60000.987654321, 0.5], dtype="f8")

        def mutate(fits):
            fits[1]["MJD"] = new_mjd

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_allclose(arr["MJD"], new_mjd, rtol=1e-14)

        _both(fname, mutate, check)


def test_whole_column_write_unsigned_trick_u4():
    """u4 column round-trips through I20 + TZERO=2^63."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)
        new_mask = np.array([0, 42, (1 << 31) - 1, (1 << 32) - 1], dtype="u4")

        def mutate(fits):
            fits[1]["MASK"] = new_mask

        def check(fits):
            arr = fits[1].read()
            assert arr.dtype["MASK"] == np.dtype("u8")
            np.testing.assert_array_equal(arr["MASK"], new_mask.astype("u8"))

        _both(fname, mutate, check)


def test_whole_column_write_a_string():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        new_names = np.array([b"alpha", b"beta", b"gamma"], dtype="S10")

        def mutate(fits):
            fits[1]["NAME"] = new_names

        def check(fits):
            arr = fits[1].read()
            assert arr["NAME"][0] == "alpha"
            assert arr["NAME"][1] == "beta"
            assert arr["NAME"][2] == "gamma"

        _both(fname, mutate, check)


def test_whole_column_unknown_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1]["bogus"] = np.array([1, 2, 3], dtype="i8")


def test_whole_column_case_insensitive_name():
    """Lower-case name lookup matches upper-case column."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        new_ids = np.array([7, 8, 9], dtype="i8")

        def mutate(fits):
            fits[1]["id"] = new_ids  # lowercase

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], new_ids)

        _both(fname, mutate, check)


def test_whole_column_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1]["ID"] = np.array([1, 2, 3], dtype="i8")


def test_whole_column_width_overflow_raises():
    """Integer too large to fit in field width raises."""
    dtype = np.dtype([("X", "i8")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            # Narrow format I3 fits 0-999
            f.create_ascii_table_hdu(dtype, nrows=2, formats={"X": "I3"})
            f[1].write(np.array([(1,), (2,)], dtype=dtype))
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1]["X"] = np.array([1, 9999], dtype="i8")


# ---------------------------------------------------------------------------
# Fancy-row writes: hdu[[i, j, k]] = arr
# ---------------------------------------------------------------------------


def test_fancy_rows_write_positive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 8)
        rep = _make_rows(3, base=6)

        def mutate(fits):
            fits[1][[1, 4, 6]] = rep

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][1] == 6000
            assert arr["ID"][4] == 6001
            assert arr["ID"][6] == 6002
            assert arr["ID"][0] == 0  # untouched
            assert arr["ID"][2] == 2  # untouched

        _both(fname, mutate, check)


def test_fancy_rows_write_with_negatives():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        rep = _make_rows(2, base=8)

        def mutate(fits):
            fits[1][[-1, -3]] = rep  # rows 4 and 2

        def check(fits):
            arr = fits[1].read()
            assert arr["ID"][4] == 8000
            assert arr["ID"][2] == 8001

        _both(fname, mutate, check)


def test_fancy_rows_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][[0, 2]] = _make_rows(3)


def test_fancy_rows_oob_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(IndexError):
                fits[1][[0, 5]] = _make_rows(2)


# ---------------------------------------------------------------------------
# Multi-column writes: hdu[["a", "b"]] = arr
# ---------------------------------------------------------------------------


def test_multi_column_write_two_columns():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 4)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(4, dtype=sub_dt)
        rep["ID"] = [11, 22, 33, 44]
        rep["FLUX"] = [0.5, 1.5, 2.5, 3.5]

        def mutate(fits):
            fits[1][["ID", "FLUX"]] = rep

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], [11, 22, 33, 44])
            np.testing.assert_allclose(
                arr["FLUX"], [0.5, 1.5, 2.5, 3.5], rtol=1e-6
            )
            # MJD untouched
            assert arr["MJD"][0] == pytest.approx(58000.0)

        _both(fname, mutate, check)


def test_multi_column_extra_field_tolerated():
    """Extra fields in value are ignored (forward-compat)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4"), ("EXTRA", "f8")])
        rep = np.zeros(3, dtype=sub_dt)
        rep["ID"] = [100, 200, 300]
        rep["FLUX"] = [1.0, 2.0, 3.0]
        rep["EXTRA"] = [9.9, 9.9, 9.9]

        def mutate(fits):
            fits[1][["ID", "FLUX"]] = rep

        def check(fits):
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["ID"], [100, 200, 300])

        _both(fname, mutate, check)


def test_multi_column_missing_field_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8")])  # missing FLUX
        rep = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][["ID", "FLUX"]] = rep


def test_multi_column_duplicate_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8"), ("FLUX", "f4")])
        rep = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][["ID", "ID"]] = rep


def test_multi_column_unknown_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 3)
        sub_dt = np.dtype([("ID", "i8"), ("BOGUS", "f4")])
        rep = np.zeros(3, dtype=sub_dt)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1][["ID", "BOGUS"]] = rep


# ---------------------------------------------------------------------------
# Cross-tool round-trip
# ---------------------------------------------------------------------------


def test_astropy_can_read_after_setitem():
    """After rustfits __setitem__, astropy reads the same values."""
    astropy = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _create_table(fname, 5)
        rep = _make_rows(3, base=7)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][1:4] = rep

        with astropy.open(fname) as hdul:
            d = hdul[1].data
            np.testing.assert_array_equal(d["ID"][1:4], rep["ID"])
