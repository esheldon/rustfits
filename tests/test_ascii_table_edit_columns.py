"""
Tests for AsciiTableHDU.insert_column and AsciiTableHDU.delete_column.

Coverage:
- insert_column: append-at-end (default), position=, after=, before=,
  by name or index (positive + negative), unsigned-int trick, string
  column, format= override, with unit=, into non-last HDU.
- delete_column: by name, by index (positive + negative), case-
  insensitive, first / last.
- TBCOL value-shift: every column at/after the slot has its byte
  position updated; verified via header inspection.
- Round-trip insert→delete restores layout.
- Cross-tool: astropy reads edited tables correctly.
- Rejection paths: duplicate name, empty name, position out of range,
  multiple location kwargs, name not found, Object dtype, wrong shape.
- Non-last HDU: later HDUs shift forward / backward in lockstep.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_dtype():
    return np.dtype([("ID", "i8"), ("FLUX", "f4"), ("NAME", "S6")])


def _basic_data(nrows=5):
    dt = _basic_dtype()
    arr = np.zeros(nrows, dtype=dt)
    arr["ID"] = np.arange(nrows, dtype="i8")
    arr["FLUX"] = np.arange(nrows, dtype="f4") * 0.5
    arr["NAME"] = [f"r{i:03d}".encode() for i in range(nrows)]
    return arr


def _make_table(fname, dtype=None, data=None, with_image_hdu_first=True):
    if dtype is None:
        dtype = _basic_dtype()
    if data is None:
        data = _basic_data()
    with rustfits.FITS(fname, "w+") as f:
        if with_image_hdu_first:
            f.create_image_hdu("i4", (1,))
        f.create_ascii_table_hdu(dtype, nrows=len(data))
        table_idx = 1 if with_image_hdu_first else 0
        f[table_idx].write(data)
    return data


def _check_both(fname, mutate_fn, predicate_fn):
    with rustfits.FITS(fname, "r+") as f:
        mutate_fn(f)
        predicate_fn(f, "same handle")
    with rustfits.FITS(fname, "r") as f:
        predicate_fn(f, "after reopen")


# ---------------------------------------------------------------------------
# insert_column — append at end
# ---------------------------------------------------------------------------


def test_insert_appends_at_end_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        orig = _make_table(fn)
        new_col = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="f8")

        def mutate(f):
            f[1].insert_column("MJD", new_col)

        def check(f, tag):
            tbl = f[1]
            assert tbl.colnames == ("ID", "FLUX", "NAME", "MJD"), tag
            arr = tbl.read()
            np.testing.assert_array_equal(arr["ID"], orig["ID"])
            np.testing.assert_allclose(arr["FLUX"], orig["FLUX"])
            assert list(arr["NAME"]) == list(
                orig["NAME"].astype("U6")), tag
            np.testing.assert_allclose(arr["MJD"], new_col)

        _check_both(fn, mutate, check)


def test_insert_at_position_zero():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        orig = _make_table(fn)
        new_col = np.array([100, 200, 300, 400, 500], dtype="i8")

        def mutate(f):
            f[1].insert_column("RANK", new_col, position=0)

        def check(f, tag):
            tbl = f[1]
            assert tbl.colnames == ("RANK", "ID", "FLUX", "NAME"), tag
            assert tbl.header["TBCOL1"] == 1
            # RANK is I20 → TBCOL2 starts at 21
            assert tbl.header["TBCOL2"] == 21
            arr = tbl.read()
            np.testing.assert_array_equal(arr["RANK"], new_col)
            np.testing.assert_array_equal(arr["ID"], orig["ID"])

        _check_both(fn, mutate, check)


def test_insert_at_middle_position():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        new_col = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="f4")

        def mutate(f):
            f[1].insert_column("ERR", new_col, position=1)

        def check(f, tag):
            tbl = f[1]
            assert tbl.colnames == ("ID", "ERR", "FLUX", "NAME"), tag
            arr = tbl.read()
            np.testing.assert_allclose(arr["ERR"], new_col, rtol=1e-6)

        _check_both(fn, mutate, check)


def test_insert_after_name():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "AFTER_ID",
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="f8"),
                after="ID")

        def check(f, tag):
            assert f[1].colnames == ("ID", "AFTER_ID", "FLUX", "NAME"), tag

        _check_both(fn, mutate, check)


def test_insert_after_index():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "AFTER1",
                np.array([1, 2, 3, 4, 5], dtype="i4"),
                after=1)

        def check(f, tag):
            assert f[1].colnames == ("ID", "FLUX", "AFTER1", "NAME"), tag

        _check_both(fn, mutate, check)


def test_insert_before_name():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "BEFORE_FLUX",
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="f4"),
                before="FLUX")

        def check(f, tag):
            assert f[1].colnames == ("ID", "BEFORE_FLUX", "FLUX", "NAME"), tag

        _check_both(fn, mutate, check)


def test_insert_before_negative_index():
    """before=-1 means before the last column."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "X",
                np.array([1, 2, 3, 4, 5], dtype="i4"),
                before=-1)

        def check(f, tag):
            assert f[1].colnames == ("ID", "FLUX", "X", "NAME"), tag

        _check_both(fn, mutate, check)


def test_insert_case_insensitive_after():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "Z",
                np.array([1, 2, 3, 4, 5], dtype="i4"),
                after="id")  # lowercase

        def check(f, tag):
            assert f[1].colnames == ("ID", "Z", "FLUX", "NAME"), tag

        _check_both(fn, mutate, check)


def test_insert_unsigned_int_trick():
    """u4 column emits I20 + TZERO=2^63 and round-trips as u8."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        mask = np.array([0, 1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1],
                        dtype="u4")

        def mutate(f):
            f[1].insert_column("MASK", mask)

        def check(f, tag):
            tbl = f[1]
            assert "MASK" in tbl.colnames
            # find index of MASK column
            idx = tbl.colnames.index("MASK") + 1
            assert tbl.header[f"TZERO{idx}"] == 1 << 63, tag
            arr = tbl.read()
            assert arr.dtype["MASK"] == np.dtype("u8")
            np.testing.assert_array_equal(arr["MASK"], mask.astype("u8"))

        _check_both(fn, mutate, check)


def test_insert_with_format_override():
    """format='E20.10' overrides the auto-picked E15.7 for f4 input."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        new_col = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype="f4")

        def mutate(f):
            f[1].insert_column("X", new_col, format="E20.10")

        def check(f, tag):
            tbl = f[1]
            idx = tbl.colnames.index("X") + 1
            assert tbl.header[f"TFORM{idx}"] == "E20.10", tag
            arr = tbl.read()
            np.testing.assert_allclose(arr["X"], new_col, rtol=1e-6)

        _check_both(fn, mutate, check)


def test_insert_with_unit():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].insert_column(
                "MJD",
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="f8"),
                unit="day")

        def check(f, tag):
            assert f[1].units.get("MJD") == "day", tag

        _check_both(fn, mutate, check)


# ---------------------------------------------------------------------------
# TBCOL value shifts on insert
# ---------------------------------------------------------------------------


def test_insert_at_zero_shifts_all_tbcols():
    """Insert at position 0 shifts every existing TBCOL by new_col_width."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        # Capture original TBCOLs
        with rustfits.FITS(fn) as f:
            tbcol1 = f[1].header["TBCOL1"]  # ID @ 1
            tbcol2 = f[1].header["TBCOL2"]  # FLUX @ 21
            tbcol3 = f[1].header["TBCOL3"]  # NAME @ 36

        def mutate(f):
            # I20 column → 20-byte shift
            f[1].insert_column("R", np.array([1, 2, 3, 4, 5], dtype="i8"),
                               position=0)

        def check(f, tag):
            # New R column at TBCOL1
            assert f[1].header["TBCOL1"] == 1, tag
            # ID is now column 2 at original_offset + 20
            assert f[1].header["TBCOL2"] == tbcol1 + 20, tag
            assert f[1].header["TBCOL3"] == tbcol2 + 20, tag
            assert f[1].header["TBCOL4"] == tbcol3 + 20, tag

        _check_both(fn, mutate, check)


def test_delete_shifts_later_tbcols():
    """Delete shifts every TBCOL after the deleted slot by -deleted_width."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn) as f:
            tbcol1 = f[1].header["TBCOL1"]
            tbcol2 = f[1].header["TBCOL2"]
            tbcol3 = f[1].header["TBCOL3"]

        def mutate(f):
            # Delete FLUX (E15.7 → 15 bytes wide)
            f[1].delete_column("FLUX")

        def check(f, tag):
            # ID stays at TBCOL1
            assert f[1].header["TBCOL1"] == tbcol1, tag
            # NAME shifts forward by 15 bytes (FLUX's width)
            assert f[1].header["TBCOL2"] == tbcol3 - 15, tag
            _ = tbcol2  # silence linter

        _check_both(fn, mutate, check)


# ---------------------------------------------------------------------------
# delete_column
# ---------------------------------------------------------------------------


def test_delete_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        orig = _make_table(fn)

        def mutate(f):
            f[1].delete_column("FLUX")

        def check(f, tag):
            tbl = f[1]
            assert tbl.colnames == ("ID", "NAME"), tag
            arr = tbl.read()
            np.testing.assert_array_equal(arr["ID"], orig["ID"])
            assert list(arr["NAME"]) == list(orig["NAME"].astype("U6"))

        _check_both(fn, mutate, check)


def test_delete_by_index():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].delete_column(1)  # FLUX

        def check(f, tag):
            assert f[1].colnames == ("ID", "NAME"), tag

        _check_both(fn, mutate, check)


def test_delete_by_negative_index():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].delete_column(-1)  # NAME (last)

        def check(f, tag):
            assert f[1].colnames == ("ID", "FLUX"), tag

        _check_both(fn, mutate, check)


def test_delete_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)

        def mutate(f):
            f[1].delete_column("flux")  # lowercase

        def check(f, tag):
            assert f[1].colnames == ("ID", "NAME"), tag

        _check_both(fn, mutate, check)


def test_delete_first_then_last():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            f[1].delete_column(0)
            assert f[1].colnames == ("FLUX", "NAME")
            f[1].delete_column(-1)
            assert f[1].colnames == ("FLUX",)
        with rustfits.FITS(fn) as f:
            assert f[1].colnames == ("FLUX",)


# ---------------------------------------------------------------------------
# Round-trip insert → delete restores layout
# ---------------------------------------------------------------------------


def test_round_trip_insert_then_delete_restores_layout():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        orig = _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            naxis1_before = f[1].header["NAXIS1"]
            tbcols_before = [
                f[1].header[f"TBCOL{i+1}"] for i in range(f[1].ncols)
            ]
            f[1].insert_column(
                "TMP",
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="f8"),
                position=1)
            assert f[1].ncols == 4
            f[1].delete_column("TMP")
            assert f[1].ncols == 3
            assert f[1].header["NAXIS1"] == naxis1_before
            tbcols_after = [
                f[1].header[f"TBCOL{i+1}"] for i in range(f[1].ncols)
            ]
            assert tbcols_after == tbcols_before
            arr = f[1].read()
            np.testing.assert_array_equal(arr["ID"], orig["ID"])
            np.testing.assert_allclose(arr["FLUX"], orig["FLUX"])


# ---------------------------------------------------------------------------
# Non-last HDU shifts later HDUs
# ---------------------------------------------------------------------------


def test_insert_non_last_hdu_shifts_later_hdus():
    """Insert into a table that's not the last HDU on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_ascii_table_hdu(_basic_dtype(), nrows=5)
            f[1].write(_basic_data())
            # Add a later image HDU we'll need to remain readable.
            f.create_image_hdu("f4", (4, 4))
            sentinel_img = (np.arange(16, dtype="f4") * 0.25).reshape(4, 4)
            f[2].write(sentinel_img)

        with rustfits.FITS(fn, "r+") as f:
            f[1].insert_column(
                "ERR",
                np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="f8"),
                after="FLUX")
            # Later HDU still readable + correct
            np.testing.assert_allclose(
                f[2].read(),
                (np.arange(16, dtype="f4") * 0.25).reshape(4, 4))
        with rustfits.FITS(fn) as f:
            # After reopen
            np.testing.assert_allclose(
                f[2].read(),
                (np.arange(16, dtype="f4") * 0.25).reshape(4, 4))
            assert f[1].colnames == ("ID", "FLUX", "ERR", "NAME")


def test_delete_non_last_hdu_shifts_later_hdus():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (1,))
            f.create_ascii_table_hdu(_basic_dtype(), nrows=5)
            f[1].write(_basic_data())
            f.create_image_hdu("f4", (4, 4))
            img = (np.arange(16, dtype="f4") * 0.5).reshape(4, 4)
            f[2].write(img)

        with rustfits.FITS(fn, "r+") as f:
            f[1].delete_column("FLUX")
            np.testing.assert_allclose(f[2].read(), img)
        with rustfits.FITS(fn) as f:
            np.testing.assert_allclose(f[2].read(), img)
            assert f[1].colnames == ("ID", "NAME")


# ---------------------------------------------------------------------------
# Cross-tool
# ---------------------------------------------------------------------------


def test_astropy_reads_inserted_table():
    astropy_fits = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        new_col = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype="f4")
        with rustfits.FITS(fn, "r+") as f:
            f[1].insert_column("EXTRA", new_col, after="FLUX")
        with astropy_fits.open(fn) as hdul:
            d = hdul[1].data
            np.testing.assert_allclose(d["EXTRA"], new_col, rtol=1e-6)


def test_astropy_reads_deleted_table():
    astropy_fits = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        orig = _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            f[1].delete_column("FLUX")
        with astropy_fits.open(fn) as hdul:
            d = hdul[1].data
            np.testing.assert_array_equal(d["ID"], orig["ID"])
            assert "FLUX" not in d.dtype.names


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_insert_duplicate_name_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "ID", np.array([1, 2, 3, 4, 5], dtype="i8"))


def test_insert_duplicate_name_case_insensitive_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "id", np.array([1, 2, 3, 4, 5], dtype="i8"))


def test_insert_empty_name_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "", np.array([1, 2, 3, 4, 5], dtype="i8"))


def test_insert_position_out_of_range_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "X", np.array([1, 2, 3, 4, 5], dtype="i8"),
                    position=99)


def test_insert_multiple_location_kwargs_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "X", np.array([1, 2, 3, 4, 5], dtype="i8"),
                    position=0, after="ID")


def test_insert_object_dtype_rejected():
    """ASCII tables don't support VLA — Object dtype is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        obj_data = np.empty(5, dtype="O")
        for i in range(5):
            obj_data[i] = np.array([i, i + 1, i + 2], dtype="i4")
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column("VLA", obj_data)


def test_insert_wrong_length_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "X", np.array([1, 2, 3], dtype="i8"))  # only 3 rows


def test_insert_wrong_shape_rejected():
    """2-D input rejected (ASCII columns are scalar)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].insert_column(
                    "X", np.zeros((5, 3), dtype="i8"))


def test_delete_name_not_found_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].delete_column("BOGUS")


def test_delete_index_out_of_range_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _make_table(fn)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError):
                f[1].delete_column(99)
