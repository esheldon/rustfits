"""
Tests: dict + list+names input forms for TableHDU.write,
plus structured-array field-order normalization.

Three input forms are accepted:
  - Structured ndarray (existing) — fast path when fields are in HDU
    order, slow path otherwise (Phase 1e adds the reorder branch).
  - Dict of {name: ndarray} — slow path; each column has its own
    buffer.
  - List/tuple of ndarrays + names=[...] — same as dict.

Each input form must round-trip through both same-handle and post-
reopen reads (per CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- structured array, reordered --------------------


def test_structured_field_order_differs_from_hdu():
    """
    Input dtype has the same names as the HDU but in a different
    order — the slow path repacks per column, so this round-trips
    correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        hdu_dt = np.dtype([("id", "i4"), ("flux", "f8"), ("snr", "f4")])
        # Input dtype: same names, reversed order.
        arr_dt = np.dtype([("snr", "f4"), ("flux", "f8"), ("id", "i4")])
        arr = np.zeros(5, dtype=arr_dt)
        arr["id"] = np.arange(5)
        arr["flux"] = np.arange(5) * 2.5
        arr["snr"] = np.arange(5) * 0.5

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(hdu_dt, nrows=5)
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["flux"], arr["flux"])
            np.testing.assert_array_equal(got["snr"], arr["snr"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["flux"], arr["flux"])
            np.testing.assert_array_equal(got["snr"], arr["snr"])


def test_structured_missing_field_rejected():
    """
    Input dtype is missing one of the HDU's columns — rejected up
    front rather than writing a partial table.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        hdu_dt = np.dtype([("id", "i4"), ("flux", "f8")])
        arr_dt = np.dtype([("id", "i4")])
        arr = np.zeros(3, dtype=arr_dt)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(hdu_dt, nrows=3)
            with pytest.raises(ValueError, match="(missing field|input has)"):
                fits[1].write(arr)


def test_structured_extra_field_rejected():
    """
    Input dtype has a field not present in the HDU — count mismatch
    triggers rejection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        hdu_dt = np.dtype([("id", "i4")])
        arr_dt = np.dtype([("id", "i4"), ("extra", "f8")])
        arr = np.zeros(3, dtype=arr_dt)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(hdu_dt, nrows=3)
            with pytest.raises(ValueError, match="input has"):
                fits[1].write(arr)


# -------------------- dict input --------------------


def test_dict_input_basic():
    """
    Basic dict form: {colname: 1-D ndarray}.  All HDU columns must
    be present.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8"), ("snr", "f4")])
        n = 6
        data = {
            "id": np.arange(n, dtype="i4"),
            "flux": np.arange(n, dtype="f8") * 1.5,
            "snr": np.arange(n, dtype="f4") - 1.0,
        }

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(data)
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], data["id"])
            np.testing.assert_array_equal(got["flux"], data["flux"])
            np.testing.assert_array_equal(got["snr"], data["snr"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], data["id"])
            np.testing.assert_array_equal(got["flux"], data["flux"])


def test_dict_input_missing_column_rejected():
    """
    Every HDU column must be in the dict.  Missing 'flux' → reject.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="missing column"):
                fits[1].write({"id": np.arange(3, dtype="i4")})


def test_dict_input_extra_key_rejected():
    """
    A dict key not in the HDU is rejected as a typo guard.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="extra key"):
                fits[1].write(
                    {
                        "id": np.arange(3, dtype="i4"),
                        "stray": np.zeros(3, dtype="f4"),
                    }
                )


def test_dict_input_with_subarray_column():
    """
    Dict input with a subarray column: each value is a (nrows, ...)
    ndarray with the per-cell shape matching the column's TDIM-
    derived shape.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("img", "f4", (3, 4))])
        n = 4
        data = {
            "id": np.arange(n, dtype="i4"),
            "img": np.arange(n * 12, dtype="f4").reshape(n, 3, 4),
        }

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(data)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], data["id"])
            assert got["img"].shape == (n, 3, 4)
            np.testing.assert_array_equal(got["img"], data["img"])


def test_dict_input_with_string_column():
    """
    Dict input with a U string column — exercises the per-column
    UnicodeToAscii transform through the slow path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("name", "U8"), ("flux", "f4")])
        n = 3
        data = {
            "name": np.array(["alpha", "beta", "gamma"], dtype="U8"),
            "flux": np.array([1.5, 2.5, 3.5], dtype="f4"),
        }

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(data)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["name"], data["name"])
            np.testing.assert_array_equal(got["flux"], data["flux"])


def test_dict_input_wrong_length_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=5)
            with pytest.raises(ValueError, match="first axis"):
                fits[1].write({"id": np.arange(2, dtype="i4")})


def test_dict_input_wrong_dtype_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("flux", "f4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="expected"):
                fits[1].write({"flux": np.arange(3, dtype="f8")})


# -------------------- list + names input --------------------


def test_list_names_input_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8")])
        n = 4
        id_arr = np.arange(n, dtype="i4")
        flux_arr = np.arange(n, dtype="f8") * 1.25

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write([id_arr, flux_arr], names=["id", "flux"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], id_arr)
            np.testing.assert_array_equal(got["flux"], flux_arr)


def test_list_names_input_reordered():
    """
    Names can be in any order; columns are looked up by name (not by
    position).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f8")])
        n = 4
        id_arr = np.arange(n, dtype="i4")
        flux_arr = np.arange(n, dtype="f8") + 10

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            # Pass flux first, then id.
            fits[1].write([flux_arr, id_arr], names=["flux", "id"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], id_arr)
            np.testing.assert_array_equal(got["flux"], flux_arr)


def test_list_names_tuple_form():
    """
    A tuple of arrays is also accepted; same handling as a list.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        n = 3
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write((np.arange(n, dtype="i4"),), names=("id",))

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], np.arange(n))


def test_list_names_missing_names_kwarg_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="names="):
                fits[1].write([np.arange(3, dtype="i4")])


def test_list_names_length_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="len"):
                fits[1].write(
                    [np.arange(3, dtype="i4")],
                    names=["id", "extra"],
                )


def test_list_names_duplicate_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4"), ("flux", "f4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            arr = np.zeros(3, dtype="i4")
            with pytest.raises(ValueError, match="duplicate"):
                fits[1].write([arr, arr], names=["id", "id"])


# -------------------- names= forbidden with other forms --------------------


def test_names_rejected_with_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="names="):
                fits[1].write({"id": np.arange(3, dtype="i4")}, names=["id"])


def test_names_rejected_with_structured_array():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("id", "i4")])
        arr = np.zeros(3, dtype=dt)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            with pytest.raises(ValueError, match="names="):
                fits[1].write(arr, names=["id"])


# -------------------- input forms produce identical files --------------------


def test_three_input_forms_produce_same_bytes():
    """
    All three input forms must produce byte-identical FITS data
    sections for the same logical content.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dt = np.dtype([("id", "i4"), ("flux", "f8"), ("name", "U6")])
        n = 4
        id_arr = np.arange(n, dtype="i4") + 100
        flux_arr = np.arange(n, dtype="f8") * 0.25
        name_arr = np.array(["alpha", "beta", "gamma", "delta"], dtype="U6")

        # Form 1: structured array
        struct = np.zeros(n, dtype=dt)
        struct["id"] = id_arr
        struct["flux"] = flux_arr
        struct["name"] = name_arr
        f1 = os.path.join(tmpdir, "struct.fits")
        with rustfits.FITS(f1, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(struct)

        # Form 2: dict
        f2 = os.path.join(tmpdir, "dict.fits")
        with rustfits.FITS(f2, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(
                {
                    "id": id_arr,
                    "flux": flux_arr,
                    "name": name_arr,
                }
            )

        # Form 3: list + names
        f3 = os.path.join(tmpdir, "list.fits")
        with rustfits.FITS(f3, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(
                [flux_arr, name_arr, id_arr],
                names=["flux", "name", "id"],
            )

        # Compare data sections.
        b1 = open(f1, "rb").read()
        b2 = open(f2, "rb").read()
        b3 = open(f3, "rb").read()
        assert b1 == b2 == b3, (
            "three input forms produced different file bytes"
        )
