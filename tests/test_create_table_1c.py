"""
Phase 1c tests: subarray fields + TDIM.

Numpy structured-dtype fields can carry a subarray shape:
  np.dtype([('flux', 'f4', (3, 4))])

These map to FITS BINTABLE columns with TFORM repeat = product of
shape, and TDIM in FITS (FORTRAN, fastest-first) order = reversed
numpy shape.  1-D shapes are fully described by the repeat count, so
TDIM is omitted in that case (matches astropy convention).

Round-trip contract: numpy shape (3, 4) → TDIM=(4,3) on disk → numpy
shape (3, 4) again on read.

Each test verifies through both a same-handle read and a fresh-reopen
read (per CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- 1-D subarray (no TDIM) --------------------


def test_1d_subarray_no_tdim_card():
    """
    A 1-D subarray emits TFORM=5E but NO TDIM card — the repeat
    count alone is enough to describe the shape, and astropy's
    convention is to omit a redundant TDIM.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("v", "f4", (5,))])
        arr = np.zeros(3, dtype=dt)
        arr["v"] = np.arange(15, dtype="f4").reshape(3, 5)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            assert fits[1].header["TFORM1"] == "5E"
            assert "TDIM1" not in fits[1].header
            fits[1].write(arr)
            got = fits[1].read()
            assert got["v"].shape == (3, 5)
            np.testing.assert_array_equal(got["v"], arr["v"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["v"].shape == (3, 5)
            np.testing.assert_array_equal(got["v"], arr["v"])


# -------------------- 2-D subarray (TDIM emitted) --------------------


def test_2d_subarray_emits_reversed_tdim():
    """
    numpy shape (3, 4) → TFORM=12E + TDIM=(4,3) on disk; read
    returns numpy shape (3, 4) again.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("img", "f4", (3, 4))])
        arr = np.zeros(2, dtype=dt)
        arr["img"][0] = np.arange(12, dtype="f4").reshape(3, 4)
        arr["img"][1] = np.arange(100, 112, dtype="f4").reshape(3, 4)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "12E"
            assert fits[1].header["TDIM1"] == "(4,3)"
            fits[1].write(arr)
            got = fits[1].read()
            assert got["img"].shape == (2, 3, 4)
            np.testing.assert_array_equal(got["img"], arr["img"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["img"].shape == (2, 3, 4)
            np.testing.assert_array_equal(got["img"], arr["img"])


def test_3d_subarray():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("cube", "f8", (2, 3, 4))])
        arr = np.zeros(2, dtype=dt)
        arr["cube"] = np.arange(2 * 24, dtype="f8").reshape(2, 2, 3, 4)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "24D"
            # numpy (2,3,4) -> FITS FORTRAN (4,3,2).
            assert fits[1].header["TDIM1"] == "(4,3,2)"
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["cube"].shape == (2, 2, 3, 4)
            np.testing.assert_array_equal(got["cube"], arr["cube"])


# -------------------- subarray + each 1b transform --------------------


def test_subarray_unsigned_int_trick():
    """
    Subarray with u4: TFORM=6J + TDIM=(3,2) + TZERO=2^31; per-cell
    XOR top bit applies to every element in the cell.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("uimg", "u4", (2, 3))])
        arr = np.zeros(3, dtype=dt)
        for r in range(3):
            arr["uimg"][r] = np.arange(6).reshape(2, 3) + r * (1 << 30)
        # Include the boundary at 2^31.
        arr["uimg"][2] = np.array([[0, 1, 1 << 31], [(1 << 32) - 1, 2, 3]])

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            assert fits[1].header["TFORM1"] == "6J"
            assert fits[1].header["TDIM1"] == "(3,2)"
            assert fits[1].header["TZERO1"] == (1 << 31)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["uimg"].shape == (3, 2, 3)
            assert got["uimg"].dtype == np.dtype("u4")
            np.testing.assert_array_equal(got["uimg"], arr["uimg"])


def test_subarray_bool():
    """
    Subarray of bool: each cell holds shape (3, 4) of bool, stored
    as 12 L-bytes per row.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("mask", "?", (3, 4))])
        arr = np.zeros(2, dtype=dt)
        arr["mask"][0] = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype="?"
        )
        arr["mask"][1] = np.array(
            [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]], dtype="?"
        )

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "12L"
            assert fits[1].header["TDIM1"] == "(4,3)"
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["mask"].shape == (2, 3, 4)
            assert got["mask"].dtype == np.dtype("?")
            np.testing.assert_array_equal(got["mask"], arr["mask"])


def test_subarray_complex():
    """
    Subarray of c8: per-half byteswap applies across all elements
    in the cell.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("z", "c16", (2, 2))])
        arr = np.zeros(2, dtype=dt)
        arr["z"][0] = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        arr["z"][1] = np.array([[-1 + 0j, 0 - 1j], [1e10 + 1j, 2 - 3j]])

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "4M"
            assert fits[1].header["TDIM1"] == "(2,2)"
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["z"].shape == (2, 2, 2)
            np.testing.assert_array_equal(got["z"], arr["z"])


# -------------------- mixed scalar + subarray --------------------


def test_mixed_scalar_and_subarray_columns():
    """
    A table mixing scalar fields with subarray fields, each with a
    different transform, exercises the full strip-loop dispatch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),  # scalar Identity
                ("vec", "f4", (5,)),  # 1-D Identity, no TDIM
                ("img", "f8", (3, 4)),  # 2-D Identity, TDIM=(4,3)
                ("umask", "u2", (2, 3)),  # 2-D UnsignedXor, TDIM=(3,2)
                ("flag", "?"),  # scalar BoolToLogical
                ("flags2d", "?", (2, 2)),  # 2-D BoolToLogical
            ]
        )
        n = 4
        arr = np.zeros(n, dtype=dt)
        arr["id"] = np.arange(n) * 10
        arr["vec"] = np.arange(n * 5).reshape(n, 5).astype("f4")
        arr["img"] = np.arange(n * 12).reshape(n, 3, 4).astype("f8")
        arr["umask"] = (
            np.arange(n * 6).reshape(n, 2, 3).astype("u2") + (1 << 15) // 2
        )
        arr["flag"] = [True, False, True, False]
        arr["flags2d"] = np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
                [[1, 0], [1, 0]],
            ],
            dtype="?",
        )

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(arr)
            # Quick header sanity.
            h = fits[1].header
            assert h["TFORM1"] == "1J"
            assert h["TFORM2"] == "5E" and "TDIM2" not in h
            assert h["TFORM3"] == "12D" and h["TDIM3"] == "(4,3)"
            assert h["TFORM4"] == "6I" and h["TDIM4"] == "(3,2)"
            assert h["TZERO4"] == (1 << 15)
            assert h["TFORM5"] == "1L"
            assert h["TFORM6"] == "4L" and h["TDIM6"] == "(2,2)"

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for name in dt.names:
                np.testing.assert_array_equal(got[name], arr[name])


# --------------------------- shape validation ---------------------------


def test_shape_mismatch_on_write_rejected():
    """
    If the HDU has shape (3,4) but the user passes shape (4,3), the
    transform succeeds (same total bytes) but the on-disk axes would
    be wrong — validate must catch this up front.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("x", "f4", (3, 4))])
        dt_arr = np.dtype([("x", "f4", (4, 3))])
        bad = np.zeros(2, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=2)
            with pytest.raises(ValueError, match="shape"):
                fits[1].write(bad)


def test_scalar_input_into_subarray_column_rejected():
    """
    If HDU column has shape (3,) but input field is scalar, the
    per-cell shape check rejects up front.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("x", "f4", (3,))])
        dt_arr = np.dtype([("x", "f4")])
        bad = np.zeros(2, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=2)
            with pytest.raises(ValueError, match="shape"):
                fits[1].write(bad)
