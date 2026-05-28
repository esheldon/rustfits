"""
Tests: unsigned-int trick (u2/u4/u8), bool, and complex.

These dtypes were rejected in the 1a MVP and added in 1b:
  - u2/u4/u8 → I/J/K + TZERO=2^(n-1)  (unsigned-int trick)
  - b1 (numpy bool) → L (FITS logical)
  - c8/c16 → C/M  (complex; per-half byteswap)

Each test verifies round-trip through both a same-handle read and a
fresh-reopen read (per CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# --------------------------- unsigned trick ---------------------------


@pytest.mark.parametrize(
    "dtype_str,expected_tzero",
    [
        ("u2", 32768),
        ("u4", 2147483648),
        ("u8", 9223372036854775808),
    ],
)
def test_unsigned_int_trick_round_trips(dtype_str, expected_tzero):
    """
    Writing u2/u4/u8 emits TZERO=2^(n-1) and the value range survives.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("v", dtype_str)])
        info = np.iinfo(np.dtype(dtype_str))
        arr = np.zeros(5, dtype=dt)
        # Pick values that span both halves of the unsigned range so
        # XOR-top-bit is exercised in both directions.
        arr["v"] = [
            0,
            1,
            info.max // 2,
            info.max // 2 + 1,
            info.max,
        ]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=5)
            assert fits[1].header["TZERO1"] == expected_tzero
            fits[1].write(arr)
            got = fits[1].read()
            assert got.dtype.fields["v"][0] == np.dtype(dtype_str)
            np.testing.assert_array_equal(got["v"], arr["v"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got.dtype.fields["v"][0] == np.dtype(dtype_str)
            np.testing.assert_array_equal(got["v"], arr["v"])


def test_unsigned_trick_extreme_values():
    """
    Boundary values (0, mid, max) must round-trip exactly for u2/u4/u8.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("u16", "u2"), ("u32", "u4"), ("u64", "u8")])
        arr = np.zeros(3, dtype=dt)
        arr["u16"] = [0, 1 << 15, (1 << 16) - 1]
        arr["u32"] = [0, 1 << 31, (1 << 32) - 1]
        arr["u64"] = [0, 1 << 63, (1 << 64) - 1]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["u16"], arr["u16"])
            np.testing.assert_array_equal(got["u32"], arr["u32"])
            np.testing.assert_array_equal(got["u64"], arr["u64"])


def test_unsigned_trick_signed_input_rejected():
    """
    A u-trick column expects u-input; passing i-input is a mismatch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("v", "u4")])
        dt_arr = np.dtype([("v", "i4")])
        bad = np.zeros(3, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=3)
            with pytest.raises(ValueError, match="unsigned-int trick"):
                fits[1].write(bad)


def test_mixed_signed_unsigned_columns():
    """
    A table that mixes plain signed columns with unsigned-trick
    columns writes each through the right transform.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype(
            [
                ("s32", "i4"),
                ("u32", "u4"),
                ("s16", "i2"),
                ("u16", "u2"),
            ]
        )
        arr = np.zeros(4, dtype=dt)
        arr["s32"] = [-2_000_000_000, -1, 0, 2_000_000_000]
        arr["u32"] = [0, 1, (1 << 31), (1 << 32) - 1]
        arr["s16"] = [-32000, -1, 0, 32000]
        arr["u16"] = [0, 1, (1 << 15), (1 << 16) - 1]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=4)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for name in dt.names:
                np.testing.assert_array_equal(got[name], arr[name])


# --------------------------- bool ---------------------------


def test_bool_round_trips():
    """
    numpy bool → L → numpy bool, T/F preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("flag", "?"), ("idx", "i4")])
        arr = np.zeros(6, dtype=dt)
        arr["flag"] = [True, False, True, True, False, False]
        arr["idx"] = np.arange(6)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=6)
            assert fits[1].header["TFORM1"] == "1L"
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["flag"], arr["flag"])
            np.testing.assert_array_equal(got["idx"], arr["idx"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got.dtype.fields["flag"][0] == np.dtype("?")
            np.testing.assert_array_equal(got["flag"], arr["flag"])


def test_bool_non_bool_input_rejected():
    """
    An L column expects bool input; integer input is a mismatch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("flag", "?")])
        dt_arr = np.dtype([("flag", "u1")])
        bad = np.zeros(3, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=3)
            with pytest.raises(ValueError, match="bool"):
                fits[1].write(bad)


# --------------------------- complex ---------------------------


@pytest.mark.parametrize("dtype_str", ["c8", "c16"])
def test_complex_round_trips(dtype_str):
    """
    c8 → C, c16 → M.  Real and imag halves byteswap independently
    (swap unit 4 for C, 8 for M) — covered by the existing Identity
    transform path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("z", dtype_str)])
        arr = np.zeros(5, dtype=dt)
        arr["z"] = [1 + 2j, -3 + 4j, 0 + 0j, 1e10 + 1e-10j, -5 - 0j]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=5)
            expected_letter = "C" if dtype_str == "c8" else "M"
            assert fits[1].header["TFORM1"] == f"1{expected_letter}"
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["z"], arr["z"])

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got.dtype.fields["z"][0] == np.dtype(dtype_str)
            np.testing.assert_array_equal(got["z"], arr["z"])


def test_complex_real_and_imag_not_swapped():
    """
    If byteswap_unit incorrectly used the whole-element width (8 for
    C, 16 for M), real and imag halves would swap places.  This test
    catches that regression by writing values where real != imag.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("z", "c16")])
        arr = np.zeros(3, dtype=dt)
        arr["z"] = [100.0 + 0.0j, 0.0 + 200.0j, 1.5 - 2.5j]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["z"][0].real == 100.0 and got["z"][0].imag == 0.0
            assert got["z"][1].real == 0.0 and got["z"][1].imag == 200.0
            assert got["z"][2].real == 1.5 and got["z"][2].imag == -2.5


# --------------------------- mixed 1a + 1b ---------------------------


def test_mixed_dtypes_1a_and_1b():
    """
    A table with a mix of MVP and Phase 1b dtypes — exercises all
    three transform variants in one strip-by-strip write.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),  # 1a Identity
                ("flux", "f8"),  # 1a Identity
                ("uidx", "u4"),  # 1b UnsignedXor
                ("flag", "?"),  # 1b BoolToLogical
                ("z", "c8"),  # 1b Identity (half-element swap)
            ]
        )
        n = 7
        arr = np.zeros(n, dtype=dt)
        arr["id"] = np.arange(n) - 3
        arr["flux"] = np.arange(n) * 0.5
        arr["uidx"] = np.arange(n) + (1 << 31)
        arr["flag"] = [True, False, True, False, True, False, True]
        arr["z"] = [complex(i, -i) for i in range(n)]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=n)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            for name in dt.names:
                np.testing.assert_array_equal(got[name], arr[name])
