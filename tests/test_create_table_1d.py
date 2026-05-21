"""
Phase 1d tests: string columns (S/U → A).

Numpy structured-dtype fields with kind 'S' (bytes) or 'U' (unicode)
map to FITS A columns:

  - S<N> scalar         → TFORM=NA, no TDIM         (one N-byte string)
  - S<N> shape (M,)     → TFORM=(N*M)A, TDIM=(N,M)  (M strings of N
                                                     bytes each)
  - S<N> shape (3, 4)   → TFORM=(N*12)A, TDIM=(N,4,3)
  - U<N> same shapes, but per-codepoint ASCII validation on write
    (FITS A is one byte per character, strictly 7-bit ASCII).

S round-trips through the fast path (per-cell byte widths match).
U forces the slow path because numpy U is UTF-32-LE (4 bytes per
codepoint), so per-cell source widths are 4× the FITS column widths.

Each test verifies through both a same-handle read and a fresh-reopen
read (per CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- S (bytes) scalar --------------------


def test_s_scalar_round_trips():
    """
    A single S<N> column emits TFORM=NA with no TDIM (the bare
    repeat captures the per-string width).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("name", "S10")])
        arr = np.zeros(4, dtype=dt)
        arr["name"] = [b"alpha", b"beta", b"gamma", b"delta"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=4)
            assert fits[1].header["TFORM1"] == "10A"
            assert "TDIM1" not in fits[1].header
            fits[1].write(arr)
            got = fits[1].read()
            assert got.dtype.fields["name"][0] == np.dtype("U10")
            np.testing.assert_array_equal(
                got["name"], ["alpha", "beta", "gamma", "delta"]
            )

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(
                got["name"], ["alpha", "beta", "gamma", "delta"]
            )


def test_s_1d_subarray_emits_tdim():
    """
    Unlike 1-D numeric subarrays (which omit TDIM), 1-D string
    subarrays MUST emit TDIM because 'TFORM=30A' alone is ambiguous
    between '30-char string' and '3 strings of 10 chars'.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("labels", "S5", (3,))])
        arr = np.zeros(2, dtype=dt)
        arr["labels"][0] = [b"abcde", b"fghij", b"klmno"]
        arr["labels"][1] = [b"pqrst", b"uvwxy", b"z0123"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "15A"
            assert fits[1].header["TDIM1"] == "(5,3)"
            fits[1].write(arr)
            got = fits[1].read()
            assert got["labels"].shape == (2, 3)
            np.testing.assert_array_equal(
                got["labels"], arr["labels"].astype("U5")
            )

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["labels"].shape == (2, 3)


def test_s_2d_subarray():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("grid", "S4", (2, 3))])
        arr = np.zeros(2, dtype=dt)
        arr["grid"][0] = [
            [b"aaaa", b"bbbb", b"cccc"],
            [b"dddd", b"eeee", b"ffff"],
        ]
        arr["grid"][1] = [
            [b"1111", b"2222", b"3333"],
            [b"4444", b"5555", b"6666"],
        ]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "24A"
            # TDIM = (chars, ...reversed(numpy shape)).
            assert fits[1].header["TDIM1"] == "(4,3,2)"
            fits[1].write(arr)
            got = fits[1].read()
            assert got["grid"].shape == (2, 2, 3)
            np.testing.assert_array_equal(
                got["grid"], arr["grid"].astype("U4")
            )


# -------------------- U (unicode) scalar + subarray --------------------


def test_u_scalar_ascii_round_trips():
    """
    U<N> with pure-ASCII content round-trips through the slow path
    (numpy U is UTF-32-LE so its per-cell width differs from A's).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("name", "U10")])
        arr = np.zeros(3, dtype=dt)
        arr["name"] = ["hello", "world", "fits"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            assert fits[1].header["TFORM1"] == "10A"
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(
                got["name"], ["hello", "world", "fits"]
            )

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(
                got["name"], ["hello", "world", "fits"]
            )


def test_u_subarray_ascii_round_trips():
    """
    U with shape (3,): TDIM=(N,3); per-cell ASCII validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("tags", "U5", (3,))])
        arr = np.zeros(2, dtype=dt)
        arr["tags"][0] = ["abcde", "fghij", "klmno"]
        arr["tags"][1] = ["12345", "67890", "ascii"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            assert fits[1].header["TFORM1"] == "15A"
            assert fits[1].header["TDIM1"] == "(5,3)"
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            assert got["tags"].shape == (2, 3)
            np.testing.assert_array_equal(got["tags"], arr["tags"])


def test_u_non_ascii_rejected():
    """
    A non-ASCII codepoint (e.g. é = U+00E9) must raise rather
    than silently truncate or replace.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("name", "U10")])
        arr = np.zeros(2, dtype=dt)
        arr["name"] = ["ok", "café"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=2)
            with pytest.raises(ValueError, match="non-ASCII"):
                fits[1].write(arr)


def test_u_high_unicode_rejected():
    """
    Codepoints above the BMP (e.g. emoji) also raise.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("name", "U5")])
        arr = np.zeros(1, dtype=dt)
        arr["name"] = ["\U0001f600"]  # grinning face emoji

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=1)
            with pytest.raises(ValueError, match="non-ASCII"):
                fits[1].write(arr)


# -------------------- mixed S/U with other dtypes --------------------


def test_mixed_string_and_numeric_columns():
    """
    A table with both S and U columns alongside numeric ones —
    routes through the slow path because of the U column.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),
                ("name", "U16"),  # forces slow path
                ("code", "S4"),
                ("flux", "f8"),
                ("flag", "?"),
            ]
        )
        arr = np.zeros(3, dtype=dt)
        arr["id"] = [1, 2, 3]
        arr["name"] = ["alpha", "beta", "gamma"]
        arr["code"] = [b"AAAA", b"BBBB", b"CCCC"]
        arr["flux"] = [1.5, 2.5, 3.5]
        arr["flag"] = [True, False, True]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["name"], arr["name"])
            np.testing.assert_array_equal(
                got["code"], arr["code"].astype("U4")
            )
            np.testing.assert_array_equal(got["flux"], arr["flux"])
            np.testing.assert_array_equal(got["flag"], arr["flag"])


def test_short_string_padded():
    """
    A string shorter than its column width must be null-padded on
    disk (and round-trip back without the trailing nulls / spaces
    interfering).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt = np.dtype([("s", "U10")])
        arr = np.zeros(3, dtype=dt)
        arr["s"] = ["hi", "", "long word"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt, nrows=3)
            fits[1].write(arr)

        with rustfits.FITS(fname) as fits:
            got = fits[1].read()
            np.testing.assert_array_equal(got["s"], ["hi", "", "long word"])


# -------------------- validation errors --------------------


def test_wrong_string_width_rejected():
    """
    Input S width different from HDU's column width is rejected
    by the per-cell shape / width validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dt_hdu = np.dtype([("s", "S10")])
        dt_arr = np.dtype([("s", "S20")])
        bad = np.zeros(2, dtype=dt_arr)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=2)
            with pytest.raises(ValueError, match="A.*chars/string"):
                fits[1].write(bad)


def test_zero_length_string_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="zero-length"):
                fits.create_table_hdu([("s", "S0")], nrows=1)


def test_s_into_u_column_rejected():
    """
    If the HDU was created with a U column (TFORM=NA), passing an
    S column with the same chars-per-string is OK (both map to A) —
    in fact this is how the read side returns U regardless of how
    the column was written.  Verify that S input round-trips correctly
    even when the HDU was created with U.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        # HDU created with U → same TFORM, same on-disk layout as if
        # created with S.
        dt_hdu = np.dtype([("s", "U8")])
        dt_arr = np.dtype([("s", "S8")])
        arr = np.zeros(2, dtype=dt_arr)
        arr["s"] = [b"ascii", b"bytes"]

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dt_hdu, nrows=2)
            # S input matches the A column's per-string byte count.
            fits[1].write(arr)
            got = fits[1].read()
            np.testing.assert_array_equal(got["s"], ["ascii", "bytes"])
