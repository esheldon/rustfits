"""
Tests for `hdu[5]` returning a single record from a TableHDU.

`hdu[i]` is the natural Python-ism that matches `structured_arr[i]`:
returns a 0-d numpy record (np.void), with field access yielding
scalars / ndarrays / inner ndarrays as appropriate.  The length-1
form `hdu[[5]]` or `hdu[5:6]` still returns a shape-(1,) structured
array — those existing paths are unaffected.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

import rustfits


CARDS_PER_BLOCK = 36
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b):
    return b + b"\x00" * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    with open(path, "wb") as f:
        for cards, data in parts:
            f.write(_pad_cards(cards))
            if data:
                f.write(_pad_to_block(data))


def _primary_no_data():
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]


def _bintable_ext(naxis1, naxis2, fields, extras=()):
    cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {naxis1:>20d}",
        f"NAXIS2  = {naxis2:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        f"TFIELDS = {len(fields):>20d}",
    ]
    for i, (ttype, tform, *opt) in enumerate(fields, start=1):
        cards.append(f"TTYPE{i:<3d}= '{ttype:<8s}'")
        cards.append(f"TFORM{i:<3d}= '{tform:<8s}'")
        if opt:
            cards.append(f"TDIM{i:<4d}= '{opt[0]:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


# ---------------------------------------------------------------------------
# Basic single-row return
# ---------------------------------------------------------------------------


def test_single_row_returns_np_void():
    """hdu[i] returns a numpy 0-d record (np.void), not an array."""
    fields = [("X", "1J"), ("Y", "1D")]
    rows_data = b""
    for i in range(3):
        rows_data += struct.pack(">id", 10 + i, 1.5 + i)
    cards = _bintable_ext(12, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            rec = fits[1][1]
            assert isinstance(rec, np.void)
            # Field access yields a scalar.
            assert int(rec["X"]) == 11
            assert float(rec["Y"]) == pytest.approx(2.5)


def test_single_row_first_and_last():
    """hdu[0] and hdu[-1] reach the right rows."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">5i", 10, 20, 30, 40, 50)
    cards = _bintable_ext(4, 5, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            assert int(fits[1][0]["X"]) == 10
            assert int(fits[1][-1]["X"]) == 50
            assert int(fits[1][-5]["X"]) == 10


def test_single_row_out_of_range_raises():
    """Out-of-range int raises (via resolve_rows)."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">3i", 1, 2, 3)
    cards = _bintable_ext(4, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            with pytest.raises((ValueError, IndexError)):
                _ = fits[1][3]   # NAXIS2 == 3, valid indices 0..2
            with pytest.raises((ValueError, IndexError)):
                _ = fits[1][-4]


# ---------------------------------------------------------------------------
# Non-integer keys are still rejected; the new path doesn't break them
# ---------------------------------------------------------------------------


def test_single_row_bool_rejected():
    """
    Bool key must NOT silently route to int (Python bool is a
    subclass of int)."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">i", 0)
    cards = _bintable_ext(4, 1, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            with pytest.raises((ValueError, TypeError)):
                _ = fits[1][True]


def test_single_row_float_rejected():
    """Float key is not an integer index; reject."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">i", 0)
    cards = _bintable_ext(4, 1, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            with pytest.raises((ValueError, TypeError)):
                _ = fits[1][0.0]


# ---------------------------------------------------------------------------
# Array / VLA / TDIM column fields on a single record
# ---------------------------------------------------------------------------


def test_single_row_array_column_field_shape():
    """
    A repeat>1 column's field on a single record is a numpy
    ndarray of the cell shape."""
    fields = [("VEC", "3J")]
    rows_data = struct.pack(">6i", 1, 2, 3, 10, 20, 30)
    cards = _bintable_ext(12, 2, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            rec = fits[1][1]
            vec = rec["VEC"]
            assert vec.shape == (3,)
            assert vec.tolist() == [10, 20, 30]


def test_single_row_tdim_column_field_shape():
    """TDIM-reshaped column field has the reversed-TDIM (numpy) shape."""
    fields = [("M", "6J", "(3,2)")]
    rows_data = struct.pack(">6i", 1, 2, 3, 4, 5, 6)
    cards = _bintable_ext(24, 1, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            rec = fits[1][0]
            m = rec["M"]
            # TDIM=(3,2) FITS axis order → numpy shape (2, 3).
            assert m.shape == (2, 3)


def test_single_row_vla_column_field():
    """
    A VLA column field on a single record yields the inner ndarray
    (the Object cell)."""
    # Build a 1PJ column with two rows, varying heap-cell sizes.
    descriptors = struct.pack(">ii", 3, 0) + struct.pack(">ii", 2, 12)
    heap = struct.pack(">iii", 1, 2, 3) + struct.pack(">ii", 100, 200)
    fields = [("V", "1PJ(10)")]
    cards = _bintable_ext(8, 2, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""),
                    (cards, descriptors + heap))
        with rustfits.FITS(fname) as fits:
            rec0 = fits[1][0]
            rec1 = fits[1][1]
            v0 = rec0["V"]
            v1 = rec1["V"]
            assert isinstance(v0, np.ndarray)
            assert v0.tolist() == [1, 2, 3]
            assert v1.tolist() == [100, 200]


# ---------------------------------------------------------------------------
# Length-1 forms are unaffected — still return shape-(1,) structured arrays
# ---------------------------------------------------------------------------


def test_length_one_list_still_returns_array():
    """
    hdu[[5]] (length-1 list) still goes through the Rows path and
    returns a shape-(1,) structured array, not np.void.  Confirms the
    new single-int path doesn't accidentally hijack iterables."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">3i", 10, 20, 30)
    cards = _bintable_ext(4, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            arr = fits[1][[1]]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (1,)
            assert arr["X"].tolist() == [20]


def test_length_one_slice_still_returns_array():
    """hdu[5:6] still returns a shape-(1,) structured array."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">3i", 10, 20, 30)
    cards = _bintable_ext(4, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            arr = fits[1][1:2]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (1,)
            assert arr["X"].tolist() == [20]


# ---------------------------------------------------------------------------
# numpy integer types should work (np.int64, np.int32, ...)
# ---------------------------------------------------------------------------


def test_single_row_accepts_numpy_int():
    """
    numpy integer scalars are accepted as the index, returning the
    same np.void as a Python int would."""
    fields = [("X", "1J")]
    rows_data = struct.pack(">3i", 10, 20, 30)
    cards = _bintable_ext(4, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, rows_data))
        with rustfits.FITS(fname) as fits:
            rec = fits[1][np.int64(1)]
            assert isinstance(rec, np.void)
            assert int(rec["X"]) == 20
