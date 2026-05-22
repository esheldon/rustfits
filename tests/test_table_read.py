"""
Tests for BINTABLE read API (read + read_column).

We build BINTABLE files by hand at the byte level so we control exactly
what's on disk — no library's write-side cleansing interferes with
the corner-case tests (embedded null in A, non-ASCII byte in A, TDIM
reshape orientation, etc).
"""

import os
import struct
import tempfile

import numpy as np
import pytest

import rustfits


CARDS_PER_BLOCK = 36  # 2880 / 80
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b):
    return b + b"\x00" * ((BLOCK - len(b) % BLOCK) % BLOCK)


_PRIMARY = [
    "SIMPLE  =                    T",
    "BITPIX  =                    8",
    "NAXIS   =                    0",
    "EXTEND  =                    T",
    "END",
]


def _write_bintable(path, ext_cards, data_bytes):
    """Build a FITS file: minimal primary + given BINTABLE extension."""
    with open(path, "wb") as f:
        f.write(_pad_cards(_PRIMARY))
        f.write(_pad_cards(ext_cards))
        f.write(_pad_to_block(data_bytes))


def _bintable_cards(naxis1, naxis2, fields):
    """
    Minimal BINTABLE header cards.  `fields` is a list of
    (ttype, tform, optional tdim) tuples."""
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
    cards.append("END")
    return cards


# ---------------------------------------------------------------------------
# full-table round-trip for each TFORM type
# ---------------------------------------------------------------------------


def test_read_all_basic_numeric_types():
    """
    Cover B / I / J / K / E / D round-trip + endian swap.

    Demonstrates the default-mode form: FITS(fname) opens for reading
    just like the built-in open(fname).  Most read-only tests in this
    file still use the explicit FITS(fname, "r") form for historical
    reasons — either is fine.
    """
    fields = [
        ("B1", "1B"),
        ("I1", "1I"),
        ("J1", "1J"),
        ("K1", "1K"),
        ("E1", "1E"),
        ("D1", "1D"),
    ]
    row_w = 1 + 2 + 4 + 8 + 4 + 8  # 27 bytes/row
    rows_data = b""
    for i in range(3):
        rows_data += bytes([100 + i])
        rows_data += struct.pack(">h", -200 + i)
        rows_data += struct.pack(">i", 30000 + i)
        rows_data += struct.pack(">q", 10**10 + i)
        rows_data += struct.pack(">f", 1.5 + i)
        rows_data += struct.pack(">d", 3.14 + i)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(row_w, 3, fields), rows_data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read()
            assert a["B1"].tolist() == [100, 101, 102]
            assert a["I1"].tolist() == [-200, -199, -198]
            assert a["J1"].tolist() == [30000, 30001, 30002]
            assert a["K1"].tolist() == [10**10, 10**10 + 1, 10**10 + 2]
            np.testing.assert_allclose(a["E1"], [1.5, 2.5, 3.5])
            np.testing.assert_allclose(a["D1"], [3.14, 4.14, 5.14])


def test_read_all_logical_t_f_other():
    """L: 'T' -> True, 'F' -> False, anything else also False."""
    fields = [("FLAG", "1L")]
    rows_data = b"T" + b"F" + b"X"  # T, F, garbage
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 3, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a.dtype["FLAG"] == np.bool_
            assert a["FLAG"].tolist() == [True, False, False]


def test_read_all_complex_round_trip():
    """C (c8) and M (c16): paired float swap."""
    fields = [("C1", "1C"), ("M1", "1M")]
    rows_data = b""
    for i in range(2):
        rows_data += struct.pack(">ff", 1.0 + i, -2.0 - i)
        rows_data += struct.pack(">dd", 0.5 + i, 0.25 + i)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(24, 2, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            np.testing.assert_allclose(a["C1"], [1.0 - 2.0j, 2.0 - 3.0j])
            np.testing.assert_allclose(a["M1"], [0.5 + 0.25j, 1.5 + 1.25j])


def test_read_all_repeat_numeric_column():
    """TFORM='3I' (no TDIM) yields a (N, 3) field."""
    fields = [("TRIO", "3I")]
    rows_data = b""
    for i in range(3):
        rows_data += struct.pack(">3h", i, i + 1, i + 2)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(6, 3, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["TRIO"].shape == (3, 3)
            assert a["TRIO"].tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]


def test_read_all_single_char_a_column():
    """TFORM='1A' yields a scalar U1 field, no shape."""
    fields = [("CH", "1A")]
    rows_data = b"abc"
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 3, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["CH"].dtype.kind == "U"
            assert a["CH"].dtype.itemsize == 4  # numpy U1 == 4 bytes
            assert a["CH"].tolist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TDIM reshape
# ---------------------------------------------------------------------------


def test_tdim_numeric_transpose_orientation():
    """
    TFORM='6D' TDIM='(2,3)' (2 fast, 3 slow) → numpy shape (3,2).
    On-disk linear order is FORTRAN; numpy reads in row-major; the
    result is the transpose of the FITS-convention matrix."""
    fields = [("M", "6D", "(2,3)")]
    # Pack 6 D values in FORTRAN order:
    # M[i,j] = 10*i + j, with i in 0..2 (dim-0 fast), j in 0..3 (dim-1 slow)
    disk = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]
    rows_data = struct.pack(">6d", *disk)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(48, 1, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            m = a[0]["M"]
            assert m.shape == (3, 2)
            np.testing.assert_array_equal(
                m, [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]
            )


def test_tdim_a_column_string_array():
    """
    TFORM='20A' TDIM='(4,5)' = 5 strings of length 4 per row.
    Numpy field is U4 shape (5,)."""
    fields = [("WORDS", "20A", "(4,5)")]
    row = b"abcdABCDefghEFGHijkl"  # 5 strings of 4 chars
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(20, 1, fields), row)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            w = a[0]["WORDS"]
            assert w.shape == (5,)
            assert w.dtype.itemsize == 4 * 4
            assert w.tolist() == ["abcd", "ABCD", "efgh", "EFGH", "ijkl"]


# ---------------------------------------------------------------------------
# columns= subset
# ---------------------------------------------------------------------------


def _three_col_file(tmp):
    """Tiny 3-row file with ID (J), MASS (D), NAME (4A)."""
    fields = [("ID", "1J"), ("MASS", "1D"), ("NAME", "4A")]
    rows_data = b""
    for i in range(3):
        rows_data += struct.pack(">i", 100 + i)
        rows_data += struct.pack(">d", 1.5 + i)
        rows_data += f"r{i}  ".encode("ascii")
    fname = os.path.join(tmp, "t.fits")
    _write_bintable(fname, _bintable_cards(16, 3, fields), rows_data)
    return fname


def test_columns_subset_reorder():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(columns=["NAME", "ID"])
            assert a.dtype.names == ("NAME", "ID")
            assert a["ID"].tolist() == [100, 101, 102]


def test_columns_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(columns=["mass", "Id"])
            assert a.dtype.names == ("MASS", "ID")


def test_columns_unknown_name_lists_available():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="unknown column name"):
                fits[1].read(columns=["BOGUS"])
            # The error message should mention the available columns.
            try:
                fits[1].read(columns=["BOGUS"])
            except ValueError as e:
                assert "ID" in str(e) and "NAME" in str(e)


def test_columns_duplicate_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="duplicate column name"):
                fits[1].read(columns=["ID", "id"])  # dup, case-insens


def test_columns_empty_list_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="empty list"):
                fits[1].read(columns=[])


# ---------------------------------------------------------------------------
# rows= subset
# ---------------------------------------------------------------------------


def _ten_row_file(tmp):
    """10 rows: ID = 1000+i, NAME = 'r<i:>3>'."""
    fields = [("ID", "1J"), ("NAME", "4A")]
    rows_data = b""
    for i in range(10):
        rows_data += struct.pack(">i", 1000 + i)
        rows_data += f"r{i:>3}".encode("ascii")
    fname = os.path.join(tmp, "t.fits")
    _write_bintable(fname, _bintable_cards(8, 10, fields), rows_data)
    return fname


def test_rows_sequential_contiguous():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[3, 4, 5])
            assert a["ID"].tolist() == [1003, 1004, 1005]


def test_rows_scattered_preserves_user_order():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[7, 2, 5])
            assert a["ID"].tolist() == [1007, 1002, 1005]


def test_rows_dedup_keeps_first_occurrence_order():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[5, 2, 5, 9, 2])
            assert a["ID"].tolist() == [1005, 1002, 1009]


def test_rows_negative_indices():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[-1, -3])
            assert a["ID"].tolist() == [1009, 1007]


def test_rows_slice_positive_step():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=slice(1, 8, 2))
            assert a["ID"].tolist() == [1001, 1003, 1005, 1007]


def test_rows_slice_reversed():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=slice(None, None, -1))
            assert a["ID"].tolist() == list(reversed(range(1000, 1010)))


def test_rows_numpy_int_array():
    """numpy int arrays should work just like a list."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=np.array([4, 1, 6], dtype="i8"))
            assert a["ID"].tolist() == [1004, 1001, 1006]


def test_rows_tuple_input():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=(5, 0, 3))
            assert a["ID"].tolist() == [1005, 1000, 1003]


def test_rows_out_of_range_positive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(IndexError, match="out of range"):
                fits[1].read(rows=[3, 99])


def test_rows_out_of_range_negative():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(IndexError, match="out of range"):
                fits[1].read(rows=[-99])


def test_rows_empty_list_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="zero rows"):
                fits[1].read(rows=[])


def test_rows_and_columns_combined():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[3, 1], columns=["NAME"])
            assert a.dtype.names == ("NAME",)
            assert a["NAME"].tolist() == ["r  3", "r  1"]


# ---------------------------------------------------------------------------
# read_column
# ---------------------------------------------------------------------------


def test_read_column_scalar():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            ids = fits[1].read_column("ID")
            assert ids.shape == (3,)
            assert ids.tolist() == [100, 101, 102]


def test_read_column_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].read_column("mass").tolist() == [1.5, 2.5, 3.5]
            assert fits[1].read_column("Id").tolist() == [100, 101, 102]


def test_read_column_rows_subset():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read_column("ID", rows=[5, 1, 9])
            assert a.tolist() == [1005, 1001, 1009]


def test_read_column_multi_element_keeps_shape():
    """TFORM='3I' on a 3-row table: read_column yields shape (3, 3)."""
    fields = [("TRIO", "3I")]
    rows_data = b""
    for i in range(3):
        rows_data += struct.pack(">3h", i, i + 1, i + 2)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(6, 3, fields), rows_data)
        with rustfits.FITS(fname, "r") as fits:
            t = fits[1].read_column("TRIO")
            assert t.shape == (3, 3)
            assert t.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]


def test_read_column_unknown_name_lists_available():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            try:
                fits[1].read_column("nope")
            except ValueError as e:
                msg = str(e)
                assert "unknown column name" in msg
                assert "ID" in msg and "MASS" in msg and "NAME" in msg


# ---------------------------------------------------------------------------
# A column: truncate-at-null + rstrip + strict ASCII + as_bytes escape
# ---------------------------------------------------------------------------


def _file_with_dirty_strings(tmp):
    """
    A column with rows that exercise null truncation, rstrip, and
    a non-ASCII byte (Latin-1 'é' = 0xE9) in row 1."""
    names = [
        b"good    ",  # 4 chars + 4 spaces
        b"bad\xe9stf ",  # non-ASCII byte 0xE9
        b"AB\x00DEF  ",  # embedded null then more bytes
        b"r3      ",  # short with trailing spaces
        b"\x00\x00\x00\x00\x00\x00\x00\x00",  # all-null
    ]
    assert all(len(n) == 8 for n in names)
    rows_data = b""
    for i, n in enumerate(names):
        rows_data += struct.pack(">i", 100 + i)
        rows_data += n
    fields = [("ID", "1J"), ("NAME", "8A")]
    fname = os.path.join(tmp, "t.fits")
    _write_bintable(fname, _bintable_cards(12, 5, fields), rows_data)
    return fname, names


def test_a_column_trailing_spaces_stripped():
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read_column("NAME", rows=[0])
            assert a.tolist() == ["good"]


def test_a_column_truncates_at_first_null():
    """b"AB\\x00DEF  " → "AB" (not "AB\\x00DEF" and not "AB\\x00DEF ")."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read_column("NAME", rows=[2])
            assert a.tolist() == ["AB"]


def test_a_column_all_null_is_empty_string():
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read_column("NAME", rows=[4])
            assert a.tolist() == [""]


def test_a_column_strict_ascii_error_message():
    """
    Error names the column, the disk row, the byte, the position,
    and points at read_column(..., as_bytes=True) as the escape."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError) as exc:
                fits[1].read_column("NAME")
            msg = str(exc.value)
            assert "'NAME'" in msg
            assert "row 1" in msg
            assert "0xE9" in msg
            assert "as_bytes=True" in msg


def test_a_column_strict_error_via_full_table_read():
    """
    A non-ASCII A column also blows up the full table read; the
    error is the same."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="non-ASCII byte"):
                fits[1].read()


def test_a_column_strict_succeeds_when_skipping_bad_row():
    """rows= avoiding the bad row should succeed in strict mode."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read_column("NAME", rows=[0, 2, 3])
            assert a.tolist() == ["good", "AB", "r3"]


def test_a_column_as_bytes_returns_raw():
    """
    as_bytes=True returns the on-disk bytes byte-for-byte (no
    decode, no truncate, no rstrip; numpy S<n> strips only trailing
    NULL bytes per its dtype semantics)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, names = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            raw = fits[1].read_column("NAME", as_bytes=True)
            assert raw.dtype == np.dtype("S8")
            assert raw[0] == b"good    "
            assert raw[1] == b"bad\xe9stf "  # non-ASCII preserved
            assert raw[2] == b"AB\x00DEF  "  # embedded null preserved
            assert raw[3] == b"r3      "
            assert raw[4] == b""  # numpy strips all-null


def test_as_bytes_rejected_on_non_character_column():
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="character"):
                fits[1].read_column("ID", as_bytes=True)


# ---------------------------------------------------------------------------
# unsupported TFORM types are rejected up front
# ---------------------------------------------------------------------------


def test_variable_length_repeat_gt_1_rejected():
    """
    Multi-descriptor variable columns (e.g. '2PE') aren't supported
    yet."""
    fields = [("VLA", "2PE(100)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(16, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="repeat>1"):
                fits[1].read()


# ---------------------------------------------------------------------------
# X (bit) columns — MSB-packed bits, unpacked to numpy bool
# ---------------------------------------------------------------------------


def test_x_full_byte_two_rows():
    """8X = 8 bits = 1 byte per row.  No padding bits."""
    rows = bytes([0b10110100, 0b00001111])
    fields = [("F", "8X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 2, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a.dtype == np.dtype([("F", "?", (8,))])
            assert a["F"][0].tolist() == [
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
            ]
            assert a["F"][1].tolist() == [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ]


def test_x_scalar_1x():
    """
    1X = single bit per row.  Shape () (scalar bool) — consistent
    with the rest of the dtype convention (repeat==1, no TDIM)."""
    rows = bytes([0b10000000, 0b00000000])
    fields = [("F", "1X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 2, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a.dtype == np.dtype([("F", "?")])
            assert a["F"].tolist() == [True, False]


def test_x_partial_byte_padding_bits_ignored():
    """
    7X = 7 bits in 1 byte; the 8th (low) bit is padding and must
    not appear in the unpacked output."""
    rows = bytes([0b10101011])  # last 1 is padding
    fields = [("F", "7X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 1, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["F"][0].tolist() == [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
            ]


def test_x_multi_byte_with_partial_last_byte():
    """13X = 13 bits in 2 bytes; 3 padding bits in the second byte."""
    # First byte: 8 bits.  Second byte: top 5 bits used.
    rows = bytes([0b11110000, 0b10101000])
    fields = [("F", "13X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(2, 1, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["F"][0].tolist() == [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
            ]


def test_x_with_tdim_reshape():
    """
    TDIM='(2,3)' over '6X' should reshape each cell to numpy shape
    (3, 2) using the same FORTRAN-to-numpy transpose rule as numeric
    TDIM."""
    # 6 bits packed MSB-first: bits b0..b5 (b5 is the 6th bit).
    # bit pattern: 1,0,1,1,0,0  → byte 0b101100xx (last 2 bits padding).
    rows = bytes([0b10110000])
    fields = [("M", "6X", "(2,3)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 1, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            # field shape is reversed(tdim) = (3, 2)
            assert a["M"].shape == (1, 3, 2)
            # FITS FORTRAN-flat order: b0..b5 = [1,0,1,1,0,0].  With
            # TDIM=(2,3), the FORTRAN-indexed element [i,j] (i in [0,2),
            # j in [0,3)) at flat k = i + j*2 maps to numpy [r=j, c=i].
            # So numpy[0] = (b0,b1), numpy[1] = (b2,b3), numpy[2]=(b4,b5).
            assert a["M"][0].tolist() == [
                [True, False],
                [True, True],
                [False, False],
            ]


def test_x_read_column_returns_plain_bool_array():
    rows = bytes([0b11000000])
    fields = [("F", "8X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 1, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            col = fits[1].read_column("F")
            assert col.dtype == np.bool_
            assert col.shape == (1, 8)
            assert col[0].tolist() == [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ]


def test_x_rows_subset():
    rows = bytes([0b10000000, 0b01000000, 0b00100000, 0b00010000])
    fields = [("F", "3X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 4, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[3, 1])
            assert a["F"][0].tolist() == [False, False, False]
            assert a["F"][1].tolist() == [False, True, False]


def test_x_mixed_with_other_columns():
    """
    X alongside fixed numeric — both must come out correctly in the
    same structured array."""
    # row 0: ID=10, FLAGS bits = [T,F,T,F,T,F,T,F] = 0b10101010
    # row 1: ID=20, FLAGS bits = [F,T,F,T,F,T,F,T] = 0b01010101
    rows = (
        struct.pack(">i", 10)
        + bytes([0b10101010])
        + struct.pack(">i", 20)
        + bytes([0b01010101])
    )
    fields = [("ID", "1J"), ("FLAGS", "8X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(5, 2, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["ID"].tolist() == [10, 20]
            assert a["FLAGS"][0].tolist() == [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ]
            assert a["FLAGS"][1].tolist() == [
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
            ]


def test_x_column_subset_via_getitem():
    rows = bytes([0b11110000])
    fields = [("F", "8X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 1, fields), rows)
        with rustfits.FITS(fname, "r") as fits:
            t = fits[1]
            col = t["F"][:]
            assert col.tolist() == [
                [True, True, True, True, False, False, False, False]
            ]


def test_x_dtype_property_matches_read():
    rows = bytes([0])
    fields = [("F", "12X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(2, 1, fields), rows + b"\x00")
        with rustfits.FITS(fname, "r") as fits:
            dt = fits[1].dtype
            a = fits[1].read()
            assert dt == a.dtype
            assert dt == np.dtype([("F", "?", (12,))])


# ---------------------------------------------------------------------------
# dtype property: structured dtype matches what read() returns
# ---------------------------------------------------------------------------


def test_dtype_property_matches_read_output():
    """
    TableHDU.dtype should equal the dtype of the array read()
    returns."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            dt = fits[1].dtype
            a = fits[1].read()
            assert dt == a.dtype


# ---------------------------------------------------------------------------
# empty table (NAXIS2=0)
# ---------------------------------------------------------------------------


def test_read_empty_table():
    fields = [("ID", "1J"), ("NAME", "4A")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(8, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a.shape == (0,)
            assert a.dtype.names == ("ID", "NAME")
            col = fits[1].read_column("ID")
            assert col.shape == (0,)


# ---------------------------------------------------------------------------
# __getitem__: hdu[key] is shorthand for hdu.read(rows=key)
# ---------------------------------------------------------------------------


def test_getitem_slice_basic():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][2:5]
            assert a.dtype.names == ("ID", "NAME")
            assert a["ID"].tolist() == [1002, 1003, 1004]


def test_getitem_slice_step():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][1:8:2]
            assert a["ID"].tolist() == [1001, 1003, 1005, 1007]


def test_getitem_slice_open_ended():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][:3]
            assert a["ID"].tolist() == [1000, 1001, 1002]
            b = fits[1][7:]
            assert b["ID"].tolist() == [1007, 1008, 1009]


def test_getitem_slice_negative_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][-3:]
            assert a["ID"].tolist() == [1007, 1008, 1009]
            b = fits[1][-5:-2]
            assert b["ID"].tolist() == [1005, 1006, 1007]


def test_getitem_slice_reversed():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][::-2]
            assert a["ID"].tolist() == [1009, 1007, 1005, 1003, 1001]


def test_getitem_list_in_user_order():
    """
    Returned rows must be in the user's requested order, not on-disk
    order — that's what the run-planner's output_indices is for."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[5, 1, 8, 3]]
            assert a["ID"].tolist() == [1005, 1001, 1008, 1003]


def test_getitem_numpy_array():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][np.array([0, 4, 9])]
            assert a["ID"].tolist() == [1000, 1004, 1009]


def test_getitem_tuple_treated_as_iterable():
    """A tuple key is iterable-of-ints, equivalent to a list."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][(2, 5, 7)]
            assert a["ID"].tolist() == [1002, 1005, 1007]


def test_getitem_negative_indices_in_list():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[-1, -3, 0]]
            assert a["ID"].tolist() == [1009, 1007, 1000]


def test_getitem_duplicates_deduped_preserving_first_order():
    """
    Matches read(rows=) dedup behavior: duplicates dropped, first
    occurrence wins the position in the output."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[3, 1, 3, 5, 1]]
            assert a["ID"].tolist() == [1003, 1001, 1005]


def test_getitem_matches_read_rows_for_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][2:9:3]
            b = fits[1].read(rows=slice(2, 9, 3))
            assert (a == b).all()
            assert a.dtype == b.dtype


def test_getitem_matches_read_rows_for_list():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[7, 0, 4, 2]]
            b = fits[1].read(rows=[7, 0, 4, 2])
            assert (a == b).all()


def test_getitem_out_of_range_index_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(IndexError):
                fits[1][[0, 20]]


def test_getitem_non_integer_in_list_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                fits[1][[0, 1.5]]


def test_getitem_bad_key_type_raises():
    """A bare float (non-iterable, non-slice) is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                fits[1][1.5]


def test_getitem_on_string_columns():
    """
    Make sure A-column decode goes through __getitem__ just like
    through read()."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[2, 0]]
            assert a["NAME"].tolist() == ["r  2", "r  0"]
            assert a["ID"].tolist() == [1002, 1000]


# ---------------------------------------------------------------------------
# __getitem__ column subsets: hdu[col] / hdu[[cols]] do NOT read; the
# subsequent [rows] does.
# ---------------------------------------------------------------------------


def test_getitem_single_column_returns_subset_no_read():
    """hdu['col'] returns a SingleColumnSubset; no I/O yet."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[1]["ID"]
            assert "TableColumn" in repr(sub)
            assert "ID" in repr(sub)


def test_getitem_single_column_then_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1]["ID"][:]
            assert a.tolist() == [100, 101, 102]


def test_getitem_single_column_then_rows_list():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1]["MASS"][[2, 0]]
            assert a.tolist() == [3.5, 1.5]


def test_getitem_single_column_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1]["id"][:]
            assert a.tolist() == [100, 101, 102]


def test_getitem_single_column_bytes_name():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][b"ID"][:]
            assert a.tolist() == [100, 101, 102]


def test_getitem_single_column_numpy_str_name():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][np.str_("ID")][:]
            assert a.tolist() == [100, 101, 102]


def test_getitem_single_column_numpy_bytes_name():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][np.bytes_(b"ID")][:]
            assert a.tolist() == [100, 101, 102]


def test_getitem_single_column_unknown_lists_available():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="unknown column"):
                fits[1]["BOGUS"][:]


def test_getitem_single_column_matches_read_column():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1]["MASS"][[0, 2]]
            b = fits[1].read_column("MASS", rows=[0, 2])
            assert (a == b).all()
            assert a.dtype == b.dtype


def test_getitem_multi_column_returns_subset_no_read():
    """hdu[[c1, c2]] returns a ColumnSubset; no I/O yet."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[1][["NAME", "ID"]]
            assert "TableColumns" in repr(sub)
            assert "NAME" in repr(sub)
            assert "ID" in repr(sub)


def test_getitem_multi_column_then_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][["NAME", "ID"]][:]
            assert a.dtype.names == ("NAME", "ID")
            assert a["ID"].tolist() == [100, 101, 102]


def test_getitem_multi_column_then_rows_list():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][["MASS", "ID"]][[2, 0]]
            assert a.dtype.names == ("MASS", "ID")
            assert a["ID"].tolist() == [102, 100]
            assert a["MASS"].tolist() == [3.5, 1.5]


def test_getitem_multi_column_tuple_of_names():
    """A tuple of strings should also be treated as column names."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][("ID", "MASS")][:]
            assert a.dtype.names == ("ID", "MASS")


def test_getitem_multi_column_numpy_str_array():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][np.array(["ID", "MASS"])][:]
            assert a.dtype.names == ("ID", "MASS")


def test_getitem_multi_column_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][["id", "Mass"]][:]
            assert a.dtype.names == ("ID", "MASS")


def test_getitem_multi_column_matches_read_with_kwargs():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][["NAME", "ID"]][[2, 0]]
            b = fits[1].read(columns=["NAME", "ID"], rows=[2, 0])
            assert (a == b).all()
            assert a.dtype == b.dtype


def test_getitem_multi_column_unknown_lists_available():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="unknown column"):
                fits[1][["ID", "BOGUS"]][:]


def test_getitem_multi_column_duplicate_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="duplicate column"):
                fits[1][["ID", "ID"]][:]


def test_getitem_empty_sequence_rejected():
    """Empty sequence is ambiguous (rows or columns?)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="empty sequence"):
                fits[1][[]]


def test_getitem_mixed_types_in_sequence_rejected():
    """A sequence mixing strings and ints is ambiguous."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="mixes column names"):
                fits[1][["ID", 0]]


def test_getitem_dispatch_int_list_still_rows():
    """
    [int, int, ...] must stay on the rows path even after we added
    column-name dispatch — this is the regression case for the
    Vec<u8>::extract foot-gun (a list of small ints would have silently
    decoded as a 'bytes column name')."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[2, 0]]
            assert a.dtype.names == ("ID", "MASS", "NAME")
            assert a["ID"].tolist() == [102, 100]


def test_getitem_dispatch_numpy_int_array_still_rows():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][np.array([2, 0])]
            assert a.dtype.names == ("ID", "MASS", "NAME")
            assert a["ID"].tolist() == [102, 100]


def test_getitem_bytearray_not_treated_as_column_name():
    """
    bytearray is not a `bytes` subclass; the explicit type-instance
    check excludes it.  It is also not an int-iterable in our
    classifier's eyes, so it errors via the iterable path."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            # bytearray IS iterable of small ints, so it routes to rows;
            # at resolve_rows we'll get out-of-range or extract failures.
            # Either way, no silent column-name match.
            with pytest.raises((ValueError, IndexError, TypeError)):
                fits[1][bytearray(b"ID")]


def test_getitem_single_column_bad_rows_type():
    """
    A SingleColumnSubset still requires slice or iterable-of-int for
    rows; bare float should raise."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                fits[1]["ID"][1.5]


def test_getitem_multi_column_bad_rows_type():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _three_col_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError):
                fits[1][["ID", "MASS"]][1.5]


# ---------------------------------------------------------------------------
# variable-length (P/Q descriptor) columns
# ---------------------------------------------------------------------------


def _write_var_table(path, fields, descriptors, heap, theap=None):
    """
    Build a BINTABLE file with a variable-length column.

    `fields` is the usual list of (ttype, tform) tuples; the variable
    column's TFORM is e.g. '1PE(maxlen)'.  `descriptors` are the raw
    main-data bytes (one descriptor per row, big-endian).  `heap` is
    the raw heap bytes laid out immediately after the main array (or
    at THEAP if given).  PCOUNT is set to len(heap).
    """
    row_width = 0
    for _, tform in fields:
        if "P" in tform and not tform.lstrip("0123456789").startswith("PA"):
            row_width += 8  # default: assume 1P descriptor
        elif "P" in tform:
            row_width += 8
        elif "Q" in tform:
            row_width += 16
        else:
            raise ValueError(f"unsupported tform in helper: {tform}")
    naxis2 = len(descriptors) // row_width
    cards = _bintable_cards(row_width, naxis2, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    if theap is not None:
        cards = cards[:-1] + [f"THEAP   = {theap:>20d}", "END"]
    payload = descriptors
    if theap is not None and theap > row_width * naxis2:
        payload += b"\x00" * (theap - row_width * naxis2)
    payload += heap
    _write_bintable(path, cards, payload)


def test_var_p_float32_basic():
    """1PE: per-row float32 array; lengths vary."""
    descriptors = (
        struct.pack(">ii", 3, 0)
        + struct.pack(">ii", 0, 12)
        + struct.pack(">ii", 2, 12)
    )
    heap = struct.pack(">fff", 1.0, 2.0, 3.0) + struct.pack(">ff", 10.0, 20.0)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("VLA", "1PE(20)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a.dtype == np.dtype([("VLA", "O")])
            assert a["VLA"][0].tolist() == [1.0, 2.0, 3.0]
            assert a["VLA"][1].tolist() == []
            assert a["VLA"][2].tolist() == [10.0, 20.0]
            for v in a["VLA"]:
                assert v.dtype == np.float32


def test_var_p_int64():
    descriptors = struct.pack(">ii", 2, 0) + struct.pack(">ii", 1, 16)
    heap = struct.pack(">qq", 10**10, -1) + struct.pack(">q", 42)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PK(5)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"][0].dtype == np.int64
            assert a["X"][0].tolist() == [10**10, -1]
            assert a["X"][1].tolist() == [42]


def test_var_q_descriptor_i64():
    """Q descriptors use two i64 (16 bytes per row)."""
    descriptors = struct.pack(">qq", 2, 0) + struct.pack(">qq", 4, 8)
    heap = struct.pack(">ii", 100, 200) + struct.pack(">iiii", 1, 2, 3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("Q1", "1QJ(10)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["Q1"][0].tolist() == [100, 200]
            assert a["Q1"][1].tolist() == [1, 2, 3, 4]


def test_var_logical_l():
    descriptors = struct.pack(">ii", 3, 0)
    heap = b"TFT"
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("FLAGS", "1PL(5)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            v = a["FLAGS"][0]
            assert v.dtype == np.bool_
            assert v.tolist() == [True, False, True]


def test_var_a_string_basic():
    descriptors = (
        struct.pack(">ii", 5, 0)
        + struct.pack(">ii", 3, 5)
        + struct.pack(">ii", 0, 8)
    )
    heap = b"hello" + b"foo"
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("NAME", "1PA(20)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["NAME"][0] == "hello"
            assert a["NAME"][1] == "foo"
            assert a["NAME"][2] == ""


def test_var_a_non_ascii_error_with_escape_hint():
    descriptors = struct.pack(">ii", 4, 0)
    heap = b"ab\xffc"
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("NAME", "1PA(10)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError) as ei:
                fits[1].read()
            msg = str(ei.value)
            assert "NAME" in msg
            assert "as_bytes=True" in msg


def test_var_a_as_bytes_returns_raw_bytes():
    descriptors = struct.pack(">ii", 4, 0) + struct.pack(">ii", 3, 4)
    heap = b"ab\xffc" + b"\x00\x00\x00"
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("NAME", "1PA(10)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            col = fits[1].read_column("NAME", as_bytes=True)
            assert col[0] == b"ab\xffc"
            assert col[1] == b"\x00\x00\x00"


def test_var_complex_c8():
    descriptors = struct.pack(">ii", 2, 0)
    # 2 c8 elements = 4 float32 values
    heap = struct.pack(">ffff", 1.0, 2.0, 3.0, 4.0)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("Z", "1PC(5)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["Z"][0].dtype == np.complex64
            assert a["Z"][0].tolist() == [complex(1, 2), complex(3, 4)]


def test_var_empty_cell_is_empty_ndarray():
    """
    A descriptor with nelements=0 should yield a 0-length ndarray
    (not None, not a scalar), so downstream code can unconditionally
    index into the cell."""
    descriptors = struct.pack(">ii", 0, 0)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PE(10)")], descriptors, b"")
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            v = a["X"][0]
            assert isinstance(v, np.ndarray)
            assert v.shape == (0,)
            assert v.dtype == np.float32


def test_var_rows_subset():
    descriptors = (
        struct.pack(">ii", 1, 0)
        + struct.pack(">ii", 2, 4)
        + struct.pack(">ii", 3, 12)
    )
    heap = (
        struct.pack(">i", 100)
        + struct.pack(">ii", 200, 201)
        + struct.pack(">iii", 300, 301, 302)
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PJ(5)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            t = fits[1]
            a = t.read(rows=[2, 0])
            assert a["X"][0].tolist() == [300, 301, 302]
            assert a["X"][1].tolist() == [100]
            # __getitem__ paths
            b = t[[2, 0]]
            assert b["X"][0].tolist() == [300, 301, 302]
            c = t["X"][[2, 0]]
            assert c[0].tolist() == [300, 301, 302]
            assert c[1].tolist() == [100]


def test_var_mixed_with_fixed_columns():
    """
    A row has a fixed J and a variable PE; both must come out
    correctly into the same structured array."""
    descriptors = b""
    for nelem, off, ident in [(2, 0, 10), (3, 8, 11), (1, 20, 12)]:
        descriptors += struct.pack(">i", ident)  # ID (J, 4 bytes)
        descriptors += struct.pack(">ii", nelem, off)  # VLA descriptor
    heap = (
        struct.pack(">ff", 1.0, 2.0)
        + struct.pack(">fff", 3.0, 4.0, 5.0)
        + struct.pack(">f", 6.0)
    )
    fields = [("ID", "1J"), ("VLA", "1PE(10)")]
    cards = _bintable_cards(12, 3, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            t = fits[1]
            a = t.read()
            assert a["ID"].tolist() == [10, 11, 12]
            assert a["VLA"][0].tolist() == [1.0, 2.0]
            assert a["VLA"][1].tolist() == [3.0, 4.0, 5.0]
            assert a["VLA"][2].tolist() == [6.0]


def test_var_columns_subset_excludes_variable():
    """
    columns= without the variable column should not need any heap
    reads (and skipping is harmless when the heap is unreadable)."""
    descriptors = b""
    for ident in [10, 11]:
        descriptors += struct.pack(">i", ident)
        descriptors += struct.pack(">ii", 0, 0)
    heap = b""
    fields = [("ID", "1J"), ("VLA", "1PE(10)")]
    cards = _bintable_cards(12, 2, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(columns=["ID"])
            assert a.dtype == np.dtype([("ID", "<i4")])
            assert a["ID"].tolist() == [10, 11]


def test_var_columns_subset_includes_variable():
    descriptors = b""
    for ident in [10, 11]:
        descriptors += struct.pack(">i", ident)
        descriptors += struct.pack(">ii", 1, 0 if ident == 10 else 4)
    heap = struct.pack(">f", 1.0) + struct.pack(">f", 2.0)
    fields = [("ID", "1J"), ("VLA", "1PE(10)")]
    cards = _bintable_cards(12, 2, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(columns=["VLA"])
            assert a.dtype == np.dtype([("VLA", "O")])
            assert a["VLA"][0].tolist() == [1.0]
            assert a["VLA"][1].tolist() == [2.0]


def test_var_theap_keyword_respected():
    """
    THEAP says the heap starts at a non-default offset (i.e. there's
    a gap between the main rows and the heap)."""
    descriptors = struct.pack(">ii", 2, 0)
    gap = b"\x00" * 20
    heap = struct.pack(">ff", 7.0, 8.0)
    row_width = 8
    naxis2 = 1
    theap = row_width * naxis2 + len(gap)
    fields = [("X", "1PE(10)")]
    cards = _bintable_cards(row_width, naxis2, fields)
    cards = [
        f"PCOUNT  = {len(gap) + len(heap):>20d}"
        if c.startswith("PCOUNT")
        else c
        for c in cards
    ]
    cards = cards[:-1] + [f"THEAP   = {theap:>20d}", "END"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + gap + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"][0].tolist() == [7.0, 8.0]


def test_var_dtype_property_object():
    descriptors = struct.pack(">ii", 0, 0)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PE(10)")], descriptors, b"")
        with rustfits.FITS(fname, "r") as fits:
            dt = fits[1].dtype
            assert dt == np.dtype([("X", "O")])


def test_var_negative_nelements_raises():
    """
    A bad descriptor (nelements < 0) is a corrupt file; reject with
    a clear error rather than crashing."""
    descriptors = struct.pack(">ii", -1, 0)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PE(10)")], descriptors, b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises((OSError, ValueError)):
                fits[1].read()


def test_var_aliased_heap_offset_allowed():
    """
    Two rows pointing at the same heap region must each get their
    own ndarray (correct + safe — no shared mutability)."""
    descriptors = struct.pack(">ii", 2, 0) + struct.pack(">ii", 2, 0)
    heap = struct.pack(">ff", 1.5, 2.5)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_var_table(fname, [("X", "1PE(5)")], descriptors, heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"][0].tolist() == [1.5, 2.5]
            assert a["X"][1].tolist() == [1.5, 2.5]
            # Verify they are independent objects.
            a["X"][0][0] = 99.0
            assert a["X"][1][0] == 1.5


def test_var_tdim_on_p_rejected():
    """
    TDIM on a P column is not yet supported; reject at parse
    time."""
    fields = [("VLA", "1PE(10)", "(2,5)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        # Build cards manually since _bintable_cards adds TDIMn.
        cards = _bintable_cards(8, 0, fields)
        _write_bintable(fname, cards, b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="TDIM"):
                fits[1].read()


def test_var_bad_inner_letter_rejected():
    fields = [("VLA", "1PZ(10)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(8, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="inner element"):
                fits[1].read()


def test_var_missing_inner_letter_rejected():
    fields = [("VLA", "1P")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(8, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="inner element"):
                fits[1].read()


def test_var_inner_x_rejected():
    fields = [("VLA", "1PX(10)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(8, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="bit"):
                fits[1].read()


# ---------------------------------------------------------------------------
# TSCAL/TZERO scaling: unsigned-int trick + general linear scaling
# ---------------------------------------------------------------------------


def _bintable_with_scaling(naxis1, naxis2, fields, tscals_tzeros):
    """
    Like _bintable_cards but inserts TSCAL/TZERO cards.

    `tscals_tzeros` is a dict {col_index_1based: (tscal_str, tzero_str)}.
    Each value is the literal numeric string to write into the card.
    """
    cards = _bintable_cards(naxis1, naxis2, fields)
    extras = []
    for i, (tscal, tzero) in tscals_tzeros.items():
        if tscal is not None:
            extras.append(f"TSCAL{i:<3d}= {tscal:>20s}")
        if tzero is not None:
            extras.append(f"TZERO{i:<3d}= {tzero:>20s}")
    # Insert before the END card.
    return cards[:-1] + extras + ["END"]


# ---------------- unsigned-int trick: I → u2 ----------------


def test_scaling_unsigned_int_trick_u16():
    """TFORM=I, TSCAL=1, TZERO=32768 → uint16 output."""
    data = struct.pack(">hhh", -32768, 0, 32767)
    fields = [("U16", "1I")]
    cards = _bintable_with_scaling(2, 3, fields, {1: ("1", "32768")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["U16"].dtype == np.uint16
            assert a["U16"].tolist() == [0, 32768, 65535]


def test_scaling_unsigned_int_trick_u32():
    data = struct.pack(">iii", -2147483648, 0, 2147483647)
    fields = [("U32", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: ("1", "2147483648")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["U32"].dtype == np.uint32
            assert a["U32"].tolist() == [0, 2147483648, 4294967295]


def test_scaling_unsigned_int_trick_u64():
    """K (int64) + TZERO=2^63 → uint64, no precision loss."""
    data = struct.pack(">qqq", -(2**63), 0, 2**63 - 1)
    fields = [("U64", "1K")]
    cards = _bintable_with_scaling(
        8, 3, fields, {1: ("1", "9223372036854775808")}
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["U64"].dtype == np.uint64
            assert a["U64"].tolist() == [0, 2**63, 2**64 - 1]


def test_scaling_signed_byte_trick_i8():
    """B (uint8) + TZERO=-128 → int8."""
    data = bytes([0, 128, 255])  # stored unsigned 0, 128, 255
    fields = [("S8", "1B")]
    cards = _bintable_with_scaling(1, 3, fields, {1: ("1", "-128")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["S8"].dtype == np.int8
            assert a["S8"].tolist() == [-128, 0, 127]


# ---------------- general (TSCAL!=1 or non-trick TZERO) ----------------


def test_scaling_general_linear_on_int_column():
    """TSCAL=2, TZERO=10 on i32 → f8 output with physical = 2*x+10."""
    data = struct.pack(">iii", 0, 5, 100)
    fields = [("X", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: ("2.0", "10.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"].dtype == np.float64
            assert a["X"].tolist() == [10.0, 20.0, 210.0]


def test_scaling_general_on_float_column():
    """TSCAL=0.5 on f4 → f8 output with physical = 0.5*x."""
    data = struct.pack(">ff", 2.0, 4.0)
    fields = [("X", "1E")]
    cards = _bintable_with_scaling(4, 2, fields, {1: ("0.5", "0.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"].dtype == np.float64
            assert a["X"].tolist() == [1.0, 2.0]


def test_scaling_tscal_only():
    """TSCAL=3, TZERO absent → general scaling (physical = 3*x)."""
    data = struct.pack(">iii", 1, 2, 3)
    fields = [("X", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: ("3.0", None)})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"].dtype == np.float64
            assert a["X"].tolist() == [3.0, 6.0, 9.0]


def test_scaling_tzero_only_nontrick():
    """
    TZERO set to a non-trick value → general scaling, not unsigned
    trick.  E.g. TZERO=100 on i32."""
    data = struct.pack(">iii", -10, 0, 10)
    fields = [("X", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: (None, "100.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"].dtype == np.float64
            assert a["X"].tolist() == [90.0, 100.0, 110.0]


# ---------------- scale=False opt-out ----------------


def test_scaling_disabled_returns_raw():
    data = struct.pack(">hhh", -32768, 0, 32767)
    fields = [("U16", "1I")]
    cards = _bintable_with_scaling(2, 3, fields, {1: ("1", "32768")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(scale=False)
            assert a["U16"].dtype == np.int16
            assert a["U16"].tolist() == [-32768, 0, 32767]


def test_scaling_read_column_scale_false():
    data = struct.pack(">iii", 0, 5, 100)
    fields = [("X", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: ("2.0", "10.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            col_scaled = fits[1].read_column("X")
            assert col_scaled.dtype == np.float64
            assert col_scaled.tolist() == [10.0, 20.0, 210.0]
            col_raw = fits[1].read_column("X", scale=False)
            assert col_raw.dtype == np.int32
            assert col_raw.tolist() == [0, 5, 100]


# ---------------- defaults are no-op (fast path preserved) ----------------


def test_scaling_defaults_no_change():
    """
    A column with default TSCAL=1 and TZERO=0 (or missing) gives
    the same dtype and values regardless of scale=True/False."""
    data = struct.pack(">iii", 1, 2, 3)
    fields = [("X", "1J")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(4, 3, fields), data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            b = fits[1].read(scale=False)
            assert a.dtype == b.dtype
            assert a.dtype == np.dtype([("X", "<i4")])
            assert a["X"].tolist() == b["X"].tolist() == [1, 2, 3]


def test_scaling_explicit_defaults_no_change():
    """TSCAL=1.0, TZERO=0.0 explicitly set: still treated as no-op."""
    data = struct.pack(">ii", 7, 8)
    fields = [("X", "1J")]
    cards = _bintable_with_scaling(4, 2, fields, {1: ("1.0", "0.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["X"].dtype == np.int32
            assert a["X"].tolist() == [7, 8]


# ---------------- complex column: scaling raises ----------------


def test_scaling_on_complex_raises():
    fields = [("Z", "1C")]
    cards = _bintable_with_scaling(8, 0, fields, {1: ("2.0", "0.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="complex"):
                fits[1].read()


def test_scaling_on_complex_scale_false_ok():
    """
    scale=False bypasses scaling — even C/M with TSCAL/TZERO set
    should read without error."""
    data = struct.pack(">ff", 1.0, 2.0)
    fields = [("Z", "1C")]
    cards = _bintable_with_scaling(8, 1, fields, {1: ("2.0", "0.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(scale=False)
            assert a["Z"][0] == complex(1, 2)


# ---------------- non-numeric types: scaling ignored ----------------


def test_scaling_on_logical_silently_ignored():
    """TSCAL/TZERO on L is meaningless; defaults pass through."""
    data = bytes([ord("T"), ord("F")])
    fields = [("FLAG", "1L")]
    cards = _bintable_with_scaling(1, 2, fields, {1: ("5.0", "10.0")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["FLAG"].dtype == np.bool_
            assert a["FLAG"].tolist() == [True, False]


# ---------------- multi-element repeat with scaling ----------------


def test_scaling_unsigned_trick_multi_element():
    """TFORM='3I' with the u16 trick: 3 elements per row, each scaled."""
    data = struct.pack(">hhhhhh", -32768, 0, 32767, 1, 2, 3)
    fields = [("V", "3I")]
    cards = _bintable_with_scaling(6, 2, fields, {1: ("1", "32768")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            assert a["V"].dtype == np.uint16
            assert a["V"][0].tolist() == [0, 32768, 65535]
            assert a["V"][1].tolist() == [32769, 32770, 32771]


# ---------------- read_column on scaled column ----------------


def test_scaling_read_column_unsigned_trick():
    data = struct.pack(">iii", -1, 0, 1)
    fields = [("U32", "1J")]
    cards = _bintable_with_scaling(4, 3, fields, {1: ("1", "2147483648")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            col = fits[1].read_column("U32")
            assert col.dtype == np.uint32
            assert col.tolist() == [2147483647, 2147483648, 2147483649]


# ---------------- dtype property reflects scaled dtype ----------------


def test_scaling_dtype_property_promotes():
    fields = [("U16", "1I"), ("F", "1J")]
    cards = _bintable_with_scaling(
        6,
        0,
        fields,
        {
            1: ("1", "32768"),  # unsigned trick → u2
            2: ("2.0", "10.0"),  # general → f8
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, b"")
        with rustfits.FITS(fname, "r") as fits:
            dt = fits[1].dtype
            assert dt == np.dtype([("U16", "<u2"), ("F", "<f8")])


# ---------------- subset paths still work with scaling ----------------


def test_scaling_with_rows_and_columns_subsets():
    data = b""
    for stored in [-32768, 0, 32767]:
        data += struct.pack(">h", stored) + struct.pack(">i", stored)
    fields = [("U16", "1I"), ("RAW", "1J")]
    cards = _bintable_with_scaling(6, 3, fields, {1: ("1", "32768")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(rows=[2, 0], columns=["U16"])
            assert a["U16"].dtype == np.uint16
            assert a["U16"].tolist() == [65535, 0]


def test_scaling_getitem_column_subset():
    data = struct.pack(">h", 0) + struct.pack(">h", 100)
    fields = [("U16", "1I")]
    cards = _bintable_with_scaling(2, 2, fields, {1: ("1", "32768")})
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname, "r") as fits:
            col = fits[1]["U16"][:]
            assert col.dtype == np.uint16
            assert col.tolist() == [32768, 32868]


# ---------------- variable-length scaling ----------------


def test_scaling_variable_length_unsigned_trick():
    """1PJ with TSCAL=1, TZERO=2^31 → each heap cell becomes a u4 ndarray."""
    descriptors = struct.pack(">ii", 3, 0) + struct.pack(">ii", 2, 12)
    heap = struct.pack(">iii", -1, 0, 1) + struct.pack(
        ">ii", -(2**31), 2**31 - 1
    )  # noqa
    fields = [("V", "1PJ(10)")]
    row_width = 8
    naxis2 = 2
    cards = _bintable_cards(row_width, naxis2, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    cards = cards[:-1] + [
        "TSCAL1  =                    1",
        "TZERO1  =           2147483648",
        "END",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            v0 = a["V"][0]
            v1 = a["V"][1]
            assert v0.dtype == np.uint32
            assert v0.tolist() == [2147483647, 2147483648, 2147483649]
            assert v1.tolist() == [0, 4294967295]


def test_scaling_variable_length_general():
    """1PE with TSCAL=10.0, TZERO=0.0 → each cell is f8 (promoted)."""
    descriptors = struct.pack(">ii", 3, 0)
    heap = struct.pack(">fff", 1.0, 2.0, 3.0)
    fields = [("V", "1PE(10)")]
    row_width = 8
    cards = _bintable_cards(row_width, 1, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    cards = cards[:-1] + [
        "TSCAL1  =                 10.0",
        "TZERO1  =                  0.0",
        "END",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read()
            v = a["V"][0]
            assert v.dtype == np.float64
            assert v.tolist() == [10.0, 20.0, 30.0]


def test_scaling_variable_length_scale_false():
    descriptors = struct.pack(">ii", 3, 0)
    heap = struct.pack(">iii", -1, 0, 1)
    fields = [("V", "1PJ(10)")]
    cards = _bintable_cards(8, 1, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    cards = cards[:-1] + [
        "TSCAL1  =                    1",
        "TZERO1  =           2147483648",
        "END",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, descriptors + heap)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1].read(scale=False)
            v = a["V"][0]
            assert v.dtype == np.int32
            assert v.tolist() == [-1, 0, 1]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
