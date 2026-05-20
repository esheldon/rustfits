"""Tests for BINTABLE read API (read + read_column).

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


CARDS_PER_BLOCK = 36   # 2880 / 80
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
    """Minimal BINTABLE header cards.  `fields` is a list of
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
    """Cover B / I / J / K / E / D round-trip + endian swap."""
    fields = [
        ("B1", "1B"), ("I1", "1I"), ("J1", "1J"),
        ("K1", "1K"), ("E1", "1E"), ("D1", "1D"),
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
        with rustfits.FITS(fname, "r") as fits:
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
    rows_data = b"T" + b"F" + b"X"   # T, F, garbage
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
            assert a["TRIO"].tolist() == [
                [0, 1, 2], [1, 2, 3], [2, 3, 4]
            ]


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
            assert a["CH"].dtype.itemsize == 4   # numpy U1 == 4 bytes
            assert a["CH"].tolist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TDIM reshape
# ---------------------------------------------------------------------------


def test_tdim_numeric_transpose_orientation():
    """TFORM='6D' TDIM='(2,3)' (2 fast, 3 slow) → numpy shape (3,2).
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
    """TFORM='20A' TDIM='(4,5)' = 5 strings of length 4 per row.
    Numpy field is U4 shape (5,)."""
    fields = [("WORDS", "20A", "(4,5)")]
    row = b"abcdABCDefghEFGHijkl"   # 5 strings of 4 chars
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
                fits[1].read(columns=["ID", "id"])   # dup, case-insens


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
            assert t.tolist() == [
                [0, 1, 2], [1, 2, 3], [2, 3, 4]
            ]


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
    """A column with rows that exercise null truncation, rstrip, and
    a non-ASCII byte (Latin-1 'é' = 0xE9) in row 1."""
    names = [
        b"good    ",          # 4 chars + 4 spaces
        b"bad\xe9stf ",       # non-ASCII byte 0xE9
        b"AB\x00DEF  ",       # embedded null then more bytes
        b"r3      ",          # short with trailing spaces
        b"\x00\x00\x00\x00\x00\x00\x00\x00",   # all-null
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
    """Error names the column, the disk row, the byte, the position,
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
    """A non-ASCII A column also blows up the full table read; the
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
    """as_bytes=True returns the on-disk bytes byte-for-byte (no
    decode, no truncate, no rstrip; numpy S<n> strips only trailing
    NULL bytes per its dtype semantics)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname, names = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            raw = fits[1].read_column("NAME", as_bytes=True)
            assert raw.dtype == np.dtype("S8")
            assert raw[0] == b"good    "
            assert raw[1] == b"bad\xe9stf "       # non-ASCII preserved
            assert raw[2] == b"AB\x00DEF  "      # embedded null preserved
            assert raw[3] == b"r3      "
            assert raw[4] == b""                  # numpy strips all-null


def test_as_bytes_rejected_on_non_character_column():
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _file_with_dirty_strings(tmp)
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="character"):
                fits[1].read_column("ID", as_bytes=True)


# ---------------------------------------------------------------------------
# unsupported TFORM types are rejected up front
# ---------------------------------------------------------------------------


def test_variable_length_p_column_rejected():
    fields = [("VLA", "1PE(100)")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(8, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="variable-length"):
                fits[1].read()


def test_bit_x_column_rejected():
    # TFORM='8X' = 8 bits packed in 1 byte
    fields = [("BITS", "8X")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, _bintable_cards(1, 0, fields), b"")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="bit columns"):
                fits[1].read()


# ---------------------------------------------------------------------------
# dtype property: structured dtype matches what read() returns
# ---------------------------------------------------------------------------


def test_dtype_property_matches_read_output():
    """TableHDU.dtype should equal the dtype of the array read()
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
    """Returned rows must be in the user's requested order, not on-disk
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
    """Matches read(rows=) dedup behavior: duplicates dropped, first
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
    """Make sure A-column decode goes through __getitem__ just like
    through read()."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ten_row_file(tmp)
        with rustfits.FITS(fname, "r") as fits:
            a = fits[1][[2, 0]]
            assert a["NAME"].tolist() == ["r  2", "r  0"]
            assert a["ID"].tolist() == [1002, 1000]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
