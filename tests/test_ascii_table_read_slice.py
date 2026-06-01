"""
ASCII-table read with rows= / columns= / mask_null / __getitem__.

Phase 2 surface:
- read(rows=, columns=, scale=, mask_null=)
- read_column(name, ...)
- __getitem__: int -> np.void; slice; list-of-int; str -> subset;
  list-of-str -> subset
- AsciiSingleColumnSubset / AsciiColumnSubset .read() and .[rows]
- TNULL masking returns numpy.ma.MaskedArray
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# fixture helpers (same shape as test_ascii_table_read.py)
# ---------------------------------------------------------------------------

CARDS_PER_BLOCK = 36
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b, pad_byte=b" "):
    return b + pad_byte * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    with open(path, "wb") as f:
        for cards, data, pad in parts:
            f.write(_pad_cards(cards))
            if data:
                f.write(_pad_to_block(data, pad))


def _primary_no_data():
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]


def _ascii_ext(naxis1, naxis2, cols, extras=()):
    cards = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {naxis1:>20d}",
        f"NAXIS2  = {naxis2:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        f"TFIELDS = {len(cols):>20d}",
    ]
    for i, c in enumerate(cols, start=1):
        cards.append(f"TTYPE{i:<3d}= '{c['name']:<8s}'")
        cards.append(f"TBCOL{i:<3d}= {c['tbcol']:>20d}")
        cards.append(f"TFORM{i:<3d}= '{c['tform']:<8s}'")
        if c.get("unit") is not None:
            cards.append(f"TUNIT{i:<3d}= '{c['unit']:<8s}'")
        if c.get("tscal") is not None:
            cards.append(f"TSCAL{i:<3d}= {c['tscal']!r:>20s}")
        if c.get("tzero") is not None:
            cards.append(f"TZERO{i:<3d}= {c['tzero']!r:>20s}")
        if c.get("tnull") is not None:
            cards.append(f"TNULL{i:<3d}= '{c['tnull']:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


def _make_file(tmp, rows_text, cols, extras=()):
    naxis1 = len(rows_text[0]) if rows_text else 0
    for r in rows_text:
        assert len(r) == naxis1
    naxis2 = len(rows_text)
    data = "".join(rows_text).encode("ascii")
    fname = os.path.join(tmp, "t.fits")
    _write_file(
        fname,
        (_primary_no_data(), b"", b" "),
        (_ascii_ext(naxis1, naxis2, cols, extras), data, b" "),
    )
    return fname


# Standard 3-col fixture used by many tests below.
# NAXIS1 = 19: A8 (cols 1-8), gap col 9, I4 (cols 10-13), gap col 14,
# F5.2 (cols 15-19).  Each row below is exactly 19 bytes:
#   positions 0-7 = NAME (rstrip-trimmed), pos 8 = gap, pos 9-12 = I4,
#   pos 13 = gap, pos 14-18 = F5.2.
_COLS = [
    {"name": "NAME", "tform": "A8", "tbcol": 1},
    {"name": "X", "tform": "I4", "tbcol": 10},
    {"name": "Y", "tform": "F5.2", "tbcol": 15},
]
# Build _ROWS programmatically so the column-width math is auditable.
_VALUES = [
    ("alice", 1, 1.50),
    ("bob", 2, -2.25),
    ("carol", 3, 3.00),
    ("dave", 4, -4.75),
    ("eve", 5, 5.00),
]
_ROWS = [f"{name:<8s} {x:4d} {y:5.2f}" for (name, x, y) in _VALUES]
assert all(len(r) == 19 for r in _ROWS), [len(r) for r in _ROWS]


def _std_fixture(tmp):
    return _make_file(tmp, _ROWS, _COLS)


# ---------------------------------------------------------------------------
# read(rows=, columns=, ...)
# ---------------------------------------------------------------------------


def test_read_rows_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(rows=slice(1, 4))
            assert len(arr) == 3
            assert list(arr["NAME"]) == ["bob", "carol", "dave"]
            np.testing.assert_array_equal(arr["X"], [2, 3, 4])
            np.testing.assert_allclose(
                arr["Y"], [-2.25, 3.0, -4.75], rtol=1e-5
            )


def test_read_rows_negative_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(rows=slice(-2, None))
            assert len(arr) == 2
            assert list(arr["NAME"]) == ["dave", "eve"]


def test_read_rows_list_dedup_first_occurrence():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(rows=[4, 0, 4, 2])
            # dedup keeps first occurrence; result order matches user
            assert len(arr) == 3
            assert list(arr["NAME"]) == ["eve", "alice", "carol"]


def test_read_rows_negative_index():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(rows=[-1, -3])
            assert list(arr["NAME"]) == ["eve", "carol"]


def test_read_rows_out_of_range_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(IndexError):
                f[1].read(rows=[0, 999])


def test_read_columns_subset_reorder():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(columns=["Y", "NAME"])
            assert arr.dtype.names == ("Y", "NAME")
            np.testing.assert_allclose(
                arr["Y"], [1.5, -2.25, 3.0, -4.75, 5.0], rtol=1e-5
            )


def test_read_columns_case_insensitive():
    """Lookup is case-insensitive; dtype carries on-disk case."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(columns=["x", "name"])
            # User passed "x" / "name"; on-disk names are "X" / "NAME".
            # Field names in the result come from the on-disk column
            # (AsciiColumn::clone preserves storage spelling).  Matches
            # the BINTABLE convention.
            assert arr.dtype.names == ("X", "NAME")
            np.testing.assert_array_equal(arr["X"], [1, 2, 3, 4, 5])


def test_read_columns_unknown_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="unknown column"):
                f[1].read(columns=["NOSUCH"])


def test_read_columns_duplicate_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="duplicate"):
                f[1].read(columns=["X", "X"])


def test_read_rows_and_columns_combined():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(rows=slice(1, 4), columns=["NAME"])
            assert len(arr) == 3
            assert list(arr["NAME"]) == ["bob", "carol", "dave"]


# ---------------------------------------------------------------------------
# read_column
# ---------------------------------------------------------------------------


def test_read_column_plain_ndarray():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            x = f[1].read_column("X")
            assert x.dtype == np.dtype("i8")
            assert x.shape == (5,)
            np.testing.assert_array_equal(x, [1, 2, 3, 4, 5])


def test_read_column_rows_subset():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            x = f[1].read_column("X", rows=[3, 0])
            np.testing.assert_array_equal(x, [4, 1])


def test_read_column_as_bytes_on_A():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            name = f[1].read_column("NAME", as_bytes=True)
            assert name.dtype == np.dtype("S8")
            # raw bytes, no trim — trailing spaces preserved
            assert name[0] == b"alice   "


def test_read_column_as_bytes_rejected_on_non_A():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="character"):
                f[1].read_column("X", as_bytes=True)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


def test_getitem_int_returns_record():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            rec = f[1][2]
            # numpy 0-d record (np.void); supports field-name indexing
            assert isinstance(rec, np.void)
            assert rec["NAME"] == "carol"
            assert rec["X"] == 3


def test_getitem_negative_int():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            rec = f[1][-1]
            assert rec["NAME"] == "eve"


def test_getitem_slice():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1][1:4]
            assert len(arr) == 3
            assert list(arr["NAME"]) == ["bob", "carol", "dave"]


def test_getitem_fancy_rows():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1][[4, 1, 3]]
            assert list(arr["NAME"]) == ["eve", "bob", "dave"]


def test_getitem_single_str_returns_subset():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            col = f[1]["NAME"]
            assert isinstance(col, rustfits.AsciiSingleColumnSubset)


def test_getitem_list_str_returns_subset():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            sub = f[1][["NAME", "X"]]
            assert isinstance(sub, rustfits.AsciiColumnSubset)


def test_getitem_empty_iterable_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="empty sequence"):
                f[1][[]]


def test_getitem_mixed_iterable_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="all int|all str"):
                f[1][["NAME", 5]]


# ---------------------------------------------------------------------------
# Subset objects
# ---------------------------------------------------------------------------


def test_single_column_subset_getitem():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            col = f[1]["X"]
            all_x = col[:]
            np.testing.assert_array_equal(all_x, [1, 2, 3, 4, 5])
            subset = col[1:4]
            np.testing.assert_array_equal(subset, [2, 3, 4])
            fancy = col[[4, 0]]
            np.testing.assert_array_equal(fancy, [5, 1])


def test_single_column_subset_read():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            col = f[1]["X"]
            all_x = col.read()
            np.testing.assert_array_equal(all_x, [1, 2, 3, 4, 5])
            subset = col.read(rows=[3, 1])
            np.testing.assert_array_equal(subset, [4, 2])


def test_single_column_subset_repr():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            col = f[1]["NAME"]
            r = repr(col)
            assert "AsciiTableColumn" in r
            assert "NAME" in r


def test_column_subset_getitem():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            sub = f[1][["NAME", "Y"]]
            arr = sub[:]
            assert arr.dtype.names == ("NAME", "Y")
            assert len(arr) == 5
            assert list(arr["NAME"]) == [
                "alice",
                "bob",
                "carol",
                "dave",
                "eve",
            ]
            sl = sub[1:3]
            assert list(sl["NAME"]) == ["bob", "carol"]


def test_column_subset_read():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            sub = f[1][["NAME", "Y"]]
            arr = sub.read(rows=slice(2, 5))
            assert arr.dtype.names == ("NAME", "Y")
            assert list(arr["NAME"]) == ["carol", "dave", "eve"]


def test_column_subset_repr():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            sub = f[1][["NAME", "X"]]
            r = repr(sub)
            assert "AsciiTableColumns" in r
            assert "NAME" in r
            assert "X" in r


# ---------------------------------------------------------------------------
# mask_null (TNULL string sentinel)
# ---------------------------------------------------------------------------


def test_mask_null_string_sentinel():
    """TNULL is a string; the trimmed field matches the trimmed TNULL.

    The TNULL short-circuit also makes the parse silent on null
    sentinels (otherwise "NA"/"NULL" would raise a parse error
    even when mask_null=False).  The cell defaults to the dtype's
    zero in that case.
    """
    cols = [
        {"name": "X", "tform": "I5", "tbcol": 1, "tnull": "  NA"},
        {"name": "Y", "tform": "I5", "tbcol": 6},
    ]
    rows = [
        "   42   10",
        "   NA   20",  # X null; Y normal
        "  100   30",  # both normal
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as f:
            # Without mask_null: NA short-circuited to zero
            plain = f[1].read()
            np.testing.assert_array_equal(plain["X"], [42, 0, 100])
            np.testing.assert_array_equal(plain["Y"], [10, 20, 30])

            # With mask_null: X cell 1 masked, others not
            masked = f[1].read(mask_null=True)
            assert isinstance(masked, np.ma.MaskedArray)
            mask = np.ma.getmaskarray(masked)
            assert list(mask["X"]) == [False, True, False]
            # Y has no TNULL, so all unmasked
            assert list(mask["Y"]) == [False, False, False]


def test_mask_null_single_column():
    cols = [{"name": "X", "tform": "I5", "tbcol": 1, "tnull": "NULL "}]
    rows = ["    1", " NULL", "    3"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as f:
            arr = f[1].read_column("X", mask_null=True)
            assert isinstance(arr, np.ma.MaskedArray)
            assert list(np.ma.getmaskarray(arr)) == [False, True, False]


def test_mask_null_no_tnull_returns_nomask():
    """No TNULL on any selected column -> MaskedArray with nomask."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            arr = f[1].read(mask_null=True)
            assert isinstance(arr, np.ma.MaskedArray)
            # No element is masked (whether mask is structured or nomask
            # depends on numpy version; check element-wise).
            assert not np.ma.getmaskarray(arr["X"]).any()


# ---------------------------------------------------------------------------
# tainted flag rejects reads
# ---------------------------------------------------------------------------


def test_taint_rejects_read():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _std_fixture(tmp)
        with rustfits.FITS(fname) as f:
            f[1]._force_taint()
            with pytest.raises(IOError):
                f[1].read()
            with pytest.raises(IOError):
                f[1][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
