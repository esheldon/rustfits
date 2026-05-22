"""
Tests for TNULLn integer-sentinel masking on BINTABLE reads.

`mask_null=True` returns a numpy.ma.MaskedArray with per-element masks
set wherever a stored integer equals the column's TNULLn.  Compare is
in stored (pre-scaling) space — independent of TSCAL/TZERO.

Same byte-level BINTABLE construction approach as test_table_read.py.
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
    with open(path, "wb") as f:
        f.write(_pad_cards(_PRIMARY))
        f.write(_pad_cards(ext_cards))
        f.write(_pad_to_block(data_bytes))


def _bintable_cards(naxis1, naxis2, fields, extras=None):
    """
    Minimal BINTABLE header cards + optional extras.

    `fields` is a list of (ttype, tform, optional tdim) tuples.  `extras`
    is a list of fully-formed card strings inserted before END.
    """
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
    if extras:
        cards.extend(extras)
    cards.append("END")
    return cards


def _tnull_card(col_index, value):
    return f"TNULL{col_index:<3d}= {value:>20d}"


def _tscal_card(col_index, value):
    return f"TSCAL{col_index:<3d}= {value:>20s}"


def _tzero_card(col_index, value):
    return f"TZERO{col_index:<3d}= {value:>20s}"


# ---------------------------------------------------------------------------
# default: mask_null=False returns a plain structured ndarray
# ---------------------------------------------------------------------------


def test_default_returns_plain_ndarray():
    """
    Without mask_null=True the result is a regular structured array,
    even when TNULL is set and rows hit it."""
    fields = [("X", "1I")]
    data = struct.pack(">hhh", 10, -32768, 30)
    extras = [_tnull_card(1, -32768)]
    cards = _bintable_cards(2, 3, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read()
            assert not isinstance(a, np.ma.MaskedArray)
            # Sentinel rows come through as the raw integer.
            assert a["X"].tolist() == [10, -32768, 30]


# ---------------------------------------------------------------------------
# basic masking for each integer TFORM letter
# ---------------------------------------------------------------------------


def test_tnull_int16_basic():
    fields = [("X", "1I")]
    data = struct.pack(">hhhh", 10, -32768, 30, -32768)
    cards = _bintable_cards(2, 4, fields, extras=[_tnull_card(1, -32768)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert isinstance(a, np.ma.MaskedArray)
            assert a["X"].mask.tolist() == [False, True, False, True]
            # Data values still present in the underlying array.
            assert a["X"].data.tolist() == [10, -32768, 30, -32768]


def test_tnull_int32_basic():
    fields = [("X", "1J")]
    data = struct.pack(">iiii", 1, -1, 2, -1)
    cards = _bintable_cards(4, 4, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].mask.tolist() == [False, True, False, True]


def test_tnull_int64_basic():
    fields = [("X", "1K")]
    data = struct.pack(">qqq", 10**10, -(2**63), 10**10 + 1)
    cards = _bintable_cards(8, 3, fields, extras=[_tnull_card(1, -(2**63))])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].mask.tolist() == [False, True, False]


def test_tnull_byte_basic():
    fields = [("X", "1B")]
    data = bytes([5, 255, 7, 255])
    cards = _bintable_cards(1, 4, fields, extras=[_tnull_card(1, 255)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].mask.tolist() == [False, True, False, True]


# ---------------------------------------------------------------------------
# nomask paths
# ---------------------------------------------------------------------------


def test_tnull_no_sentinel_hit():
    """
    TNULLn declared in header but no row matches: mask all-False but
    the result is still a MaskedArray (consistent return type)."""
    fields = [("X", "1J")]
    data = struct.pack(">iii", 1, 2, 3)
    cards = _bintable_cards(4, 3, fields, extras=[_tnull_card(1, -99999)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert isinstance(a, np.ma.MaskedArray)
            assert a["X"].mask.tolist() == [False, False, False]


def test_mask_null_true_with_no_tnull_anywhere():
    """
    mask_null=True on a table where no column declares TNULL.  Result
    is MaskedArray; nothing is masked.

    Note: for STRUCTURED arrays numpy.ma always materializes a bool
    structured mask on construction, so `.mask is np.ma.nomask` does
    not hold here — we verify the weaker (and more meaningful)
    invariant that no element is actually masked.
    """
    fields = [("X", "1D")]
    data = struct.pack(">ddd", 1.0, 2.0, 3.0)
    cards = _bintable_cards(8, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert isinstance(a, np.ma.MaskedArray)
            assert not a["X"].mask.any()


def test_tnull_silently_ignored_for_float_column():
    """
    TNULL on an E (float) column has no meaning per the FITS spec
    and is silently ignored: no row is masked even with mask_null=True.
    """
    fields = [("X", "1E")]
    data = struct.pack(">fff", 1.0, 2.0, 3.0)
    cards = _bintable_cards(4, 3, fields, extras=[_tnull_card(1, 0)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert isinstance(a, np.ma.MaskedArray)
            assert not a["X"].mask.any()


# ---------------------------------------------------------------------------
# multi-column tables: per-field masking
# ---------------------------------------------------------------------------


def test_tnull_per_field_isolation():
    """
    Multi-column: TNULL set only on one int field; other fields'
    masks stay all-False even when their values happen to coincide
    with the masked column's sentinel."""
    fields = [("A", "1J"), ("B", "1J"), ("C", "1D")]
    rows = b""
    for a, b, c in [(10, -1, 1.5), (-1, 99, 2.5), (20, -1, 3.5)]:
        rows += struct.pack(">iid", a, b, c)
    cards = _bintable_cards(16, 3, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["A"].mask.tolist() == [False, True, False]
            # B has matching -1 values but no TNULL → not masked.
            assert a["B"].mask.tolist() == [False, False, False]
            # C is float, not eligible.
            assert a["C"].mask.tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# interplay with TSCAL/TZERO scaling
# ---------------------------------------------------------------------------


def test_tnull_with_unsigned_int_trick():
    """
    I + TZERO=32768 produces u2 output (unsigned trick).  Mask is
    computed against the STORED i2 sentinel — independent of scaling."""
    # Stored values: [0, -32768, 32767, -32768] → physical u2:
    # [32768, 0, 65535, 0]
    data = struct.pack(">hhhh", 0, -32768, 32767, -32768)
    fields = [("X", "1I")]
    extras = [
        _tscal_card(1, "1"),
        _tzero_card(1, "32768"),
        _tnull_card(1, -32768),
    ]
    cards = _bintable_cards(2, 4, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].dtype == np.uint16
            assert a["X"].mask.tolist() == [False, True, False, True]
            # Underlying data still reflects the scaled physical values.
            assert a["X"].data.tolist() == [32768, 0, 65535, 0]


def test_tnull_with_general_scaling():
    """I + TSCAL=2 → f8 output.  Mask computed on stored i2 sentinel."""
    data = struct.pack(">hhh", 5, -1, 10)
    fields = [("X", "1I")]
    extras = [_tscal_card(1, "2.0"), _tnull_card(1, -1)]
    cards = _bintable_cards(2, 3, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].dtype == np.float64
            assert a["X"].mask.tolist() == [False, True, False]
            # Physical = 2 * stored: [10.0, -2.0, 20.0]
            np.testing.assert_allclose(a["X"].data, [10.0, -2.0, 20.0])


def test_tnull_scale_false():
    """
    scale=False returns raw stored ints.  mask_null still masks the
    sentinel rows since the compare is in stored space anyway."""
    data = struct.pack(">hhh", 5, -1, 10)
    fields = [("X", "1I")]
    extras = [_tscal_card(1, "2.0"), _tnull_card(1, -1)]
    cards = _bintable_cards(2, 3, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True, scale=False)
            assert a["X"].dtype == np.int16
            assert a["X"].mask.tolist() == [False, True, False]
            assert a["X"].data.tolist() == [5, -1, 10]


# ---------------------------------------------------------------------------
# multi-element columns: repeat > 1 and TDIM
# ---------------------------------------------------------------------------


def test_tnull_repeat_gt_1_per_element():
    """3J column: per-element mask, shape (n_rows, 3)."""
    fields = [("X", "3J")]
    rows = struct.pack(">iii", 1, -1, 3) + struct.pack(">iii", -1, 5, -1)
    cards = _bintable_cards(12, 2, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            assert a["X"].shape == (2, 3)
            assert a["X"].mask.tolist() == [
                [False, True, False],
                [True, False, True],
            ]


def test_tnull_with_tdim_reshape():
    """
    6J with TDIM=(3,2) → per-row shape (2, 3) (reversed TDIM).
    Mask reshapes alongside the data."""
    fields = [("X", "6J", "(3,2)")]
    rows = b""
    for vals in [[1, -1, 3, 4, -1, 6], [-1, 8, 9, 10, 11, -1]]:
        rows += struct.pack(">6i", *vals)
    cards = _bintable_cards(24, 2, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True)
            # numpy axis order is reversed-TDIM, so (2, 3).
            assert a["X"].shape == (2, 2, 3)
            # Mask shape mirrors data shape.
            assert a["X"].mask.shape == (2, 2, 3)
            # In FORTRAN order the row [1,-1,3,4,-1,6] reshapes to:
            # column 0 = [1, -1, 3], column 1 = [4, -1, 6].
            # numpy reverses dims → shape (2, 3): rows are TDIM-cols.
            assert a["X"].mask[0].tolist() == [
                [False, True, False],
                [False, True, False],
            ]


# ---------------------------------------------------------------------------
# row subset
# ---------------------------------------------------------------------------


def test_tnull_with_rows_subset():
    """
    rows= subset: mask is aligned with the output (user-requested)
    order, not the on-disk order."""
    fields = [("X", "1J")]
    rows = struct.pack(">5i", 10, -1, 30, -1, 50)
    cards = _bintable_cards(4, 5, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True, rows=[4, 1, 0])
            assert a["X"].data.tolist() == [50, -1, 10]
            assert a["X"].mask.tolist() == [False, True, False]


def test_tnull_with_columns_subset():
    """columns= subset: mask is built only for the selected columns."""
    fields = [("A", "1J"), ("B", "1J")]
    rows = struct.pack(">6i", 10, -1, 20, -2, 30, -1)
    extras = [_tnull_card(1, -1), _tnull_card(2, -2)]
    cards = _bintable_cards(8, 3, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read(mask_null=True, columns=["B"])
            assert a.dtype.names == ("B",)
            assert a["B"].mask.tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# read_column (single-column path)
# ---------------------------------------------------------------------------


def test_read_column_tnull_basic():
    """
    read_column with mask_null=True on an int col with TNULL → plain
    MaskedArray (not structured)."""
    fields = [("X", "1J")]
    data = struct.pack(">iii", 7, -1, 9)
    cards = _bintable_cards(4, 3, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            col = fits[1].read_column("X", mask_null=True)
            assert isinstance(col, np.ma.MaskedArray)
            assert col.dtype == np.int32
            assert col.mask.tolist() == [False, True, False]
            assert col.data.tolist() == [7, -1, 9]


def test_read_column_tnull_default_off():
    """Default mask_null=False returns a plain ndarray."""
    fields = [("X", "1J")]
    data = struct.pack(">iii", 7, -1, 9)
    cards = _bintable_cards(4, 3, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            col = fits[1].read_column("X")
            assert not isinstance(col, np.ma.MaskedArray)
            assert col.tolist() == [7, -1, 9]


def test_read_column_tnull_no_tnull():
    """
    read_column with mask_null=True on a col without TNULL: returns
    a MaskedArray with nomask (consistent type, zero overhead)."""
    fields = [("X", "1D")]
    data = struct.pack(">ddd", 1.0, 2.0, 3.0)
    cards = _bintable_cards(8, 3, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            col = fits[1].read_column("X", mask_null=True)
            assert isinstance(col, np.ma.MaskedArray)
            assert col.mask is np.ma.nomask


def test_read_column_tnull_multi_element():
    """
    read_column with mask_null=True on a 3J col → MaskedArray shape
    (n_rows, 3) with per-element mask."""
    fields = [("X", "3J")]
    rows = struct.pack(">iii", 1, -1, 3) + struct.pack(">iii", -1, 5, -1)
    cards = _bintable_cards(12, 2, fields, extras=[_tnull_card(1, -1)])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, rows)
        with rustfits.FITS(fname) as fits:
            col = fits[1].read_column("X", mask_null=True)
            assert col.shape == (2, 3)
            assert col.mask.tolist() == [
                [False, True, False],
                [True, False, True],
            ]


# ---------------------------------------------------------------------------
# variable-length: rejection up-front when mask_null=True
# ---------------------------------------------------------------------------


def _vla_with_tnull_cards():
    """Build a 1PJ VLA column with TNULL1 set in the header."""
    descriptors = struct.pack(">ii", 3, 0)
    heap = struct.pack(">iii", -1, 0, 1)
    fields = [("V", "1PJ(10)")]
    cards = _bintable_cards(8, 1, fields)
    cards = [
        f"PCOUNT  = {len(heap):>20d}" if c.startswith("PCOUNT") else c
        for c in cards
    ]
    cards = cards[:-1] + [_tnull_card(1, -1), "END"]
    return cards, descriptors + heap


def test_vla_tnull_mask_null_true_raises():
    """
    VLA + TNULL + mask_null=True is not yet supported and must error
    cleanly before any I/O."""
    cards, data = _vla_with_tnull_cards()
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError, match="variable-length"):
                fits[1].read(mask_null=True)
            with pytest.raises(ValueError, match="variable-length"):
                fits[1].read_column("V", mask_null=True)


def test_vla_tnull_mask_null_false_ok():
    """
    VLA with TNULL in header is fine to read when mask_null=False
    (the default).  Result is the usual Object-dtype column."""
    cards, data = _vla_with_tnull_cards()
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_bintable(fname, cards, data)
        with rustfits.FITS(fname) as fits:
            a = fits[1].read()
            assert a["V"][0].tolist() == [-1, 0, 1]
