"""
Tests for HDU accessor properties.

Base HDU:           extname, extver, has_data
ImageHDU:           shape, dtype, ndim, size, bitpix, __len__
TableHDU:           nrows, ncols, colnames, __len__
AsciiTableHDU:      nrows, __len__
"""

import os
import struct
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# byte-level fixture helpers (mirrors test_table_read.py / test_repr.py)
# ---------------------------------------------------------------------------

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


def _primary_no_data(extras=()):
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        *extras,
        "END",
    ]


def _image_primary(bitpix, dims, extras=()):
    cards = [
        "SIMPLE  =                    T",
        f"BITPIX  = {bitpix:>20d}",
        f"NAXIS   = {len(dims):>20d}",
    ]
    for i, d in enumerate(dims, start=1):
        cards.append(f"NAXIS{i:<3d}= {d:>20d}")
    cards.append("EXTEND  =                    T")
    cards.extend(extras)
    cards.append("END")
    return cards


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
    for i, (ttype, tform) in enumerate(fields, start=1):
        cards.append(f"TTYPE{i:<3d}= '{ttype:<8s}'")
        cards.append(f"TFORM{i:<3d}= '{tform:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


def _ascii_table_ext(naxis1, naxis2, extras=()):
    cards = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {naxis1:>20d}",
        f"NAXIS2  = {naxis2:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    0",
        *extras,
        "END",
    ]
    return cards


# ---------------------------------------------------------------------------
# extname / extver (base HDU)
# ---------------------------------------------------------------------------


def test_extname_present():
    """EXTNAME set on an extension HDU → returned verbatim."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            4, 1, [("X", "1J")], extras=["EXTNAME = 'CATALOG '"],
        )
        _write_file(fname, (primary, b""), (ext, bytes(4)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].extname == "CATALOG"


def test_extname_absent_returns_none():
    """No EXTNAME → extname returns None."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].extname is None


def test_extver_default_is_one():
    """EXTVER absent → default 1 per FITS standard."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].extver == 1


def test_extver_set():
    """EXTVER set in header → returned verbatim."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            4, 1, [("X", "1J")],
            extras=["EXTNAME = 'SCI     '", "EXTVER  =                    7"],
        )
        _write_file(fname, (primary, b""), (ext, bytes(4)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].extver == 7


# ---------------------------------------------------------------------------
# has_data (base HDU)
# ---------------------------------------------------------------------------


def test_has_data_primary_naxis_zero_is_false():
    """Primary HDU with NAXIS=0 has no data section."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].has_data is False


def test_has_data_image_true():
    """Image with positive NAXISn → has data."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_image_primary(-32, [3, 4]), bytes(3 * 4 * 4)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].has_data is True


def test_has_data_table_true():
    """Table with non-zero NAXIS2 → has data."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 3, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, struct.pack(">3i", 1, 2, 3)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].has_data is True


def test_has_data_empty_table_is_false():
    """Table with NAXIS2=0 (no rows) → no data."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 0, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, b""))
        with rustfits.FITS(fname) as fits:
            assert fits[1].has_data is False


def test_has_data_zero_axis_image_is_false():
    """Image with a zero-sized axis → no data section bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        # NAXIS=2, NAXIS1=0, NAXIS2=5: total pixels = 0
        _write_file(fname, (_image_primary(-32, [0, 5]), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].has_data is False


# ---------------------------------------------------------------------------
# ImageHDU accessors
# ---------------------------------------------------------------------------


def test_image_shape_2d():
    """shape is a tuple in numpy axis order (slowest first)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        # FITS NAXIS1=4, NAXIS2=3 → numpy shape (3, 4)
        _write_file(fname, (_image_primary(-64, [4, 3]), bytes(4 * 3 * 8)))
        with rustfits.FITS(fname) as fits:
            img = fits[0]
            assert img.shape == (3, 4)
            assert isinstance(img.shape, tuple)


def test_image_shape_1d():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_image_primary(32, [10]), bytes(10 * 4)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].shape == (10,)


def test_image_shape_naxis_zero_is_empty_tuple():
    """Primary HDU with no data → shape is ()."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].shape == ()


@pytest.mark.parametrize(
    "bitpix,expected_dtype",
    [
        (8, np.uint8),
        (16, np.int16),
        (32, np.int32),
        (64, np.int64),
        (-32, np.float32),
        (-64, np.float64),
    ],
)
def test_image_dtype_matches_bitpix(bitpix, expected_dtype):
    """ImageHDU.dtype is a numpy.dtype matching BITPIX."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        bpp = abs(bitpix) // 8
        _write_file(fname, (_image_primary(bitpix, [2, 2]), bytes(4 * bpp)))
        with rustfits.FITS(fname) as fits:
            dt = fits[0].dtype
            assert isinstance(dt, np.dtype)
            assert dt == np.dtype(expected_dtype)


def test_image_ndim():
    """ndim is len(shape)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname, (_image_primary(-32, [2, 3, 4]), bytes(2 * 3 * 4 * 4))
        )
        with rustfits.FITS(fname) as fits:
            assert fits[0].ndim == 3


def test_image_ndim_naxis_zero():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].ndim == 0


def test_image_size():
    """size is the total pixel count (product of all NAXISn)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_image_primary(-32, [4, 3]), bytes(12 * 4)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].size == 12


def test_image_size_naxis_zero():
    """NAXIS=0 → size = 0."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert fits[0].size == 0


def test_image_bitpix_raw():
    """bitpix returns the raw FITS value (positive or negative)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_image_primary(-64, [2, 2]), bytes(4 * 8)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].bitpix == -64


def test_image_len_is_shape_zero():
    """len(image_hdu) == shape[0] (numpy convention)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        # numpy shape (3, 4) → len == 3
        _write_file(fname, (_image_primary(-32, [4, 3]), bytes(12 * 4)))
        with rustfits.FITS(fname) as fits:
            assert len(fits[0]) == 3


def test_image_len_naxis_zero_is_zero():
    """No data section → len = 0."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            assert len(fits[0]) == 0


# ---------------------------------------------------------------------------
# TableHDU accessors
# ---------------------------------------------------------------------------


def test_table_nrows():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 7, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, bytes(28)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 7


def test_table_ncols():
    """ncols returns TFIELDS."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            4 + 8 + 4, 1,
            [("A", "1J"), ("B", "1D"), ("C", "1J")],
        )
        _write_file(fname, (primary, b""), (ext, bytes(16)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].ncols == 3


def test_table_colnames_preserves_case_and_order():
    """colnames is a tuple, in file order, with verbatim case."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            4 + 4 + 4, 1,
            [("RA_deg", "1J"), ("Flux", "1J"), ("id", "1J")],
        )
        _write_file(fname, (primary, b""), (ext, bytes(12)))
        with rustfits.FITS(fname) as fits:
            names = fits[1].colnames
            assert isinstance(names, tuple)
            assert names == ("RA_deg", "Flux", "id")


def test_table_len_is_nrows():
    """len(table_hdu) == nrows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 5, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, bytes(20)))
        with rustfits.FITS(fname) as fits:
            assert len(fits[1]) == 5
            assert len(fits[1]) == fits[1].nrows


def test_table_empty_table_len_zero():
    """Table with NAXIS2=0 has len=0."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 0, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, b""))
        with rustfits.FITS(fname) as fits:
            assert len(fits[1]) == 0
            assert fits[1].nrows == 0


# ---------------------------------------------------------------------------
# AsciiTableHDU accessors
# ---------------------------------------------------------------------------


def test_ascii_table_nrows_and_len():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _ascii_table_ext(20, 4)
        _write_file(fname, (primary, b""), (ext, bytes(80)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 4
            assert len(fits[1]) == 4
