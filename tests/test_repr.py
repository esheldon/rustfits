"""
Tests for the multi-line, fitsio-style __repr__ on FITS and HDU objects.

The reprs are designed for interactive REPL use: typing `fits` + Enter
calls __repr__, so the rich display has to live there (not in __str__).

Tests check that the expected lines appear in the right order without
asserting exact whitespace counts, so small formatting tweaks don't
break them.

Compressed-image coverage at the bottom uses rustfits itself to
generate fixtures for Gzip1/Gzip2/Rice1/Hcompress1; the PLIO_1 case
falls back to fitsio (our encoder isn't implemented yet) and skips
when fitsio is unavailable.
"""

import os
import struct
import sys
import tempfile

import numpy as np
import pytest

import rustfits


CARDS_PER_BLOCK = 36
BLOCK = 2880


def _show(r):
    """
    Print the repr that the surrounding test is asserting on.

    The assertions do the precise machine check; this lets a human
    also eyeball the rendered repr (spacing, alignment, wording).
    Visible when running `python tests/test_repr.py` directly or
    `pytest -s -v ...`; hidden under pytest's normal output capture.

    Leading newline + dashed separator break the repr off from the
    test-name line pytest writes, and visually divide consecutive
    reprs when a single test calls _show more than once.
    """
    print()
    print("-" * 70)
    print(r, end="")
    sys.stdout.flush()


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b):
    return b + b"\x00" * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    """
    Write a FITS file from a sequence of (cards, data_bytes) tuples,
    one per HDU (cards already includes END)."""
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


def _image_extension(bitpix, dims, extras=()):
    cards = [
        "XTENSION= 'IMAGE   '",
        f"BITPIX  = {bitpix:>20d}",
        f"NAXIS   = {len(dims):>20d}",
    ]
    for i, d in enumerate(dims, start=1):
        cards.append(f"NAXIS{i:<3d}= {d:>20d}")
    cards.append("PCOUNT  =                    0")
    cards.append("GCOUNT  =                    1")
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
    for i, (ttype, tform, *opt) in enumerate(fields, start=1):
        cards.append(f"TTYPE{i:<3d}= '{ttype:<8s}'")
        cards.append(f"TFORM{i:<3d}= '{tform:<8s}'")
        if opt:
            cards.append(f"TDIM{i:<4d}= '{opt[0]:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


def _ascii_table_ext(naxis1, naxis2, extras=()):
    """
    Minimal ASCII table extension cards.  We don't try to make this
    actually readable as a table — just enough to be parsed as an
    AsciiTableHDU so we can repr it."""
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
# FITS repr
# ---------------------------------------------------------------------------


def test_fits_repr_image_only_default_mode():
    """
    Primary-only file opened with default mode ('r').  Repr shows
    file, mode, and the single IMAGE_HDU line."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        _write_file(fname, (_image_primary(-64, [4, 3]), bytes(4 * 3 * 8)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits)
            _show(r)
            assert f"file: {fname}" in r
            assert "mode: r" in r
            assert "extnum  hdutype" in r
            assert "0" in r
            assert "IMAGE_HDU" in r
            # No extensions → only one HDU line.
            assert r.count("IMAGE_HDU") == 1
            assert "BINARY_TBL" not in r


def test_fits_repr_image_and_table():
    """
    Primary image + binary table extension with EXTNAME.  Repr
    shows both HDU lines in order, BINARY_TBL row carries the EXTNAME.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "mix.fits")
        primary = _image_primary(-32, [10])
        primary_data = bytes(10 * 4)
        ext = _bintable_ext(
            8,
            2,
            [("X", "1D")],
            extras=["EXTNAME = 'MYTABLE '"],
        )
        ext_data = struct.pack(">dd", 1.0, 2.0)
        _write_file(fname, (primary, primary_data), (ext, ext_data))
        with rustfits.FITS(fname) as fits:
            r = repr(fits)
            _show(r)
            assert "IMAGE_HDU" in r
            assert "BINARY_TBL" in r
            assert "MYTABLE" in r
            # IMAGE_HDU appears before BINARY_TBL.
            assert r.index("IMAGE_HDU") < r.index("BINARY_TBL")


def test_fits_repr_ascii_table_label():
    """
    ASCII table extension shows 'ASCII_TBL' in the FITS-level
    table."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "ascii.fits")
        primary = _primary_no_data()
        ext = _ascii_table_ext(20, 3)
        _write_file(fname, (primary, b""), (ext, bytes(60)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits)
            _show(r)
            assert "ASCII_TBL" in r


def test_fits_repr_mode_rplus():
    """Opening with mode='r+' shows 'mode: r+' in the repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "rw.fits")
        _write_file(fname, (_image_primary(8, []), b""))
        with rustfits.FITS(fname, "r+") as fits:
            r = repr(fits)
            _show(r)
            assert "mode: r+" in r


def test_fits_repr_closed_file():
    """
    After close() the repr shows 'status: closed' and skips the
    per-HDU table."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "closed.fits")
        _write_file(fname, (_image_primary(8, []), b""))
        fits = rustfits.FITS(fname)
        fits.close()
        r = repr(fits)
        _show(r)
        assert "status: closed" in r
        assert "extnum" not in r


# ---------------------------------------------------------------------------
# ImageHDU repr
# ---------------------------------------------------------------------------


def test_image_repr_f8_2d():
    """
    BITPIX=-64 + dims = (4, 3) → 'data type: f8' and numpy-order
    dims [3, 4] (FITS NAXIS1=4 is the fastest axis → trailing in
    numpy)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        _write_file(fname, (_image_primary(-64, [4, 3]), bytes(4 * 3 * 8)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert "type: IMAGE_HDU" in r
            assert "extension: 0" in r
            assert "data type: f8" in r
            # numpy axis order: slowest first.  FITS NAXIS1=4, NAXIS2=3
            # → numpy shape (3, 4).
            assert "dims: [3, 4]" in r


def test_image_repr_i4_1d():
    """1D int32 image."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        _write_file(fname, (_image_primary(32, [10]), bytes(10 * 4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert "data type: i4" in r
            assert "dims: [10]" in r


def test_image_repr_primary_no_data():
    """Primary HDU with NAXIS=0 — repr shows dims: []"""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert "dims: []" in r


def test_image_repr_extname_only_when_set():
    """EXTNAME line appears when EXTNAME is set, otherwise absent."""
    with tempfile.TemporaryDirectory() as tmp:
        # Primary has no EXTNAME → no extname line.
        # Extension carries EXTNAME → line appears.
        fname = os.path.join(tmp, "img.fits")
        primary = _primary_no_data()
        ext = _image_extension(-64, [2, 2], extras=["EXTNAME = 'SCI     '"])
        _write_file(fname, (primary, b""), (ext, bytes(2 * 2 * 8)))
        with rustfits.FITS(fname) as fits:
            r0 = repr(fits[0])
            r1 = repr(fits[1])
            _show(r0)
            _show(r1)
            assert "extname:" not in r0
            assert "extname: SCI" in r1


def test_image_repr_filename_present():
    """File path appears in each HDU's repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        _write_file(fname, (_image_primary(8, []), b""))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert f"file: {fname}" in r


# ---------------------------------------------------------------------------
# TableHDU repr
# ---------------------------------------------------------------------------


def test_table_repr_basic_scalar_columns():
    """One scalar int column: name + dtype, no shape annotation."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(4, 2, [("X", "1J")])
        _write_file(fname, (primary, b""), (ext, bytes(8)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL" in r
            assert "rows: 2" in r
            assert "column info:" in r
            # X column: scalar i4, no "array[" annotation.
            lines = r.splitlines()
            x_line = [ln for ln in lines if "X" in ln and "i4" in ln][0]
            assert "array[" not in x_line


def test_table_repr_repeat_array():
    """
    3J column shows 'array[3]'.  Column name uses mixed case to
    also confirm case is preserved in the repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(12, 1, [("Vec", "3J")])
        _write_file(fname, (primary, b""), (ext, bytes(12)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            vec_line = [ln for ln in r.splitlines() if "Vec" in ln][0]
            assert "i4" in vec_line
            assert "array[3]" in vec_line


def test_table_repr_tdim_array():
    """
    6J + TDIM=(3,2) shows array[2,3] (numpy axis order, slowest
    first)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(24, 1, [("M", "6J", "(3,2)")])
        _write_file(fname, (primary, b""), (ext, bytes(24)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            m_line = [ln for ln in r.splitlines() if "M" in ln and "i4" in ln][
                0
            ]
            assert "array[2,3]" in m_line


def test_table_repr_vla():
    """1PE column shows 'f4  array[var]'."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(8, 1, [("V", "1PE(10)")])
        _write_file(fname, (primary, b""), (ext, bytes(8)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            v_line = [ln for ln in r.splitlines() if "V" in ln and "f4" in ln][
                0
            ]
            assert "array[var]" in v_line


def test_table_repr_unsigned_int_trick_dtype():
    """
    A column with TSCAL=1, TZERO=32768 (the I→u2 unsigned trick)
    shows the scaled dtype (u2) in the repr, not the raw on-disk i2."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            2,
            1,
            [("U", "1I")],
            extras=[
                "TSCAL1  =                    1",
                "TZERO1  =                32768",
            ],
        )
        _write_file(fname, (primary, b""), (ext, bytes(2)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            u_line = [ln for ln in r.splitlines() if "U" in ln and "u2" in ln]
            assert u_line, f"expected a u2 line for column U; got:\n{r}"


def test_table_repr_character_column():
    """
    8A column shows 'U8' (one string of length 8 per cell).
    Column name uses CamelCase to also confirm case preservation."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(8, 1, [("Label", "8A")])
        _write_file(fname, (primary, b""), (ext, bytes(8)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            label_line = [ln for ln in r.splitlines() if "Label" in ln][0]
            assert "U8" in label_line
            assert "array[" not in label_line


def test_table_repr_with_extname():
    """EXTNAME appears as 'extname: NAME' line."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(
            4,
            1,
            [("X", "1J")],
            extras=["EXTNAME = 'CATALOG '"],
        )
        _write_file(fname, (primary, b""), (ext, bytes(4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "extname: CATALOG" in r


def test_table_repr_mixed_columns_alignment():
    """
    Columns of different name lengths align to the longest name.
    Names cover lower / Mixed / snake_case to also confirm the repr
    preserves the column-name case verbatim."""
    fields = [
        ("id", "1J"),
        ("Magnitude", "1D"),
        ("ra_deg", "3E"),
    ]
    naxis1 = 4 + 8 + 12  # i4 + f8 + 3*f4
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(naxis1, 1, fields)
        _write_file(fname, (primary, b""), (ext, bytes(naxis1)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            # All three columns appear with their case intact.
            for name, dtype in [
                ("id", "i4"),
                ("Magnitude", "f8"),
                ("ra_deg", "f4"),
            ]:
                line = [
                    ln for ln in r.splitlines() if name in ln and dtype in ln
                ]
                assert line, f"missing line for {name} {dtype} in:\n{r}"
            # Alignment check: dtype column is at the same offset on
            # every column row (since name field is left-padded).
            id_line = [
                ln for ln in r.splitlines() if "id" in ln and "i4" in ln
            ][0]
            mag_line = [ln for ln in r.splitlines() if "Magnitude" in ln][0]
            # Find the dtype-string start positions.
            assert id_line.index("i4") == mag_line.index("f8")


def test_table_repr_preserves_column_case():
    """
    FITS allows arbitrary case in TTYPE values.  Confirm the repr
    shows column names verbatim — no normalization to upper/lower."""
    fields = [
        ("lowerCase", "1J"),
        ("UPPERCASE", "1J"),
        ("MixedCase", "1J"),
        ("snake_case", "1J"),
    ]
    naxis1 = 4 * 4
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        primary = _primary_no_data()
        ext = _bintable_ext(naxis1, 1, fields)
        _write_file(fname, (primary, b""), (ext, bytes(naxis1)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            for name in ("lowerCase", "UPPERCASE", "MixedCase", "snake_case"):
                assert name in r, f"missing column name {name!r} in:\n{r}"
            # Confirm no accidental upper/lower normalization: if the
            # original spelling were dropped, only one of these would
            # remain.
            assert "LOWERCASE" not in r
            assert "MIXEDCASE" not in r
            assert "uppercase" not in r


# ---------------------------------------------------------------------------
# AsciiTableHDU repr
# ---------------------------------------------------------------------------


def test_ascii_table_repr_basic():
    """ASCII table extension shows ASCII_TBL type and row count."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "a.fits")
        primary = _primary_no_data()
        ext = _ascii_table_ext(20, 5)
        _write_file(fname, (primary, b""), (ext, bytes(100)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: ASCII_TBL" in r
            assert "rows: 5" in r
            # Column info is intentionally absent for ASCII tables.
            assert "column info:" not in r


def test_ascii_table_repr_with_extname():
    """ASCII table EXTNAME shows up in repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "a.fits")
        primary = _primary_no_data()
        ext = _ascii_table_ext(20, 1, extras=["EXTNAME = 'ASCIITAB'"])
        _write_file(fname, (primary, b""), (ext, bytes(20)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "extname: ASCIITAB" in r


# ---------------------------------------------------------------------------
# CompressedImageHDU repr
# ---------------------------------------------------------------------------
#
# Fixtures are generated by rustfits's own write side for the four
# algorithms we can write (Gzip1, Gzip2, Rice1, Hcompress1).  The
# PLIO_1 case falls back to fitsio (no rustfits encoder yet) and
# skips if fitsio isn't installed.
#
# Each test asserts:
#   - `type: COMPRESSED_IMAGE_HDU` header line
#   - The `compression:` line includes the algorithm's class name
#     (Gzip1, Rice1, ...) and tile_shape, plus any algorithm-
#     specific parameters in the config object's __repr__.
# The class name is the Pythonic form (Gzip1) not the FITS-spec
# ZCMPTYPE (GZIP_1) — `.compression.zcmptype` returns the FITS
# string when needed.


def _data_2d(shape, dtype="i4", seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, shape, dtype=dtype)


def test_compressed_image_repr_gzip1():
    """
    Gzip1 HDU repr inlines the config object's __repr__:
    `compression: Gzip1(tile_shape=..., heap_format='P')`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "g1.fits.fz")
        data = _data_2d((32, 48))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 24)),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: COMPRESSED_IMAGE_HDU" in r
            assert "data type: i4" in r
            assert "dims: [32, 48]" in r
            assert "compression: Gzip1(" in r
            assert "tile_shape=[16, 24]" in r
            assert "heap_format='P'" in r


def test_compressed_image_repr_gzip2():
    """Gzip2 HDU repr distinguishable from Gzip1."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "g2.fits.fz")
        data = _data_2d((32, 48), dtype="i2")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i2",
                data.shape,
                compress=rustfits.Gzip2(tile_shape=(16, 16)),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "compression: Gzip2(" in r
            assert "tile_shape=[16, 16]" in r
            # Sanity: not the wrong algorithm in the repr.
            assert "Gzip1(" not in r


def test_compressed_image_repr_rice1_default_blocksize():
    """Rice1 surfaces blocksize in the repr (default 32)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "r1.fits.fz")
        data = _data_2d((32, 48))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "compression: Rice1(" in r
            assert "tile_shape=[16, 24]" in r
            assert "blocksize=32" in r


def test_compressed_image_repr_rice1_custom_blocksize():
    """Non-default Rice1 blocksize round-trips into the repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "r1.fits.fz")
        data = _data_2d((32, 32))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(32, 32), blocksize=64),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "blocksize=64" in r


def test_compressed_image_repr_hcompress1_lossless():
    """Hcompress1 lossless (scale=0) shows scale=0, smooth=False."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h1.fits.fz")
        data = _data_2d((32, 48))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(tile_shape=(16, 24)),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "compression: Hcompress1(" in r
            assert "scale=0" in r
            assert "smooth=False" in r


def test_compressed_image_repr_hcompress1_lossy_smooth():
    """Hcompress1 with scale > 0 and smooth=True round-trips into repr."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h1.fits.fz")
        data = _data_2d((32, 32), dtype="i2")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i2",
                data.shape,
                compress=rustfits.Hcompress1(
                    tile_shape=(32, 32),
                    scale=8,
                    smooth=True,
                ),
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "scale=8" in r
            assert "smooth=True" in r


def test_compressed_image_repr_plio1():
    """
    PLIO_1 HDU repr: shows `compression: Plio1(...)`.  Our PLIO_1
    encoder isn't implemented yet, so the fixture comes from fitsio
    (skip if unavailable).
    """
    fitsio = pytest.importorskip("fitsio")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "p1.fits.fz")
        rng = np.random.default_rng(0)
        mask = rng.integers(0, 8, (32, 32), dtype="i4")
        fitsio.write(fname, mask, compress="PLIO_1", tile_dims=(16, 16))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: COMPRESSED_IMAGE_HDU" in r
            assert "compression: Plio1(" in r
            assert "tile_shape=[16, 16]" in r


def test_compressed_image_repr_with_extname():
    """EXTNAME line is present in compressed-HDU repr when set."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "e.fits.fz")
        data = _data_2d((16, 16))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 16)),
                extname="SCI",
            )
            fits[1].write(data)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "extname: SCI" in r


def test_compressed_image_repr_unknown_zcmptype_fallback():
    """
    Malformed file with present-but-unrecognized ZCMPTYPE: repr
    shows the raw ZCMPTYPE string rather than crashing.  Constructs
    a minimal ZIMAGE BINTABLE by hand and patches ZCMPTYPE to a
    bogus value via header write.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "x.fits.fz")
        # Write a valid GZIP_1 file first, then corrupt its ZCMPTYPE.
        data = _data_2d((16, 16))
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 16)),
            )
            fits[1].write(data)
        # ZCMPTYPE is a protected key (we can't __setitem__ it),
        # so patch the bytes on disk directly.  Find the card,
        # rewrite its value.
        with open(fname, "rb") as f:
            raw = f.read()
        target = b"ZCMPTYPE= 'GZIP_1  '"
        replacement = b"ZCMPTYPE= 'XYZ_99  '"
        assert target in raw, "expected ZCMPTYPE card not found in fixture"
        with open(fname, "wb") as f:
            f.write(raw.replace(target, replacement, 1))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            # Falls back to showing the raw ZCMPTYPE string verbatim.
            assert "compression: XYZ_99" in r


# ---------------------------------------------------------------------------
# TableHDU repr — wide dtype coverage
# ---------------------------------------------------------------------------
#
# Round-trip tables created by rustfits's own writer through every
# supported fixed dtype, then assert each column appears with the
# expected dtype label in the repr.  A separate test covers the
# variable-length + bit-packed special cases.


def test_table_repr_wide_dtype_matrix():
    """
    A table with one column of every supported fixed-width scalar
    dtype.  Repr shows each column with its expected dtype label and
    they line up under the widest column name.
    """
    dt = np.dtype(
        [
            ("flag", "?"),
            ("byte", "u1"),
            ("small", "i2"),
            ("mid", "i4"),
            ("big", "i8"),
            ("real32", "f4"),
            ("real64", "f8"),
            ("cplx64", "c8"),
            ("cplx128", "c16"),
            ("label", "S12"),
        ]
    )
    rows = np.zeros(3, dtype=dt)
    rows["label"] = [b"alpha", b"beta", b"gamma"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "wide.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(rows, extname="WIDE")
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL" in r
            assert "rows: 3" in r
            assert "extname: WIDE" in r
            # Each column appears with its expected scaled dtype label.
            # bool → ?, byte → u1, A12 → U12, etc.
            expected = [
                ("flag", "?"),
                ("byte", "u1"),
                ("small", "i2"),
                ("mid", "i4"),
                ("big", "i8"),
                ("real32", "f4"),
                ("real64", "f8"),
                ("cplx64", "c8"),
                ("cplx128", "c16"),
                ("label", "U12"),
            ]
            for name, dtype in expected:
                line = [
                    ln for ln in r.splitlines() if name in ln and dtype in ln
                ]
                assert line, f"missing line for {name} {dtype} in:\n{r}"
            # Alignment check: dtype-column starts at the same offset
            # in every column row.
            lines = {
                name: [
                    ln for ln in r.splitlines() if name in ln and dtype in ln
                ][0]
                for name, dtype in expected
            }
            ref_offset = lines["flag"].index("?")
            assert lines["byte"].index("u1") == ref_offset
            assert lines["label"].index("U12") == ref_offset


def test_table_repr_vla_and_bit_packed():
    """
    Wide-format VLA + bit-packed X coverage in one table.  Tests:
      - VLA numeric (f4) → 'f4  array[var]'
      - VLA string (PA)  → 'S   array[var]'
      - Fixed X (bit)    → '?   array[N]'   (bit_columns=)
      - VLA X (PX/QX)    → '?   array[var]' (bit_columns= + var_dtypes=)
    """
    dt = np.dtype(
        [
            ("samples", "O"),
            ("name", "O"),
            ("bits", "?", (16,)),
            ("flags", "O"),
        ]
    )
    rows = np.empty(2, dtype=dt)
    rows["samples"][0] = np.array([1.0, 2.0, 3.0], dtype="f4")
    rows["samples"][1] = np.array([], dtype="f4")
    rows["name"][0] = "first"
    rows["name"][1] = "second_row_string"
    rows["bits"] = False
    rows["flags"][0] = np.array([True, False, True], dtype="?")
    rows["flags"][1] = np.array([False, True], dtype="?")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "vla.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(
                rows,
                extname="MIXED",
                var_dtypes={
                    "samples": "f4",
                    "name": "S",
                    "flags": "?",
                },
                bit_columns=["bits", "flags"],
            )
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "extname: MIXED" in r

            def _col_line(col_name):
                """Lines whose first whitespace-stripped token is col_name."""
                hits = [
                    ln
                    for ln in r.splitlines()
                    if ln.strip().split(" ", 1)[:1] == [col_name]
                ]
                assert hits, f"no column-info line for {col_name!r} in:\n{r}"
                return hits[0]

            # VLA numeric.
            samples_line = _col_line("samples")
            assert "f4" in samples_line
            assert "array[var]" in samples_line
            # VLA string (PA) renders as "U array[var]".  Matches
            # the fixed-A path (which emits "U<n>") for consistency
            # — both kinds of string columns show "U" because the
            # read side returns Python str.  VLA cells have no
            # single length, so no suffix.
            name_line = _col_line("name")
            assert " U " in f" {name_line} "
            assert "array[var]" in name_line
            # Fixed X column — bool with array[16].
            bits_line = _col_line("bits")
            assert "?" in bits_line
            assert "array[16]" in bits_line
            # VLA X column — column_repr_info's VLA arm has no 'X'
            # case, so the dtype label is the raw FITS letter 'X'
            # rather than '?'.  Documents current behavior; revisit
            # if column_repr_info grows an X→? case for VLA.
            flags_line = _col_line("flags")
            assert " X " in f" {flags_line} "
            assert "array[var]" in flags_line


def test_table_repr_units_displayed():
    """TUNITn cards surface as `(unit)` after the dtype label."""
    dt = np.dtype([("ra", "f8"), ("flux", "f4"), ("flag", "i4")])
    rows = np.zeros(2, dtype=dt)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "u.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(
                rows,
                extname="CAT",
                units={"ra": "deg", "flux": "Jy"},
            )
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            # `(deg)` appears on the ra line; `(Jy)` on flux.
            ra_line = [
                ln for ln in r.splitlines() if "ra" in ln and "f8" in ln
            ][0]
            assert "(deg)" in ra_line
            flux_line = [
                ln for ln in r.splitlines() if "flux" in ln and "f4" in ln
            ][0]
            assert "(Jy)" in flux_line
            # Column without TUNIT has no trailing parens.
            flag_line = [
                ln for ln in r.splitlines() if "flag" in ln and "i4" in ln
            ][0]
            assert "(" not in flag_line


# ---------------------------------------------------------------------------
# CompressedTableHDU repr
# ---------------------------------------------------------------------------
#
# A ZTABLE shell carries the original schema in Z-prefixed cards; the
# repr surfaces the ORIGINAL row count / per-column dtypes (what you
# get back from .read()), the tile count, ZTILELEN, and the per-column
# algorithm.  Fixtures are built with rustfits's own ZTABLE writer.


def _mixed_table_rows(n=50):
    """A few-column table wide enough to give cfitsio's per-dtype
    algorithm defaults a chance to vary (u1/i2/i4/f4/f8 + a string)."""
    dt = np.dtype(
        [
            ("byte", "u1"),
            ("small", "i2"),
            ("mid", "i4"),
            ("real32", "f4"),
            ("real64", "f8"),
            ("label", "S8"),
        ]
    )
    rng = np.random.default_rng(0)
    rows = np.zeros(n, dtype=dt)
    rows["byte"] = rng.integers(0, 256, n, dtype="u1")
    rows["small"] = rng.integers(-100, 100, n, dtype="i2")
    rows["mid"] = rng.integers(-1000, 1000, n, dtype="i4")
    rows["real32"] = rng.standard_normal(n).astype("f4")
    rows["real64"] = rng.standard_normal(n)
    rows["label"] = [f"r{i:03d}".encode("ascii") for i in range(n)]
    return rows


def test_compressed_table_repr_default_algorithms():
    """
    compress=True picks cfitsio's per-dtype defaults; the repr's
    `compression:` line should list each user-visible column with
    its assigned algorithm name (GZIP_1 / GZIP_2 / RICE_1).  The
    `type:` line marks it as compressed; `rows:` shows the original
    (uncompressed) count.
    """
    rows = _mixed_table_rows(50)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "c.fits.fz")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(
                rows,
                extname="CMP",
                compress=True,
                ztilelen=10,
            )
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL (compressed)" in r
            assert "extname: CMP" in r
            assert "rows: 50" in r
            # Default tiling: 50 rows / 10-per-tile = 5 tiles.
            assert "tiles: 5" in r
            assert "rows per tile: 10" in r
            assert "compression:" in r
            # Per-cfitsio defaults: u1=GZIP_1, i2=GZIP_2, i4=RICE_1,
            # f4=GZIP_2, f8=GZIP_2, A=GZIP_1.  Each named column shows
            # up with its algorithm name on the compression line.
            comp_line = [ln for ln in r.splitlines() if "compression:" in ln][
                0
            ]
            assert "byte=GZIP_1" in comp_line
            assert "small=GZIP_2" in comp_line
            assert "mid=RICE_1" in comp_line
            assert "real32=GZIP_2" in comp_line
            assert "real64=GZIP_2" in comp_line
            assert "label=GZIP_1" in comp_line
            # column info: each column appears with its ORIGINAL
            # (uncompressed) dtype — NOT the on-disk 1QB descriptor.
            for name, dtype in [
                ("byte", "u1"),
                ("small", "i2"),
                ("mid", "i4"),
                ("real32", "f4"),
                ("real64", "f8"),
                ("label", "U8"),
            ]:
                line = [
                    ln for ln in r.splitlines() if name in ln and dtype in ln
                ]
                assert line, (
                    f"compressed table missing {name} {dtype} in:\n{r}"
                )


def test_compressed_table_repr_single_algorithm():
    """
    compress='GZIP_2' applies one algorithm to every column; the
    compression line shows the same algorithm for each.  Uses a
    string-free fixture because A columns only permit GZIP_1.
    """
    dt = np.dtype(
        [
            ("byte", "u1"),
            ("small", "i2"),
            ("mid", "i4"),
            ("real32", "f4"),
            ("real64", "f8"),
        ]
    )
    rng = np.random.default_rng(0)
    rows = np.zeros(20, dtype=dt)
    rows["byte"] = rng.integers(0, 256, 20, dtype="u1")
    rows["small"] = rng.integers(-100, 100, 20, dtype="i2")
    rows["mid"] = rng.integers(-1000, 1000, 20, dtype="i4")
    rows["real32"] = rng.standard_normal(20).astype("f4")
    rows["real64"] = rng.standard_normal(20)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "g2.fits.fz")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(rows, compress="GZIP_2", ztilelen=10)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL (compressed)" in r
            assert "rows: 20" in r
            assert "tiles: 2" in r
            comp_line = [ln for ln in r.splitlines() if "compression:" in ln][
                0
            ]
            for name in ("byte", "small", "mid", "real32", "real64"):
                assert f"{name}=GZIP_2" in comp_line


def test_compressed_table_repr_per_column_overrides():
    """
    compress={col: algo, ...} dict assigns per-column algorithms;
    the repr's compression line reflects the dict mapping.
    """
    rows = _mixed_table_rows(15)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "mix.fits.fz")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(
                rows,
                compress={
                    "byte": "GZIP_1",
                    "small": rustfits.Gzip2(),
                    "mid": rustfits.Rice1(blocksize=32),
                    "real32": rustfits.Gzip2(),
                    "real64": "GZIP_1",
                    "label": rustfits.Gzip1(),
                },
                ztilelen=5,
            )
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            comp_line = [ln for ln in r.splitlines() if "compression:" in ln][
                0
            ]
            assert "byte=GZIP_1" in comp_line
            assert "small=GZIP_2" in comp_line
            assert "mid=RICE_1" in comp_line
            assert "real32=GZIP_2" in comp_line
            assert "real64=GZIP_1" in comp_line
            assert "label=GZIP_1" in comp_line


def test_compressed_table_repr_with_vla_column():
    """
    A compressed table with a VLA column shows the VLA column with
    'array[var]' in column info, and includes it on the compression
    line.  VLA-bearing tables are a separate code path
    (dual-descriptor heap) and the repr should handle them too.
    """
    dt = np.dtype([("id", "i4"), ("samples", "O")])
    rows = np.empty(8, dtype=dt)
    rows["id"] = np.arange(8, dtype="i4")
    for i in range(8):
        rows["samples"][i] = np.arange(i + 1, dtype="f4")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "vla.fits.fz")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(
                rows,
                extname="VTAB",
                var_dtypes={"samples": "f4"},
                compress=True,
                ztilelen=4,
            )
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL (compressed)" in r
            assert "extname: VTAB" in r
            assert "rows: 8" in r
            assert "tiles: 2" in r
            # VLA column appears in column info as 'f4 array[var]'.
            samples_line = [
                ln for ln in r.splitlines() if "samples" in ln and "f4" in ln
            ][0]
            assert "array[var]" in samples_line
            # And on the compression line.  cfitsio defaults still
            # apply: id=RICE_1, samples=GZIP_1 (VLA-bearing tables
            # default to GZIP_1 for per-cell streams).
            comp_line = [ln for ln in r.splitlines() if "compression:" in ln][
                0
            ]
            assert "id=" in comp_line
            assert "samples=" in comp_line


def test_compressed_table_repr_short_table_no_tile_line():
    """
    Without explicit ztilelen=, the default tile size is large enough
    that small tables get a single tile.  The 'rows per tile:' line
    is still present when ZTILELEN is set (which the writer always
    emits, even on single-tile tables) — verify it shows a reasonable
    value rather than 0 / negative.
    """
    rows = _mixed_table_rows(5)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "s.fits.fz")
        with rustfits.FITS(fname, "w+") as fits:
            fits.write_table(rows, compress=True)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            assert "type: BINARY_TBL (compressed)" in r
            assert "rows: 5" in r
            tile_lines = [
                ln for ln in r.splitlines() if "rows per tile:" in ln
            ]
            assert tile_lines, f"expected a 'rows per tile:' line in:\n{r}"
            # Parse the integer after the colon and confirm > 0.
            n = int(tile_lines[0].split(":", 1)[1].strip())
            assert n > 0


if __name__ == "__main__":
    # Running the file directly drives the same tests pytest would,
    # but with output capture disabled and verbose mode on, so each
    # test's _show(r) output appears under its name.  Useful for
    # eyeballing changes to the repr formatting.
    sys.exit(pytest.main([__file__, "-s", "-v"]))
