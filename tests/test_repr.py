"""
Tests for the multi-line, fitsio-style __repr__ on FITS and HDU objects.

The reprs are designed for interactive REPL use: typing `fits` + Enter
calls __repr__, so the rich display has to live there (not in __str__).

Tests check that the expected lines appear in the right order without
asserting exact whitespace counts, so small formatting tweaks don't
break them.

TODO: add repr coverage for CompressedImageHDU here, alongside the
ImageHDU/TableHDU/AsciiTableHDU cases below.  The Phase 1 file
test_compressed_image_phase1.py::test_repr_contains_compression_info
has a minimal smoke test, but the full _show-instrumented visual-
inspection suite in this file doesn't include compressed images yet.
Needs fitsio (pytest.importorskip) to generate the fixture.
"""

import os
import struct
import sys
import tempfile

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


if __name__ == "__main__":
    # Running the file directly drives the same tests pytest would,
    # but with output capture disabled and verbose mode on, so each
    # test's _show(r) output appears under its name.  Useful for
    # eyeballing changes to the repr formatting.
    sys.exit(pytest.main([__file__, "-s", "-v"]))
