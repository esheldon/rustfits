"""
ASCII-table (XTENSION='TABLE') read tests — Phase 1 MVP.

Covers whole-table read, accessors (nrows, ncols, colnames, dtype,
units, __len__), per-TFORM-letter parsing (A/I/F/E/D), TSCAL/TZERO
scaling, gaps between columns, FORTRAN 'D' exponent marker, and
astropy round-trip.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

# ---------------------------------------------------------------------------
# byte-level fixture helpers (same pattern as test_hdu_accessors.py)
# ---------------------------------------------------------------------------

CARDS_PER_BLOCK = 36
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b, pad_byte=b" "):
    # ASCII tables pad with SPACE, not NUL (per FITS standard).
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
    """
    cols: list of dicts with keys name, tform, tbcol, optional unit /
    tscal / tzero / tnull.
    """
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
            # Use repr to round-trip f64 exactly through the card value.
            cards.append(f"TSCAL{i:<3d}= {c['tscal']!r:>20s}")
        if c.get("tzero") is not None:
            cards.append(f"TZERO{i:<3d}= {c['tzero']!r:>20s}")
        if c.get("tnull") is not None:
            cards.append(f"TNULL{i:<3d}= '{c['tnull']:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


def _make_file(tmp, rows_text, cols, extras=()):
    """rows_text: list[str] of row text, each already padded to NAXIS1."""
    naxis1 = len(rows_text[0]) if rows_text else 0
    for r in rows_text:
        assert len(r) == naxis1, f"row width mismatch: {len(r)} vs {naxis1}"
    naxis2 = len(rows_text)
    data = "".join(rows_text).encode("ascii")
    fname = os.path.join(tmp, "t.fits")
    _write_file(
        fname,
        (_primary_no_data(), b"", b" "),
        (_ascii_ext(naxis1, naxis2, cols, extras), data, b" "),
    )
    return fname


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def test_accessors_basic():
    """All five accessors return the expected shape on a 3-col table."""
    cols = [
        {"name": "NAME", "tform": "A8", "tbcol": 1},
        {"name": "X", "tform": "I5", "tbcol": 10},
        {"name": "Y", "tform": "F8.2", "tbcol": 16, "unit": "m"},
    ]
    # NAXIS1 = 23: cols 1-8 (A8), gap 9, cols 10-14 (I5),
    # gap 15, cols 16-23 (F8.2)
    rows = [
        "alice    1234     3.14 ",
        "bob          1    -.50 ",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            hdu = fits[1]
            assert isinstance(hdu, rustfits.AsciiTableHDU)
            assert hdu.nrows == 2
            assert hdu.ncols == 3
            assert len(hdu) == 2
            assert hdu.colnames == ("NAME", "X", "Y")
            assert hdu.units == {"NAME": None, "X": None, "Y": "m"}
            dt = hdu.dtype
            assert dt.names == ("NAME", "X", "Y")
            assert dt["NAME"] == np.dtype("U8")
            assert dt["X"] == np.dtype("i8")
            assert dt["Y"] == np.dtype("f4")


def test_repr_includes_columns():
    cols = [
        {"name": "X", "tform": "I3", "tbcol": 1},
        {"name": "Y", "tform": "F6.2", "tbcol": 5, "unit": "Jy"},
    ]
    # NAXIS1 = 10: cols 1-3 (I3), gap 4, cols 5-10 (F6.2)
    rows = ["  1   1.50", "  2  -0.25"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            assert "ASCII_TBL" in r
            assert "rows: 2" in r
            assert "X" in r
            assert "Y" in r
            assert "(Jy)" in r


# ---------------------------------------------------------------------------
# per-TFORM-letter parsing
# ---------------------------------------------------------------------------


def test_read_string_column():
    cols = [{"name": "NAME", "tform": "A5", "tbcol": 1}]
    rows = ["alice", "bob  ", "  hi ", "     "]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["NAME"] == np.dtype("U5")
            assert list(arr["NAME"]) == ["alice", "bob", "  hi", ""]


def test_read_integer_column():
    cols = [{"name": "X", "tform": "I6", "tbcol": 1}]
    rows = ["     0", "    42", "   -17", "999999"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("i8")
            np.testing.assert_array_equal(arr["X"], [0, 42, -17, 999999])


def test_read_float_F_column():
    cols = [{"name": "X", "tform": "F8.3", "tbcol": 1}]
    rows = ["   0.000", "   1.500", "  -2.250", "1234.567"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("f4")
            np.testing.assert_allclose(
                arr["X"], [0.0, 1.5, -2.25, 1234.567], rtol=1e-5
            )


def test_read_float_E_column():
    cols = [{"name": "X", "tform": "E12.4", "tbcol": 1}]
    rows = [
        "  1.5000E+03",
        " -2.5000E-02",
        "  0.0000E+00",
    ]
    naxis1 = 12
    for r in rows:
        assert len(r) == naxis1
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("f4")
            np.testing.assert_allclose(
                arr["X"], [1500.0, -0.025, 0.0], rtol=1e-5
            )


def test_read_double_D_column():
    cols = [{"name": "X", "tform": "D20.12", "tbcol": 1}]
    rows = [
        "  1.500000000000D+03",
        " -2.500000000000E-02",
        "  0.000000000000D+00",
    ]
    naxis1 = 20
    for r in rows:
        assert len(r) == naxis1
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("f8")
            np.testing.assert_allclose(
                arr["X"], [1500.0, -0.025, 0.0], rtol=1e-12
            )


# ---------------------------------------------------------------------------
# layout (gaps between columns)
# ---------------------------------------------------------------------------


def test_read_columns_with_gaps():
    """Two columns with a 2-byte gap; the gap bytes are ignored."""
    cols = [
        {"name": "A", "tform": "I3", "tbcol": 1},
        # gap at byte positions 4-5 (1-based)
        {"name": "B", "tform": "I3", "tbcol": 6},
    ]
    rows = [
        "  1XX  2",  # naxis1=8; gap holds 'XX' as filler
        "  3XX  4",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["A"], [1, 3])
            np.testing.assert_array_equal(arr["B"], [2, 4])


def test_overlap_rejected():
    """Overlapping columns are rejected at parse time."""
    cols = [
        {"name": "A", "tform": "I5", "tbcol": 1},
        {"name": "B", "tform": "I5", "tbcol": 3},  # overlaps A
    ]
    rows = ["1234567"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError, match="overlap"):
                fits[1].read()


# ---------------------------------------------------------------------------
# scaling (TSCAL / TZERO)
# ---------------------------------------------------------------------------


def test_general_scaling_promotes_to_f8():
    """TSCAL=0.5 + TZERO=10 on I -> f8 with physical = 0.5*stored + 10."""
    cols = [
        {"name": "X", "tform": "I5", "tbcol": 1, "tscal": 0.5, "tzero": 10.0},
    ]
    rows = ["    0", "   10", "  -20"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("f8")
            np.testing.assert_allclose(arr["X"], [10.0, 15.0, 0.0])
            # scale=False returns raw i8
            arr_raw = fits[1].read(scale=False)
            assert arr_raw.dtype["X"] == np.dtype("i8")
            np.testing.assert_array_equal(arr_raw["X"], [0, 10, -20])


def test_unsigned_trick_on_I():
    """I + TSCAL=1 + TZERO=2^63 -> u8 with bias subtracted."""
    cols = [
        {
            "name": "X",
            "tform": "I20",
            "tbcol": 1,
            "tscal": 1.0,
            "tzero": 9.223372036854776e18,  # 2^63
        },
    ]
    # stored values 0, 1, -1 in i8 -> u8 of 2^63, 2^63+1, 2^63-1
    rows = [
        "                   0",
        "                   1",
        "                  -1",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("u8")
            expected = np.array(
                [1 << 63, (1 << 63) + 1, (1 << 63) - 1], dtype=np.uint64
            )
            np.testing.assert_array_equal(arr["X"], expected)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_table():
    """Zero-row table reads as a length-0 structured array."""
    cols = [{"name": "X", "tform": "I3", "tbcol": 1}]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b"", b" "),
            (_ascii_ext(3, 0, cols), b"", b" "),
        )
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert len(arr) == 0
            assert arr.dtype["X"] == np.dtype("i8")


def test_single_column():
    cols = [{"name": "X", "tform": "I3", "tbcol": 1}]
    rows = ["  7", " 42", "  0"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["X"], [7, 42, 0])


def test_blank_integer_field_reads_as_zero():
    """All-blank I field is 'undefined' per FITS spec; read as 0."""
    cols = [{"name": "X", "tform": "I5", "tbcol": 1}]
    rows = ["     ", "   42", "     "]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            np.testing.assert_array_equal(arr["X"], [0, 42, 0])


def test_parse_failure_raises_with_row_context():
    """Garbage in an I field raises ValueError naming the column + row."""
    cols = [{"name": "X", "tform": "I5", "tbcol": 1}]
    rows = ["    1", " junk", "    3"]
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_file(tmp, rows, cols)
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError, match="row 1.*junk"):
                fits[1].read()


# ---------------------------------------------------------------------------
# cross-tool: astropy round-trip
# ---------------------------------------------------------------------------


def test_fitsio_round_trip():
    """A fitsio-written ASCII table reads identically.

    fitsio's ASCII-table writer goes through cfitsio's
    fits_create_ascii_tbl, which picks a TFORM letter per column
    dtype.  Notably, cfitsio emits both ``f4`` and ``f8`` numpy
    columns as ``E26.17`` (NOT ``E15.7`` for f4 + ``D25.17`` for
    f8 as one might expect).  This forces rustfits to use the
    decimal-place count, not just the letter, to pick the read
    dtype: ``Ew.d`` with ``d > 7`` reads as ``f8``.  See the
    ``F_E_F4_MAX_DECIMALS`` rule in ``src/hdu_ascii_table/read.rs``.

    Result here: BOTH the f4 source and f8 source come back as
    f8 from a fitsio-written file (no precision loss, just
    slightly bigger arrays).  Astropy's narrower ``E10.4`` reads
    as f4 — see ``test_astropy_round_trip``.
    """
    fitsio = pytest.importorskip("fitsio")

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "fitsio.fits")
        data = np.zeros(
            3,
            dtype=[
                ("ID", "i4"),
                ("FLUX", "f4"),
                ("MJD", "f8"),
                ("NAME", "S8"),
            ],
        )
        data["ID"] = [1, 2, 3]
        data["FLUX"] = [1.5, -2.25, 0.0]
        data["MJD"] = [58000.123456789, 58001.987654321, 58002.5]
        data["NAME"] = ["alpha", "beta", "g"]
        with fitsio.FITS(fname, "rw") as f:
            f.write_table(data, table_type="ascii")

        with rustfits.FITS(fname) as fits:
            assert isinstance(fits[1], rustfits.AsciiTableHDU)
            arr = fits[1].read()
            # ID always reads as i8 per the rustfits Iw -> i8 rule
            # (the source was i4 but the on-disk TFORM is Iw text).
            assert arr.dtype["ID"] == np.dtype("i8")
            # fitsio writes E26.17 for both f4 and f8; rustfits picks
            # f8 because d=17 > 7 (see test docstring).
            assert arr.dtype["FLUX"] == np.dtype("f8")
            assert arr.dtype["MJD"] == np.dtype("f8")
            np.testing.assert_array_equal(arr["ID"], [1, 2, 3])
            np.testing.assert_allclose(
                arr["FLUX"], [1.5, -2.25, 0.0], rtol=1e-12
            )
            np.testing.assert_allclose(
                arr["MJD"],
                [58000.123456789, 58001.987654321, 58002.5],
                rtol=1e-12,
            )
            assert list(arr["NAME"]) == ["alpha", "beta", "g"]


def test_astropy_all_float_formats():
    """Every Fw.d / Ew.d / Dw.d shape astropy can emit.

    Verifies the rustfits f4/f8 dispatch:
    - F/E with d <= 7 -> f4
    - F/E with d > 7  -> f8
    - D (any d)       -> f8

    astropy itself returns f8 for ALL float columns regardless of
    TFORM letter — rustfits is between cfitsio (always-narrow) and
    astropy (always-wide).  See class docstring on AsciiTableHDU.
    """
    astropy_fits = pytest.importorskip("astropy.io.fits")

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "astropy_floats.fits")
        cols = [
            astropy_fits.Column(
                name="F4_NARROW",
                format="F10.3",
                array=np.array([1.5, -2.25, 0.0]),
            ),
            astropy_fits.Column(
                name="F8_WIDE",
                format="F25.15",
                array=np.array([1.5, -2.25, 0.0]),
            ),
            astropy_fits.Column(
                name="E4_NARROW",
                format="E12.4",
                array=np.array([1.5e3, -2.25e-5, 0.0]),
            ),
            astropy_fits.Column(
                name="E8_WIDE",
                format="E26.17",
                array=np.array([1.5e3, -2.25e-5, 0.0]),
            ),
            astropy_fits.Column(
                name="D8",
                format="D25.17",
                array=np.array([1.5e10, -2.25e-15, 0.0]),
            ),
        ]
        hdu = astropy_fits.TableHDU.from_columns(cols)
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fname, overwrite=True
        )

        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["F4_NARROW"] == np.dtype("f4")
            assert arr.dtype["F8_WIDE"] == np.dtype("f8")
            assert arr.dtype["E4_NARROW"] == np.dtype("f4")
            assert arr.dtype["E8_WIDE"] == np.dtype("f8")
            assert arr.dtype["D8"] == np.dtype("f8")
            np.testing.assert_allclose(
                arr["F4_NARROW"], [1.5, -2.25, 0.0], rtol=1e-5
            )
            np.testing.assert_allclose(
                arr["F8_WIDE"], [1.5, -2.25, 0.0], rtol=1e-12
            )
            np.testing.assert_allclose(
                arr["E4_NARROW"], [1.5e3, -2.25e-5, 0.0], rtol=1e-5
            )
            np.testing.assert_allclose(
                arr["E8_WIDE"], [1.5e3, -2.25e-5, 0.0], rtol=1e-12
            )
            np.testing.assert_allclose(
                arr["D8"], [1.5e10, -2.25e-15, 0.0], rtol=1e-12
            )


@pytest.mark.parametrize("width", [3, 5, 10, 15, 20])
def test_astropy_int_widths_all_i8(width):
    """astropy's I3 / I5 / I10 / I15 / I20 all read as i8 in rustfits.

    rustfits always maps Iw -> i8 regardless of width; astropy's
    own typed-read picks i2/i4/i8 by width.  Confirms the choice
    doesn't lose information across the int width range.
    """
    astropy_fits = pytest.importorskip("astropy.io.fits")

    values = np.array([0, 42, -17], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "astropy_ints.fits")
        col = astropy_fits.Column(
            name="X",
            format=f"I{width}",
            array=values,
        )
        hdu = astropy_fits.TableHDU.from_columns([col])
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fname, overwrite=True
        )

        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert arr.dtype["X"] == np.dtype("i8")
            np.testing.assert_array_equal(arr["X"], values)


def test_astropy_multi_chunk():
    """Many rows force the streaming reader through multiple chunks.

    The 1 MiB row-buffer budget means a 20-byte-wide table holds
    ~52k rows per chunk; a 100k-row table spans at least 2 chunks
    and exercises the chunk-refill path that a small fixture
    wouldn't touch.
    """
    astropy_fits = pytest.importorskip("astropy.io.fits")

    n = 100_000
    ids = np.arange(n, dtype=np.int64)
    flux = np.sin(np.arange(n) * 0.001).astype(np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "many.fits")
        cols = [
            astropy_fits.Column(name="ID", format="I10", array=ids),
            astropy_fits.Column(name="FLUX", format="D20.12", array=flux),
        ]
        hdu = astropy_fits.TableHDU.from_columns(cols)
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fname, overwrite=True
        )

        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert len(arr) == n
            np.testing.assert_array_equal(arr["ID"], ids)
            np.testing.assert_allclose(arr["FLUX"], flux, rtol=1e-12)


def test_astropy_strings_and_negative_values():
    """A-column trimming and signed-int / signed-float round-trip."""
    astropy_fits = pytest.importorskip("astropy.io.fits")

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "astropy_mixed.fits")
        cols = [
            astropy_fits.Column(
                name="LABEL",
                format="A10",
                array=np.array(["short", "longer one", "x", ""]),
            ),
            astropy_fits.Column(
                name="SIGNED",
                format="I8",
                array=np.array([0, -1, 99999, -99999], dtype=np.int32),
            ),
            astropy_fits.Column(
                name="DELTA",
                format="E12.4",
                array=np.array([0.0, -1.5e-10, 3.14e15, -2.71]),
            ),
        ]
        hdu = astropy_fits.TableHDU.from_columns(cols)
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fname, overwrite=True
        )

        with rustfits.FITS(fname) as fits:
            arr = fits[1].read()
            assert list(arr["LABEL"]) == ["short", "longer one", "x", ""]
            np.testing.assert_array_equal(
                arr["SIGNED"], [0, -1, 99999, -99999]
            )
            np.testing.assert_allclose(
                arr["DELTA"],
                [0.0, -1.5e-10, 3.14e15, -2.71],
                rtol=1e-5,
            )


def test_astropy_round_trip():
    """An astropy-written ASCII table reads identically."""
    astropy_fits = pytest.importorskip("astropy.io.fits")

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "astropy.fits")
        # astropy TableHDU = ASCII table
        col1 = astropy_fits.Column(
            name="ID",
            format="I5",
            array=np.array([1, 2, 3], dtype=np.int32),
        )
        col2 = astropy_fits.Column(
            name="FLUX",
            format="F10.4",
            array=np.array([1.5, -2.25, 0.0]),
        )
        col3 = astropy_fits.Column(
            name="NAME",
            format="A8",
            array=np.array(["alpha", "beta", "g"]),
        )
        hdu = astropy_fits.TableHDU.from_columns([col1, col2, col3])
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fname, overwrite=True
        )

        with rustfits.FITS(fname) as fits:
            assert isinstance(fits[1], rustfits.AsciiTableHDU)
            arr = fits[1].read()
            assert arr.dtype["ID"] == np.dtype("i8")
            assert arr.dtype["FLUX"] == np.dtype("f4")
            np.testing.assert_array_equal(arr["ID"], [1, 2, 3])
            np.testing.assert_allclose(
                arr["FLUX"], [1.5, -2.25, 0.0], rtol=1e-5
            )
            assert list(arr["NAME"]) == ["alpha", "beta", "g"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
