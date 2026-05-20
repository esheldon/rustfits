"""
Tests for the FITS header parser's handling of:

    - HIERARCH long keywords
    - `''` escape inside string values
    - trailing-space stripping in string values
    - `D` (Fortran double precision) exponent in floats
    - complex-number literals
    - blank-keyword commentary cards

Each test reads from a single hand-crafted FITS file that exercises all six
features in one primary HDU with no data.
"""

import os
import tempfile

import pytest

import rustfits


def _card(text):
    """Pad text to exactly 80 chars (one FITS card)."""
    assert len(text) <= 80, f"card too long ({len(text)} chars): {text!r}"
    return text.ljust(80)


def _write_extended_fits(fname):
    cards = [
        _card("SIMPLE  =                    T / conforms to FITS standard"),
        _card(
            "BITPIX  =                    8 / number of bits per data pixel"
        ),
        _card("NAXIS   =                    0 / number of data axes"),
        # HIERARCH long keyword (keyword contains spaces).
        _card("HIERARCH ESO INS DET TEMP = 12.5 / instrument temperature"),
        _card("HIERARCH ESO TEL ALT = -20.25 / telescope altitude"),
        # `''` escape inside a quoted string.
        _card("OBSERVER= 'O''Brien' / observer surname"),
        # Several escapes in one string.
        _card("QUOTED  = 'a''b''c' / multiple escapes"),
        # Trailing-space stripping (per FITS standard).
        _card("OBJECT  = 'M31     ' / target"),
        # `D` exponent (Fortran double precision), both cases.
        _card("EXPTIME = 1.5D-3 / exposure in seconds"),
        _card("BIGNUM  = 2.5d10 / lower-case d exponent"),
        # Complex literals: float and integer components.
        _card("IMPED   = (50.0, -25.0) / complex impedance"),
        _card("CINT    = (3, 4) / complex with integer parts"),
        # Blank-keyword commentary cards (cols 1-8 all spaces).
        _card("        Some commentary text without a keyword"),
        _card("        Another line of blank-keyword commentary"),
        _card("END"),
    ]
    header = "".join(cards).encode("ascii")
    header += b" " * ((-len(header)) % 2880)
    with open(fname, "wb") as f:
        f.write(header)


@pytest.fixture
def header():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_extended_fits(fname)
        with rustfits.FITS(fname, "r") as fits:
            yield fits.hdus[0].header


def test_hierarch_keyword(header):
    assert header["ESO INS DET TEMP"] == 12.5
    assert (
        header.comment_of("ESO INS DET TEMP") == "instrument temperature"
    )
    assert header["ESO TEL ALT"] == -20.25
    # HIERARCH should not appear as a standalone key.
    assert "HIERARCH" not in header


def test_escaped_single_quote_in_string(header):
    assert header["OBSERVER"] == "O'Brien"
    assert header["QUOTED"] == "a'b'c"


def test_trailing_space_trimmed_in_string(header):
    assert header["OBJECT"] == "M31"


def test_d_exponent_floats(header):
    assert header["EXPTIME"] == pytest.approx(1.5e-3)
    assert isinstance(header["EXPTIME"], float)
    assert header["BIGNUM"] == pytest.approx(2.5e10)


def test_complex_values(header):
    v = header["IMPED"]
    assert isinstance(v, complex)
    assert v == complex(50.0, -25.0)

    w = header["CINT"]
    assert isinstance(w, complex)
    assert w == complex(3, 4)


def test_blank_keyword_cards_accumulated(header):
    assert "" in header
    assert isinstance(header[""], list)
    assert header[""] == [
        "Some commentary text without a keyword",
        "Another line of blank-keyword commentary",
    ]


if __name__ == "__main__":
    # Allow running directly without pytest.
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_extended_fits(fname)
        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header
        from pprint import pprint

        pprint(hd)
