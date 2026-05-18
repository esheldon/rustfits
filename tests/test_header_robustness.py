"""Robustness / correctness tests for FITS header parsing.

Covers:
    - parse_keyword('NAXIS') must not match NAXIS1, NAXIS2, ... even when
      a NAXISn card appears before NAXIS in card order.
    - The END card check must not be satisfied by other keywords whose
      names start with "END" (e.g. ENDIAN).
    - Non-printable bytes (outside 0x20-0x7E) in a header block must
      cause the reader to error rather than silently substitute.
    - Mandatory keywords (SIMPLE, BITPIX, NAXIS, NAXISn) must be present;
      the reader rejects headers missing any of them.
"""

import os
import tempfile

import pytest

import rustfits


def _card(text):
    assert len(text) <= 80, f"card too long ({len(text)} chars): {text!r}"
    return text.ljust(80)


def _write_header_only(fname, cards):
    """
    Write a single primary HDU with the given cards, padded to a 2880-byte
    block.
    """
    block = "".join(cards).encode("ascii")
    block += b" " * ((-len(block)) % 2880)
    with open(fname, "wb") as f:
        f.write(block)


def _write_two_hdus(fname, cards1, cards2):
    blocks = []
    for cards in (cards1, cards2):
        b = "".join(cards).encode("ascii")
        b += b" " * ((-len(b)) % 2880)
        blocks.append(b)
    with open(fname, "wb") as f:
        for b in blocks:
            f.write(b)


# ----------------------- parse_keyword disambiguation -----------------------


def test_naxis_keyword_not_matched_by_naxis1():
    """parse_keyword('NAXIS') must not match a NAXIS1 card listed before NAXIS.

    Constructs a malformed-order header where NAXIS1 appears first.  With a
    lenient `starts_with` match, the data-size calculation would use 99
    instead of 0, mis-locating the second HDU.
    """
    cards1 = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS1  =                   99"),  # stray, before NAXIS
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    cards2 = [
        _card("XTENSION= 'IMAGE   '"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("EXTNAME = 'hdu1'"),
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_two_hdus(fname, cards1, cards2)
        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 2
            assert fits.hdus[1].header["EXTNAME"] == "hdu1"


# --------------------------- END card detection ----------------------------


def test_end_card_does_not_match_endian():
    """A regular keyword like ENDIAN must not be misidentified as END."""
    cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("ENDIAN  = 'little' / endianness"),
        _card("OTHER   =                   42 / some other thing"),
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_header_only(fname, cards)
        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header
        assert hd["ENDIAN"] == "little"
        assert hd["OTHER"] == 42


# ----------------------- printable-ASCII validation ------------------------


def test_non_printable_byte_rejected():
    cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    raw = "".join(cards).encode("ascii")
    raw += b" " * ((-len(raw)) % 2880)
    corrupted = bytearray(raw)
    corrupted[120] = 0xFF  # somewhere inside the BITPIX card
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        with open(fname, "wb") as f:
            f.write(corrupted)
        with pytest.raises((ValueError, OSError)):
            rustfits.FITS(fname, "r")


# ----------------------- mandatory-keyword validation ------------------------


def test_missing_simple_rejected():
    cards = [
        # SIMPLE intentionally missing — primary HDU must start with SIMPLE
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_header_only(fname, cards)
        with pytest.raises((ValueError, OSError)):
            rustfits.FITS(fname, "r")


def test_missing_bitpix_rejected():
    cards = [
        _card("SIMPLE  =                    T"),
        # BITPIX missing
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_header_only(fname, cards)
        with pytest.raises((ValueError, OSError)):
            rustfits.FITS(fname, "r")


def test_missing_naxis_rejected():
    cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        # NAXIS missing
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_header_only(fname, cards)
        with pytest.raises((ValueError, OSError)):
            rustfits.FITS(fname, "r")


def test_missing_naxisn_rejected():
    """NAXIS=2 declared but only NAXIS1 present — must be rejected."""
    cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    2"),
        _card("NAXIS1  =                   10"),
        # NAXIS2 missing
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_header_only(fname, cards)
        with pytest.raises((ValueError, OSError)):
            rustfits.FITS(fname, "r")


if __name__ == "__main__":
    test_naxis_keyword_not_matched_by_naxis1()
    test_end_card_does_not_match_endian()
    test_non_printable_byte_rejected()
    test_missing_simple_rejected()
    test_missing_bitpix_rejected()
    test_missing_naxis_rejected()
    test_missing_naxisn_rejected()
    print("all tests passed")
