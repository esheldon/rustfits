"""
Robustness / correctness tests for FITS header parsing.

Covers:
    - parse_keyword('NAXIS') must not match NAXIS1, NAXIS2, ... even when
      a NAXISn card appears before NAXIS in card order.
    - The END card check must not be satisfied by other keywords whose
      names start with "END" (e.g. ENDIAN).
    - Non-printable bytes (outside 0x20-0x7E) in a header block must
      cause the reader to error rather than silently substitute.
    - Mandatory keywords (SIMPLE, BITPIX, NAXIS, NAXISn) must be present;
      the reader rejects headers missing any of them.
    - Extension headers must start with XTENSION (not a stray SIMPLE).
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
    """
    parse_keyword('NAXIS') must not match a NAXIS1 card listed before NAXIS.

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


def test_extension_starts_with_simple_rejected():
    """
    A multi-HDU file whose second HDU's header starts with SIMPLE
    instead of XTENSION must be rejected with a clear error.  This
    is what a file looks like when an extension is missing its
    XTENSION marker (or, equivalently, when two primary headers
    were concatenated by mistake).  Regression pin against the
    fitsio fixture in fitsio/tests/test_header_junk.py
    ::test_missing_xtension_keyword (same shape on a real-world
    malformed file).
    """
    primary = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("EXTEND  =                    T"),
        _card("END"),
    ]
    # Extension HDU but starts with SIMPLE — illegal.
    extension = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                   32"),
        _card("NAXIS   =                    2"),
        _card("NAXIS1  =                   30"),
        _card("NAXIS2  =                   30"),
        _card("END"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rb.fits")
        _write_two_hdus(fname, primary, extension)
        with pytest.raises(ValueError, match="XTENSION"):
            rustfits.FITS(fname, "r")


# --------------------------- CONTINUE quirks -------------------------------


def _write_primary_with_cards(fname, extra_cards):
    """
    Single-HDU file with NAXIS=0 (no data) plus the given user cards.
    """
    cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
    ]
    cards.extend(extra_cards)
    cards.append(_card("END"))
    _write_header_only(fname, cards)


def test_orphan_continue_card_does_not_crash():
    """
    A CONTINUE card with no preceding string ending in '&' is
    malformed per the FITS spec, but archive files contain them
    (and fitsio + astropy both tolerate them).  rustfits must
    parse the file without crashing and the neighboring cards
    must still read correctly; the orphan CONTINUE is silently
    dropped from the keys() view (CONTINUE is in the commentary-
    key exclusion list).  Regression pin against
    fitsio/tests/test_header.py::test_corrupt_continue (the
    first half of the smoke test).
    """
    orphan = [
        _card("IVAL    =                   35 / integer value"),
        _card("SHORTS  = 'hello world'"),
        _card("CONTINUE= '        '           / orphan CONTINUE here"),
        _card("UND     ="),
        _card("DBL     =                 1.25"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "orphan.fits")
        _write_primary_with_cards(fname, orphan)
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            # Neighboring cards still parse, the orphan CONTINUE
            # doesn't propagate into them.
            assert h["IVAL"] == 35
            assert h["SHORTS"] == "hello world"
            assert h["UND"] is None
            assert h["DBL"] == 1.25


def test_multi_card_continue_chain_does_not_crash():
    """
    A 3-card CONTINUE chain (PROGRAM + 2 CONTINUE cards, where the
    last CONTINUE has '&' as payload) must parse without crashing.
    The exact final value text is implementation-defined for this
    edge case (the spec is ambiguous about what '&' as payload on
    the last card means), so this test pins only the
    no-crash + chain-detection invariant + the neighboring keys.
    Regression pin against fitsio/tests/test_header.py
    ::test_corrupt_continue (the second half of the smoke test).
    """
    chain = [
        _card("IVAL    =                   35 / integer value"),
        _card("SHORTS  = 'hello world'"),
        _card(
            "PROGRAM = 'Setting the Scale: "
            "Determining the Absolute Mass Norm. and &'"
        ),
        _card("CONTINUE  'Scaling Relations for Clusters at z~0.1&'"),
        _card("CONTINUE  '&' / Current observing program"),
        _card("UND     ="),
        _card("DBL     =                 1.25"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "chain.fits")
        _write_primary_with_cards(fname, chain)
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            # The chain follower actually concatenates the segments
            # (don't pin the trailing-'&' edge case — start-match
            # only, so the test stays robust to future tightening
            # of the spec-ambiguous tail handling).
            assert isinstance(h["PROGRAM"], str)
            assert h["PROGRAM"].startswith("Setting the Scale:")
            assert "Scaling Relations" in h["PROGRAM"]
            # Neighboring cards unaffected by the chain logic.
            assert h["IVAL"] == 35
            assert h["SHORTS"] == "hello world"
            assert h["UND"] is None
            assert h["DBL"] == 1.25


if __name__ == "__main__":
    test_naxis_keyword_not_matched_by_naxis1()
    test_end_card_does_not_match_endian()
    test_non_printable_byte_rejected()
    test_missing_simple_rejected()
    test_missing_bitpix_rejected()
    test_missing_naxis_rejected()
    test_missing_naxisn_rejected()
    test_extension_starts_with_simple_rejected()
    test_orphan_continue_card_does_not_crash()
    test_multi_card_continue_chain_does_not_crash()
    print("all tests passed")
