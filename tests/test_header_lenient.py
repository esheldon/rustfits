"""
Opt-in lenient header parsing (FITS(..., lenient=True)).

rustfits defaults to strict header validation — files with bytes
outside the FITS-standard printable-ASCII range (0x20-0x7E) are
rejected on open.  This is the right default for most workflows
(integrity guarantee: if the file opened, every byte was
spec-compliant), but it makes archive files written by older
non-conforming tools unreadable.

`lenient=True` on the constructor relaxes the byte-level check:
non-printable bytes are substituted with underscores in place,
then parsing proceeds normally.  Cards with illegal characters
in the keyword (e.g. '@', '.') or unconventional value text
(unquoted NAN / INF, numbers inside quotes) work transparently
because rustfits's value parser already falls back to string on
unparseable input and key lookup is a raw substring match.

Regression pin: the IDL MWRFITS fixture from fitsio's test suite
(fitsio/tests/test_header_junk.py).  Real-world archive files
emit this kind of garbage; opting into lenient mode should make
them readable.

Divergence from fitsio worth knowing about: where fitsio returns
the string 'NAN' / 'INF' for unquoted NaN-like value text,
rustfits returns float('nan') / float('inf') because Rust's
f64::from_str accepts those tokens.  Arguably more useful for
downstream code (NaN-as-float composes with numpy / math.isnan),
and the round-trip-string form is still recoverable as a
deliberate quoted value.
"""

import math
import os
import tempfile

import pytest

import rustfits


# Verbatim from fitsio/tests/test_header_junk.py — the literal
# byte string of a real broken file emitted by IDL MWRFITS.  Two
# non-ASCII bytes (0xF4, 0x04) sit between 'QUI' and the trailing
# spaces of that key.
_IDL_MWRFITS_FIXTURE = b"""SIMPLE  =                    T /Primary Header created by MWRFITS v1.11         BITPIX  =                   16 /                                                NAXIS   =                    0 /                                                EXTEND  =                    T /Extensions may be present                       BLAT    =                    1 /integer                                         FOO     =              1.00000 /float (or double?)                              BAR@    =                  NAN /float NaN                                       BI.Z    =                  NaN /double NaN                                      BAT     =                  INF /1.0 / 0.0                                       BOO     =                 -INF /-1.0 / 0.0                                      QUAT    = '        '           /blank string                                    QUIP    = '1.0     '           /number in quotes                                QUIZ    = ' 1.0    '           /number in quotes with a leading space           QUI\xf4\x04   = 'NaN     '           /NaN in quotes                                   HIERARCH QU.@D = 'Inf     '                                                     END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             """  # noqa: E501


def _write_fixture(tmpdir):
    fname = os.path.join(tmpdir, "junk.fits")
    with open(fname, "wb") as f:
        f.write(_IDL_MWRFITS_FIXTURE)
    return fname


# ---------------------------------------------------------------
# Strict mode (default) rejects with a helpful error
# ---------------------------------------------------------------


def test_strict_default_rejects_non_printable_byte():
    """
    Default behavior: any non-printable byte in a header block
    raises immediately on open with a message that points at
    lenient=True as the opt-in fix.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with pytest.raises(ValueError, match="non-printable byte"):
            rustfits.FITS(fname, "r")


def test_strict_error_message_mentions_lenient():
    """
    The rejection message must tell the user how to recover —
    otherwise they hit a wall.  Specifically: name 'lenient=True'.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with pytest.raises(ValueError, match="lenient=True"):
            rustfits.FITS(fname, "r")


# ---------------------------------------------------------------
# Lenient mode parses every key the fitsio test asserts on
# ---------------------------------------------------------------


def test_lenient_opens_idl_mwrfits_fixture():
    """File opens without error in lenient mode."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            assert len(f) >= 1


def test_lenient_preserves_well_formed_values():
    """The standard-conforming cards parse normally — integer and
    float values are NOT side-effected by lenient mode."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert h["BLAT"] == 1
            assert h["FOO"] == 1.0


def test_lenient_accepts_keys_with_illegal_chars():
    """Keys containing '@' or '.' (illegal per the 8-char keyword
    spec but emitted by IDL MWRFITS) are accepted in lenient mode
    and looked up case-insensitively like any other key."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            # Just confirm the key is present and not raising
            # KeyError; the value semantics are checked separately.
            assert "BAR@" in h
            assert "BI.Z" in h


def test_lenient_parses_unquoted_nan_as_float_nan():
    """
    Unquoted NAN as a value (legal in IEEE-754, illegal per the
    FITS spec) is parsed as float NaN, not a string.  This is a
    deliberate divergence from fitsio (which returns 'NAN' as a
    string) — see the module docstring for the rationale.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert math.isnan(h["BAR@"])
            assert math.isnan(h["BI.Z"])


def test_lenient_parses_unquoted_inf_as_float_inf():
    """Unquoted INF / -INF parse as +/- infinity."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert math.isinf(h["BAT"])
            assert h["BAT"] > 0
            assert math.isinf(h["BOO"])
            assert h["BOO"] < 0


def test_lenient_preserves_quoted_numeric_strings():
    """
    Numbers inside quotes are strings per the FITS spec; lenient
    mode must NOT promote them to numeric.  Includes preservation
    of leading whitespace inside the quoted string.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert h["QUAT"] == ""
            assert h["QUIP"] == "1.0"
            assert h["QUIZ"] == " 1.0"  # leading space inside quotes


def test_lenient_substitutes_non_ascii_with_underscore():
    """
    Non-printable / non-ASCII bytes in keys are substituted with
    '_' (matches astropy's substitution rule).  The QUI key in
    the fixture has two non-ASCII bytes after 'QUI', so the
    lenient-mode key text becomes 'QUI__'.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert h["QUI__"] == "NaN"


def test_lenient_hierarch_with_illegal_chars():
    """
    HIERARCH long keys with '@' and '.' already work in strict
    mode (rustfits's HIERARCH validator is broader than the
    8-char-key validator); lenient mode obviously also accepts
    them.  This test pins the behavior so a future tightening
    of HIERARCH validation doesn't silently break the case.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _write_fixture(tmp)
        with rustfits.FITS(fname, "r", lenient=True) as f:
            h = f[0].header
            assert h["qu.@d"] == "Inf"


# ---------------------------------------------------------------
# Cross-tests: a well-formed file opens identically in either mode
# ---------------------------------------------------------------


def test_lenient_is_a_no_op_on_well_formed_files():
    """
    `lenient=True` is purely additive — a file with no illegal
    bytes opens and reads identically under either mode.  Verified
    against a freshly-created rustfits file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "clean.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu("i4", (3, 3))
            f[0].header["MYINT"] = 42
            f[0].header["MYFLOAT"] = 3.14
            f[0].header["MYSTR"] = "hello"
        with rustfits.FITS(fname, "r") as strict:
            strict_keys = sorted(strict[0].header.keys())
        with rustfits.FITS(fname, "r", lenient=True) as lenient:
            lenient_keys = sorted(lenient[0].header.keys())
        assert strict_keys == lenient_keys
        with rustfits.FITS(fname, "r", lenient=True) as f:
            assert f[0].header["MYINT"] == 42
            assert f[0].header["MYFLOAT"] == 3.14
            assert f[0].header["MYSTR"] == "hello"
