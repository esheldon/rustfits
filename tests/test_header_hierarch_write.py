"""Phase 2c, step 3: HIERARCH-on-write for long keywords.

Long keywords (>8 chars or containing spaces) are written as HIERARCH cards:

    HIERARCH <long-key> = <value> [/ comment]

Each test verifies its assertion through BOTH the same FITS handle that did
the mutation AND a fresh reopen.

Out of scope (phase 2c step 3):
    - Case preservation on HIERARCH keys (we always normalize to uppercase,
      same as standard keys).
    - CONTINUE-chained HIERARCH long-string values (deferred).
"""

import os
import tempfile
import contextlib

import pytest

import rustfits


@contextlib.contextmanager
def _new_file(shape=(4, 6), dtype="i4"):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


def _check_both(fname, fits, predicate):
    predicate(fits[0].header)
    with rustfits.FITS(fname, "r") as fits2:
        predicate(fits2[0].header)


# ============================================================================
# Detection & basic write
# ============================================================================


def test_long_single_word_key_writes_hierarch():
    """A >8-char key with no spaces becomes a HIERARCH card."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONGKEYWORD"] = 42

            def check(hd):
                assert hd["LONGKEYWORD"] == 42
                cards = hd.cards
                assert any(c.startswith("HIERARCH LONGKEYWORD") for c in cards)
                # Not a standard "LONGKEYW" card by truncation.
                assert not any(c.startswith("LONGKEYW ") for c in cards)

            _check_both(fname, fits, check)


def test_multi_word_key_writes_hierarch():
    """A key containing spaces (any length) is HIERARCH."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 12.5

            def check(hd):
                assert hd["ESO INS TEMP"] == 12.5
                cards = hd.cards
                assert any(c.startswith("HIERARCH ESO INS TEMP") for c in cards)

            _check_both(fname, fits, check)


def test_short_key_with_space_uses_hierarch():
    """Even an 8-char key with a space is HIERARCH-shaped."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ABC DEFG"] = 5

            def check(hd):
                assert hd["ABC DEFG"] == 5
                assert any(c.startswith("HIERARCH ABC DEFG") for c in hd.cards)

            _check_both(fname, fits, check)


# ============================================================================
# Value types
# ============================================================================


def test_hierarch_with_int_value():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 42

            def check(hd):
                v = hd["ESO INS TEMP"]
                assert v == 42 and isinstance(v, int) and not isinstance(v, bool)

            _check_both(fname, fits, check)


def test_hierarch_with_float_value():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 1.5e-3

            def check(hd):
                v = hd["ESO INS TEMP"]
                assert isinstance(v, float)
                assert v == pytest.approx(1.5e-3)

            _check_both(fname, fits, check)


def test_hierarch_with_bool_value():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO IS COOL"] = True

            def check(hd):
                assert hd["ESO IS COOL"] is True

            _check_both(fname, fits, check)


def test_hierarch_with_string_value():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS NAME"] = "FORS2"

            def check(hd):
                assert hd["ESO INS NAME"] == "FORS2"

            _check_both(fname, fits, check)


def test_hierarch_with_value_and_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = (12.5, "[C] sensor 1")

            def check(hd):
                assert hd["ESO INS TEMP"] == 12.5
                assert hd.comment_of("ESO INS TEMP") == "[C] sensor 1"

            _check_both(fname, fits, check)


def test_hierarch_string_with_embedded_quote():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["OBS TARGET"] = "M31's center"

            def check(hd):
                assert hd["OBS TARGET"] == "M31's center"

            _check_both(fname, fits, check)


# ============================================================================
# Case normalization
# ============================================================================


def test_lowercase_hierarch_key_uppercased():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["eso ins temp"] = 12.5
            # Both case variants resolve to the same uppercase key.
            assert fits[0].header["ESO INS TEMP"] == 12.5
            assert fits[0].header["eso ins temp"] == 12.5
            assert fits[0].header["Eso Ins Temp"] == 12.5
            # On disk, the card is uppercase.
            assert any(c.startswith("HIERARCH ESO INS TEMP") for c in fits[0].header.cards)
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["ESO INS TEMP"] == 12.5


def test_hierarch_key_with_surrounding_spaces_normalized():
    """Leading/trailing whitespace on the user-supplied key is trimmed."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["  ESO INS TEMP  "] = 12.5
            # The on-disk key is the trimmed form.
            assert fits[0].header["ESO INS TEMP"] == 12.5
            cards = fits[0].header.cards
            assert any(c.startswith("HIERARCH ESO INS TEMP =") for c in cards)


# ============================================================================
# Update / delete
# ============================================================================


def test_hierarch_update_preserves_position():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 12.5
            keys_before = list(fits[0].header)
            fits[0].header["ESO INS TEMP"] = 13.5  # update
            assert fits[0].header["ESO INS TEMP"] == 13.5
            assert list(fits[0].header) == keys_before
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["ESO INS TEMP"] == 13.5


def test_hierarch_bare_update_preserves_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = (12.5, "[C]")
            fits[0].header["ESO INS TEMP"] = 15.0  # bare value, no comment

            def check(hd):
                assert hd["ESO INS TEMP"] == 15.0
                assert hd.comment_of("ESO INS TEMP") == "[C]"

            _check_both(fname, fits, check)


def test_hierarch_delete():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 12.5
            assert "ESO INS TEMP" in fits[0].header
            del fits[0].header["ESO INS TEMP"]

            def check(hd):
                assert "ESO INS TEMP" not in hd
                # No leftover HIERARCH card.
                assert not any(c.startswith("HIERARCH ESO INS TEMP") for c in hd.cards)

            _check_both(fname, fits, check)


def test_hierarch_delete_lowercase_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS TEMP"] = 12.5
            del fits[0].header["eso ins temp"]
            assert "ESO INS TEMP" not in fits[0].header


# ============================================================================
# Coexistence with standard keys & each other
# ============================================================================


def test_hierarch_and_standard_keys_coexist():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5.0
            fits[0].header["ESO INS TEMP"] = 12.5
            fits[0].header["OBJECT"] = "M31"
            fits[0].header["ESO OBS ID"] = 1234

            def check(hd):
                assert hd["EXPTIME"] == 5.0
                assert hd["ESO INS TEMP"] == 12.5
                assert hd["OBJECT"] == "M31"
                assert hd["ESO OBS ID"] == 1234

            _check_both(fname, fits, check)


def test_update_with_dict_mixed_keys():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.update({
                "EXPTIME": 5.0,
                "ESO INS TEMP": (12.5, "[C]"),
                "OBJECT": "M31",
            })

            def check(hd):
                assert hd["EXPTIME"] == 5.0
                assert hd["ESO INS TEMP"] == 12.5
                assert hd.comment_of("ESO INS TEMP") == "[C]"
                assert hd["OBJECT"] == "M31"

            _check_both(fname, fits, check)


# ============================================================================
# Inside FITSHeaderEdit batch
# ============================================================================


def test_hierarch_in_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["ESO INS TEMP"] = 12.5
                h["ESO OBS ID"] = 7
                assert h["ESO INS TEMP"] == 12.5
                assert h["ESO OBS ID"] == 7

            def check(hd):
                assert hd["ESO INS TEMP"] == 12.5
                assert hd["ESO OBS ID"] == 7

            _check_both(fname, fits, check)


def test_hierarch_edit_rollback():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h["ESO INS TEMP"] = 12.5
                    raise RuntimeError("boom")

            def check(hd):
                assert "ESO INS TEMP" not in hd

            _check_both(fname, fits, check)


# ============================================================================
# Validation
# ============================================================================


def test_literal_hierarch_keyword_rejected():
    """The bare literal 'HIERARCH' is the convention prefix, not a key."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["HIERARCH"] = 1


def test_hierarch_card_too_long_rejected():
    """If key + value + comment + framing exceeds 80 chars, the write is
    rejected with a clear error (no silent truncation)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                # 60-char key + long string value definitely won't fit in 80 chars.
                fits[0].header["A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"] = "X" * 50


def test_hierarch_invalid_chars_rejected():
    """HIERARCH allows extra chars (space, '.', '+') but still rejects others."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["ESO INS#TEMP"] = 5    # '#' not allowed
            with pytest.raises(ValueError):
                fits[0].header["LONG KEY/PATH"] = 5    # '/' not allowed


def test_hierarch_dot_and_plus_allowed():
    """HIERARCH should accept '.' and '+' (some conventions use them)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS.TEMP"] = 12.5
            fits[0].header["ESO+SENSOR"] = 1
            assert fits[0].header["ESO INS.TEMP"] == 12.5
            assert fits[0].header["ESO+SENSOR"] == 1
