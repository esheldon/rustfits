"""Phase 2c, step 3: HIERARCH-on-write for long keywords.

Long keywords (>8 chars or containing spaces) are written as HIERARCH cards:

    HIERARCH <long-key> = <value> [/ comment]

Long string values auto-chain via the FITS Long Strings convention: the
first card carries the HIERARCH prefix and a `&'`-terminated chunk,
followed by N standard CONTINUE cards with the comment on the last one.

Each test verifies its assertion through BOTH the same FITS handle that did
the mutation AND a fresh reopen.

Out of scope (phase 2c step 3):
    - Case preservation on HIERARCH keys (we always normalize to uppercase,
      same as standard keys).
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
                assert any(
                    c.startswith("HIERARCH ESO INS TEMP") for c in cards
                )

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
                assert (
                    v == 42 and isinstance(v, int) and not isinstance(v, bool)
                )

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
            assert any(
                c.startswith("HIERARCH ESO INS TEMP")
                for c in fits[0].header.cards
            )
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
                assert not any(
                    c.startswith("HIERARCH ESO INS TEMP") for c in hd.cards
                )

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


def test_hierarch_non_string_card_too_long_rejected():
    """Non-string HIERARCH values can't chain, so a key + framing + value +
    comment that doesn't fit in 80 chars is rejected."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                # 55-char key + int (2) + 30-char comment + framing = 102.
                fits[0].header["A" * 55] = (42, "X" * 30)


def test_hierarch_key_too_long_for_chain_rejected():
    """A HIERARCH key >= 65 chars leaves no room for first-card payload."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["A" * 65] = "X" * 100


def test_hierarch_invalid_chars_rejected():
    """
    HIERARCH allows extra chars (space, '.', '+') but still rejects others.
    """
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


# ============================================================================
# CONTINUE-chained HIERARCH string values
# ============================================================================


def test_hierarch_short_string_emits_single_card():
    """A HIERARCH string that fits in one card stays a single card (no
    CONTINUE)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS NAME"] = "FORS2"

            def check(hd):
                assert hd["ESO INS NAME"] == "FORS2"
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


def test_hierarch_long_string_chains():
    """A HIERARCH string that doesn't fit on one card auto-chains."""
    with _new_file() as fname:
        s = "X" * 100
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESCRIPTION"] = s

            def check(hd):
                assert hd["ESO INS DESCRIPTION"] == s
                cards = hd.cards
                # First card is the HIERARCH card.
                first_idx = next(
                    i for i, c in enumerate(cards)
                    if c.startswith("HIERARCH ESO INS DESCRIPTION")
                )
                # At least one CONTINUE card follows.
                assert cards[first_idx + 1].startswith("CONTINUE")

            _check_both(fname, fits, check)


def test_hierarch_long_string_round_trip_with_comment():
    """Comment lands on the final CONTINUE card and round-trips intact."""
    with _new_file() as fname:
        s = "Y" * 200
        c = "long hierarch value"
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESCRIPTION"] = (s, c)

            def check(hd):
                assert hd["ESO INS DESCRIPTION"] == s
                assert hd.comment_of("ESO INS DESCRIPTION") == c

            _check_both(fname, fits, check)


def test_hierarch_long_string_with_embedded_quotes_round_trips():
    """The chunker must not split a `''` escape pair across cards."""
    with _new_file() as fname:
        # Many quotes, ending in a non-space so trailing spaces aren't
        # trimmed by the FITS string rule.
        s = ("It's M31's brightest core; " * 8)[:199] + "Z"
        assert "'" in s
        assert not s.endswith(" ")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO OBS COMMENT"] = s

            def check(hd):
                assert hd["ESO OBS COMMENT"] == s

            _check_both(fname, fits, check)


def test_hierarch_chain_layout_starts_hierarch_then_continue():
    """On-disk layout: first card has the HIERARCH prefix; the rest are
    standard CONTINUE cards."""
    with _new_file() as fname:
        s = "Z" * 200
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESCRIPTION"] = s

            def check(hd):
                cards = hd.cards
                first_idx = next(
                    i for i, c in enumerate(cards)
                    if c.startswith("HIERARCH ESO INS DESCRIPTION")
                )
                chain = [cards[first_idx]]
                i = first_idx + 1
                while i < len(cards) and cards[i].startswith("CONTINUE"):
                    chain.append(cards[i])
                    i += 1
                assert len(chain) >= 3
                # Non-last cards in the chain end with `&'`.
                for card in chain[:-1]:
                    assert "&'" in card
                assert not chain[-1].rstrip().endswith("&'")

            _check_both(fname, fits, check)


def test_replacing_hierarch_chain_with_short_string_removes_extras():
    """Overwriting a chained HIERARCH value with a short one removes the
    whole chain, not just the first card."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESC"] = "Q" * 200
            chain_before = sum(
                1 for c in fits[0].header.cards if c.startswith("CONTINUE")
            )
            assert chain_before >= 2

            fits[0].header["ESO INS DESC"] = "short"

            def check(hd):
                assert hd["ESO INS DESC"] == "short"
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


def test_replacing_hierarch_chain_with_longer_chain():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESC"] = "A" * 200
            fits[0].header["ESO INS DESC"] = "B" * 400

            def check(hd):
                assert hd["ESO INS DESC"] == "B" * 400

            _check_both(fname, fits, check)


def test_del_hierarch_chain_removes_all_cards():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DESC"] = "X" * 200
            assert any(
                c.startswith("CONTINUE") for c in fits[0].header.cards
            )

            del fits[0].header["ESO INS DESC"]

            def check(hd):
                assert "ESO INS DESC" not in hd
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


def test_hierarch_chain_in_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["ESO INS DESC"] = "P" * 200
                assert h["ESO INS DESC"] == "P" * 200

            def check(hd):
                assert hd["ESO INS DESC"] == "P" * 200

            _check_both(fname, fits, check)


def test_hierarch_chain_edit_rollback():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h["ESO INS DESC"] = "Q" * 200
                    raise RuntimeError("boom")

            def check(hd):
                assert "ESO INS DESC" not in hd
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


def test_hierarch_comment_too_long_for_chain_rejected():
    """Comments are capped at 64 chars for CONTINUE-chained values."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["ESO INS DESC"] = ("Z" * 200, "C" * 80)


def test_hierarch_chain_can_overflow_slack():
    """A chain too big for the reserved header blocks is rejected, and
    in-memory + on-disk state remain consistent (write-disk-first)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            initial = len(h.cards)
            block_count = (initial + 35) // 36
            slots_free = block_count * 36 - initial

            # First-card payload ~= 65 - len(key); continuation cards ~67.
            # (slots_free + 5) * 60 chars is comfortably over budget.
            big = "X" * (60 * (slots_free + 5))

            with pytest.raises(ValueError):
                h["ESO INS DESC"] = big
            assert "ESO INS DESC" not in h
        with rustfits.FITS(fname, "r") as fits:
            assert "ESO INS DESC" not in fits[0].header
