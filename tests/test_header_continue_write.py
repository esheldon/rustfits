"""
Phase 2c, step 2: CONTINUE-on-write for long string values.

When a string value's FITS-escaped length exceeds 68 chars (or fits in 68
but the comment wouldn't), the writer auto-splits it across one keyword
card + N CONTINUE cards.  Comments land on the last card.  An existing
CONTINUE chain is fully removed before its replacement is inserted.

Every test verifies its assertion through BOTH the same FITS handle that
did the mutation AND a fresh reopen.
"""

import os
import tempfile
import contextlib

import pytest

import rustfits

# 36 cards per 2880-byte FITS header block (BLOCK_SIZE / CARD_SIZE).
# Used in the slack-fill arithmetic below.
CARDS_PER_BLOCK = 2880 // 80


@contextlib.contextmanager
def _new_file(shape=(4, 6), dtype="i4"):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


def _check_both(fname, fits, predicate):
    """
    Run `predicate(header)` on the live-handle header, then on a fresh
    reopen.  Used by tests where the same assertion holds in both views."""
    predicate(fits[0].header)
    with rustfits.FITS(fname, "r") as fits2:
        predicate(fits2[0].header)


# ============================================================================
# Single-card path is unchanged
# ============================================================================


def test_short_string_emits_single_card():
    """A string that fits comfortably in one card produces exactly one card."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["OBJECT"] = "M31"

            def check(hd):
                assert hd["OBJECT"] == "M31"
                # No CONTINUE cards introduced.
                continues = [c for c in hd.cards if c.startswith("CONTINUE")]
                assert continues == []

            _check_both(fname, fits, check)


def test_string_at_68_chars_still_one_card():
    """The boundary: 68 chars (max payload with no comment) — one card."""
    with _new_file() as fname:
        s = "X" * 68
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = s

            def check(hd):
                assert hd["LONG"] == s
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


# ============================================================================
# Multi-card CONTINUE chain
# ============================================================================


def test_string_just_over_68_chars_uses_continue():
    with _new_file() as fname:
        s = "X" * 69
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = s

            def check(hd):
                assert hd["LONG"] == s
                continues = [c for c in hd.cards if c.startswith("CONTINUE")]
                assert len(continues) >= 1

            _check_both(fname, fits, check)


def test_long_string_round_trip_no_comment():
    """200-char string round-trips through CONTINUE write/read."""
    with _new_file() as fname:
        s = ("ABCDEFGHIJ" * 25)[:200]  # 200 chars
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = s

            def check(hd):
                assert hd["LONG"] == s

            _check_both(fname, fits, check)


def test_long_string_round_trip_with_comment():
    with _new_file() as fname:
        s = "Y" * 200
        c = "long string value"
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = (s, c)

            def check(hd):
                assert hd["LONG"] == s
                # Comment lands on the LAST card; readback joins comments
                # space-separated across CONTINUE cards.  Since we only put
                # the comment on the last card, the result is exactly `c`.
                assert hd.comment_of("LONG") == c

            _check_both(fname, fits, check)


def test_chain_uses_keyword_then_continue_cards():
    """
    Inspect the on-disk card layout: first card begins with KEYWORD,
    each subsequent card begins with CONTINUE."""
    with _new_file() as fname:
        s = "Z" * 200
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = s

            def check(hd):
                cards = hd.cards
                # Find the LONG card.
                first_idx = next(
                    i for i, c in enumerate(cards) if c.startswith("LONG")
                )
                # All cards immediately after, until the chain ends, must be
                # CONTINUE cards.
                chain = [cards[first_idx]]
                i = first_idx + 1
                while i < len(cards) and cards[i].startswith("CONTINUE"):
                    chain.append(cards[i])
                    i += 1
                # We need at least 3 cards for 200 chars (67 + 67 + 66).
                assert len(chain) >= 3
                # Non-last cards inside the chain end with `&'` before the
                # 80-byte boundary (some trailing space is fine since `cards`
                # is trimmed).  Last card has no `&` before the closing `'`.
                for card in chain[:-1]:
                    # The card carries an inner string ending with `&`.
                    assert "&'" in card
                assert not chain[-1].rstrip().endswith("&'")

            _check_both(fname, fits, check)


# ============================================================================
# Embedded single quotes survive chunking
# ============================================================================


def test_long_string_with_embedded_quotes_round_trips():
    """The chunker must not split inside a `''` escape pair."""
    with _new_file() as fname:
        # 200 chars with several quotes sprinkled throughout, ending in a
        # non-space so the FITS "trailing spaces not significant" rule
        # doesn't trim our test data.
        s = ("It's M31's brightest core; " * 8)[:199] + "Z"
        assert "'" in s
        assert not s.endswith(" ")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = s

            def check(hd):
                assert hd["LONG"] == s

            _check_both(fname, fits, check)


# ============================================================================
# Updating an existing CONTINUE chain
# ============================================================================


def test_replacing_chain_with_short_string_removes_extra_cards():
    """
    Overwrite a long value with a short one — the CONTINUE cards must
    all go, not just the first card of the chain."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = "Q" * 200
            chain_cards_before = sum(
                1 for c in fits[0].header.cards if c.startswith("CONTINUE")
            )
            assert chain_cards_before >= 2

            fits[0].header["LONG"] = "short"

            def check(hd):
                assert hd["LONG"] == "short"
                # No orphaned CONTINUE cards from the old chain.
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


def test_replacing_chain_with_longer_chain():
    """
    Overwrite a 200-char value with a 400-char value — new chain replaces
    old, no leftovers."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = "A" * 200
            fits[0].header["LONG"] = "B" * 400

            def check(hd):
                assert hd["LONG"] == "B" * 400

            _check_both(fname, fits, check)


def test_del_chain_removes_all_cards():
    """del header[key] for a CONTINUE-chained value removes the whole chain."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = "X" * 200
            assert any(c.startswith("CONTINUE") for c in fits[0].header.cards)

            del fits[0].header["LONG"]

            def check(hd):
                assert "LONG" not in hd
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


# ============================================================================
# Comment-too-long rejection
# ============================================================================


def test_comment_too_long_for_chained_value_rejected():
    """Comments are capped at 64 chars for CONTINUE-chained values."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                # Long value triggers CONTINUE; comment > 64 chars must error.
                fits[0].header["LONG"] = ("Z" * 200, "C" * 80)


# ============================================================================
# Position semantics
# ============================================================================


def test_new_chain_lands_before_end():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = "W" * 150

            def check(hd):
                cards = hd.cards
                assert cards[-1].startswith("END")
                # The chain occupies the last few cards before END.
                first_idx = next(
                    i for i, c in enumerate(cards) if c.startswith("LONG")
                )
                # Every card from first_idx to second-to-last is part of the
                # chain.
                for card in cards[first_idx + 1 : -1]:
                    assert card.startswith("CONTINUE")

            _check_both(fname, fits, check)


def test_replacing_existing_key_preserves_position():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            keys_before = list(fits[0].header)

            # Now upgrade it to a long string value (different chain length).
            fits[0].header["EXPTIME"] = "X" * 150
            keys_after = list(fits[0].header)
            assert keys_after == keys_before

        with rustfits.FITS(fname, "r") as fits:
            assert list(fits[0].header) == keys_before


# ============================================================================
# Inside a FITSHeaderEdit batch
# ============================================================================


def test_long_string_in_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["LONG"] = "P" * 200
                # Staged read agrees.
                assert h["LONG"] == "P" * 200

            def check(hd):
                assert hd["LONG"] == "P" * 200

            _check_both(fname, fits, check)


def test_edit_rollback_discards_long_string_chain():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h["LONG"] = "Q" * 200
                    raise RuntimeError("boom")

            def check(hd):
                assert "LONG" not in hd
                # No orphaned CONTINUE cards from the staged but uncommitted
                # chain.
                assert not any(c.startswith("CONTINUE") for c in hd.cards)

            _check_both(fname, fits, check)


# ============================================================================
# Long string spilling past reserved slack triggers the header grow path
# ============================================================================


def test_long_string_triggers_header_grow():
    """
    A CONTINUE chain too long for the reserved header block(s) triggers
    the file-tail shift and grows the reserved region rather than failing."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            initial = len(h.cards)
            block_count = (initial + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK
            slots_free = block_count * CARDS_PER_BLOCK - initial

            # Build a value long enough to require more cards than the slack
            # supports.  ~67 chars per non-last card, so (slots_free + 5) * 67
            # chars is comfortably over budget — overflow triggers grow.
            big = "X" * (67 * (slots_free + 5))

            h["LONG"] = big
            assert h["LONG"] == big
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["LONG"] == big
