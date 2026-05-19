"""Tests for FITSHeader mutation basics:
    - __setitem__ (bare value and (value, comment) tuple)
    - __delitem__
    - update() from dict, mapping, FITSHeader source
    - Position semantics (existing key in place, new key before END)
    - Persistence across close/reopen
    - Case-insensitive keyword normalization
    - Header overflow triggers the file-tail shift (grow) path
    - Commentary key assignment via subscript raises (use the dedicated
      add_comment/add_history/add_blank helpers instead).
"""

import os
import tempfile
import contextlib

import pytest

import rustfits

# 36 cards per 2880-byte FITS header block (BLOCK_SIZE / CARD_SIZE).
CARDS_PER_BLOCK = 2880 // 80


@contextlib.contextmanager
def _new_file(shape=(4, 6), dtype="i4"):
    """
    Create a fresh single-HDU file with the given shape/dtype, yield path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


def _reopen(fname):
    """Open file fresh; returns the header view of HDU 0."""
    fits = rustfits.FITS(fname, "r")
    return fits, fits[0].header


# ----------------------- __setitem__: bare value ------------------------


def test_set_new_int_key_and_persist():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
        # Reopen and verify the new card landed on disk.
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["EXPTIME"] == 5


def test_set_new_float_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["TEMP"] = 12.5
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["TEMP"] == 12.5
            assert isinstance(h["TEMP"], float)


def test_set_new_string_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["OBJECT"] = "M31"
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["OBJECT"] == "M31"


def test_set_new_bool_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["DOPHOT"] = True
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["DOPHOT"] is True


# -------- __setitem__: (value, comment) tuple -------


def test_set_with_comment_tuple():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = (5.0, "exposure (s)")
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["EXPTIME"] == 5.0
            assert h.comment_of("EXPTIME") == "exposure (s)"


def test_set_bare_value_preserves_existing_comment():
    """Updating an existing key with a bare value keeps the prior comment."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = (5.0, "exposure (s)")
            fits[0].header["EXPTIME"] = 10.0   # bare value
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["EXPTIME"] == 10.0
            assert h.comment_of("EXPTIME") == "exposure (s)"


def test_set_with_explicit_empty_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = (5.0, "exposure (s)")
            fits[0].header["EXPTIME"] = (10.0, "")  # explicit empty
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header.comment_of("EXPTIME") == ""


def test_set_with_malformed_tuple_raises():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["EXPTIME"] = (5.0, "extra", "elements")


# --------------------------- Position semantics ----------------------------


def test_update_existing_key_preserves_position():
    """An update to an existing key keeps the card at its original index."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            # Seed a non-protected key, then verify mutating it preserves
            # its position.  (Structural keys like BITPIX are protected and
            # cannot be set directly — see test_header_protected.py.)
            fits[0].header["OBJECT"] = "M31"
            keys_before = list(fits[0].header)
            fits[0].header["OBJECT"] = "NGC 224"   # update existing
            keys_after = list(fits[0].header)
        # Same keys in the same order — only OBJECT's value changed.
        assert keys_before == keys_after


def test_new_key_inserted_before_end():
    """
    A brand-new key lands just before END (so END remains last in the file).
    """
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            cards = fits[0].header.cards
        # END must still be the last card; EXPTIME must be the second-to-last.
        assert cards[-1].startswith("END")
        assert cards[-2].startswith("EXPTIME")


# -------------------------- __delitem__ ---------------------------


def test_delete_key_and_persist():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            assert "EXPTIME" in fits[0].header
            del fits[0].header["EXPTIME"]
            assert "EXPTIME" not in fits[0].header
        with rustfits.FITS(fname, "r") as fits:
            assert "EXPTIME" not in fits[0].header


def test_delete_keeps_end_at_last_position():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            fits[0].header["OBJECT"] = "M31"
            del fits[0].header["EXPTIME"]
            cards = fits[0].header.cards
        # END must still be the last card; OBJECT is now second-to-last.
        assert cards[-1].startswith("END")
        assert cards[-2].startswith("OBJECT")


def test_delete_missing_key_raises_keyerror():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(KeyError):
                del fits[0].header["NOPE"]


# --------------------------- update(): dict ----------------------------


def test_update_with_dict_bare_values():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.update({"EXPTIME": 5, "OBJECT": "M31"})
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["EXPTIME"] == 5
            assert h["OBJECT"] == "M31"


def test_update_with_dict_value_comment_tuples():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.update({
                "EXPTIME": (5.0, "exposure (s)"),
                "OBJECT":  ("M31", "target"),
            })
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["EXPTIME"] == 5.0
            assert h.comment_of("EXPTIME") == "exposure (s)"
            assert h["OBJECT"] == "M31"
            assert h.comment_of("OBJECT") == "target"


def test_update_with_commentary_key_raises():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header.update({"COMMENT": "no good"})


# ----------------- update(): FITSHeader source --------------------


def test_update_with_fitsheader_source_copies_comments():
    """update() from a FITSHeader carries comments from the source's cards."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["EXPTIME"] = (5.0, "exposure (s)")
            a[0].header["OBJECT"] = ("M31", "target")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header)
        with rustfits.FITS(b_name, "r") as b:
            h = b[0].header
            assert h["EXPTIME"] == 5.0
            assert h.comment_of("EXPTIME") == "exposure (s)"
            assert h["OBJECT"] == "M31"
            assert h.comment_of("OBJECT") == "target"


# --------------------------- Case insensitivity ----------------------------


def test_setitem_lowercase_key_normalized_to_uppercase():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["exptime"] = 5
        # Both case variants resolve to the same card.
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["EXPTIME"] == 5
            assert h["exptime"] == 5     # lookup is case-insensitive too
            assert h["Exptime"] == 5
            # The card on disk is the uppercase form.
            card = next(c for c in h.cards if c.startswith("EXPTIME"))
            assert card.startswith("EXPTIME")


def test_delitem_lowercase_key_works():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            del fits[0].header["exptime"]
            assert "EXPTIME" not in fits[0].header


def test_invalid_keyword_chars_rejected():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["BAD*KEY"] = 5


def test_long_keyword_promoted_to_hierarch():
    """Keys longer than 8 characters are now written as HIERARCH cards
    instead of being rejected (phase 2c).  Comprehensive HIERARCH behavior
    is covered in tests/test_header_hierarch_write.py; this is a regression
    guard against the old "reject long keys" contract from phase 2a."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["TOOMANYCHARS"] = 5
            assert fits[0].header["TOOMANYCHARS"] == 5


# --------------------------- Slack-only overflow ----------------------------


def test_header_overflow_triggers_grow():
    """One card past capacity triggers the file-tail shift and grows the
    reserved header blocks; the new key lands on disk and round-trips
    through close-and-reopen."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            initial_cards = len(h.cards)
            block_count = (initial_cards + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK
            capacity = block_count * CARDS_PER_BLOCK
            slots_free = capacity - initial_cards
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            # One more triggers the grow path rather than raising.
            h["OVERFLOW"] = 1
            assert h["OVERFLOW"] == 1
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["OVERFLOW"] == 1


# -------------------- Shared state across views --------------------


def test_two_header_views_share_state():
    """Two FITSHeader handles from the same HDU observe each other's writes."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h1 = fits[0].header
            h2 = fits[0].header
            h1["EXPTIME"] = 5
            # Without reopening, h2 sees the change because cards are shared.
            assert h2["EXPTIME"] == 5


# ------------------- Auto-flush per mutation ---------------------


def test_each_setitem_persists_independently():
    """Each __setitem__ writes immediately — no explicit flush required."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["A"] = 1
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["B"] = 2
        with rustfits.FITS(fname, "r") as fits:
            h = fits[0].header
            assert h["A"] == 1
            assert h["B"] == 2


if __name__ == "__main__":
    # Quick smoke run without pytest.
    test_set_new_int_key_and_persist()
    test_set_with_comment_tuple()
    test_update_existing_key_preserves_position()
    test_new_key_inserted_before_end()
    test_delete_keeps_end_at_last_position()
    test_update_with_dict_value_comment_tuples()
    test_update_with_fitsheader_source_copies_comments()
    test_header_overflow_triggers_grow()
    test_two_header_views_share_state()
    print("smoke tests passed")
