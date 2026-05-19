"""Case-preservation tests for HIERARCH long keys (ESO convention).

Contract:
  - On WRITE, the user's case is preserved verbatim on disk (only internal
    whitespace is collapsed to single spaces).  Standard 8-char keys are
    still uppercased (FITS standard requires uppercase).
  - On LOOKUP (__getitem__, __contains__, __delitem__, update()), HIERARCH
    matching is case-insensitive — `h["Eso Ins Det1"]` and
    `h["ESO INS DET1"]` and `h["eso ins det1"]` find the same card.
  - On UPDATE of an existing key, the on-disk card's spelling is KEPT —
    only the value changes.  This matches the "in-place update preserves
    position" rule extended to "...and spelling".  Matches astropy.

These tests pair same-handle and post-reopen assertions per the project
testing convention.
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


# ---------------------------------------------------------------------------
# storage preserves user case
# ---------------------------------------------------------------------------


def test_mixed_case_hierarch_preserves_case_on_disk():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["Eso Ins Det1 ExpTime"] = 5.0
            cards = fits[0].header.cards
            assert any(
                c.startswith("HIERARCH Eso Ins Det1 ExpTime") for c in cards
            )
        with rustfits.FITS(fname, "r") as fits:
            cards = fits[0].header.cards
            assert any(
                c.startswith("HIERARCH Eso Ins Det1 ExpTime") for c in cards
            )


def test_all_lowercase_hierarch_preserves_lowercase():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["eso tel airm start"] = 1.42
            cards = fits[0].header.cards
            assert any(c.startswith("HIERARCH eso tel airm start") for c in cards)


def test_uppercase_hierarch_still_uppercase_on_disk():
    """Existing all-uppercase HIERARCH callers stay byte-equivalent."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO INS DET1 EXPTIME"] = 5.0
            cards = fits[0].header.cards
            assert any(
                c.startswith("HIERARCH ESO INS DET1 EXPTIME") for c in cards
            )


def test_standard_short_key_still_uppercased_on_disk():
    """The case-preservation rule applies ONLY to HIERARCH.  A standard
    8-char key in lowercase is still uppercased on disk (FITS standard)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["exptime"] = 5
            cards = fits[0].header.cards
            assert any(c.startswith("EXPTIME ") for c in cards)
            assert not any(c.startswith("exptime ") for c in cards)


# ---------------------------------------------------------------------------
# lookup is case-insensitive
# ---------------------------------------------------------------------------


def test_lookup_is_case_insensitive_across_variants():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            h["Eso Ins Det1 ExpTime"] = 5.0
            assert h["Eso Ins Det1 ExpTime"] == 5.0
            assert h["ESO INS DET1 EXPTIME"] == 5.0
            assert h["eso ins det1 exptime"] == 5.0
            assert h["eSo InS deT1 ExPtImE"] == 5.0
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["ESO INS DET1 EXPTIME"] == 5.0


def test_contains_is_case_insensitive():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            h["Eso Tel Airm Start"] = 1.42
            assert "Eso Tel Airm Start" in h
            assert "ESO TEL AIRM START" in h
            assert "eso tel airm start" in h
            assert "ESO TEL FOO" not in h


def test_delete_is_case_insensitive():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            h["Eso Tel Airm Start"] = 1.42
            del h["eso tel airm start"]
            assert "Eso Tel Airm Start" not in h
        with rustfits.FITS(fname, "r") as fits:
            assert "ESO TEL AIRM START" not in fits[0].header


# ---------------------------------------------------------------------------
# update() preserves existing card spelling
# ---------------------------------------------------------------------------


def test_update_existing_key_keeps_original_spelling():
    """Setting an existing HIERARCH key with a different-case spelling
    must KEEP the original card's spelling — value changes, key text
    on disk does not."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            h["Eso Tel Airm Start"] = 1.0
            h["ESO TEL AIRM START"] = 2.0   # same key, different case
            cards = h.cards
            # Original spelling preserved; new value committed.
            assert any(c.startswith("HIERARCH Eso Tel Airm Start") for c in cards)
            assert not any(c.startswith("HIERARCH ESO TEL AIRM START") for c in cards)
            assert h["ESO TEL AIRM START"] == 2.0
        with rustfits.FITS(fname, "r") as fits:
            cards = fits[0].header.cards
            assert any(c.startswith("HIERARCH Eso Tel Airm Start") for c in cards)
            assert fits[0].header["eso tel airm start"] == 2.0


def test_dict_update_preserves_user_case_on_new_key():
    """For brand-new keys via update(dict), the user's case from the
    dict literal is preserved on disk."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.update({"Eso Ins Det1 Gain": 1.5})
            cards = fits[0].header.cards
            assert any(c.startswith("HIERARCH Eso Ins Det1 Gain") for c in cards)


def test_dict_update_existing_key_keeps_spelling():
    """update(dict) on an existing HIERARCH key keeps the existing
    card's spelling, regardless of dict-side case."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            h["Eso Ins Det1 Gain"] = 1.5
            h.update({"ESO INS DET1 GAIN": 2.5})
            assert any(
                c.startswith("HIERARCH Eso Ins Det1 Gain")
                for c in h.cards
            )
            assert h["eso ins det1 gain"] == 2.5


# ---------------------------------------------------------------------------
# whitespace canonicalization
# ---------------------------------------------------------------------------


def test_extra_internal_spaces_collapsed_on_write():
    """ESO convention is single-space separators between words.  User
    input with multiple spaces is collapsed at write time.  (Non-space
    whitespace like tab is still rejected by validate_keyword — keywords
    must be space-separated per the convention.)"""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["Eso  Ins   Det1 Gain"] = 1.5
            cards = fits[0].header.cards
            # Exactly single spaces between words on disk.
            assert any(c.startswith("HIERARCH Eso Ins Det1 Gain") for c in cards)
            # Lookup matches both the canonical form and the as-written form.
            assert fits[0].header["ESO INS DET1 GAIN"] == 1.5
            assert fits[0].header["Eso  Ins  Det1  Gain"] == 1.5


# ---------------------------------------------------------------------------
# iteration / keys() returns the storage form (case-preserved)
# ---------------------------------------------------------------------------


def test_keys_returns_storage_form():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["Eso Tel Airm Start"] = 1.42
            keys = list(fits[0].header)
            assert "Eso Tel Airm Start" in keys
            assert "ESO TEL AIRM START" not in keys


# ---------------------------------------------------------------------------
# CONTINUE-chained long string with mixed-case HIERARCH key
# ---------------------------------------------------------------------------


def test_long_string_value_with_mixed_case_hierarch_key():
    """The CONTINUE-chained HIERARCH first card must carry the user's
    case in the keyword position."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            big = "X" * 300
            fits[0].header["Eso Ins Det1 Desc"] = big
            assert fits[0].header["ESO INS DET1 DESC"] == big
            cards = fits[0].header.cards
            assert any(
                c.startswith("HIERARCH Eso Ins Det1 Desc") for c in cards
            )
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["eso ins det1 desc"] == big


if __name__ == "__main__":
    test_mixed_case_hierarch_preserves_case_on_disk()
    test_all_lowercase_hierarch_preserves_lowercase()
    test_uppercase_hierarch_still_uppercase_on_disk()
    test_standard_short_key_still_uppercased_on_disk()
    test_lookup_is_case_insensitive_across_variants()
    test_contains_is_case_insensitive()
    test_delete_is_case_insensitive()
    test_update_existing_key_keeps_original_spelling()
    test_dict_update_preserves_user_case_on_new_key()
    test_dict_update_existing_key_keeps_spelling()
    test_extra_internal_spaces_collapsed_on_write()
    test_keys_returns_storage_form()
    test_long_string_value_with_mixed_case_hierarch_key()
    print("smoke tests passed")
