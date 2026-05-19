"""Phase 2c, step 1: commentary-key write API.

Each test verifies its assertion through BOTH:
    - the same FITS handle that did the mutation (same-handle / in-memory
      consistency check), and
    - a fresh reopen of the file (on-disk persistence check).

This catches the entire class of bugs where the in-memory cards Vec and
the file would diverge — exactly the kind of issue the phase-2 write-disk-
first invariant is designed to prevent.

Covers:
    - add_comment / add_history / add_blank append a card (or multiple for
      long text).
    - New cards cluster with existing same-keyword cards (immediately after
      the last one), or land just before END if none exist.
    - The three families are independent.
    - del header["COMMENT"] removes all matching cards; KeyError if none.
    - Subscript assignment to a commentary key is rejected with a message
      pointing to the add_* methods.
    - Non-printable text is rejected.
    - All of the above also work inside a FITSHeaderEdit batch, with
      rollback on exception.
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


def _read_only(fname):
    """Open a fresh read-only FITS handle for the persistence check."""
    return rustfits.FITS(fname, "r")


# ============================================================================
# add_comment / add_history / add_blank
# ============================================================================


def test_add_comment_appends_one_card():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("first line")
            # same-handle
            assert fits[0].header["COMMENT"] == ["first line"]
        # post-reopen
        with _read_only(fname) as fits:
            assert fits[0].header["COMMENT"] == ["first line"]


def test_add_history_appends_one_card():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_history("processed step 1")
            assert fits[0].header["HISTORY"] == ["processed step 1"]
        with _read_only(fname) as fits:
            assert fits[0].header["HISTORY"] == ["processed step 1"]


def test_add_blank_appends_one_card():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_blank("notes without a keyword")
            assert fits[0].header[""] == ["notes without a keyword"]
        with _read_only(fname) as fits:
            assert fits[0].header[""] == ["notes without a keyword"]


def test_multiple_comments_in_order():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("one")
            fits[0].header.add_comment("two")
            fits[0].header.add_comment("three")
            assert fits[0].header["COMMENT"] == ["one", "two", "three"]
        with _read_only(fname) as fits:
            assert fits[0].header["COMMENT"] == ["one", "two", "three"]


def test_new_comments_cluster_with_existing_ones():
    """The second add_comment lands immediately after the first one — even
    if other keys have been added in between."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("first")
            fits[0].header["EXPTIME"] = 5      # regular key in between
            fits[0].header.add_comment("second")
            cards_same = fits[0].header.cards
        with _read_only(fname) as fits:
            cards_reopen = fits[0].header.cards
        for cards in (cards_same, cards_reopen):
            idx1 = next(i for i, c in enumerate(cards) if c.startswith("COMMENT") and "first" in c)
            idx2 = next(i for i, c in enumerate(cards) if c.startswith("COMMENT") and "second" in c)
            assert idx2 == idx1 + 1


def test_first_comment_lands_before_end():
    """With no existing COMMENT cards, the new one goes just before END."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("only one")
            cards_same = fits[0].header.cards
        with _read_only(fname) as fits:
            cards_reopen = fits[0].header.cards
        for cards in (cards_same, cards_reopen):
            assert cards[-1].startswith("END")
            assert cards[-2].startswith("COMMENT")
            assert "only one" in cards[-2]


def test_comment_history_blank_are_independent():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("c1")
            fits[0].header.add_history("h1")
            fits[0].header.add_blank("b1")
            fits[0].header.add_comment("c2")
            hd = fits[0].header
            assert hd["COMMENT"] == ["c1", "c2"]
            assert hd["HISTORY"] == ["h1"]
            assert hd[""] == ["b1"]
        with _read_only(fname) as fits:
            hd = fits[0].header
            assert hd["COMMENT"] == ["c1", "c2"]
            assert hd["HISTORY"] == ["h1"]
            assert hd[""] == ["b1"]


# ============================================================================
# Long-text splitting (no CONTINUE for commentary keys)
# ============================================================================


def test_long_comment_splits_across_cards():
    with _new_file() as fname:
        long = "X" * 150  # 150 / 72 = 3 cards (72 + 72 + 6)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment(long)
            parts_same = fits[0].header["COMMENT"]
        with _read_only(fname) as fits:
            parts_reopen = fits[0].header["COMMENT"]
        for parts in (parts_same, parts_reopen):
            assert len(parts) == 3
            assert "".join(parts) == long


def test_exactly_72_chars_one_card():
    with _new_file() as fname:
        text = "Y" * 72
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment(text)
            assert fits[0].header["COMMENT"] == [text]
        with _read_only(fname) as fits:
            assert fits[0].header["COMMENT"] == [text]


def test_empty_text_produces_one_blank_card():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("")
            entries_same = fits[0].header["COMMENT"]
        with _read_only(fname) as fits:
            entries_reopen = fits[0].header["COMMENT"]
        for entries in (entries_same, entries_reopen):
            assert len(entries) == 1
            assert entries[0].strip() == ""


# ============================================================================
# Delete (del header["COMMENT"])
# ============================================================================


def test_del_comment_removes_all():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("a")
            fits[0].header.add_comment("b")
            assert fits[0].header["COMMENT"] == ["a", "b"]
            del fits[0].header["COMMENT"]
            assert "COMMENT" not in fits[0].header
        with _read_only(fname) as fits:
            assert "COMMENT" not in fits[0].header


def test_del_history_removes_all_and_leaves_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("c1")
            fits[0].header.add_history("h1")
            fits[0].header.add_history("h2")
            del fits[0].header["HISTORY"]
            hd = fits[0].header
            assert hd["COMMENT"] == ["c1"]
            assert "HISTORY" not in hd
        with _read_only(fname) as fits:
            hd = fits[0].header
            assert hd["COMMENT"] == ["c1"]
            assert "HISTORY" not in hd


def test_del_blank_removes_all():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_blank("b1")
            fits[0].header.add_blank("b2")
            del fits[0].header[""]
            assert "" not in fits[0].header
        with _read_only(fname) as fits:
            assert "" not in fits[0].header


def test_del_missing_commentary_raises_keyerror():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(KeyError):
                del fits[0].header["COMMENT"]


# ============================================================================
# Subscript-assignment to commentary still rejected
# ============================================================================


def test_subscript_set_commentary_rejected():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header["COMMENT"] = "no good"
            with pytest.raises(ValueError):
                fits[0].header["COMMENT"] = ["list", "form", "also no"]
            with pytest.raises(ValueError):
                fits[0].header["HISTORY"] = "nope"
            with pytest.raises(ValueError):
                fits[0].header[""] = "nope"


# ============================================================================
# Input validation
# ============================================================================


def test_non_printable_text_rejected():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header.add_comment("oops\nhas newline")
            with pytest.raises(ValueError):
                fits[0].header.add_history("tab\there")
            # And the file is unchanged.
            assert "COMMENT" not in fits[0].header
            assert "HISTORY" not in fits[0].header
        with _read_only(fname) as fits:
            assert "COMMENT" not in fits[0].header
            assert "HISTORY" not in fits[0].header


# ============================================================================
# Inside FITSHeaderEdit (staged batching)
# ============================================================================


def test_add_comment_inside_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h.add_comment("from batch")
                h.add_history("step A")
                # staged-state visible through the handle
                assert h["COMMENT"] == ["from batch"]
                assert h["HISTORY"] == ["step A"]
            # post-commit, visible via the parent (same handle, no reopen)
            hd = fits[0].header
            assert hd["COMMENT"] == ["from batch"]
            assert hd["HISTORY"] == ["step A"]
        with _read_only(fname) as fits:
            hd = fits[0].header
            assert hd["COMMENT"] == ["from batch"]
            assert hd["HISTORY"] == ["step A"]


def test_del_comment_inside_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("will be removed")
            with fits[0].header.edit() as h:
                del h["COMMENT"]
                assert "COMMENT" not in h
            # post-commit
            assert "COMMENT" not in fits[0].header
        with _read_only(fname) as fits:
            assert "COMMENT" not in fits[0].header


def test_edit_batch_rollback_keeps_comments_unchanged():
    """A failed edit block leaves the parent header — including its
    commentary — unchanged."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("keep me")
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h.add_comment("about to die")
                    del h["COMMENT"]
                    raise RuntimeError("boom")
            # same handle: rollback observable
            assert fits[0].header["COMMENT"] == ["keep me"]
        with _read_only(fname) as fits:
            assert fits[0].header["COMMENT"] == ["keep me"]


def test_add_comment_outside_with_raises():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            e = fits[0].header.edit()
            with pytest.raises(ValueError):
                e.add_comment("nope")
            # nothing committed
            assert "COMMENT" not in fits[0].header


# ============================================================================
# Coexistence with the existing scalar setitem path
# ============================================================================


def test_regular_setitem_still_works_alongside_commentary():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = (5.0, "exposure (s)")
            fits[0].header.add_comment("processed")
            hd = fits[0].header
            assert hd["EXPTIME"] == 5.0
            assert hd.comment_of("EXPTIME") == "exposure (s)"
            assert hd["COMMENT"] == ["processed"]
        with _read_only(fname) as fits:
            hd = fits[0].header
            assert hd["EXPTIME"] == 5.0
            assert hd.comment_of("EXPTIME") == "exposure (s)"
            assert hd["COMMENT"] == ["processed"]


if __name__ == "__main__":
    test_add_comment_appends_one_card()
    test_multiple_comments_in_order()
    test_long_comment_splits_across_cards()
    test_del_comment_removes_all()
    test_subscript_set_commentary_rejected()
    test_add_comment_inside_edit_batch()
    test_edit_batch_rollback_keeps_comments_unchanged()
    print("smoke tests passed")
