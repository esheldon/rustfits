"""
Phase 2c, step 1: commentary-key write API.

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
    """
    The second add_comment lands immediately after the first one — even
    if other keys have been added in between."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("first")
            fits[0].header["EXPTIME"] = 5  # regular key in between
            fits[0].header.add_comment("second")
            cards_same = fits[0].header.cards
        with _read_only(fname) as fits:
            cards_reopen = fits[0].header.cards
        for cards in (cards_same, cards_reopen):
            idx1 = next(
                i
                for i, c in enumerate(cards)
                if c.startswith("COMMENT") and "first" in c
            )
            idx2 = next(
                i
                for i, c in enumerate(cards)
                if c.startswith("COMMENT") and "second" in c
            )
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
    """
    A failed edit block leaves the parent header — including its
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


# ============================================================================
# update(other, copy_commentary=...) from a FITSHeader source
# ============================================================================


def test_update_default_silently_skips_commentary_in_source():
    """
    Default: copy_commentary=False.  Commentary cards in a FITSHeader
    source are silently dropped (parallels how protected keys are handled
    from a FITSHeader source).  No exception, no leaked HISTORY/COMMENT."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
            a[0].header.add_comment("from source")
            a[0].header.add_history("ran step 1")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header)
            assert b[0].header["OBJECT"] == "M31"
            with pytest.raises(KeyError):
                _ = b[0].header["COMMENT"]
            with pytest.raises(KeyError):
                _ = b[0].header["HISTORY"]
        with _read_only(b_name) as b:
            assert b[0].header["OBJECT"] == "M31"
            with pytest.raises(KeyError):
                _ = b[0].header["HISTORY"]


def test_update_copy_commentary_appends_history_and_comment():
    """
    With copy_commentary=True, COMMENT and HISTORY cards in the source
    are appended verbatim to the destination."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
            a[0].header.add_comment("from source")
            a[0].header.add_history("did a thing")
            a[0].header.add_history("did another thing")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header, copy_commentary=True)
            assert b[0].header["OBJECT"] == "M31"
            assert b[0].header["COMMENT"] == ["from source"]
            assert b[0].header["HISTORY"] == [
                "did a thing",
                "did another thing",
            ]
        with _read_only(b_name) as b:
            assert b[0].header["COMMENT"] == ["from source"]
            assert b[0].header["HISTORY"] == [
                "did a thing",
                "did another thing",
            ]


def test_update_copy_commentary_appends_blank_commentary():
    """Blank-keyword commentary (cols 1-8 spaces) is also copied."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header.add_blank("blank-key text")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header, copy_commentary=True)
            assert b[0].header[""] == ["blank-key text"]


def test_update_copy_commentary_preserves_source_card_split():
    """
    A long commentary that the source split across multiple cards stays
    split — one append per source card, no concatenation."""
    long_text = "X" * 200  # splits into 72 + 72 + 56 = 3 cards
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header.add_comment(long_text)
            # Sanity: source has 3 COMMENT cards.
            n_src = sum(
                1 for c in a[0].header.cards if c.startswith("COMMENT")
            )
            assert n_src == 3
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header, copy_commentary=True)
            n_dst = sum(
                1 for c in b[0].header.cards if c.startswith("COMMENT")
            )
            assert n_dst == 3
            # Joining the 3 entries reconstructs the original text.
            assert "".join(b[0].header["COMMENT"]) == long_text


def test_update_copy_commentary_repeated_calls_accumulate():
    """
    Documented hazard: calling update(..., copy_commentary=True) twice
    duplicates the source's commentary in the destination.  This is by
    design — that's why the default is False and no deduplication is done.
    Users who want a one-shot copy should opt in once."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header.add_history("first run")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header, copy_commentary=True)
            b[0].header.update(a[0].header, copy_commentary=True)
            assert b[0].header["HISTORY"] == ["first run", "first run"]


def test_update_copy_commentary_coexists_with_dest_commentary():
    """
    Source commentary is appended after the destination's existing
    commentary cards (cluster-with-same-keyword position rule)."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header.add_history("from source")
        with rustfits.FITS(b_name, "r+") as b:
            b[0].header.add_history("from dest")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header, copy_commentary=True)
            assert b[0].header["HISTORY"] == ["from dest", "from source"]


def test_update_dict_source_still_raises_on_commentary():
    """
    Dict source's COMMENT/HISTORY keys raise regardless of the flag —
    the flag is meaningful only for header-to-header copy.  An explicit
    `"COMMENT"` in a dict is almost certainly a mistake (single value vs
    append is ambiguous), so we reject it loudly."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header.update({"COMMENT": "no good"})
            with pytest.raises(ValueError):
                fits[0].header.update(
                    {"COMMENT": "no good"},
                    copy_commentary=True,
                )


def test_update_copy_commentary_inside_edit_batch():
    """
    copy_commentary=True works inside header.edit() and commits
    atomically with the rest of the staged changes."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
            a[0].header.add_history("from source")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            with b[0].header.edit() as h:
                h.update(a[0].header, copy_commentary=True)
            assert b[0].header["OBJECT"] == "M31"
            assert b[0].header["HISTORY"] == ["from source"]
        with _read_only(b_name) as b:
            assert b[0].header["HISTORY"] == ["from source"]


def test_update_default_silently_skips_commentary_inside_edit_batch():
    """
    Edit-batched update() with default copy_commentary=False also
    silently skips commentary — parallel to the non-batched path."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
            a[0].header.add_history("dropped")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            with b[0].header.edit() as h:
                h.update(a[0].header)
            assert b[0].header["OBJECT"] == "M31"
            with pytest.raises(KeyError):
                _ = b[0].header["HISTORY"]


def test_update_copy_commentary_edit_rollback_discards_appends():
    """
    An exception inside the edit() block discards both staged set-key
    actions and staged commentary appends."""
    with _new_file() as a_name, _new_file() as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header.add_history("staged then rolled back")
        with rustfits.FITS(a_name, "r") as a, rustfits.FITS(b_name, "r+") as b:
            with pytest.raises(RuntimeError):
                with b[0].header.edit() as h:
                    h.update(a[0].header, copy_commentary=True)
                    raise RuntimeError("boom")
            with pytest.raises(KeyError):
                _ = b[0].header["HISTORY"]


if __name__ == "__main__":
    test_add_comment_appends_one_card()
    test_multiple_comments_in_order()
    test_long_comment_splits_across_cards()
    test_del_comment_removes_all()
    test_subscript_set_commentary_rejected()
    test_add_comment_inside_edit_batch()
    test_edit_batch_rollback_keeps_comments_unchanged()
    print("smoke tests passed")
