"""
Tests for the FITSHeaderEdit context-manager batching API.

Verifies that:
    - Mutations inside a `with header.edit():` block stage in memory and
      commit on successful exit (one disk rewrite per block).
    - An exception inside the block discards staged changes (rollback).
    - Reads inside the block observe staged state; reads on the parent
      header during the block still see the pre-batch state.
    - The handle rejects mutations outside a `with` and after commit.
    - update() and __delitem__ behave the same as on FITSHeader, but
      batched.
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
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


# --------------------------- happy path ----------------------------


def test_edit_commits_on_normal_exit():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["EXPTIME"] = 5
                h["OBJECT"] = "M31"
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["EXPTIME"] == 5
            assert fits[0].header["OBJECT"] == "M31"


def test_edit_supports_value_comment_tuples():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["EXPTIME"] = (5.0, "exposure (s)")
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header.comment_of("EXPTIME") == "exposure (s)"


def test_edit_supports_delete():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            with fits[0].header.edit() as h:
                del h["EXPTIME"]
        with rustfits.FITS(fname, "r") as fits:
            assert "EXPTIME" not in fits[0].header


def test_edit_supports_update_with_dict():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h.update({"A": 1, "B": 2, "C": (3.0, "with comment")})
        with rustfits.FITS(fname, "r") as fits:
            hd = fits[0].header
            assert hd["A"] == 1
            assert hd["B"] == 2
            assert hd["C"] == 3.0
            assert hd.comment_of("C") == "with comment"


# -------------------- rollback on exception ----------------------


def test_edit_rolls_back_on_exception():
    """
    If an exception escapes the with-block, staged mutations are
    discarded — the on-disk file and parent header are unchanged."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h["EXPTIME"] = 5
                    raise RuntimeError("boom")
            # Parent header sees nothing.
            assert "EXPTIME" not in fits[0].header
        # Disk sees nothing.
        with rustfits.FITS(fname, "r") as fits:
            assert "EXPTIME" not in fits[0].header


# --------------------- staged-vs-parent reads ---------------------


def test_edit_reads_show_staged_state():
    """Subscript on the edit handle observes the staged state immediately."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["EXPTIME"] = 5
                assert h["EXPTIME"] == 5
                assert "EXPTIME" in h
                h["EXPTIME"] = 10
                assert h["EXPTIME"] == 10


def test_edit_parent_unchanged_until_commit():
    """
    During the batch, the parent FITSHeader still shows the pre-batch
    state — committing only happens on __exit__."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            header = fits[0].header
            with header.edit() as h:
                h["EXPTIME"] = 5
                # Parent (re-read via the HDU) still doesn't have it.
                assert "EXPTIME" not in header
            # Now committed.
            assert header["EXPTIME"] == 5


# --------------------------- guard rails ----------------------------


def test_edit_setitem_outside_with_raises():
    """
    Trying to use the handle without entering the `with` is an error
    (otherwise the user would silently lose data)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            e = fits[0].header.edit()
            with pytest.raises(ValueError):
                e["EXPTIME"] = 5


def test_edit_delitem_outside_with_raises():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            e = fits[0].header.edit()
            with pytest.raises(ValueError):
                del e["EXPTIME"]


def test_edit_update_outside_with_raises():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            e = fits[0].header.edit()
            with pytest.raises(ValueError):
                e.update({"X": 1})


def test_edit_setitem_after_commit_raises():
    """After the with-block exits and commit fires, the handle is spent."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["EXPTIME"] = 5
            with pytest.raises(ValueError):
                h["OTHER"] = 1


# ------------------- batching is atomic on disk --------------------


def test_edit_overflow_triggers_grow_at_commit():
    """
    When staged mutations exceed the reserved block(s), commit triggers
    the file-tail shift, grows the reserved region, and lands every staged
    key — there is no longer a slack-overflow error to roll back from."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            header = fits[0].header
            initial = len(header.cards)
            block_count = (initial + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK
            slots_free = block_count * CARDS_PER_BLOCK - initial
            with header.edit() as h:
                for i in range(slots_free + 1):
                    h[f"PAD{i:04d}"] = i
            # All committed via the grow path.
            for i in range(slots_free + 1):
                assert header[f"PAD{i:04d}"] == i
        with rustfits.FITS(fname, "r") as fits:
            for i in range(slots_free + 1):
                assert fits[0].header[f"PAD{i:04d}"] == i


def test_edit_position_semantics_match_setitem():
    """
    An update to an existing key keeps its position; new keys land
    immediately before END — same as outside a batch."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            # Seed a non-protected key so we have something to update in
            # place (BITPIX et al. are protected and rejected).
            fits[0].header["OBJECT"] = "M31"
            keys_before = list(fits[0].header)
            with fits[0].header.edit() as h:
                h["OBJECT"] = "NGC 224"   # update existing
                h["EXPTIME"] = 5          # new key
            cards = fits[0].header.cards
            assert cards[-1].startswith("END")
            assert cards[-2].startswith("EXPTIME")
            keys_after = list(fits[0].header)
            # Same keys as before, plus EXPTIME at the end.
            assert keys_after[:-1] == keys_before
            assert keys_after[-1] == "EXPTIME"


# --------------------------- repr ----------------------------


def test_edit_repr_reflects_state():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            e = fits[0].header.edit()
            assert "pending" in repr(e)
            with e as h:
                assert "active" in repr(h)
            assert "committed" in repr(e)


if __name__ == "__main__":
    test_edit_commits_on_normal_exit()
    test_edit_rolls_back_on_exception()
    test_edit_parent_unchanged_until_commit()
    test_edit_setitem_outside_with_raises()
    test_edit_overflow_triggers_grow_at_commit()
    test_edit_position_semantics_match_setitem()
    print("smoke tests passed")
