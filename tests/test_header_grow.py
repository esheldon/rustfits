"""Tests for the file-rewrite (grow) path that fires when a header
mutation needs more cards than the currently reserved blocks can hold.

Before the grow path landed, overflowing the reserved blocks raised
ValueError; now the file tail is shifted forward, the reserved header
region grows by N blocks, and the new card lands on disk.  These tests
cover:

  - single-HDU grow round-trips (same-handle + post-reopen)
  - multi-HDU files: data integrity of subsequent HDUs after a grow
  - previously-issued HDU / FITSHeader handles see post-grow offsets
    transparently (the Arc<HduOffsets> is shared)
  - grow inside FITSHeaderEdit commit
  - grow on the last HDU (no tail to shift, just file extend)
  - multi-block grow in one shot
  - sequential grows compose
  - grow triggered by add_comment / update()
"""

import os
import tempfile
import contextlib

import numpy as np

import rustfits

# 36 cards per 2880-byte FITS header block (BLOCK_SIZE / CARD_SIZE).
# Used in the slack-fill arithmetic below.
CARDS_PER_BLOCK = 2880 // 80


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _new_single_hdu(shape=(4, 6), dtype="i4"):
    """One image HDU, fresh file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


@contextlib.contextmanager
def _new_three_image_hdus():
    """Three image HDUs with distinct shapes/dtypes and recognisable data.
    Returns (fname, [arr0, arr1, arr2])."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "multi.fits")
        arr0 = np.arange(5 * 7, dtype="i4").reshape(5, 7)
        arr1 = (np.arange(3 * 11, dtype="f8") * 0.5).reshape(3, 11)
        arr2 = np.arange(4 * 4 * 4, dtype="i2").reshape(4, 4, 4)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=[5, 7])
            fits[0].write(arr0)
            fits.create_image_hdu(dtype="f8", dims=[3, 11], extname="IMG1")
            fits[1].write(arr1)
            fits.create_image_hdu(dtype="i2", dims=[4, 4, 4], extname="IMG2")
            fits[2].write(arr2)
        yield fname, [arr0, arr1, arr2]


def _slack_capacity(header):
    """Return slots_free = number of cards that can still be added before
    the next write spills past the currently reserved-block boundary."""
    initial = len(header.cards)
    block_count = (initial + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK
    return block_count * CARDS_PER_BLOCK - initial


# ---------------------------------------------------------------------------
# basic grow round-trips
# ---------------------------------------------------------------------------


def test_grow_one_card_past_boundary_same_handle_and_reopen():
    """The simplest case: fill the slack, then push one card past — same
    handle sees the new key, and a fresh reopen confirms persistence."""
    with _new_single_hdu() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            old_blocks = (len(h.cards) + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK  # noqa
            h["TRIGGER"] = "grow"
            new_blocks = (len(h.cards) + CARDS_PER_BLOCK - 1) // CARDS_PER_BLOCK  # noqa
            assert new_blocks > old_blocks
            assert h["TRIGGER"] == "grow"
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["TRIGGER"] == "grow"
            for i in range(slots_free):
                assert fits[0].header[f"PAD{i:04d}"] == i


def test_grow_by_many_cards_in_one_setitem():
    """A single CONTINUE chain can span multiple new blocks."""
    with _new_single_hdu() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            # ~67 chars per CONTINUE card; this is comfortably 4+ blocks worth.
            big = "Z" * (67 * CARDS_PER_BLOCK * 4)
            h["HUGE"] = big
            assert h["HUGE"] == big
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["HUGE"] == big


# ---------------------------------------------------------------------------
# multi-HDU integrity: subsequent HDU data survives the shift
# ---------------------------------------------------------------------------


def test_grow_primary_preserves_subsequent_hdu_data():
    """Grow HDU 0's header.  HDUs 1 and 2 (which got shifted forward in
    the file) must still read back as the same bytes."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            h["TRIGGER"] = "grow"

            # Same-handle reads through HDU references taken AFTER the grow.
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["TRIGGER"] == "grow"
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


def test_grow_middle_hdu_preserves_neighbours():
    """Grow HDU 1's header.  HDU 0 stays put (it's before the insertion
    point); HDU 2 shifts forward.  Both must still read correctly."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h1 = fits[1].header
            slots_free = _slack_capacity(h1)
            for i in range(slots_free):
                h1[f"PAD{i:04d}"] = i
            h1["TRIGGER"] = "grow"

            np.testing.assert_array_equal(fits[0].read(), arrays[0])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), arrays[0])
            assert fits[1].header["TRIGGER"] == "grow"
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


def test_grow_last_hdu_no_tail_to_shift():
    """Grow the LAST HDU's header.  There's nothing past it, so the
    shift loop has zero iterations — only the file-extend path runs."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h2 = fits[2].header
            slots_free = _slack_capacity(h2)
            for i in range(slots_free):
                h2[f"PAD{i:04d}"] = i
            h2["TRIGGER"] = "grow"
            assert h2["TRIGGER"] == "grow"
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            assert fits[2].header["TRIGGER"] == "grow"
            np.testing.assert_array_equal(fits[0].read(), arrays[0])
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# previously-issued handles remain valid after a grow
# ---------------------------------------------------------------------------


def test_old_hdu_handle_sees_post_grow_offsets():
    """A HDU reference issued BEFORE the grow must transparently read
    the right data afterward — the Arc<HduOffsets> shared with the
    layout is updated in place."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            # Capture HDU 2 reference before the grow on HDU 0.
            hdu2_before = fits[2]
            h0 = fits[0].header
            slots_free = _slack_capacity(h0)
            for i in range(slots_free):
                h0[f"PAD{i:04d}"] = i
            h0["TRIGGER"] = "grow"

            # The stale-looking reference still works — offsets updated.
            np.testing.assert_array_equal(hdu2_before.read(), arrays[2])


def test_old_fitsheader_handle_sees_post_grow_offsets():
    """Same as above but for a FITSHeader handle taken before the grow."""
    with _new_three_image_hdus() as (fname, _arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h2_before = fits[2].header
            h2_before["MARKER"] = "before"

            h0 = fits[0].header
            slots_free = _slack_capacity(h0)
            for i in range(slots_free):
                h0[f"PAD{i:04d}"] = i
            h0["TRIGGER"] = "grow"

            # h2_before's offsets have been bumped by the grow; it can
            # still read its own values and accept new writes.
            assert h2_before["MARKER"] == "before"
            h2_before["AFTER"] = "ok"
            assert h2_before["AFTER"] == "ok"

        with rustfits.FITS(fname, "r") as fits:
            assert fits[2].header["MARKER"] == "before"
            assert fits[2].header["AFTER"] == "ok"


# ---------------------------------------------------------------------------
# grow inside FITSHeaderEdit
# ---------------------------------------------------------------------------


def test_grow_inside_edit_commit():
    """The batched-edit commit path also routes through the grow code."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            with h.edit() as e:
                for i in range(slots_free + 5):
                    e[f"PAD{i:04d}"] = i
            for i in range(slots_free + 5):
                assert h[f"PAD{i:04d}"] == i
            # Subsequent HDUs still intact.
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            for i in range(slots_free + 5):
                assert fits[0].header[f"PAD{i:04d}"] == i
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# grow triggered by commentary paths and update()
# ---------------------------------------------------------------------------


def test_grow_triggered_by_add_comment():
    """COMMENT cards are also routed through rewrite_header_to_disk and
    can therefore trigger the grow path."""
    with _new_single_hdu() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            # add_comment splits at 72 chars per card; one card is enough
            # to push past the boundary.
            h.add_comment("grow via comment")
            assert h["COMMENT"][-1] == "grow via comment"
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["COMMENT"][-1] == "grow via comment"


def test_grow_triggered_by_update_from_dict():
    """update() with many new keys can overflow the slack and grow."""
    with _new_single_hdu() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            payload = {f"K{i:04d}": i for i in range(slots_free + 10)}
            h.update(payload)
            for k, v in payload.items():
                assert h[k] == v
        with rustfits.FITS(fname, "r") as fits:
            for k, v in payload.items():
                assert fits[0].header[k] == v


# ---------------------------------------------------------------------------
# sequential grows
# ---------------------------------------------------------------------------


def test_two_sequential_grows_compose():
    """Grow twice in a row — both grows must successfully shift the
    file tail and update layout offsets."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            slots_free = _slack_capacity(h)
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            h["GROW1"] = "first"
            # Now the second grow has to operate from the new offsets.
            slots_free_2 = _slack_capacity(h)
            for i in range(slots_free_2):
                h[f"QAD{i:04d}"] = i
            h["GROW2"] = "second"
            assert h["GROW1"] == "first"
            assert h["GROW2"] == "second"
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])

        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["GROW1"] == "first"
            assert fits[0].header["GROW2"] == "second"
            np.testing.assert_array_equal(fits[1].read(), arrays[1])
            np.testing.assert_array_equal(fits[2].read(), arrays[2])


# ---------------------------------------------------------------------------
# slice reads through post-grow HDUs
# ---------------------------------------------------------------------------


def test_image_slice_after_grow_returns_correct_bytes():
    """Slice-read a post-grow HDU and compare against the expected
    sub-array to make sure offsets + strides line up."""
    with _new_three_image_hdus() as (fname, arrays):
        with rustfits.FITS(fname, "r+") as fits:
            h0 = fits[0].header
            slots_free = _slack_capacity(h0)
            for i in range(slots_free):
                h0[f"PAD{i:04d}"] = i
            h0["TRIGGER"] = "grow"

            np.testing.assert_array_equal(
                fits[2][1:3, :, 2], arrays[2][1:3, :, 2]
            )

        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(
                fits[2][1:3, :, 2], arrays[2][1:3, :, 2]
            )


if __name__ == "__main__":
    test_grow_one_card_past_boundary_same_handle_and_reopen()
    test_grow_by_many_cards_in_one_setitem()
    test_grow_primary_preserves_subsequent_hdu_data()
    test_grow_middle_hdu_preserves_neighbours()
    test_grow_last_hdu_no_tail_to_shift()
    test_old_hdu_handle_sees_post_grow_offsets()
    test_old_fitsheader_handle_sees_post_grow_offsets()
    test_grow_inside_edit_commit()
    test_grow_triggered_by_add_comment()
    test_grow_triggered_by_update_from_dict()
    test_two_sequential_grows_compose()
    test_image_slice_after_grow_returns_correct_bytes()
    print("smoke tests passed")
