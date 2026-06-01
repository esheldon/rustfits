"""
Tests for CompressedImageHDU.repack() — rebuild the tile-data heap
with only live cells, dropping orphans left behind by
extend() / __setitem__() / boundary-tile re-encoding.

When the heap shrinks past a 2880-byte block boundary the on-disk
file shrinks too (last HDU: set_len; non-last: tail shifts backward).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _filesize(fname):
    return os.path.getsize(fname)


def _make_compressed(tmpdir, dtype, shape, compress, fill=None):
    """Create a one-HDU compressed image, optionally pre-write `fill`."""
    fname = os.path.join(tmpdir, "c.fits")
    with rustfits.FITS(fname, "w+") as f:
        f.create_image_hdu(dtype=dtype, dims=shape, compress=compress)
        if fill is not None:
            f[1].write(fill)
    return fname


# ---------------------------------------------------------------------------
# Repack drops orphans
# ---------------------------------------------------------------------------


def test_repack_after_setitem_drops_orphans():
    """
    Modify the same tile repeatedly via __setitem__ so the heap
    accumulates orphans; repack should bring PCOUNT back to the
    sum of live-tile bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        shape = (40, 40)
        data = rng.integers(0, 1000, size=shape, dtype="i4")
        fname = _make_compressed(
            tmp,
            "i4",
            shape,
            compress=rustfits.Gzip1(tile_shape=(10, 10)),
            fill=data,
        )
        with rustfits.FITS(fname, "r") as fits:
            base_pcount = int(fits[1].header["PCOUNT"])

        # Hammer one tile repeatedly to grow orphans.
        with rustfits.FITS(fname, "r+") as fits:
            for k in range(10):
                fits[1][0:10, 0:10] = np.full((10, 10), k * 100, dtype="i4")

        with rustfits.FITS(fname, "r") as fits:
            mid_pcount = int(fits[1].header["PCOUNT"])
        assert mid_pcount > base_pcount

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            final_pcount = int(fits[1].header["PCOUNT"])
            assert final_pcount < mid_pcount
            # Final tile contents reflect the last __setitem__.
            expected = data.copy()
            expected[0:10, 0:10] = 9 * 100
            np.testing.assert_array_equal(fits[1].read(), expected)


def test_repack_shrinks_last_hdu_file():
    """
    Enough orphans on a last-HDU compressed image should make the
    file shrink after repack.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        shape = (60, 60)
        data = rng.integers(0, 1000, size=shape, dtype="i4")
        fname = _make_compressed(
            tmp,
            "i4",
            shape,
            compress=rustfits.Gzip1(tile_shape=(10, 10)),
            fill=data,
        )
        with rustfits.FITS(fname, "r+") as fits:
            # Repeatedly rewrite the whole image (each pass orphans
            # every tile and re-encodes them at the heap end).
            for k in range(8):
                fits[1][:] = rng.integers(0, 1000, size=shape, dtype="i4")
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        post_size = _filesize(fname)
        assert post_size < pre_size


def test_repack_no_op_when_already_compact():
    """Fresh-after-write compressed image: repack is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(2)
        shape = (30, 30)
        data = rng.integers(0, 100, size=shape, dtype="i2")
        fname = _make_compressed(
            tmp,
            "i2",
            shape,
            compress=rustfits.Rice1(tile_shape=(10, 10)),
            fill=data,
        )
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r") as fits:
            pre_pcount = int(fits[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
        with rustfits.FITS(fname) as fits:
            assert int(fits[1].header["PCOUNT"]) == pre_pcount
            np.testing.assert_array_equal(fits[1].read(), data)
        assert _filesize(fname) == pre_size


# ---------------------------------------------------------------------------
# Non-last HDU shrink
# ---------------------------------------------------------------------------


def test_repack_non_last_hdu_shifts_tail_backward():
    """
    Repack on a non-last compressed HDU shifts the following HDUs
    backward; previously-issued handles to later HDUs still work
    (shared Arc<HduOffsets>).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "two.fits")
        rng = np.random.default_rng(3)
        shape = (40, 40)
        data = rng.integers(0, 1000, size=shape, dtype="i4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "i4",
                shape,
                compress=rustfits.Gzip1(tile_shape=(10, 10)),
            )
            f[1].write(data)
            f.create_image_hdu("i4", (5,), extname="AFTER")
            f[2].write(np.arange(5, dtype="i4") + 1000)
        # Inflate orphans.
        with rustfits.FITS(fname, "r+") as fits:
            for k in range(8):
                fits[1][:] = rng.integers(0, 1000, size=shape, dtype="i4")
        pre_size = _filesize(fname)
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
            # Same-handle: second HDU still reads correctly.
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
        assert _filesize(fname) < pre_size
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(5, dtype="i4") + 1000,
            )
            # First HDU still readable.
            _ = fits[1].read()


# ---------------------------------------------------------------------------
# Algorithm + dtype matrix (correctness preserved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,algo,tile_shape",
    [
        ("u1", rustfits.Gzip1(tile_shape=(10, 10)), (10, 10)),
        ("i2", rustfits.Gzip2(tile_shape=(10, 10)), (10, 10)),
        ("i4", rustfits.Rice1(tile_shape=(10, 10)), (10, 10)),
        (
            "i2",
            rustfits.Hcompress1(tile_shape=(10, 10), scale=4),
            (10, 10),
        ),
    ],
)
def test_repack_preserves_content_across_algorithms(dtype, algo, tile_shape):
    """
    Whatever the compression algorithm, repack-after-setitem must
    leave the decoded image bit-exact (apart from the lossy hcompress
    case which loses precision per-write, not per-repack).
    """
    with tempfile.TemporaryDirectory() as tmp:
        shape = (30, 30)
        rng = np.random.default_rng(4)
        data = rng.integers(0, 100, size=shape, dtype=dtype)
        fname = _make_compressed(tmp, dtype, shape, algo, fill=data)
        # Modify a few tiles to leave orphans.
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][0:10, 0:10] = np.full((10, 10), 5, dtype=dtype)
            fits[1][10:20, 10:20] = np.full((10, 10), 7, dtype=dtype)
            ref_after = fits[1].read()
            fits[1].repack()
            np.testing.assert_array_equal(fits[1].read(), ref_after)
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(fits[1].read(), ref_after)


# ---------------------------------------------------------------------------
# Float-quantized: repack on quantized floats
# ---------------------------------------------------------------------------


def test_repack_quantized_float_preserves_content():
    """
    Repack on a quantized-float HDU after several setitems must
    preserve every pixel value bit-exact (quantization losses happen
    on the original encode + per-setitem encode, NOT on repack since
    repack just relocates raw bytes).
    """
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        shape = (40, 40)
        data = rng.standard_normal(shape).astype("f4")
        fname = os.path.join(tmp, "q.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "f4",
                shape,
                compress=rustfits.Rice1(tile_shape=(10, 10)),
                quantize=rustfits.Quantize(level=4.0, method="dither1"),
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r+") as fits:
            patch = rng.standard_normal((10, 10)).astype("f4")
            fits[1][0:10, 0:10] = patch
            ref = fits[1].read()
            fits[1].repack()
            np.testing.assert_array_equal(fits[1].read(), ref)
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(fits[1].read(), ref)


def test_repack_unquantized_float_preserves_content():
    """Unquantized float HDU (Gzip + quantize=None implicit)."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(6)
        shape = (40, 40)
        data = rng.standard_normal(shape).astype("f4")
        fname = _make_compressed(
            tmp,
            "f4",
            shape,
            compress=rustfits.Gzip1(tile_shape=(10, 10)),
            fill=data,
        )
        with rustfits.FITS(fname, "r+") as fits:
            patch = rng.standard_normal((10, 10)).astype("f4")
            fits[1][0:10, 0:10] = patch
            ref = fits[1].read()
            fits[1].repack()
            np.testing.assert_array_equal(fits[1].read(), ref)
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(fits[1].read(), ref)


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def test_repack_slow_path_non_last_hdu_no_corruption():
    """
    Engineer a fast-path-unsafe orphan layout and verify repack
    takes the slow (stage-past-EOF) path on a non-last HDU without
    corrupting the trailing HDU.

    Pattern: write a 9-tile image of all zeros (each tile
    compresses to ~24 B with Gzip1), then ``__setitem__`` a
    MIDDLE tile with random i4 data (compresses to ~400 B —
    much larger than the original).  The walked-row-by-row
    move plan sorted by old_off then contains:

        - tiles 0..4 with new_off == old_off (all unchanged)
        - tile 5 (modified) at the END with old_off=heap_end_orig
          but new_off=cum_through_4
        - tiles 6..8 sorted earlier with new_off bumped by
          L_modified - L_orig (a large positive delta)

    For tile 6, ``new_off + length = L_orig_through_4 + L_modified
    + L_6`` while ``next_read_start = old_off_of_tile_7 ≈
    L_orig_through_6``.  Since ``L_modified > L_orig``, the
    check ``new_off + length > next_read_start`` fires → slow
    path runs.

    Before the 2026-05-31 fix, the slow path used ``set_len``
    directly to allocate staging room, silently overwriting the
    trailing HDU's bytes.  This test would have failed with a
    "non-printable byte in header" error on the post-reopen
    read of the trailing HDU.  Now both that bug and the
    related post-grow shrink bug are fixed; the trailing HDU
    must survive bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "two.fits")
        # Size the image so the post-setitem heap fills the
        # current block-padded extent: (100, 100) i4 with
        # (10, 10) tiles → 100 tiles × ~24 B all-zeros
        # compressed + the modified tile at heap end → total
        # heap ~2.8 KB; main descriptor table ~1.6 KB; total
        # data ~4.4 KB padded to 5.76 KB.  Slow-path staging
        # of another ~2.8 KB then spills 1.4 KB past the
        # current padded extent into the trailing HDU's
        # territory — which is exactly what the (now-fixed)
        # bare-set_len bug would have silently overwritten.
        shape = (100, 100)
        zeros = np.zeros(shape, dtype="i4")
        rng = np.random.default_rng(42)
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "i4",
                shape,
                compress=rustfits.Gzip1(tile_shape=(10, 10)),
            )
            f[1].write(zeros)
            f.create_image_hdu("i4", (7,), extname="AFTER")
            f[2].write(np.arange(7, dtype="i4") + 5000)
        # Setitem a MIDDLE tile with random data — large
        # compressed size compared to the all-zeros originals.
        big_patch = rng.integers(
            -(2**30), 2**30 - 1, size=(10, 10), dtype="i4"
        )
        with rustfits.FITS(fname, "r+") as fits:
            fits[1][50:60, 50:60] = big_patch
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].repack()
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(7, dtype="i4") + 5000,
            )
        with rustfits.FITS(fname) as fits:
            # Trailing HDU survived the slow-path repack bit-exact.
            np.testing.assert_array_equal(
                fits[2].read(),
                np.arange(7, dtype="i4") + 5000,
            )
            # First HDU still reads correctly — every tile that
            # wasn't modified should be all zeros, and the
            # modified tile contains the random patch.
            got = fits[1].read()
            expected = zeros.copy()
            expected[50:60, 50:60] = big_patch
            np.testing.assert_array_equal(got, expected)


def test_repack_clears_tile_cache():
    """
    After repack, decoded tiles whose descriptors moved must come
    from the new heap (not from any cached entry referencing the
    old layout).  Easiest check: cache_used should be 0 right after
    repack (before any further read).
    """
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(7)
        shape = (30, 30)
        data = rng.integers(0, 100, size=shape, dtype="i2")
        fname = _make_compressed(
            tmp,
            "i2",
            shape,
            compress=rustfits.Gzip1(tile_shape=(10, 10)),
            fill=data,
        )
        with rustfits.FITS(fname, "r+") as fits:
            _ = fits[1].read()  # warm the cache
            assert fits[1].tile_cache_used > 0
            fits[1][0:10, 0:10] = np.full((10, 10), 9, dtype="i2")
            _ = fits[1].read()
            fits[1].repack()
            assert fits[1].tile_cache_used == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
