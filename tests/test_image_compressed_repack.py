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
