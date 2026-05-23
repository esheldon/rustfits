"""
Compressed image __setitem__: in-place pixel modification.

CompressedImageHDU.__setitem__ accepts the same slice surface as
__getitem__ (slice / int / ellipsis per axis, stepped slices,
mixed combinations) and a RHS that's either a scalar (broadcast
across the selection) or an ndarray with shape matching the
selection.

For each tile that overlaps the selection, the tile is decoded,
modified in numpy with the user's value, and re-encoded.  New
bytes are appended to the heap; the old bytes for modified tiles
become orphans (left in place; descriptors no longer reference
them).  PCOUNT grows by the total appended; the file may grow
(and later HDUs shift forward) if the heap exceeds the current
padded extent.

Tests cover:
    - Single-pixel writes (int per axis).
    - Row / column writes (mixed int + slice).
    - Contiguous slice writes across multiple tiles.
    - Stepped slices (numpy ::N convention).
    - Scalar broadcast vs array RHS.
    - 1-D / 2-D / 3-D images.
    - Dtype matrix including unsigned-int trick (i1/u2/u4).
    - Algorithm matrix (Gzip1/Gzip2/Rice1/Hcompress1).
    - Multiple sequential modifications.
    - astropy cross-read agreement.
    - Non-last HDU growth (heap appends → shift later HDUs).
    - Empty slice no-op.
    - Rejections: quantized-float, shape mismatch, fancy indexing.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")


def _seq(shape, dtype, start=0):
    """
    Deterministic integer/float test array of `shape` and `dtype`.
    """
    np_dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    if np_dtype.kind in ("u", "i"):
        info = np.iinfo(np_dtype)
        maxv = min(info.max, 1_000_000)
        return (
            ((np.arange(n) + start) % (maxv + 1))
            .astype(np_dtype)
            .reshape(shape)
        )
    else:
        return (np.arange(n) + start).astype(np_dtype).reshape(
            shape
        ) * 0.5 - 3.0


# ---------------------- single-pixel writes ------------------------


def test_single_pixel_write_2d():
    """hdu[i, j] = scalar writes one pixel; rest of HDU unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((8, 8), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(data)
            f[1][3, 5] = 999
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        expected = data.copy()
        expected[3, 5] = 999
        np.testing.assert_array_equal(same, expected)
        np.testing.assert_array_equal(reopen, expected)


def test_single_pixel_write_1d():
    """1-D pixel write — healsparse-style sparse update."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((50,), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (50,), compress=rustfits.Gzip1(tile_shape=(16,))
            )
            f[1].write(data)
            f[1][7] = -7
            f[1][33] = -33
            f[1][49] = -49
            rt = f[1].read()
        expected = data.copy()
        expected[7] = -7
        expected[33] = -33
        expected[49] = -49
        np.testing.assert_array_equal(rt, expected)


# ---------------------- mixed int + slice --------------------------


def test_row_write():
    """
    hdu[i, :] = row — single tile row or multiple tiles depending
    on layout.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((8, 8), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(data)
            f[1][2, :] = np.arange(8, dtype="i4") + 100
            rt = f[1].read()
        expected = data.copy()
        expected[2, :] = np.arange(8) + 100
        np.testing.assert_array_equal(rt, expected)


def test_column_write():
    """hdu[:, j] = col — column-axis write."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((8, 8), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(data)
            f[1][:, 5] = np.arange(8, dtype="i4") - 100
            rt = f[1].read()
        expected = data.copy()
        expected[:, 5] = np.arange(8) - 100
        np.testing.assert_array_equal(rt, expected)


# ---------------------- multi-tile slice ---------------------------


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_multi_tile_slice_write(AlgoCls):
    """
    Slice that spans multiple tiles must modify each tile
    correctly without disturbing the others.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((16, 16), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (16, 16), compress=AlgoCls(tile_shape=(8, 8))
            )
            f[1].write(data)
            new = _seq((6, 10), "i4", start=10_000)
            f[1][5:11, 3:13] = new
            rt = f[1].read()
        expected = data.copy()
        expected[5:11, 3:13] = new
        np.testing.assert_array_equal(rt, expected)


def test_hcompress_setitem():
    """Hcompress1 + slice write."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((16, 16), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (16, 16),
                compress=rustfits.Hcompress1(tile_shape=(8, 8)),
            )
            f[1].write(data)
            f[1][4:12, 4:12] = 0
            rt = f[1].read()
        expected = data.copy()
        expected[4:12, 4:12] = 0
        np.testing.assert_array_equal(rt, expected)


# ---------------------- scalar broadcast ---------------------------


def test_scalar_broadcast_int():
    """Integer scalar broadcast across a slice selection."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((10, 10), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (10, 10), compress=rustfits.Gzip1(tile_shape=(5, 5))
            )
            f[1].write(data)
            f[1][2:8, 2:8] = -1
            rt = f[1].read()
        expected = data.copy()
        expected[2:8, 2:8] = -1
        np.testing.assert_array_equal(rt, expected)


def test_scalar_broadcast_whole_image():
    """hdu[:] = 0 zeros the entire image."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((8, 8), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(data)
            f[1][:] = 0
            rt = f[1].read()
        np.testing.assert_array_equal(rt, np.zeros((8, 8), dtype="i4"))


# ---------------------- stepped slice ------------------------------


def test_stepped_slice_1d():
    """1-D stepped slice: hdu[::2] = ..."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((32,), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (32,), compress=rustfits.Gzip1(tile_shape=(8,))
            )
            f[1].write(data)
            f[1][::2] = 0
            rt = f[1].read()
        expected = data.copy()
        expected[::2] = 0
        np.testing.assert_array_equal(rt, expected)


def test_stepped_slice_2d_one_axis():
    """2-D stepped slice on one axis."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((10, 10), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (10, 10), compress=rustfits.Gzip1(tile_shape=(5, 5))
            )
            f[1].write(data)
            f[1][1::3, :] = 0
            rt = f[1].read()
        expected = data.copy()
        expected[1::3, :] = 0
        np.testing.assert_array_equal(rt, expected)


# ---------------------- 3-D ----------------------------------------


def test_3d_setitem():
    """3-D image __setitem__."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((4, 5, 6), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (4, 5, 6),
                compress=rustfits.Gzip1(tile_shape=(2, 5, 6)),
            )
            f[1].write(data)
            f[1][1:3, 1:4, 2:5] = -1
            rt = f[1].read()
        expected = data.copy()
        expected[1:3, 1:4, 2:5] = -1
        np.testing.assert_array_equal(rt, expected)


# ---------------------- dtype matrix -------------------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i1", "u2", "u4"])
def test_dtype_matrix(dtype):
    """Dtype matrix including unsigned-int trick (i1/u2/u4)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((10, 10), dtype)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype, (10, 10), compress=rustfits.Gzip1(tile_shape=(5, 5))
            )
            f[1].write(data)
            new = _seq((4, 4), dtype, start=1234)
            f[1][3:7, 3:7] = new
            rt = f[1].read()
        expected = data.copy()
        expected[3:7, 3:7] = new
        np.testing.assert_array_equal(rt, expected)
        assert rt.dtype == np.dtype(dtype)


# ---------------------- unquantized float --------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_unquantized_float_setitem(dtype):
    """Unquantized float HDU (quantize=None) supports __setitem__."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        data = rng.standard_normal((8, 8)).astype(dtype)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                quantize=None,
            )
            f[1].write(data)
            f[1][2:6, 2:6] = 0.0
            rt = f[1].read()
        expected = data.copy()
        expected[2:6, 2:6] = 0.0
        np.testing.assert_array_equal(rt, expected)


# ---------------------- multiple sequential mods -------------------


def test_many_sequential_setitems():
    """
    Many sequential __setitem__ calls in a row.  Each appends to
    the heap (old bytes orphaned).  All updates must compose
    correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((20,), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (20,), compress=rustfits.Gzip1(tile_shape=(8,))
            )
            f[1].write(data)
            for idx, val in [
                (0, 100),
                (5, 105),
                (10, 110),
                (15, 115),
                (19, 119),
            ]:
                f[1][idx] = val
            rt = f[1].read()
        expected = data.copy()
        for idx, val in [(0, 100), (5, 105), (10, 110), (15, 115), (19, 119)]:
            expected[idx] = val
        np.testing.assert_array_equal(rt, expected)


# ---------------------- astropy cross-read -------------------------


def test_astropy_reads_after_setitem():
    """rustfits-modified file must read back bit-exact via astropy."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((16, 16), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(8, 8)),
            )
            f[1].write(data)
            f[1][4:12, 4:12] = -42
        with astropy_fits.open(fn) as h:
            ap = h[1].data
        expected = data.copy()
        expected[4:12, 4:12] = -42
        np.testing.assert_array_equal(ap, expected)


# ---------------------- non-last HDU growth -----------------------


def test_setitem_shifts_later_hdus():
    """
    __setitem__ may grow the file (modified tiles append to the
    heap).  Later HDUs must shift forward and remain readable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _seq((16, 16), "i4")
        later = _seq((10, 10), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (16, 16),
                compress=rustfits.Gzip1(tile_shape=(8, 8)),
                extname="COMP",
            )
            f.create_image_hdu("i2", (10, 10), extname="LATER")
            f[1].write(comp_data)
            f[2].write(later)
            # Big slice modification — likely grows the heap past
            # the current padded extent.
            f[1][:] = np.full((16, 16), -1, dtype="i4")
            np.testing.assert_array_equal(
                f[1].read(), np.full((16, 16), -1, dtype="i4")
            )
            np.testing.assert_array_equal(f[2].read(), later)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(
                f[1].read(), np.full((16, 16), -1, dtype="i4")
            )
            np.testing.assert_array_equal(f[2].read(), later)


# ---------------------- empty slice no-op --------------------------


def test_empty_slice_is_noop():
    """hdu[5:5] = ... is a no-op (matches numpy)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((8, 8), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(data)
            # Empty slice on axis 0
            f[1][5:5] = -1
            rt = f[1].read()
        np.testing.assert_array_equal(rt, data)


# ---------------------- rejections --------------------------------


# test_quantized_float_setitem_rejected: removed — quantized-float
# __setitem__ is now supported with the careful re-encoding scheme
# (re-uses existing per-tile bscale/bzero/seed to avoid compounding
# loss on unchanged pixels).  See
# tests/test_compressed_image_quant_mutation.py.


def test_shape_mismatch_rejected():
    """Array RHS with wrong shape relative to the selection is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
            # Selection is (4, 4); pass wrong-shape array
            with pytest.raises(ValueError, match="shape"):
                f[1][2:6, 2:6] = np.zeros((3, 3), dtype="i4")


def test_fancy_index_rejected():
    """Lists / arrays as indexers are not supported (same as __getitem__)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
            with pytest.raises(ValueError, match="unsupported index"):
                f[1][[1, 3, 5]] = 0


def test_int_out_of_bounds_rejected():
    """Out-of-bounds single integer index → IndexError."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
            with pytest.raises(IndexError):
                f[1][8, 0] = 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
