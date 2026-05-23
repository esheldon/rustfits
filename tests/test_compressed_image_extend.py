"""
Compressed image extend (append-along-axis-0).

CompressedImageHDU.extend(data) appends rows along the slow axis
of an existing tile-compressed image.  Existing tiles outside the
last tile row are preserved untouched on disk; the partial last
tile row (if any) gets re-encoded to absorb the first portion of
new data; truly-new tile rows are encoded from new data alone.

Tests cover:
    - 1-D extends (the healsparse use case): tile-aligned and
      partial-last-tile.
    - 2-D extends across the algorithm matrix.
    - 3-D extends.
    - Dtype matrix: u1/i2/i4 + unsigned-int trick (i1/u2/u4).
    - Algorithm matrix: Gzip1/Gzip2/Rice1/Hcompress1.  (Plio1 is
      excluded — it'd reject most non-mask data; covered separately
      in the unit tests file.)
    - Multiple sequential extends on the same HDU.
    - Cross-check with astropy: rustfits writes + extends, astropy
      reads back bit-exact.
    - Non-last HDU growth (later HDUs shift forward correctly).
    - Unquantized-float (quantize=None) extend.
    - Rejections: shape mismatch on non-extend axes, empty input,
      quantized-float extend, wrong axis count.  (Note: no start=
      kwarg, unlike ImageHDU.extend — in-place writes are
      __setitem__'s job; extend only appends.)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")


def _seq(shape, dtype, start=0):
    """
    Deterministic integer test array of `shape` and `dtype`,
    clamped into the dtype's range to avoid overflow surprises.
    """
    np_dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    if np_dtype.kind in ("u", "i"):
        info = np.iinfo(np_dtype)
        # Cycle through a safe window: 0..min(max, 1_000_000)
        maxv = min(info.max, 1_000_000)
        return (
            ((np.arange(n) + start) % (maxv + 1))
            .astype(np_dtype)
            .reshape(shape)
        )
    else:
        # float: scale into a moderate range
        return (np.arange(n) + start).astype(np_dtype).reshape(
            shape
        ) * 0.5 - 3.0


# ---------------------- 1-D (healsparse) ---------------------------


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_1d_tile_aligned(AlgoCls):
    """
    1-D image where initial_size and tile_shape align cleanly
    (no partial last tile).  After extend, both pieces are
    preserved bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((64,), "i4")
        more = _seq((48,), "i4", start=64)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (64,), compress=AlgoCls(tile_shape=(16,)))
            f[1].write(initial)
            f[1].extend(more)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        expected = np.concatenate([initial, more])
        np.testing.assert_array_equal(same, expected)
        np.testing.assert_array_equal(reopen, expected)


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
@pytest.mark.parametrize(
    "initial_n,extend_n,tile_n",
    [
        (50, 48, 16),  # partial last tile (2 rows), extend fills it + adds
        (50, 4, 16),  # extend stays within the partial tile (50 → 54 < 64)
        (50, 14, 16),  # extend fills exactly to next tile boundary (50→64)
        (1, 31, 16),  # tiny initial + extend across tiles
        (16, 1, 16),  # tile-aligned + extend creates partial
    ],
)
def test_1d_partial_last_tile(AlgoCls, initial_n, extend_n, tile_n):
    """
    1-D image where the initial size leaves a partial last tile,
    or where the extend creates a new partial last tile.  The
    boundary tile must be re-encoded correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((initial_n,), "i4")
        more = _seq((extend_n,), "i4", start=initial_n)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (initial_n,), compress=AlgoCls(tile_shape=(tile_n,))
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more])
        np.testing.assert_array_equal(rt, expected)
        assert rt.shape == (initial_n + extend_n,)


# ---------------------- 2-D ---------------------------------------


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_2d_tile_aligned(AlgoCls):
    """
    2-D image whose initial rows align with the tile rows.  No
    boundary re-encoding needed; just truly-new tile rows added.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((32, 50), "i2")
        more = _seq((16, 50), "i2", start=32 * 50)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (32, 50),
                compress=AlgoCls(tile_shape=(16, 50)),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more], axis=0)
        np.testing.assert_array_equal(rt, expected)


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_2d_partial_last_tile(AlgoCls):
    """
    2-D image whose initial rows leave a partial last tile row.
    The partial tile(s) get re-encoded to absorb new data.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        # 50 rows / tile=16 → 4 tile rows, last has 2 rows
        initial = _seq((50, 30), "i2")
        more = _seq((30, 30), "i2", start=50 * 30)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (50, 30),
                compress=AlgoCls(tile_shape=(16, 30)),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more], axis=0)
        np.testing.assert_array_equal(rt, expected)


def test_hcompress_2d_extend():
    """
    HCOMPRESS_1 + 2-D extend.  HCOMPRESS requires tiles >= 4 on
    each dim, so use 16x16 tiles on a 16x16 image and extend by 16
    more rows (tile-aligned).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((16, 16), "i2")
        more = _seq((16, 16), "i2", start=256)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (16, 16),
                compress=rustfits.Hcompress1(tile_shape=(16, 16)),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more], axis=0)
        np.testing.assert_array_equal(rt, expected)


# ---------------------- 3-D ---------------------------------------


def test_3d_extend():
    """
    3-D image extend along numpy axis 0 (FITS-slowest).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((4, 5, 6), "i4")
        more = _seq((6, 5, 6), "i4", start=4 * 5 * 6)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (4, 5, 6),
                compress=rustfits.Gzip1(tile_shape=(2, 5, 6)),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more], axis=0)
        np.testing.assert_array_equal(rt, expected)


# ---------------------- dtype matrix (including unsigned trick) ----


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i1", "u2", "u4"])
def test_dtype_matrix_with_partial_tile(dtype):
    """
    Dtype matrix including unsigned-int trick (i1/u2/u4).  Partial-
    last-tile case to exercise the most complex code path on each
    dtype.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((25,), dtype)
        more = _seq((20,), dtype, start=25)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype, (25,), compress=rustfits.Gzip1(tile_shape=(8,))
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more])
        np.testing.assert_array_equal(rt, expected)
        assert rt.dtype == np.dtype(dtype)


# ---------------------- unquantized float --------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("AlgoCls", [rustfits.Gzip1, rustfits.Gzip2])
def test_unquantized_float_extend(AlgoCls, dtype):
    """
    Unquantized float HDU (quantize=None) extend.  Bit-exact
    round-trip required.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        initial = rng.standard_normal(40).astype(dtype)
        more = rng.standard_normal(20).astype(dtype)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                (40,),
                compress=AlgoCls(tile_shape=(16,)),
                quantize=None,
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        expected = np.concatenate([initial, more])
        np.testing.assert_array_equal(rt, expected)


# ---------------------- multiple sequential extends ----------------


def test_many_sequential_extends_with_partial_tiles():
    """
    Five small extends in a row, each potentially crossing tile
    boundaries.  Sanity-check that the boundary handling stays
    correct across multiple invocations.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        pieces = [_seq((20,), "i4", start=i * 20) for i in range(5)]
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (20,), compress=rustfits.Gzip1(tile_shape=(8,))
            )
            f[1].write(pieces[0])
            for p in pieces[1:]:
                f[1].extend(p)
            rt = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            rt_re = f[1].read()
        expected = np.concatenate(pieces)
        np.testing.assert_array_equal(rt, expected)
        np.testing.assert_array_equal(rt_re, expected)


# ---------------------- astropy cross-read ------------------------


@pytest.mark.parametrize(
    "AlgoCls,zcmptype",
    [
        (rustfits.Gzip1, "GZIP_1"),
        (rustfits.Gzip2, "GZIP_2"),
        (rustfits.Rice1, "RICE_1"),
    ],
)
def test_astropy_reads_extended_file(AlgoCls, zcmptype):
    """
    rustfits-extended file must read back bit-exact via astropy.
    Exercises the partial-last-tile case (50 % 16 != 0).
    ZCMPTYPE is checked via the raw BINTABLE header (astropy's
    high-level CompImageHDU.header hides Z-prefixed cards).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        initial = _seq((50, 30), "i2")
        more = _seq((30, 30), "i2", start=50 * 30)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (50, 30),
                compress=AlgoCls(tile_shape=(16, 30)),
            )
            f[1].write(initial)
            f[1].extend(more)
        expected = np.concatenate([initial, more], axis=0)
        # Data round-trip via the high-level interface
        with astropy_fits.open(fn) as h:
            np.testing.assert_array_equal(h[1].data, expected)
        # Raw BINTABLE header carries the Z-prefixed cards
        with astropy_fits.open(fn, disable_image_compression=True) as h:
            assert h[1].header["ZCMPTYPE"] == zcmptype


# ---------------------- non-last HDU growth -----------------------


def test_extend_shifts_later_hdus():
    """
    Compressed HDU followed by a second HDU.  Extending the first
    grows the file; the second HDU's offsets must shift forward
    and both must read back correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_initial = _seq((50,), "i4")
        comp_extend = _seq((40,), "i4", start=50)
        later = _seq((10, 10), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (50,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                extname="COMP",
            )
            f.create_image_hdu("i2", (10, 10), extname="LATER")
            f[1].write(comp_initial)
            f[2].write(later)
            # Now extend COMP — must shift LATER forward
            f[1].extend(comp_extend)
            # Same-handle reads of both
            np.testing.assert_array_equal(
                f[1].read(), np.concatenate([comp_initial, comp_extend])
            )
            np.testing.assert_array_equal(f[2].read(), later)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(
                f[1].read(), np.concatenate([comp_initial, comp_extend])
            )
            np.testing.assert_array_equal(f[2].read(), later)


# ---------------------- rejections --------------------------------


def test_start_kwarg_not_accepted():
    """
    Unlike ImageHDU.extend(data, start=None), compressed extend
    has no start= kwarg.  In-place writes are __setitem__'s job.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (16,), compress=rustfits.Gzip1(tile_shape=(16,))
            )
            f[1].write(_seq((16,), "i4"))
            with pytest.raises(TypeError):
                f[1].extend(_seq((8,), "i4"), start=[0])


def test_shape_mismatch_rejected():
    """
    Data shape on axes 1.. must match the HDU's shape; only axis 0
    can grow.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (10, 20),
                compress=rustfits.Gzip1(tile_shape=(5, 20)),
            )
            f[1].write(_seq((10, 20), "i2"))
            with pytest.raises(ValueError, match="axis"):
                f[1].extend(_seq((5, 25), "i2"))


def test_empty_data_rejected():
    """
    Extending with shape[0] == 0 is rejected (no-op would be
    surprising; explicit error makes intent clear).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (16,), compress=rustfits.Gzip1(tile_shape=(16,))
            )
            f[1].write(_seq((16,), "i4"))
            with pytest.raises(ValueError, match="data.shape"):
                f[1].extend(np.empty(0, dtype="i4"))


def test_quantized_float_extend_rejected():
    """
    Extend on quantized-float HDUs (4-column schema) is deferred.
    Clear error pointing the user at quantize=None.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        initial = rng.standard_normal(32).astype("f4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (32,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                quantize=rustfits.Quantize(method="dither1"),
            )
            f[1].write(initial)
            with pytest.raises(NotImplementedError, match="quantized"):
                f[1].extend(np.zeros(8, dtype="f4"))


def test_axis_count_mismatch_rejected():
    """
    Input ndarray must have the same number of axes as the HDU.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (16,), compress=rustfits.Gzip1(tile_shape=(16,))
            )
            f[1].write(_seq((16,), "i4"))
            with pytest.raises(ValueError, match="axes"):
                f[1].extend(_seq((4, 4), "i4"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
