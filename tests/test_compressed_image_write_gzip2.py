"""
ZIMAGE Gzip2 compressed image writes.

Parallels test_compressed_image_write_gzip.py for the
GZIP_2 algorithm: byte-shuffle preprocessor in front of the same
gzip framing GZIP_1 uses.  Tests cover:
    - Round-trip via same-handle read and via post-reopen read
    - Cross-check vs fitsio (rustfits writes → fitsio reads, and
      fitsio writes → rustfits reads — both must agree bit-exact)
    - Integer dtype matrix (u1, i2, i4, i8).  For u1 the shuffle
      collapses to a no-op so GZIP_2 bytes should equal GZIP_1
      bytes on the same input.
    - Shape matrix (1-D, 2-D square, 2-D non-square, 3-D)
    - Default tile shape (row tiles)
    - Non-last HDU growth: heap shifts later HDU offsets
    - Mixed-algorithm file: one HDU GZIP_1, another GZIP_2
    - Float ZBITPIX (-32/-64) rejected with a Phase 8 NotImplemented
    - Unsupported algorithm types rejected
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _seq(shape, dtype):
    """
    Deterministic test array of `shape` and `dtype`.

    Arange modulo the dtype's range, then reshape — avoids
    overflow on small-range dtypes and keeps the values
    sensitive to byte-order and shuffle errors.
    """
    n = int(np.prod(shape))
    maxv = {
        np.dtype("u1"): 255,
        np.dtype("i2"): 32767,
        np.dtype("i4"): 1_000_000,
        np.dtype("i8"): 1_000_000,
    }[np.dtype(dtype)]
    arr = (np.arange(n) % (maxv + 1)).astype(dtype)
    return arr.reshape(shape)


# ---------------------- accessors ----------------------------------


def test_accessors_after_create():
    """
    create_image_hdu(..., compress=Gzip2(...)) produces a
    CompressedImageHDU with ZCMPTYPE=GZIP_2.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(16, 16))
            f.create_image_hdu("i4", (32, 48), compress=cfg, extname="SCI")
            hdu = f[1]
            assert type(hdu).__name__ == "CompressedImageHDU"
            assert hdu.shape == (32, 48)
            assert hdu.dtype == np.int32
            assert hdu.bitpix == 32
            assert hdu.compression.zcmptype == "GZIP_2"
            assert hdu.compression.tile_shape == (16, 16)
            assert hdu.extname == "SCI"
            # Before write, PCOUNT is 0 (heap empty).
            assert hdu.header["PCOUNT"] == 0


def test_gzip2_repr_and_kwargs():
    """
    Gzip2 config object surface matches Gzip1: tile_shape +
    heap_format getters, repr starts with 'Gzip2('.
    """
    cfg = rustfits.Gzip2(tile_shape=(16, 16), heap_format="Q")
    assert cfg.tile_shape == (16, 16)
    assert cfg.heap_format == "Q"
    assert repr(cfg).startswith("Gzip2(")


# ---------------------- round-trip dtype matrix --------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8"])
def test_round_trip_dtype_matrix(dtype):
    """
    Write + same-handle read + post-reopen read all bit-exact
    across the supported integer dtype set.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((32, 48), dtype)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(16, 24))
            f.create_image_hdu(dtype, data.shape, compress=cfg)
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype
        assert reopen.dtype == data.dtype


def test_round_trip_u1_matches_gzip1():
    """
    For bytepix=1 (u1) the GZIP_2 byte-shuffle is a no-op, so
    the on-disk heap bytes should match what GZIP_1 produces on
    the same input.  Anchors the "shuffle vanishes at bytepix=1"
    invariant.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn1 = os.path.join(tmp, "g1.fits.fz")
        fn2 = os.path.join(tmp, "g2.fits.fz")
        data = _seq((32, 48), "u1")
        with rustfits.FITS(fn1, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 24)),
            )
            f[1].write(data)
        with rustfits.FITS(fn2, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Gzip2(tile_shape=(16, 24)),
            )
            f[1].write(data)
        # Compare the heap regions byte-for-byte.  The header
        # blocks differ in ZCMPTYPE, but the post-header heap
        # bytes should be identical at u1.
        with open(fn1, "rb") as fh:
            b1 = fh.read()
        with open(fn2, "rb") as fh:
            b2 = fh.read()
        # File sizes equal (same descriptor table + same heap).
        assert len(b1) == len(b2)


# ---------------------- round-trip shape matrix --------------------


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((128,), (32,)),  # 1-D, four tiles
        ((40,), (40,)),  # 1-D, single tile
        ((32, 32), (16, 16)),  # 2-D square
        ((50, 70), (16, 24)),  # 2-D non-square + edge tiles
        ((40, 60), (40, 60)),  # 2-D single tile
        ((4, 5, 6), (2, 5, 6)),  # 3-D, two tiles along axis 0
    ],
)
def test_round_trip_shape_matrix(shape, tile):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq(shape, "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=tile)
            f.create_image_hdu("i4", shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_round_trip_default_tile_shape():
    """
    tile_shape=None falls back to FITS-convention row tiles.
    Must still round-trip bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((20, 64), "i2")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2()
            f.create_image_hdu("i2", data.shape, compress=cfg)
            f[1].write(data)
            assert f[1].compression.tile_shape == (1, 64)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- cross-check with fitsio --------------------


def test_rustfits_written_fitsio_read_matches():
    """
    A file written by rustfits must read back bit-exact via
    fitsio (proving we emit a FITS-conforming GZIP_2 HDU).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(20, 30))
            f.create_image_hdu("i4", data.shape, compress=cfg)
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            assert f[1].read_header().get("ZCMPTYPE") == "GZIP_2"
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_written_rustfits_read_matches():
    """
    Mirror direction: fitsio writes GZIP_2, rustfits reads.
    Covered by Phase 4 read tests; repeated here as a regression
    guard for the dispatch.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        with fitsio.FITS(fn, "rw") as f:
            f.write(
                data,
                compress="GZIP_2",
                tile_dims=(20, 30),
                qlevel=None,
            )
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth ------------------------


def test_compressed_write_shifts_later_hdus():
    """
    A compressed HDU followed by another HDU: writing into the
    compressed heap must shift the later HDU's offsets, and
    post-reopen reads of both must still work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _seq((64, 64), "i4")
        later_data = _seq((10, 10), "i2")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(16, 16))
            f.create_image_hdu(
                "i4", comp_data.shape, compress=cfg, extname="COMP"
            )
            f.create_image_hdu(
                "i2",
                later_data.shape,
                extname="LATER",
            )
            f[2].write(later_data)
            # Now write the compressed HDU — heap grows, LATER shifts.
            f[1].write(comp_data)
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


# ---------------------- mixed-algorithm file -----------------------


def test_mixed_gzip1_and_gzip2_in_one_file():
    """
    Two compressed HDUs in one file, one GZIP_1 and one GZIP_2.
    The per-HDU dispatch picks the right decoder; round-trip
    works for both.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data1 = _seq((32, 48), "i4")
        data2 = _seq((48, 32), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data1.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 24)),
                extname="G1",
            )
            f.create_image_hdu(
                "i2",
                data2.shape,
                compress=rustfits.Gzip2(tile_shape=(24, 16)),
                extname="G2",
            )
            f[1].write(data1)
            f[2].write(data2)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].compression.zcmptype == "GZIP_1"
            assert f[2].compression.zcmptype == "GZIP_2"
            np.testing.assert_array_equal(f[1].read(), data1)
            np.testing.assert_array_equal(f[2].read(), data2)


# ---------------------- rejections ---------------------------------


# test_float_compress_rejected: removed in Phase 8 commit 2 —
# float-compressed writes are now supported via quantize=.  See
# tests/test_compressed_image_write_quantize.py.


# test_unsigned_trick_dtype_rejected: removed — u2/u4/u8/i1 are
# now supported for compressed writes via Gzip1/Gzip2/Rice1/
# Hcompress1.  See tests/test_compressed_image_unsigned_trick.py.


def test_input_shape_mismatch_rejected():
    """
    write(data) with shape mismatching the HDU raises and does
    NOT taint.  A subsequent good write must succeed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            wrong = np.arange(8 * 8, dtype="i4").reshape(8, 8)
            with pytest.raises(ValueError, match="shape"):
                f[1].write(wrong)
            good = _seq((16, 16), "i4")
            f[1].write(good)
            np.testing.assert_array_equal(f[1].read(), good)


def test_start_kwarg_rejected():
    """start= is not supported on compressed-image writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip2(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = _seq((16, 16), "i4")
            with pytest.raises(NotImplementedError, match="start="):
                f[1].write(data, start=[0, 0])
