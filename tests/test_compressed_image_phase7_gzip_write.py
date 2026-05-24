"""
ZIMAGE Phase 7: Gzip1 compressed image writes.

Tests cover:
    - Round-trip via same-handle read and via post-reopen read
    - Cross-check vs fitsio (rustfits writes → fitsio reads, and
      fitsio writes → rustfits reads — both must agree bit-exact)
    - Integer dtype matrix (u1, i2, i4, i8)
    - Shape matrix (1-D, 2-D square, 2-D non-square, 3-D)
    - Tile shapes including default (row tiles), explicit small
      tiles with edge tiles, single whole-image tile
    - Non-last HDU growth: a compressed HDU followed by another
      HDU; the later HDU's offsets must shift to make room for
      the heap and post-reopen reads still work.
    - Float ZBITPIX (-32/-64) rejected with a Phase 8 NotImplemented
    - Unsupported algorithm types rejected (only Gzip1 in Phase 7)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _seq(shape, dtype):
    """
    Build a deterministic test array of `shape` and `dtype`: an
    arange clamped into the dtype's range, then reshaped.  Avoids
    overflow for small-range dtypes while keeping the values
    sensitive to byte-order errors.
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
    """create_image_hdu(..., compress=Gzip1(...)) produces a
    CompressedImageHDU with the right metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (32, 48), compress=cfg, extname="SCI")
            hdu = f[1]
            assert type(hdu).__name__ == "CompressedImageHDU"
            assert hdu.shape == (32, 48)
            assert hdu.dtype == np.int32
            assert hdu.bitpix == 32
            assert hdu.compression.zcmptype == "GZIP_1"
            assert hdu.compression.tile_shape == (16, 16)
            assert hdu.extname == "SCI"
            # Before write, PCOUNT is 0 (heap empty).
            assert hdu.header["PCOUNT"] == 0


# ---------------------- round-trip dtype matrix --------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8"])
def test_round_trip_dtype_matrix(dtype):
    """Write + same-handle read + post-reopen read all bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((32, 48), dtype)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(16, 24))
            f.create_image_hdu(dtype, data.shape, compress=cfg)
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype
        assert reopen.dtype == data.dtype


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
            cfg = rustfits.Gzip1(tile_shape=tile)
            f.create_image_hdu("i4", shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_round_trip_default_tile_shape():
    """tile_shape=None → FITS-convention row tiles.  Must still
    round-trip bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((20, 64), "i2")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1()
            f.create_image_hdu("i2", data.shape, compress=cfg)
            f[1].write(data)
            # Default row tiles: ZTILE1 = NAXIS1, others = 1
            # In numpy axis order (slowest first), that's (1, 64).
            assert f[1].compression.tile_shape == (1, 64)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- cross-check with fitsio --------------------


def test_rustfits_written_fitsio_read_matches():
    """A file written by rustfits must read back bit-exact via
    fitsio (proving we emit a FITS-conforming compressed HDU)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(20, 30))
            f.create_image_hdu("i4", data.shape, compress=cfg)
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            assert f[1].read_header().get("ZCMPTYPE") == "GZIP_1"
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_written_rustfits_read_matches():
    """The mirror direction: fitsio writes, rustfits reads.  This
    was already covered by the Phase 4 GZIP read tests; repeating
    here as a sanity check that the dispatch hasn't regressed."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        with fitsio.FITS(fn, "rw") as f:
            f.write(data, compress="GZIP_1", tile_dims=(20, 30), qlevel=None)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth ------------------------


def test_compressed_write_shifts_later_hdus():
    """Create a compressed HDU, then a second HDU after it, then
    write a large compressed image — the second HDU's offsets must
    bump and post-reopen reads of both must still work."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _seq((64, 64), "i4")
        later_data = _seq((10, 10), "i2")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(16, 16))
            f.create_image_hdu(
                "i4", comp_data.shape, compress=cfg, extname="COMP"
            )
            # Second HDU created *before* the write so the heap
            # grow has to shift it.
            f.create_image_hdu("i2", later_data.shape, extname="LATER")
            f[2].write(later_data)
            # Now write the compressed HDU.  Its heap grows; LATER's
            # offsets must shift to make room.
            f[1].write(comp_data)
            # Same-handle reads of both.
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        # Post-reopen.
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


# ---------------------- rejections ---------------------------------


# test_float_compress_rejected: removed in Phase 8 commit 2 —
# float-compressed writes are now supported via quantize=.  See
# tests/test_compressed_image_phase8_quantize_write.py.


# test_unsigned_trick_dtype_rejected: removed — u2/u4/u8/i1 are
# now supported for compressed writes via Gzip1/Gzip2/Rice1/
# Hcompress1.  See tests/test_compressed_image_unsigned_trick.py.


def test_compress_not_a_config_rejected():
    """
    Passing a non-config, non-string object to compress= raises
    TypeError.  (Strings — "GZIP_1" etc. — are accepted as aliases
    for the default-constructed class; this test covers the truly
    invalid-type case.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(TypeError, match="compress="):
                f.create_image_hdu("i4", (16, 16), compress=42)


def test_compress_unknown_string_rejected():
    """An algorithm-name string that isn't recognized raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="unknown compression"):
                f.create_image_hdu("i4", (16, 16), compress="SQUEEZE_1")


@pytest.mark.parametrize(
    "alias,zcmptype",
    [
        ("GZIP_1", "GZIP_1"),
        ("gzip_1", "GZIP_1"),
        ("GZIP", "GZIP_1"),
        ("GZIP_2", "GZIP_2"),
        ("RICE_1", "RICE_1"),
        ("rice", "RICE_1"),
        ("RICE_ONE", "RICE_1"),
        ("HCOMPRESS_1", "HCOMPRESS_1"),
        ("HCOMPRESS", "HCOMPRESS_1"),
        ("PLIO_1", "PLIO_1"),
    ],
)
def test_compress_string_alias_resolves_to_class(alias, zcmptype):
    """
    compress='<alias>' is equivalent to compress=<Class>() with all
    other parameters default.  Verifies via the ZCMPTYPE the on-disk
    header lands with.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        # PLIO_1 only accepts integer images; use small mask-style
        # input for the PLIO case to avoid the algorithm-specific
        # validation kicking in.
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i2", (16, 16), compress=alias)
            f[1].write(np.zeros((16, 16), dtype="i2"))
        with rustfits.FITS(fn, "r") as f:
            assert f[1].header["ZCMPTYPE"] == zcmptype


def test_input_shape_mismatch_rejected():
    """write(data) with shape mismatching the HDU raises and does
    NOT taint."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            wrong = np.arange(8 * 8, dtype="i4").reshape(8, 8)
            with pytest.raises(ValueError, match="shape"):
                f[1].write(wrong)
            # Subsequent good write must succeed (no taint).
            good = _seq((16, 16), "i4")
            f[1].write(good)
            np.testing.assert_array_equal(f[1].read(), good)


def test_start_kwarg_rejected():
    """start= is not supported on compressed-image writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Gzip1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = _seq((16, 16), "i4")
            with pytest.raises(NotImplementedError, match="start="):
                f[1].write(data, start=[0, 0])
