"""
ZIMAGE Phase 2: RICE_1 whole-image read.

Round-trip tests use fitsio to write RICE_1-compressed fixtures
and rustfits to read them back, checking byte-exactness of the
recovered array.  Also covers:
    - isinstance(hdu, ImageHDU) after the inheritance restructure
    - scale=True / scale=False behavior on scaled HDUs
    - NotImplementedError on __getitem__ / write / extend /
      __setitem__ until later phases land
    - Non-RICE compression types reject with NotImplementedError
    - Float ZBITPIX rejects with a clear "Phase 5" message
    - mask_blank=True rejects until ZBLANK plumbing exists
    - Various tile shapes including edge tiles, default tiles,
      and whole-image tiles
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_rice(
    tmpdir, shape, dtype, tile_dims=None, extname=None, start_value=0
):
    """
    Build a compressed-image fixture with fitsio.  Data is a
    contiguous range starting at `start_value`, then reshaped to
    `shape` — gives a known sequence of differences so round-trip
    failures are easy to debug.
    """
    fname = os.path.join(tmpdir, "t.fits.fz")
    n = int(np.prod(shape))
    data = np.arange(
        start_value,
        start_value + n,
        dtype=dtype,
    ).reshape(shape)
    kw = {"compress": "RICE_1"}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    if extname is not None:
        kw["extname"] = extname
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    return fname, data


# -------------------- inheritance ----------------------------------


def test_compressed_image_is_image_hdu():
    """isinstance(hdu, ImageHDU) returns True after the Phase 2
    inheritance restructure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (8, 8), "i4", tile_dims=(4, 4))
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert isinstance(hdu, rustfits.ImageHDU)
        assert isinstance(hdu, rustfits.CompressedImageHDU)
        assert isinstance(hdu, rustfits.HDU)


# -------------------- round-trip exactness -------------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_roundtrip_2d(dtype):
    """Various BITPIX, 2-D image with explicit tile shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            dtype,
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_roundtrip_1d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (50,),
            "i4",
            tile_dims=(50,),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_roundtrip_3d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (4, 6, 8),
            "i4",
            tile_dims=(2, 3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_roundtrip_negative_values():
    """ZigZag decoding correctly handles negative differences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = np.array(
            [
                [-100, 50, -50, 75],
                [10, -10, 20, -20],
                [0, -1, 1, -2],
                [1000, -1000, 500, -500],
            ],
            dtype="i4",
        )
        with fitsio.FITS(fname, "rw") as f:
            f.write(data, compress="RICE_1", tile_dims=(2, 2))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_roundtrip_constant_image():
    """All-same-value image — exercises the fs=-1 low-entropy run
    branch of the decoder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = np.full((8, 8), 42, dtype="i4")
        with fitsio.FITS(fname, "rw") as f:
            f.write(data, compress="RICE_1", tile_dims=(4, 4))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_roundtrip_high_entropy():
    """Random pixel values exercise the high-entropy / raw branch
    (fs == fsmax) for some blocks."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = rng.integers(
            -1_000_000,
            1_000_000,
            size=(32, 32),
            dtype="i4",
        )
        with fitsio.FITS(fname, "rw") as f:
            f.write(data, compress="RICE_1", tile_dims=(16, 16))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


# -------------------- tile-shape variations ------------------------


def test_edge_tiles_image_not_multiple_of_tile():
    """Image dims aren't multiples of tile dims → edge tiles are
    smaller than nominal.  Tests the tile_origin_and_shape edge-
    clipping logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (7, 11),
            "i4",
            tile_dims=(3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_default_tile_shape():
    """No explicit tile_dims → fitsio writes ZTILE1=ZNAXIS1 and
    ZTILE2=1 (row tiles)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 15),
            "i4",
            tile_dims=None,
        )
        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].tile_shape == (1, 15)
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_whole_image_single_tile():
    """A single tile covering the whole image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (8, 12),
            "i4",
            tile_dims=(8, 12),
        )
        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].n_tiles == 1
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def test_many_small_tiles():
    """Many tiles to exercise the per-tile loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (20, 20),
            "i4",
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].n_tiles == 100
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


# -------------------- scale=True/False -----------------------------


def test_scale_false_returns_native_dtype():
    """scale=False returns the BITPIX-native dtype without
    BSCALE/BZERO applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (4, 4),
            "i4",
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read(scale=False)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, data)


def test_scaled_compressed_hdu():
    """Add BSCALE/BZERO after fitsio writes the file; read with
    scale=True should apply scaling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (4, 4),
            "i4",
            tile_dims=(2, 2),
        )
        # BSCALE/BZERO aren't protected; patch them in.
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].header["BSCALE"] = 2.0
            fits[1].header["BZERO"] = 10.0
        with rustfits.FITS(fname, "r") as fits:
            scaled = fits[1].read(scale=True)
            raw = fits[1].read(scale=False)
        # scale=True: physical = stored * 2 + 10 (promoted to f8)
        np.testing.assert_array_equal(
            scaled,
            data.astype("f8") * 2.0 + 10.0,
        )
        np.testing.assert_array_equal(raw, data)


# -------------------- NotImplementedError stubs --------------------


# __getitem__ shipped in Phase 3 — see test_compressed_image_phase3_slice.py.
# write() for Rice1 shipped in the Phase 7 follow-up — see
# test_compressed_image_phase7_rice_write.py.


def test_extend_raises_not_implemented():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r+") as fits:
            hdu = fits[1]
            with pytest.raises(NotImplementedError, match="Phase 7"):
                hdu.extend(np.zeros((4, 4), dtype="i4"))


def test_setitem_raises_not_implemented():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r+") as fits:
            hdu = fits[1]
            with pytest.raises(NotImplementedError, match="Phase 7"):
                hdu[0:2, 0:2] = np.zeros((2, 2), dtype="i4")


# -------------------- mask_blank rejection -------------------------


def test_mask_blank_rejected_with_clear_message():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(NotImplementedError, match="ZBLANK"):
                fits[1].read(mask_blank=True)


# -------------------- repr / accessors after Phase 2 ---------------


def test_phase1_accessors_still_work():
    """Phase 2 didn't break the Phase 1 accessor surface."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (6, 8),
            "i4",
            tile_dims=(3, 4),
            extname="SCI",
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.shape == (6, 8)
        assert hdu.dtype == np.int32
        assert hdu.bitpix == 32
        assert hdu.ndim == 2
        assert hdu.size == 48
        assert len(hdu) == 6
        assert hdu.compression_type == "RICE_1"
        assert hdu.tile_shape == (3, 4)
        assert hdu.n_tiles == 4
        assert hdu.extname == "SCI"
        assert hdu.extver == 1
        assert hdu.has_data is True
        assert hdu.tile_cache_size == 32 * 1024 * 1024


# -------------------- mixed HDU file -------------------------------


def test_read_compressed_from_mixed_file():
    """File with primary + compressed + table — reads the
    compressed one correctly without disturbing the others."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "mixed.fits.fz")
        img = np.arange(64, dtype="i4").reshape(8, 8)
        rec = np.zeros(3, dtype=[("a", "i4"), ("b", "f8")])
        rec["a"] = [1, 2, 3]
        with fitsio.FITS(fname, "rw") as f:
            f.write(img, compress="RICE_1", tile_dims=(4, 4))
            f.write(rec)
        with rustfits.FITS(fname, "r") as fits:
            assert isinstance(fits[1], rustfits.CompressedImageHDU)
            assert isinstance(fits[2], rustfits.TableHDU)
            got = fits[1].read()
        np.testing.assert_array_equal(got, img)
