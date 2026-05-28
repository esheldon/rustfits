"""
ZIMAGE slicing on CompressedImageHDU + LRU tile cache.

Covers:
    - __getitem__ with slice/int/ellipsis matches numpy semantics
    - Stepped slices, multi-dim slicing, all-int scalar return
    - LRU cache populates on access, evicts past tile_cache_size,
      drops on set_tile_cache_size(0) and clear_tile_cache()
    - read() populates the cache; subsequent slicing hits warm tiles
    - tile_cache_used reports current bytes
    - Scaling on __getitem__ (BSCALE/BZERO applied automatically)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_rice(tmpdir, shape, dtype, tile_dims=None):
    fname = os.path.join(tmpdir, "t.fits.fz")
    n = int(np.prod(shape))
    data = np.arange(n, dtype=dtype).reshape(shape)
    kw = {"compress": "RICE_1"}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    return fname, data


# -------------------- slice correctness ---------------------------


def test_whole_image_slice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][:, :]
        np.testing.assert_array_equal(got, data)


def test_partial_2d_slice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][2:7, 3:8]
        np.testing.assert_array_equal(got, data[2:7, 3:8])


def test_slice_within_single_tile():
    """Slice fits entirely within one tile — only that tile decoded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            got = hdu[0:3, 0:3]
            # Only 1 tile (5*5*4=100 bytes) cached.
            assert hdu.tile_cache_used == 100
        np.testing.assert_array_equal(got, data[0:3, 0:3])


def test_slice_spanning_multiple_tiles():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            got = hdu[3:7, 3:7]
            # Slice spans all 4 tiles.
            assert hdu.tile_cache_used == 400
        np.testing.assert_array_equal(got, data[3:7, 3:7])


def test_stepped_slice_2d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][::2, ::3]
        np.testing.assert_array_equal(got, data[::2, ::3])


def test_stepped_slice_with_start_stop():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (20, 20),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][2:18:3, 1:19:4]
        np.testing.assert_array_equal(got, data[2:18:3, 1:19:4])


def test_mixed_int_and_slice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            got_row = hdu[3, 2:8]
            got_col = hdu[2:8, 3]
        np.testing.assert_array_equal(got_row, data[3, 2:8])
        np.testing.assert_array_equal(got_col, data[2:8, 3])


def test_all_int_returns_scalar():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            v = fits[1][5, 6]
        # numpy scalar, not ndarray.
        assert isinstance(v, np.int32)
        assert v == data[5, 6]


def test_all_int_3d_returns_scalar():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (4, 5, 6),
            "i4",
            tile_dims=(2, 2, 3),
        )
        with rustfits.FITS(fname, "r") as fits:
            v = fits[1][1, 2, 3]
        assert isinstance(v, np.int32)
        assert v == data[1, 2, 3]


def test_negative_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            v = hdu[-1, -1]
            sub = hdu[-3:, -3:]
        assert v == data[-1, -1]
        np.testing.assert_array_equal(sub, data[-3:, -3:])


def test_ellipsis():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (3, 4, 5),
            "i4",
            tile_dims=(3, 4, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][..., 2:4]
        np.testing.assert_array_equal(got, data[..., 2:4])


def test_1d_slice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (50,),
            "i4",
            tile_dims=(10,),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            assert np.array_equal(hdu[:], data)
            assert np.array_equal(hdu[10:30], data[10:30])
            assert np.array_equal(hdu[5:45:7], data[5:45:7])


def test_3d_slice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (4, 6, 8),
            "i4",
            tile_dims=(2, 3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][1:3, 1:5, 2:6]
        np.testing.assert_array_equal(got, data[1:3, 1:5, 2:6])


def test_edge_tile_slicing():
    """Image dims aren't multiples of tile dims — edge tiles are
    smaller.  Slicing must still produce correct output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (7, 11),
            "i4",
            tile_dims=(3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            np.testing.assert_array_equal(hdu[:, :], data)
            np.testing.assert_array_equal(hdu[5:7, 9:11], data[5:7, 9:11])
            np.testing.assert_array_equal(hdu[::2, ::3], data[::2, ::3])


# -------------------- cache behavior ------------------------------


def test_read_populates_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            assert hdu.tile_cache_used == 0
            _ = hdu.read()
            # 4 tiles × 25 i4 pixels × 4 bytes = 400 bytes.
            assert hdu.tile_cache_used == 400


def test_subsequent_slice_hits_warm_cache():
    """After read(), slicing doesn't grow the cache (all tiles
    already there)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            _ = hdu.read()
            before = hdu.tile_cache_used
            _ = hdu[2:8, 2:8]
            assert hdu.tile_cache_used == before


def test_set_tile_cache_size_zero_disables():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            hdu.set_tile_cache_size(0)
            _ = hdu.read()
            assert hdu.tile_cache_used == 0


def test_clear_tile_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            _ = hdu.read()
            assert hdu.tile_cache_used > 0
            hdu.clear_tile_cache()
            assert hdu.tile_cache_used == 0
            # Cache still works after clear.
            _ = hdu.read()
            assert hdu.tile_cache_used > 0


def test_lru_eviction_when_over_capacity():
    """tile_cache_used never exceeds tile_cache_size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Many 2x2 i4 tiles → each tile = 16 bytes.
        fname, _ = _write_rice(
            tmpdir,
            (20, 20),
            "i4",
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            # Set a tiny cap → only a few tiles fit.
            hdu.set_tile_cache_size(64)  # 4 tiles
            _ = hdu.read()
            assert hdu.tile_cache_used <= 64


def test_shrinking_capacity_evicts():
    """Setting a smaller cap drops LRU entries to fit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            _ = hdu.read()
            assert hdu.tile_cache_used == 400
            hdu.set_tile_cache_size(200)
            assert hdu.tile_cache_used <= 200


def test_default_cache_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (4, 4),
            "i4",
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.tile_cache_size == 32 * 1024 * 1024


def test_tile_cache_used_starts_at_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (4, 4),
            "i4",
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            assert hdu.tile_cache_used == 0


# -------------------- scaling on __getitem__ -----------------------


def test_getitem_applies_scaling():
    """__getitem__ always scales (matches table-side convention)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].header["BSCALE"] = 2.0
            fits[1].header["BZERO"] = 100.0
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[1][2:5, 2:5]
        expected = data[2:5, 2:5].astype("f8") * 2.0 + 100.0
        np.testing.assert_array_equal(sub, expected)


# -------------------- dtype preservation ---------------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_slice_dtype_matches_image(dtype):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_rice(
            tmpdir,
            (10, 10),
            dtype,
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[1][3:7, 3:7]
        assert sub.dtype == data.dtype
        np.testing.assert_array_equal(sub, data[3:7, 3:7])


# -------------------- empty slices --------------------------------


def test_empty_slice_returns_empty_array():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            sub = fits[1][5:5, :]
            assert sub.shape == (0, 10)
            assert sub.dtype == np.int32


# -------------------- still rejects what should --------------------


# test_setitem_still_raises: removed — CompressedImageHDU.__setitem__
# is now supported.  See tests/test_image_compressed_setitem.py.


def test_fancy_list_raises_clear_error():
    """Fancy list indexing isn't supported (matches ImageHDU)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 10),
            "i4",
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="unsupported index"):
                _ = fits[1][[1, 3, 5]]
