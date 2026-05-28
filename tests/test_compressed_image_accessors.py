"""
ZIMAGE detection, dispatch, accessors.

Confirms that a tile-compressed image HDU is parsed as a
CompressedImageHDU (not TableHDU), and that the image-side and
compression-side accessors all return the right values.  Reading
the actual pixel data is Phase 2 work.

Test fixtures are built with fitsio (already a dependency in the
dev env).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_rice(tmpdir, shape, dtype, tile_dims=None, extname=None):
    """
    Helper: write a single tile-compressed (RICE_1) image to a
    temp file.  fitsio always emits an empty primary + the
    compressed extension at index 1.
    """
    fname = os.path.join(tmpdir, "t.fits.fz")
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    kw = {"compress": "RICE_1"}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    if extname is not None:
        kw["extname"] = extname
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    return fname, data


# -------------------- dispatch -------------------------------------


def test_compressed_image_dispatched_to_compressed_pyclass():
    """
    A BINTABLE HDU with ZIMAGE=T should land as CompressedImageHDU,
    not TableHDU.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (10, 10), "i4", tile_dims=(5, 5))
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert isinstance(hdu, rustfits.CompressedImageHDU)
        assert not isinstance(hdu, rustfits.TableHDU)


def test_uncompressed_bintable_still_dispatches_to_tablehdu():
    """
    Sanity check: a plain (uncompressed) BINTABLE still goes to
    TableHDU.  ZIMAGE detection must not fire on the wrong files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        dtype = np.dtype([("x", "i4"), ("y", "f8")])
        rows = np.zeros(3, dtype=dtype)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(dtype=dtype, nrows=3)
            fits[1].write(rows)
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert isinstance(hdu, rustfits.TableHDU)
        assert not isinstance(hdu, rustfits.CompressedImageHDU)


# -------------------- image-side accessors -------------------------


def test_shape_in_numpy_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (7, 13), "i4", tile_dims=(7, 13))
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        # FITS stores ZNAXIS1=13, ZNAXIS2=7; numpy order is (7, 13).
        assert hdu.shape == (7, 13)


def test_dtype_and_bitpix_track_zbitpix():
    """ZBITPIX=32 → dtype int32, bitpix attribute returns 32."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.dtype == np.int32
        assert hdu.bitpix == 32


def test_dtype_i2():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i2")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.dtype == np.int16
        assert hdu.bitpix == 16


def test_ndim_size_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (5, 8), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.ndim == 2
        assert hdu.size == 40
        assert len(hdu) == 5


def test_1d_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (20,), "i4", tile_dims=(20,))
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.shape == (20,)
        assert hdu.ndim == 1
        assert len(hdu) == 20


def test_unit_when_unset_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.unit is None


def test_unit_when_set():
    """BUNIT is allowed on compressed images (informational)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        # BUNIT isn't protected; set it after the fact.
        with rustfits.FITS(fname, "r+") as fits:
            fits[1].header["BUNIT"] = "counts/s"
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.unit == "counts/s"


# -------------------- compression-specific accessors ---------------


def test_compression_type_rice():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.compression.zcmptype == "RICE_1"


def test_tile_shape_explicit():
    """Explicit ZTILE1, ZTILE2 → returned in numpy order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 12),
            "i4",
            tile_dims=(5, 6),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        # FITS order ZTILE1=6, ZTILE2=5 → numpy order (5, 6).
        assert hdu.compression.tile_shape == (5, 6)


def test_n_tiles_explicit():
    """ceil(10/5) * ceil(12/6) = 2 * 2 = 4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 12),
            "i4",
            tile_dims=(5, 6),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.n_tiles == 4


def test_n_tiles_non_aligned_divisor():
    """ceil(10/3) * ceil(12/4) = 4 * 3 = 12 (tiles overhang)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 12),
            "i4",
            tile_dims=(3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.n_tiles == 12


# -------------------- tile cache plumbing --------------------------


def test_tile_cache_default_is_32_mib():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.tile_cache_size == 32 * 1024 * 1024


def test_set_tile_cache_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            hdu.set_tile_cache_size(1024 * 1024)
            assert hdu.tile_cache_size == 1024 * 1024
            hdu.set_tile_cache_size(0)
            assert hdu.tile_cache_size == 0


# -------------------- inherited HDU accessors ----------------------


def test_extname_when_unset_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.extname is None


def test_extname_when_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (4, 4),
            "i4",
            extname="COMP_SCI",
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.extname == "COMP_SCI"


def test_extver_defaults_to_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.extver == 1


def test_has_data_true_for_real_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.has_data is True


def test_index_correct():
    """Compressed image typically sits at index 1 after fitsio's
    empty primary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            assert fits[1].index == 1


def test_header_returns_bintable_view():
    """
    The on-disk header is a BINTABLE header — raw lookups for
    BITPIX should return 8 (the BINTABLE bitpix), not the image
    ZBITPIX.  The .bitpix accessor returns the image-side value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(tmpdir, (4, 4), "i4")
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.header["BITPIX"] == 8
        assert hdu.header["XTENSION"] == "BINTABLE"
        assert hdu.header["ZBITPIX"] == 32
        assert hdu.bitpix == 32


# -------------------- repr -----------------------------------------


def test_repr_contains_compression_info():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_rice(
            tmpdir,
            (10, 12),
            "i4",
            tile_dims=(5, 6),
        )
        with rustfits.FITS(fname, "r") as fits:
            r = repr(fits[1])
        assert "COMPRESSED_IMAGE_HDU" in r
        # Repr inlines the algorithm-config's own __repr__ via
        # `compression: Rice1(...)`; the Pythonic class name (not
        # the FITS-spec ZCMPTYPE) is what shows up.
        assert "Rice1(" in r
        assert "tile_shape=[5, 6]" in r
        assert "i4" in r


# -------------------- iteration / mixed file -----------------------


def test_iterate_over_mixed_hdus():
    """
    File with [primary IMAGE, compressed IMAGE, table] — iterating
    yields the right pyclass for each.  Confirms dispatch in the
    presence of multiple HDU types.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "mixed.fits.fz")
        with fitsio.FITS(fname, "rw") as f:
            img = np.arange(16, dtype="i4").reshape(4, 4)
            f.write(img, compress="RICE_1", tile_dims=(2, 2))
            rec = np.zeros(3, dtype=[("a", "i4"), ("b", "f8")])
            f.write(rec)
        with rustfits.FITS(fname, "r") as fits:
            hdus = list(fits)
        assert len(hdus) == 3
        assert isinstance(hdus[0], rustfits.ImageHDU)
        assert isinstance(hdus[1], rustfits.CompressedImageHDU)
        assert isinstance(hdus[2], rustfits.TableHDU)


# -------------------- different compression types parse ok ---------


@pytest.mark.parametrize(
    "compress,expected",
    [("GZIP_1", "GZIP_1"), ("GZIP_2", "GZIP_2"), ("PLIO_1", "PLIO_1")],
)
def test_other_compression_types_dispatched(compress, expected):
    """
    Phase 1 detects ZIMAGE regardless of compression algorithm.
    Decoding non-RICE algorithms is Phase 4/6; here we only check
    that dispatch + the compression_type accessor work.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        img = np.arange(16, dtype="i4").reshape(4, 4)
        with fitsio.FITS(fname, "rw") as f:
            f.write(img, compress=compress, tile_dims=(2, 2))
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert isinstance(hdu, rustfits.CompressedImageHDU)
        assert hdu.compression.zcmptype == expected
