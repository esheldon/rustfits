"""
ZIMAGE Phase 6: HCOMPRESS_1 read.

Round-trip tests use fitsio to write HCOMPRESS_1-compressed fixtures
and rustfits to read them back, checking byte-exactness of the
recovered array (lossless mode) or per-pixel error bounds (lossy
mode).

Covers:
    - Integer ZBITPIX 8 / 16 / 32, lossless (hcomp_scale=0)
    - Non-square shapes and edge tiles
    - Whole-image read + slicing parity with ImageHDU
    - SMOOTH=1 rejection (hsmooth boundary clauses not ported yet)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_hcompress(
    tmpdir,
    shape,
    dtype,
    *,
    tile_dims=None,
    hcomp_scale=0.0,
    hcomp_smooth=0,
    extname=None,
    start_value=0,
):
    """
    Build a HCOMPRESS_1-compressed fixture with fitsio.  Data is a
    contiguous range starting at ``start_value``, reshaped to
    ``shape`` — a known sequence of differences makes round-trip
    failures easy to diff.
    """
    fname = os.path.join(tmpdir, "t.fits.fz")
    n = int(np.prod(shape))
    data = np.arange(
        start_value,
        start_value + n,
        dtype=dtype,
    ).reshape(shape)
    kw = {"compress": "HCOMPRESS"}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    if extname is not None:
        kw["extname"] = extname
    kw["qlevel"] = None  # disable quantization for integer images
    with fitsio.FITS(fname, "rw") as f:
        f.write(
            data,
            hcomp_scale=hcomp_scale,
            hcomp_smooth=hcomp_smooth,
            **kw,
        )
    return fname, data


# ---------------------- accessors ----------------------------------


def test_hcompress_accessors_report_correct_metadata():
    """compression_type / shape / dtype / tile_shape all line up
    with what fitsio wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (32, 32)
        fname, _ = _write_hcompress(
            tmp,
            shape,
            np.int16,
            tile_dims=(16, 16),
        )
        with rustfits.FITS(fname, "r") as f:
            hdu = f[1]
            assert hdu.compression_type == "HCOMPRESS_1"
            assert hdu.shape == shape
            assert hdu.dtype == np.int16
            assert hdu.bitpix == 16
            # tile_shape is numpy order; fitsio's tile_dims arg is
            # numpy order too for square tiles so they match.
            assert hdu.tile_shape == (16, 16)


# ---------------------- lossless round trip ------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32])
def test_hcompress_lossless_whole_image(dtype):
    """hcomp_scale=0 → bit-exact recovery on integer ZBITPIX."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (32, 48)
        fname, data = _write_hcompress(
            tmp,
            shape,
            dtype,
            tile_dims=(16, 24),
            hcomp_scale=0.0,
        )
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        assert out.dtype == data.dtype
        np.testing.assert_array_equal(out, data)


def test_hcompress_lossless_non_square_tiles():
    """Tile dimensions different from image dimensions; edge tiles
    smaller than nominal."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (50, 70)  # neither dim divides tile shape
        fname, data = _write_hcompress(
            tmp,
            shape,
            np.int16,
            tile_dims=(16, 24),
        )
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_hcompress_lossless_whole_image_default_tiles():
    """No ``tile_dims`` arg → fitsio's default (row tiles).  Should
    still round-trip bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (20, 64)
        fname, data = _write_hcompress(
            tmp,
            shape,
            np.int16,
        )
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


# ---------------------- slicing ------------------------------------


def test_hcompress_slicing_matches_whole_read():
    """Slicing a HCOMPRESS HDU yields the same pixels as the
    corresponding slice of the whole-image read."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (40, 60)
        fname, _ = _write_hcompress(
            tmp,
            shape,
            np.int16,
            tile_dims=(20, 30),
        )
        with rustfits.FITS(fname, "r") as f:
            whole = f[1].read()
            # Two slices: interior, and a region crossing tile edges.
            assert (f[1][10:30, 15:50] == whole[10:30, 15:50]).all()
            assert (f[1][0:1, :] == whole[0:1, :]).all()
            # Single-pixel int-int indexing.
            assert int(f[1][7, 11]) == int(whole[7, 11])


# ---------------------- lossy round trip ---------------------------


def test_hcompress_lossy_scale4_matches_cfitsio():
    """hcomp_scale=4 + SMOOTH=0: lossy compression.  The decoded
    pixels must match cfitsio's own decode bit-exactly (read the
    same file via fitsio to confirm)."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (32, 32)
        # Use a slowly-varying pattern so quantization noise stays
        # bounded; pure np.arange would have a sharp gradient that
        # exposes quantization more.
        data = np.fromfunction(
            lambda i, j: ((i * 13 + j * 7) % 100).astype(np.int16),
            shape,
        )
        fname = os.path.join(tmp, "t.fits.fz")
        with fitsio.FITS(fname, "rw") as f:
            f.write(
                data,
                compress="HCOMPRESS",
                tile_dims=(16, 16),
                hcomp_scale=4.0,
                hcomp_smooth=0,
                qlevel=None,
            )
        # Reference: what does cfitsio (via fitsio) produce?
        with fitsio.FITS(fname, "r") as f:
            cfitsio_out = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            our_out = f[1].read()
        assert our_out.dtype == cfitsio_out.dtype
        np.testing.assert_array_equal(our_out, cfitsio_out)


# ---------------------- SMOOTH=1 round trip ------------------------


@pytest.mark.parametrize("hcomp_scale", [4.0, 16.0])
def test_hcompress_smooth_matches_cfitsio(hcomp_scale):
    """SMOOTH=1 + lossy: the hsmooth pass must reproduce cfitsio's
    output bit-exactly.  Tested across two scales so the smoothing
    has visible effect (smax = scale/2)."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (32, 48)
        # Slowly varying pattern so the smoothing has meaningful
        # interpolated targets; pure noise would be uninteresting.
        data = np.fromfunction(
            lambda i, j: ((i * 11 + j * 5) % 200).astype(np.int16),
            shape,
        )
        fname = os.path.join(tmp, "t.fits.fz")
        with fitsio.FITS(fname, "rw") as f:
            f.write(
                data,
                compress="HCOMPRESS",
                tile_dims=(16, 24),
                hcomp_scale=hcomp_scale,
                hcomp_smooth=1,
                qlevel=None,
            )
        with fitsio.FITS(fname, "r") as f:
            cfitsio_out = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            our_out = f[1].read()
        assert our_out.dtype == cfitsio_out.dtype
        np.testing.assert_array_equal(our_out, cfitsio_out)


def test_hcompress_smooth_int32_matches_cfitsio():
    """SMOOTH=1 on ZBITPIX=32 — exercises the i64-internal path's
    hsmooth_i64."""
    with tempfile.TemporaryDirectory() as tmp:
        shape = (32, 48)
        data = np.fromfunction(
            lambda i, j: ((i * 1009 + j * 503) % 50_000).astype(np.int32),
            shape,
        )
        fname = os.path.join(tmp, "t.fits.fz")
        with fitsio.FITS(fname, "rw") as f:
            f.write(
                data,
                compress="HCOMPRESS",
                tile_dims=(16, 24),
                hcomp_scale=8.0,
                hcomp_smooth=1,
                qlevel=None,
            )
        with fitsio.FITS(fname, "r") as f:
            cfitsio_out = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            our_out = f[1].read()
        np.testing.assert_array_equal(our_out, cfitsio_out)
