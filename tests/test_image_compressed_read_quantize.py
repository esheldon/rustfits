"""
ZIMAGE quantized-float read.

Floating-point ZBITPIX (-32 / -64) reads go through a per-tile
dequantization step: ZSCALE and ZZERO from adjacent BINTABLE
columns, plus (for SUBTRACTIVE_DITHER_1/2) a per-pixel offset
from the FITS-spec Park-Miller PRNG seeded with ZDITHER0.

Coverage:
    - Round-trip exactness vs. fitsio across the {RICE_1, GZIP_1,
      GZIP_2} x {f4, f8} x {NO_DITHER, SUBTRACTIVE_DITHER_1,
      SUBTRACTIVE_DITHER_2} matrix.  cfitsio is the reference;
      rustfits must match it bit-for-bit (not just approximately
      — the dither offsets and dequant formula are exact).
    - NaN preservation through SUBTRACTIVE_DITHER_2 (the spec
      reserves stored value -2147483647 for NaN).
    - dtype agreement: ZBITPIX=-32 → float32, -64 → float64.
    - Slicing path (__getitem__) uses the same dequant code so it
      should match too.
    - Inheritance: still isinstance(hdu, ImageHDU) for compressed
      float HDUs.
    - Tile cache stores the post-dequant float bytes — same
      semantics as the integer cache, just different per-pixel
      width.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


# On macOS, cfitsio's dequantization produces slightly different
# float results than on Linux — almost certainly compiler-codegen
# variance (FMA fusion, Apple libm vs glibc).  rustfits's Rust
# dequant bit-matches Linux cfitsio.  We pin the divergence here
# so the level is documented (current observation: max ~4.6e-6
# relative on f4, ~3.6e-16 absolute on near-zero values) and a
# future regression beyond this bound surfaces visibly.  Tighten
# the rtol/atol if a future fitsio release narrows the gap.
_MACOS_FP_RTOL = 1e-5
_MACOS_FP_ATOL = 1e-10
_IS_MACOS = sys.platform == "darwin"


def _write_quantized(
    tmpdir,
    shape,
    dtype,
    compress,
    *,
    tile_dims=None,
    qmethod=None,
    qlevel=16.0,
    inject_nans=False,
):
    """
    Build a quantized-float compressed-image fixture with fitsio.
    Reads it back through fitsio too and returns (fname, ref_decoded)
    so tests can compare rustfits's read against cfitsio's.
    """
    fname = os.path.join(tmpdir, "t.fits.fz")
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, shape).astype(dtype)
    if inject_nans:
        # Sprinkle a couple of NaNs at known positions.
        flat = data.reshape(-1)
        flat[3] = np.nan
        flat[len(flat) // 2] = np.nan
    kw = {"compress": compress, "qlevel": qlevel}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    if qmethod is not None:
        kw["qmethod"] = qmethod
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    with fitsio.FITS(fname, "r") as f:
        ref = f[1].read()
    return fname, ref


def _arrays_match(got, ref):
    """
    NaN-aware cross-tool equality.

    Linux: bit-for-bit (the dequant formula is deterministic and
    rustfits's Rust port matches conda-forge linux cfitsio
    exactly).  macOS: allclose at the level documented by
    ``_MACOS_FP_RTOL`` / ``_MACOS_FP_ATOL`` — cfitsio's dequant
    on macOS computes the same formula but the platform's
    compiler/libm produces ULP-level differences.
    """
    assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    got_nan = np.isnan(got)
    ref_nan = np.isnan(ref)
    np.testing.assert_array_equal(got_nan, ref_nan)
    if _IS_MACOS:
        np.testing.assert_allclose(
            got[~got_nan],
            ref[~ref_nan],
            rtol=_MACOS_FP_RTOL,
            atol=_MACOS_FP_ATOL,
        )
    else:
        np.testing.assert_array_equal(got[~got_nan], ref[~ref_nan])


# -------------------- exhaustive round-trip matrix -----------------


@pytest.mark.parametrize("compress", ["RICE_1", "GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize(
    "qmethod,label",
    [
        (-1, "NO_DITHER"),
        (1, "SUBTRACTIVE_DITHER_1"),
        (2, "SUBTRACTIVE_DITHER_2"),
    ],
)
def test_roundtrip_matrix(compress, dtype, qmethod, label):
    """
    Exhaustive round-trip across algorithms x dtypes x dither
    methods.  Must match fitsio exactly (no approximate-equality
    tolerance — the dequantization is deterministic).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (16, 16),
            dtype,
            compress,
            tile_dims=(8, 8),
            qmethod=qmethod,
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        _arrays_match(got, ref)


# -------------------- 1D quantized image ---------------------------


@pytest.mark.parametrize("compress", ["RICE_1", "GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_1d_quantized(compress, dtype):
    """
    1-D quantized image — the simplest shape, but worth covering
    because the per-axis tile-iteration / origin math collapses
    to the trivial case and any off-by-one would surface here.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (50,),
            dtype,
            compress,
            tile_dims=(50,),
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        _arrays_match(got, ref)


def test_1d_quantized_multiple_tiles():
    """1-D image split across multiple tiles so multiple
    ZSCALE/ZZERO rows get read; covers the per-tile column
    indexing on a 1-D image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (40,),
            "f4",
            "RICE_1",
            tile_dims=(10,),
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        _arrays_match(got, ref)


# -------------------- NaN preservation (DITHER_2) ------------------


@pytest.mark.parametrize("compress", ["RICE_1", "GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_dither2_preserves_nan(compress, dtype):
    """
    SUBTRACTIVE_DITHER_2 reserves stored value -2147483647 for
    NaN.  cfitsio writes it; the rustfits dequant path must
    decode it back to NaN and not silently turn it into a number.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (12, 12),
            dtype,
            compress,
            tile_dims=(6, 6),
            qmethod=2,
            inject_nans=True,
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        # Refs from fitsio should have NaN where we injected them.
        assert np.isnan(ref).sum() > 0
        # Same NaN mask in both.
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        _arrays_match(got, ref)


# -------------------- output dtype agreement -----------------------


def test_f4_output_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_quantized(tmpdir, (8, 8), "f4", "RICE_1")
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        assert got.dtype == np.float32


def test_f8_output_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_quantized(tmpdir, (8, 8), "f8", "RICE_1")
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        assert got.dtype == np.float64


# -------------------- slicing path uses dequant --------------------


@pytest.mark.parametrize("compress", ["RICE_1", "GZIP_1", "GZIP_2"])
def test_slice_quantized(compress):
    """
    __getitem__ on a quantized HDU shares the same get_or_decode_tile
    path, so dequant should apply there too.  Tile across boundaries
    to make sure multiple tiles' ZSCALE/ZZERO get read correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (12, 12),
            "f4",
            compress,
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1][2:10, 3:11]
        np.testing.assert_array_equal(got, ref[2:10, 3:11])


def test_slice_int_scalar_quantized():
    """All-int multi-axis returns a numpy scalar of the right
    dtype, matching ImageHDU semantics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (8, 8),
            "f4",
            "RICE_1",
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1][3, 5]
        assert isinstance(got, np.floating)
        assert got == ref[3, 5]


# -------------------- inheritance unchanged ------------------------


def test_compressed_float_still_image_hdu():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_quantized(
            tmpdir,
            (8, 8),
            "f4",
            "RICE_1",
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as f:
            hdu = f[1]
        assert isinstance(hdu, rustfits.ImageHDU)
        assert isinstance(hdu, rustfits.CompressedImageHDU)


# -------------------- accessors report float dtype -----------------


def test_dtype_accessor_returns_float():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_quantized(tmpdir, (8, 8), "f8", "RICE_1")
        with rustfits.FITS(fname, "r") as f:
            hdu = f[1]
            assert hdu.dtype == np.float64
            assert hdu.bitpix == -64


# -------------------- cache stores float bytes ---------------------


def test_cache_holds_dequantized_bytes():
    """After a full read of an f4 quantized HDU the cache should
    hold ~n_tiles * tile_n_pixels * 4 bytes of decoded f4 data
    — not the raw i32 quantized bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_quantized(
            tmpdir,
            (8, 8),
            "f4",
            "RICE_1",
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as f:
            hdu = f[1]
            assert hdu.tile_cache_used == 0
            _ = hdu.read()
            # 4 tiles x 16 pixels x 4 bytes = 256.
            assert hdu.tile_cache_used == 4 * 16 * 4


# -------------------- 3D quantized image ---------------------------


def test_3d_quantized():
    """Quantization composes with N-D images — the per-tile
    ZSCALE/ZZERO indexing is by tile-row, independent of image
    dimensionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(
            tmpdir,
            (4, 6, 8),
            "f4",
            "RICE_1",
            tile_dims=(2, 3, 4),
        )
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        _arrays_match(got, ref)


# -------------------- unquantized float HDU ------------------------
#
# When ZQUANTIZ='NONE' (astropy's convention for "no quantization
# happened") or ZSCALE/ZZERO columns are absent, the on-disk bytes
# are raw f4/f8 (compressed by GZIP) — not quantized i32.  The
# reader must skip dequant in this case and decode at the float
# bytepix.  Real-world astropy files use this configuration; we hit
# it on a 12 GB GZIP_2 + f8 sky-mosaic file in the wild.

astropy_fits = pytest.importorskip("astropy.io.fits")


def test_unquantized_float_astropy_no_dither_no_columns():
    """
    astropy's CompImageHDU with quantize_level=0 writes
    ZQUANTIZ='NO_DITHER' but emits NO ZSCALE/ZZERO columns —
    the bytes in COMPRESSED_DATA are raw GZIP-compressed floats.
    Reader must take the no-quant path (missing columns → dequant
    skipped).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = np.arange(64, dtype="f8").reshape(8, 8) * 1.5 - 17.25
        hdul = astropy_fits.HDUList(
            [
                astropy_fits.PrimaryHDU(),
                astropy_fits.CompImageHDU(
                    data,
                    compression_type="GZIP_1",
                    tile_shape=(4, 4),
                    quantize_level=0.0,
                ),
            ]
        )
        hdul.writeto(fname, overwrite=True)
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("compress", ["GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_unquantized_float_matrix(compress, dtype):
    """
    Unquantized float reads work for both GZIP variants and both
    float widths.  Bit-exact round-trip since no quantization is
    in play.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        rng = np.random.RandomState(7)
        data = rng.normal(0, 1, (10, 10)).astype(dtype)
        hdul = astropy_fits.HDUList(
            [
                astropy_fits.PrimaryHDU(),
                astropy_fits.CompImageHDU(
                    data,
                    compression_type=compress,
                    tile_shape=(5, 5),
                    quantize_level=0.0,
                ),
            ]
        )
        hdul.writeto(fname, overwrite=True)
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read()
        assert got.dtype == data.dtype
        np.testing.assert_array_equal(got, data)


def test_unquantized_float_slice():
    """Slicing an unquantized float HDU also takes the no-quant
    path and matches astropy's read exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = np.arange(144, dtype="f4").reshape(12, 12)
        hdul = astropy_fits.HDUList(
            [
                astropy_fits.PrimaryHDU(),
                astropy_fits.CompImageHDU(
                    data,
                    compression_type="GZIP_1",
                    tile_shape=(4, 4),
                    quantize_level=0.0,
                ),
            ]
        )
        hdul.writeto(fname, overwrite=True)
        with rustfits.FITS(fname, "r") as f:
            got = f[1][3:9, 5:10]
        np.testing.assert_array_equal(got, data[3:9, 5:10])


# -------------------- scale=False on quantized HDU -----------------


def test_scale_false_still_returns_dequantized():
    """
    For a quantized-float HDU `scale=False` is intended to bypass
    BSCALE/BZERO (the *post*-dequant scaling, almost never set on
    a quantized HDU).  The dequantization itself is not bypassed
    by scale=False — if it were, the user would just get the
    raw i32 quantized integers, which is rarely what anyone wants.
    Document the current behavior with a test so we notice if it
    ever changes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, ref = _write_quantized(tmpdir, (8, 8), "f4", "RICE_1")
        with rustfits.FITS(fname, "r") as f:
            got = f[1].read(scale=False)
        # Same dequantized values, same float dtype.
        _arrays_match(got, ref)
