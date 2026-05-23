"""
Quantized-float compressed image mutation: extend + __setitem__.

The careful re-encoding scheme: when modifying a tile that was
originally stored in the primary (quantized) column, rustfits
re-uses the EXISTING per-tile bscale/bzero (read from the
ZSCALE/ZZERO columns) AND the existing dither seed.  This makes
`requantize(dequantize(stored))` idempotent for unchanged pixels,
so they round-trip with NO compounding quantization loss.

For values that don't fit the existing per-tile scale, the
encoder returns a clear error pointing the user at `quantize=None`
for unrestricted (lossless) mutation.  Tiles originally in the
GZIP fallback (raw float bytes — lossless backup for ranges too
wide to quantize) stay in the fallback after modification.

Tests cover:
    - Untouched-tile bit-exact round-trip after extend / __setitem__.
    - Unchanged-pixels-in-modified-tile bit-exact round-trip.
    - Out-of-range rejection (clear error).
    - 1-D and 2-D images; f4 and f8 dtypes.
    - All three dither methods (no_dither, dither1, dither2).
    - Multi-tile slice modifications.
    - Partial-last-tile extend on quantized HDU.
    - astropy cross-read after mutation.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")


def _write_quant(fn, shape, dtype, tile_shape, method, data):
    """Write a fresh quantized-float HDU with `data`."""
    with rustfits.FITS(fn, "w+") as f:
        f.create_image_hdu(
            dtype,
            shape,
            compress=rustfits.Gzip1(tile_shape=tile_shape),
            quantize=rustfits.Quantize(method=method, seed=1),
        )
        f[1].write(data)


# ---------------------- extend: no compounding loss ----------------


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_extend_quantized_untouched_tiles_idempotent(method, dtype):
    """
    Extend on a quantized HDU.  Tiles entirely in the OLD image
    (not touched by extend) must round-trip BIT-EXACT relative to
    a reference file with no extend.  This is the no-compounding-
    loss guarantee.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        fn_ext = os.path.join(tmp, "ext.fits.fz")
        rng = np.random.default_rng(0)
        # 32 rows / tile=16 = 2 tiles (clean alignment, no boundary)
        initial = rng.standard_normal(32).astype(dtype)
        more = rng.standard_normal(16).astype(dtype)
        # Reference: just write initial
        _write_quant(fn_ref, (32,), dtype, (16,), method, initial)
        # Extended: write initial + extend
        with rustfits.FITS(fn_ext, "w+") as f:
            f.create_image_hdu(
                dtype,
                (32,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                quantize=rustfits.Quantize(method=method, seed=1),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn_ref, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fn_ext, "r") as f:
            rt = f[1].read()
        # Tiles 0..31 (entirely old) must be bit-exact
        np.testing.assert_array_equal(ref[:32], rt[:32])


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
def test_extend_quantized_boundary_pixels_idempotent(method):
    """
    Extend that crosses a partial-last-tile boundary.  The pixels
    in the boundary tile that EXISTED before extend must round-
    trip bit-exact (since they're re-quantized with the same
    bscale/bzero/seed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        fn_ext = os.path.join(tmp, "ext.fits.fz")
        rng = np.random.default_rng(1)
        # 20 rows / tile=16 → 2 tiles, last is partial (4 rows)
        initial = rng.standard_normal(20).astype("f4")
        more = rng.standard_normal(15).astype("f4")
        _write_quant(fn_ref, (20,), "f4", (16,), method, initial)
        with rustfits.FITS(fn_ext, "w+") as f:
            f.create_image_hdu(
                "f4",
                (20,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                quantize=rustfits.Quantize(method=method, seed=1),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn_ref, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fn_ext, "r") as f:
            rt = f[1].read()
        # Pixels 0..15 are entirely in tile 0 (not touched): bit-exact
        np.testing.assert_array_equal(ref[:16], rt[:16])
        # Pixels 16..19 are in boundary tile (re-encoded with same
        # scale): bit-exact
        np.testing.assert_array_equal(ref[16:20], rt[16:20])


# ---------------------- __setitem__: no compounding loss -----------


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_setitem_quantized_untouched_tiles_idempotent(method, dtype):
    """
    __setitem__ on a quantized HDU.  Tiles outside the modified
    region must be bit-exact relative to a reference file with no
    mutation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        fn_mod = os.path.join(tmp, "mod.fits.fz")
        rng = np.random.default_rng(2)
        data = rng.standard_normal((16, 16)).astype(dtype)
        _write_quant(fn_ref, (16, 16), dtype, (8, 8), method, data)
        # Modify only one tile (bottom-right, [8:, 8:])
        with rustfits.FITS(fn_mod, "w+") as f:
            f.create_image_hdu(
                dtype,
                (16, 16),
                compress=rustfits.Gzip1(tile_shape=(8, 8)),
                quantize=rustfits.Quantize(method=method, seed=1),
            )
            f[1].write(data)
            f[1][12:14, 12:14] = 0.0
        with rustfits.FITS(fn_ref, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fn_mod, "r") as f:
            rt = f[1].read()
        # Untouched tiles: top-left, top-right, bottom-left
        np.testing.assert_array_equal(ref[:8, :8], rt[:8, :8])
        np.testing.assert_array_equal(ref[:8, 8:], rt[:8, 8:])
        np.testing.assert_array_equal(ref[8:, :8], rt[8:, :8])


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
def test_setitem_quantized_unchanged_pixels_in_modified_tile(method):
    """
    Within a tile that gets modified by __setitem__, the pixels
    that were NOT touched by the user's selection must still round-
    trip bit-exact (since they're re-quantized with the same
    per-tile bscale/bzero/seed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        fn_mod = os.path.join(tmp, "mod.fits.fz")
        rng = np.random.default_rng(3)
        data = rng.standard_normal((16, 16)).astype("f4")
        _write_quant(fn_ref, (16, 16), "f4", (8, 8), method, data)
        # Modify [8:12, 8:12] — top-left 4x4 of the bottom-right tile
        # (which spans [8:, 8:]).  Pixels in [12:16, 12:16] of that
        # tile are unchanged and should be bit-exact.
        with rustfits.FITS(fn_mod, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Gzip1(tile_shape=(8, 8)),
                quantize=rustfits.Quantize(method=method, seed=1),
            )
            f[1].write(data)
            f[1][8:12, 8:12] = 0.0
        with rustfits.FITS(fn_ref, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fn_mod, "r") as f:
            rt = f[1].read()
        # Unchanged sub-regions in the modified tile must be bit-exact
        np.testing.assert_array_equal(ref[12:16, 8:16], rt[12:16, 8:16])
        np.testing.assert_array_equal(ref[8:12, 12:16], rt[8:12, 12:16])


# ---------------------- modified pixels approximate ----------------


def test_setitem_modified_pixels_round_trip_within_noise():
    """
    The pixels we DID modify get one round of quantization noise.
    They should be close to the value we wrote, but not exactly
    (within roughly ±bscale of the per-tile scale).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(4)
        data = rng.standard_normal((8, 8)).astype("f4")
        _write_quant(fn, (8, 8), "f4", (8, 8), "dither1", data)
        with rustfits.FITS(fn, "r+") as f:
            # Write a known value
            f[1][3:5, 3:5] = 0.0
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        # Modified pixels should be near 0.0 (within tile's quant step)
        # The original data has std ~1.0, so bscale ~0.25 with qlevel=4.
        # So tolerance ~0.5 is generous.
        assert np.abs(rt[3:5, 3:5] - 0.0).max() < 0.5


# ---------------------- out-of-range rejection ---------------------


def test_setitem_out_of_range_rejected():
    """
    A scalar value far outside the existing tile's quantization
    range must be rejected with a clear error pointing at
    quantize=None.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(5)
        data = rng.standard_normal((8, 8)).astype("f4")  # range ~[-3, 3]
        _write_quant(fn, (8, 8), "f4", (8, 8), "dither1", data)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError, match="outside"):
                f[1][3, 3] = 1e30


def test_extend_out_of_range_in_boundary_rejected():
    """
    Extend that puts an out-of-range value into the partial last
    tile (boundary tile) must be rejected.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(6)
        # 10 pixels / tile=16 = 1 partial tile of 10 pixels
        initial = rng.standard_normal(10).astype("f4")
        # Extend by 6 pixels with one wildly out-of-range value
        more = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1e30], dtype="f4")
        _write_quant(fn, (10,), "f4", (16,), "dither1", initial)
        with rustfits.FITS(fn, "r+") as f:
            with pytest.raises(ValueError, match="outside"):
                f[1].extend(more)


# ---------------------- multi-tile slice setitem -------------------


def test_setitem_multi_tile_slice():
    """
    Slice that spans multiple tiles modifies each tile correctly
    without disturbing the others.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(7)
        data = rng.standard_normal((16, 16)).astype("f4")
        _write_quant(fn, (16, 16), "f4", (8, 8), "dither1", data)
        # Modify a 6x6 block spanning the 2x2 tile grid
        new = np.full((6, 6), 0.5, dtype="f4")
        with rustfits.FITS(fn, "r+") as f:
            f[1][5:11, 5:11] = new
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        # Modified region near 0.5 within quant noise
        assert np.abs(rt[5:11, 5:11] - 0.5).max() < 0.5


# ---------------------- astropy cross-read -------------------------


def test_astropy_reads_after_quantized_setitem():
    """
    A quantized-float HDU modified by rustfits must read back via
    astropy.  We don't assert bit-exact (quantized output already
    has noise vs the original input), but the read must succeed
    and return the same dtype/shape.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(8)
        data = rng.standard_normal((16, 16)).astype("f4")
        _write_quant(fn, (16, 16), "f4", (8, 8), "dither1", data)
        with rustfits.FITS(fn, "r+") as f:
            f[1][6:10, 6:10] = 0.0
        with rustfits.FITS(fn, "r") as f:
            rust_rt = f[1].read()
        with astropy_fits.open(fn) as h:
            ap = h[1].data
        assert ap.shape == rust_rt.shape
        assert ap.dtype == rust_rt.dtype
        # Both decoded values should be bit-exact (same dequant
        # formula, same per-tile bscale/bzero/dither).
        np.testing.assert_array_equal(ap, rust_rt)


# ---------------------- 1-D quantized extend (healsparse) ----------


def test_1d_quantized_extend_healsparse_pattern():
    """
    1-D quantized float extend — the healsparse use case, but with
    lossy compression.  Verify the no-compounding-loss property
    holds for the bulk of the data.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        fn_ext = os.path.join(tmp, "ext.fits.fz")
        rng = np.random.default_rng(9)
        initial = rng.standard_normal(64).astype("f4")
        more = rng.standard_normal(48).astype("f4")
        _write_quant(fn_ref, (64,), "f4", (16,), "dither1", initial)
        with rustfits.FITS(fn_ext, "w+") as f:
            f.create_image_hdu(
                "f4",
                (64,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                quantize=rustfits.Quantize(method="dither1", seed=1),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn_ref, "r") as f:
            ref = f[1].read()
        with rustfits.FITS(fn_ext, "r") as f:
            rt = f[1].read()
        # All 4 original tiles (rows 0..64) untouched
        np.testing.assert_array_equal(ref, rt[:64])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
