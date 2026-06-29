"""
ZIMAGE quantized float compressed image writes.

Float images are encoded by quantizing each tile to i32 (per-tile
bscale/bzero from a noise estimate), then compressing the i32
stream with the chosen algorithm (Rice1/Gzip1/Gzip2/Hcompress1).
Tiles that can't be quantized fall back to GZIP-compressed raw
float bytes.

Coverage:
    - Schema validation: 4 columns (COMPRESSED_DATA + ZSCALE +
      ZZERO + GZIP_COMPRESSED_DATA), ZQUANTIZ + ZDITHER0 + ZBLANK
      cards
    - Round-trip across f4/f8, all three dither methods
      (NO_DITHER / SUBTRACTIVE_DITHER_1 / SUBTRACTIVE_DITHER_2),
      same-handle + post-reopen
    - DITHER_2 exact-zero preservation (ZERO_VALUE_I32 sentinel)
    - NaN preservation for ALL dither methods (NULL_VALUE_I32
      sentinel; cfitsio interop)
    - GZIP fallback on constant-tile / unquantizable input
    - fitsio cross-read agreement across (algorithm, method) matrix
    - Rejection paths: quantize= on integer dtype, quantize=
      without compress=
"""

import hashlib
import os
import sys
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


# On macOS, cfitsio's dequantization produces slightly different
# float results than on Linux — compiler-codegen variance (FMA
# fusion, Apple libm vs glibc).  rustfits's Rust dequant bit-matches
# Linux cfitsio, and test_rustfits_quantized_bytes_stable_across_os
# (below) pins rustfits's own written + decoded bytes byte-identical
# across platforms — so the drift is empirically isolated to
# fitsio/cfitsio, not rustfits.  Linux remains strictly bit-exact in
# the matrix test below; macOS allows allclose at the documented
# level (rtol up to ~1.6e-5 on near-zero values, atol up to ~2.6e-9).
_MACOS_FP_RTOL = 1e-5
_MACOS_FP_ATOL = 1e-8
_IS_MACOS = sys.platform == "darwin"


def _smooth(shape, dtype="f4", seed=0):
    """
    Smooth-ish float data with enough variation that the per-tile
    noise estimate finds something to work with.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    x = np.arange(n) * 0.1
    base = np.sin(x) * 5.0 + np.cos(x * 0.3) * 2.0
    noise = rng.standard_normal(n) * 0.3
    return (base + noise).astype(dtype).reshape(shape)


# ---------------------- schema -------------------------------------


def test_create_emits_quantize_schema():
    """create_image_hdu(..., compress=Rice1, quantize=Quantize)
    emits a 4-column BINTABLE with ZQUANTIZ + ZDITHER0 cards."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (32, 48),
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=42,
                ),
            )
            hdr = f[1].header
            assert hdr["ZBITPIX"] == -32
            assert hdr["TFIELDS"] == 4
            assert hdr["TTYPE1"].strip() == "COMPRESSED_DATA"
            assert hdr["TTYPE2"].strip() == "ZSCALE"
            assert hdr["TTYPE3"].strip() == "ZZERO"
            assert hdr["TTYPE4"].strip() == "GZIP_COMPRESSED_DATA"
            assert hdr["TFORM2"].strip() == "1D"
            assert hdr["TFORM3"].strip() == "1D"
            assert hdr["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
            assert hdr["ZDITHER0"] == 42


# ---------------------- round-trip ---------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_round_trip_no_dither(dtype):
    """NO_DITHER round-trip: max error bounded by ~bscale (= noise/4)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), dtype=dtype, seed=11)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="no_dither",
                ),
            )
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        assert same.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(same, reopen)
        # Quantization step ≈ noise/4 ≈ 0.3/4 ≈ 0.075 for this
        # data; allow plenty of headroom.
        assert np.abs(same - data).max() < 1.0


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_round_trip_dither1(dtype):
    """SUBTRACTIVE_DITHER_1 round-trip; same precision target as
    NO_DITHER (dither just whitens the quantization noise)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), dtype=dtype, seed=22)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=99,
                ),
            )
            f[1].write(data)
            out = f[1].read()
        assert np.abs(out - data).max() < 1.5


def test_quantize_none_requires_gzip():
    """quantize=None on a float HDU without compress=Gzip1 or Gzip2
    is rejected with a clear error.  Other algorithms (Rice1,
    Hcompress1, Plio1) cannot round-trip raw float bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="unquantized float"):
                f.create_image_hdu(
                    "f4",
                    (20, 20),
                    compress=rustfits.Rice1(tile_shape=(20, 20)),
                )


# ---------------------- fitsio interop -----------------------------


def test_fitsio_read_matches_rustfits_read():
    """
    rustfits-written quantized-float file: cfitsio's reader (via
    fitsio) must return the same physical values rustfits would.
    Bit-exact because both decoders go through the same Park-Miller
    dither table and the same dequant formula.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), seed=44)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method="dither1",
                    seed=12345,
                ),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            rust_out = f[1].read()
        with fitsio.FITS(fn) as f:
            fitsio_out = f[1].read()
        np.testing.assert_array_equal(rust_out, fitsio_out)


# ---------------------- GZIP fallback ------------------------------


def test_constant_tile_triggers_gzip_fallback():
    """
    A constant-value tile has zero noise, so quantize_float returns
    None and the encoder falls back to GZIP-compressing the raw
    float bytes into the GZIP_COMPRESSED_DATA column.  Round-trip
    must be exact (no quantization loss).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.full((16, 16), 3.14, dtype="f4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method="dither1"),
            )
            f[1].write(data)
            out = f[1].read()
        # Lossless fallback → exact round-trip.
        np.testing.assert_array_equal(out, data)
        # Heap is non-empty (the fallback bytes live there).
        with rustfits.FITS(fn, "r") as f:
            assert f[1].header["PCOUNT"] > 0


# ---------------------- rejections ---------------------------------


def test_quantize_on_integer_dtype_rejected():
    """quantize= with integer ZBITPIX raises (it's float-only)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="quantize"):
                f.create_image_hdu(
                    "i4",
                    (16, 16),
                    compress=rustfits.Rice1(tile_shape=(16, 16)),
                    quantize=rustfits.Quantize(),
                )


def test_quantize_without_compress_rejected():
    """quantize= without compress= raises (no compressed HDU to
    quantize against)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="quantize"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    quantize=rustfits.Quantize(),
                )


# ---------------------- DITHER_2 + NaN -----------------------------


def test_dither2_exact_zero_round_trips_exactly():
    """
    DITHER_2 reserves ZERO_VALUE_I32 for exact-zero pixels so they
    round-trip bit-exact (not through the dequant formula's noise).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(55)
        data = rng.standard_normal((32, 32)).astype("f4") * 5
        # Inject several exact-zero pixels.
        data[5, 5] = 0.0
        data[10, 20] = 0.0
        data[15, 0] = 0.0
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method="dither2"),
            )
            f[1].write(data)
            out = f[1].read()
        # Exact-zero pixels round-trip bit-exact under DITHER_2.
        assert out[5, 5] == 0.0
        assert out[10, 20] == 0.0
        assert out[15, 0] == 0.0


@pytest.mark.parametrize("method", ["no_dither", "dither1", "dither2"])
def test_nan_round_trips_for_all_methods(method):
    """
    NaN input pixels map to NULL_VALUE_I32 on encode and back to
    NaN on decode for all three dither methods (cfitsio's
    convention).  Non-NaN pixels round-trip within the usual
    quantization tolerance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(77)
        data = rng.standard_normal((32, 32)).astype("f4") * 5
        # Inject NaNs at deterministic locations.
        nan_positions = [(3, 5), (10, 20), (25, 0)]
        for r, c in nan_positions:
            data[r, c] = np.nan
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(method=method),
            )
            f[1].write(data)
            out = f[1].read()
        # NaN positions preserved exactly.
        assert np.array_equal(np.isnan(out), np.isnan(data))
        # Non-NaN pixels within quantization tolerance (loose
        # bound — bscale ~ noise/4 for this data).
        valid = ~np.isnan(data)
        assert np.abs(out[valid] - data[valid]).max() < 3.0


def test_zblank_card_emitted_for_float():
    """
    Float HDUs carry ZBLANK = -2147483647 so fitsio / astropy
    readers see the quantized null sentinel cfitsio expects.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(),
            )
            assert f[1].header["ZBLANK"] == -2147483647


def test_zblank_absent_for_integer():
    """
    Integer compressed HDUs do not get a ZBLANK card from the
    Phase 8 wiring — quantization isn't in play, and the sentinel
    would be meaningless.  (The Phase 1 ZBLANK plumbing for
    integer images via the header is a separate, unrelated path.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
            )
            assert "ZBLANK" not in f[1].header


# ---------------------- fitsio cross-check matrix ------------------


# fitsio + cfitsio agree with rustfits on the same physical
# pixel values across the supported (algorithm, method) matrix.
# Byte-exact heap agreement is NOT asserted here — fitsio's
# noise2/3/5 estimator picks a slightly different bscale on some
# inputs due to f32 vs f64 precision in the per-row diff arrays.
# The dequantized output still matches bit-for-bit because the
# encoded i32 stream + the on-disk ZSCALE/ZZERO together fully
# specify the decoded values.


@pytest.mark.parametrize(
    "algorithm_name",
    ["Rice1", "Gzip1", "Gzip2", "Hcompress1"],
)
@pytest.mark.parametrize(
    "method",
    ["no_dither", "dither1", "dither2"],
)
def test_fitsio_read_agrees_across_matrix(algorithm_name, method):
    """rustfits-written → fitsio-decoded bit-exact across the
    (algorithm, dither method) matrix."""
    algo_cls = getattr(rustfits, algorithm_name)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _smooth((32, 48), seed=88)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=algo_cls(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method=method,
                    seed=2024,
                ),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            rust_out = f[1].read()
        with fitsio.FITS(fn) as f:
            fitsio_out = f[1].read()
        if _IS_MACOS:
            np.testing.assert_allclose(
                rust_out,
                fitsio_out,
                rtol=_MACOS_FP_RTOL,
                atol=_MACOS_FP_ATOL,
            )
        else:
            np.testing.assert_array_equal(rust_out, fitsio_out)


# ------------------ cross-OS rustfits byte stability ----------------
#
# The complement to test_fitsio_read_agrees_across_matrix above.  That
# test documents that *fitsio*'s decoded floats drift on macOS (forcing
# the allclose fallback at the top of this file).  Attributing that
# drift to fitsio rather than rustfits rested on the inference that
# "rustfits's Rust dequant is deterministic and bit-matches Linux
# cfitsio" — but nothing pinned that rustfits's OWN output is identical
# across platforms.  This test supplies that missing link.
#
# Each (algorithm, dither-method) combo is pinned to two SHA-256
# digests captured on Linux x86_64: one over the written .fz bytes
# (rustfits ENCODE is OS-invariant) and one over the decoded f4 bytes
# (rustfits DECODE / dequant is OS-invariant).  Both supported
# platforms (Linux x86_64, macOS arm64) are little-endian, so the
# decoded bytes compare directly.  A macOS CI pass here therefore
# proves rustfits is byte-identical macOS↔Linux end to end, which —
# together with the matrix test showing fitsio drifts on the same
# files — isolates the documented dequant divergence entirely to
# fitsio/cfitsio (compiler codegen / Apple libm), not rustfits.
#
# Notes:
#   - The decode digest is the same across all four codecs for a given
#     dither method (only 3 distinct values): the codecs losslessly
#     compress the SAME quantized i32 stream, so the decoded output
#     depends only on the quantization, not the codec.
#   - The encode digest depends on the pinned miniz_oxide version
#     (Cargo.lock, committed) for the gzip payloads; a deliberate
#     encoder/quantizer or flate2 bump may require regenerating these.
#     To regenerate: for each combo, write the file exactly as below,
#     sha256 the .fz bytes and sha256(np.ascontiguousarray(out)
#     .tobytes()) of the read-back array.
_GOLDEN = {
    ("Rice1", "no_dither"): (
        "5b6642a8a04aae8a08dddaf6dbe11ce8f46f9e15a1efb645ca4e43f29c915161",
        "29d9f483f6ff296bb0ebce1c506c8cdceaab1a82f5f76bc25644cb41af15f2d8",
    ),
    ("Rice1", "dither1"): (
        "093b5bd4de9ff0c4b1b8b83ca5fdbb9f741f5a916457992228ed068c737cc7b7",
        "3f03700d94cb30c18934f6dc89e573664adfe02ef5e41a76de275edaa102c595",
    ),
    ("Rice1", "dither2"): (
        "8c1ef03a32f296a4d392d16ba160023dc78fda7e7fdb44c37a73f72b122e4f8e",
        "c6c867886dce1ec8c7059e60fecd0d3cd03673099f1fbd5324a9878808459407",
    ),
    ("Gzip1", "no_dither"): (
        "1054c62dc5869513edb2f27ebcce5753d5618a3d4fc67f8fab2695c52dc52975",
        "29d9f483f6ff296bb0ebce1c506c8cdceaab1a82f5f76bc25644cb41af15f2d8",
    ),
    ("Gzip1", "dither1"): (
        "bd834607acebf70774402fe33f462c859fb1b50125c6d115bf2b551b5c1e610b",
        "3f03700d94cb30c18934f6dc89e573664adfe02ef5e41a76de275edaa102c595",
    ),
    ("Gzip1", "dither2"): (
        "d16fd5158c0c75ae63f1e9950befe602cd9fb498260b811debba57ebf40d0e97",
        "c6c867886dce1ec8c7059e60fecd0d3cd03673099f1fbd5324a9878808459407",
    ),
    ("Gzip2", "no_dither"): (
        "df799ab20fa6d04b4ddc7cd63fd99a77fc740b2398fd0d9b621bcfb1854610c7",
        "29d9f483f6ff296bb0ebce1c506c8cdceaab1a82f5f76bc25644cb41af15f2d8",
    ),
    ("Gzip2", "dither1"): (
        "21e2e70ef68a38d4199788e2270ed181f2dfa41392078a03aed25467962ba25f",
        "3f03700d94cb30c18934f6dc89e573664adfe02ef5e41a76de275edaa102c595",
    ),
    ("Gzip2", "dither2"): (
        "6ba177d563318f662ad9ddb92fdc12e91eb139e3d0b056cfeef016398910d828",
        "c6c867886dce1ec8c7059e60fecd0d3cd03673099f1fbd5324a9878808459407",
    ),
    ("Hcompress1", "no_dither"): (
        "573834ec20a9d79d473e18a25a7f4445628edde6feae415a225f59b77b622b13",
        "29d9f483f6ff296bb0ebce1c506c8cdceaab1a82f5f76bc25644cb41af15f2d8",
    ),
    ("Hcompress1", "dither1"): (
        "e8e22e1400247cf9bdd7b452fe443da3a79904cd19dfacab9bf69487a44c39a2",
        "3f03700d94cb30c18934f6dc89e573664adfe02ef5e41a76de275edaa102c595",
    ),
    ("Hcompress1", "dither2"): (
        "feb74cd00e2c767758ca1ea8cd06919863d534084f8b793889a8ecfb576480b4",
        "c6c867886dce1ec8c7059e60fecd0d3cd03673099f1fbd5324a9878808459407",
    ),
}


@pytest.mark.parametrize(
    "algorithm_name",
    ["Rice1", "Gzip1", "Gzip2", "Hcompress1"],
)
@pytest.mark.parametrize(
    "method",
    ["no_dither", "dither1", "dither2"],
)
def test_rustfits_quantized_bytes_stable_across_os(algorithm_name, method):
    """
    rustfits's written and decoded bytes are byte-identical across
    platforms (pinned to Linux-captured goldens).

    Complement to test_fitsio_read_agrees_across_matrix: that test
    shows fitsio drifts on macOS; this one shows rustfits does not,
    isolating the documented dequant divergence to fitsio/cfitsio.
    """
    algo_cls = getattr(rustfits, algorithm_name)
    data = _smooth((32, 48), seed=88)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=algo_cls(tile_shape=(16, 24)),
                quantize=rustfits.Quantize(
                    level=4.0,
                    method=method,
                    seed=2024,
                ),
            )
            f[1].write(data)
        with open(fn, "rb") as fh:
            fz_sha = hashlib.sha256(fh.read()).hexdigest()
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()

    want_fz, want_dec = _GOLDEN[(algorithm_name, method)]
    dec_sha = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()
    assert fz_sha == want_fz, (
        f"encoded .fz bytes drifted from the Linux golden for "
        f"{algorithm_name}/{method}: rustfits encode is not "
        f"OS-invariant (got {fz_sha})"
    )
    assert dec_sha == want_dec, (
        f"decoded f4 bytes drifted from the Linux golden for "
        f"{algorithm_name}/{method}: rustfits dequant is not "
        f"OS-invariant (got {dec_sha})"
    )


def test_default_seed_emits_one():
    """Quantize(seed=0) → on-disk ZDITHER0=1 (cfitsio default)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                quantize=rustfits.Quantize(seed=0),
            )
            assert f[1].header["ZDITHER0"] == 1
