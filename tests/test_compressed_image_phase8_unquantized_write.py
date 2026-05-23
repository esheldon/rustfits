"""
ZIMAGE Phase 8: unquantized-float compressed image writes
(quantize=None or omitted kwarg).

When a float HDU is created without a Quantize config (or with
quantize=None), every tile is stored as raw float bytes
compressed through GZIP_1 or GZIP_2 — no quantization, no
precision loss.  Matches astropy's quantize_level=0 layout:
single COMPRESSED_DATA column with raw GZIP'd float bytes, no
ZSCALE/ZZERO, no ZQUANTIZ keyword.

Tests cover:
    - Schema validation: TFIELDS=1, COMPRESSED_DATA is the
      only column, ZQUANTIZ + ZDITHER0 + ZBLANK absent.
    - Round-trip via same-handle read and post-reopen read.
    - Cross-check vs astropy + fitsio: rustfits writes →
      both libraries read back bit-exact; reverse direction
      also bit-exact.
    - Dtype matrix (f4, f8) × algorithm (Gzip1, Gzip2).
    - Non-last HDU growth: a compressed HDU followed by a
      second HDU; the later HDU's offsets must shift to make
      room for the heap.
    - omit-quantize semantics: not passing quantize= at all
      is equivalent to passing quantize=None.
    - Rejections: Rice1, Hcompress1, Plio1 with quantize=None
      (or omitted) raise clear errors.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")
fitsio = pytest.importorskip("fitsio")


def _rand(shape, dtype, seed=0):
    """
    Deterministic random float test array, scaled so values span
    a sensible dynamic range without being all-positive (which
    would mask sign-bit handling).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype) * 17.0 - 3.5


# ---------------------- schema ------------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize(
    "AlgoCls,zcmptype",
    [(rustfits.Gzip1, "GZIP_1"), (rustfits.Gzip2, "GZIP_2")],
)
def test_schema_single_column_no_zquantiz(dtype, AlgoCls, zcmptype):
    """
    A float HDU created with quantize=None must emit a
    single-column BINTABLE (TFIELDS=1, COMPRESSED_DATA only) and
    skip ZQUANTIZ / ZDITHER0 / ZBLANK.  Matches astropy's
    quantize_level=0 schema bit-for-bit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                (20, 30),
                compress=AlgoCls(tile_shape=(10, 15)),
                quantize=None,
            )
            hdr = f[1].header
            assert hdr["TFIELDS"] == 1
            assert hdr["ZCMPTYPE"] == zcmptype
            assert "ZQUANTIZ" not in hdr
            assert "ZDITHER0" not in hdr
            assert "ZBLANK" not in hdr
            assert hdr["TTYPE1"] == "COMPRESSED_DATA"


def test_omit_quantize_kwarg_equivalent_to_none():
    """
    Not passing quantize= at all on a float HDU is equivalent to
    passing quantize=None: the schema is single-column and the
    Z-quantization keywords are absent.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (16, 16),
                compress=rustfits.Gzip1(tile_shape=(16, 16)),
            )
            hdr = f[1].header
            assert hdr["TFIELDS"] == 1
            assert "ZQUANTIZ" not in hdr


# ---------------------- round-trip --------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("AlgoCls", [rustfits.Gzip1, rustfits.Gzip2])
def test_round_trip_bit_exact(dtype, AlgoCls):
    """
    Lossless round-trip across the dtype × algorithm matrix:
    same-handle read and post-reopen read both bit-exact equal
    to the input.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand((30, 40), dtype, seed=11)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=AlgoCls(tile_shape=(15, 20)),
                quantize=None,
            )
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype
        assert reopen.dtype == data.dtype


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((40,), (40,)),  # 1-D single tile
        ((100,), (32,)),  # 1-D, four tiles with edge
        ((32, 32), (16, 16)),  # 2-D square
        ((50, 70), (16, 24)),  # 2-D non-square with edge tiles
        ((4, 5, 6), (2, 5, 6)),  # 3-D
    ],
)
def test_round_trip_shape_matrix(shape, tile):
    """
    Shape coverage: 1-D, 2-D, 3-D with both whole-image tiles
    and partial edge tiles must all round-trip bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand(shape, "f4", seed=22)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                shape,
                compress=rustfits.Gzip1(tile_shape=tile),
                quantize=None,
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- cross-checks ------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize(
    "AlgoCls,ap_algo",
    [
        (rustfits.Gzip1, "GZIP_1"),
        (rustfits.Gzip2, "GZIP_2"),
    ],
)
def test_rustfits_writes_astropy_reads(dtype, AlgoCls, ap_algo):
    """
    A rustfits-written quantize=None file must read back
    bit-exact via astropy (proving the schema is FITS-conformant
    and astropy's reader handles the missing ZQUANTIZ).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand((30, 40), dtype, seed=33)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=AlgoCls(tile_shape=(15, 20)),
                quantize=None,
            )
            f[1].write(data)
        with astropy_fits.open(fn) as h:
            ap_data = h[1].data
        np.testing.assert_array_equal(ap_data, data)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize(
    "AlgoCls,ap_algo",
    [
        (rustfits.Gzip1, "GZIP_1"),
        (rustfits.Gzip2, "GZIP_2"),
    ],
)
def test_astropy_writes_rustfits_reads(dtype, AlgoCls, ap_algo):
    """
    The mirror direction: astropy writes a quantize_level=0 file,
    rustfits reads it bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand((30, 40), dtype, seed=44)
        hdu = astropy_fits.CompImageHDU(
            data,
            compression_type=ap_algo,
            tile_shape=(15, 20),
            quantize_level=0.0,
        )
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
            fn, overwrite=True
        )
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_reads_rustfits_written():
    """
    fitsio should also read our quantize=None output bit-exact
    (cfitsio under the hood; another standards-conformance check).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand((30, 40), "f4", seed=55)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(15, 20)),
                quantize=None,
            )
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth -----------------------


def test_non_last_hdu_growth():
    """
    Write a compressed float HDU, then a second HDU after it,
    then trigger the compressed write — the second HDU's offsets
    must shift to make room for the heap, and post-reopen reads
    of both HDUs must still work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _rand((40, 60), "f4", seed=66)
        later_data = np.arange(100, dtype="i4").reshape(10, 10)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                comp_data.shape,
                compress=rustfits.Gzip2(tile_shape=(20, 30)),
                quantize=None,
                extname="COMP",
            )
            f.create_image_hdu(
                "i4",
                later_data.shape,
                extname="LATER",
            )
            f[2].write(later_data)
            # Now write the compressed HDU — heap grows; LATER
            # offsets must shift.
            f[1].write(comp_data)
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


# ---------------------- compression accessor ----------------------


@pytest.mark.parametrize(
    "AlgoCls,zcmptype",
    [(rustfits.Gzip1, "GZIP_1"), (rustfits.Gzip2, "GZIP_2")],
)
def test_compression_accessor_round_trips(AlgoCls, zcmptype):
    """
    .compression on a reopened quantize=None HDU returns the
    matching Gzip1/Gzip2 config with the correct tile_shape.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (20, 30),
                compress=AlgoCls(tile_shape=(10, 15)),
                quantize=None,
            )
            f[1].write(np.zeros((20, 30), dtype="f4"))
        with rustfits.FITS(fn, "r") as f:
            comp = f[1].compression
            assert comp.zcmptype == zcmptype
            assert comp.tile_shape == (10, 15)


# ---------------------- rejections --------------------------------


@pytest.mark.parametrize(
    "AlgoCls,name",
    [
        (lambda: rustfits.Rice1(tile_shape=(16, 16)), "Rice1"),
        (
            lambda: rustfits.Hcompress1(tile_shape=(16, 16)),
            "Hcompress1",
        ),
    ],
)
def test_non_gzip_with_quantize_none_rejected(AlgoCls, name):
    """
    Rice1 and Hcompress1 paired with quantize=None must raise a
    ValueError pointing at Gzip1/Gzip2.  These algorithms silently
    corrupt unquantized float data (the H-transform and Rice
    coding are integer-only operations).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="unquantized float"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    compress=AlgoCls(),
                    quantize=None,
                )


def test_omitted_quantize_with_non_gzip_rejected():
    """
    Same rejection as quantize=None — omitting the kwarg
    entirely should also reject Rice1 / Hcompress1 (since the
    behavior is the same as quantize=None).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="unquantized float"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    compress=rustfits.Rice1(tile_shape=(16, 16)),
                )


def test_plio_with_quantize_none_rejected_with_plio_message():
    """
    PLIO + float gets PLIO's specific "does not support float"
    error message rather than the generic "unquantized float
    requires Gzip" message — PLIO + float never works regardless
    of quantize, so the PLIO-specific guidance is more useful.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="PLIO"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    compress=rustfits.Plio1(tile_shape=(16, 16)),
                    quantize=None,
                )


def test_quantize_object_still_works_for_quantized_floats():
    """
    Sanity check: explicit quantize=Quantize(...) still produces
    the quantized 4-column schema (i.e. our API change didn't
    break the existing path).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _rand((20, 30), "f4", seed=77)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                data.shape,
                compress=rustfits.Gzip1(tile_shape=(10, 15)),
                quantize=rustfits.Quantize(method="dither1"),
            )
            f[1].write(data)
            hdr = f[1].header
            assert hdr["TFIELDS"] == 4
            assert hdr["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
            assert "ZDITHER0" in hdr
            assert "ZBLANK" in hdr


def test_quantize_none_method_string_rejected():
    """
    The old API spelling Quantize(method='none') is removed — it
    should now raise from Quantize.__init__.
    """
    with pytest.raises(ValueError, match="unknown method"):
        rustfits.Quantize(method="none")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
