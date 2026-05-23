"""
ZIMAGE Phase 7 follow-up: Plio1 compressed image writes.

Bit-exact port of cfitsio's `pl_p2li` from `<cfitsio>/pliocomp.c`.
The SPP/f2c goto soup is replaced by a single linear loop in
`src/zimage/plio.rs::encode_plio_i32`.

Tests cover:
    - Lossless round-trip (same-handle + post-reopen) across u1/i2/i4
    - **Byte-exact heap agreement with fitsio** for both single-tile
      and multi-tile cases — strongest correctness signal for the
      encoder.
    - Mask-style fixtures (mostly zero with non-zero runs), plus
      degenerate cases: all-zeros, all-solid-one-value, sparse.
    - Bidirectional cross-check with fitsio (rustfits-written read
      by fitsio AND fitsio-written read by rustfits).
    - Non-last HDU growth (heap shifts later HDU forward).
    - Mixed-algorithm file: Plio1 + Gzip2.
    - Rejection paths: float ZBITPIX, i8 dtype (no 64-bit PLIO
      variant), negative pixels (PLIO is non-negative-only),
      shape mismatch, start kwarg.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _mask(shape, seed=0, max_val=10, dtype="i4"):
    """
    Mask-style array: mostly zeros, with a few non-zero runs.  This
    is the typical PLIO_1 input (the algorithm is built for IRAF
    pixel lists, which are mostly-zero masks with runs of small
    positive values).
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=dtype)
    # Sprinkle 10 random rectangles.
    if len(shape) == 2:
        for _ in range(10):
            r0 = rng.integers(0, shape[0])
            r1 = rng.integers(r0, shape[0] + 1)
            c0 = rng.integers(0, shape[1])
            c1 = rng.integers(c0, shape[1] + 1)
            arr[r0:r1, c0:c1] = rng.integers(1, max_val + 1)
    return arr


def _heap_bytes(fn):
    """Extract the heap bytes of HDU 1 from a tile-compressed FITS file."""
    with rustfits.FITS(fn, "r") as f:
        hdr = f[1].header
        pcount = hdr["PCOUNT"]
        naxis1 = hdr["NAXIS1"]
        naxis2 = hdr["NAXIS2"]
    with open(fn, "rb") as fh:
        raw = fh.read()
    i = 2880
    data_start = None
    while i < len(raw):
        block = raw[i : i + 2880]
        if b"END     " in block:
            data_start = i + 2880
            break
        i += 2880
    assert data_start is not None
    main_bytes = naxis1 * naxis2
    heap_start = data_start + main_bytes
    return raw[heap_start : heap_start + pcount]


# ---------------------- accessors ----------------------------------


def test_accessors_after_create():
    """
    create_image_hdu(..., compress=Plio1(...)) produces a
    CompressedImageHDU with ZCMPTYPE=PLIO_1 and TFORM1='1PI'
    (i16 inner type, not '1PB' like the other algorithms).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (32, 48), compress=cfg, extname="MASK")
            hdu = f[1]
            assert type(hdu).__name__ == "CompressedImageHDU"
            assert hdu.shape == (32, 48)
            assert hdu.dtype == np.int32
            assert hdu.compression.zcmptype == "PLIO_1"
            assert hdu.compression.tile_shape == (16, 16)
            assert hdu.extname == "MASK"
            assert hdu.header["TFORM1"].strip() == "1PI"


def test_plio1_repr_and_kwargs():
    """Plio1 config-object surface: tile_shape + heap_format getters."""
    cfg = rustfits.Plio1(tile_shape=(16, 16), heap_format="Q")
    assert cfg.tile_shape == (16, 16)
    assert cfg.heap_format == "Q"
    assert cfg.zcmptype == "PLIO_1"
    assert repr(cfg).startswith("Plio1(")


# ---------------------- round-trip dtype matrix --------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_round_trip_dtype_matrix_lossless(dtype):
    """Lossless round-trip across the supported integer dtype set."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _mask((32, 48), seed=11, max_val=20, dtype=dtype)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(16, 24))
            f.create_image_hdu(dtype, data.shape, compress=cfg)
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype


# ---------------------- shape matrix -------------------------------


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((32, 32), (16, 16)),
        ((48, 72), (16, 24)),
        ((40, 60), (40, 60)),  # single tile
        ((32, 48), (16, 24)),
        ((64, 64), (32, 32)),
    ],
)
def test_round_trip_shape_matrix(shape, tile):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _mask(shape, seed=22, max_val=5)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=tile)
            f.create_image_hdu("i4", shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_round_trip_default_tile_shape():
    """tile_shape=None defaults to FITS-convention row tiles."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _mask((20, 64), seed=33)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", data.shape, compress=rustfits.Plio1())
            f[1].write(data)
            assert f[1].compression.tile_shape == (1, 64)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- degenerate cases ---------------------------


def test_all_zeros_round_trip():
    """All-zero mask: encoder emits only zero-run words."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.zeros((32, 32), dtype="i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_all_solid_round_trip():
    """All-same-non-zero-value mask: encoder emits solid-pv runs."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.full((32, 32), 7, dtype="i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_large_pv_value_round_trip():
    """Large pv values (> 4095) exercise the two-word set-pv path."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.zeros((16, 16), dtype="i4")
        data[2:5, 3:8] = 1_000_000  # well above 4095
        data[10:12, 10:14] = 5000
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(16, 16)),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_single_pixel_writes_round_trip():
    """
    Single-pixel non-zero writes exercise the opcode-6/7 combo
    (set-pv-and-write-single-pixel).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.zeros((16, 16), dtype="i4")
        data[0, 0] = 5
        data[5, 7] = 9
        data[10, 3] = 2
        data[15, 15] = 4
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(16, 16)),
            )
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- byte-exact cross-check ---------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_byte_exact_heap_matches_fitsio(dtype):
    """
    Heap bytes produced by rustfits must equal cfitsio's output
    byte-for-byte on the same input.  Catches any drift in the
    opcode dispatch, single-pixel combinations, or wire format.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = _mask((32, 48), seed=44, max_val=10, dtype=dtype)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Plio1(tile_shape=(32, 48)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="PLIO_1",
            tile_dims=(32, 48),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


def test_byte_exact_heap_multi_tile():
    """Multi-tile case: byte-exact across every tile."""
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = _mask((50, 70), seed=55, max_val=5)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(16, 24)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="PLIO_1",
            tile_dims=(16, 24),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


def test_byte_exact_all_zeros():
    """All-zero tiles: trivial encoding must match fitsio byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = np.zeros((32, 32), dtype="i4")
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="PLIO_1",
            tile_dims=(32, 32),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


def test_byte_exact_solid_value():
    """Solid-value tiles: encoder must match cfitsio byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = np.full((32, 32), 3, dtype="i4")
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="PLIO_1",
            tile_dims=(32, 32),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


# ---------------------- bidirectional fitsio interop ---------------


def test_rustfits_written_fitsio_read_matches():
    """rustfits-written → fitsio-read bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _mask((40, 60), seed=66, max_val=8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Plio1(tile_shape=(20, 30)),
            )
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            assert f[1].read_header().get("ZCMPTYPE") == "PLIO_1"
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_written_rustfits_read_matches():
    """fitsio-written → rustfits-read bit-exact (regression guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _mask((40, 60), seed=77, max_val=8)
        fitsio.write(fn, data, compress="PLIO_1", tile_dims=(20, 30))
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth ------------------------


def test_compressed_write_shifts_later_hdus():
    """
    PLIO HDU followed by another HDU; heap write must shift the
    later HDU's offsets, and post-reopen reads of both still work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _mask((64, 64), seed=88, max_val=5)
        later_data = _mask((10, 10), seed=99, max_val=3, dtype="i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                comp_data.shape,
                compress=rustfits.Plio1(tile_shape=(16, 16)),
                extname="MASK",
            )
            f.create_image_hdu("i2", later_data.shape, extname="LATER")
            f[2].write(later_data)
            f[1].write(comp_data)
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "MASK"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


# ---------------------- mixed-algorithm file -----------------------


def test_mixed_plio_and_gzip2_in_one_file():
    """One Plio1 + one Gzip2 HDU in the same file."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        d1 = _mask((32, 48), seed=111, max_val=5)
        rng = np.random.default_rng(222)
        d2 = rng.integers(-10000, 10000, (48, 32), dtype="i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                d1.shape,
                compress=rustfits.Plio1(tile_shape=(16, 24)),
                extname="MASK",
            )
            f.create_image_hdu(
                "i2",
                d2.shape,
                compress=rustfits.Gzip2(tile_shape=(24, 16)),
                extname="IMG",
            )
            f[1].write(d1)
            f[2].write(d2)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].compression.zcmptype == "PLIO_1"
            assert f[2].compression.zcmptype == "GZIP_2"
            np.testing.assert_array_equal(f[1].read(), d1)
            np.testing.assert_array_equal(f[2].read(), d2)


# ---------------------- rejections ---------------------------------


def test_i8_plio_rejected():
    """i8 (bitpix=64) rejected — PLIO has no 64-bit variant."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="64-bit"):
                f.create_image_hdu(
                    "i8",
                    (16, 16),
                    compress=rustfits.Plio1(),
                )


def test_float_compress_rejected():
    """
    PLIO + float rejected.  PLIO encodes mask data with
    non-negative integer values; quantize_float produces an i32
    stream with negative values (bzero shifts the range), which
    PLIO can't represent.  We reject at create time so the user
    gets a clear error instead of a downstream "pixel is negative"
    failure from the encoder.  Other algorithms now accept float
    via quantize=; PLIO is the one exception.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="float"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    compress=rustfits.Plio1(tile_shape=(16, 16)),
                )


def test_unsigned_trick_dtype_rejected():
    """
    PLIO + unsigned-int trick (i1/u2/u4/u8) is rejected with the
    PLIO-specific error: the reverse XOR produces signed stored
    values that include negatives, which PLIO's non-negative-only
    encoder can't represent.  Other algorithms now accept these
    dtypes — see tests/test_compressed_image_unsigned_trick.py.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="PLIO"):
                f.create_image_hdu(
                    "u4",
                    (16, 16),
                    compress=rustfits.Plio1(tile_shape=(16, 16)),
                )


def test_negative_pixel_rejected():
    """
    PLIO_1 is built around pv += increments from a non-negative
    starting state; negative pixels can't be represented.  Encoder
    rejects with a clear error rather than silently clamping.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = np.zeros((16, 16), dtype="i4")
            data[3, 5] = -1
            with pytest.raises(ValueError, match="negative"):
                f[1].write(data)


def test_value_too_large_rejected():
    """
    PLIO_1's two-word set-pv encoding tops out at 2^27 - 1.  Values
    above that are rejected at encode time.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = np.zeros((16, 16), dtype="i4")
            data[0, 0] = 2**28  # > 2^27 - 1
            with pytest.raises(ValueError, match="exceeds"):
                f[1].write(data)


def test_input_shape_mismatch_rejected():
    """write(data) with wrong shape raises; subsequent good write works."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            wrong = np.zeros((8, 8), dtype="i4")
            with pytest.raises(ValueError, match="shape"):
                f[1].write(wrong)
            good = _mask((16, 16), seed=333, max_val=4)
            f[1].write(good)
            np.testing.assert_array_equal(f[1].read(), good)


def test_start_kwarg_rejected():
    """start= is not supported on compressed-image writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Plio1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = _mask((16, 16), seed=444, max_val=4)
            with pytest.raises(NotImplementedError, match="start="):
                f[1].write(data, start=[0, 0])
