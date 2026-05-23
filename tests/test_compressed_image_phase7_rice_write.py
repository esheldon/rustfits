"""
ZIMAGE Phase 7 follow-up: Rice1 compressed image writes.

Bit-exact port of cfitsio's `fits_rcomp` / `_short` / `_byte`
encoder family.  Tests cover:
    - Round-trip via same-handle and post-reopen read
    - **Byte-exact heap agreement with fitsio (cfitsio)** — the
      strongest correctness signal; catches any drift in the
      per-block fs heuristic or unary/raw bit packing.
    - Integer dtype matrix (u1, i2, i4).  i8 is rejected; see
      below.
    - Shape matrix (1-D, 2-D square + non-square, 3-D, single-tile,
      multi-tile)
    - Default tile shape (FITS-convention row tiles)
    - Custom blocksize parameter
    - Mixed-algorithm file: Rice1 + Gzip2 in one file
    - Non-last HDU growth (heap shifts later HDU forward)
    - Float ZBITPIX (-32/-64) rejected with a Phase 8 NotImplemented
    - **i8 (BYTEPIX=8) rejected**: no canonical FITS writer emits
      such files; cfitsio refuses outright, astropy silently
      downcasts to i32 (lossy).  rustfits rejects with a clean
      NotImplementedError pointing at Gzip2 instead.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _seq(shape, dtype):
    """
    Deterministic test array of `shape` and `dtype`.

    Arange modulo the dtype's range so values fit and the
    sequence has enough variation to exercise the encoder's
    per-block fs heuristic.
    """
    n = int(np.prod(shape))
    maxv = {
        np.dtype("u1"): 255,
        np.dtype("i2"): 32767,
        np.dtype("i4"): 1_000_000,
    }[np.dtype(dtype)]
    arr = (np.arange(n) % (maxv + 1)).astype(dtype)
    return arr.reshape(shape)


def _heap_bytes(fn):
    """
    Extract the heap bytes of HDU 1 from a tile-compressed FITS
    file.  Used for byte-exact comparison between fitsio-written
    and rustfits-written outputs.
    """
    with rustfits.FITS(fn, "r") as f:
        hdr = f[1].header
        pcount = hdr["PCOUNT"]
        naxis1 = hdr["NAXIS1"]
        naxis2 = hdr["NAXIS2"]
    with open(fn, "rb") as fh:
        raw = fh.read()
    # Primary header is 2880 bytes; find HDU 1 header end.
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
    create_image_hdu(..., compress=Rice1(...)) produces a
    CompressedImageHDU with ZCMPTYPE=RICE_1 and BLOCKSIZE/BYTEPIX
    ZNAMEn cards.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (32, 48), compress=cfg, extname="SCI")
            hdu = f[1]
            assert type(hdu).__name__ == "CompressedImageHDU"
            assert hdu.shape == (32, 48)
            assert hdu.dtype == np.int32
            assert hdu.compression.zcmptype == "RICE_1"
            assert hdu.compression.tile_shape == (16, 16)
            assert hdu.extname == "SCI"
            assert hdu.header["PCOUNT"] == 0
            # ZNAMEn / ZVALn cards for RICE parameters.
            names = {
                hdu.header[f"ZNAME{n}"]: hdu.header[f"ZVAL{n}"] for n in (1, 2)
            }
            assert names["BLOCKSIZE"] == 32
            assert names["BYTEPIX"] == 4  # i4 → 4 bytes


def test_rice1_repr_and_kwargs():
    """
    Rice1 config object surface: tile_shape + heap_format +
    blocksize getters, repr.
    """
    cfg = rustfits.Rice1(
        tile_shape=(16, 16),
        heap_format="Q",
        blocksize=64,
    )
    assert cfg.tile_shape == (16, 16)
    assert cfg.heap_format == "Q"
    assert cfg.blocksize == 64
    assert repr(cfg).startswith("Rice1(")


def test_rice1_blocksize_validation():
    """blocksize must be > 0."""
    with pytest.raises(ValueError, match="blocksize"):
        rustfits.Rice1(blocksize=0)
    with pytest.raises(ValueError, match="blocksize"):
        rustfits.Rice1(blocksize=-1)


# ---------------------- round-trip dtype matrix --------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_round_trip_dtype_matrix(dtype):
    """
    Write + same-handle read + post-reopen read all bit-exact
    across the supported integer dtype set.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((32, 48), dtype)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=(16, 24))
            f.create_image_hdu(dtype, data.shape, compress=cfg)
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype


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
        # Tile smaller than blocksize — exercises the partial-block path.
        ((10, 10), (10, 10)),
    ],
)
def test_round_trip_shape_matrix(shape, tile):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq(shape, "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=tile)
            f.create_image_hdu("i4", shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_round_trip_default_tile_shape():
    """tile_shape=None defaults to FITS-convention row tiles."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((20, 64), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i2", data.shape, compress=rustfits.Rice1())
            f[1].write(data)
            assert f[1].compression.tile_shape == (1, 64)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_round_trip_custom_blocksize():
    """Non-default blocksize round-trips."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((32, 32), "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=(32, 32), blocksize=16)
            f.create_image_hdu("i4", data.shape, compress=cfg)
            f[1].write(data)
            # BLOCKSIZE ZVAL must reflect the config.
            names = {
                f[1].header[f"ZNAME{n}"]: f[1].header[f"ZVAL{n}"]
                for n in (1, 2)
            }
            assert names["BLOCKSIZE"] == 16
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- byte-exact cross-check ---------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_byte_exact_heap_matches_fitsio(dtype):
    """
    The heap bytes produced by rustfits must equal cfitsio's
    output byte-for-byte on the same input.  Catches any drift in
    the fs heuristic, ZigZag formula, or bit-packing logic.
    Strongest correctness signal we have for the RICE encoder.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fitsio = os.path.join(tmp, "fitsio.fits.fz")
        data = _seq((32, 48), dtype)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Rice1(tile_shape=(32, 48)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fitsio,
            data,
            compress="RICE_1",
            tile_dims=(32, 48),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fitsio)


def test_byte_exact_heap_multi_tile():
    """
    Multi-tile case: same byte-exact agreement across every tile
    in a non-trivial layout (edge tiles, multiple blocks per tile).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fitsio = os.path.join(tmp, "fitsio.fits.fz")
        # 50x70 with 16x24 tiles → 4x3 = 12 tiles, including edge tiles.
        data = _seq((50, 70), "i4")
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fitsio,
            data,
            compress="RICE_1",
            tile_dims=(16, 24),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fitsio)


def test_byte_exact_heap_low_entropy():
    """
    Constant-value tile triggers the low-entropy branch (fs=0 +
    pixelsum=0); must match cfitsio's output byte-for-byte.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fitsio = os.path.join(tmp, "fitsio.fits.fz")
        data = np.full((32, 32), 42, dtype=np.int32)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fitsio,
            data,
            compress="RICE_1",
            tile_dims=(32, 32),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fitsio)


def test_byte_exact_heap_high_entropy():
    """
    Random data forces the high-entropy raw branch (fs>=fsmax)
    for many blocks; must still match cfitsio byte-for-byte.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fitsio = os.path.join(tmp, "fitsio.fits.fz")
        rng = np.random.default_rng(123)
        data = rng.integers(
            np.iinfo(np.int32).min // 2,
            np.iinfo(np.int32).max // 2,
            (32, 32),
            dtype=np.int32,
        )
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(32, 32)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fitsio,
            data,
            compress="RICE_1",
            tile_dims=(32, 32),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fitsio)


# ---------------------- bidirectional fitsio read/write ------------


def test_rustfits_written_fitsio_read_matches():
    """rustfits-written → fitsio-read bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(20, 30)),
            )
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            assert f[1].read_header().get("ZCMPTYPE") == "RICE_1"
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_written_rustfits_read_matches():
    """fitsio-written → rustfits-read bit-exact (regression guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _seq((40, 60), "i4")
        fitsio.write(fn, data, compress="RICE_1", tile_dims=(20, 30))
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth ------------------------


def test_compressed_write_shifts_later_hdus():
    """
    Compressed HDU followed by another HDU; heap write must shift
    the later HDU's offsets, and post-reopen reads of both still
    work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _seq((64, 64), "i4")
        later_data = _seq((10, 10), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                comp_data.shape,
                compress=rustfits.Rice1(tile_shape=(16, 16)),
                extname="COMP",
            )
            f.create_image_hdu(
                "i2",
                later_data.shape,
                extname="LATER",
            )
            f[2].write(later_data)
            f[1].write(comp_data)
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


# ---------------------- mixed-algorithm file -----------------------


def test_mixed_rice_and_gzip2_in_one_file():
    """One Rice1 + one Gzip2 HDU in the same file."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        d1 = _seq((32, 48), "i4")
        d2 = _seq((48, 32), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                d1.shape,
                compress=rustfits.Rice1(tile_shape=(16, 24)),
                extname="R1",
            )
            f.create_image_hdu(
                "i2",
                d2.shape,
                compress=rustfits.Gzip2(tile_shape=(24, 16)),
                extname="G2",
            )
            f[1].write(d1)
            f[2].write(d2)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].compression.zcmptype == "RICE_1"
            assert f[2].compression.zcmptype == "GZIP_2"
            np.testing.assert_array_equal(f[1].read(), d1)
            np.testing.assert_array_equal(f[2].read(), d2)


# ---------------------- rejections ---------------------------------


def test_i8_rice_rejected():
    """
    BYTEPIX=8 (i8 dtype, bitpix=64) is rejected at create time:
    no canonical FITS writer produces such files, so they'd be
    unreadable outside rustfits.  cfitsio refuses outright;
    astropy silently downcasts to i32 (lossy).  rustfits raises
    NotImplementedError pointing at Gzip2 instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="64-bit"):
                f.create_image_hdu(
                    "i8",
                    (16, 16),
                    compress=rustfits.Rice1(),
                )


# test_float_compress_rejected: removed in Phase 8 commit 2 —
# float-compressed writes are now supported via the new
# quantize= parameter.  See
# tests/test_compressed_image_phase8_quantize_write.py for the
# float round-trip coverage.

def test_unsigned_trick_dtype_rejected():
    """u2/u4/u8/i1 not yet supported on the compressed-write side."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="unsigned"):
                f.create_image_hdu(
                    "u4",
                    (16, 16),
                    compress=rustfits.Rice1(tile_shape=(16, 16)),
                )


def test_input_shape_mismatch_rejected():
    """write(data) with wrong shape raises; subsequent good write works."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            wrong = np.arange(8 * 8, dtype="i4").reshape(8, 8)
            with pytest.raises(ValueError, match="shape"):
                f[1].write(wrong)
            good = _seq((16, 16), "i4")
            f[1].write(good)
            np.testing.assert_array_equal(f[1].read(), good)


def test_start_kwarg_rejected():
    """start= is not supported on compressed-image writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Rice1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = _seq((16, 16), "i4")
            with pytest.raises(NotImplementedError, match="start="):
                f[1].write(data, start=[0, 0])
