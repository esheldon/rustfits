"""
ZIMAGE Phase 7 follow-up: Hcompress1 compressed image writes.

Bit-exact port of cfitsio's `fits_hcompress` family (htrans, digitize,
encode, doencode, qtree_encode, qtree_onebit, qtree_reduce, bufcopy,
write_bdirect plus the output_nbits/nybble/nnybble bit-output state).
Tests cover:
    - Lossless round-trip (same-handle + post-reopen) across u1/i2/i4
    - Lossy round-trip with various scale values
    - Smooth=True round-trip (smoothing happens on read; encoded bytes
      unchanged, but the SMOOTH ZNAME card must be written)
    - **Byte-exact heap agreement with fitsio (cfitsio)** on single-
      tile and on multi-tile cases where the image divides evenly.
      Tile shapes that leave a < 4 pixel edge are rejected upstream
      (see test_thin_edge_tile_rejected), so byte-exact agreement is
      the right invariant for everything we do accept.
    - Bidirectional cross-check with fitsio (rustfits → fitsio,
      fitsio → rustfits)
    - Non-last HDU growth (heap shifts later HDU forward)
    - Mixed-algorithm file (Hcompress1 + Gzip2)
    - Rejection paths: float ZBITPIX, i8 dtype, unsigned trick, 1-D /
      3-D images, shape mismatch, start kwarg, dim<4, tile<4, thin
      edge tile (with cfitsio-style suggested fix in the error)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _data(shape, dtype, seed=0):
    """Deterministic test data with enough variation to exercise the
    H-transform across bit planes."""
    rng = np.random.default_rng(seed)
    if dtype == "u1":
        return rng.integers(0, 200, shape, dtype="u1")
    if dtype == "i2":
        return rng.integers(-20000, 20000, shape, dtype="i2")
    if dtype == "i4":
        return rng.integers(-1_000_000, 1_000_000, shape, dtype="i4")
    raise ValueError(dtype)


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
    create_image_hdu(..., compress=Hcompress1(...)) produces a
    CompressedImageHDU with ZCMPTYPE=HCOMPRESS_1 and the SCALE +
    SMOOTH ZNAMEn cards.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(16, 16))
            f.create_image_hdu("i4", (32, 48), compress=cfg, extname="SCI")
            hdu = f[1]
            assert type(hdu).__name__ == "CompressedImageHDU"
            assert hdu.shape == (32, 48)
            assert hdu.dtype == np.int32
            assert hdu.compression.zcmptype == "HCOMPRESS_1"
            assert hdu.compression.tile_shape == (16, 16)
            assert hdu.extname == "SCI"
            assert hdu.header["PCOUNT"] == 0
            names = {
                hdu.header[f"ZNAME{n}"]: hdu.header[f"ZVAL{n}"] for n in (1, 2)
            }
            assert names["SCALE"] == 0
            assert names["SMOOTH"] == 0


def test_hcompress1_repr_and_kwargs():
    """Hcompress1 config-object surface."""
    cfg = rustfits.Hcompress1(
        tile_shape=(16, 16),
        heap_format="Q",
        scale=4,
        smooth=True,
    )
    assert cfg.tile_shape == (16, 16)
    assert cfg.heap_format == "Q"
    assert cfg.scale == 4
    assert cfg.smooth is True
    assert repr(cfg).startswith("Hcompress1(")


def test_hcompress1_scale_validation():
    """scale must be >= 0."""
    with pytest.raises(ValueError, match="scale"):
        rustfits.Hcompress1(scale=-1)


def test_hcompress1_tile_must_be_2d():
    """HCOMPRESS_1 is a 2-D algorithm; non-2D tile_shape rejected."""
    with pytest.raises(ValueError, match="2-D"):
        rustfits.Hcompress1(tile_shape=(16,))
    with pytest.raises(ValueError, match="2-D"):
        rustfits.Hcompress1(tile_shape=(8, 8, 8))


# ---------------------- round-trip dtype matrix --------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_round_trip_dtype_matrix_lossless(dtype):
    """Lossless round-trip across the supported integer dtype set."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((32, 48), dtype)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(16, 24))
            f.create_image_hdu(dtype, data.shape, compress=cfg)
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == data.dtype


# ---------------------- shape matrix (always divisible) ------------


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((32, 32), (16, 16)),  # 2-D square, multi-tile
        ((48, 72), (16, 24)),  # 2-D non-square, no edge tiles
        ((40, 60), (40, 60)),  # 2-D single tile
        ((32, 48), (16, 24)),  # 2x2 tiles
        ((64, 64), (32, 32)),  # 2x2 large tiles
    ],
)
def test_round_trip_shape_matrix(shape, tile):
    """Shapes chosen so image % tile == 0 along both axes (no edge
    tiles).  Multi-tile cases with valid edge tiles (remain >= 4) are
    also valid but harder to enumerate; the byte-exact tests below
    cover them."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data(shape, "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=tile)
            f.create_image_hdu("i4", shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_round_trip_default_tile_shape_small_image():
    """
    tile_shape=None on a small image (NAXIS2 <= 30) defaults to the
    whole image as a single tile (matches cfitsio's small-image
    heuristic).  20x64 → single (20, 64) tile.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((20, 64), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                data.shape,
                compress=rustfits.Hcompress1(),
            )
            f[1].write(data)
            assert f[1].compression.tile_shape == (20, 64)
            assert f[1].n_tiles == 1
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_round_trip_default_tile_shape_large_image():
    """
    tile_shape=None on a larger image (NAXIS2 > 30) defaults to
    16-row stripes (cfitsio's preferred stripe height when it leaves
    a valid edge tile).  200x64 → tile (16, 64), n_tiles = 13
    (200 / 16 = 12 full + 1 edge of 8 rows, which is >= 4).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((200, 64), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(),
            )
            f[1].write(data)
            assert f[1].compression.tile_shape == (16, 64)
            assert f[1].n_tiles == 13
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_round_trip_default_tile_shape_fallback():
    """
    tile_shape=None on a shape where 16 leaves a bad edge falls
    through to the next preferred value.  34x64: 34 > 30, try 16:
    34 % 16 = 2 (bad), try 24: 34 % 24 = 10 (good) → tile (24, 64).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((34, 64), "i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(),
            )
            f[1].write(data)
            assert f[1].compression.tile_shape == (24, 64)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_edge_tile_remain_at_least_4():
    """Shape where image %% tile leaves remainder >= 4 — valid HCOMPRESS_1
    (edge tile satisfies the 4-pixel minimum)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        # 50x70 with 16x16: rows 50 % 16 = 2 (BAD), use 14x14:
        # 50 % 14 = 8 (good); 70 % 14 = 0 (good).
        data = _data((50, 70), "i4")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(14, 14))
            f.create_image_hdu("i4", data.shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- lossy round-trip ---------------------------


@pytest.mark.parametrize("scale", [4, 8, 16])
def test_lossy_round_trip(scale):
    """
    Lossy scale > 1 round-trips correctly (approximately): the
    decoded image differs from the original by at most ~scale, and
    the difference is bounded.  Exact bit-pattern is checked in the
    byte-exact tests below.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((32, 32), "i2", seed=11)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(32, 32), scale=scale)
            f.create_image_hdu("i2", data.shape, compress=cfg)
            f[1].write(data)
        with rustfits.FITS(fn, "r") as f:
            out = f[1].read()
        diff = np.abs(out.astype("f8") - data.astype("f8"))
        # cfitsio guarantees |diff| <= scale for digitized HCOMPRESS;
        # bound loosely to avoid flakiness on different RNG seeds.
        assert diff.max() <= scale + 1


def test_smooth_true_round_trip():
    """
    SMOOTH=True doesn't change encoded bytes (the smoothing happens
    on read).  Verify the ZNAME card is set and that the smoothed
    read matches an un-smoothed read within the scale's precision.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((32, 32), "i2", seed=22)
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(
                tile_shape=(32, 32),
                scale=8,
                smooth=True,
            )
            f.create_image_hdu("i2", data.shape, compress=cfg)
            f[1].write(data)
        # SMOOTH card must reflect True.
        with rustfits.FITS(fn, "r") as f:
            znames = {
                f[1].header[f"ZNAME{n}"]: f[1].header[f"ZVAL{n}"]
                for n in (1, 2)
            }
            assert znames["SMOOTH"] == 1
            assert znames["SCALE"] == 8
            out = f[1].read()
        # Smoothed scale=8 read should be within ~scale of the input.
        diff = np.abs(out.astype("f8") - data.astype("f8"))
        assert diff.max() <= 10  # generous; smoothing can adjust by ~scale


# ---------------------- byte-exact heap with fitsio ----------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_byte_exact_single_tile(dtype):
    """
    Single-tile lossless heap bytes must match cfitsio's output
    byte-for-byte.  Catches any drift in the H-transform, digitize,
    or bit-output paths.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = _data((32, 48), dtype)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=rustfits.Hcompress1(tile_shape=(32, 48)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="HCOMPRESS_1",
            tile_dims=(32, 48),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


def test_byte_exact_multi_tile_evenly_divisible():
    """
    Multi-tile case with image exactly divisible by tile (no edge
    tiles).  Heap bytes must match cfitsio byte-for-byte across
    every tile.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        # 48x72 with 16x24 → 3x3 = 9 full-size tiles
        data = _data((48, 72), "i4", seed=33)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(tile_shape=(16, 24)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="HCOMPRESS_1",
            tile_dims=(16, 24),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


def test_byte_exact_multi_tile_with_valid_edge():
    """
    Multi-tile case with edge tile whose remainder is exactly 4
    pixels (the lower bound for HCOMPRESS_1).  rustfits keeps the
    tile_shape literally; cfitsio's adjustment only fires when
    remainder < 4, so on this case the two layouts agree.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        # 32x32 with 16x24: cols 32%24 = 8 (>= 4, valid edge);
        # rows 32%16 = 0 (no edge).
        data = _data((32, 32), "i4", seed=44)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(tile_shape=(16, 24)),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="HCOMPRESS_1",
            tile_dims=(16, 24),
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


@pytest.mark.parametrize("scale", [4, 8, 16])
def test_byte_exact_lossy_single_tile(scale):
    """
    Lossy single-tile heap bytes match cfitsio byte-for-byte.
    fitsio's `hcomp_scale > 0` triggers per-tile noise-based scaling;
    `hcomp_scale < 0` uses the absolute value as a fixed scale.  We
    pass a fixed scale, so the comparison goes through hcomp_scale=
    -scale on the fitsio side.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_rust = os.path.join(tmp, "rust.fits.fz")
        fn_fits = os.path.join(tmp, "fitsio.fits.fz")
        data = _data((32, 48), "i2", seed=55)
        with rustfits.FITS(fn_rust, "w+") as f:
            f.create_image_hdu(
                "i2",
                data.shape,
                compress=rustfits.Hcompress1(
                    tile_shape=(32, 48),
                    scale=scale,
                ),
            )
            f[1].write(data)
        fitsio.write(
            fn_fits,
            data,
            compress="HCOMPRESS_1",
            tile_dims=(32, 48),
            hcomp_scale=-scale,
        )
        assert _heap_bytes(fn_rust) == _heap_bytes(fn_fits)


# ---------------------- bidirectional fitsio interop ---------------


def test_rustfits_written_fitsio_read_matches():
    """rustfits-written → fitsio-read bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((40, 60), "i4", seed=66)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=rustfits.Hcompress1(tile_shape=(20, 30)),
            )
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            assert f[1].read_header().get("ZCMPTYPE") == "HCOMPRESS_1"
            np.testing.assert_array_equal(f[1].read(), data)


def test_fitsio_written_rustfits_read_matches():
    """fitsio-written → rustfits-read bit-exact (regression guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _data((40, 60), "i4", seed=77)
        fitsio.write(
            fn,
            data,
            compress="HCOMPRESS_1",
            tile_dims=(20, 30),
        )
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


# ---------------------- non-last HDU growth ------------------------


def test_compressed_write_shifts_later_hdus():
    """
    Compressed HCOMPRESS HDU followed by another HDU; heap write
    must shift the later HDU's offsets, and post-reopen reads of
    both still work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _data((64, 64), "i4", seed=88)
        later_data = _data((10, 10), "i2", seed=99)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                comp_data.shape,
                compress=rustfits.Hcompress1(tile_shape=(16, 16)),
                extname="COMP",
            )
            f.create_image_hdu("i2", later_data.shape, extname="LATER")
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


def test_mixed_hcompress_and_rice_in_one_file():
    """One Hcompress1 + one Rice1 HDU in the same file."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        d1 = _data((32, 48), "i4")
        d2 = _data((48, 32), "i2")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                d1.shape,
                compress=rustfits.Hcompress1(tile_shape=(16, 24)),
                extname="HC",
            )
            f.create_image_hdu(
                "i2",
                d2.shape,
                compress=rustfits.Rice1(tile_shape=(24, 16)),
                extname="RC",
            )
            f[1].write(d1)
            f[2].write(d2)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].compression.zcmptype == "HCOMPRESS_1"
            assert f[2].compression.zcmptype == "RICE_1"
            np.testing.assert_array_equal(f[1].read(), d1)
            np.testing.assert_array_equal(f[2].read(), d2)


# ---------------------- rejections ---------------------------------


def test_i8_hcompress_rejected():
    """i8 (bitpix=64) rejected — HCOMPRESS_1 has no 64-bit variant."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="64-bit"):
                f.create_image_hdu(
                    "i8",
                    (16, 16),
                    compress=rustfits.Hcompress1(),
                )


def test_float_compress_rejected():
    """Float ZBITPIX writes raise NotImplementedError (Phase 8)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="float"):
                f.create_image_hdu(
                    "f4",
                    (16, 16),
                    compress=rustfits.Hcompress1(tile_shape=(16, 16)),
                )


def test_unsigned_trick_dtype_rejected():
    """u2/u4/u8/i1 not yet supported on the compressed-write side."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="unsigned"):
                f.create_image_hdu(
                    "u4",
                    (16, 16),
                    compress=rustfits.Hcompress1(tile_shape=(16, 16)),
                )


def test_non_2d_image_rejected():
    """HCOMPRESS_1 is 2-D only; 1-D and 3-D images rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="2-D"):
                f.create_image_hdu(
                    "i4",
                    (32,),
                    compress=rustfits.Hcompress1(),
                )
            with pytest.raises(ValueError, match="2-D"):
                f.create_image_hdu(
                    "i4",
                    (4, 4, 4),
                    compress=rustfits.Hcompress1(tile_shape=(4, 4, 4)),
                )


def test_image_smaller_than_minimum_rejected():
    """HCOMPRESS_1 requires at least 4 pixels per dimension."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="minimum of 4"):
                f.create_image_hdu(
                    "i4",
                    (3, 16),
                    compress=rustfits.Hcompress1(),
                )


def test_tile_smaller_than_minimum_rejected():
    """tile_shape[i] < 4 rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="minimum of 4"):
                f.create_image_hdu(
                    "i4",
                    (16, 16),
                    compress=rustfits.Hcompress1(tile_shape=(2, 16)),
                )


def test_thin_edge_tile_rejected_with_suggestion():
    """
    Edge tile with < 4 pixels rejected (astropy-style policy).
    The error message must include the cfitsio-style adjusted tile
    value so the user can copy it back into their config.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            # 50x32 with 16x16 → row remain = 50 % 16 = 2 (< 4).
            # cfitsio's adjustment: ndiv=3, add=ceil(2/3)=1 → tile=17.
            with pytest.raises(ValueError) as exc:
                f.create_image_hdu(
                    "i4",
                    (50, 32),
                    compress=rustfits.Hcompress1(tile_shape=(16, 16)),
                )
            msg = str(exc.value)
            assert "last tile of 2 pixels" in msg
            assert "tile_shape[0]=17" in msg


def test_input_shape_mismatch_rejected():
    """write(data) with wrong shape raises; subsequent good write works."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            wrong = np.arange(8 * 8, dtype="i4").reshape(8, 8)
            with pytest.raises(ValueError, match="shape"):
                f[1].write(wrong)
            good = _data((16, 16), "i4")
            f[1].write(good)
            np.testing.assert_array_equal(f[1].read(), good)


def test_start_kwarg_rejected():
    """start= is not supported on compressed-image writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            cfg = rustfits.Hcompress1(tile_shape=(8, 8))
            f.create_image_hdu("i4", (16, 16), compress=cfg)
            data = _data((16, 16), "i4")
            with pytest.raises(NotImplementedError, match="start="):
                f[1].write(data, start=[0, 0])
