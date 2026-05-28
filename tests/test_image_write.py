"""
Tests for ImageHDU.write.

Covers:
    - full overwrite (data shape matches HDU shape, start defaults to origin)
    - partial write with `start=` placing a sub-region inside the HDU
    - dtype / shape / bounds validation
    - byte order: input arrays in big-endian, little-endian, and native order
      all land as big-endian on disk (FITS standard)
    - non-contiguous numpy input is rejected with a hint
    - 1D and 3D images
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- helpers --------------------


def _make_file_with_image(tmpdir, dtype, dims, extname="img"):
    """Create a fresh FITS file with one image HDU and return its filename."""
    fname = os.path.join(tmpdir, "t.fits")
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=dims, extname=extname)
    return fname


def _read_image_bytes(fname, hdu_index=0):
    """
    Read the on-disk data section bytes for the given HDU.  Returns the
    raw bytes (length = product(dims) * bpp, NOT padded to 2880)."""
    with rustfits.FITS(fname, "r") as fits:
        hd = fits.hdus[hdu_index].header
        bitpix = hd["BITPIX"]
        naxis = hd["NAXIS"]
        fits_dims = [hd[f"NAXIS{i}"] for i in range(1, naxis + 1)]
    bpp = abs(bitpix) // 8
    nbytes = bpp
    for d in fits_dims:
        nbytes *= d
    # data section follows the header; we always emit a single 2880-block
    # header for the small HDUs in these tests.
    with open(fname, "rb") as f:
        f.seek(2880)
        return f.read(nbytes)


# -------------------- full write, round-trip --------------------


def test_full_write_roundtrip_f8():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (5, 20))
        data = np.arange(5 * 20, dtype="f8").reshape(5, 20) * 0.5 - 3.0

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(data)

        # Verify on-disk bytes are exactly big-endian f8 of `data`.
        on_disk = _read_image_bytes(fname)
        expected = data.astype(">f8").tobytes()
        assert on_disk == expected

        # And we can recover the values by reading as big-endian f8.
        recovered = np.frombuffer(on_disk, dtype=">f8").reshape(5, 20)
        np.testing.assert_array_equal(recovered, data)


def test_full_write_roundtrip_i4():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (3, 7))
        data = np.arange(3 * 7, dtype="i4").reshape(3, 7) - 5

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(data)

        on_disk = _read_image_bytes(fname)
        assert on_disk == data.astype(">i4").tobytes()


def test_full_write_u1_zero_copy():
    """
    uint8 has no byte order — the write path is zero-copy.  Verify the
    bytes land verbatim."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "u1", (4, 6))
        data = np.arange(4 * 6, dtype="u1").reshape(4, 6)

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(data)

        on_disk = _read_image_bytes(fname)
        assert on_disk == data.tobytes()


# -------------------- byte order --------------------


def test_write_already_big_endian_input():
    """Big-endian input takes the zero-copy path; result must match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (3, 4))
        data_be = np.arange(12, dtype=">f8").reshape(3, 4)

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(data_be)

        on_disk = _read_image_bytes(fname)
        assert on_disk == data_be.tobytes()  # already big-endian


def test_write_little_endian_input_swapped():
    """Little-endian input must be byte-swapped before landing on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (2, 3))
        data_le = np.array([[1, 2, 3], [4, 5, 6]], dtype="<i4")

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(data_le)

        on_disk = _read_image_bytes(fname)
        # Disk must be big-endian i4.
        recovered = np.frombuffer(on_disk, dtype=">i4").reshape(2, 3)
        np.testing.assert_array_equal(recovered, data_le)


# -------------------- partial write with start --------------------


def test_partial_write_2d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (5, 10))
        # Write a 3x4 sub-region at numpy [1, 2].
        sub = np.arange(12, dtype="i4").reshape(3, 4) + 100

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(sub, start=(1, 2))

        on_disk = _read_image_bytes(fname)
        img = np.frombuffer(on_disk, dtype=">i4").reshape(5, 10).copy()

        # The sub-region landed in the right place.
        np.testing.assert_array_equal(img[1:4, 2:6], sub)

        # Everything else is still zero (from create_image_hdu zero-fill).
        mask = np.ones(img.shape, dtype=bool)
        mask[1:4, 2:6] = False
        assert (img[mask] == 0).all()


def test_partial_write_1d():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (20,))
        sub = np.array([1.5, 2.5, 3.5, 4.5], dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(sub, start=(5,))

        on_disk = _read_image_bytes(fname)
        img = np.frombuffer(on_disk, dtype=">f4").copy()
        np.testing.assert_array_equal(img[5:9], sub)
        assert (img[:5] == 0).all()
        assert (img[9:] == 0).all()


def test_partial_write_3d():
    """3D sub-region exercises multi-axis strip iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (4, 5, 6))
        sub = np.arange(2 * 3 * 4, dtype="f8").reshape(2, 3, 4) + 10.0

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(sub, start=(1, 1, 1))

        on_disk = _read_image_bytes(fname)
        img = np.frombuffer(on_disk, dtype=">f8").reshape(4, 5, 6).copy()
        np.testing.assert_array_equal(img[1:3, 1:4, 1:5], sub)

        mask = np.ones(img.shape, dtype=bool)
        mask[1:3, 1:4, 1:5] = False
        assert (img[mask] == 0).all()


def test_partial_write_full_inner_axis_coalesces():
    """
    When the fast axis fully spans the HDU, the strip coalesces into a
    single larger contiguous write per outer step; the result must still be
    correct (exercises compute_strip_layout's coalescing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i2", (5, 8))
        # Fast axis = 8 (matches HDU); outer axis is partial.
        sub = np.arange(3 * 8, dtype="i2").reshape(3, 8) + 50

        with rustfits.FITS(fname, "r+") as fits:
            fits.hdus[0].write(sub, start=(1, 0))

        on_disk = _read_image_bytes(fname)
        img = np.frombuffer(on_disk, dtype=">i2").reshape(5, 8).copy()
        np.testing.assert_array_equal(img[1:4], sub)
        assert (img[0] == 0).all()
        assert (img[4] == 0).all()


# -------------------- validation errors --------------------


def test_wrong_dtype_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "i4", (3, 3))
        bad = np.zeros((3, 3), dtype="f8")

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="does not match HDU BITPIX"):
                fits.hdus[0].write(bad)


def test_wrong_naxis_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (3, 3))
        bad = np.zeros((3,), dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="axes"):
                fits.hdus[0].write(bad)


def test_out_of_bounds_start_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (5, 5))
        sub = np.zeros((3, 3), dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="exceeds HDU dim"):
                fits.hdus[0].write(sub, start=(3, 3))


def test_negative_start_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f4", (5, 5))
        sub = np.zeros((2, 2), dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="must be >= 0"):
                fits.hdus[0].write(sub, start=(-1, 0))


def test_non_contiguous_rejected():
    """
    A non-C-contiguous array (here, a transpose) must be rejected with
    a hint about np.ascontiguousarray."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (4, 4))
        base = np.arange(16, dtype="f8").reshape(4, 4)
        non_contig = base.T  # not C-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="C-contiguous"):
                fits.hdus[0].write(non_contig)


# -------------------- create_image_hdu dtype-like inputs --------------------


@pytest.mark.parametrize(
    "dtype_input, expected_bitpix",
    [
        # Plain numpy short-codes (existing behavior).
        ("f8", -64),
        ("i4", 32),
        ("u2", 16),
        # Endianness-prefixed numpy short-codes — stripped by
        # dtype_to_bitpix's leading-prefix trim.
        ("<i4", 32),
        (">f4", -32),
        # Long-form numpy aliases handled by np.dtype(...).str
        # normalization on the Rust side.
        ("int32", 32),
        ("float64", -64),
        ("uint8", 8),
        # numpy scalar types.
        (np.int32, 32),
        (np.float64, -64),
        (np.uint8, 8),
        (np.int8, 8),  # unsigned-int trick (i1 stored as i2 + BZERO).
        # Python builtins (numpy maps int → i8, float → f8).
        (int, 64),
        (float, -64),
        # An existing np.dtype object.
        (np.dtype("i2"), 16),
        (np.dtype(">f8"), -64),
    ],
)
def test_image_create_hdu_dtype_like_inputs(dtype_input, expected_bitpix):
    """
    create_image_hdu accepts anything np.dtype() accepts — strings
    (with or without endianness), long-form aliases, numpy scalar
    types, Python builtins, and np.dtype objects all normalize to
    the same BITPIX.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "dtype.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype_input, (4,), extname="X")
            assert fits[0].bitpix == expected_bitpix


def test_image_create_hdu_dtype_object_write_roundtrip():
    """
    Passing np.dtype(...) directly (instead of a short-code) yields
    an HDU that round-trips a matching-dtype write.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "rt.fits")
        data = np.arange(12, dtype="f4").reshape(3, 4)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(np.dtype("f4"), data.shape, extname="X")
            fits[0].write(data)
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_image_create_hdu_python_float_writes_f8():
    """
    `float` is a stand-in for `np.float64` per numpy's convention;
    repeat the round-trip end-to-end so the BITPIX + data path agree.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "pyfloat.fits")
        data = np.linspace(0.0, 1.0, 8, dtype="f8")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(float, data.shape, extname="X")
            assert fits[0].bitpix == -64
            fits[0].write(data)
        with rustfits.FITS(fname) as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_image_create_hdu_rejects_unsupported_dtype_object():
    """
    Complex dtypes still raise the dtype_to_bitpix error message
    (np.complex64 normalizes to '<c8' which is not supported).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "bad.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="unsupported numpy dtype"):
                fits.create_image_hdu(np.complex64, (4,))


def test_image_create_hdu_rejects_garbage_dtype():
    """
    np.dtype() raises TypeError on completely bogus input
    (e.g. a list or a random object); the error surfaces unchanged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "bad.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises((TypeError, ValueError)):
                fits.create_image_hdu(object(), (4,))


# -------------------- extension HDU write --------------------


def test_write_to_extension_hdu():
    """
    Writes addressed via hdus[1] must land in the second HDU's data
    section, not the primary's."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(3, 4), extname="primary")
            fits.create_image_hdu(dtype="f8", dims=(2, 5), extname="second")

            primary = np.arange(12, dtype="i4").reshape(3, 4) + 1
            second = np.arange(10, dtype="f8").reshape(2, 5) - 0.5

            fits.hdus[0].write(primary)
            fits.hdus[1].write(second)

        # Primary HDU: 1 header block + 1 data block (48 bytes inside 2880).
        with open(fname, "rb") as f:
            f.seek(2880)
            primary_bytes = f.read(48)
        np.testing.assert_array_equal(
            np.frombuffer(primary_bytes, dtype=">i4").reshape(3, 4),
            primary,
        )
        # Extension HDU follows: primary header (2880) + primary data (2880)
        # + extension header (2880) = 8640.
        with open(fname, "rb") as f:
            f.seek(2880 + 2880 + 2880)
            second_bytes = f.read(80)
        np.testing.assert_array_equal(
            np.frombuffer(second_bytes, dtype=">f8").reshape(2, 5),
            second,
        )


# -------------------- empty-shape create + extend later --------------------
#
# `create_image_hdu` accepts a 0 on numpy axis 0 (= FITS NAXIS-last) so
# callers can size the rest of the shape up-front and stream rows in via
# `ImageHDU.extend`.  Inner-axis 0 stays rejected (FITS standard forbids
# zero pixels on inner axes).


def test_empty_2d_create_then_extend_uncompressed():
    """shape=(0, M) HDU + later extend with N rows lands as shape=(N, M)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("f4", (0, 1024), extname="SCI")
            assert fits[0].shape == (0, 1024)
            assert fits[0].dtype == np.dtype("f4")
            data = np.arange(10 * 1024, dtype="f4").reshape(10, 1024) * 0.5
            fits[0].extend(data)
            assert fits[0].shape == (10, 1024)
            np.testing.assert_array_equal(fits[0].read(), data)
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].shape == (10, 1024)
            np.testing.assert_array_equal(fits[0].read(), data)


def test_empty_1d_create_then_extend_uncompressed():
    """shape=(0,) HDU + extend turns into a regular 1-D image."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("i4", (0,))
            assert fits[0].shape == (0,)
            data = np.arange(7, dtype="i4")
            fits[0].extend(data)
            np.testing.assert_array_equal(fits[0].read(), data)
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_empty_3d_create_then_extend_uncompressed():
    """shape=(0, M, K) HDU only grows axis 0 on extend."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("f8", (0, 4, 5))
            data = np.arange(3 * 4 * 5, dtype="f8").reshape(3, 4, 5)
            fits[0].extend(data)
            assert fits[0].shape == (3, 4, 5)
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_empty_create_then_multiple_extends_accumulate():
    """Successive extends append along axis 0."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        a = np.arange(6, dtype="i2").reshape(2, 3)
        b = (np.arange(9, dtype="i2") + 100).reshape(3, 3)
        c = (np.arange(3, dtype="i2") - 50).reshape(1, 3)
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("i2", (0, 3))
            fits[0].extend(a)
            fits[0].extend(b)
            fits[0].extend(c)
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(
                fits[0].read(), np.concatenate([a, b, c], axis=0)
            )


def test_empty_read_returns_empty_array_uncompressed():
    """Reading a freshly-created shape=(0, M) HDU yields a 0-row array."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("f4", (0, 8))
            arr = fits[0].read()
            assert arr.shape == (0, 8)
            assert arr.dtype == np.dtype("f4")
        with rustfits.FITS(fname, "r") as fits:
            arr = fits[0].read()
            assert arr.shape == (0, 8)


def test_empty_create_rejects_zero_on_inner_axes():
    """Zero on any axis other than axis 0 is rejected — FITS forbids it."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match=r"dimension 1 must be > 0"):
                fits.create_image_hdu("f4", (10, 0))
            with pytest.raises(ValueError, match=r"dimension 1 must be > 0"):
                fits.create_image_hdu("f4", (0, 0))
            with pytest.raises(ValueError, match=r"dimension 2 must be > 0"):
                fits.create_image_hdu("f4", (10, 4, 0))


def test_empty_create_rejects_negative_axis0():
    """Axis 0 may be 0 (empty) but not negative."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match=r"dimension 0 must be >= 0"):
                fits.create_image_hdu("f4", (-1, 1024))


def test_empty_create_then_extend_not_last_hdu_uncompressed():
    """Growing an empty HDU that's not the last shifts the trailing HDU."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        trailing = np.arange(20, dtype="i4").reshape(4, 5) - 7
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu("f4", (0, 8), extname="EMPTY")
            fits.create_image_hdu("i4", (4, 5), extname="TRAIL")
            fits[1].write(trailing)
            data = np.arange(6 * 8, dtype="f4").reshape(6, 8) * 0.25
            fits[0].extend(data)
            np.testing.assert_array_equal(fits[0].read(), data)
            np.testing.assert_array_equal(fits[1].read(), trailing)
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), data)
            np.testing.assert_array_equal(fits[1].read(), trailing)


def test_empty_2d_create_then_extend_compressed_gzip1():
    """Empty compressed image + extend round-trips through Gzip1."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "c.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                (0, 1024),
                extname="SCI",
                compress=rustfits.Gzip1(tile_shape=(50, 1024)),
            )
            assert isinstance(fits[1], rustfits.CompressedImageHDU)
            assert fits[1].shape == (0, 1024)
            assert fits[1].n_tiles == 0
            data = np.arange(100 * 1024, dtype="i4").reshape(100, 1024)
            fits[1].extend(data)
            assert fits[1].shape == (100, 1024)
            assert fits[1].n_tiles == 2
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[1].read(), data)


def test_empty_2d_create_then_extend_compressed_rice1():
    """Same flow under Rice1 (different encoder)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "c.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(
                "i4",
                (0, 256),
                compress=rustfits.Rice1(tile_shape=(32, 256)),
            )
            data = np.arange(32 * 256, dtype="i4").reshape(32, 256)
            fits[1].extend(data)
        with rustfits.FITS(fname, "r") as fits:
            np.testing.assert_array_equal(fits[1].read(), data)


def test_empty_compressed_rejects_zero_on_inner_axes():
    """Compressed path also rejects 0 on inner axes."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "c.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match=r"dimension 1 must be > 0"):
                fits.create_image_hdu(
                    "i4",
                    (10, 0),
                    compress=rustfits.Gzip1(tile_shape=(5, 1024)),
                )


def test_empty_compressed_hcompress_still_rejects_axis0_zero():
    """HCOMPRESS_1's own dim >= 4 check still rejects axis 0 == 0.

    The looser create-time validation allows axis 0 == 0, but HCOMPRESS
    has algorithm-specific minimum-dim constraints (every dim >= 4) that
    fire on top of the create check, so the empty-create + extend-later
    pattern is not available under HCOMPRESS.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match=r"axis 0 has size 0"):
                fits.create_image_hdu(
                    "i4",
                    (0, 64),
                    compress=rustfits.Hcompress1(tile_shape=(16, 64)),
                )


def test_empty_create_copy_loop_pattern():
    """fitsio-style copy loop: empty HDU per source + extend with .read()."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.fits")
        dst = os.path.join(td, "dst.fits")
        img1 = np.arange(20, dtype="i4").reshape(4, 5) + 1
        img2 = (np.arange(60, dtype="f4").reshape(5, 12) - 30.0) * 0.1
        img3 = np.arange(8, dtype="i2") - 4
        with rustfits.FITS(src, "w+") as fits:
            fits.write(img1)
            fits.write(img2)
            fits.write(img3)
        with rustfits.FITS(src, "r") as srcf, rustfits.FITS(dst, "w+") as dstf:
            for hdu in srcf:
                if not hdu.has_data:
                    continue
                if not isinstance(hdu, rustfits.ImageHDU):
                    continue
                empty_shape = (0,) + tuple(hdu.shape[1:])
                target_idx = len(dstf)
                dstf.create_image_hdu(str(hdu.dtype.str), empty_shape)
                dstf[target_idx].extend(hdu.read())
        with rustfits.FITS(dst, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), img1)
            np.testing.assert_array_equal(fits[1].read(), img2)
            np.testing.assert_array_equal(fits[2].read(), img3)


if __name__ == "__main__":
    test_full_write_roundtrip_f8()
    test_full_write_roundtrip_i4()
    test_full_write_u1_zero_copy()
    test_write_already_big_endian_input()
    test_write_little_endian_input_swapped()
    test_partial_write_2d()
    test_partial_write_1d()
    test_partial_write_3d()
    test_partial_write_full_inner_axis_coalesces()
    test_write_to_extension_hdu()
    test_empty_2d_create_then_extend_uncompressed()
    test_empty_1d_create_then_extend_uncompressed()
    test_empty_3d_create_then_extend_uncompressed()
    test_empty_create_then_multiple_extends_accumulate()
    test_empty_read_returns_empty_array_uncompressed()
    test_empty_create_rejects_zero_on_inner_axes()
    test_empty_create_rejects_negative_axis0()
    test_empty_create_then_extend_not_last_hdu_uncompressed()
    test_empty_2d_create_then_extend_compressed_gzip1()
    test_empty_2d_create_then_extend_compressed_rice1()
    test_empty_compressed_rejects_zero_on_inner_axes()
    test_empty_compressed_hcompress_still_rejects_axis0_zero()
    test_empty_create_copy_loop_pattern()
    print("all tests passed")
