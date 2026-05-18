"""Tests for ImageHDU.write.

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
    """Read the on-disk data section bytes for the given HDU.  Returns the
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
    """uint8 has no byte order — the write path is zero-copy.  Verify the
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
    """When the fast axis fully spans the HDU, the strip coalesces into a
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
    """A non-C-contiguous array (here, a transpose) must be rejected with
    a hint about np.ascontiguousarray."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_file_with_image(tmpdir, "f8", (4, 4))
        base = np.arange(16, dtype="f8").reshape(4, 4)
        non_contig = base.T  # not C-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="C-contiguous"):
                fits.hdus[0].write(non_contig)


# -------------------- extension HDU write --------------------


def test_write_to_extension_hdu():
    """Writes addressed via hdus[1] must land in the second HDU's data
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
    print("all tests passed")
