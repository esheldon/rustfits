"""
Tests for FITS.create_image_hdu.

Verifies:
    - first call creates a primary image HDU (SIMPLE/EXTEND); subsequent
      calls create IMAGE extensions (XTENSION/PCOUNT/GCOUNT)
    - numpy-style row-major dims are reversed to FITS NAXISn ordering
      (NAXIS1 is the fastest-varying axis)
    - supported numpy dtype short codes map to the correct BITPIX
    - EXTNAME / EXTVER are written when supplied
    - on-disk file is well-formed and round-trips via a fresh open
    - data section is allocated and zero-filled (for the supported dtypes)
"""

import os
import tempfile

import pytest

import rustfits


def test_create_primary_image_hdu():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            assert len(fits.hdus) == 0
            fits.create_image_hdu(dtype="f8", dims=(5, 20), extname="image1")

            assert len(fits.hdus) == 1
            hdu = fits.hdus[0]
            assert isinstance(hdu, rustfits.ImageHDU)
            assert hdu.index == 0

            hd = hdu.header
            assert hd["SIMPLE"] is True
            assert hd["BITPIX"] == -64
            assert hd["NAXIS"] == 2
            # numpy dims (5, 20) -> FITS NAXIS1=20, NAXIS2=5
            assert hd["NAXIS1"] == 20
            assert hd["NAXIS2"] == 5
            assert hd["EXTEND"] is True
            assert hd["EXTNAME"] == "image1"


def test_create_image_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(3, 4), extname="image1")
            fits.create_image_hdu(
                dtype="f4",
                dims=(2, 5, 6),
                extname="image2",
                extver=3,
            )

            assert len(fits.hdus) == 2
            assert isinstance(fits.hdus[1], rustfits.ImageHDU)
            assert fits.hdus[1].index == 1

            hd = fits.hdus[1].header
            # Extension headers begin with XTENSION = 'IMAGE   '
            assert hd["XTENSION"] == "IMAGE"
            assert hd["BITPIX"] == -32
            assert hd["NAXIS"] == 3
            # numpy (2, 5, 6) -> FITS NAXIS1=6, NAXIS2=5, NAXIS3=2
            assert hd["NAXIS1"] == 6
            assert hd["NAXIS2"] == 5
            assert hd["NAXIS3"] == 2
            assert hd["PCOUNT"] == 0
            assert hd["GCOUNT"] == 1
            assert hd["EXTNAME"] == "image2"
            assert hd["EXTVER"] == 3


@pytest.mark.parametrize(
    "dtype,bitpix",
    [
        ("u1", 8),
        ("i2", 16),
        ("i4", 32),
        ("i8", 64),
        ("f4", -32),
        ("f8", -64),
        ("uint8", 8),
        ("int32", 32),
        ("float64", -64),
        ("<f8", -64),  # endianness prefix stripped
        (">i4", 32),
    ],
)
def test_dtype_maps_to_bitpix(dtype, bitpix):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=(2, 3))
            assert fits.hdus[0].header["BITPIX"] == bitpix


def test_unsupported_dtype_raises():
    # c8 (complex64) is not a FITS image storage type; bool, datetime,
    # and strings are also rejected.  u2/u4/u8/i1 are now accepted via
    # the unsigned-int trick (BZERO offset) — covered in
    # test_image_scale_write.py.
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="unsupported numpy dtype"):
                fits.create_image_hdu(dtype="c8", dims=(2, 3))


def test_zero_dim_on_inner_axis_rejected():
    """
    Zero is only allowed on axis 0 (numpy slowest = FITS NAXIS-last) so
    callers can stream rows in via `ImageHDU.extend`.  Inner axes must
    stay strictly positive — the FITS standard forbids zero pixels on
    inner axes.  See `test_write_image.py::test_empty_*` for the
    supported axis-0 case.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            with pytest.raises(ValueError, match="must be > 0"):
                fits.create_image_hdu(dtype="f8", dims=(3, 0))


def test_roundtrip_via_reopen():
    """
    Write a primary image, close, reopen read-only, verify the parser
    can re-read what we wrote and finds the data section in the right place
    (a misaligned data section would corrupt the HDU offset for any later
    extension).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=(5, 20), extname="img")
            fits.create_image_hdu(dtype="f8", dims=(3, 4), extname="ext1")

        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 2

            hd0 = fits.hdus[0].header
            assert hd0["BITPIX"] == 32
            assert hd0["NAXIS1"] == 20
            assert hd0["NAXIS2"] == 5
            assert hd0["EXTNAME"] == "img"

            hd1 = fits.hdus[1].header
            assert hd1["XTENSION"] == "IMAGE"
            assert hd1["BITPIX"] == -64
            assert hd1["NAXIS1"] == 4
            assert hd1["NAXIS2"] == 3
            assert hd1["EXTNAME"] == "ext1"


def test_data_section_padded_and_zero():
    """
    File on disk must have the data section padded to a 2880-byte boundary
    and filled with zeros (allocated via sparse extension)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            # 5 x 20 i4 = 400 bytes -> padded to 2880
            fits.create_image_hdu(dtype="i4", dims=(5, 20))

        size = os.path.getsize(fname)
        # exactly one header block + one data block
        assert size == 2880 * 2

        with open(fname, "rb") as f:
            data = f.read()
        # data section starts at 2880; should be all zero
        assert data[2880:] == b"\x00" * 2880


def test_naxis0_has_no_data_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="u1", dims=[])

            hd = fits.hdus[0].header
            assert hd["NAXIS"] == 0

        # one header block only (no data unit when NAXIS=0)
        assert os.path.getsize(fname) == 2880


def test_extname_with_embedded_quote():
    """
    Single quotes inside EXTNAME must be doubled per the FITS standard,
    and the round-trip must recover the original string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=(2, 2), extname="O'Brien")

        with rustfits.FITS(fname, "r") as fits:
            assert fits.hdus[0].header["EXTNAME"] == "O'Brien"


if __name__ == "__main__":
    test_create_primary_image_hdu()
    test_create_image_extension()
    test_roundtrip_via_reopen()
    test_data_section_padded_and_zero()
    test_naxis0_has_no_data_section()
    test_extname_with_embedded_quote()
    print("all tests passed")
