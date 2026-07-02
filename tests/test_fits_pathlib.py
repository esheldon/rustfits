"""
Tests that FITS (and the convenience wrappers) accept os.PathLike
filenames — pathlib.Path in particular — alongside plain strings.

Matches astropy / fitsio, which both take Path objects.  A str is
still taken verbatim (so the driver URL schemes mem:// / http:// /
ftp:// and the .gz suffix are preserved byte-for-byte); anything else
is routed through os.fspath().
"""

import pathlib
import tempfile
import contextlib

import numpy as np
import pytest

import rustfits


@contextlib.contextmanager
def _tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


def test_fits_write_and_read_via_path():
    with _tmpdir() as d:
        p = d / "img.fits"
        data = np.arange(12, dtype="i4").reshape(3, 4)
        with rustfits.FITS(p, "w+") as fits:
            fits.create_image_hdu("i4", (3, 4))
            fits[0].write(data)
        # read via a fresh Path pointing at the same file
        with rustfits.FITS(pathlib.Path(p), "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_filename_attr_is_string_form_of_path():
    with _tmpdir() as d:
        p = d / "img.fits"
        with rustfits.FITS(p, "w+") as fits:
            fits.create_image_hdu("i4", (2, 2))
            assert fits.filename == str(p)
            assert isinstance(fits.filename, str)


def test_str_and_path_open_same_file():
    with _tmpdir() as d:
        p = d / "img.fits"
        data = np.arange(4, dtype="f4").reshape(2, 2)
        with rustfits.FITS(str(p), "w+") as fits:
            fits.create_image_hdu("f4", (2, 2))
            fits[0].write(data)
        with rustfits.FITS(p, "r") as fits:  # Path
            np.testing.assert_array_equal(fits[0].read(), data)


def test_convenience_read_write_via_path():
    with _tmpdir() as d:
        p = d / "conv.fits"
        data = np.arange(6, dtype="f4")
        rustfits.write(p, data)  # Path
        np.testing.assert_array_equal(rustfits.read(p), data)  # Path


def test_convenience_read_header_via_path():
    with _tmpdir() as d:
        p = d / "img.fits"
        with rustfits.FITS(p, "w+") as fits:
            fits.create_image_hdu("i4", (2, 2))
            fits[0].header["OBJECT"] = "M31"
        hdr = rustfits.read_header(p)  # Path
        assert hdr["OBJECT"] == "M31"


def test_gz_suffix_detected_through_path():
    """A .gz Path still routes through the gzip write-back path."""
    with _tmpdir() as d:
        p = d / "img.fits.gz"
        data = np.arange(9, dtype="i4").reshape(3, 3)
        with rustfits.FITS(p, "w+") as fits:
            fits.create_image_hdu("i4", (3, 3))
            fits[0].write(data)
        # The on-disk bytes must be gzip-compressed (magic 0x1f 0x8b).
        with open(p, "rb") as fh:
            assert fh.read(2) == b"\x1f\x8b"
        with rustfits.FITS(p, "r") as fits:
            np.testing.assert_array_equal(fits[0].read(), data)


def test_mem_url_str_preserved_verbatim():
    """A str is taken verbatim — mem:// is NOT path-normalized."""
    with rustfits.FITS("mem://", "w+") as fits:
        fits.create_image_hdu("i4", (2, 2))
        assert fits.filename == "mem://"


@pytest.mark.parametrize("bad", [12345, 3.14, b"/tmp/x.fits", None])
def test_non_pathlike_rejected(bad):
    with pytest.raises(TypeError):
        rustfits.FITS(bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
