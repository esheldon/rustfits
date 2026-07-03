import os
import shutil
import tempfile

import pytest

import rustfits
from pprint import pprint

# Canonical fixtures committed under tests/data/ (cfitsio-produced
# reference bytes).  Copying a frozen file instead of writing one via
# fitsio keeps these open/parse tests free of any FITS-writer
# dependency at runtime, so they run on platforms with no fitsio build.
# Regenerate with tests/data/regenerate.py.
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _write_test_file_one_image(fname):
    shutil.copy(os.path.join(_DATA, 'one_image.fits'), fname)


def test_fits_open_single_image():
    """
    The mode argument defaults to 'r', matching the built-in
    open() convention.  Most read-only tests can be written as
    FITS(fname) with no mode argument."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        _write_test_file_one_image(fname)

        with rustfits.FITS(fname) as fits:
            print(fits)

            hdus = fits.hdus

            assert len(hdus) == 1
            assert isinstance(hdus[0], rustfits.HDU)
            assert isinstance(hdus[0], rustfits.ImageHDU)
            assert hdus[0].index == 0
            print(hdus)
            print(hdus[0])
            pprint(hdus[0].header)


def test_fits_open_default_mode_is_read_only():
    """
    Explicit-default sanity check: passing no mode gives a read-only
    handle.  Writes through that handle should fail; reopening with
    mode='r+' should let them succeed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'ro.fits')
        _write_test_file_one_image(fname)

        with rustfits.FITS(fname) as fits:
            import pytest

            with pytest.raises((OSError, IOError)):
                fits[0].header['NEWKEY'] = 1

        with rustfits.FITS(fname, 'r+') as fits:
            fits[0].header['NEWKEY'] = 1

        with rustfits.FITS(fname) as fits:
            assert fits[0].header['NEWKEY'] == 1


# ------------- empty / truncated files are rejected at open -------------


def test_fits_open_empty_file_raises_at_open():
    """
    A 0-byte file is not a FITS file: 'r' and 'r+' raise OSError at
    open (matching fitsio and astropy), not later on first HDU access.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'empty.fits')
        open(fname, 'w').close()

        for mode in ('r', 'r+'):
            with pytest.raises(OSError, match='empty'):
                rustfits.FITS(fname, mode)


def test_fits_open_short_garbage_file_raises_at_open():
    """
    A non-empty file shorter than one 2880-byte FITS block is rejected
    at open, with the byte count in the message.  (A full block of
    garbage is caught separately by header validation.)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'garbage.fits')
        with open(fname, 'wb') as fh:
            fh.write(b'this is not a FITS file')

        with pytest.raises(OSError, match='23 bytes'):
            rustfits.FITS(fname)


def test_fits_open_wplus_on_existing_empty_file_creates():
    """
    'w+' truncates/creates, so zero HDUs is its legal starting state —
    an existing empty file opens fine and can be populated.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'empty.fits')
        open(fname, 'w').close()

        with rustfits.FITS(fname, 'w+') as fits:
            assert len(fits) == 0
            fits.create_image_hdu('i4', (2, 2))

        with rustfits.FITS(fname, 'r') as fits:
            assert len(fits) == 1


def test_fits_from_bytes_empty_or_short_raises():
    """from_bytes only has read modes, so empty input always raises."""
    for blob in (b'', b'junk'):
        with pytest.raises(ValueError, match='not a valid FITS'):
            rustfits.FITS.from_bytes(blob)


def test_fits_open_mem_url_read_modes_raise():
    """
    mem:// starts as an empty buffer, so only 'w+' makes sense; the
    read modes get the same empty-file rejection as a disk file.
    """
    for mode in ('r', 'r+'):
        with pytest.raises(OSError, match='empty'):
            rustfits.FITS('mem://', mode)


if __name__ == '__main__':
    test_fits_open_single_image()
    test_fits_open_default_mode_is_read_only()
    test_fits_open_empty_file_raises_at_open()
    test_fits_open_short_garbage_file_raises_at_open()
    test_fits_open_wplus_on_existing_empty_file_creates()
    test_fits_from_bytes_empty_or_short_raises()
    test_fits_open_mem_url_read_modes_raise()
