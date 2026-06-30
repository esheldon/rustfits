import os
import shutil
import tempfile
import rustfits
from pprint import pprint

# Canonical fixtures committed under tests/data/ (cfitsio-produced
# reference bytes).  See the note in test_fits_open.py / the docstring
# in tests/data/regenerate.py for why these are frozen files rather
# than fitsio-written at runtime.
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _write_test_file_two_images(fname):
    shutil.copy(os.path.join(_DATA, 'two_images.fits'), fname)


def _write_test_file_two_images_and_table(fname):
    shutil.copy(os.path.join(_DATA, 'two_images_table.fits'), fname)


def test_fits_open_two_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        _write_test_file_two_images(fname)

        with rustfits.FITS(fname, 'r') as fits:
            print(fits)

            hdus = fits.hdus

            assert len(hdus) == 2
            assert isinstance(hdus[0], rustfits.HDU)
            assert isinstance(hdus[0], rustfits.ImageHDU)
            assert hdus[0].index == 0
            assert isinstance(hdus[1], rustfits.HDU)
            assert isinstance(hdus[1], rustfits.ImageHDU)
            assert hdus[1].index == 1


def test_fits_open_two_images_and_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        _write_test_file_two_images_and_table(fname)

        with rustfits.FITS(fname, 'r') as fits:
            print(fits)

            hdus = fits.hdus

            assert len(hdus) == 3
            print(hdus)
            print(hdus[0])

            assert isinstance(hdus[0], rustfits.HDU)
            assert isinstance(hdus[0], rustfits.ImageHDU)
            assert hdus[0].index == 0
            assert isinstance(hdus[1], rustfits.HDU)
            assert isinstance(hdus[1], rustfits.ImageHDU)
            assert hdus[1].index == 1
            assert isinstance(hdus[2], rustfits.HDU)
            assert isinstance(hdus[2], rustfits.TableHDU)
            assert hdus[2].index == 2
            pprint(hdus[2].header)
            pprint(hdus[2].header)


if __name__ == '__main__':
    test_fits_open_two_images_and_table()
