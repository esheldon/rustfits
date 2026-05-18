import os
import tempfile
import numpy as np
import rustfits
from pprint import pprint


def _write_test_file_two_images(fname):
    import fitsio

    with fitsio.FITS(fname, 'rw', clobber=True) as fits:
        data1 = np.arange(5 * 20, dtype='i4').reshape(5, 20)
        fits.write(data1, extname='image1')

        data2 = np.arange(3 * 15, dtype='f8').reshape(3, 15)
        fits.write(data2, extname='image2')


def _write_test_file_two_images_and_table(fname):
    import fitsio

    with fitsio.FITS(fname, 'rw', clobber=True) as fits:
        data1 = np.arange(5 * 20, dtype='i4').reshape(5, 20)
        fits.write(data1, extname='image1')

        data2 = np.arange(3 * 15, dtype='f8').reshape(3, 15)
        fits.write(data2, extname='image2')

        tab1 = np.zeros(3, dtype=[('index', 'i4'), ('x', 'f4')])
        tab1['index'] = np.arange(tab1.size)
        tab1['x'] = [8, -2.25, 5.51]
        fits.write(tab1, extname='table1')


def test_open_two_images():
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


def test_open_two_images_and_table():
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
    test_open_two_images_and_table()
