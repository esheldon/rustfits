import os
import tempfile
import numpy as np
import rustfits
from pprint import pprint


def _write_test_file_one_image(fname):
    import fitsio

    with fitsio.FITS(fname, 'rw', clobber=True) as fits:
        data = np.arange(5 * 20, dtype='i4').reshape(5, 20)
        fits.write(data, extname='image1')


def test_open_single_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        _write_test_file_one_image(fname)

        with rustfits.FITS(fname, 'r') as fits:
            print(fits)

            hdus = fits.hdus

            assert len(hdus) == 1
            assert isinstance(hdus[0], rustfits.HDU)
            assert isinstance(hdus[0], rustfits.ImageHDU)
            assert hdus[0].index == 0
            print(hdus)
            print(hdus[0])
            pprint(hdus[0].header)


if __name__ == '__main__':
    test_open_single_image()
