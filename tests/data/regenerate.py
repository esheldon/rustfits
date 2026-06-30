"""
Regenerate the canonical FITS fixtures used by the open/parse tests.

These ``.fits`` files are committed to the repo and read directly by
``test_fits_open.py`` / ``test_fits_open_multi.py`` so those tests need
no FITS *writer* at runtime — they exercise only rustfits's open/parse
path against a frozen, known-good reference.

The bytes are produced by fitsio (the cfitsio reference implementation),
so the fixtures are a genuine external reference, not a rustfits
round-trip.  They're frozen on purpose: an open/parse test wants a
stable input, not one that drifts with the writer.  (Cross-tool
*agreement* tests are the opposite case and stay on live fitsio.)

Freezing the fixtures also drops the runtime fitsio dependency from
these tests, which is what lets them run on platforms with no fitsio
build (Windows) — the FS / locking / path code is exactly the
platform-specific surface a Windows CI leg should cover.

Run this by hand on a machine with fitsio installed after changing any
fixture's content; commit the regenerated ``.fits`` files alongside the
test change::

    python tests/data/regenerate.py
"""

import os

import numpy as np

import fitsio

HERE = os.path.dirname(os.path.abspath(__file__))


def _one_image(fname):
    with fitsio.FITS(fname, "rw", clobber=True) as fits:
        data = np.arange(5 * 20, dtype="i4").reshape(5, 20)
        fits.write(data, extname="image1")


def _two_images(fname):
    with fitsio.FITS(fname, "rw", clobber=True) as fits:
        data1 = np.arange(5 * 20, dtype="i4").reshape(5, 20)
        fits.write(data1, extname="image1")

        data2 = np.arange(3 * 15, dtype="f8").reshape(3, 15)
        fits.write(data2, extname="image2")


def _two_images_table(fname):
    with fitsio.FITS(fname, "rw", clobber=True) as fits:
        data1 = np.arange(5 * 20, dtype="i4").reshape(5, 20)
        fits.write(data1, extname="image1")

        data2 = np.arange(3 * 15, dtype="f8").reshape(3, 15)
        fits.write(data2, extname="image2")

        tab1 = np.zeros(3, dtype=[("index", "i4"), ("x", "f4")])
        tab1["index"] = np.arange(tab1.size)
        tab1["x"] = [8, -2.25, 5.51]
        fits.write(tab1, extname="table1")


def main():
    _one_image(os.path.join(HERE, "one_image.fits"))
    _two_images(os.path.join(HERE, "two_images.fits"))
    _two_images_table(os.path.join(HERE, "two_images_table.fits"))
    print("regenerated fixtures in", HERE)


if __name__ == "__main__":
    main()
