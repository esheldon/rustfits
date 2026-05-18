"""Tests for HDU data-unit size calculation.

The FITS formula is:

    data_bytes = |BITPIX|/8 * GCOUNT * (PCOUNT + Π NAXISn)

PCOUNT carries the heap size for BINTABLE variable-length array columns and
must be included in the offset to the next HDU.  GCOUNT defaults to 1; in
the rare cases where it is greater (Random Groups, or pathological headers),
it multiplies the data size.

These tests construct multi-HDU files where the buggy "image-only" formula
(ignoring PCOUNT/GCOUNT) would mis-locate a later HDU, and verify that the
reader finds it correctly.
"""

import os
import tempfile

import rustfits


def _card(text):
    assert len(text) <= 80, f"card too long ({len(text)} chars): {text!r}"
    return text.ljust(80)


def _pad_block(b):
    """Pad bytes to the next 2880-byte boundary with spaces."""
    return b + b" " * ((-len(b)) % 2880)


# --------------------------- PCOUNT ----------------------------


def test_pcount_used_for_bintable_heap():
    """A BINTABLE declares PCOUNT=5000 (variable-length heap).  The reader
    must add PCOUNT to the row-array size, otherwise it lands inside the
    heap when searching for the next HDU."""
    primary_cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    table_cards = [
        _card("XTENSION= 'BINTABLE'           / binary table"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    2"),
        _card("NAXIS1  =                   10 / row width in bytes"),
        _card("NAXIS2  =                    5 / number of rows"),
        _card("PCOUNT  =                 5000 / heap size in bytes"),
        _card("GCOUNT  =                    1"),
        _card("TFIELDS =                    1"),
        _card("TFORM1  = '1PE(100)'           / variable-length floats"),
        _card("END"),
    ]
    final_cards = [
        _card("XTENSION= 'IMAGE   '"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("EXTNAME = 'last'"),
        _card("END"),
    ]

    # bpp=1, GCOUNT=1, PCOUNT=5000, Π NAXISn = 50, so
    # data_bytes = 1 * 1 * (50 + 5000) = 5050, padded to 5760 (2 blocks).
    table_data = b"\x00" * 5760

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "bt.fits")
        with open(fname, "wb") as f:
            f.write(_pad_block("".join(primary_cards).encode("ascii")))
            f.write(_pad_block("".join(table_cards).encode("ascii")))
            f.write(table_data)
            f.write(_pad_block("".join(final_cards).encode("ascii")))

        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 3
            assert isinstance(fits.hdus[1], rustfits.TableHDU)
            assert fits.hdus[2].header["EXTNAME"] == "last"


# --------------------------- GCOUNT ----------------------------


def test_gcount_multiplies_data_size():
    """A header with GCOUNT=2 must produce a data unit twice as large as
    GCOUNT=1 would imply, even with PCOUNT=0.  Constructs an artificial
    header where this difference crosses a 2880-byte block boundary."""
    primary_cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("END"),
    ]
    # GCOUNT=2, PCOUNT=0, NAXIS1=2000, NAXIS2=1, bpp=1.
    # data_bytes = 1 * 2 * (0 + 2000) = 4000, padded to 5760 (2 blocks).
    # Without GCOUNT: 1 * 1 * 2000 = 2000, padded to 2880 (1 block).
    table_cards = [
        _card("XTENSION= 'BINTABLE'"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    2"),
        _card("NAXIS1  =                 2000"),
        _card("NAXIS2  =                    1"),
        _card("PCOUNT  =                    0"),
        _card("GCOUNT  =                    2"),
        _card("TFIELDS =                    0"),
        _card("END"),
    ]
    final_cards = [
        _card("XTENSION= 'IMAGE   '"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("EXTNAME = 'after-gcount'"),
        _card("END"),
    ]

    table_data = b"\x00" * 5760

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "gc.fits")
        with open(fname, "wb") as f:
            f.write(_pad_block("".join(primary_cards).encode("ascii")))
            f.write(_pad_block("".join(table_cards).encode("ascii")))
            f.write(table_data)
            f.write(_pad_block("".join(final_cards).encode("ascii")))

        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 3
            assert (
                fits.hdus[2].header["EXTNAME"] == "after-gcount"
            )


# --------------------------- defaults still work ----------------------------


def test_image_hdu_unaffected_by_defaults():
    """An image HDU without PCOUNT/GCOUNT keywords must still compute the
    same data size as before (GCOUNT defaults to 1, PCOUNT to 0)."""
    primary_cards = [
        _card("SIMPLE  =                    T"),
        _card("BITPIX  =                   32"),
        _card("NAXIS   =                    2"),
        _card("NAXIS1  =                   20"),
        _card("NAXIS2  =                    5"),
        _card("END"),
    ]
    final_cards = [
        _card("XTENSION= 'IMAGE   '"),
        _card("BITPIX  =                    8"),
        _card("NAXIS   =                    0"),
        _card("EXTNAME = 'after-image'"),
        _card("END"),
    ]

    # 32/8 * 20 * 5 = 400 bytes -> padded to 2880 (1 block).
    image_data = b"\x00" * 2880

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        with open(fname, "wb") as f:
            f.write(_pad_block("".join(primary_cards).encode("ascii")))
            f.write(image_data)
            f.write(_pad_block("".join(final_cards).encode("ascii")))

        with rustfits.FITS(fname, "r") as fits:
            assert len(fits.hdus) == 2
            assert (
                fits.hdus[1].header["EXTNAME"] == "after-image"
            )


if __name__ == "__main__":
    test_pcount_used_for_bintable_heap()
    test_gcount_multiplies_data_size()
    test_image_hdu_unaffected_by_defaults()
    print("all tests passed")
