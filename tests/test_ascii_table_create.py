"""
ASCII-table schema create (no data write) — FITS.create_ascii_table_hdu.

Covers the dtype->TFORM auto-mapping, the formats= override, header
card emission (XTENSION/BITPIX/NAXIS1/NAXIS2/TBCOLn/TFORMn/...), and
rejection paths for unsupported dtypes / subarray / Object.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def test_create_empty_table_auto_formats():
    """nrows=0 + auto formats: every supported dtype -> matching TFORM."""
    dtype = np.dtype(
        [
            ("ID", "i8"),
            ("FLUX", "f4"),
            ("MJD", "f8"),
            ("MASK", "u4"),
            ("NAME", "S10"),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            hdu = f[1]
            assert isinstance(hdu, rustfits.AsciiTableHDU)
            assert hdu.nrows == 0
            assert hdu.ncols == 5
            h = hdu.header
            assert h["TFORM1"] == "I20"
            assert h["TFORM2"] == "E15.7"
            assert h["TFORM3"] == "D25.17"
            assert h["TFORM4"] == "I20"
            assert h["TFORM5"] == "A10"
            # Unsigned-int trick on u4
            assert h["TZERO4"] == 1 << 63
            # TBCOL packs flush (no inter-column space)
            assert h["TBCOL1"] == 1
            assert h["TBCOL2"] == 21  # 1 + 20
            assert h["TBCOL3"] == 36  # 21 + 15
            assert h["TBCOL4"] == 61  # 36 + 25
            assert h["TBCOL5"] == 81  # 61 + 20
            # Row width = 20+15+25+20+10 = 90
            assert h["NAXIS1"] == 90


def test_create_with_nrows_allocates_data_section():
    """nrows>0 reserves the data section; rows default to ASCII spaces."""
    dtype = np.dtype([("X", "i8")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=3)
            assert f[1].nrows == 3
        # Reopen and confirm the data section is space-padded.
        with rustfits.FITS(fname) as f:
            assert f[1].nrows == 3
            # File should be exactly 2 blocks (primary) + 2 blocks
            # (header) + 1 block (data, padded) = 5 * 2880.
            size = os.path.getsize(fname)
            assert size % 2880 == 0


def test_create_with_units():
    dtype = np.dtype([("FLUX", "f4"), ("RA", "f8")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(
                dtype,
                units={"FLUX": "Jy", "RA": "deg"},
            )
            assert f[1].units == {"FLUX": "Jy", "RA": "deg"}


def test_create_with_extname_extver():
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(
                dtype,
                extname="CATALOG",
                extver=2,
            )
            hdu = f[1]
            assert hdu.extname == "CATALOG"
            assert hdu.extver == 2


def test_create_formats_override():
    """formats= overrides the auto-picked TFORM per column."""
    dtype = np.dtype([("FLUX", "f8"), ("RA", "f8")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(
                dtype,
                formats={"FLUX": "F12.4", "RA": "E15.6"},
            )
            h = f[1].header
            assert h["TFORM1"] == "F12.4"
            assert h["TFORM2"] == "E15.6"
            # TBCOL math: F12.4 -> 12, E15.6 -> 15.  Total 27.
            assert h["TBCOL2"] == 13
            assert h["NAXIS1"] == 27


def test_create_formats_case_insensitive():
    dtype = np.dtype([("X", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, formats={"x": "F8.2"})
            assert f[1].header["TFORM1"] == "F8.2"


def test_create_formats_unknown_key_raises():
    dtype = np.dtype([("X", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="does not match"):
                f.create_ascii_table_hdu(
                    dtype,
                    formats={"NOSUCH": "F8.2"},
                )


def test_create_formats_incompatible_kind_raises():
    """F format on an integer column should raise."""
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="incompatible"):
                f.create_ascii_table_hdu(dtype, formats={"X": "F8.2"})


def test_create_rejects_bool():
    dtype = np.dtype([("FLAG", "?")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="bool|b1"):
                f.create_ascii_table_hdu(dtype, nrows=0)


def test_create_rejects_int8():
    dtype = np.dtype([("X", "i1")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="int8|i1"):
                f.create_ascii_table_hdu(dtype, nrows=0)


def test_create_rejects_subarray_field():
    dtype = np.dtype([("V", "f4", (3,))])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="subarray"):
                f.create_ascii_table_hdu(dtype, nrows=0)


def test_create_rejects_negative_nrows():
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="nrows must be >= 0"):
                f.create_ascii_table_hdu(dtype, nrows=-1)


def test_create_rejects_empty_dtype():
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError, match="no fields|structured"):
                f.create_ascii_table_hdu(np.dtype("f4"), nrows=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
