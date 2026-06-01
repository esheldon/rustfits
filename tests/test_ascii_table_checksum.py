"""
FITS Checksum Convention for AsciiTableHDU.

Same four methods as TableHDU / ImageHDU:
    add_datasum    — compute DATASUM from data, update card.
    add_checksum   — compute DATASUM + CHECKSUM, update both.
    verify_datasum  — True / False / None.
    verify_checksum — True / False / None.

The shared `checksum_hdu_*` helpers in `src/hdu_image.rs` are
HDU-type-agnostic — they stream the padded data section.  For an
ASCII table that means NAXIS1×NAXIS2 bytes of text + ASCII-space
pad to the 2880-byte block.  Cross-tool verification uses astropy
and fitsio to confirm rustfits's stored values are correct.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


DT = np.dtype([("ID", "i8"), ("FLUX", "f4"), ("NAME", "S6")])


def _make_rows(n):
    arr = np.zeros(n, dtype=DT)
    arr["ID"] = np.arange(n, dtype="i8")
    arr["FLUX"] = np.arange(n, dtype="f4") * 0.5
    arr["NAME"] = [f"r{i:04d}".encode() for i in range(n)]
    return arr


def _create_ascii(fname, nrows):
    arr = _make_rows(nrows)
    with rustfits.FITS(fname, "w+") as f:
        f.create_ascii_table_hdu(DT, nrows=nrows)
        f[1].write(arr)
    return arr


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip():
    """add_checksum + verify both pass; survives a reopen."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 5)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
        with rustfits.FITS(fn) as f:
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True


def test_add_datasum_only():
    """add_datasum leaves CHECKSUM absent (verify returns None for it)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 3)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_datasum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is None  # not written
            assert "DATASUM" in f[1].header
            assert "CHECKSUM" not in f[1].header


def test_verify_returns_none_when_absent():
    """A fresh table has no DATASUM/CHECKSUM cards."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 3)
        with rustfits.FITS(fn) as f:
            assert f[1].verify_datasum() is None
            assert f[1].verify_checksum() is None


# ---------------------------------------------------------------------------
# Stale-after-mutation: rustfits does NOT auto-refresh
# ---------------------------------------------------------------------------


def test_setitem_invalidates_checksum():
    """After __setitem__, the previous checksum no longer matches."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 5)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_checksum()
            assert f[1].verify_checksum() is True
            # Mutate a single cell — checksum is now stale.
            f[1]["ID"][2] = 9999
            assert f[1].verify_checksum() is False
            assert f[1].verify_datasum() is False
            # Re-add brings it back.
            f[1].add_checksum()
            assert f[1].verify_checksum() is True


def test_append_invalidates_checksum():
    """After append, the previous checksum no longer matches."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 3)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            f[1].append(_make_rows(2))
            assert f[1].verify_datasum() is False
            f[1].add_checksum()
            assert f[1].verify_datasum() is True


# ---------------------------------------------------------------------------
# Cross-tool: astropy + fitsio see the same values
# ---------------------------------------------------------------------------


def test_astropy_verifies_what_we_write():
    astropy_fits = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 4)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_checksum()
        # astropy's checksum=True open mode verifies cards on read.
        with astropy_fits.open(fn, checksum=True) as hdul:
            # No exceptions raised = valid checksums.
            assert hdul[1].header["DATASUM"] is not None
            assert hdul[1].header["CHECKSUM"] is not None


def test_fitsio_verifies_what_we_write():
    fitsio = pytest.importorskip("fitsio")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        _create_ascii(fn, 4)
        with rustfits.FITS(fn, "r+") as f:
            f[1].add_checksum()
        with fitsio.FITS(fn) as f:
            # fitsio's verify_checksum returns None on success
            # (raises an explicit error on failure).
            assert f[1].verify_checksum() is None


def test_read_astropy_written_checksum():
    """rustfits verifies an astropy-written CHECKSUM."""
    astropy_fits = pytest.importorskip("astropy.io.fits")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        # Build an ASCII table via astropy.
        cols = astropy_fits.ColDefs(
            [
                astropy_fits.Column(
                    name="ID",
                    format="I8",
                    array=np.array([1, 2, 3], dtype="i8"),
                ),
                astropy_fits.Column(
                    name="FLUX",
                    format="F10.4",
                    array=np.array([0.5, 1.5, 2.5]),
                ),
            ],
            ascii=True,
        )
        hdu = astropy_fits.TableHDU.from_columns(cols)
        primary = astropy_fits.PrimaryHDU()
        hdul = astropy_fits.HDUList([primary, hdu])
        hdul.writeto(fn, checksum=True)
        with rustfits.FITS(fn) as f:
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
