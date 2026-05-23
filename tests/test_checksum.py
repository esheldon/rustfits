"""
FITS Checksum Convention (Pence & Seaman 2010): CHECKSUM /
DATASUM for ImageHDU + TableHDU, ZHECKSUM / ZDATASUM for
CompressedImageHDU.

Four methods on each HDU type:
    add_datasum    — compute DATASUM (or ZDATASUM) from data,
                     update card.
    add_checksum   — compute DATASUM + CHECKSUM, update both.
    verify_datasum  — True / False / None.
    verify_checksum — True / False / None.

Cross-tool verification:
    - astropy reads our uncompressed CHECKSUM correctly.
    - fitsio reads our uncompressed CHECKSUM correctly
      (`verify_checksum()` returns None on success, raises on
      failure).
    - astropy's compressed-HDU verify_checksum has its own bug
      (TypeError on internal `_compute_checksum(None)`) that
      triggers on its OWN written files too, so we don't
      cross-verify compressed against astropy.  ZDATASUM /
      ZHECKSUM values agree with what astropy writes — our
      self-verify catches corruption correctly.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")
fitsio = pytest.importorskip("fitsio")


# ---------------------- uncompressed ImageHDU ---------------------


def test_image_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (8, 8))
            f[0].write(data)
            f[0].add_checksum()
            assert f[0].verify_datasum() is True
            assert f[0].verify_checksum() is True
        with rustfits.FITS(fn) as f:
            # Survives reopen
            assert f[0].verify_datasum() is True
            assert f[0].verify_checksum() is True
            # DATASUM is the FITS-spec decimal string of the sum.
            # arange(64) BE i4 = 0+1+...+63 = 2016.
            assert f[0].header["DATASUM"] == "2016"


def test_image_add_datasum_only():
    """add_datasum doesn't touch CHECKSUM."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (4, 4))
            f[0].write(np.arange(16, dtype="i4").reshape(4, 4))
            f[0].add_datasum()
            assert f[0].verify_datasum() is True
            # CHECKSUM absent → verify_checksum returns None.
            assert f[0].verify_checksum() is None


def test_image_verify_none_when_absent():
    """verify_*_=None when the card is absent."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (4, 4))
            f[0].write(np.arange(16, dtype="i4").reshape(4, 4))
            assert f[0].verify_datasum() is None
            assert f[0].verify_checksum() is None


def test_image_corruption_detected():
    """Mutating data after add_checksum makes verify_* return False."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (8, 8))
            f[0].write(data)
            f[0].add_checksum()
        # Flip one bit in the data section (offset 2890 = early
        # into the 256-byte data block).
        with open(fn, "r+b") as f:
            f.seek(2890)
            f.write(b"\xff")
        with rustfits.FITS(fn) as f:
            assert f[0].verify_checksum() is False
            assert f[0].verify_datasum() is False


def test_astropy_verifies_rustfits_uncompressed():
    """Our CHECKSUM passes astropy's verify_checksum."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (8, 8))
            f[0].write(data)
            f[0].add_checksum()
        with astropy_fits.open(fn) as h:
            assert h[0].verify_checksum() == 1
            assert h[0].verify_datasum() == 1


def test_rustfits_verifies_astropy_uncompressed():
    """astropy's CHECKSUM passes our verify_checksum."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        hdu = astropy_fits.PrimaryHDU(data)
        hdu.add_checksum()
        hdu.writeto(fn, overwrite=True)
        with rustfits.FITS(fn) as f:
            assert f[0].verify_checksum() is True
            assert f[0].verify_datasum() is True


def test_fitsio_verifies_rustfits_uncompressed():
    """fitsio's verify_checksum (raises on failure, None on OK)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (8, 8))
            f[0].write(data)
            f[0].add_checksum()
        with fitsio.FITS(fn) as f:
            # No raise = success.
            f[0].verify_checksum()


# ---------------------- TableHDU ---------------------------------


def test_table_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        dtype = np.dtype([("a", "i4"), ("b", "f8")])
        data = np.zeros(8, dtype=dtype)
        data["a"] = np.arange(8)
        data["b"] = np.linspace(0.0, 1.0, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dtype, nrows=8)
            f[1].write(data)
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
        with rustfits.FITS(fn) as f:
            assert f[1].verify_checksum() is True


def test_table_corruption_detected():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        dtype = np.dtype([("a", "i4")])
        data = np.zeros(8, dtype=dtype)
        data["a"] = np.arange(8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dtype, nrows=8)
            f[1].write(data)
            f[1].add_checksum()
        # Corrupt data byte
        with open(fn, "r+b") as f:
            f.seek(5762)  # well inside the table data
            f.write(b"\xff")
        with rustfits.FITS(fn) as f:
            assert f[1].verify_checksum() is False


def test_astropy_verifies_rustfits_table():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        dtype = np.dtype([("a", "i4"), ("b", "f8")])
        data = np.zeros(8, dtype=dtype)
        data["a"] = np.arange(8)
        data["b"] = np.linspace(0, 1, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dtype, nrows=8)
            f[1].write(data)
            f[1].add_checksum()
        with astropy_fits.open(fn) as h:
            assert h[1].verify_checksum() == 1
            assert h[1].verify_datasum() == 1


# ---------------------- CompressedImageHDU ------------------------


def test_compressed_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
        with rustfits.FITS(fn) as f:
            # Survives reopen.
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
            # ZDATASUM of arange(64) BE i4 = 2016.
            assert f[1].header["ZDATASUM"] == "2016"


def test_compressed_zdatasum_card_corruption_detected():
    """Mutating the ZDATASUM card itself yields verify=False."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1].add_checksum()
        # Find ZDATASUM card and corrupt its value.
        with open(fn, "r+b") as f:
            buf = f.read()
        idx = buf.find(b"ZDATASUM= '")
        with open(fn, "r+b") as f:
            f.seek(idx + 11)
            f.write(b"99")  # '2016    ' → '9916    '
        with rustfits.FITS(fn) as f:
            assert f[1].verify_datasum() is False
            assert f[1].verify_checksum() is False


def test_compressed_heap_corruption_raises():
    """
    Corrupting a heap byte breaks gzip decompression — the
    verify_* methods propagate the ValueError rather than
    silently masking the problem.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1].add_checksum()
        # Corrupt a byte at a known heap offset (after the main
        # descriptor table).
        with open(fn, "r+b") as f:
            f.seek(5806)  # 30 bytes into the heap
            cur = f.read(1)
            f.seek(5806)
            f.write(bytes([cur[0] ^ 0xFF]))
        with rustfits.FITS(fn) as f:
            with pytest.raises(ValueError, match="decompression"):
                f[1].verify_datasum()


def test_compressed_verify_none_when_absent():
    """verify_*=None when ZHECKSUM/ZDATASUM cards are absent."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            assert f[1].verify_datasum() is None
            assert f[1].verify_checksum() is None


def test_compressed_add_datasum_only():
    """add_datasum on compressed: only ZDATASUM, no ZHECKSUM."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1].add_datasum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is None


@pytest.mark.parametrize(
    "AlgoCls",
    [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1],
)
def test_compressed_round_trip_algo_matrix(AlgoCls):
    """Checksum works across algorithms."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=AlgoCls(tile_shape=(4, 4))
            )
            f[1].write(data)
            f[1].add_checksum()
        with rustfits.FITS(fn) as f:
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True


def test_compressed_zdatasum_matches_uncompressed():
    """
    ZDATASUM (checksum of the conceptual uncompressed data) must
    equal what an equivalent uncompressed ImageHDU's DATASUM
    would be — the whole point of the ZDATASUM mechanism.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn_unc = os.path.join(tmp, "unc.fits")
        fn_cmp = os.path.join(tmp, "cmp.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn_unc, "w+") as f:
            f.create_image_hdu("i4", (8, 8))
            f[0].write(data)
            f[0].add_datasum()
        with rustfits.FITS(fn_cmp, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1].add_datasum()
        with rustfits.FITS(fn_unc) as f:
            unc_datasum = f[0].header["DATASUM"]
        with rustfits.FITS(fn_cmp) as f:
            cmp_zdatasum = f[1].header["ZDATASUM"]
        assert unc_datasum == cmp_zdatasum


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
