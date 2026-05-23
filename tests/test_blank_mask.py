"""
BLANK/ZBLANK sentinel + mask_blank read on both uncompressed and
compressed image HDUs.

API surface added in this PR:

1. **Read.**  `hdu.read(mask_blank=True)` returns a
   `numpy.ma.MaskedArray` with True at pixels whose stored value
   matches the header's `BLANK` (uncompressed) or `ZBLANK`
   (compressed).  Same semantics on both HDU types.

2. **Create.**  `create_image_hdu(..., blank=<sentinel>)` emits
   the `BLANK` card (uncompressed) or `ZBLANK` card (compressed)
   so reads with `mask_blank=True` find the sentinel.  Value is
   in PHYSICAL space (user's dtype); transformed to STORED space
   for the card when the unsigned-int trick is in play.

3. **Write.**  Two paths to populate masked pixels:
   - Pre-fill: user writes a plain ndarray with the sentinel value
     at masked positions.
   - MaskedArray input: `write`/`__setitem__`/`extend` accept
     `numpy.ma.MaskedArray`; masked positions are auto-filled with
     the sentinel from the header.  For float HDUs, NaN is used
     (no header dependency).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")
fitsio = pytest.importorskip("fitsio")


# ---------------------- create + read round-trip ------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_uncompressed_blank_round_trip(dtype):
    """
    Uncompressed: create with blank=sentinel, write data containing
    sentinel at some positions, read with mask_blank=True returns
    MaskedArray with True at those positions.
    """
    np_dtype = np.dtype(dtype)
    sentinel = int(np.iinfo(np_dtype).max) - 1
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(16, dtype=dtype).reshape(4, 4)
        data[1, 2] = sentinel
        data[3, 0] = sentinel
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(dtype, (4, 4), blank=sentinel)
            f[0].write(data)
        with rustfits.FITS(fn) as f:
            assert f[0].header["BLANK"] == sentinel
            masked = f[0].read(mask_blank=True)
            assert int(masked.mask.sum()) == 2
            assert masked.mask[1, 2]
            assert masked.mask[3, 0]
            assert not masked.mask[0, 0]


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_compressed_blank_round_trip(dtype, AlgoCls):
    """
    Compressed: same round trip via ZBLANK.
    """
    np_dtype = np.dtype(dtype)
    sentinel = int(np.iinfo(np_dtype).max) - 1
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype=dtype).reshape(8, 8)
        data[3:5, 3:5] = sentinel
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                (8, 8),
                compress=AlgoCls(tile_shape=(4, 4)),
                blank=sentinel,
            )
            f[1].write(data)
        with rustfits.FITS(fn) as f:
            assert f[1].header["ZBLANK"] == sentinel
            masked = f[1].read(mask_blank=True)
            assert int(masked.mask.sum()) == 4


def test_no_blank_returns_nomask():
    """
    When BLANK/ZBLANK is absent from the header, mask_blank=True
    still returns a MaskedArray (consistent return type) but with
    no positions masked.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (8, 8), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
        with rustfits.FITS(fn) as f:
            masked = f[1].read(mask_blank=True)
            assert type(masked).__name__ == "MaskedArray"
            assert int(np.ma.getmaskarray(masked).sum()) == 0


# ---------------------- unsigned-int trick interaction ------------


def test_unsigned_trick_blank():
    """
    For unsigned-trick dtypes (e.g. u2 stored as i2 + BZERO=32768),
    `blank=` is in PHYSICAL space.  The BLANK/ZBLANK card on disk
    is in STORED space (i.e., transformed by subtracting BZERO).
    Round-trip via mask_blank=True works against the user's
    physical dtype.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        # u2 dtype, physical sentinel 65535
        data = np.arange(16, dtype="u2").reshape(4, 4)
        data[1, 2] = 65535
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "u2",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                blank=65535,
            )
            f[1].write(data)
        with rustfits.FITS(fn) as f:
            # ZBLANK card is in stored space (i2 → 65535-32768=32767)
            assert f[1].header["ZBLANK"] == 32767
            masked = f[1].read(mask_blank=True)
            assert int(masked.mask.sum()) == 1
            assert masked.mask[1, 2]
            assert masked.dtype == np.dtype("u2")


# ---------------------- MaskedArray input ------------------------


def test_masked_array_write_compressed():
    """
    Compressed write accepts numpy.ma.MaskedArray; masked positions
    are auto-filled with the header's ZBLANK sentinel.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        ma = np.ma.MaskedArray(data, mask=mask)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                blank=-99,
            )
            f[1].write(ma)
        with rustfits.FITS(fn) as f:
            rt = f[1].read(mask_blank=True)
            assert int(rt.mask.sum()) == 9
            # Plain read shows the sentinel
            plain = f[1].read()
            assert (plain[2:5, 2:5] == -99).all()


def test_masked_array_write_uncompressed():
    """Same as above but for uncompressed."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        ma = np.ma.MaskedArray(data, mask=mask)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (4, 4), blank=-99)
            f[0].write(ma)
        with rustfits.FITS(fn) as f:
            rt = f[0].read(mask_blank=True)
            assert int(rt.mask.sum()) == 4
            assert f[0].header["BLANK"] == -99


def test_masked_array_float_fills_nan():
    """
    Float HDU + MaskedArray: masked positions fill with NaN (no
    BLANK header needed for floats).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 4)).astype("f4")
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        ma = np.ma.MaskedArray(data, mask=mask)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                quantize=None,
            )
            f[1].write(ma)
        with rustfits.FITS(fn) as f:
            rt = f[1].read()
            assert np.isnan(rt[1:3, 1:3]).all()
            assert not np.isnan(rt[0, 0])
            assert not np.isnan(rt[3, 3])


def test_masked_array_setitem_compressed():
    """
    Compressed __setitem__ accepts MaskedArray (auto-fills with
    ZBLANK).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                blank=-77,
            )
            f[1].write(data)
            # Now overwrite a 2x2 region with a masked patch
            new = np.array([[1, 2], [3, 4]], dtype="i4")
            ma = np.ma.MaskedArray(new, mask=[[True, False], [False, True]])
            f[1][3:5, 3:5] = ma
        with rustfits.FITS(fn) as f:
            rt = f[1].read(mask_blank=True)
            # [3,3] and [4,4] are masked (the True positions in ma)
            assert rt.mask[3, 3]
            assert rt.mask[4, 4]
            assert not rt.mask[3, 4]
            assert not rt.mask[4, 3]


def test_masked_array_extend_compressed():
    """
    Compressed extend accepts MaskedArray.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(16, dtype="i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (16,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
                blank=-1,
            )
            f[1].write(data)
            # Extend with a MaskedArray
            more = np.arange(16, 24, dtype="i4")
            mask = np.zeros(8, dtype=bool)
            mask[3:5] = True
            ma = np.ma.MaskedArray(more, mask=mask)
            f[1].extend(ma)
        with rustfits.FITS(fn) as f:
            rt = f[1].read(mask_blank=True)
            assert rt.shape == (24,)
            assert int(rt.mask.sum()) == 2
            assert rt.mask[19]
            assert rt.mask[20]


# ---------------------- rejection paths --------------------------


def test_blank_on_float_rejected_uncompressed():
    """blank= on float dtype must reject (spec forbids BLANK on floats)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="float"):
                f.create_image_hdu("f4", (4, 4), blank=-1)


def test_blank_on_float_rejected_compressed():
    """blank= on float compressed dtype must reject."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="float"):
                f.create_image_hdu(
                    "f4",
                    (4, 4),
                    compress=rustfits.Gzip1(tile_shape=(4, 4)),
                    blank=-1,
                )


def test_mask_blank_on_float_compressed_rejected():
    """
    mask_blank=True on float compressed must reject (matches the
    uncompressed BLANK-on-float rejection).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "f4",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                quantize=None,
            )
            f[1].write(rng.standard_normal((4, 4)).astype("f4"))
        with rustfits.FITS(fn) as f:
            with pytest.raises(ValueError, match="float"):
                f[1].read(mask_blank=True)


def test_masked_array_integer_no_header_rejected():
    """
    Integer HDU + MaskedArray + no BLANK/ZBLANK in header → clear
    error pointing user at create_image_hdu(blank=).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        ma = np.ma.MaskedArray(data, mask=np.ones((4, 4), dtype=bool))
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (4, 4), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )  # no blank=
            with pytest.raises(ValueError, match="blank"):
                f[1].write(ma)


def test_blank_out_of_range_rejected():
    """blank= value outside the BITPIX range must reject."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="range"):
                f.create_image_hdu("i2", (4, 4), blank=70000)


# ---------------------- nomask MaskedArray = no-op -----------------


def test_masked_array_with_nomask_no_blank_needed():
    """
    If the user passes a MaskedArray with the singleton nomask (or
    an all-False mask), no sentinel is needed — the helper just
    unwraps to the underlying data.  So no BLANK header is
    required.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        # MaskedArray with nomask
        ma = np.ma.MaskedArray(data)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4", (4, 4), compress=rustfits.Gzip1(tile_shape=(4, 4))
            )
            f[1].write(ma)  # should NOT raise
        with rustfits.FITS(fn) as f:
            rt = f[1].read()
            np.testing.assert_array_equal(rt, data)


# ---------------------- astropy / fitsio cross-read ---------------


def test_astropy_reads_rustfits_blank_compressed():
    """
    rustfits-written compressed file with ZBLANK reads back via
    astropy.  Astropy's convention for integer + BLANK/ZBLANK is
    to promote to float and put NaN at the sentinel positions
    (rather than returning a MaskedArray) — so we check for NaN.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        data[3:5, 3:5] = -1
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                blank=-1,
            )
            f[1].write(data)
        with astropy_fits.open(fn) as h:
            ap = h[1].data
            assert np.isnan(ap[3:5, 3:5]).all()
            assert not np.isnan(ap[0, 0])


def test_fitsio_reads_rustfits_zblank():
    """
    fitsio reads the file too; sentinel values are visible (fitsio
    doesn't auto-mask).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        data[3:5, 3:5] = -1
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
                blank=-1,
            )
            f[1].write(data)
        with fitsio.FITS(fn) as f:
            rt = f[1].read()
            assert (rt[3:5, 3:5] == -1).all()
            hdr = f[1].read_header()
            assert hdr["ZBLANK"] == -1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
