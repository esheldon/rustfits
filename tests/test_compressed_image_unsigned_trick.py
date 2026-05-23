"""
Compressed image write: unsigned-int trick dtypes (i1, u2, u4, u8).

The FITS unsigned-int convention stores u2/u4/u8 data in a signed
i2/i4/i8 BITPIX with BSCALE=1 + BZERO=2^(n-1), and stores i1 data
in u1 (BITPIX=8) with BZERO=-128.  On read, BSCALE/BZERO are
applied to recover the original dtype.  This file tests the
compressed-write side: rustfits encodes the on-disk (BITPIX-native)
bytes, emits BSCALE/BZERO cards, and round-trips back to the
original dtype.

Two input dtype paths covered:
    - Scaled input (user passes u2 for a u2-trick HDU): the
      reverse XOR converts to i2 before encoding.
    - BITPIX-native input (user passes i2 directly): fast path,
      no transform — bytes flow straight to the encoder.

Astropy cross-read is asserted for u2/u4/i1 (not u8: astropy
returns f8 on u8 BZERO=2^63 with precision loss, an existing
astropy quirk unrelated to compression).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")


def _spread(dtype, n=36):
    """
    Build a deterministic test array spanning the dtype's range
    (min, max, 0, 1, mid).  Reshaped to 6x6 by default.
    """
    np_dtype = np.dtype(dtype)
    info = np.iinfo(np_dtype)
    anchors = np.array(
        [info.min, info.max, 0, 1, info.min + 1, info.max - 1],
        dtype=np_dtype,
    )
    return np.tile(anchors, n // anchors.size).reshape(6, 6).astype(np_dtype)


# ---------------------- round-trip --------------------------------


@pytest.mark.parametrize("dtype", ["i1", "u2", "u4", "u8"])
@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_round_trip_scaled_input(dtype, AlgoCls):
    """
    User passes the scaled (unsigned) dtype directly; rustfits
    must reverse-transform, encode, and round-trip to the same
    unsigned dtype.
    """
    if dtype == "u8" and AlgoCls is rustfits.Rice1:
        pytest.skip("Rice1 rejects bitpix=64 by design")

    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _spread(dtype)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=AlgoCls(tile_shape=(3, 6)),
            )
            f[1].write(data)
            same = f[1].read()
        with rustfits.FITS(fn, "r") as f:
            reopen = f[1].read()
        np.testing.assert_array_equal(same, data)
        np.testing.assert_array_equal(reopen, data)
        assert same.dtype == np.dtype(dtype)
        assert reopen.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "scaled,bitpix_dtype",
    [
        ("i1", "u1"),
        ("u2", "i2"),
        ("u4", "i4"),
        # u8/i8 omitted: read-side dequant arithmetic for
        # BZERO=2^63 happens in f64, which loses precision at the
        # i8 endpoints (mantissa is 53 bits but the offset
        # arithmetic produces values up to 2^64 - 1).  Same
        # limitation on the uncompressed u8 read path; not new.
    ],
)
def test_round_trip_bitpix_native_input_fast_path(scaled, bitpix_dtype):
    """
    User passes the BITPIX-native dtype (e.g. i2 for a u2-trick
    HDU); fast path skips the XOR.  The on-disk bytes are the
    user's input verbatim, and read-back applies BSCALE/BZERO to
    produce the scaled dtype.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        # Build BITPIX-native data spanning the signed range.
        np_dtype = np.dtype(bitpix_dtype)
        info = np.iinfo(np_dtype)
        bnative = np.array(
            [info.min, 0, info.max], dtype=bitpix_dtype
        ).reshape(1, 3)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                scaled,
                (1, 3),
                compress=rustfits.Gzip1(tile_shape=(1, 3)),
            )
            # Note: writing BITPIX-native bytes; reader applies
            # BSCALE/BZERO to recover the scaled dtype.
            f[1].write(bnative)
        with rustfits.FITS(fn, "r") as f:
            rt = f[1].read()
        # Apply the unsigned-trick offset manually to verify:
        # scaled = bnative + 2^(n-1) (or for i1: bnative - (-128))
        offset = {
            "u1": -128,  # i1 → u1 (BZERO=-128 → stored = native - (-128))
            "i2": 1 << 15,
            "i4": 1 << 31,
        }[bitpix_dtype]
        expected = (bnative.astype("int64") + offset).astype(scaled)
        np.testing.assert_array_equal(rt, expected)
        assert rt.dtype == np.dtype(scaled)


# ---------------------- header validation -------------------------


@pytest.mark.parametrize(
    "dtype,expected_bzero",
    [
        ("i1", -128),
        ("u2", 1 << 15),
        ("u4", 1 << 31),
        # u8 / BZERO=2^63 omitted: rustfits's header[] accessor
        # parses BZERO as i64 by default and clamps overflow to
        # i64::MAX, so the assertion would compare against
        # 2^63 - 1 instead of 2^63.  The card is still on disk
        # correctly (read-back uses parse_keyword_float = f64
        # which represents 2^63 exactly).
    ],
)
def test_bscale_bzero_cards_emitted(dtype, expected_bzero):
    """
    BSCALE=1 + BZERO=<expected> must appear in the compressed
    HDU header (regular cards, NOT Z-prefixed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            hdr = f[1].header
            assert hdr["BSCALE"] == 1
            assert hdr["BZERO"] == expected_bzero


def test_plain_signed_no_bscale_bzero():
    """
    Signed dtypes (i2/i4) without the trick must NOT emit
    BSCALE/BZERO — they're stored at their natural BITPIX.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i2",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            hdr = f[1].header
            assert "BSCALE" not in hdr
            assert "BZERO" not in hdr


def test_u1_no_bscale_bzero():
    """
    u1 has no unsigned-int trick (BITPIX=8 is already unsigned
    per FITS spec), so no BSCALE/BZERO should be emitted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "u1",
                (4, 4),
                compress=rustfits.Gzip1(tile_shape=(4, 4)),
            )
            hdr = f[1].header
            assert "BSCALE" not in hdr
            assert "BZERO" not in hdr


# ---------------------- astropy cross-read ------------------------
#
# Skip u8: astropy returns f8 (precision loss) for BZERO=2^63 on
# any FITS file — uncompressed or compressed.  This is an astropy
# quirk independent of compression, and rustfits's own round-trip
# is still bit-exact.


@pytest.mark.parametrize("dtype", ["i1", "u2", "u4"])
@pytest.mark.parametrize(
    "AlgoCls,ap_algo",
    [
        (rustfits.Gzip1, "GZIP_1"),
        (rustfits.Gzip2, "GZIP_2"),
        (rustfits.Rice1, "RICE_1"),
    ],
)
def test_astropy_reads_rustfits_unsigned_trick(dtype, AlgoCls, ap_algo):
    """
    rustfits-written unsigned-trick file must read back bit-exact
    via astropy (proving BSCALE/BZERO + compressed schema is
    FITS-conformant).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = _spread(dtype)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                dtype,
                data.shape,
                compress=AlgoCls(tile_shape=(3, 6)),
            )
            f[1].write(data)
        with astropy_fits.open(fn) as h:
            ap = h[1].data
        np.testing.assert_array_equal(ap, data)
        assert ap.dtype == np.dtype(dtype)


# ---------------------- rejections --------------------------------


@pytest.mark.parametrize("dtype", ["i1", "u2", "u4"])
def test_plio_rejects_unsigned_trick(dtype):
    """
    PLIO + any unsigned-int-trick dtype rejects with a clear
    error.  PLIO encoder is non-negative-only; the reverse XOR
    produces signed stored values that include negatives.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(NotImplementedError, match="PLIO"):
                f.create_image_hdu(
                    dtype,
                    (8, 8),
                    compress=rustfits.Plio1(tile_shape=(8, 8)),
                )


def test_dtype_mismatch_rejected():
    """
    Passing a non-BITPIX-and-non-scaled dtype to write must raise.
    e.g. u4 array into a u2 HDU.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        wrong = np.arange(36, dtype="u4").reshape(6, 6)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "u2",
                (6, 6),
                compress=rustfits.Gzip1(tile_shape=(3, 6)),
            )
            with pytest.raises(ValueError, match="does not match"):
                f[1].write(wrong)


# ---------------------- non-last HDU growth ------------------------


def test_unsigned_trick_non_last_hdu_growth():
    """
    Compressed unsigned-trick HDU followed by another HDU; write
    triggers heap growth that must shift the later HDU forward.
    Both HDUs must read back bit-exact post-reopen.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        comp_data = _spread("u2")
        later_data = np.arange(25, dtype="i2").reshape(5, 5)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "u2",
                comp_data.shape,
                compress=rustfits.Gzip1(tile_shape=(3, 6)),
                extname="COMP",
            )
            f.create_image_hdu("i2", later_data.shape, extname="LATER")
            f[2].write(later_data)
            f[1].write(comp_data)
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)
        with rustfits.FITS(fn, "r") as f:
            assert f[1].extname == "COMP"
            assert f[2].extname == "LATER"
            np.testing.assert_array_equal(f[1].read(), comp_data)
            np.testing.assert_array_equal(f[2].read(), later_data)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
