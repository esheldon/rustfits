"""
Tests for `image_hdu[key] = scalar` on HDUs configured with
BSCALE/BZERO scaling (unsigned-int trick OR general).

Symmetric with the ndarray RHS path: the scalar value is given in
USER-FACING space (e.g. the u2 value on a BITPIX=16 + BZERO=32768
HDU, or the physical f8 value on a generally-scaled HDU), and the
reverse transform runs in flight.  This closes the gap where the
old scalar path only accepted BITPIX-native (stored-space) values.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unsigned_trick(tmpdir, dtype, dims):
    """Create a one-HDU file with the unsigned-int trick."""
    fname = os.path.join(tmpdir, "u.fits")
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=dims)
    return fname


def _make_general_scaled(tmpdir, dtype, dims, bscale, bzero):
    """
    Create a one-HDU file with general BSCALE/BZERO scaling (i.e. not
    the unsigned-int trick).
    """
    fname = os.path.join(tmpdir, "g.fits")
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=dims)
        fits[0].header["BSCALE"] = bscale
        fits[0].header["BZERO"] = bzero
    return fname


# ---------------------------------------------------------------------------
# Unsigned-int trick: scalar broadcast accepts scaled-space values
# ---------------------------------------------------------------------------


def test_unsigned_trick_u2_scalar_above_i2_range():
    """
    `img[k] = 50000` on a u2 HDU (BITPIX=16, BZERO=32768) — value
    is outside i2 range but valid u2, used to raise OverflowError
    on the BITPIX-native fast path.  Now accepted as a u2 value.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (4,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1] = 50000
            assert int(fits[0][1]) == 50000
        with rustfits.FITS(fname, "r") as fits:
            assert int(fits[0][1]) == 50000


def test_unsigned_trick_u2_scalar_broadcast_full():
    """`img[:] = 60000` broadcasts the u2 value to every pixel."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3, 4))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][:] = 60000
            got = fits[0].read()
            assert got.dtype == np.uint16
            assert np.all(got == 60000)


def test_unsigned_trick_u2_scalar_broadcast_slice():
    """Mid-range u2 value broadcasts to a row slice."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (4, 3))
        with rustfits.FITS(fname, "r+") as fits:
            # Pre-fill with u2 zero so unwritten rows compare against
            # u2(0); the file's initial zero bytes read back as 32768
            # in u2 space (the unsigned-trick bias) and would confuse
            # the assertion below.
            fits[0].write(np.zeros((4, 3), dtype="u2"))
            fits[0][1:3, :] = 40000
            got = fits[0].read()
            assert np.all(got[1:3] == 40000)
            assert np.all(got[0] == 0)
            assert np.all(got[3] == 0)


def test_unsigned_trick_u4_scalar_above_i4_range():
    """u4 HDU: scalar above i4 max accepted as a u4 value."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u4", (4,))
        big = 3_000_000_000  # > 2^31 - 1
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][2] = big
            assert int(fits[0][2]) == big
        with rustfits.FITS(fname, "r") as fits:
            assert int(fits[0][2]) == big


def test_unsigned_trick_u8_scalar_above_i8_range():
    """u8 HDU: scalar above i8 max accepted as a u8 value."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u8", (4,))
        big = (1 << 63) + 12345  # > i64 max, fits in u64
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1] = big
            assert int(fits[0][1]) == big
        with rustfits.FITS(fname, "r") as fits:
            assert int(fits[0][1]) == big


def test_unsigned_trick_i1_scalar_negative():
    """
    i1 HDU (BITPIX=8, BZERO=-128): scalar in [-128, 127] range
    works.  This is the inverse of u2/u4/u8 — the user-facing
    type is signed and the stored type is unsigned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "i1", (4,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][0] = -100
            fits[0][1] = 100
            assert int(fits[0][0]) == -100
            assert int(fits[0][1]) == 100
        with rustfits.FITS(fname, "r") as fits:
            assert int(fits[0][0]) == -100
            assert int(fits[0][1]) == 100


def test_unsigned_trick_u2_numpy_scalar_broadcasts():
    """np.uint16 scalar broadcasts the same as a Python int."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3, 3))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1, 1] = np.uint16(45000)
            assert int(fits[0][1, 1]) == 45000


def test_unsigned_trick_u2_zero_d_ndarray_broadcasts():
    """0-d u2 ndarray RHS broadcasts (ndim == 0 path)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][...] = np.array(55555, dtype="u2")
            got = fits[0].read()
            assert np.all(got == 55555)


def test_unsigned_trick_u2_scalar_overflow_raises():
    """
    Value larger than u2 max raises (NEP 50: numpy rejects on
    `asarray(value, dtype='u2')`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3,))
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises((OverflowError, ValueError)):
                fits[0][0] = 70000  # > 65535


def test_unsigned_trick_u2_scalar_negative_raises():
    """Negative value on a u2 HDU raises (out of unsigned range)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3,))
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises((OverflowError, ValueError)):
                fits[0][0] = -1


def test_unsigned_trick_u2_other_pixels_untouched():
    """
    Single-pixel scaled scalar write leaves the rest of the image
    at its previously-written values.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_unsigned_trick(tmp, "u2", (3, 3))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(np.full((3, 3), 12345, dtype="u2"))
            fits[0][1, 1] = 60000
            got = fits[0].read()
            assert got[1, 1] == 60000
            # Other pixels still the pre-write value.
            assert got[0, 0] == 12345
            assert got[2, 2] == 12345


# ---------------------------------------------------------------------------
# General scaling: scalar broadcast accepts f8 (physical) values
# ---------------------------------------------------------------------------


def test_general_scaling_int_physical_scalar():
    """
    BITPIX=16, BSCALE=2, BZERO=10: scalar f8 physical value
    reverse-transforms to integer stored.  Previously the old
    scalar path would have read 42.0 at i16, succeeding silently
    with the wrong meaning.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "i2", (4,), 2.0, 10.0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1] = 42.0  # stored = (42 - 10) / 2 = 16
            stored = fits[0].read(scale=False)
            assert stored[1] == 16
            assert fits[0].read()[1] == 42.0


def test_general_scaling_int_broadcast_full():
    """f8 physical scalar broadcast across the whole image."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "i2", (3, 4), 0.5, 10.0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][:] = 12.5  # stored = (12.5 - 10) / 0.5 = 5
            stored = fits[0].read(scale=False)
            assert np.all(stored == 5)
            assert np.all(fits[0].read() == 12.5)


def test_general_scaling_int_python_int_scalar():
    """
    Python int scalar on a generally-scaled integer HDU.  The
    int is promoted to f8 (since the scaled dtype is 'f8'),
    reverse-transform applies.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "i2", (3,), 1.0, 1.0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1] = 42  # stored = 42 - 1 = 41
            stored = fits[0].read(scale=False)
            assert stored[1] == 41


def test_general_scaling_float_image_physical_scalar():
    """
    BITPIX=-32, BSCALE=2, BZERO=5: physical scalar reverse-
    transforms to f32 stored (no rounding/bounds check for floats).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "f4", (3,), 2.0, 5.0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][0] = 11.0  # stored = (11 - 5) / 2 = 3.0
            stored = fits[0].read(scale=False)
            assert stored[0] == np.float32(3.0)
            assert fits[0].read()[0] == 11.0


def test_general_scaling_int_overflow_raises():
    """
    Physical value that overflows BITPIX range after reverse-
    transform raises (uses the same shared check as the ndarray
    path via normalize_input_dtype → reverse_general_scaling).
    """
    with tempfile.TemporaryDirectory() as tmp:
        # BSCALE=2 (not 1) → ScalingKind::General; without it, the
        # path falls into the no-scaling BITPIX-native branch which
        # rejects the float scalar with TypeError instead.
        fname = _make_general_scaled(tmp, "i2", (3,), 2.0, 0.0)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="overflow"):
                fits[0][0] = 1e9  # stored = 5e8 > i2 max


def test_general_scaling_int_nonfinite_raises():
    """NaN on integer BITPIX scaled HDU raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "i2", (3,), 2.0, 0.0)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="non-finite"):
                fits[0][0] = float("nan")


def test_general_scaling_other_pixels_untouched():
    """Scalar write to one pixel leaves the rest as previously written."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _make_general_scaled(tmp, "i2", (3, 3), 1.0, 10.0)
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(np.full((3, 3), 100.0, dtype="f8"))
            fits[0][1, 1] = 200.0
            got = fits[0].read()
            assert got[1, 1] == 200.0
            assert got[0, 0] == 100.0
            assert got[2, 2] == 100.0


# ---------------------------------------------------------------------------
# No-scaling fast path: unchanged behavior
# ---------------------------------------------------------------------------


def test_no_scaling_path_unchanged():
    """
    HDU with no scaling: scalar is extracted at BITPIX-native dtype
    (existing fast path), out-of-range still raises.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "p.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i2", dims=(3,))
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][0] = 100
            assert int(fits[0][0]) == 100
            # 99999 > i16 max; native fast path still rejects.
            with pytest.raises((OverflowError, ValueError)):
                fits[0][0] = 99999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
