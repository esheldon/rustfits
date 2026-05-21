"""
Tests for ImageHDU general BSCALE/BZERO reverse-scaling on write.

Covers:
    - write() / __setitem__ / extend() accept f8 (physical) input
      and reverse-transform to BITPIX-native dtype via
      stored = (physical - BZERO) / BSCALE
    - Round-trip exactness through the scaled write path
    - Integer BITPIX: half-to-even rounding, NaN/Inf rejection,
      overflow rejection on both ends, post-round bounds catch
    - Float BITPIX (-32) with non-trivial scaling: no rounding,
      cast precision only
    - BITPIX-native fast-path still works on scaled HDUs (no
      regression from the new branch)
    - Error message mentions the f8 alternative when wrong dtype
      is passed
    - f4 input to a scaled int HDU is rejected (only f8 triggers
      the reverse-transform branch)
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_scaled(tmpdir, dtype, dims, bscale, bzero, fill=None):
    """
    Create a one-HDU FITS file with the given BITPIX dtype/dims and
    BSCALE/BZERO header cards.  Optionally pre-write `fill` (stored
    values) so reads have content.  Returns the path.
    """
    fname = os.path.join(tmpdir, "t.fits")
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_image_hdu(dtype=dtype, dims=dims)
        fits[0].header["BSCALE"] = bscale
        fits[0].header["BZERO"] = bzero
        if fill is not None:
            fits[0].write(np.asarray(fill, dtype=dtype))
    return fname


# -------------------- round-trip via write() ------------------------


def test_int_roundtrip_via_f8_write():
    """
    BITPIX=16 with BSCALE=0.5, BZERO=10.  Physical input that maps
    to exact integer stored values round-trips losslessly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(5,), bscale=0.5, bzero=10.0,
        )
        physical = np.array(
            [10.0, 10.5, 11.0, 11.5, 12.0], dtype="f8",
        )
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            got = fits[0].read(scale=True)
        with rustfits.FITS(fname, "r") as fits:
            got2 = fits[0].read(scale=True)
        np.testing.assert_array_equal(got, physical)
        np.testing.assert_array_equal(got2, physical)


def test_int_stored_values_match_expected():
    """
    Check the actual stored bytes (read with scale=False) match
    the expected (physical - BZERO) / BSCALE.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(4,), bscale=2.0, bzero=0.0,
        )
        physical = np.array([2.0, 4.0, 6.0, 8.0], dtype="f8")
        expected_stored = np.array([1, 2, 3, 4], dtype="i2")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(stored, expected_stored)


def test_native_dtype_still_works_on_scaled_hdu():
    """
    Fast path: passing the BITPIX-native dtype to a scaled HDU
    writes the bytes verbatim (no reverse-transform).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=10.0, bzero=0.0,
        )
        stored = np.array([7, 8, 9], dtype="i2")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(stored)
            got_raw = fits[0].read(scale=False)
            got_scaled = fits[0].read(scale=True)
        np.testing.assert_array_equal(got_raw, stored)
        np.testing.assert_array_equal(
            got_scaled, stored.astype("f8") * 10.0,
        )


# -------------------- rounding (half-to-even) -----------------------


def test_half_to_even_rounding():
    """
    Halfway values round to the nearest even (banker's rounding)
    via np.rint — matches numpy default.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # BSCALE=2 + BZERO=0 → kind=General; stored = physical / 2.
        fname = _make_scaled(
            tmpdir, "i2", dims=(5,), bscale=2.0, bzero=0.0,
        )
        # Physical 1.0 → stored 0.5 → 0 (even); 3.0 → 1.5 → 2;
        # 5.0 → 2.5 → 2; 7.0 → 3.5 → 4; 9.0 → 4.5 → 4.
        physical = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype="f8")
        expected_stored = np.array([0, 2, 2, 4, 4], dtype="i2")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(stored, expected_stored)


def test_negative_half_to_even_rounding():
    """Half-to-even with negative values too."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(4,), bscale=2.0, bzero=0.0,
        )
        # -1.0 → -0.5 → 0; -3.0 → -1.5 → -2; -5.0 → -2.5 → -2;
        # -7.0 → -3.5 → -4.
        physical = np.array([-1.0, -3.0, -5.0, -7.0], dtype="f8")
        expected_stored = np.array([0, -2, -2, -4], dtype="i2")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(stored, expected_stored)


# -------------------- overflow rejection ---------------------------


def test_overflow_int_upper_raises():
    """BITPIX=16 + value beyond 32767 stored raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # BSCALE=1, BZERO=1 → kind=General; stored = physical - 1.
        fname = _make_scaled(
            tmpdir, "i2", dims=(2,), bscale=1.0, bzero=1.0,
        )
        # Physical 100000 → stored 99999 → out of i2 range.
        physical = np.array([1.0, 100000.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="overflow"):
                fits[0].write(physical)


def test_overflow_int_lower_raises():
    """BITPIX=8 (unsigned) + negative value raises."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "u1", dims=(2,), bscale=1.0, bzero=10.0,
        )
        # Physical 9 → stored -1 → out of u1 range [0, 255].
        physical = np.array([15.0, 9.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="overflow"):
                fits[0].write(physical)


def test_overflow_caught_after_rounding():
    """
    Post-rounding bounds check: stored value rounds to 32768 which
    overflows i2.  Raise, don't wrap.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # BSCALE=1, BZERO=1 → stored = physical - 1.
        fname = _make_scaled(
            tmpdir, "i2", dims=(1,), bscale=1.0, bzero=1.0,
        )
        # Physical 32768.6 → stored 32767.6 → rint → 32768 → overflow.
        physical = np.array([32768.6], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="overflow"):
                fits[0].write(physical)


# -------------------- NaN / Inf rejection --------------------------


def test_nan_raises_on_int_bitpix():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=0.5, bzero=10.0,
        )
        physical = np.array([10.0, np.nan, 11.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="non-finite"):
                fits[0].write(physical)


def test_inf_raises_on_int_bitpix():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i4", dims=(2,), bscale=2.0, bzero=0.0,
        )
        physical = np.array([1.0, np.inf], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="non-finite"):
                fits[0].write(physical)


def test_neg_inf_raises_on_int_bitpix():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i4", dims=(2,), bscale=2.0, bzero=0.0,
        )
        physical = np.array([1.0, -np.inf], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="non-finite"):
                fits[0].write(physical)


# -------------------- float BITPIX with scaling --------------------


def test_float_bitpix_minus32_with_scaling_roundtrips():
    """
    BITPIX=-32, BSCALE=2, BZERO=5.  f8 input reverse-transforms to
    f4, no rounding/bounds, round-trip exact within f4 precision.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "f4", dims=(4,), bscale=2.0, bzero=5.0,
        )
        physical = np.array([5.0, 7.0, 9.0, 11.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            got = fits[0].read(scale=True)
        # Allow f4 precision (~1e-7 rel).  Inputs chosen to round-
        # trip exactly: stored = (physical - 5) / 2 → 0, 1, 2, 3.
        np.testing.assert_allclose(got, physical, rtol=1e-6)


def test_float_bitpix_allows_nan():
    """
    Float BITPIX: NaN is fine (no rounding/bounds checks).  Stored
    as NaN in f4, read back as NaN.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "f4", dims=(3,), bscale=2.0, bzero=5.0,
        )
        physical = np.array([5.0, np.nan, 7.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            got = fits[0].read(scale=True)
        assert np.isnan(got[1])
        np.testing.assert_allclose(got[[0, 2]], [5.0, 7.0], rtol=1e-6)


# -------------------- __setitem__ with f8 RHS ----------------------


def test_setitem_slice_with_f8_on_scaled_hdu():
    """__setitem__ on a scaled HDU accepts f8 RHS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(6,), bscale=0.5, bzero=10.0,
            fill=[0, 0, 0, 0, 0, 0],
        )
        with rustfits.FITS(fname, "r+") as fits:
            fits[0][1:4] = np.array(
                [10.5, 11.0, 11.5], dtype="f8",
            )
            got = fits[0].read(scale=True)
        # First, last two untouched; middle three set.
        # Original stored 0 → physical 0*0.5+10 = 10.0.
        expected = np.array(
            [10.0, 10.5, 11.0, 11.5, 10.0, 10.0], dtype="f8",
        )
        np.testing.assert_array_equal(got, expected)


# -------------------- extend with f8 input -------------------------


def test_extend_with_f8_on_scaled_hdu():
    """ImageHDU.extend accepts f8 input on a scaled HDU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=0.5, bzero=10.0,
            fill=[0, 0, 0],
        )
        with rustfits.FITS(fname, "r+") as fits:
            # extend writes starting at index 3, growing to length 6.
            fits[0].extend(
                np.array([10.5, 11.0, 11.5], dtype="f8"),
                start=[3],
            )
            got = fits[0].read(scale=True)
        expected = np.array(
            [10.0, 10.0, 10.0, 10.5, 11.0, 11.5], dtype="f8",
        )
        np.testing.assert_array_equal(got, expected)


# -------------------- error messages -------------------------------


def test_error_message_mentions_f8_for_general_scaled():
    """
    Mismatched dtype on a generally-scaled HDU should mention the
    'or scaled f8' alternative.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=0.5, bzero=10.0,
        )
        bogus = np.array([1, 2, 3], dtype="i4")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(
                ValueError,
                match="scaled 'f8'",
            ):
                fits[0].write(bogus)


def test_f4_input_rejected_on_scaled_int_hdu():
    """
    Only f8 triggers the general-scaling reverse branch.  f4 is
    rejected with the standard dtype-mismatch message.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=0.5, bzero=10.0,
        )
        # f4 doesn't match i2 (fast path) and doesn't match f8
        # (general-scaling path) — rejected.
        bogus = np.array([10.0, 10.5, 11.0], dtype="f4")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="does not match"):
                fits[0].write(bogus)


# -------------------- BITPIX=64 with scaling -----------------------


def test_bitpix_64_with_general_scaling_roundtrips():
    """
    BITPIX=64 with non-trivial scaling.  Values well within f64
    precision round-trip cleanly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i8", dims=(3,), bscale=1000.0, bzero=0.0,
        )
        physical = np.array(
            [1000.0, 2000.0, 3000.0], dtype="f8",
        )
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].write(physical)
            got = fits[0].read(scale=True)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(got, physical)
        np.testing.assert_array_equal(
            stored, np.array([1, 2, 3], dtype="i8"),
        )


# -------------------- failed write leaves file untouched -----------


def test_nan_rejection_leaves_file_untouched():
    """
    NaN rejection happens in normalize_input_dtype, before any
    file mutation.  Pre-existing stored values stay intact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(3,), bscale=0.5, bzero=10.0,
            fill=[7, 8, 9],
        )
        bad = np.array([10.0, np.nan, 11.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].write(bad)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(
            stored, np.array([7, 8, 9], dtype="i2"),
        )


def test_overflow_rejection_leaves_file_untouched():
    """Overflow rejection also pre-mutation; stored values intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _make_scaled(
            tmpdir, "i2", dims=(2,), bscale=1.0, bzero=1.0,
            fill=[5, 6],
        )
        bad = np.array([1.0, 100000.0], dtype="f8")
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].write(bad)
            stored = fits[0].read(scale=False)
        np.testing.assert_array_equal(
            stored, np.array([5, 6], dtype="i2"),
        )
