"""
GZIP compression level kwarg: Gzip1(level=...) and Gzip2(level=...).

Level is a write-only zlib knob that controls the compression
effort:
- 0 = no compression (just gzip framing)
- 1 = fastest, least compression
- 9 = slowest, most compression
- None (default) = codec default = level 6 (zlib/cfitsio/astropy)

The level is NOT recorded in the file — it only affects how the
encoder produces bytes; the decoder handles any level identically.
So `.compression.level` returns the user's value within the same
Python session (we store the full config on the HDU), but
`level=None` after reopen (we can't recover what the encoder used).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------- config-class API --------------------------


def test_default_level_is_none():
    """Default level is None → codec default (zlib level 6)."""
    assert rustfits.Gzip1().level is None
    assert rustfits.Gzip2().level is None


def test_level_set():
    """level=N stored on the config object and shown in repr."""
    cfg = rustfits.Gzip1(level=9)
    assert cfg.level == 9
    assert "level=9" in repr(cfg)
    cfg2 = rustfits.Gzip2(level=0)
    assert cfg2.level == 0
    assert "level=0" in repr(cfg2)


@pytest.mark.parametrize("level", [0, 1, 5, 6, 9])
def test_level_valid_range(level):
    """All zlib levels 0..=9 accepted."""
    rustfits.Gzip1(level=level)
    rustfits.Gzip2(level=level)


@pytest.mark.parametrize("level", [10, 100])
def test_level_too_large_rejected(level):
    """Levels above 9 rejected with our error message."""
    with pytest.raises(ValueError, match="0..=9"):
        rustfits.Gzip1(level=level)


def test_level_negative_rejected():
    """Negative levels rejected at the pyo3 boundary (u32 conversion)."""
    with pytest.raises(OverflowError):
        rustfits.Gzip1(level=-1)


def test_level_equality():
    """__eq__ considers level (so configs with different levels differ)."""
    assert rustfits.Gzip1(level=1) != rustfits.Gzip1(level=9)
    assert rustfits.Gzip1(level=1) == rustfits.Gzip1(level=1)
    assert rustfits.Gzip2(level=5) == rustfits.Gzip2(level=5)


# ---------------------- same-session round-trip ------------------


def test_same_session_round_trip():
    """
    .compression returns the stored config (with level preserved)
    within the same Python session.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(level=9, tile_shape=(8, 8)),
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
            comp = f[1].compression
            assert comp.level == 9
            assert comp.zcmptype == "GZIP_1"


def test_reopen_loses_level():
    """
    After close+reopen, .compression.level returns None — the level
    isn't recoverable from the gzip stream itself.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(level=9, tile_shape=(8, 8)),
            )
            f[1].write(np.arange(64, dtype="i4").reshape(8, 8))
        with rustfits.FITS(fn) as f:
            assert f[1].compression.level is None


# ---------------------- round-trip data fidelity -----------------


@pytest.mark.parametrize("level", [0, 1, 6, 9])
@pytest.mark.parametrize("AlgoCls", [rustfits.Gzip1, rustfits.Gzip2])
def test_round_trip_bit_exact_across_levels(level, AlgoCls):
    """
    The compression level affects file size but never the decoded
    data — every level must round-trip bit-exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        data = rng.integers(-1000, 1000, size=(32, 32), dtype="i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                data.shape,
                compress=AlgoCls(level=level, tile_shape=(16, 16)),
            )
            f[1].write(data)
        with rustfits.FITS(fn) as f:
            rt = f[1].read()
        np.testing.assert_array_equal(rt, data)


# ---------------------- level actually affects size --------------


def test_level_9_smaller_than_level_1_on_compressible_data():
    """
    On compressible (smooth) data, level=9 produces a smaller file
    than level=1.  Quick proxy: the file size should differ
    measurably.  (For random/incompressible data the two levels
    can be similar; we use a smooth pattern that gives gzip room
    to work.)
    """
    sizes = {}
    for level in [1, 9]:
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, "t.fits.fz")
            rng = np.random.default_rng(0)
            x, y = np.meshgrid(np.arange(500), np.arange(500))
            data = (
                1000 * np.sin(0.01 * x) * np.cos(0.01 * y)
                + rng.standard_normal((500, 500))
            ).astype("i4")
            with rustfits.FITS(fn, "w+") as f:
                f.create_image_hdu(
                    "i4",
                    data.shape,
                    compress=rustfits.Gzip1(
                        level=level, tile_shape=(100, 100)
                    ),
                )
                f[1].write(data)
            sizes[level] = os.path.getsize(fn)
    # Expect ≥ 10% reduction going from level=1 to level=9
    assert sizes[9] < sizes[1] * 0.9, (
        f"expected level=9 noticeably smaller than level=1, got "
        f"level=1={sizes[1]} bytes, level=9={sizes[9]} bytes"
    )


# ---------------------- level applies to all write paths ---------


def test_level_applies_to_extend():
    """Level used by extend() too (not just initial write)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        rng = np.random.default_rng(0)
        initial = rng.integers(-100, 100, size=32, dtype="i4")
        more = rng.integers(-100, 100, size=16, dtype="i4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (32,),
                compress=rustfits.Gzip1(level=9, tile_shape=(16,)),
            )
            f[1].write(initial)
            f[1].extend(more)
        with rustfits.FITS(fn) as f:
            rt = f[1].read()
        np.testing.assert_array_equal(rt, np.concatenate([initial, more]))


def test_level_applies_to_setitem():
    """Level used by __setitem__ too."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        data = np.arange(64, dtype="i4").reshape(8, 8)
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (8, 8),
                compress=rustfits.Gzip1(level=9, tile_shape=(4, 4)),
            )
            f[1].write(data)
            f[1][3:5, 3:5] = -1
        with rustfits.FITS(fn) as f:
            rt = f[1].read()
        expected = data.copy()
        expected[3:5, 3:5] = -1
        np.testing.assert_array_equal(rt, expected)


# ---------------------- non-GZIP algorithms ignore level ---------


def test_rice1_has_no_level():
    """Rice1 doesn't have a level kwarg (zlib not used)."""
    with pytest.raises(TypeError):
        rustfits.Rice1(level=9)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
