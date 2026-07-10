"""
RICE decode regression: long unary runs crossing the 64-bit
bit-buffer boundary.

A u8 image whose flat runs jump by ~50 (a healsparse widemask)
zigzags to diffs > 100, which small-fs blocks encode as unary
runs of > 100 zero bits.  When a refill landed the decoder's
64-bit buffer in the exact state "63 zeros + terminating 1 in
bit 0", the single-shift consume (`buf <<= lz + 1`) overflowed
the shift count: debug builds panicked; release builds left a
stale bit that corrupted the rest of the stream and surfaced as
"RICE decode: unary code exceeded 1024 zeros (corrupt stream)".

These tests are pure rustfits round-trips (no fitsio needed, so
they run on every platform).  The decoder-only counterpart — a
hand-crafted stream that deterministically forces the lz=63
state — lives in the Rust unit tests in src/zimage/rice.rs.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _roundtrip_mem(arr, compress='RICE_1'):
    """
    Encode + decode through an in-memory file, return the decoded
    array."""
    with rustfits.FITS('mem://', 'w+') as f:
        f.write_image(arr, compress=compress)
        return f[1].read()


def test_u8_long_unary_run_roundtrip_disk():
    """
    The minimal reproducer found for the shift overflow: 128 u8
    zeros with a 3-pixel spike of 63 at position 61.  Pre-fix this
    panicked in debug builds and raised the misleading
    corrupt-stream ValueError in release builds."""
    arr = np.zeros(128, dtype=np.uint8)
    arr[61:64] = 63

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'unary.fits.fz')

        with rustfits.FITS(fname, 'w+') as f:
            f.write_image(arr, compress='RICE_1')
            np.testing.assert_array_equal(f[1].read(), arr)

        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[1].read(), arr)


def test_u8_spike_position_sweep():
    """
    Sweep the spike across every position so the unary run lands
    at every possible alignment against the decoder's 64-bit
    refill boundary (the failing state only occurs at specific
    alignments — position 61 pre-fix)."""
    n = 128
    for start in range(n - 3):
        arr = np.zeros(n, dtype=np.uint8)
        arr[start : start + 3] = 63
        np.testing.assert_array_equal(_roundtrip_mem(arr), arr)


@pytest.mark.parametrize('value', [7, 31, 53, 63, 101, 127, 255])
def test_u8_spike_value_sweep(value):
    """
    Different jump sizes give different unary-run lengths; sweep
    positions for each at the alignment-critical region."""
    n = 128
    for start in range(48, 80):
        arr = np.zeros(n, dtype=np.uint8)
        arr[start : start + 3] = value
        np.testing.assert_array_equal(_roundtrip_mem(arr), arr)


def test_widemask_like_multi_tile_1d():
    """
    Healsparse-widemask-shaped data: a 1-D u8 image, mostly zeros
    with scattered constant-value runs, multiple tiles (the shape
    of the file this bug was reported against)."""
    rng = np.random.default_rng(42)
    n = 8192
    arr = np.zeros(n, dtype=np.uint8)
    for _ in range(60):
        start = int(rng.integers(0, n - 16))
        length = int(rng.integers(1, 16))
        value = int(rng.integers(1, 64))
        arr[start : start + length] = value

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'widemask.fits.fz')

        cfg = rustfits.Rice1(tile_shape=(2048,))
        with rustfits.FITS(fname, 'w+') as f:
            f.write_image(arr, compress=cfg)
            np.testing.assert_array_equal(f[1].read(), arr)

        with rustfits.FITS(fname) as f:
            np.testing.assert_array_equal(f[1].read(), arr)
            # slicing exercises the per-tile decode path
            np.testing.assert_array_equal(f[1][100:3000], arr[100:3000])
            np.testing.assert_array_equal(f[1][::7], arr[::7])
