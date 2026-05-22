"""
ZIMAGE Phase 6: PLIO_1 read.

Round-trip tests use fitsio to write PLIO_1-compressed fixtures
(mask-style images: mostly zero, with runs of non-zero values)
and rustfits to read them back, checking bit-exact agreement
with fitsio's own decode.

PLIO is designed for mask / pixel-list data; ZBITPIX is typically
8 / 16 / 32 (negative ZBITPIX makes no sense for masks).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_plio(tmpdir, data, *, tile_dims=None):
    fname = os.path.join(tmpdir, "t.fits.fz")
    kw = {"compress": "PLIO"}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    kw["qlevel"] = None
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    return fname


def _mask_pattern(shape, *, dtype, n_runs=8):
    """
    Build a mask-style array: mostly zero with a handful of non-zero
    runs.  Deterministic per (shape, dtype) so tests are stable.
    """
    arr = np.zeros(shape, dtype=dtype)
    flat = arr.reshape(-1)
    n = flat.size
    rng = np.random.default_rng(42)
    for k in range(n_runs):
        start = rng.integers(0, n - 1)
        length = int(min(rng.integers(2, 25), n - start))
        value = int(rng.integers(1, 100))
        flat[start : start + length] = value
    return arr


# ---------------------- accessors ----------------------------------


def test_plio_compression_type_reported():
    """compression_type returns 'PLIO_1' for a PLIO-compressed HDU."""
    with tempfile.TemporaryDirectory() as tmp:
        data = _mask_pattern((32, 32), dtype=np.int32)
        fname = _write_plio(tmp, data, tile_dims=(16, 16))
        with rustfits.FITS(fname, "r") as f:
            assert f[1].compression.zcmptype == "PLIO_1"


# ---------------------- round trip ---------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32])
def test_plio_round_trip_matches_cfitsio(dtype):
    """Bit-exact agreement with fitsio's own decode across the
    supported integer dtypes."""
    with tempfile.TemporaryDirectory() as tmp:
        # Restrict values to the dtype's range for u8 / i16.
        maxv = {np.uint8: 100, np.int16: 100, np.int32: 100}[dtype]
        data = _mask_pattern((40, 60), dtype=dtype)
        # Clamp to range.
        data = np.clip(data, 0, maxv).astype(dtype)
        fname = _write_plio(tmp, data, tile_dims=(16, 24))
        with fitsio.FITS(fname, "r") as f:
            cfitsio_out = f[1].read()
        with rustfits.FITS(fname, "r") as f:
            our_out = f[1].read()
        assert our_out.dtype == cfitsio_out.dtype
        np.testing.assert_array_equal(our_out, cfitsio_out)


def test_plio_all_zeros_tile():
    """An all-zero image (which exercises the degenerate header-only
    encoded tile, len=7 shorts, no data words)."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.zeros((32, 48), dtype=np.int32)
        fname = _write_plio(tmp, data, tile_dims=(16, 24))
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_plio_single_solid_run():
    """A whole image of a single constant value: exercises the
    SH opcode (set high pv) followed by a long PN run."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((24, 36), 17, dtype=np.int32)
        fname = _write_plio(tmp, data, tile_dims=(12, 18))
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out, data)


def test_plio_slicing_matches_whole_read():
    """Slicing a PLIO HDU yields the same pixels as the matching
    slice of the whole-image read."""
    with tempfile.TemporaryDirectory() as tmp:
        data = _mask_pattern((40, 60), dtype=np.int32)
        fname = _write_plio(tmp, data, tile_dims=(20, 30))
        with rustfits.FITS(fname, "r") as f:
            whole = f[1].read()
            assert (f[1][10:30, 15:50] == whole[10:30, 15:50]).all()
            assert (f[1][0:1, :] == whole[0:1, :]).all()
            assert int(f[1][5, 7]) == int(whole[5, 7])
