"""
ASCII-table append — AsciiTableHDU.append + extend alias.

Covers NAXIS2 grow, data-section grow (incl. file shift when the
HDU is not last), validate-then-mutate, and cross-tool readback.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def test_append_basic():
    """Create with 0 rows, append in batches, verify accumulated state."""
    dtype = np.dtype([("X", "i4"), ("Y", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            f[1].append(
                np.array(
                    [(1, 1.5), (2, 2.5)],
                    dtype=dtype,
                )
            )
            assert f[1].nrows == 2
            f[1].append(np.array([(3, 3.5)], dtype=dtype))
            assert f[1].nrows == 3
        # Reopen and confirm bytes survive.
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], [1, 2, 3])
            np.testing.assert_allclose(
                arr["Y"],
                [1.5, 2.5, 3.5],
                rtol=1e-5,
            )


def test_append_crosses_block_boundary():
    """Append enough rows to span multiple 2880-byte blocks."""
    dtype = np.dtype([("X", "i4")])
    # I20 = 20 bytes/row; 2880 / 20 = 144 rows/block.  Append 200
    # rows in 50-row chunks (4 chunks); should span 2 blocks.
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            for i in range(0, 200, 50):
                batch = np.array(
                    [(j,) for j in range(i, i + 50)],
                    dtype=dtype,
                )
                f[1].append(batch)
            assert f[1].nrows == 200
        with rustfits.FITS(fname) as f:
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], np.arange(200))


def test_append_non_last_hdu_shifts_tail():
    """Append to HDU [1] when HDU [2] follows; verify HDU [2] survives."""
    dtype1 = np.dtype([("X", "i4")])
    dtype2 = np.dtype([("Y", "i8")])
    sentinel = np.array([(99,), (98,), (97,)], dtype=dtype2)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            # HDU 1: small ASCII table to grow into.
            f.create_ascii_table_hdu(dtype1, nrows=0)
            # HDU 2: sentinel BINTABLE to detect tail corruption.
            f.create_table_hdu(dtype2, nrows=3)
            f[2].write(sentinel)
        with rustfits.FITS(fname, "r+") as f:
            # Append enough rows that the data section grows past
            # the original block.
            f[1].append(
                np.array(
                    [(i,) for i in range(200)],
                    dtype=dtype1,
                )
            )
            assert f[1].nrows == 200
        with rustfits.FITS(fname) as f:
            # HDU 1 still readable.
            assert f[1].nrows == 200
            arr = f[1].read()
            np.testing.assert_array_equal(arr["X"], np.arange(200))
            # HDU 2 sentinel bytes survived the shift.
            t = f[2].read()
            np.testing.assert_array_equal(t["Y"], [99, 98, 97])


def test_append_dict_input():
    dtype = np.dtype([("X", "i4"), ("Y", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            f[1].append({"X": np.array([1, 2]), "Y": np.array([0.5, 1.5])})
            assert f[1].nrows == 2


def test_append_zero_rows_is_noop():
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            f[1].append(np.array([], dtype=dtype))
            assert f[1].nrows == 0


def test_append_validate_then_mutate():
    """Invalid input doesn't change NAXIS2."""
    dtype = np.dtype([("X", "i4")])
    bad = np.array([(99,)], dtype=np.dtype([("WRONG", "i4")]))
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=1)
            f[1].write(np.array([(7,)], dtype=dtype))
            with pytest.raises(ValueError):
                f[1].append(bad)
            # NAXIS2 unchanged.
            assert f[1].nrows == 1


def test_extend_alias():
    """extend is the same as append (parity with TableHDU)."""
    dtype = np.dtype([("X", "i4")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            f[1].extend(np.array([(1,), (2,)], dtype=dtype))
            assert f[1].nrows == 2


def test_round_trip_after_multiple_appends():
    """Many small appends produce a final file astropy can read."""
    astropy_fits = pytest.importorskip("astropy.io.fits")
    dtype = np.dtype([("ID", "i8"), ("FLUX", "f8")])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_ascii_table_hdu(dtype, nrows=0)
            for i in range(10):
                batch = np.array(
                    [(j, j * 0.5) for j in range(i * 3, i * 3 + 3)],
                    dtype=dtype,
                )
                f[1].append(batch)
            assert f[1].nrows == 30
        with astropy_fits.open(fname) as hdul:
            t = hdul[1].data
            np.testing.assert_array_equal(t["ID"], np.arange(30))
            np.testing.assert_allclose(
                t["FLUX"],
                0.5 * np.arange(30),
                rtol=1e-12,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
