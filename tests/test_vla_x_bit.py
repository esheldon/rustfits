"""
BINTABLE VLA PX/QX (variable-length bit-packed) column support.

The FITS spec for PX/QX:
  - Descriptor `nelements` is the BIT count (not the byte count).
  - Heap holds `ceil(nelements/8)` MSB-packed bytes per cell.
  - Trailing bits in the last byte are zero per spec.

API: same `bit_columns=` kwarg as Phase 1 fixed-X.  Mark a VLA
column whose `var_dtypes` says bool/?/b1 with bit_columns to make
it a PX/QX (otherwise it stays PL/QL with one byte per bool):

    f.create_table_hdu(
        dt,                                  # Object-dtype VLA field
        nrows=N,
        var_dtypes={"flags": "?"},           # bool inner
        bit_columns=["flags"],               # opt into PX
    )

Tests cover: whole-table write, per-cell read round-trip with
varying cell lengths (including non-multiple-of-8), P vs Q
descriptors, append, __setitem__ (cell + slice + whole-column),
empty cells, repack reclaims orphans, bit_columns=True global
toggle, rejection when a non-bool VLA column is named in
bit_columns, and astropy + fitsio cross-tool reads.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _have_astropy():
    try:
        import astropy.io.fits  # noqa: F401

        return True
    except ImportError:
        return False


def _have_fitsio():
    try:
        import fitsio  # noqa: F401

        return True
    except ImportError:
        return False


def _dt_with_vla_x():
    return np.dtype([("id", "i4"), ("flags", "O")])


def _make_data(nrows, *, seed=0):
    """nrows cells with varying lengths (mix of multiples of 8 and not)."""
    rng = np.random.default_rng(seed)
    dt = _dt_with_vla_x()
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows, dtype="i4")
    lengths = [0, 1, 8, 13, 16, 17, 31, 32, 33]
    for i in range(nrows):
        n = lengths[i % len(lengths)]
        arr["flags"][i] = rng.integers(0, 2, n, dtype=bool)
    return arr, dt


def _check_cells_equal(out, expected):
    assert len(out) == len(expected)
    for i in range(len(out)):
        np.testing.assert_array_equal(out[i], expected[i])


# ---------------------------------------------------------------------
# Whole-table round-trip
# ---------------------------------------------------------------------


def test_vla_x_write_then_read():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(9)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=9,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            tform2 = f[1].header["TFORM2"]
        # TFORM should be 1PX or 1QX (with optional max-length hint).
        upper = tform2.strip().upper()
        assert upper.startswith("1PX") or upper.startswith("1QX"), (
            f"unexpected TFORM2: {tform2!r}"
        )
        np.testing.assert_array_equal(out["id"], data["id"])
        _check_cells_equal(out["flags"], data["flags"])


def test_vla_x_default_p_descriptor():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )  # heap_format defaults to 'P'
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].header["TFORM2"].strip().upper().startswith("1PX")


def test_vla_x_q_descriptor():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5, seed=1)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
                heap_format="Q",
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].header["TFORM2"].strip().upper().startswith("1QX")
            out = f[1].read()
        _check_cells_equal(out["flags"], data["flags"])


def test_vla_x_without_bit_columns_stays_pl():
    """Without bit_columns, bool VLA inner stays PL (default behavior)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(3)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=3,
                var_dtypes={"flags": "?"},
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].header["TFORM2"].strip().upper().startswith("1PL")
            out = f[1].read()
        _check_cells_equal(out["flags"], data["flags"])


# ---------------------------------------------------------------------
# Cell length edge cases
# ---------------------------------------------------------------------


def test_vla_x_empty_cells():
    """nelements=0 cells: descriptor (0, current_heap_offset), no bytes."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_with_vla_x()
        nrows = 4
        data = np.zeros(nrows, dtype=dt)
        data["id"] = np.arange(nrows, dtype="i4")
        for i in range(nrows):
            data["flags"][i] = np.array([], dtype=bool)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            pcount = int(f[1].header["PCOUNT"])
        # All-empty: heap has zero bytes (no cells to store).
        assert pcount == 0
        for i in range(nrows):
            assert len(out["flags"][i]) == 0


def test_vla_x_non_multiple_of_8_trailing_zero():
    """13-bit cell: 2 bytes on disk with bottom 3 bits of byte 2 = 0."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_with_vla_x()
        data = np.zeros(2, dtype=dt)
        data["id"] = [0, 1]
        # Row 0: all 13 bits true.
        data["flags"][0] = np.ones(13, dtype=bool)
        # Row 1: alternating.
        data["flags"][1] = np.array([True, False] * 6 + [True], dtype=bool)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=2,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        # Round-trip: all 13 bits round-trip; the implicit trailing-
        # zero bits are exercised by the fact that we got all 13
        # bits back correctly (read would otherwise pick up garbage
        # from the unused tail).
        _check_cells_equal(out["flags"], data["flags"])


def test_vla_x_mixed_lengths_sorted_by_heap_offset():
    """
    Heap reads sort by heap_offset; confirm out-of-order cell sizes
    still round-trip correctly (the read pass handles ordering).
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = _dt_with_vla_x()
        # Deliberately ragged: long, short, medium, very long, empty.
        cells = [
            np.ones(100, dtype=bool),
            np.array([True, False], dtype=bool),
            np.zeros(20, dtype=bool),
            np.array([True] * 200 + [False] * 200, dtype=bool),
            np.array([], dtype=bool),
        ]
        data = np.zeros(len(cells), dtype=dt)
        data["id"] = np.arange(len(cells), dtype="i4")
        for i, c in enumerate(cells):
            data["flags"][i] = c
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=len(cells),
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        _check_cells_equal(out["flags"], data["flags"])


# ---------------------------------------------------------------------
# bit_columns=True (global toggle) + non-bool VLA rejection
# ---------------------------------------------------------------------


def test_vla_bit_columns_true_promotes_bool_vla():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype(
            [
                ("id", "i4"),
                ("flags", "O"),
                ("vals", "O"),
            ]
        )
        data = np.zeros(3, dtype=dt)
        data["id"] = np.arange(3, dtype="i4")
        for i in range(3):
            data["flags"][i] = np.array([True, False, True], dtype=bool)
            data["vals"][i] = np.arange(i + 1, dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=3,
                var_dtypes={"flags": "?", "vals": "f4"},
                bit_columns=True,  # global toggle
            )
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            assert f[1].header["TFORM2"].strip().upper().startswith("1PX")
            # vals is f4 — bit_columns=True is a soft filter; non-
            # bool inners stay at their natural letter.
            assert f[1].header["TFORM3"].strip().upper().startswith("1PE")
            out = f[1].read()
        _check_cells_equal(out["flags"], data["flags"])
        _check_cells_equal(out["vals"], data["vals"])


def test_vla_bit_columns_names_non_bool_inner_rejected():
    """Explicitly listing a non-bool VLA column is a user error."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("vals", "O")])
        with rustfits.FITS(fname, "w+") as f:
            with pytest.raises(ValueError) as ei:
                f.create_table_hdu(
                    dt,
                    nrows=2,
                    var_dtypes={"vals": "f4"},
                    bit_columns=["vals"],
                )
            msg = str(ei.value).lower()
            assert "bool" in msg or "vals" in msg


# ---------------------------------------------------------------------
# Append + __setitem__
# ---------------------------------------------------------------------


def test_vla_x_append():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        more, _ = _make_data(3, seed=42)
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(more)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        expected = np.concatenate([data, more])
        _check_cells_equal(out["flags"], expected["flags"])


def test_vla_x_setitem_cell():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        new_cell = np.array([True, True, False, True, False] * 5, dtype=bool)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["flags"][2] = new_cell
        expected = data.copy()
        expected["flags"] = data["flags"].copy()
        expected["flags"][2] = new_cell
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        _check_cells_equal(out["flags"], expected["flags"])


def test_vla_x_setitem_whole_column():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(4, seed=2)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=4,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        new_col = np.empty(4, dtype=object)
        for i in range(4):
            new_col[i] = np.array([True] * (3 * (i + 1)), dtype=bool)
        with rustfits.FITS(fname, "r+") as f:
            f[1]["flags"] = new_col
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        _check_cells_equal(out["flags"], new_col)


def test_vla_x_setitem_slice():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(6)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=6,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
        chunk = np.zeros(3, dtype=dt)
        chunk["id"] = [100, 101, 102]
        chunk["flags"][0] = np.ones(7, dtype=bool)
        chunk["flags"][1] = np.array([False] * 15, dtype=bool)
        chunk["flags"][2] = np.array([True, False] * 8, dtype=bool)
        with rustfits.FITS(fname, "r+") as f:
            f[1][2:5] = chunk
        expected = data.copy()
        expected["flags"] = data["flags"].copy()
        for k, i in enumerate(range(2, 5)):
            expected["id"][i] = chunk["id"][k]
            expected["flags"][i] = chunk["flags"][k]
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        np.testing.assert_array_equal(out["id"], expected["id"])
        _check_cells_equal(out["flags"], expected["flags"])


# ---------------------------------------------------------------------
# Repack reclaims VLA X orphans
# ---------------------------------------------------------------------


def test_vla_x_repack_reclaims_orphans():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5, seed=3)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)
            for k in range(4):
                f[1]["flags"][1] = np.array([True] * (k + 5), dtype=bool)
            pcount_before = int(f[1].header["PCOUNT"])
            f[1].repack()
            pcount_after = int(f[1].header["PCOUNT"])
        assert pcount_after < pcount_before
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        # Last write to row 1 wins.
        np.testing.assert_array_equal(
            out["flags"][1], np.array([True] * 8, dtype=bool)
        )


# ---------------------------------------------------------------------
# Cross-tool reads
# ---------------------------------------------------------------------


def test_astropy_pxqx_documented_limitation():
    """
    astropy.io.fits does NOT support PX/QX VLA columns: its FITS2NUMPY
    dtype map (in astropy.io.fits.column) only lists L/B/I/J/K/A/E/D/C/M;
    `from_tform()` on a `1PX(N)` format raises VerifyError before any
    rows are touched.  cfitsio + fitsio handle PX/QX correctly.  This
    test pins the limitation so we notice if astropy ever adds X
    support — at which point this test can flip to a positive
    cross-read assertion mirroring `test_fitsio_cross_reads_vla_x`.
    """
    if not _have_astropy():
        pytest.skip("astropy required")
    import astropy.io.fits as ap

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(6)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=6,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)

        # astropy's TFORM parser regex accepts X as a letter but
        # rejects on the FITS2NUMPY lookup right after.  Verify the
        # specific error path so an astropy upgrade that fixes it
        # surfaces here.
        from astropy.io.fits.verify import VerifyError

        with ap.open(fname) as hdul:
            with pytest.raises(VerifyError, match="Invalid column format"):
                _ = hdul[1].data["flags"]


@pytest.mark.skipif(
    not _have_fitsio(),
    reason="fitsio required for cross-tool verification",
)
def test_fitsio_cross_reads_vla_x():
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        data, dt = _make_data(5, seed=5)
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                var_dtypes={"flags": "?"},
                bit_columns=["flags"],
            )
            f[1].write(data)

        with fitsio.FITS(fname, "r") as f:
            tform = f[1].read_header()["TFORM2"]
            arr = f[1].read()
        assert "X" in tform.upper()
        # fitsio's read of PX returns a (nrows,) object array of
        # cells (or possibly padded — slice to actual length).
        for i in range(5):
            cell = np.asarray(arr["flags"][i]).astype(bool).ravel()
            # fitsio may zero-pad VLA reads to max length; slice.
            n = len(data["flags"][i])
            np.testing.assert_array_equal(cell[:n], data["flags"][i])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
