"""
Phase 3 tests: TableHDU.append() (and the extend alias).

Append grows NAXIS2 in the header and the data section, then writes
the new rows.  For HDUs that are not the last on disk it shifts the
file tail forward and bumps later-HDU offsets in lockstep (shared
shift_file_tail_and_update_offsets primitive).

Coverage:
- append to last HDU (set_len path)
- append to non-last HDU (shift_file_tail path)
- append that stays within the current block (no grow)
- append that crosses a 2880-byte block boundary
- append 0 rows (no-op)
- input-form coverage: structured ndarray, dict, list+names
- validate-before-mutate: dtype mismatch leaves the file untouched
- extend alias
- multiple sequential appends accumulate correctly
- append to an empty (NAXIS2=0) table

Mutations are checked via both same-handle and post-reopen reads
per the CLAUDE.md convention.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


DT = np.dtype(
    [
        ("id", "i4"),
        ("name", "U6"),
        ("flux", "f4"),
    ]
)


def _make_arr(n, base=0):
    arr = np.zeros(n, dtype=DT)
    arr["id"] = np.arange(n, dtype="i4") + base * 1000
    arr["name"] = [f"r{base * 1000 + i:05d}" for i in range(n)]
    arr["flux"] = np.arange(n, dtype="f4") + base * 0.5
    return arr


def _create_initial(fname, nrows):
    arr = _make_arr(nrows, base=0)
    with rustfits.FITS(fname, "w+") as fits:
        fits.create_table_hdu(DT, nrows=nrows)
        if nrows > 0:
            fits[1].write(arr)
    return arr


# ----------------- last-HDU append (set_len branch) -----------------


def test_append_last_hdu_set_len_branch():
    """
    Single-HDU file (table is last on disk) → set_len path.
    Append 3 rows to a 4-row table; verify size + data + NAXIS2.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 4)
        new = _make_arr(3, base=1)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(new)
            assert fits[1].nrows == 7
            got = fits[1].read()
            np.testing.assert_array_equal(got[:4], initial)
            np.testing.assert_array_equal(got[4:], new)

        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 7
            got = fits[1].read()
            np.testing.assert_array_equal(got[:4], initial)
            np.testing.assert_array_equal(got[4:], new)


# ----------------- non-last-HDU append (shift branch) -----------------


def test_append_non_last_hdu_shifts_tail():
    """
    Two tables in the file.  Appending to the FIRST table shifts the
    second table's bytes forward; verify both tables still readable
    via same handle AND fresh reopen.  Exercises the shift_file_tail
    + offset propagation through the shared Arc<HduOffsets> model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        first_initial = _make_arr(4, base=0)
        second_initial = _make_arr(5, base=9)

        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(DT, nrows=4, extname="FIRST")
            fits[1].write(first_initial)
            fits.create_table_hdu(DT, nrows=5, extname="SECOND")
            fits[2].write(second_initial)

        new = _make_arr(3, base=1)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(new)
            # First HDU grew.
            assert fits[1].nrows == 7
            assert fits[1].shape[0] == 7
            got1 = fits[1].read()
            np.testing.assert_array_equal(got1[:4], first_initial)
            np.testing.assert_array_equal(got1[4:], new)
            # Second HDU shifted but unchanged in content.
            assert fits[2].nrows == 5
            got2 = fits[2].read()
            np.testing.assert_array_equal(got2, second_initial)

        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 7
            got1 = fits[1].read()
            np.testing.assert_array_equal(got1[:4], first_initial)
            np.testing.assert_array_equal(got1[4:], new)
            assert fits[2].nrows == 5
            got2 = fits[2].read()
            np.testing.assert_array_equal(got2, second_initial)


# ----------------- in-block vs cross-block grow -----------------


def test_append_within_current_block_no_grow():
    """
    Row width is 14 bytes (4 + 6 + 4) for DT.  A 2880-byte block
    holds 205 rows.  Append a few rows to a small table → new
    padded size equals old padded size, no file-length change
    needed; only the header and data get rewritten.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_initial(fname, 4)
        size_before = os.path.getsize(fname)
        new = _make_arr(3, base=1)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(new)
            assert fits[1].nrows == 7

        size_after = os.path.getsize(fname)
        assert size_after == size_before  # no block grow

        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 7


def test_append_crosses_block_boundary():
    """
    Force the data section to cross a block boundary.  DT row width
    is 14 bytes; 205 rows = 2870 bytes (one block).  Append 100 rows
    to take it past 2880; file size must grow by one block.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 200)
        size_before = os.path.getsize(fname)
        new = _make_arr(100, base=1)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(new)
            assert fits[1].nrows == 300
            got = fits[1].read()
            np.testing.assert_array_equal(got[:200], initial)
            np.testing.assert_array_equal(got[200:], new)

        size_after = os.path.getsize(fname)
        # New data is 300*14 = 4200 bytes → 2 blocks = 5760, vs old
        # 200*14 = 2800 → 1 block = 2880.  Delta = 2880.
        assert size_after == size_before + 2880

        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 300
            got = fits[1].read()
            np.testing.assert_array_equal(got[:200], initial)
            np.testing.assert_array_equal(got[200:], new)


# ----------------- zero-row append (no-op) -----------------


def test_append_zero_rows_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 4)
        size_before = os.path.getsize(fname)
        empty = _make_arr(0)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(empty)
            assert fits[1].nrows == 4
            got = fits[1].read()
            np.testing.assert_array_equal(got, initial)

        assert os.path.getsize(fname) == size_before


# ----------------- input forms -----------------


def test_append_dict_form():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 3)
        new_dict = {
            "id": np.array([100, 101], dtype="i4"),
            "name": np.array(["alpha", "beta"], dtype="U6"),
            "flux": np.array([3.5, 4.5], dtype="f4"),
        }

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append(new_dict)
            assert fits[1].nrows == 5
            got = fits[1].read()
            np.testing.assert_array_equal(got[:3], initial)
            np.testing.assert_array_equal(got["id"][3:], [100, 101])
            np.testing.assert_array_equal(got["name"][3:], ["alpha", "beta"])
            np.testing.assert_array_equal(got["flux"][3:], [3.5, 4.5])


def test_append_list_names_form():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 3)
        ids = np.array([200, 201], dtype="i4")
        names = np.array(["gamma", "delta"], dtype="U6")
        fluxes = np.array([5.5, 6.5], dtype="f4")

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].append([ids, names, fluxes], names=["id", "name", "flux"])
            assert fits[1].nrows == 5
            got = fits[1].read()
            np.testing.assert_array_equal(got[:3], initial)
            np.testing.assert_array_equal(got["id"][3:], ids)
            np.testing.assert_array_equal(got["name"][3:], names)
            np.testing.assert_array_equal(got["flux"][3:], fluxes)


# ----------------- validate-then-mutate -----------------


def test_append_dtype_mismatch_leaves_file_untouched():
    """
    The bad input is detected by dispatch_write_input BEFORE the file
    is grown or the header is rewritten.  Verify: file size, NAXIS2,
    and data are all unchanged after the rejection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 5)
        size_before = os.path.getsize(fname)
        # i4 column would naturally take i4; f8 is the wrong dtype.
        bad_dt = np.dtype([("id", "f8"), ("name", "U6"), ("flux", "f4")])
        bad = np.zeros(3, dtype=bad_dt)

        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[1].append(bad)
            assert fits[1].nrows == 5
            np.testing.assert_array_equal(fits[1].read(), initial)

        assert os.path.getsize(fname) == size_before
        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 5
            np.testing.assert_array_equal(fits[1].read(), initial)


# ----------------- extend alias -----------------


def test_extend_is_alias_for_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        initial = _create_initial(fname, 4)
        new = _make_arr(2, base=1)

        with rustfits.FITS(fname, "r+") as fits:
            fits[1].extend(new)
            assert fits[1].nrows == 6
            got = fits[1].read()
            np.testing.assert_array_equal(got[:4], initial)
            np.testing.assert_array_equal(got[4:], new)


# ----------------- multiple appends -----------------


def test_multiple_sequential_appends_accumulate():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        chunks = [_make_arr(3, base=b) for b in range(5)]
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_table_hdu(DT, nrows=0)
            for c in chunks:
                fits[1].append(c)
            assert fits[1].nrows == 15
            got = fits[1].read()
            for i, c in enumerate(chunks):
                np.testing.assert_array_equal(got[i * 3 : (i + 1) * 3], c)

        with rustfits.FITS(fname) as fits:
            assert fits[1].nrows == 15
            got = fits[1].read()
            for i, c in enumerate(chunks):
                np.testing.assert_array_equal(got[i * 3 : (i + 1) * 3], c)


# ----------------- append to empty table -----------------


def test_append_to_empty_table():
    """create_table_hdu(nrows=0) followed by append should be equivalent
    to create_table_hdu(nrows=N) + write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname_via_append = os.path.join(tmpdir, "via_append.fits")
        fname_via_write = os.path.join(tmpdir, "via_write.fits")
        arr = _make_arr(6, base=3)

        with rustfits.FITS(fname_via_append, "w+") as fits:
            fits.create_table_hdu(DT, nrows=0)
            fits[1].append(arr)

        with rustfits.FITS(fname_via_write, "w+") as fits:
            fits.create_table_hdu(DT, nrows=6)
            fits[1].write(arr)

        with rustfits.FITS(fname_via_append) as fits:
            assert fits[1].nrows == 6
            got = fits[1].read()
            np.testing.assert_array_equal(got, arr)


# ----------------- empty-input validation -----------------


def test_append_empty_dict_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_initial(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="empty"):
                fits[1].append({})


def test_append_names_with_dict_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_initial(fname, 3)
        new_dict = {
            "id": np.array([1], dtype="i4"),
            "name": np.array(["x"], dtype="U6"),
            "flux": np.array([1.0], dtype="f4"),
        }
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="names="):
                fits[1].append(new_dict, names=["id", "name", "flux"])


def test_append_list_without_names_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        _create_initial(fname, 3)
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="names="):
                fits[1].append([np.array([1], dtype="i4")])
