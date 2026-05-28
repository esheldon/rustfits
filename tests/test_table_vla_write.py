"""
Phase 4 tests: VLA columns on write + append.

create_table_hdu accepts numpy Object fields plus a `var_dtypes`
sidecar kwarg mapping each Object column to its inner element type
(numpy dtype string).  TFORMs are emitted as 1PE / 1QE / etc.
PCOUNT is updated by write() and append() as the heap grows.

Coverage:
- create + write across the supported inner letters (B/I/J/K/E/D/L/C/M)
- P (default) and Q (opt-in) descriptors
- Mixed fixed + VLA columns; multiple VLA columns in one table
- Empty cells (length-0 ndarray) round-trip
- All three input forms: structured ndarray, dict, list+names
- append() to a VLA table — last HDU + non-last HDU
- append moves the existing heap forward to make room for new main rows
- Validate-before-mutate: bad dtype rejected without touching the file
- Unknown var_dtypes key rejected
- Object field declared without var_dtypes entry rejected

Mutations verified via both same-handle and post-reopen reads
(CLAUDE.md convention).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------- create_table_hdu header structure ----------


def test_create_emits_pE_tform_and_zero_pcount():
    """A single 'O' column with var_dtypes={'lc': 'f4'} produces
    TFORM='1PE' and PCOUNT=0 (no data yet)."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=0, var_dtypes={"lc": "f4"})
            assert f[1].header["TFORM1"] == "1PE"
            assert f[1].header["PCOUNT"] == 0


def test_heap_format_q_opt_in():
    """Passing heap_format='Q' produces 1QE (16-byte descriptors)."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(
                dt, nrows=0, var_dtypes={"lc": "f4"}, heap_format="Q"
            )
            assert f[1].header["TFORM1"] == "1QE"
            assert f[1].header["NAXIS1"] == 16


def test_object_without_var_dtypes_rejected():
    """An 'O' field without a corresponding var_dtypes entry is
    rejected — we can't pick a TFORM letter."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="var_dtypes"):
                f.create_table_hdu(dt, nrows=0)


def test_unknown_var_dtypes_key_rejected():
    """var_dtypes key with no matching column raises."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4")])
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="does not match"):
                f.create_table_hdu(dt, nrows=0, var_dtypes={"missing": "f4"})


def test_bad_heap_format_rejected():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        with rustfits.FITS(fn, "w+") as f:
            with pytest.raises(ValueError, match="must be 'P' or 'Q'"):
                f.create_table_hdu(
                    dt, nrows=0, var_dtypes={"lc": "f4"}, heap_format="R"
                )


# ---------- round-trip across inner types ----------


INNER_TYPES = [
    ("f4", "1PE"),
    ("f8", "1PD"),
    ("i2", "1PI"),
    ("i4", "1PJ"),
    ("i8", "1PK"),
    ("u1", "1PB"),
    ("c8", "1PC"),
    ("c16", "1PM"),
    ("bool", "1PL"),
]


@pytest.mark.parametrize("inner,expected_tform", INNER_TYPES)
def test_round_trip_single_vla_column(inner, expected_tform):
    """Round-trip a VLA column for each supported inner dtype."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("v", "O")])
        nrows = 4
        arr = np.zeros(nrows, dtype=dt)
        if inner == "bool":
            arr["v"][0] = np.array([True, False, True], dtype="bool")
            arr["v"][1] = np.array([], dtype="bool")
            arr["v"][2] = np.array([False], dtype="bool")
            arr["v"][3] = np.array(
                [True, True, False, False, True], dtype="bool"
            )
        elif inner.startswith("c"):
            arr["v"][0] = np.array([1 + 2j, 3 + 4j], dtype=inner)
            arr["v"][1] = np.array([], dtype=inner)
            arr["v"][2] = np.array([5 - 6j], dtype=inner)
            arr["v"][3] = np.array([0 + 0j, 1e10 + 1j], dtype=inner)
        else:
            arr["v"][0] = np.arange(3, dtype=inner)
            arr["v"][1] = np.array([], dtype=inner)
            arr["v"][2] = np.arange(5, dtype=inner)
            arr["v"][3] = np.array([42], dtype=inner)

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": inner})
            assert f[1].header["TFORM1"] == expected_tform
            f[1].write(arr)
            got = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(got["v"][i], arr["v"][i])

        with rustfits.FITS(fn) as f:
            got = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(got["v"][i], arr["v"][i])


# ---------- mixed fixed + VLA ----------


def test_mixed_fixed_and_vla_columns():
    """A table with fixed int + fixed string + VLA float."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("name", "U6"), ("lc", "O")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        arr["id"] = [1, 2, 3]
        arr["name"] = ["alpha", "beta", "gamma"]
        arr["lc"][0] = np.array([0.5, 1.0], dtype="f4")
        arr["lc"][1] = np.array([], dtype="f4")
        arr["lc"][2] = np.array([2.0, 3.0, 4.0, 5.0], dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows, var_dtypes={"lc": "f4"})
            f[1].write(arr)
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["name"], arr["name"])
            for i in range(nrows):
                np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])

        with rustfits.FITS(fn) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            np.testing.assert_array_equal(got["name"], arr["name"])
            for i in range(nrows):
                np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])


def test_multiple_vla_columns():
    """Two VLA columns of different inner dtypes in one table."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("a", "O"), ("b", "O")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        arr["a"][0] = np.array([1, 2, 3], dtype="i4")
        arr["a"][1] = np.array([4, 5], dtype="i4")
        arr["a"][2] = np.array([], dtype="i4")
        arr["b"][0] = np.array([1.5], dtype="f8")
        arr["b"][1] = np.array([2.5, 3.5, 4.5], dtype="f8")
        arr["b"][2] = np.array([5.5, 6.5], dtype="f8")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(
                dt, nrows=nrows, var_dtypes={"a": "i4", "b": "f8"}
            )
            f[1].write(arr)
            got = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(got["a"][i], arr["a"][i])
                np.testing.assert_array_equal(got["b"][i], arr["b"][i])

        with rustfits.FITS(fn) as f:
            got = f[1].read()
            for i in range(nrows):
                np.testing.assert_array_equal(got["a"][i], arr["a"][i])
                np.testing.assert_array_equal(got["b"][i], arr["b"][i])


# ---------- input forms ----------


def test_write_via_dict():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O")])
        ids = np.array([1, 2], dtype="i4")
        lcs = np.empty(2, dtype="O")
        lcs[0] = np.array([1.0, 2.0], dtype="f4")
        lcs[1] = np.array([3.0], dtype="f4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            f[1].write({"id": ids, "lc": lcs})
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], ids)
            for i in range(2):
                np.testing.assert_array_equal(got["lc"][i], lcs[i])


def test_write_via_list_names():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O")])
        ids = np.array([1, 2], dtype="i4")
        lcs = np.empty(2, dtype="O")
        lcs[0] = np.array([1.0, 2.0], dtype="f4")
        lcs[1] = np.array([3.0], dtype="f4")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            f[1].write([ids, lcs], names=["id", "lc"])
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], ids)
            for i in range(2):
                np.testing.assert_array_equal(got["lc"][i], lcs[i])


# ---------- append (last HDU) ----------


def test_append_vla_last_hdu():
    """Append to the only (last) HDU: set_len grow path."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["id"] = [1, 2]
        arr["lc"][0] = np.array([1.0, 2.0, 3.0], dtype="f4")
        arr["lc"][1] = np.array([4.0], dtype="f4")
        extra = np.zeros(2, dtype=dt)
        extra["id"] = [3, 4]
        extra["lc"][0] = np.array([5.0, 6.0], dtype="f4")
        extra["lc"][1] = np.array([], dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            f[1].write(arr)
        with rustfits.FITS(fn, "r+") as f:
            f[1].append(extra)
            assert f[1].nrows == 4
            assert f[1].header["PCOUNT"] == (3 + 1 + 2 + 0) * 4
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [1, 2, 3, 4])
            np.testing.assert_array_equal(got["lc"][0], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(got["lc"][1], [4.0])
            np.testing.assert_array_equal(got["lc"][2], [5.0, 6.0])
            np.testing.assert_array_equal(got["lc"][3], [])

        with rustfits.FITS(fn) as f:
            assert f[1].nrows == 4
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], [1, 2, 3, 4])
            np.testing.assert_array_equal(got["lc"][0], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(got["lc"][3], [])


# ---------- append (non-last HDU) ----------


def test_append_vla_non_last_hdu_shifts_tail():
    """Append to a VLA HDU that is followed by another HDU on disk.
    The follow-on HDU must be untouched in content but shifted in
    file position."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt_vla = np.dtype([("id", "i4"), ("lc", "O")])
        dt_fixed = np.dtype([("x", "i4"), ("y", "f8")])
        first = np.zeros(2, dtype=dt_vla)
        first["id"] = [1, 2]
        first["lc"][0] = np.array([1.0], dtype="f4")
        first["lc"][1] = np.array([2.0, 3.0], dtype="f4")
        second = np.zeros(3, dtype=dt_fixed)
        second["x"] = [10, 20, 30]
        second["y"] = [0.1, 0.2, 0.3]
        extra = np.zeros(2, dtype=dt_vla)
        extra["id"] = [3, 4]
        extra["lc"][0] = np.array([4.0, 5.0, 6.0], dtype="f4")
        extra["lc"][1] = np.array([7.0], dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(
                dt_vla, nrows=2, var_dtypes={"lc": "f4"}, extname="VLA"
            )
            f[1].write(first)
            f.create_table_hdu(dt_fixed, nrows=3, extname="FIXED")
            f[2].write(second)

        with rustfits.FITS(fn, "r+") as f:
            f[1].append(extra)
            assert f[1].nrows == 4
            got1 = f[1].read()
            np.testing.assert_array_equal(got1["id"], [1, 2, 3, 4])
            np.testing.assert_array_equal(got1["lc"][0], [1.0])
            np.testing.assert_array_equal(got1["lc"][1], [2.0, 3.0])
            np.testing.assert_array_equal(got1["lc"][2], [4.0, 5.0, 6.0])
            np.testing.assert_array_equal(got1["lc"][3], [7.0])
            assert f[2].nrows == 3
            got2 = f[2].read()
            np.testing.assert_array_equal(got2["x"], [10, 20, 30])
            np.testing.assert_array_equal(got2["y"], [0.1, 0.2, 0.3])

        with rustfits.FITS(fn) as f:
            assert f[1].nrows == 4
            got1 = f[1].read()
            for i in range(4):
                np.testing.assert_array_equal(
                    got1["lc"][i],
                    [
                        np.array([1.0], dtype="f4"),
                        np.array([2.0, 3.0], dtype="f4"),
                        np.array([4.0, 5.0, 6.0], dtype="f4"),
                        np.array([7.0], dtype="f4"),
                    ][i],
                )
            got2 = f[2].read()
            np.testing.assert_array_equal(got2["x"], [10, 20, 30])
            np.testing.assert_array_equal(got2["y"], [0.1, 0.2, 0.3])


def test_append_to_empty_vla_table():
    """Initial create with nrows=0, then append.  Exercises the
    current_pcount=0 / current_nrows=0 edge case."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["lc"][0] = np.arange(3, dtype="f4")
        arr["lc"][1] = np.arange(5, dtype="f4")
        arr["lc"][2] = np.array([], dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=0, var_dtypes={"lc": "f4"})
            f[1].append(arr)
            assert f[1].nrows == 3
            assert f[1].header["PCOUNT"] == (3 + 5) * 4

        with rustfits.FITS(fn) as f:
            got = f[1].read()
            for i in range(3):
                np.testing.assert_array_equal(got["lc"][i], arr["lc"][i])


def test_multiple_sequential_vla_appends():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("lc", "O")])

        def make_chunk(start_id, lengths):
            n = len(lengths)
            a = np.zeros(n, dtype=dt)
            a["id"] = np.arange(n) + start_id
            for i, ln in enumerate(lengths):
                a["lc"][i] = np.arange(ln, dtype="f4") + start_id * 100.0
            return a

        chunks = [
            make_chunk(0, [2, 3]),
            make_chunk(10, [1, 0, 4]),
            make_chunk(100, [5, 2]),
        ]
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=0, var_dtypes={"lc": "f4"})
            for c in chunks:
                f[1].append(c)
            total_rows = sum(len(c) for c in chunks)
            assert f[1].nrows == total_rows

        with rustfits.FITS(fn) as f:
            got = f[1].read()
            row = 0
            for c in chunks:
                for i in range(len(c)):
                    np.testing.assert_array_equal(
                        got["id"][row + i], c["id"][i]
                    )
                    np.testing.assert_array_equal(
                        got["lc"][row + i], c["lc"][i]
                    )
                row += len(c)


# ---------- validation rejections ----------


def test_vla_cell_wrong_dtype_rejected():
    """A VLA column declared as 'f4' but given an i4 cell ndarray
    is rejected before any byte is written."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["lc"][0] = np.arange(3, dtype="i4")  # wrong dtype
        arr["lc"][1] = np.arange(2, dtype="i4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            size_before = os.path.getsize(fn)
            with pytest.raises(ValueError, match="VLA cell dtype"):
                f[1].write(arr)
            # File untouched.
            assert os.path.getsize(fn) == size_before
            assert f[1].header["PCOUNT"] == 0


def test_vla_cell_not_ndarray_rejected():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["lc"][0] = [1.0, 2.0, 3.0]  # Python list, not ndarray
        arr["lc"][1] = np.arange(2, dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            with pytest.raises(ValueError, match="ndarray"):
                f[1].write(arr)


def test_vla_cell_2d_rejected():
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.fits")
        dt = np.dtype([("lc", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["lc"][0] = np.zeros((2, 3), dtype="f4")  # 2-D
        arr["lc"][1] = np.arange(2, dtype="f4")

        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=2, var_dtypes={"lc": "f4"})
            with pytest.raises(ValueError, match="1-D"):
                f[1].write(arr)
