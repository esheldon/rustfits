"""
Table row / chunk iteration: ``for row in hdu`` and
``hdu.iter(chunksize=..., columns=..., scale=...)``.

Covers both ``TableHDU`` and ``CompressedTableHDU`` (the iterator
refills through the HDU's own polymorphic ``read``, so the compressed
subclass works with no special-casing).  Modes:

  - row mode (chunksize=None): yields one np.void record per row
  - chunk mode (chunksize=N): yields structured ndarrays of <=N rows

Plus columns=/scale= forwarding, empty tables, the nrows snapshot
contract, VLA columns, and the multi-refill path (wide rows shrink the
internal byte-budget buffer below the row count).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# -------------------- helpers --------------------


def _basic_data(nrows):
    dt = np.dtype([("id", "i4"), ("x", "f8"), ("name", "S6")])
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows, dtype="i4")
    arr["x"] = np.arange(nrows, dtype="f8") * 0.25
    arr["name"] = [f"n{i % 1000}".encode() for i in range(nrows)]
    return arr


def _write_table(td, data, *, kind, ztilelen=None):
    """
    Write `data` as a plain or compressed BINTABLE; return the path.

    rustfits' own compress= writer always produces a CompressedTableHDU
    regardless of size (no cfitsio copy-verbatim fallback), so small
    fixtures are fine.
    """
    path = os.path.join(td, f"{kind}.fits")
    with rustfits.FITS(path, "w+") as f:
        if kind == "plain":
            f.create_table_hdu(data.dtype, nrows=len(data))
        else:
            f.create_table_hdu(
                data.dtype, nrows=len(data), compress=True, ztilelen=ztilelen
            )
        f[1].write(data)
    return path


KINDS = ["plain", "compressed"]


# -------------------- row mode --------------------


@pytest.mark.parametrize("kind", KINDS)
def test_row_iter_yields_void_records(kind):
    data = _basic_data(80)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=16)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            rows = list(hdu)
            assert len(rows) == len(data)
            assert isinstance(rows[0], np.void)
            for i, row in enumerate(rows):
                assert row["id"] == data["id"][i]
                assert row["x"] == data["x"][i]
                # A/S columns read back as str (U) by default
                assert row["name"] == data["name"][i].decode()


@pytest.mark.parametrize("kind", KINDS)
def test_row_iter_matches_getitem(kind):
    data = _basic_data(50)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=16)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            for i, row in enumerate(hdu):
                np.testing.assert_array_equal(row, hdu[i])


@pytest.mark.parametrize("kind", KINDS)
def test_iter_method_equals_dunder(kind):
    data = _basic_data(20)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            a = [r["id"] for r in hdu]
            b = [r["id"] for r in hdu.iter()]
            assert a == b


@pytest.mark.parametrize("kind", KINDS)
def test_independent_iterators(kind):
    data = _basic_data(10)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            it1 = iter(hdu)
            it2 = iter(hdu)
            assert next(it1)["id"] == 0
            assert next(it2)["id"] == 0  # independent cursor
            assert next(it1)["id"] == 1


# -------------------- chunk mode --------------------


@pytest.mark.parametrize("kind", KINDS)
def test_chunk_sizes_and_content(kind):
    data = _basic_data(250)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=64)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            chunks = list(hdu.iter(chunksize=100))
            assert [c.shape[0] for c in chunks] == [100, 100, 50]
            assert all(c.dtype.names == data.dtype.names for c in chunks)
            np.testing.assert_array_equal(np.concatenate(chunks), hdu.read())


@pytest.mark.parametrize("kind", KINDS)
def test_chunk_evenly_divides(kind):
    data = _basic_data(200)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=50)
        with rustfits.FITS(path, "r") as f:
            chunks = list(f[1].iter(chunksize=50))
            assert [c.shape[0] for c in chunks] == [50, 50, 50, 50]


@pytest.mark.parametrize("kind", KINDS)
def test_chunksize_one_yields_len1_arrays(kind):
    data = _basic_data(5)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            chunks = list(f[1].iter(chunksize=1))
            assert len(chunks) == 5
            for c in chunks:
                # chunksize=1 yields a shape-(1,) array, NOT np.void
                assert isinstance(c, np.ndarray)
                assert c.shape == (1,)


@pytest.mark.parametrize("kind", KINDS)
def test_chunk_larger_than_table(kind):
    data = _basic_data(7)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            chunks = list(f[1].iter(chunksize=1000))
            assert len(chunks) == 1
            assert chunks[0].shape[0] == 7


# -------------------- columns= / scale= forwarding --------------------


@pytest.mark.parametrize("kind", KINDS)
def test_columns_forwarding_row_mode(kind):
    data = _basic_data(40)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=16)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            rows = list(hdu.iter(columns=["x", "id"]))
            assert rows[0].dtype.names == ("x", "id")
            for i, row in enumerate(rows):
                assert row["id"] == data["id"][i]
                assert row["x"] == data["x"][i]


@pytest.mark.parametrize("kind", KINDS)
def test_columns_forwarding_chunk_mode(kind):
    data = _basic_data(120)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind, ztilelen=32)
        with rustfits.FITS(path, "r") as f:
            chunks = list(f[1].iter(chunksize=50, columns=["id"]))
            assert [c.shape[0] for c in chunks] == [50, 50, 20]
            assert all(c.dtype.names == ("id",) for c in chunks)


@pytest.mark.parametrize("kind", KINDS)
def test_single_column_iter_yields_one_field_record(kind):
    # Documented quirk: iter(columns=["x"]) yields 1-field records, not
    # bare scalars (it always goes through read()).
    data = _basic_data(6)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            rows = list(f[1].iter(columns=["x"]))
            assert rows[0].dtype.names == ("x",)
            assert rows[3]["x"] == data["x"][3]


def test_scale_false_forwarding_plain():
    # u2 column rides the unsigned-int trick (stored i2 + TZERO).
    # scale=True -> u2; scale=False -> the raw stored i2.
    dt = np.dtype([("flag", "u2")])
    arr = np.zeros(10, dtype=dt)
    arr["flag"] = np.arange(10, dtype="u2") + 60000  # forces the trick
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.fits")
        with rustfits.FITS(path, "w+") as f:
            f.create_table_hdu(dt, nrows=len(arr))
            f[1].write(arr)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            scaled = list(hdu.iter())
            raw = list(hdu.iter(scale=False))
            assert scaled[0].dtype["flag"] == np.dtype("u2")
            assert raw[0].dtype["flag"] == np.dtype("i2")
            assert scaled[0]["flag"] == 60000


# -------------------- empty + rejection --------------------


@pytest.mark.parametrize("kind", KINDS)
def test_empty_table_yields_nothing(kind):
    data = _basic_data(0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, f"{kind}.fits")
        with rustfits.FITS(path, "w+") as f:
            if kind == "plain":
                f.create_table_hdu(data.dtype, nrows=0)
            else:
                # compressed empty table needs a non-None handle; skip
                # the write (no rows) — n_tiles=0.
                f.create_table_hdu(data.dtype, nrows=0, compress=True)
        with rustfits.FITS(path, "r") as f:
            assert list(f[1]) == []
            assert list(f[1].iter(chunksize=10)) == []


@pytest.mark.parametrize("kind", KINDS)
def test_chunksize_zero_rejected(kind):
    data = _basic_data(5)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            with pytest.raises(ValueError, match="positive integer"):
                f[1].iter(chunksize=0)


@pytest.mark.parametrize("kind", KINDS)
def test_negative_chunksize_rejected(kind):
    data = _basic_data(5)
    with tempfile.TemporaryDirectory() as td:
        path = _write_table(td, data, kind=kind)
        with rustfits.FITS(path, "r") as f:
            # negative -> usize extraction fails (OverflowError/ValueError)
            with pytest.raises((ValueError, OverflowError)):
                f[1].iter(chunksize=-1)


# -------------------- nrows snapshot --------------------


def test_nrows_snapshotted_at_iter_creation():
    dt = np.dtype([("id", "i4")])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.fits")
        with rustfits.FITS(path, "w+") as f:
            f.create_table_hdu(dt, nrows=3)
            f[1].write(np.array([(0,), (1,), (2,)], dtype=dt))
        with rustfits.FITS(path, "r+") as f:
            hdu = f[1]
            it = iter(hdu)
            first = next(it)
            hdu.append(np.array([(99,)], dtype=dt))  # grow mid-iteration
            seen = [int(first["id"])] + [int(r["id"]) for r in it]
            assert seen == [0, 1, 2]  # appended row not seen


# -------------------- multi-refill (row mode) --------------------


def test_row_mode_multiple_buffer_refills():
    # Wide rows shrink the ~8 MiB byte-budget buffer below the row
    # count, forcing several internal read refills in row mode.  Each
    # row is ~16 KB, so the buffer holds ~512 rows; 1500 rows => 3
    # refills.  Verifies the buffer-boundary logic delivers every row
    # in order.
    dt = np.dtype([("idx", "i4"), ("big", "f8", (2000,))])
    nrows = 1500
    arr = np.zeros(nrows, dtype=dt)
    arr["idx"] = np.arange(nrows, dtype="i4")
    arr["big"][:, 0] = np.arange(nrows, dtype="f8")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "wide.fits")
        with rustfits.FITS(path, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows)
            f[1].write(arr)
        with rustfits.FITS(path, "r") as f:
            seen = [int(r["idx"]) for r in f[1]]
            assert seen == list(range(nrows))


# -------------------- VLA columns --------------------


@pytest.mark.parametrize("kind", KINDS)
def test_vla_column_iteration(kind):
    dt = np.dtype([("id", "i4"), ("v", "O")])
    nrows = 60
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows, dtype="i4")
    for i in range(nrows):
        arr["v"][i] = np.arange(i % 5, dtype="f4")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, f"{kind}.fits")
        with rustfits.FITS(path, "w+") as f:
            if kind == "plain":
                f.create_table_hdu(dt, nrows=nrows, var_dtypes={"v": "f4"})
            else:
                f.create_table_hdu(
                    dt,
                    nrows=nrows,
                    var_dtypes={"v": "f4"},
                    compress=True,
                    ztilelen=16,
                )
            f[1].write(arr)
        with rustfits.FITS(path, "r") as f:
            hdu = f[1]
            rows = list(hdu)
            assert len(rows) == nrows
            for i, row in enumerate(rows):
                assert row["id"] == i
                np.testing.assert_array_equal(
                    row["v"], np.arange(i % 5, dtype="f4")
                )
            # chunk mode: each chunk's VLA field is an object array
            chunks = list(hdu.iter(chunksize=25))
            assert [c.shape[0] for c in chunks] == [25, 25, 10]
            np.testing.assert_array_equal(chunks[1]["id"], np.arange(25, 50))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
