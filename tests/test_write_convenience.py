"""
Top-level + FITS-method write convenience: write_image / write_table.

Covers FITS.write_image / FITS.write_table (the one-call combinations
of create_*_hdu + write) and the rustfits.write_image /
rustfits.write_table top-level wrappers that open a file, write, close.

Each test verifies the result through both same-handle and post-reopen
reads where applicable.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# FITS.write_image
# ---------------------------------------------------------------------------


def test_write_image_returns_hdu():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ret.fits")
        data = np.arange(12, dtype="f4").reshape(3, 4)
        with rustfits.FITS(path, "w+") as f:
            hdu = f.write_image(data, extname="IMG")
            assert isinstance(hdu, rustfits.ImageHDU)
            assert hdu.extname == "IMG"
            assert hdu.shape == (3, 4)
            assert hdu.dtype == np.dtype("f4")


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8", "f4", "f8"])
def test_write_image_dtype_matrix(dtype):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"dt_{dtype}.fits")
        data = np.arange(20, dtype=dtype).reshape(4, 5)
        rustfits.write_image(path, data, extname="X")
        back = rustfits.read(path, "X")
        assert back.dtype == np.dtype(dtype)
        assert np.array_equal(back, data)


@pytest.mark.parametrize("dtype", ["i1", "u2", "u4", "u8"])
def test_write_image_unsigned_trick(dtype):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"u_{dtype}.fits")
        data = np.arange(24, dtype=dtype).reshape(4, 6)
        rustfits.write_image(path, data)
        back = rustfits.read(path)
        assert back.dtype == np.dtype(dtype)
        assert np.array_equal(back, data)


def test_write_image_extname_and_extver():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ev.fits")
        with rustfits.FITS(path, "w+") as f:
            f.write_image(np.zeros(4, dtype="i4"), extname="SCI", extver=3)
        with rustfits.FITS(path) as f:
            assert f[0].extname == "SCI"
            assert f[0].extver == 3


def test_write_image_with_compress():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "comp.fits")
        data = np.arange(100, dtype="i4").reshape(10, 10)
        with rustfits.FITS(path, "w+") as f:
            hdu = f.write_image(data, extname="C", compress="GZIP_1")
            assert isinstance(hdu, rustfits.CompressedImageHDU)
        back = rustfits.read(path, "C")
        assert np.array_equal(back, data)


def test_write_image_with_blank_and_mask_blank_read():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blank.fits")
        data = np.array([1, 2, -999, 4, 5], dtype="i4")
        rustfits.write_image(path, data, blank=-999)
        with rustfits.FITS(path) as f:
            arr = f[0].read(mask_blank=True)
            assert arr.mask.tolist() == [False, False, True, False, False]


def test_write_image_with_masked_array_input():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ma.fits")
        data = np.ma.MaskedArray(
            [1, 2, 3, 4],
            mask=[False, True, False, True],
            dtype="i4",
        )
        rustfits.write_image(path, data, blank=-9999)
        with rustfits.FITS(path) as f:
            arr = f[0].read(mask_blank=True)
            assert arr.mask.tolist() == [False, True, False, True]
            assert arr[0] == 1 and arr[2] == 3


def test_write_image_with_header_from_dict():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hdr.fits")
        hdr = {"OBJECT": "NGC 1234", "RA": 12.3456, "DEC": -45.6}
        rustfits.write_image(
            path, np.zeros(3, dtype="i4"), extname="X", header=hdr
        )
        with rustfits.FITS(path) as f:
            assert f[0].header["OBJECT"] == "NGC 1234"
            assert f[0].header["RA"] == 12.3456
            assert f[0].header["DEC"] == -45.6


def test_write_image_with_header_from_fitsheader():
    with tempfile.TemporaryDirectory() as d:
        # Build a source HDU whose header we'll copy from.
        src_path = os.path.join(d, "src.fits")
        with rustfits.FITS(src_path, "w+") as f:
            src = f.write_image(
                np.zeros(2, dtype="i4"),
                extname="S",
                header={"OBJECT": "M51"},
            )
            src_header = src.header

            # Use that header on a second write.
            dst_path = os.path.join(d, "dst.fits")
            with rustfits.FITS(dst_path, "w+") as g:
                g.write_image(np.zeros(3, dtype="i4"), header=src_header)

        with rustfits.FITS(dst_path) as f:
            assert f[0].header["OBJECT"] == "M51"


def test_write_image_accepts_list_input():
    """asanyarray promotes Python lists to ndarray."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lst.fits")
        rustfits.write_image(path, [[1, 2, 3], [4, 5, 6]])
        back = rustfits.read(path)
        assert back.shape == (2, 3)


def test_write_image_rejects_unsupported_dtype():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.fits")
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="unsupported numpy dtype"):
                f.write_image(np.array([1 + 2j], dtype="c8"))


# ---------------------------------------------------------------------------
# FITS.write_table — structured ndarray
# ---------------------------------------------------------------------------


def _example_struct():
    dt = np.dtype([("x", "f8"), ("y", "i4"), ("name", "S8")])
    return np.array(
        [(1.0, 10, b"foo"), (2.0, 20, b"bar"), (3.0, 30, b"baz")],
        dtype=dt,
    )


def test_write_table_returns_hdu():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ret.fits")
        rows = _example_struct()
        with rustfits.FITS(path, "w+") as f:
            hdu = f.write_table(rows, extname="DATA")
            assert isinstance(hdu, rustfits.TableHDU)
            assert hdu.nrows == 3
            assert hdu.colnames == ("x", "y", "name")
            assert hdu.extname == "DATA"


def test_write_table_struct_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "struct.fits")
        rows = _example_struct()
        rustfits.write_table(path, rows, extname="DATA")
        back = rustfits.read(path, "DATA")
        assert back.dtype.names == rows.dtype.names
        assert back["x"].tolist() == [1.0, 2.0, 3.0]
        assert back["y"].tolist() == [10, 20, 30]


def test_write_table_dict_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dict.fits")
        data = {
            "a": np.array([1, 2, 3], dtype="i4"),
            "b": np.array([0.1, 0.2, 0.3], dtype="f4"),
        }
        rustfits.write_table(path, data, extname="D")
        back = rustfits.read(path, "D")
        assert back.dtype.names == ("a", "b")
        assert back["a"].tolist() == [1, 2, 3]
        assert np.allclose(back["b"], [0.1, 0.2, 0.3])


def test_write_table_list_plus_names_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "list.fits")
        arrs = [
            np.array([1.5, 2.5], dtype="f8"),
            np.array([10, 20], dtype="i2"),
        ]
        rustfits.write_table(path, arrs, names=["x", "y"], extname="L")
        back = rustfits.read(path, "L")
        assert back.dtype.names == ("x", "y")


def test_write_table_dict_unequal_lengths_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "unequal.fits")
        data = {
            "a": np.array([1, 2, 3], dtype="i4"),
            "b": np.array([1, 2], dtype="i4"),
        }
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="column lengths disagree"):
                f.write_table(data)


def test_write_table_list_without_names_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "noname.fits")
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="requires names="):
                f.write_table([np.array([1, 2, 3], dtype="i4")])


def test_write_table_names_with_struct_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "redundant.fits")
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="not valid with structured"):
                f.write_table(_example_struct(), names=["x", "y", "name"])


def test_write_table_names_with_dict_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "redundant.fits")
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="not valid with dict"):
                f.write_table({"a": np.arange(3, dtype="i4")}, names=["a"])


def test_write_table_list_names_length_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lenmm.fits")
        with rustfits.FITS(path, "w+") as f:
            with pytest.raises(ValueError, match="does not match"):
                f.write_table(
                    [np.arange(3, dtype="i4"), np.arange(3, dtype="f4")],
                    names=["only_one"],
                )


def test_write_table_with_units():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "units.fits")
        data = {"x": np.arange(3, dtype="f8"), "y": np.arange(3, dtype="i4")}
        with rustfits.FITS(path, "w+") as f:
            hdu = f.write_table(data, units={"x": "m", "y": "count"})
            assert hdu.units == {"x": "m", "y": "count"}


def test_write_table_with_compress():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "comp.fits")
        data = {
            "x": np.arange(50, dtype="f8"),
            "y": np.arange(50, dtype="i4"),
        }
        with rustfits.FITS(path, "w+") as f:
            hdu = f.write_table(data, extname="T", compress=True)
            assert isinstance(hdu, rustfits.CompressedTableHDU)
        back = rustfits.read(path, "T")
        assert back["x"].tolist() == list(range(50))


def test_write_table_with_var_dtypes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vla.fits")
        # Structured dtype with an Object field declared via var_dtypes.
        dt = np.dtype([("id", "i4"), ("samples", "O")])
        rows = np.empty(3, dtype=dt)
        rows["id"] = [1, 2, 3]
        rows["samples"][0] = np.array([1.0, 2.0], dtype="f4")
        rows["samples"][1] = np.array([3.0, 4.0, 5.0], dtype="f4")
        rows["samples"][2] = np.array([], dtype="f4")
        with rustfits.FITS(path, "w+") as f:
            f.write_table(rows, extname="V", var_dtypes={"samples": "f4"})
        back = rustfits.read(path, "V")
        assert back["id"].tolist() == [1, 2, 3]
        assert back["samples"][0].tolist() == [1.0, 2.0]
        assert back["samples"][2].tolist() == []


def test_write_table_auto_creates_primary():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "auto.fits")
        with rustfits.FITS(path, "w+") as f:
            f.write_table({"x": np.arange(3, dtype="i4")}, extname="T")
            assert len(f) == 2
            assert f[0].has_data is False  # auto-primary
            assert f[1].extname == "T"
        with rustfits.FITS(path) as f:
            assert len(f) == 2
            assert f[1].extname == "T"


def test_write_table_with_header_dict():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hdr.fits")
        rustfits.write_table(
            path,
            {"x": np.arange(3, dtype="i4")},
            extname="T",
            header={"OBSERVER": "Hubble"},
        )
        with rustfits.FITS(path) as f:
            assert f[1].header["OBSERVER"] == "Hubble"


# ---------------------------------------------------------------------------
# Top-level rustfits.write_image / rustfits.write_table
# ---------------------------------------------------------------------------


def test_top_level_write_image_returns_none():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "top.fits")
        result = rustfits.write_image(path, np.arange(4, dtype="i4"))
        assert result is None


def test_top_level_write_table_returns_none():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "top.fits")
        result = rustfits.write_table(path, {"x": np.arange(3, dtype="i4")})
        assert result is None


def test_top_level_write_image_truncates_existing_file():
    """Default mode='w+' must truncate (matches fitsio 'rw' + clobber)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trunc.fits")
        rustfits.write_image(path, np.arange(100, dtype="i4"))
        # Overwrite with smaller data.
        rustfits.write_image(path, np.arange(4, dtype="i4"))
        back = rustfits.read(path)
        assert back.tolist() == [0, 1, 2, 3]


def test_top_level_write_image_append_with_rplus_mode():
    """mode='r+' should append to an existing file, not truncate."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "app.fits")
        rustfits.write_image(path, np.zeros(3, dtype="i4"), extname="FIRST")
        rustfits.write_image(
            path, np.ones(3, dtype="i4"), extname="SECOND", mode="r+"
        )
        with rustfits.FITS(path) as f:
            assert len(f) == 2
            assert f[0].extname == "FIRST"
            assert f[1].extname == "SECOND"


def test_top_level_write_table_append_with_rplus_mode():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tabapp.fits")
        rustfits.write_table(
            path, {"a": np.arange(3, dtype="i4")}, extname="T1"
        )
        rustfits.write_table(
            path,
            {"b": np.arange(2, dtype="f4")},
            extname="T2",
            mode="r+",
        )
        with rustfits.FITS(path) as f:
            # Auto-primary + T1 + T2.
            assert len(f) == 3
            assert f[1].extname == "T1"
            assert f[2].extname == "T2"


# ---------------------------------------------------------------------------
# rustfits.write — minimal auto-dispatch convenience
# ---------------------------------------------------------------------------


def test_top_level_write_dispatches_image_from_plain_ndarray():
    """A plain (non-structured) ndarray routes to write_image."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "img.fits")
        data = np.arange(12, dtype="f4").reshape(3, 4)
        rustfits.write(path, data)
        with rustfits.FITS(path) as f:
            # Image lands in the primary HDU when the file is fresh.
            assert isinstance(f[0], rustfits.ImageHDU)
        back = rustfits.read(path)
        np.testing.assert_array_equal(back, data)


def test_top_level_write_dispatches_table_from_structured_ndarray():
    """A structured ndarray (dtype.fields is not None) routes to write_table."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tab.fits")
        dt = np.dtype([("a", "i4"), ("b", "f8")])
        data = np.zeros(5, dtype=dt)
        data["a"] = np.arange(5)
        data["b"] = np.linspace(0, 1, 5)
        rustfits.write(path, data)
        with rustfits.FITS(path) as f:
            assert isinstance(f[1], rustfits.TableHDU)
        back = rustfits.read(path)
        np.testing.assert_array_equal(back["a"], data["a"])
        np.testing.assert_allclose(back["b"], data["b"])


def test_top_level_write_dispatches_table_from_dict():
    """A {name: array} dict routes to write_table."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dtab.fits")
        rustfits.write(
            path,
            {"x": np.arange(3, dtype="i4"), "y": np.arange(3, dtype="f4")},
        )
        with rustfits.FITS(path) as f:
            assert isinstance(f[1], rustfits.TableHDU)
        back = rustfits.read(path)
        assert back.dtype.names == ("x", "y")
        np.testing.assert_array_equal(back["x"], [0, 1, 2])


def test_top_level_write_rejects_list_of_arrays():
    """list/tuple-of-arrays needs write_table directly (for names=)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rej.fits")
        arrays = [np.arange(3, dtype="i4"), np.arange(3, dtype="f4")]
        with pytest.raises(ValueError, match="write_image|write_table"):
            rustfits.write(path, arrays)


def test_top_level_write_rejects_unsupported_type():
    """A plain string isn't writable."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rej.fits")
        with pytest.raises(ValueError):
            rustfits.write(path, "not data")


def test_top_level_write_truncates_then_overwrites():
    """Default mode='w+' truncates an existing file."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trunc.fits")
        rustfits.write(path, np.arange(100, dtype="i4"))
        rustfits.write(path, np.arange(4, dtype="i4"))
        back = rustfits.read(path)
        assert back.tolist() == [0, 1, 2, 3]


def test_top_level_write_appends_with_rplus_mode():
    """mode='r+' appends to an existing file without truncating."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "app.fits")
        rustfits.write(path, np.zeros(3, dtype="i4"))
        rustfits.write(path, {"y": np.ones(2, dtype="f4")}, mode="r+")
        with rustfits.FITS(path) as f:
            # Primary (with image data) + table.
            assert len(f) == 2
            assert isinstance(f[0], rustfits.ImageHDU)
            assert isinstance(f[1], rustfits.TableHDU)


def test_top_level_write_forwards_header():
    """`header=` is passed through to the underlying write_* call."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hdr.fits")
        data = np.arange(4, dtype="i4")
        rustfits.write(path, data, header={"object": "ngc1"})
        _, hdr = rustfits.read(path, header=True)
        assert hdr["object"] == "ngc1"


def test_top_level_write_forwards_extname_image():
    """`extname=` sets EXTNAME on an image HDU; FITS["name"] looks
    it up case-insensitively."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ext_img.fits")
        rustfits.write(path, np.arange(4, dtype="i4"), extname="sci")
        with rustfits.FITS(path) as f:
            # Case is preserved on disk for EXTNAME string values.
            assert f[0].extname == "sci"
            # And case-insensitive lookup works.
            assert isinstance(f["SCI"], rustfits.ImageHDU)


def test_top_level_write_forwards_extname_table():
    """`extname=` sets EXTNAME on a table HDU built from a dict."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ext_tab.fits")
        rustfits.write(
            path,
            {"a": np.arange(3, dtype="i4")},
            extname="cat",
        )
        with rustfits.FITS(path) as f:
            assert f[1].extname == "cat"
            assert isinstance(f[1], rustfits.TableHDU)
            # Case-insensitive EXTNAME lookup matches.
            assert isinstance(f["CAT"], rustfits.TableHDU)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
