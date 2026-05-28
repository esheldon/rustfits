"""
Tests for variable-length ASCII string columns on write (FITS PA).

API: pass `var_dtypes={col: 'S'}` (or 'U' / 'A' / 'S1' / 'U1') to
create_table_hdu; per-cell input is a Python str (ASCII) or bytes.
Round-trip with rustfits read returns Python strs (or bytes via
read_column(as_bytes=True)).

Cross-tool: astropy reads our files as object arrays of chararray
(one chararray per cell); fitsio reads them as padded U<maxlen>.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _make_vla_str_table(tmpdir, nrows, var_letter="S", fill=None):
    """Create a one-column VLA string table and (optionally) write fill."""
    dt = np.dtype([("name", "O")])
    fname = os.path.join(tmpdir, "t.fits")
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows, var_dtypes={"name": var_letter})
        if fill is not None:
            f[1].write(fill)
    return fname, dt


# ---------------------------------------------------------------------------
# create_table_hdu accepts all the string aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("letter", ["S", "U", "A", "a", "S1", "U1"])
def test_create_accepts_string_aliases(letter):
    """
    All ascii-string aliases produce TFORM1='1PA'.  Uppercase S/U
    match numpy's string-kind characters; bare lowercase 's' and
    'u' are NOT accepted because lowercase 'u1' is uint8 (numeric)
    and the case-insensitive collision would be ambiguous.  'a' is
    accepted as the FITS-letter form.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname, _ = _make_vla_str_table(tmp, nrows=0, var_letter=letter)
        with rustfits.FITS(fname) as f:
            assert f[1].header["TFORM1"] == "1PA"


@pytest.mark.parametrize("letter", ["s", "u"])
def test_create_rejects_lowercase_su(letter):
    """Bare lowercase 's' / 'u' are rejected — no numpy precedent."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="unsupported inner dtype"):
            _make_vla_str_table(tmp, nrows=0, var_letter=letter)


def test_create_heap_format_q():
    """heap_format='Q' produces TFORM1='1QA' (16-byte descriptors)."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        dt = np.dtype([("name", "O")])
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=0,
                var_dtypes={"name": "S"},
                heap_format="Q",
            )
            assert f[1].header["TFORM1"] == "1QA"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_basic():
    """str cells round-trip through write → read as Python str."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "alice"
        arr["name"][1] = "bob"
        arr["name"][2] = "wendy_long"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "alice"
            assert got["name"][1] == "bob"
            assert got["name"][2] == "wendy_long"


def test_round_trip_bytes_cells():
    """bytes cells round-trip verbatim via as_bytes=True read."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = b"hello"
        arr["name"][1] = b""  # empty cell
        arr["name"][2] = b"\x00abc\x00"  # embedded NULs
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read_column("name", as_bytes=True)
            assert got[0] == b"hello"
            assert got[1] == b""
            assert got[2] == b"\x00abc\x00"


def test_round_trip_empty_cells():
    """Empty str cells produce nelements=0 descriptors."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = ""
        arr["name"][1] = ""
        arr["name"][2] = ""
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        with rustfits.FITS(fname) as f:
            assert int(f[1].header["PCOUNT"]) == 0
            got = f[1].read()
            for r in range(3):
                assert got["name"][r] == ""


def test_round_trip_mixed_str_and_bytes():
    """A single column can mix str and bytes cells row by row."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "alpha"
        arr["name"][1] = b"beta"
        arr["name"][2] = "gamma_x"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "alpha"
            assert got["name"][1] == "beta"
            assert got["name"][2] == "gamma_x"


def test_round_trip_numpy_str_scalars():
    """numpy.str_ / numpy.bytes_ scalars are accepted as cells."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["name"][0] = np.str_("from_str")
        arr["name"][1] = np.bytes_(b"from_bytes")
        fname, _ = _make_vla_str_table(tmp, nrows=2, fill=arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "from_str"
            assert got["name"][1] == "from_bytes"


def test_mixed_str_vla_with_numeric_columns():
    """String VLA alongside fixed + numeric VLA columns."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        dt = np.dtype([("id", "i4"), ("name", "O"), ("lc", "O")])
        nrows = 3
        arr = np.zeros(nrows, dtype=dt)
        arr["id"] = [1, 2, 3]
        arr["name"][0] = "alpha"
        arr["name"][1] = "beta"
        arr["name"][2] = "gamma"
        arr["lc"][0] = np.array([1.0, 2.0], dtype="f4")
        arr["lc"][1] = np.array([], dtype="f4")
        arr["lc"][2] = np.array([3.0, 4.0, 5.0], dtype="f4")
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=nrows,
                var_dtypes={"name": "S", "lc": "f4"},
            )
            f[1].write(arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            np.testing.assert_array_equal(got["id"], arr["id"])
            for r in range(nrows):
                assert got["name"][r] == arr["name"][r]
                np.testing.assert_array_equal(got["lc"][r], arr["lc"][r])


# ---------------------------------------------------------------------------
# Append + setitem (existing surfaces) work with PA
# ---------------------------------------------------------------------------


def test_append_str_vla_rows():
    """append() should grow PCOUNT by sum of new cell bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(2, dtype=dt)
        arr["name"][0] = "first"
        arr["name"][1] = "second"
        fname, _ = _make_vla_str_table(tmp, nrows=2, fill=arr)
        with rustfits.FITS(fname, "r") as f:
            pc0 = int(f[1].header["PCOUNT"])
        extra = np.zeros(2, dtype=dt)
        extra["name"][0] = "third"
        extra["name"][1] = "fourth"
        with rustfits.FITS(fname, "r+") as f:
            f[1].append(extra)
        with rustfits.FITS(fname) as f:
            assert int(f[1].header["NAXIS2"]) == 4
            assert int(f[1].header["PCOUNT"]) == pc0 + len("third") + len(
                "fourth"
            )
            got = f[1].read()
            assert got["name"][0] == "first"
            assert got["name"][3] == "fourth"


def test_setitem_single_row_str_vla():
    """hdu[i] = record with a string VLA column."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "alpha"
        arr["name"][1] = "beta"
        arr["name"][2] = "gamma"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        new = np.zeros(1, dtype=dt)
        new["name"][0] = "modified"
        with rustfits.FITS(fname, "r+") as f:
            f[1][1] = new[0]
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "alpha"
            assert got["name"][1] == "modified"
            assert got["name"][2] == "gamma"


def test_setitem_whole_str_vla_column():
    """hdu['name'] = arr rewrites every cell of a string VLA column."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "old1"
        arr["name"][1] = "old2"
        arr["name"][2] = "old3"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        new = np.zeros(3, dtype="O")
        new[0] = "new_alpha"
        new[1] = "new_beta"
        new[2] = "new_gamma"
        with rustfits.FITS(fname, "r+") as f:
            f[1]["name"] = new
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["name"][0] == "new_alpha"
            assert got["name"][1] == "new_beta"
            assert got["name"][2] == "new_gamma"


def test_repack_str_vla_drops_orphans():
    """repack() compacts the heap after str VLA setitem orphans."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "a"
        arr["name"][1] = "b"
        arr["name"][2] = "c"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        new = np.zeros(1, dtype=dt)
        with rustfits.FITS(fname, "r+") as f:
            for k in range(5):
                new["name"][0] = "x" * 20
                f[1][0] = new[0]
        with rustfits.FITS(fname, "r") as f:
            mid_pc = int(f[1].header["PCOUNT"])
        with rustfits.FITS(fname, "r+") as f:
            f[1].repack()
        with rustfits.FITS(fname) as f:
            final_pc = int(f[1].header["PCOUNT"])
            assert final_pc < mid_pc
            assert final_pc == 20 + 1 + 1  # "xxx...x" + "b" + "c"
            got = f[1].read()
            assert got["name"][0] == "x" * 20
            assert got["name"][1] == "b"
            assert got["name"][2] == "c"


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_non_ascii_str_rejected():
    """str cells with non-ASCII bytes raise (matches the read-side rule)."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(1, dtype=dt)
        arr["name"][0] = "héllo"
        with pytest.raises(ValueError, match="non-ASCII"):
            _make_vla_str_table(tmp, nrows=1, fill=arr)


def test_non_ascii_bytes_pass_through():
    """bytes cells with non-ASCII bytes are accepted (raw bytes mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(1, dtype=dt)
        arr["name"][0] = b"\xff\xfe\xfd"
        fname, _ = _make_vla_str_table(tmp, nrows=1, fill=arr)
        with rustfits.FITS(fname) as f:
            got = f[1].read_column("name", as_bytes=True)
            assert got[0] == b"\xff\xfe\xfd"


def test_wrong_cell_type_rejected():
    """Non-str/bytes cell (e.g. ndarray) raises a clear error."""
    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(1, dtype=dt)
        arr["name"][0] = np.array([1, 2, 3], dtype="i4")
        with pytest.raises(ValueError, match="str.*bytes"):
            _make_vla_str_table(tmp, nrows=1, fill=arr)


# ---------------------------------------------------------------------------
# Cross-tool: astropy reads our file
# ---------------------------------------------------------------------------


def test_astropy_can_read_our_str_vla():
    """astropy reads rustfits-written PA columns (per-cell chararray)."""
    from astropy.io import fits as ap_fits

    with tempfile.TemporaryDirectory() as tmp:
        dt = np.dtype([("name", "O")])
        arr = np.zeros(3, dtype=dt)
        arr["name"][0] = "alice"
        arr["name"][1] = "bob"
        arr["name"][2] = "wendy_long"
        fname, _ = _make_vla_str_table(tmp, nrows=3, fill=arr)
        with ap_fits.open(fname) as hdul:
            assert hdul[1].header["TFORM1"].startswith("PA") or hdul[1].header[
                "TFORM1"
            ].endswith("PA")
            data = hdul[1].data
            # astropy returns each cell as chararray of single chars.
            assert "".join(str(c) for c in data["name"][0]) == "alice"
            assert "".join(str(c) for c in data["name"][1]) == "bob"
            assert "".join(str(c) for c in data["name"][2]) == "wendy_long"


def test_we_can_read_astropy_str_vla():
    """rustfits reads astropy-written PA columns (round-trip via astropy)."""
    from astropy.io import fits as ap_fits

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "ap.fits")
        col = ap_fits.Column(
            name="NAME",
            format="PA()",
            array=np.array(["alice", "bob", "wendy_long"], dtype="O"),
        )
        hdu = ap_fits.BinTableHDU.from_columns([col])
        hdu.writeto(fname)
        with rustfits.FITS(fname) as f:
            got = f[1].read()
            assert got["NAME"][0] == "alice"
            assert got["NAME"][1] == "bob"
            assert got["NAME"][2] == "wendy_long"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
