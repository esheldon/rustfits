"""
Tests for ZTABLE detection + accessors + stubbed I/O.

Phase 1 ships only the detection plumbing and accessors that report
the original (uncompressed) table's schema.  read/write/etc. raise
NotImplementedError pointing at later phases.

Fixtures are built by running `fpack -table` over a fresh fitsio-
written BINTABLE.  fpack is the only widely-available writer for
this format (astropy doesn't expose a CompTableHDU; fitsio doesn't
have a high-level wrapper for cfitsio's fits_compress_table).
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------


def _have_fpack():
    return shutil.which("fpack") is not None


# Skip the whole module if fpack isn't installed; Phase 1 has no other
# way to build a real ZTABLE fixture.
pytestmark = pytest.mark.skipif(
    not _have_fpack(),
    reason="fpack (cfitsio CLI) is required to build ZTABLE fixtures",
)


def _make_ztable_fixture(td, dtype, data, units=None):
    """
    Build a ZTABLE-compressed fixture via fitsio (write) + fpack -table.
    fpack writes alongside the input file with a .fz suffix.
    """
    import fitsio

    src = os.path.join(td, "src.fits")
    with fitsio.FITS(src, "rw", clobber=True) as f:
        f.write(data, units=units)
    subprocess.run(
        ["fpack", "-table", src],
        check=True,
        capture_output=True,
    )
    fz = src + ".fz"
    assert os.path.exists(fz), f"fpack did not produce {fz}"
    return fz


def _basic_dtype():
    return np.dtype(
        [("a", "i4"), ("b", "f8"), ("c", ("f4", (2, 3))), ("s", "S10")]
    )


def _basic_data(nrows=10000):
    dt = _basic_dtype()
    arr = np.zeros(nrows, dtype=dt)
    arr["a"] = np.arange(nrows, dtype="i4")
    arr["b"] = np.arange(nrows, dtype="f8") * 0.5
    arr["c"] = np.arange(nrows * 6, dtype="f4").reshape(nrows, 2, 3)
    arr["s"] = [f"r{i:04d}".encode() for i in range(nrows)]
    return arr


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------


def test_compressed_table_is_detected():
    """
    A ZTABLE-compressed BINTABLE should open as CompressedTableHDU,
    not the plain TableHDU.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            assert isinstance(f[1], rustfits.CompressedTableHDU)


def test_isinstance_chain():
    """
    CompressedTableHDU is a subclass of TableHDU (which is a subclass
    of HDU), so generic isinstance checks still pick up the compressed
    HDU as a table HDU.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert isinstance(hdu, rustfits.CompressedTableHDU)
            assert isinstance(hdu, rustfits.TableHDU)
            assert isinstance(hdu, rustfits.HDU)


def test_plain_table_unaffected():
    """
    An uncompressed BINTABLE in the same library should still come
    back as TableHDU (and not as CompressedTableHDU).
    """
    import fitsio

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "plain.fits")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(_basic_data())
        with rustfits.FITS(fname, "r") as f:
            assert isinstance(f[1], rustfits.TableHDU)
            assert not isinstance(f[1], rustfits.CompressedTableHDU)


# ---------------------------------------------------------------------
# Accessors — return the UNCOMPRESSED view
# ---------------------------------------------------------------------


def test_nrows_returns_uncompressed_count():
    """
    nrows reads ZNAXIS2 (the original row count), not NAXIS2 (which
    is the number of tiles after compression).
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(nrows=10000),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.nrows == 10000
            assert len(hdu) == 10000


def test_n_tiles_matches_naxis2():
    """
    n_tiles is the on-disk NAXIS2 — one row per tile in the
    compressed table.  Default fpack rowspertile is large enough
    that small fixtures land in a single tile.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.n_tiles >= 1
            # Raw header NAXIS2 should match.
            assert int(hdu.header["NAXIS2"]) == hdu.n_tiles


def test_ztile_rows_is_ztilelen():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.ztile_rows == int(hdu.header["ZTILELEN"])


def test_ncols_matches_tfields():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.ncols == 4
            assert int(hdu.header["TFIELDS"]) == 4


def test_colnames_preserved():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert hdu.colnames == ("a", "b", "c", "s")


def test_hdu_units_preserved():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            assert dict(hdu.units) == {
                "a": "count",
                "b": "Jy",
                "c": "m",
                "s": "char",
            }


def test_dtype_matches_original_schema():
    """
    dtype is built from the per-column ZFORMn (original TFORMn) +
    preserved TDIMn — should equal the dtype of the source data
    (modulo numpy returning U10 vs S10 for FITS A columns, per the
    existing TableHDU.read convention).
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            dt = hdu.dtype
            assert dt.names == ("a", "b", "c", "s")
            assert dt["a"] == np.dtype("<i4")
            assert dt["b"] == np.dtype("<f8")
            assert dt["c"].subdtype == (np.dtype("<f4"), (2, 3))
            assert dt["s"].kind == "U"
            assert dt["s"].itemsize == 40  # 10 chars * 4 bytes/U


def test_compression_dict_per_column():
    """
    The compression accessor returns a dict mapping each column name
    to the FITS-spec algorithm string from ZCTYPn.  fpack picks
    different defaults per dtype: RICE_1 for i4, GZIP_2 for float,
    GZIP_1 for strings.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            algos = dict(hdu.compression)
            assert set(algos.keys()) == {"a", "b", "c", "s"}
            for col, algo in algos.items():
                assert algo in {"RICE_1", "GZIP_1", "GZIP_2"}


def test_extname_and_repr():
    """
    Generic HDU accessors (extname, repr) work; repr names the
    compressed BINTABLE type explicitly.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            hdu = f[1]
            r = repr(hdu)
            assert "BINARY_TBL (compressed)" in r
            assert "rows: 10000" in r
            assert "compression:" in r


# ---------------------------------------------------------------------
# Stubs — every I/O entry point raises NotImplementedError
# ---------------------------------------------------------------------


def test_insert_column_stub_raises():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r+") as f:
            with pytest.raises(NotImplementedError, match="not planned"):
                f[1].insert_column("x", np.arange(10000))


def test_delete_column_stub_raises():
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r+") as f:
            with pytest.raises(NotImplementedError, match="not planned"):
                f[1].delete_column("a")


# ---------------------------------------------------------------------
# Header still accessible (raw Z-prefixed cards visible to the user)
# ---------------------------------------------------------------------


def test_raw_header_cards_visible():
    """
    The user can still inspect the raw ZTABLE/ZNAXIS*/ZFORMn cards
    through hdu.header — useful for debugging and for forwards
    compatibility with hand-written tools.
    """
    with tempfile.TemporaryDirectory() as td:
        fz = _make_ztable_fixture(
            td,
            _basic_dtype(),
            _basic_data(),
            units=["count", "Jy", "m", "char"],
        )
        with rustfits.FITS(fz, "r") as f:
            h = f[1].header
            assert h["ZTABLE"] is True
            assert int(h["ZNAXIS2"]) == 10000
            assert h["ZFORM3"] == "6E"
            assert h["ZCTYP1"] in {"RICE_1", "GZIP_1", "GZIP_2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
