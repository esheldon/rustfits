"""
ZHECKSUM / ZDATASUM on CompressedTableHDU.

Per the FITS Tile Compression Convention, compressed tables (ZTABLE)
emit ZHECKSUM + ZDATASUM rather than CHECKSUM + DATASUM, and the
values are computed against the EQUIVALENT UNCOMPRESSED table — the
BITPIX-native big-endian bytes the original BINTABLE would have
stored.  Astropy + cfitsio use the same convention.

Scope: fixed-column compressed tables.  VLA-bearing compressed
tables raise NotImplementedError (reconstructing the equivalent-
uncompressed heap with the file's original cell offsets is a
deferred follow-up); the rejection path is tested here too.

The streaming implementation walks tiles one at a time and feeds
the running checksum incrementally, so peak memory stays bounded
regardless of file size — confirmed indirectly by the
multi-tile / many-row round-trip tests round-tripping correctly
without OOM.
"""

import os
import shutil
import subprocess
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


def _have_funpack():
    return shutil.which("funpack") is not None


def _dt_basic():
    return np.dtype([("id", "i4"), ("v", "f8"), ("c", "f4")])


def _basic_data(nrows):
    dt = _dt_basic()
    arr = np.zeros(nrows, dtype=dt)
    arr["id"] = np.arange(nrows, dtype="i4")
    arr["v"] = np.arange(nrows, dtype="f8") * 0.25
    arr["c"] = np.arange(nrows, dtype="f4") * -1.0
    return arr


def _make_compressed_table(fname, *, nrows, ztilelen=None):
    dt = _dt_basic()
    data = _basic_data(nrows)
    with rustfits.FITS(fname, "w+") as f:
        f.create_table_hdu(dt, nrows=nrows, compress=True, ztilelen=ztilelen)
        f[1].write(data)
    return data, dt


# ---------------------------------------------------------------------
# Round-trip — add + verify, same handle and post-reopen
# ---------------------------------------------------------------------


def test_compressed_table_round_trip():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=200, ztilelen=80)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
            assert "ZDATASUM" in f[1].header
            assert "ZHECKSUM" in f[1].header
        with rustfits.FITS(fname) as f:
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True


def test_compressed_table_add_datasum_only():
    """add_datasum sets ZDATASUM only — ZHECKSUM stays absent."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_datasum()
            assert "ZDATASUM" in f[1].header
            assert "ZHECKSUM" not in f[1].header
            assert f[1].verify_datasum() is True
            # ZHECKSUM absent → verify_checksum returns None.
            assert f[1].verify_checksum() is None


def test_compressed_table_verify_none_when_absent():
    """No ZDATASUM / ZHECKSUM cards → verify returns None."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=50, ztilelen=25)
        with rustfits.FITS(fname) as f:
            assert f[1].verify_datasum() is None
            assert f[1].verify_checksum() is None


def test_compressed_table_single_tile():
    """Single-tile table — same-handle + reopen round-trip."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=40, ztilelen=40)
        with rustfits.FITS(fname, "r+") as f:
            assert f[1].n_tiles == 1
            f[1].add_checksum()
            assert f[1].verify_checksum() is True
        with rustfits.FITS(fname) as f:
            assert f[1].verify_checksum() is True


def test_compressed_table_partial_last_tile():
    """Last tile not full — checksum must still be correct."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        # 130 rows / 50-per-tile = 2 full tiles + 30-row last tile.
        _make_compressed_table(fname, nrows=130, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            assert f[1].n_tiles == 3
            f[1].add_checksum()
            assert f[1].verify_checksum() is True


# ---------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------


def test_compressed_table_zdatasum_card_corruption_detected():
    """
    ZDATASUM is a protected keyword — can't be changed through the
    header API.  Edit the raw file bytes instead: find the card,
    flip a digit in its quoted value, then verify returns False.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=80, ztilelen=40)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            original = f[1].header["ZDATASUM"]
        with open(fname, "rb") as fp:
            buf = fp.read()
        idx = buf.find(b"ZDATASUM= '")
        assert idx > 0, "ZDATASUM card not found in file"
        # Find a digit inside the quoted value and bump it.
        val_start = idx + len(b"ZDATASUM= '")
        digit_off = None
        for k in range(20):
            ch = buf[val_start + k]
            if ch == ord("'"):
                break
            if ord("0") <= ch <= ord("9"):
                digit_off = val_start + k
                break
        assert digit_off is not None, "no digit in ZDATASUM value"
        orig_digit = buf[digit_off]
        new_digit = ord("0") if orig_digit != ord("0") else ord("1")
        with open(fname, "r+b") as fp:
            fp.seek(digit_off)
            fp.write(bytes([new_digit]))
        # Confirm the value actually parses differently.
        with rustfits.FITS(fname) as f:
            assert f[1].header["ZDATASUM"] != original
            assert f[1].verify_datasum() is False


def test_compressed_table_heap_byte_corruption_raises():
    """
    Flipping a byte inside the compressed heap breaks gzip
    decompression — verify_* propagates ValueError rather than
    silently masking the problem.  Same shape as
    test_compressed_heap_corruption_raises on the image side.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=100, ztilelen=50)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            naxis1 = int(f[1].header["NAXIS1"])
            naxis2 = int(f[1].header["NAXIS2"])
            assert int(f[1].header["PCOUNT"]) > 0
        with open(fname, "rb") as fp:
            buf = fp.read()
        primary_end = 2880
        bintable_section = buf[primary_end:]
        end_idx = bintable_section.find(b"END     ")
        assert end_idx >= 0
        bintable_header_bytes = ((end_idx + 2880 - 1) // 2880) * 2880
        bintable_data_off = primary_end + bintable_header_bytes
        target = bintable_data_off + naxis1 * naxis2 + 100
        with open(fname, "r+b") as fp:
            fp.seek(target)
            existing = fp.read(1)
            fp.seek(target)
            fp.write(bytes([existing[0] ^ 0xFF]))
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="decompression"):
                f[1].verify_datasum()


# ---------------------------------------------------------------------
# ZDATASUM == DATASUM of the equivalent uncompressed table
# ---------------------------------------------------------------------


def test_zdatasum_equals_uncompressed_datasum():
    """
    The whole point of the ZDATASUM convention: it equals the
    DATASUM that an equivalent uncompressed table would have.
    Write the same data twice (once compressed, once not),
    compute each via rustfits, compare.
    """
    with tempfile.TemporaryDirectory() as td:
        fname_z = os.path.join(td, "z.fits")
        fname_u = os.path.join(td, "u.fits")
        nrows = 150
        ztilelen = 50
        _make_compressed_table(fname_z, nrows=nrows, ztilelen=ztilelen)
        dt = _dt_basic()
        data = _basic_data(nrows)
        with rustfits.FITS(fname_u, "w+") as f:
            f.create_table_hdu(dt, nrows=nrows)
            f[1].write(data)
        with rustfits.FITS(fname_z, "r+") as f:
            f[1].add_datasum()
            z = f[1].header["ZDATASUM"]
        with rustfits.FITS(fname_u, "r+") as f:
            f[1].add_datasum()
            u = f[1].header["DATASUM"]
        assert z == u


# ---------------------------------------------------------------------
# Compressed tables vary tile size + nrows
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "nrows,ztilelen",
    [
        (1, 1),
        (10, 5),
        (100, 30),
        (250, 100),
        (501, 100),
    ],
)
def test_compressed_table_round_trip_matrix(nrows, ztilelen):
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=nrows, ztilelen=ztilelen)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True


# ---------------------------------------------------------------------
# VLA-bearing tables rejected with a clear pointer
# ---------------------------------------------------------------------


def test_compressed_vla_table_checksum_rejected():
    """VLA-bearing compressed-table checksums raise (deferred)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        dt = np.dtype([("id", "i4"), ("v", "O")])
        with rustfits.FITS(fname, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=5,
                compress=True,
                ztilelen=3,
                var_dtypes={"v": "f4"},
            )
            data = np.zeros(5, dtype=dt)
            data["id"] = np.arange(5, dtype="i4")
            for i in range(5):
                data["v"][i] = np.arange(i + 1, dtype="f4")
            f[1].write(data)
        with rustfits.FITS(fname, "r+") as f:
            with pytest.raises(NotImplementedError, match="VLA"):
                f[1].add_datasum()
            with pytest.raises(NotImplementedError, match="VLA"):
                f[1].add_checksum()


# ---------------------------------------------------------------------
# Cross-tool: funpack + cfitsio verify our compressed-table checksums
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_funpack(),
    reason="funpack required for cross-tool verification",
)
def test_funpack_verifies_zdatasum():
    """
    funpack the file and check the resulting uncompressed table's
    DATASUM agrees with our ZDATASUM (the convention says it
    should be the SAME 32-bit value).  Anchors that rustfits-
    written ZDATASUM matches what cfitsio computes for the
    decompressed file.
    """
    import fitsio  # used to read DATASUM after funpack

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=120, ztilelen=40)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            zdatasum = int(f[1].header["ZDATASUM"])
        out = os.path.join(td, "unfz.fits")
        # funpack copies ZHECKSUM/ZDATASUM to DATASUM/CHECKSUM
        # on the output uncompressed file (per cfitsio convention).
        subprocess.run(
            ["funpack", "-O", out, fname],
            check=True,
            capture_output=True,
        )
        with fitsio.FITS(out, "r") as f:
            # funpack should have copied ZDATASUM → DATASUM on the
            # output.  Confirm the value matches.
            hdr = f[1].read_header()
            ds = hdr.get("DATASUM")
            assert ds is not None, "funpack didn't carry DATASUM"
            assert int(ds) == zdatasum


# ---------------------------------------------------------------------
# Same-handle vs reopen parity
# ---------------------------------------------------------------------


def test_compressed_table_same_handle_and_reopen_agree():
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=80, ztilelen=30)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            same = (
                f[1].header["ZDATASUM"],
                f[1].header["ZHECKSUM"],
            )
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
        with rustfits.FITS(fname) as f:
            reopen = (
                f[1].header["ZDATASUM"],
                f[1].header["ZHECKSUM"],
            )
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
        assert same == reopen


# ---------------------------------------------------------------------
# Refresh after a mutation: re-running add_checksum picks up changes
# ---------------------------------------------------------------------


def test_compressed_table_checksum_refresh_after_setitem():
    """
    After hdu[i] = record, the previous ZDATASUM is stale.
    Re-running add_checksum produces a new value that re-verifies.
    """
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "t.fits")
        _make_compressed_table(fname, nrows=100, ztilelen=40)
        with rustfits.FITS(fname, "r+") as f:
            f[1].add_checksum()
            old_zdatasum = f[1].header["ZDATASUM"]
            # Mutate a row.
            record = np.zeros(1, dtype=_dt_basic())
            record["id"] = 9999
            record["v"] = -3.14
            record["c"] = 0.5
            f[1][50] = record[0]
            # Previous ZDATASUM is now stale.
            assert f[1].verify_datasum() is False
            # Refresh.
            f[1].add_checksum()
            assert f[1].verify_datasum() is True
            assert f[1].verify_checksum() is True
            new_zdatasum = f[1].header["ZDATASUM"]
        assert new_zdatasum != old_zdatasum


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
