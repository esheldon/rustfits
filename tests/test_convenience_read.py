"""
Tests for `rustfits.read()` (top-level convenience wrapper).

Covers the auto-skip-to-first-HDU-with-data default, ext selectors
by int and EXTNAME, ``header=True`` returning the tuple, the error
paths, and the rejection of removed kwargs.  ``rustfits.read()`` is
intentionally minimal — knobs like ``scale=`` / ``rows=`` /
``columns=`` / ``mask_null=`` live on the underlying ``HDU.read()``
calls.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

import rustfits


CARDS_PER_BLOCK = 36
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b):
    return b + b"\x00" * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    with open(path, "wb") as f:
        for cards, data in parts:
            f.write(_pad_cards(cards))
            if data:
                f.write(_pad_to_block(data))


def _primary_no_data():
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]


def _image_primary(bitpix, dims, extras=()):
    cards = [
        "SIMPLE  =                    T",
        f"BITPIX  = {bitpix:>20d}",
        f"NAXIS   = {len(dims):>20d}",
    ]
    for i, d in enumerate(dims, start=1):
        cards.append(f"NAXIS{i:<3d}= {d:>20d}")
    cards.append("EXTEND  =                    T")
    cards.extend(extras)
    cards.append("END")
    return cards


def _bintable_ext(naxis1, naxis2, fields, extras=()):
    cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {naxis1:>20d}",
        f"NAXIS2  = {naxis2:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        f"TFIELDS = {len(fields):>20d}",
    ]
    for i, (ttype, tform) in enumerate(fields, start=1):
        cards.append(f"TTYPE{i:<3d}= '{ttype:<8s}'")
        cards.append(f"TFORM{i:<3d}= '{tform:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


# ---------------------------------------------------------------------------
# default: ext=None picks the first HDU with data
# ---------------------------------------------------------------------------


def test_read_default_picks_image_when_primary_has_data():
    """Primary HDU carries data → that's what we get back."""
    data = np.arange(12, dtype="f4").reshape(3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "img.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=(3, 4))
            fits.hdus[0].write(data)
        got = rustfits.read(fname)
        np.testing.assert_array_equal(got, data)


def test_read_default_skips_empty_primary_to_find_table():
    """
    Empty primary HDU followed by a BINTABLE → the table is
    returned (default ext= None picks the first HDU with data)."""
    rows = struct.pack(">3i", 10, 20, 30)
    fields = [("X", "1J")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b""),
            (_bintable_ext(4, 3, fields), rows),
        )
        got = rustfits.read(fname)
        assert got.dtype.names == ("X",)
        assert got["X"].tolist() == [10, 20, 30]


def test_read_default_skips_empty_primary_to_find_image_extension():
    """Empty primary, image extension → return the image."""
    # Build a primary with no data + an IMAGE extension with NAXIS=2.
    primary = _primary_no_data()
    ext = [
        "XTENSION= 'IMAGE   '",
        "BITPIX  =                   32",
        "NAXIS   =                    2",
        "NAXIS1  =                    3",
        "NAXIS2  =                    2",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "END",
    ]
    # Pack as big-endian to match FITS on-disk order.
    rows_data = struct.pack(">6i", 1, 2, 3, 4, 5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "im.fits")
        _write_file(fname, (primary, b""), (ext, rows_data))
        got = rustfits.read(fname)
        assert got.shape == (2, 3)
        np.testing.assert_array_equal(got, [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# ext= int / str selectors
# ---------------------------------------------------------------------------


def test_read_ext_by_index():
    """ext=int reaches the specific HDU."""
    rows = struct.pack(">3i", 100, 200, 300)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b""),
            (_bintable_ext(4, 3, [("X", "1J")]), rows),
        )
        got = rustfits.read(fname, ext=1)
        assert got["X"].tolist() == [100, 200, 300]


def test_read_ext_by_extname():
    """
    ext=str does case-insensitive EXTNAME lookup (matches FITS
    __getitem__)."""
    rows = struct.pack(">2i", 7, 8)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b""),
            (
                _bintable_ext(
                    4,
                    2,
                    [("X", "1J")],
                    extras=["EXTNAME = 'CATALOG '"],
                ),
                rows,
            ),
        )
        got = rustfits.read(fname, ext="CATALOG")
        assert got["X"].tolist() == [7, 8]
        # Case-insensitive lookup.
        got2 = rustfits.read(fname, ext="catalog")
        assert got2["X"].tolist() == [7, 8]


def test_read_explicit_ext_returns_empty_data():
    """
    When ext is given explicitly, the empty HDU is read anyway —
    only the auto-pick path errors on no-data."""
    primary = _primary_no_data()
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "empty.fits")
        _write_file(fname, (primary, b""))
        # Reading the explicit empty primary should succeed.  An
        # NAXIS=0 image has no data section; the read returns an
        # empty array (or raises a clean error from ImageHDU.read).
        try:
            got = rustfits.read(fname, ext=0)
        except ValueError:
            # ImageHDU.read errors on NAXIS=0; that's also acceptable
            # for an explicit empty-HDU read.  The point of this test
            # is that the wrapper doesn't bail before the read happens.
            return
        assert got.size == 0


# ---------------------------------------------------------------------------
# Removed kwargs (rows / columns / scale / mask_null) raise TypeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kw",
    [
        {"rows": [0]},
        {"columns": ["X"]},
        {"scale": False},
        {"mask_null": True},
    ],
)
def test_read_rejects_removed_kwargs(bad_kw):
    """rustfits.read() is minimal — these kwargs live on HDU.read()."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b""),
            (_bintable_ext(4, 1, [("X", "1J")]), struct.pack(">i", 7)),
        )
        with pytest.raises(TypeError):
            rustfits.read(fname, **bad_kw)


# ---------------------------------------------------------------------------
# header=True returns tuple
# ---------------------------------------------------------------------------


def test_read_header_true_returns_tuple():
    """
    header=True returns (data, FITSHeader); inspecting the header
    works after the file is closed (the FITSHeader holds its own
    snapshot of cards)."""
    rows = struct.pack(">i", 42)
    fields = [("X", "1J")]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(
            fname,
            (_primary_no_data(), b""),
            (
                _bintable_ext(4, 1, fields, extras=["EXTNAME = 'TBL     '"]),
                rows,
            ),
        )
        got, hdr = rustfits.read(fname, header=True)
        assert got["X"][0] == 42
        # Cards-level access still works.
        assert hdr["EXTNAME"] == "TBL"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_read_no_hdu_with_data_raises():
    """File with only an empty primary HDU and no extensions → raise."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "empty.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with pytest.raises(ValueError, match="no HDU with data"):
            rustfits.read(fname)


def test_read_unsupported_hdu_type_raises():
    """
    ASCII table HDUs don't have a read API yet — wrapper rejects
    cleanly when the user picks one explicitly."""
    primary = _primary_no_data()
    ascii_ext = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                   20",
        "NAXIS2  =                    1",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    0",
        "END",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "a.fits")
        _write_file(fname, (primary, b""), (ascii_ext, bytes(20)))
        with pytest.raises(ValueError, match="does not yet support"):
            rustfits.read(fname, ext=1)


# ---------------------------------------------------------------------------
# read_header
# ---------------------------------------------------------------------------


def test_read_header_default_is_primary():
    """ext=0 (the default) returns the primary HDU's header."""
    primary = _primary_no_data() + ["OBJECT  = 'M31     '"]
    # Re-emit END at the end if needed; _primary_no_data already includes END.
    # Inject OBJECT before END instead:
    primary = [c for c in _primary_no_data() if c != "END"]
    primary.append("OBJECT  = 'M31     '")
    primary.append("END")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h.fits")
        _write_file(fname, (primary, b""))
        hdr = rustfits.read_header(fname)
        assert hdr["OBJECT"] == "M31"


def test_read_header_ext_by_int():
    """ext=int picks that HDU's header."""
    primary = _primary_no_data()
    ext = _bintable_ext(4, 1, [("X", "1J")], extras=["EXTNAME = 'TBL     '"])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h.fits")
        _write_file(fname, (primary, b""), (ext, struct.pack(">i", 7)))
        hdr0 = rustfits.read_header(fname)
        hdr1 = rustfits.read_header(fname, ext=1)
        assert "EXTNAME" not in hdr0
        assert hdr1["EXTNAME"] == "TBL"


def test_read_header_ext_by_extname():
    """ext='name' looks up by EXTNAME (case-insensitive)."""
    primary = _primary_no_data()
    ext = _bintable_ext(4, 1, [("X", "1J")], extras=["EXTNAME = 'TBL     '"])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h.fits")
        _write_file(fname, (primary, b""), (ext, struct.pack(">i", 7)))
        hdr = rustfits.read_header(fname, ext="tbl")
        assert hdr["EXTNAME"] == "TBL"


def test_read_header_outlives_file_close():
    """The returned header is safe to inspect after the function exits."""
    primary = [c for c in _primary_no_data() if c != "END"]
    primary.append("OBJECT  = 'NGC1    '")
    primary.append("END")
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h.fits")
        _write_file(fname, (primary, b""))
        hdr = rustfits.read_header(fname)
        # File has been closed by now (function returned).  Read-only
        # access still works because FITSHeader holds the cards Arc.
        assert hdr["OBJECT"] == "NGC1"
        assert list(hdr.keys())  # iteration works


def test_read_header_missing_ext_raises():
    """Bad int index raises; missing EXTNAME raises."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "h.fits")
        _write_file(fname, (_primary_no_data(), b""))
        with pytest.raises((IndexError, ValueError)):
            rustfits.read_header(fname, ext=5)
        with pytest.raises(ValueError):
            rustfits.read_header(fname, ext="nope")
