"""
ZIMAGE Phase 4: GZIP_1 / GZIP_2 whole-image read + slicing,
plus the fallback-column dispatch (GZIP_COMPRESSED_DATA and
UNCOMPRESSED_DATA used when the primary column is empty).

Round-trip tests use fitsio to write GZIP_1 / GZIP_2 fixtures
and rustfits to read them back, checking byte-exactness.  Also
covers:
    - isinstance(hdu, ImageHDU) (inheritance is unchanged from
      Phase 2)
    - scale=True / scale=False composition with BSCALE/BZERO
    - __getitem__ slicing works (same dispatch as Phase 3, just
      a different decoder under the hood)
    - All bitpix supported by fitsio's GZIP encoder (u1, i2, i4 —
      it refuses i8; that's a cfitsio limit, not a spec limit)
    - GZIP fallback column dispatch (synthetic fixture: primary
      COMPRESSED_DATA empty, GZIP_COMPRESSED_DATA populated)
    - UNCOMPRESSED fallback column dispatch (synthetic fixture)

KNOWN FOLLOW-UP: fitsio refuses to compress i8 (TLONGLONG)
images — see CLAUDE.md ZIMAGE roadmap.  Worth checking whether
the FITS Tile Compression Convention itself permits i8 with
GZIP (almost certainly yes — GZIP is a byte-stream codec, no
inherent BITPIX restriction), in which case our reader is
already fine and only the fixture story needs work (hand-craft
a fixture, or use astropy if it has more relaxed limits).
"""

import gzip
import io
import os
import struct
import tempfile

import numpy as np
import pytest

import rustfits

fitsio = pytest.importorskip("fitsio")


def _write_gzip(tmpdir, shape, dtype, algo, tile_dims=None, start_value=0):
    """
    Build a GZIP-compressed-image fixture with fitsio.  Data is a
    contiguous range starting at `start_value`, reshaped to `shape`.
    """
    fname = os.path.join(tmpdir, "t.fits.fz")
    n = int(np.prod(shape))
    data = np.arange(
        start_value,
        start_value + n,
        dtype=dtype,
    ).reshape(shape)
    kw = {"compress": algo}
    if tile_dims is not None:
        kw["tile_dims"] = tile_dims
    with fitsio.FITS(fname, "rw") as f:
        f.write(data, **kw)
    return fname, data


# -------------------- round-trip exactness -------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_roundtrip_2d(algo, dtype):
    """Various BITPIX, 2-D image with explicit tile shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (10, 10),
            dtype,
            algo,
            tile_dims=(5, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        assert got.dtype == data.dtype
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_roundtrip_1d(algo):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (50,),
            "i4",
            algo,
            tile_dims=(50,),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_roundtrip_3d(algo):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (4, 6, 8),
            "i4",
            algo,
            tile_dims=(2, 3, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_roundtrip_edge_tiles(algo):
    """Image dims not a multiple of tile dims — edge tiles are
    smaller than nominal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (7, 11),
            "i4",
            algo,
            tile_dims=(4, 5),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_roundtrip_default_tile_shape(algo):
    """No tile_dims → fitsio uses the row-tile default (ZTILE1 =
    NAXIS1, others = 1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(tmpdir, (5, 6), "i4", algo)
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


# -------------------- inheritance ----------------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_compressed_image_is_image_hdu(algo):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_gzip(
            tmpdir,
            (8, 8),
            "i4",
            algo,
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert isinstance(hdu, rustfits.ImageHDU)
        assert isinstance(hdu, rustfits.CompressedImageHDU)


# -------------------- accessors --------------------------------


@pytest.mark.parametrize(
    "algo,expected", [("GZIP_1", "GZIP_1"), ("GZIP_2", "GZIP_2")]
)
def test_compression_type_accessor(algo, expected):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_gzip(
            tmpdir,
            (4, 4),
            "i4",
            algo,
            tile_dims=(2, 2),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
        assert hdu.compression.zcmptype == expected


# -------------------- slicing -------------------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_slice_basic(algo):
    """A 2-D slice across tile boundaries decodes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (10, 10),
            "i4",
            algo,
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][2:8, 3:9]
        np.testing.assert_array_equal(got, data[2:8, 3:9])


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_slice_int_collapses(algo):
    """All-int multi-axis index returns a numpy scalar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, data = _write_gzip(
            tmpdir,
            (6, 6),
            "i4",
            algo,
            tile_dims=(3, 3),
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1][2, 5]
        assert got == data[2, 5]
        # scalar, not ndarray
        assert isinstance(got, np.integer)


# -------------------- scaling composition --------------------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_scale_default_applies_bzero(algo):
    """fitsio writing a u2 image triggers the unsigned-int trick:
    BITPIX=16 + BZERO=32768.  Default scale=True should return u2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        # Use u2 values that include the half-range (32768) so the
        # signed/unsigned distinction matters.
        data = np.array(
            [[0, 1000, 32768, 65535], [10, 100, 50000, 200]],
            dtype="u2",
        )
        with fitsio.FITS(fname, "rw") as f:
            f.write(data, compress=algo, tile_dims=(2, 2))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_scale_false_returns_raw(algo):
    """scale=False returns the storage dtype (i2 for a u2-encoded
    image), with raw (signed) values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits.fz")
        data = np.array([[0, 32768, 65535, 1000]], dtype="u2")
        with fitsio.FITS(fname, "rw") as f:
            f.write(data, compress=algo, tile_dims=(1, 4))
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read(scale=False)
        assert got.dtype == np.int16
        # u2 0 → i2 -32768; u2 32768 → i2 0; u2 65535 → i2 32767
        np.testing.assert_array_equal(
            got,
            np.array([[-32768, 0, 32767, -31768]], dtype="i2"),
        )


# -------------------- tile cache works for GZIP too ---------------


@pytest.mark.parametrize("algo", ["GZIP_1", "GZIP_2"])
def test_cache_populates_on_gzip_read(algo):
    """The cache is algorithm-agnostic — a GZIP read should warm
    it just like a RICE read does."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname, _ = _write_gzip(
            tmpdir,
            (8, 8),
            "i4",
            algo,
            tile_dims=(4, 4),
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            assert hdu.tile_cache_used == 0
            _ = hdu.read()
            assert hdu.tile_cache_used > 0


# ==================== fallback columns ============================
#
# fitsio's high-level write API doesn't expose the fallback knobs
# (it relies on cfitsio's automatic per-tile fallback decision,
# which is hard to force from the outside).  So these tests build
# fixtures by hand: a minimal BINTABLE with TWO data columns,
# primary empty, fallback populated.


def _pad_to_block(buf: bytes) -> bytes:
    pad = (2880 - len(buf) % 2880) % 2880
    return buf + b" " * pad


def _build_card(key: str, value, comment: str = "") -> bytes:
    """
    Minimal FITS card builder for fixtures.  Handles int / bool /
    str values.  Output is exactly 80 bytes, ASCII.
    """
    if isinstance(value, bool):
        val_str = "T" if value else "F"
        card = f"{key:<8}= {val_str:>20}"
    elif isinstance(value, int):
        card = f"{key:<8}= {value:>20d}"
    elif isinstance(value, str):
        val_str = f"'{value:<8}'"
        card = f"{key:<8}= {val_str:<20}"
    else:
        raise TypeError(f"unsupported value type: {type(value)}")
    if comment:
        card = f"{card} / {comment}"
    return card.ljust(80).encode("ascii")


def _build_end_card() -> bytes:
    return ("END" + " " * 77).encode("ascii")


def _build_primary_header() -> bytes:
    """Empty primary HDU header (NAXIS=0)."""
    cards = [
        _build_card("SIMPLE", True),
        _build_card("BITPIX", 8),
        _build_card("NAXIS", 0),
        _build_card("EXTEND", True),
        _build_end_card(),
    ]
    return _pad_to_block(b"".join(cards))


def _build_fallback_fixture(
    fname: str,
    image_shape: tuple,
    tile_shape: tuple,
    zbitpix: int,
    *,
    primary_payloads: list,
    gzip_payloads: list = None,
    uncompressed_payloads: list = None,
):
    """
    Hand-build a single-tile-row ZIMAGE BINTABLE where:
      - primary COMPRESSED_DATA may be empty (zero-length descriptor)
        or populated;
      - GZIP_COMPRESSED_DATA fallback column is added iff
        gzip_payloads is given;
      - UNCOMPRESSED_DATA fallback column is added iff
        uncompressed_payloads is given.

    `*_payloads` are lists indexed by tile-row (the row count is
    derived from the tile/image shapes).  Each entry is `bytes` for
    the heap payload of that tile in that column; pass `b""` for an
    empty (descriptor = 0 nelements) entry.

    The heap is laid out as: all primary payloads concatenated,
    then all GZIP payloads, then all UNCOMPRESSED payloads.  Each
    column's descriptors point at its slice of the heap.
    """
    n_tiles = 1
    for axis_img, axis_tile in zip(image_shape, tile_shape):
        n_tiles *= (axis_img + axis_tile - 1) // axis_tile
    assert len(primary_payloads) == n_tiles
    if gzip_payloads is not None:
        assert len(gzip_payloads) == n_tiles
    if uncompressed_payloads is not None:
        assert len(uncompressed_payloads) == n_tiles

    columns = [("COMPRESSED_DATA", primary_payloads)]
    if gzip_payloads is not None:
        columns.append(("GZIP_COMPRESSED_DATA", gzip_payloads))
    if uncompressed_payloads is not None:
        columns.append(("UNCOMPRESSED_DATA", uncompressed_payloads))
    n_cols = len(columns)

    # Lay out the heap; each tile's descriptor records that column's
    # current offset into the running heap buffer.
    heap = io.BytesIO()
    col_descs = []  # list of [(nelem, offset)] per row, per column
    for _, payloads in columns:
        per_row = []
        for payload in payloads:
            if len(payload) == 0:
                per_row.append((0, 0))
            else:
                off = heap.tell()
                heap.write(payload)
                per_row.append((len(payload), off))
        col_descs.append(per_row)
    heap_bytes = heap.getvalue()
    pcount = len(heap_bytes)

    # NAXIS1 = 8 bytes per column (P descriptor = two big-endian u32s).
    row_width = 8 * n_cols
    naxis2 = n_tiles

    # Build the BINTABLE header.
    cards = [
        _build_card("XTENSION", "BINTABLE"),
        _build_card("BITPIX", 8),
        _build_card("NAXIS", 2),
        _build_card("NAXIS1", row_width),
        _build_card("NAXIS2", naxis2),
        _build_card("PCOUNT", pcount),
        _build_card("GCOUNT", 1),
        _build_card("TFIELDS", n_cols),
    ]
    for i, (ttype, _) in enumerate(columns, start=1):
        cards.append(_build_card(f"TFORM{i}", "1PB"))
        cards.append(_build_card(f"TTYPE{i}", ttype))
    # ZIMAGE plus image-shape Z-cards (FITS axis order).
    cards.append(_build_card("ZIMAGE", True))
    cards.append(_build_card("ZCMPTYPE", "RICE_1"))
    cards.append(_build_card("ZBITPIX", zbitpix))
    cards.append(_build_card("ZNAXIS", len(image_shape)))
    for i, n in enumerate(reversed(image_shape), start=1):  # FITS order
        cards.append(_build_card(f"ZNAXIS{i}", n))
    for i, t in enumerate(reversed(tile_shape), start=1):
        cards.append(_build_card(f"ZTILE{i}", t))
    # RICE params (only consulted if a primary tile actually fires
    # the RICE decoder).
    cards.append(_build_card("ZNAME1", "BLOCKSIZE"))
    cards.append(_build_card("ZVAL1", 32))
    cards.append(_build_card("ZNAME2", "BYTEPIX"))
    cards.append(_build_card("ZVAL2", abs(zbitpix) // 8))
    cards.append(_build_end_card())
    bintable_header = _pad_to_block(b"".join(cards))

    # Build the data section: descriptors (row-major) + heap.
    data_buf = io.BytesIO()
    for tile_idx in range(n_tiles):
        for col_idx in range(n_cols):
            nelem, off = col_descs[col_idx][tile_idx]
            data_buf.write(struct.pack(">II", nelem, off))
    data_buf.write(heap_bytes)
    data_section = _pad_to_block(data_buf.getvalue())

    with open(fname, "wb") as fh:
        fh.write(_build_primary_header())
        fh.write(bintable_header)
        fh.write(data_section)


def test_fallback_to_gzip_compressed_data():
    """
    Primary COMPRESSED_DATA is empty; GZIP_COMPRESSED_DATA holds
    a gzip-encoded payload.  The reader must consult the fallback
    column and decode via GZIP_1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        # 4x4 i4 image, one tile = whole image.
        data = np.arange(16, dtype="i4").reshape(4, 4)
        # Encode tile as gzip-of-big-endian-pixels.
        be_bytes = data.astype(">i4").tobytes()
        gz_payload = gzip.compress(be_bytes)
        _build_fallback_fixture(
            fname,
            image_shape=(4, 4),
            tile_shape=(4, 4),
            zbitpix=32,
            primary_payloads=[b""],  # empty primary
            gzip_payloads=[gz_payload],  # falls back to gzip
        )
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits[1]
            assert isinstance(hdu, rustfits.CompressedImageHDU)
            got = hdu.read()
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, data)


def test_fallback_to_uncompressed_data():
    """
    Primary COMPRESSED_DATA empty, GZIP fallback empty,
    UNCOMPRESSED_DATA populated with raw big-endian pixel bytes.
    The reader must byteswap (on LE hosts) and return the data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        data = np.arange(16, dtype="i4").reshape(4, 4)
        be_bytes = data.astype(">i4").tobytes()
        _build_fallback_fixture(
            fname,
            image_shape=(4, 4),
            tile_shape=(4, 4),
            zbitpix=32,
            primary_payloads=[b""],
            gzip_payloads=[b""],  # empty gzip too
            uncompressed_payloads=[be_bytes],
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, data)


def test_mixed_tile_columns():
    """
    Two tiles, primary populated for tile 0, fallback (GZIP) for
    tile 1.  Tests the per-row dispatch logic.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "t.fits")
        # 1x8 image, two tiles of shape 1x4.
        data = np.arange(8, dtype="i4").reshape(1, 8)
        tile0 = data[:, 0:4]
        tile1 = data[:, 4:8]
        # Primary column for tile 0: GZIP-compressed.  Primary
        # uses RICE_1 in normal use; for this hand-crafted fixture
        # the primary decoder is dispatched from ZCMPTYPE — easier
        # to use GZIP for both so we don't have to encode RICE
        # bit-by-bit.  Override ZCMPTYPE in the fixture builder.
        _build_fallback_fixture_with_zcmptype(
            fname,
            image_shape=(1, 8),
            tile_shape=(1, 4),
            zbitpix=32,
            zcmptype="GZIP_1",
            primary_payloads=[
                gzip.compress(tile0.astype(">i4").tobytes()),
                b"",  # empty for tile 1
            ],
            gzip_payloads=[
                b"",  # empty for tile 0 (primary is populated)
                gzip.compress(tile1.astype(">i4").tobytes()),
            ],
        )
        with rustfits.FITS(fname, "r") as fits:
            got = fits[1].read()
        np.testing.assert_array_equal(got, data)


def _build_fallback_fixture_with_zcmptype(
    fname,
    image_shape,
    tile_shape,
    zbitpix,
    zcmptype,
    *,
    primary_payloads,
    gzip_payloads=None,
    uncompressed_payloads=None,
):
    """Same as _build_fallback_fixture but with overridable
    ZCMPTYPE (needed when the primary itself is GZIP)."""
    # Reuse by patching the header build inline.  Easiest is to
    # call the main helper, then rewrite ZCMPTYPE in the file.
    _build_fallback_fixture(
        fname,
        image_shape=image_shape,
        tile_shape=tile_shape,
        zbitpix=zbitpix,
        primary_payloads=primary_payloads,
        gzip_payloads=gzip_payloads,
        uncompressed_payloads=uncompressed_payloads,
    )
    if zcmptype != "RICE_1":
        with open(fname, "rb") as fh:
            buf = bytearray(fh.read())
        needle = b"ZCMPTYPE= 'RICE_1  '"
        idx = buf.find(needle)
        assert idx != -1, "ZCMPTYPE card not found in fixture"
        replacement = f"ZCMPTYPE= '{zcmptype:<8s}'".encode("ascii")
        buf[idx : idx + len(replacement)] = replacement
        with open(fname, "wb") as fh:
            fh.write(buf)
