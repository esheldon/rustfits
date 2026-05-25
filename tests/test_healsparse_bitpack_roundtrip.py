"""
Healsparse bit-packed sparse-map layout round-trip.

Healsparse stores its bit-packed boolean maps as a 1-D, tile-compressed
uint8 IMAGE HDU (RICE_1) with a `BITPACK=T` marker card and a handful
of healsparse-specific cards (EXTNAME='SPARSE', PIXTYPE='HEALSPARSE',
SENTINEL=False, NSIDE=<int>).  The bit packing itself is done in
Python — healsparse stuffs 8 booleans into each uint8 and wraps the
buffer in `_PackedBoolArray`; FITS sees just a regular uint8 image.

The pieces rustfits has to get right:
  - 1-D uint8 tile-compressed image creation (`Rice1(tile_shape=(N,))`).
  - Partial-last-tile extend (the common case healsparse hits when
    growing the SPARSE buffer one coverage chunk at a time).
  - __setitem__ on a tile-compressed 1-D uint8 image.
  - Custom unprotected header cards round-trip through edits.
  - fitsio cross-read agreement on the produced file (so a real
    healsparse client could swap rustfits in for fitsio without
    surprises).

This test exercises the full path on a synthetic bit-packed buffer
sized like a small coverage chunk; the real workload uses much
larger arrays but the per-chunk mechanics are the same.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _have_fitsio():
    try:
        import fitsio  # noqa: F401

        return True
    except ImportError:
        return False


def _emit_healsparse_cards(hdu, *, nside_sparse, sentinel=False):
    """
    Set the SPARSE-extension header cards healsparse writes alongside
    the bit-packed uint8 buffer.  These are all unprotected keywords
    so rustfits accepts them via __setitem__.
    """
    hdu.header["EXTNAME"] = "SPARSE"
    hdu.header["PIXTYPE"] = "HEALSPARSE"
    hdu.header["BITPACK"] = True
    hdu.header["SENTINEL"] = sentinel
    hdu.header["NSIDE"] = nside_sparse


def _make_packed_buffer(nfine_per_cov, n_cov_chunks, *, seed=0):
    """
    Synthesize a healsparse-shaped sparse buffer: `n_cov_chunks`
    coverage chunks each holding `nfine_per_cov // 8` packed bytes.
    The pattern is deterministic but covers a range of bit densities
    (some chunks dense, some sparse, some all-zero).
    """
    rng = np.random.default_rng(seed)
    bytes_per_chunk = nfine_per_cov // 8
    total = n_cov_chunks * bytes_per_chunk
    # Mix of all-zero (typical for masked-out coverage), uniform
    # random, dense (most pixels set), and sparse (a few pixels set)
    # chunks — mimics real coverage patterns.
    buf = np.zeros(total, dtype=np.uint8)
    for chunk in range(n_cov_chunks):
        beg = chunk * bytes_per_chunk
        end = beg + bytes_per_chunk
        mode = chunk % 4
        if mode == 0:
            pass  # all-zero
        elif mode == 1:
            buf[beg:end] = rng.integers(0, 256, bytes_per_chunk, "u1")
        elif mode == 2:
            buf[beg:end] = 0xFF
            # punch a few holes
            for _ in range(5):
                k = rng.integers(0, bytes_per_chunk)
                buf[beg + k] &= rng.integers(0, 256, dtype="u1")
        else:
            # sparse: a few bytes have a single bit set
            for _ in range(3):
                k = rng.integers(0, bytes_per_chunk)
                bit = rng.integers(0, 8)
                buf[beg + k] |= np.uint8(1) << np.uint8(bit)
    return buf


# ---------------------------------------------------------------------
# Whole-file round-trip
# ---------------------------------------------------------------------


def test_healsparse_layout_round_trip_within_session():
    """
    Create the SPARSE extension exactly as healsparse would, write
    the bit-packed bytes, read them back through the same handle —
    bytes round-trip exactly, cards survive.
    """
    nfine_per_cov = 8 * 32  # 32 packed bytes per coverage chunk
    n_cov_chunks = 16
    data = _make_packed_buffer(nfine_per_cov, n_cov_chunks)
    tile_size = nfine_per_cov // 8  # 32

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=131072)
            f[1].write(data)
            same = f[1].read()
        np.testing.assert_array_equal(same, data)


def test_healsparse_layout_round_trip_post_reopen():
    """
    Re-open the file after writing — bytes AND header cards survive.
    """
    nfine_per_cov = 8 * 32
    n_cov_chunks = 8
    data = _make_packed_buffer(nfine_per_cov, n_cov_chunks, seed=1)
    tile_size = nfine_per_cov // 8

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=4096)
            f[1].write(data)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            hdr = f[1].header
            assert f[1].extname == "SPARSE"
            assert hdr["BITPACK"] is True
            assert hdr["SENTINEL"] is False
            assert hdr["PIXTYPE"] == "HEALSPARSE"
            assert int(hdr["NSIDE"]) == 4096
        np.testing.assert_array_equal(out, data)


# ---------------------------------------------------------------------
# Extend: append a new coverage chunk (1-D extend with partial-last-tile)
# ---------------------------------------------------------------------


def test_healsparse_extend_one_coverage_chunk():
    """
    Healsparse grows the SPARSE buffer one coverage chunk at a time
    when new pixels become valid.  The extend lands on a partial
    last tile in the general case.  Confirm the result round-trips.
    """
    nfine_per_cov = 8 * 32
    bytes_per_chunk = nfine_per_cov // 8
    tile_size = bytes_per_chunk
    initial_chunks = 5
    initial = _make_packed_buffer(nfine_per_cov, initial_chunks, seed=2)
    added = _make_packed_buffer(nfine_per_cov, 1, seed=3)

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                initial.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=8192)
            f[1].write(initial)

        # Now grow by one chunk along axis 0 — this is the workload
        # the memory note flagged as the common case (partial-last-
        # tile when bytes_per_chunk doesn't divide a fresh boundary).
        with rustfits.FITS(fname, "r+") as f:
            f[1].extend(added)

        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        expected = np.concatenate([initial, added])
        np.testing.assert_array_equal(out, expected)


def test_healsparse_extend_partial_last_tile_off_boundary():
    """
    Extend where the appended length forces a real partial-last-tile
    case: starting nbytes is not a tile-shape multiple, and the
    extension adds another non-multiple chunk.  The existing partial
    tile must be decoded + the new bytes merged + the rest of the
    extension encoded as fresh tiles.
    """
    tile_size = 64
    # Initial length: 1.5 tiles worth = 96 bytes.
    initial = np.arange(96, dtype=np.uint8) ^ 0xA5
    # Append 50 bytes (less than a full tile).
    added = (np.arange(50, dtype=np.uint8) * 7) & 0xFF

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                initial.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=2048)
            f[1].write(initial)
        with rustfits.FITS(fname, "r+") as f:
            f[1].extend(added)
        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        expected = np.concatenate([initial, added])
        assert out.shape == expected.shape
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------
# __setitem__: modify a packed slice in place
# ---------------------------------------------------------------------


def test_healsparse_setitem_slice_round_trip():
    """
    Mutate a slice of the bit-packed buffer — exercises the same
    "decode tile → modify → re-encode + append-and-orphan heap"
    machinery used for all tile-compressed image __setitem__ writes.
    """
    nfine_per_cov = 8 * 32
    bytes_per_chunk = nfine_per_cov // 8
    n_cov_chunks = 6
    data = _make_packed_buffer(nfine_per_cov, n_cov_chunks, seed=4)
    tile_size = bytes_per_chunk

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=1024)
            f[1].write(data)

        # Overwrite the middle coverage chunk + a bit on either side
        # so the write crosses tile boundaries (this is the realistic
        # "I just updated a few pixels" mutation).
        new_chunk = (
            np.arange(bytes_per_chunk + 8, dtype=np.uint8) * 13
        ) & 0xFF
        beg = 2 * bytes_per_chunk - 4
        end = beg + new_chunk.size
        with rustfits.FITS(fname, "r+") as f:
            f[1][beg:end] = new_chunk

        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
        expected = data.copy()
        expected[beg:end] = new_chunk
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------
# Custom header cards survive header edits
# ---------------------------------------------------------------------


def test_healsparse_header_cards_survive_extend_and_setitem():
    """
    The BITPACK / SENTINEL / PIXTYPE / NSIDE cards must remain in
    place after the data extents grow (extend) or the heap mutates
    (__setitem__).  rustfits doesn't auto-strip user keywords; this
    test pins that behavior so future header-rewrite changes don't
    silently drop them.
    """
    tile_size = 32
    data = _make_packed_buffer(8 * 32, 3, seed=5)

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=65536)
            f[1].write(data)
            f[1].extend(np.zeros(16, dtype=np.uint8))
            f[1][0:4] = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)

        with rustfits.FITS(fname, "r") as f:
            hdr = f[1].header
            assert hdr["BITPACK"] is True
            assert hdr["SENTINEL"] is False
            assert hdr["PIXTYPE"] == "HEALSPARSE"
            assert int(hdr["NSIDE"]) == 65536


# ---------------------------------------------------------------------
# Cross-tool: fitsio reads the file we wrote
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_fitsio(),
    reason="fitsio required for cross-tool verification",
)
def test_fitsio_cross_reads_healsparse_layout():
    """
    A real healsparse client uses fitsio.read() to load the SPARSE
    extension.  Confirm rustfits-written files are byte-equivalent
    from fitsio's perspective.
    """
    import fitsio

    nfine_per_cov = 8 * 32
    n_cov_chunks = 12
    data = _make_packed_buffer(nfine_per_cov, n_cov_chunks, seed=6)
    tile_size = nfine_per_cov // 8

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=16384)
            f[1].write(data)

        with fitsio.FITS(fname, "r") as f:
            out = f[1].read()
            hdr = f[1].read_header()
        np.testing.assert_array_equal(out, data)
        assert hdr["BITPACK"] is True
        assert hdr["PIXTYPE"].strip() == "HEALSPARSE"
        assert int(hdr["NSIDE"]) == 16384


@pytest.mark.skipif(
    not _have_fitsio(),
    reason="fitsio required for cross-tool verification",
)
def test_fitsio_cross_reads_after_extend_and_setitem():
    """
    Same as above but after a couple of mutations — fitsio still
    sees consistent bytes.
    """
    import fitsio

    tile_size = 32
    data = _make_packed_buffer(8 * 32, 4, seed=7)

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.create_image_hdu(
                "u1",
                data.shape,
                compress=rustfits.Rice1(tile_shape=(tile_size,)),
            )
            _emit_healsparse_cards(f[1], nside_sparse=4096)
            f[1].write(data)
        extra = _make_packed_buffer(8 * 32, 1, seed=8)
        with rustfits.FITS(fname, "r+") as f:
            f[1].extend(extra)
            f[1][10:14] = np.array([0x01, 0x02, 0x04, 0x08], dtype=np.uint8)

        with fitsio.FITS(fname, "r") as f:
            out = f[1].read()
        expected = np.concatenate([data, extra])
        expected[10:14] = [0x01, 0x02, 0x04, 0x08]
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------
# Cross-tool: rustfits reads a fitsio-written healsparse-shaped file
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_fitsio(),
    reason="fitsio required for cross-tool verification",
)
def test_rustfits_reads_fitsio_written_healsparse_layout():
    """
    Mirror of the above: fitsio writes the file, rustfits reads it
    back.  Covers the other direction in case a healsparse user
    starts with fitsio-written files and switches to rustfits for
    further mutations.
    """
    import fitsio

    nfine_per_cov = 8 * 32
    n_cov_chunks = 10
    data = _make_packed_buffer(nfine_per_cov, n_cov_chunks, seed=9)
    tile_size = nfine_per_cov // 8

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "hsp.fits")
        with fitsio.FITS(fname, "rw", clobber=True) as f:
            f.write(np.zeros(1, dtype="i4"))  # primary
            header = [
                {"name": "EXTNAME", "value": "SPARSE"},
                {"name": "PIXTYPE", "value": "HEALSPARSE"},
                {"name": "BITPACK", "value": True},
                {"name": "SENTINEL", "value": False},
                {"name": "NSIDE", "value": 4096},
            ]
            f.write(
                data,
                extname="SPARSE",
                header=header,
                compress="RICE_1",
                tile_dims=[tile_size],
            )

        with rustfits.FITS(fname, "r") as f:
            out = f[1].read()
            hdr = f[1].header
        np.testing.assert_array_equal(out, data)
        assert hdr["BITPACK"] is True
        assert hdr["PIXTYPE"] == "HEALSPARSE"
        assert int(hdr["NSIDE"]) == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
