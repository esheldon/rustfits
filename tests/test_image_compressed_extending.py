"""
Batched compressed-image extend context manager.

``with hdu.extending():`` buffers ``extend()`` inputs in RAM and
flushes once on ``__exit__``, collapsing N partial-last-tile
re-encodes into 1.  See CLAUDE.md Performance TODO #12 and
``hdu_image_compressed/extending.rs`` for the design.

Tests cover:
    - Round-trip equivalence (buffered N small extends == one big
      extend).
    - Auto-flush on normal __exit__.
    - Auto-flush on exception inside the context.
    - Rejection of read / __getitem__ / write / __setitem__ /
      repack / add_checksum / verify_checksum while a context is
      open.
    - Rejection of nested extending() (one HDU, two contexts).
    - Rejection of FITS.close() while a context is open.
    - Rejection of `with FITS` __exit__ when a context is still
      open (the same close() check fires).
    - Cross-tool: astropy reads the result bit-exact.
    - All major algorithms (Gzip1, Gzip2, Rice1) round-trip
      identically through the context.
    - Streaming-create pattern: create_image_hdu(shape=(0, ...))
      + with extending(): + extend() repeatedly.
    - Mixed input dtypes (BITPIX-native, unsigned-int trick) are
      buffered + concatenated correctly.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits

astropy_fits = pytest.importorskip("astropy.io.fits")


def _seq(shape, dtype, start=0):
    """Deterministic integer ndarray for round-trip checks."""
    np_dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    info = np.iinfo(np_dtype)
    maxv = min(info.max, 1_000_000)
    return (
        ((np.arange(n) + start) % (maxv + 1)).astype(np_dtype).reshape(shape)
    )


# -------- round-trip equivalence --------


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_buffered_equals_single_extend_1d(AlgoCls):
    """
    Buffered N small extends produces a file equivalent (read-back)
    to one big extend with the concatenated data.
    """
    data = _seq((97,), "i4")
    with tempfile.TemporaryDirectory() as tmp:
        # Buffered: 13 chunks averaging ~7 rows each
        fn_buf = os.path.join(tmp, "buf.fits.fz")
        with rustfits.FITS(fn_buf, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=AlgoCls(tile_shape=(16,)),
            )
            hdu = f[1]
            with hdu.extending():
                cuts = np.array_split(data, 13)
                for c in cuts:
                    hdu.extend(c)

        # Reference: one big extend
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        with rustfits.FITS(fn_ref, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=AlgoCls(tile_shape=(16,)),
            )
            hdu = f[1]
            hdu.extend(data)

        # Same-handle + post-reopen reads of both files match data
        with rustfits.FITS(fn_buf, "r") as f:
            np.testing.assert_equal(f[1].read(), data)
        with rustfits.FITS(fn_ref, "r") as f:
            np.testing.assert_equal(f[1].read(), data)


@pytest.mark.parametrize(
    "AlgoCls", [rustfits.Gzip1, rustfits.Gzip2, rustfits.Rice1]
)
def test_buffered_equals_single_extend_2d(AlgoCls):
    """2-D mosaic pattern: 20 batches of 5 rows into a tile of 100."""
    data = _seq((100, 50), "i4")
    chunks = np.split(data, 20)  # 20 chunks of 5 rows each
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0, 50),
                compress=AlgoCls(tile_shape=(100, 50)),
            )
            hdu = f[1]
            with hdu.extending():
                for c in chunks:
                    hdu.extend(c)
            np.testing.assert_equal(hdu.read(), data)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_equal(f[1].read(), data)


# -------- auto-flush semantics --------


def test_flush_on_exception_inside_context():
    """
    An exception raised inside the `with` block still triggers
    __exit__ → drain → file contains the buffered rows.
    """
    data = _seq((24,), "i4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(8,)),
            )
            hdu = f[1]
            with pytest.raises(RuntimeError, match="boom"):
                with hdu.extending():
                    hdu.extend(data[:16])
                    raise RuntimeError("boom")
            # buffer was flushed before the raise propagated;
            # only the rows extended before the raise are on disk.
            np.testing.assert_equal(hdu.read(), data[:16])


def test_empty_context_is_noop():
    """
    Entering + exiting without calling extend() leaves the file
    unchanged.
    """
    data = _seq((32,), "i4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(8,)),
            )
            hdu = f[1]
            hdu.extend(data)
            with hdu.extending():
                pass  # no extend calls
            np.testing.assert_equal(hdu.read(), data)


# -------- rejection paths --------


def _open_with_context():
    """Helper: returns (fits, hdu, ctx_obj). ctx is __enter__'d."""
    tmpdir = tempfile.TemporaryDirectory()
    fn = os.path.join(tmpdir.name, "t.fits.fz")
    f = rustfits.FITS(fn, "w+")
    f.create_image_hdu(
        "i4",
        (0,),
        compress=rustfits.Gzip1(tile_shape=(16,)),
    )
    hdu = f[1]
    ctx = hdu.extending()
    ctx.__enter__()
    return f, hdu, ctx, tmpdir  # tmpdir kept alive by caller


def test_reject_read_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((8,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu.read()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_getitem_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((8,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu[0:4]
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_write_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        # First put a tile's worth of data so write() target is valid
        hdu.extend(_seq((16,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu.write(_seq((16,), "i4"))
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_setitem_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((16,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu[0] = 999
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_repack_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((8,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu.repack()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_add_checksum_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((8,), "i4"))
        with pytest.raises(ValueError, match="extending"):
            hdu.add_checksum()
        with pytest.raises(ValueError, match="extending"):
            hdu.add_datasum()
        with pytest.raises(ValueError, match="extending"):
            hdu.verify_checksum()
        with pytest.raises(ValueError, match="extending"):
            hdu.verify_datasum()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_nested_context():
    """Opening a second extending() context on the same HDU raises."""
    f, hdu, ctx, _td = _open_with_context()
    try:
        with pytest.raises(ValueError, match="already inside"):
            hdu.extending().__enter__()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_close_while_context_open():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.extend(_seq((8,), "i4"))
        with pytest.raises(
            ValueError,
            match="cannot close FITS while HDU at index .* extending",
        ):
            f.close()
        # Verify f is still open and usable after the rejection.
        # First close the context so subsequent ops can run.
        ctx.__exit__(None, None, None)
        # Now we can read, prove the file survived.
        assert hdu.read().shape == (8,)
    finally:
        f.close()


def test_reject_fits_with_exit_while_context_open():
    """
    The same close() check fires from `with FITS` __exit__ when
    a nested context is still open.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        ctx_obj = None
        with pytest.raises(
            ValueError,
            match="cannot close FITS while HDU at index .* extending",
        ):
            with rustfits.FITS(fn, "w+") as f:
                f.create_image_hdu(
                    "i4",
                    (0,),
                    compress=rustfits.Gzip1(tile_shape=(8,)),
                )
                hdu = f[1]
                ctx_obj = hdu.extending()
                ctx_obj.__enter__()
                hdu.extend(_seq((8,), "i4"))
                # forgot ctx_obj.__exit__(...)
                # outer `with FITS` __exit__ should raise
        # Recover so the temp dir cleans up cleanly.
        if ctx_obj is not None:
            try:
                ctx_obj.__exit__(None, None, None)
            except Exception:
                pass


# -------- input validation --------


def test_reject_empty_data_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        empty = np.zeros((0,), dtype="i4")
        with pytest.raises(ValueError, match="data.shape\\[0\\]"):
            hdu.extend(empty)
    finally:
        ctx.__exit__(None, None, None)
        f.close()


# -------- cross-tool --------


def test_astropy_can_read_buffered_extends():
    """
    astropy reads back the bit-exact image that buffered extends
    produced.
    """
    data = _seq((48,), "i4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(8,)),
            )
            hdu = f[1]
            with hdu.extending():
                for chunk in np.array_split(data, 6):
                    hdu.extend(chunk)
        with astropy_fits.open(fn) as hdus:
            np.testing.assert_equal(hdus[1].data, data)


# -------- mid-context drain (RAM cap) --------


def test_mid_context_drain_keeps_data_correct():
    """
    Push enough data through ``extend()`` calls that the 32 MB cap
    should trigger at least one mid-context drain.  The final
    image must still round-trip bit-exact.

    Image: 50000 rows x 200 cols i4 = 40 MB raw, comfortably over
    the 32 MB cap.  Chunks of 50 rows = 40 KB each, so ~800
    chunks; drains fire when the buffer crosses ~32 MB.  Uses
    arange data + i4 dtype for fast generation (the cap logic is
    indifferent to dtype).
    """
    rows, cols = 50_000, 200
    chunk_rows = 50
    full = (np.arange(rows * cols, dtype="i4") % 1_000_000).reshape(rows, cols)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0, cols),
                compress=rustfits.Gzip1(tile_shape=(100, cols)),
            )
            hdu = f[1]
            with hdu.extending():
                for r0 in range(0, rows, chunk_rows):
                    hdu.extend(full[r0 : r0 + chunk_rows])
            np.testing.assert_array_equal(hdu.read(), full)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f[1].read(), full)


def test_mid_context_drain_aligned_chunks_streaming():
    """
    Tile-aligned chunks: each mid-context drain ends at exactly
    NAXIS0 == k * tile_rows, so the final residual drain has
    nothing to do (no partial-tile re-encode).  This covers the
    "tile-aligned input + cap-triggered drain" interaction.
    """
    rows, cols = 50_000, 200
    tile_rows = 100
    full = (np.arange(rows * cols, dtype="i4") % 1_000_000).reshape(rows, cols)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0, cols),
                compress=rustfits.Gzip1(tile_shape=(tile_rows, cols)),
            )
            hdu = f[1]
            with hdu.extending():
                for r0 in range(0, rows, tile_rows):
                    hdu.extend(full[r0 : r0 + tile_rows])
            np.testing.assert_array_equal(hdu.read(), full)


# -------- unsigned-int trick survives buffering --------


def test_unsigned_trick_buffered():
    """
    A u2 HDU (stored as i2+BZERO=32768) accepts buffered u2 input
    inside the context.  Each buffered chunk stays as u2 until the
    drain step, where np.concatenate produces a single u2 array
    that the existing extend code's reverse-transform handles.
    """
    data = (
        np.arange(64, dtype="u2") + 30000  # forces top bit
    ).astype("u2")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "u2",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
            )
            hdu = f[1]
            with hdu.extending():
                for chunk in np.array_split(data, 8):
                    hdu.extend(chunk)
            np.testing.assert_equal(hdu.read(), data)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
