"""
No-op `extending()` / `appending()` context on uncompressed
HDUs.

`ImageHDU.extending()`, `TableHDU.appending()`, and
`TableHDU.extending()` (alias) all return a no-op context
manager.  This exists for API symmetry with the compressed
subclasses (where the context does real buffered-batched-extend
work) so generic code that iterates HDUs of mixed types can
use the pattern uniformly::

    for hdu in fits:
        with hdu.extending():
            for batch in batches:
                hdu.extend(batch)

The no-op contract:
    - __enter__ returns the HDU (so `with hdu.extending() as h`
      yields the same object).
    - __exit__ returns False (don't suppress exceptions).
    - No file state changes from entering / exiting alone.
    - No restrictions on other operations inside the block
      (read, getitem, etc. all work as normal).

Tests also pin that the COMPRESSED subclasses' real contexts
still override the parent's no-op via Python MRO (so a generic
loop that always uses `with hdu.extending():` gets the buffered
behavior on compressed HDUs and the no-op on uncompressed).
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ----- ImageHDU.extending() no-op -----


def test_image_extending_returns_same_hdu():
    """`__enter__` yields the same HDU object (for `with ... as h`)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (0, 10))
            hdu = f[0]
            with hdu.extending() as h:
                assert h is hdu


def test_image_extending_is_noop_round_trip():
    """
    Several extend calls inside extending() produce the same file as
    the same extends outside.  The context is just transparent.
    """
    chunks = [np.full((5, 10), v, dtype="i4") for v in range(4)]
    expected = np.concatenate(chunks, axis=0)
    with tempfile.TemporaryDirectory() as tmp:
        # Inside context
        fn1 = os.path.join(tmp, "ctx.fits")
        with rustfits.FITS(fn1, "w+") as f:
            f.create_image_hdu("i4", (0, 10))
            hdu = f[0]
            with hdu.extending():
                for c in chunks:
                    hdu.extend(c)
            np.testing.assert_array_equal(hdu.read(), expected)
        # Outside context
        fn2 = os.path.join(tmp, "noctx.fits")
        with rustfits.FITS(fn2, "w+") as f:
            f.create_image_hdu("i4", (0, 10))
            hdu = f[0]
            for c in chunks:
                hdu.extend(c)
            np.testing.assert_array_equal(hdu.read(), expected)


def test_image_extending_allows_read_inside():
    """
    Unlike compressed extending(), the no-op imposes no restrictions
    on operations inside the block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (0, 5))
            hdu = f[0]
            with hdu.extending():
                hdu.extend(np.arange(15, dtype="i4").reshape(3, 5))
                # Read inside the context — fine for uncompressed.
                np.testing.assert_array_equal(
                    hdu.read(), np.arange(15, dtype="i4").reshape(3, 5)
                )


def test_image_extending_exception_propagates():
    """`__exit__` returns False, so in-flight exceptions propagate."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (0, 5))
            hdu = f[0]
            with pytest.raises(RuntimeError, match="boom"):
                with hdu.extending():
                    hdu.extend(np.zeros((2, 5), dtype="i4"))
                    raise RuntimeError("boom")
            # Pre-raise extend stays on disk (no rollback semantics).
            assert hdu.shape == (2, 5)


# ----- TableHDU.appending() / extending() no-op -----


def test_table_appending_returns_same_hdu():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))  # primary placeholder
            f.create_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            with tbl.appending() as h:
                assert h is tbl


def test_table_extending_alias_returns_same_hdu():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            with tbl.extending() as h:
                assert h is tbl


def test_table_appending_is_noop_round_trip():
    dt = np.dtype([("x", "f4"), ("y", "i4")])
    chunks = []
    for v in range(5):
        a = np.zeros(10, dtype=dt)
        a["x"] = v
        a["y"] = v * 100
        chunks.append(a)
    expected = np.concatenate(chunks)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_table_hdu(dt, nrows=0)
            tbl = f[1]
            with tbl.appending():
                for c in chunks:
                    tbl.append(c)
            np.testing.assert_array_equal(tbl.read(), expected)


def test_table_extending_alias_is_same_machinery():
    """extending() and appending() yield the same context type."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            c1 = tbl.appending()
            c2 = tbl.extending()
            assert type(c1) is type(c2)


# ----- AsciiTableHDU.appending() / extending() no-op -----


def test_ascii_table_appending_returns_same_hdu():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))  # primary placeholder
            f.create_ascii_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            with tbl.appending() as h:
                assert h is tbl


def test_ascii_table_extending_alias_returns_same_hdu():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_ascii_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            with tbl.extending() as h:
                assert h is tbl


def test_ascii_table_appending_is_noop_round_trip():
    dt = np.dtype([("x", "f4"), ("y", "i4")])
    chunks = []
    for v in range(5):
        a = np.zeros(10, dtype=dt)
        a["x"] = v
        a["y"] = v * 100
        chunks.append(a)
    expected = np.concatenate(chunks)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_ascii_table_hdu(dt, nrows=0)
            tbl = f[1]
            with tbl.appending():
                for c in chunks:
                    tbl.append(c)
            got = tbl.read()
            np.testing.assert_array_equal(got["x"], expected["x"])
            np.testing.assert_array_equal(got["y"], expected["y"])


def test_ascii_table_extending_alias_is_same_machinery():
    """extending() and appending() yield the same context type."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (10,))
            f.create_ascii_table_hdu(np.dtype([("x", "f4")]), nrows=0)
            tbl = f[1]
            c1 = tbl.appending()
            c2 = tbl.extending()
            assert type(c1) is type(c2)


# ----- Generic loop pattern (the actual point of the no-op) -----


def test_generic_loop_works_across_hdu_types():
    """
    The headline use case: `with hdu.extending():` works on every
    HDU type — no-op on uncompressed, real buffering on compressed.
    """
    dt = np.dtype([("x", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            # Mix of all four data-bearing HDU types.
            f.create_image_hdu("i4", (0,))  # uncompressed primary image
            f.create_image_hdu(
                "i4",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
            )  # compressed image
            f.create_table_hdu(dt, nrows=0)  # uncompressed table
            f.create_table_hdu(
                dt,
                nrows=0,
                compress=True,
                ztilelen=50,
            )  # compressed table

            for hdu in f:
                with hdu.extending():
                    for _ in range(3):
                        if isinstance(hdu, rustfits.ImageHDU):
                            hdu.extend(np.arange(10, dtype="i4"))
                        else:
                            hdu.extend(np.zeros(10, dtype=dt))

            for i in range(len(f)):
                assert len(f[i]) == 30, (
                    f"HDU {i} ({type(f[i]).__name__}) has len={len(f[i])}"
                )


# ----- Compressed subclass overrides (sanity) -----


def test_compressed_image_override_still_routes_through_buffer():
    """
    CompressedImageHDU.extending() is the real context (not the
    no-op).  We assert that by trying a forbidden operation inside
    — the no-op would allow it; the compressed override rejects it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu(
                "i4",
                (0,),
                compress=rustfits.Gzip1(tile_shape=(16,)),
            )
            hdu = f[1]
            with hdu.extending():
                hdu.extend(np.arange(8, dtype="i4"))
                # Compressed override rejects read inside;
                # the no-op would have allowed it.
                with pytest.raises(ValueError, match="extending"):
                    hdu.read()


def test_compressed_table_override_still_routes_through_buffer():
    """
    Same check, table side: appending() override rejects mid-context
    read.
    """
    dt = np.dtype([("x", "f4")])
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(
                dt,
                nrows=0,
                compress=True,
                ztilelen=50,
            )
            hdu = f[1]
            with hdu.appending():
                hdu.append(np.zeros(10, dtype=dt))
                with pytest.raises(ValueError, match="appending"):
                    hdu.read()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
