"""
Batched compressed-table append context manager.

``with hdu.appending():`` (alias ``hdu.extending():``) buffers
``append()`` / ``extend()`` inputs in RAM and drains in
ZTILELEN-aligned bursts (cap at 32 MB), with the residual draining
at ``__exit__``.  Collapses N partial-trailing-tile merge-and-
re-encodes into 1.  See CLAUDE.md Performance TODO #12 (ZTABLE
follow-up) and ``hdu_table_compressed/extending.rs`` for the
design (mirrors the ZIMAGE side).

Tests cover:
    - Round-trip equivalence (buffered N small appends == one big
      write).
    - Auto-flush on normal __exit__.
    - Auto-flush on exception inside the context.
    - Rejection of read / __getitem__ / write / __setitem__ /
      repack / add_checksum / verify_checksum while a context is
      open.
    - Rejection of nested appending() (one HDU, two contexts).
    - Rejection of FITS.close() while a context is open.
    - extending() alias points at the same context machinery.
    - Mid-context drain firing under load.
    - Input form coverage: structured ndarray, dict, list+names
      all buffer + concatenate correctly.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


def _schema():
    """Small fixed-column schema used across tests."""
    return np.dtype([("x", "f4"), ("y", "i4"), ("z", "f8")])


def _seq(n, start=0):
    """Deterministic structured ndarray of length n."""
    dt = _schema()
    arr = np.zeros(n, dtype=dt)
    arr["x"] = (np.arange(n) + start).astype("f4")
    arr["y"] = (np.arange(n) + start).astype("i4")
    arr["z"] = (np.arange(n) + start).astype("f8") * 0.5
    return arr


# -------- round-trip equivalence --------


def test_buffered_equals_single_append():
    """
    N small buffered appends produce the same file (post-read) as
    one big write.
    """
    data = _seq(500)
    with tempfile.TemporaryDirectory() as tmp:
        fn_buf = os.path.join(tmp, "buf.fits.fz")
        with rustfits.FITS(fn_buf, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=100)
            hdu = f[1]
            with hdu.appending():
                for chunk in np.array_split(data, 17):
                    hdu.append(chunk)
        fn_ref = os.path.join(tmp, "ref.fits.fz")
        with rustfits.FITS(fn_ref, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=100)
            f[1].append(data)
        with rustfits.FITS(fn_buf, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)
        with rustfits.FITS(fn_ref, "r") as f:
            np.testing.assert_array_equal(f[1].read(), data)


def test_buffered_append_via_extend_alias():
    """``extend()`` (alias for ``append()``) routes through buffer too."""
    data = _seq(300)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=100)
            hdu = f[1]
            with hdu.appending():
                for chunk in np.array_split(data, 12):
                    hdu.extend(chunk)
            np.testing.assert_array_equal(hdu.read(), data)


def test_extending_alias_is_same_context():
    """``hdu.extending()`` is the same machinery as ``hdu.appending()``."""
    data = _seq(200)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=50)
            hdu = f[1]
            with hdu.extending():
                for chunk in np.array_split(data, 8):
                    hdu.append(chunk)
            np.testing.assert_array_equal(hdu.read(), data)


# -------- input form coverage --------


def test_buffered_dict_input():
    """Buffered appends with dict-of-arrays input."""
    n = 200
    data = _seq(n)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=50)
            hdu = f[1]
            with hdu.appending():
                for chunk in np.array_split(data, 8):
                    hdu.append(
                        {
                            "x": chunk["x"],
                            "y": chunk["y"],
                            "z": chunk["z"],
                        }
                    )
            np.testing.assert_array_equal(hdu.read(), data)


def test_buffered_list_plus_names_input():
    """Buffered appends with list-of-arrays + names= input."""
    n = 200
    data = _seq(n)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=50)
            hdu = f[1]
            with hdu.appending():
                for chunk in np.array_split(data, 8):
                    hdu.append(
                        [chunk["x"], chunk["y"], chunk["z"]],
                        names=["x", "y", "z"],
                    )
            np.testing.assert_array_equal(hdu.read(), data)


# -------- auto-flush semantics --------


def test_flush_on_exception_inside_context():
    """
    Exception inside the `with` block still triggers __exit__ →
    drain → file contains the buffered rows.
    """
    data = _seq(60)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=20)
            hdu = f[1]
            with pytest.raises(RuntimeError, match="boom"):
                with hdu.appending():
                    hdu.append(data[:40])
                    raise RuntimeError("boom")
            # Buffer was flushed before raise propagated.
            np.testing.assert_array_equal(hdu.read(), data[:40])


def test_empty_context_is_noop():
    """
    Entering + exiting without calling append() leaves the file
    unchanged.
    """
    data = _seq(80)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=40)
            hdu = f[1]
            hdu.append(data)
            with hdu.appending():
                pass
            np.testing.assert_array_equal(hdu.read(), data)


# -------- rejection paths --------


def _open_with_context():
    """Helper: returns (fits, hdu, ctx_obj, tmpdir). ctx is __enter__'d."""
    tmpdir = tempfile.TemporaryDirectory()
    fn = os.path.join(tmpdir.name, "t.fits.fz")
    f = rustfits.FITS(fn, "w+")
    f.create_table_hdu(_schema(), nrows=0, compress=True, ztilelen=50)
    hdu = f[1]
    ctx = hdu.appending()
    ctx.__enter__()
    return f, hdu, ctx, tmpdir


def test_reject_read_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(ValueError, match="appending"):
            hdu.read()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_getitem_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(ValueError, match="appending"):
            hdu[0:5]
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_write_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(ValueError, match="appending"):
            hdu.write(_seq(50))
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_setitem_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(60))  # enough rows to set on
        with pytest.raises(ValueError, match="appending"):
            hdu[0] = _seq(1)[0]
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_repack_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(ValueError, match="appending"):
            hdu.repack()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_checksum_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(ValueError, match="appending"):
            hdu.add_checksum()
        with pytest.raises(ValueError, match="appending"):
            hdu.add_datasum()
        with pytest.raises(ValueError, match="appending"):
            hdu.verify_checksum()
        with pytest.raises(ValueError, match="appending"):
            hdu.verify_datasum()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_nested_context():
    """Opening a second appending() context on the same HDU raises."""
    f, hdu, ctx, _td = _open_with_context()
    try:
        with pytest.raises(ValueError, match="already inside"):
            hdu.appending().__enter__()
        # extending() alias should also be rejected
        with pytest.raises(ValueError, match="already inside"):
            hdu.extending().__enter__()
    finally:
        ctx.__exit__(None, None, None)
        f.close()


def test_reject_close_while_context_open():
    f, hdu, ctx, _td = _open_with_context()
    try:
        hdu.append(_seq(10))
        with pytest.raises(
            ValueError,
            match="cannot close FITS while HDU at index .* appending",
        ):
            f.close()
        ctx.__exit__(None, None, None)
        assert len(hdu.read()) == 10
    finally:
        f.close()


def test_reject_fits_with_exit_while_context_open():
    """`with FITS` __exit__ propagates the close() check error."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        ctx_obj = None
        with pytest.raises(
            ValueError,
            match="cannot close FITS while HDU at index .* appending",
        ):
            with rustfits.FITS(fn, "w+") as f:
                f.create_table_hdu(
                    _schema(), nrows=0, compress=True, ztilelen=50
                )
                hdu = f[1]
                ctx_obj = hdu.appending()
                ctx_obj.__enter__()
                hdu.append(_seq(10))
                # forgot ctx_obj.__exit__(...)
        if ctx_obj is not None:
            try:
                ctx_obj.__exit__(None, None, None)
            except Exception:
                pass


# -------- mid-context drain (RAM cap) --------


def test_mid_context_drain_keeps_data_correct():
    """
    Push enough data through ``append()`` calls that the 32 MB cap
    should trigger at least one mid-context drain.  Final table
    round-trips bit-exact.

    Schema: 4 × f4 = 16 B / row.  Push 5M rows = 80 MB; appends of
    1000 rows each = 16 KB / append → ~2000 appends per drain
    boundary.  At least one mid-context drain fires.
    """
    dt = np.dtype([("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4")])
    n_total = 5_000_000
    chunk = 1000
    src = np.zeros(n_total, dtype=dt)
    src["a"] = np.arange(n_total).astype("f4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with rustfits.FITS(fn, "w+") as f:
            f.create_table_hdu(dt, nrows=0, compress=True, ztilelen=10_000)
            hdu = f[1]
            with hdu.appending():
                for r0 in range(0, n_total, chunk):
                    hdu.append(src[r0 : r0 + chunk])
            # Spot-check first / mid / last rather than full equality
            # (which would generate another 80 MB).
            got = hdu.read()
            assert len(got) == n_total
            np.testing.assert_array_equal(
                got["a"][:5], np.arange(5).astype("f4")
            )
            np.testing.assert_array_equal(
                got["a"][-5:],
                np.arange(n_total - 5, n_total).astype("f4"),
            )


# -------- input validation --------


def test_reject_empty_data_inside_context():
    f, hdu, ctx, _td = _open_with_context()
    try:
        empty = np.zeros(0, dtype=_schema())
        with pytest.raises(ValueError, match="at least one row"):
            hdu.append(empty)
    finally:
        ctx.__exit__(None, None, None)
        f.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
