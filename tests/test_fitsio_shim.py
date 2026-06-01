"""
Tests for the fitsio-style shim at ``rustfits.fitsio``.

The shim is intentionally narrow — it translates the constructor
(fitsio's ``mode='rw'`` + ``clobber=True``) and forwards everything
else.  These tests pin:

* the four mode / clobber combinations land on the right rustfits
  open mode,
* the fitsio synonyms ``'READONLY'`` / ``'READWRITE'`` / ``0`` / ``1``
  work,
* unknown modes raise with a clear message,
* the ``compress=`` string passthrough produces a real compressed
  HDU (rustfits already accepts strings; this is a regression pin),
* indexing / iteration / ``len`` / context manager all forward,
* :func:`fitsio.read` / :func:`fitsio.read_header` / :func:`fitsio.write`
  are the same callables as rustfits's convenience surface.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits
from rustfits import fitsio


# ----- mode + clobber translation -----


def test_mode_r_opens_existing_readonly():
    """``mode='r'`` opens an existing file read-only."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (5,))
            f[0].write(np.arange(5, dtype="i4"))
        with fitsio.FITS(fn, "r") as f:
            np.testing.assert_array_equal(
                f[0].read(), np.arange(5, dtype="i4")
            )
            with pytest.raises(IOError):
                f[0].write(np.zeros(5, dtype="i4"))


def test_mode_rw_opens_existing_read_write():
    """``mode='rw'`` on an existing file opens for r+ (no truncate)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (5,))
            f[0].write(np.arange(5, dtype="i4"))
        with fitsio.FITS(fn, "rw") as f:
            np.testing.assert_array_equal(
                f[0].read(), np.arange(5, dtype="i4")
            )
            f[0].write(np.full(5, 42, dtype="i4"))
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(
                f[0].read(), np.full(5, 42, dtype="i4")
            )


def test_mode_rw_creates_nonexistent():
    """
    ``mode='rw'`` on a nonexistent file creates it.

    This is the case where fitsio differs from rustfits's native
    ``'r+'`` (which requires the file to exist); the shim picks
    ``'w+'`` when the path doesn't exist yet.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "fresh.fits")
        assert not os.path.exists(fn)
        with fitsio.FITS(fn, "rw") as f:
            f.write_image(np.arange(7, dtype="i4"), extname="sci")
        assert os.path.exists(fn)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(
                f["sci"].read(), np.arange(7, dtype="i4")
            )


def test_mode_rw_clobber_true_truncates_existing():
    """``clobber=True`` discards an existing file's contents."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.create_image_hdu("i4", (5,))
            f[0].write(np.full(5, 99, dtype="i4"))
        with fitsio.FITS(fn, "rw", clobber=True) as f:
            f.write_image(np.arange(3, dtype="i4"), extname="sci")
        with rustfits.FITS(fn, "r") as f:
            assert len(f) == 1
            np.testing.assert_array_equal(
                f["sci"].read(), np.arange(3, dtype="i4")
            )


def test_mode_rw_clobber_false_preserves_existing():
    """``clobber=False`` (default) on an existing file does NOT truncate."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.write_image(np.arange(10, dtype="i4"), extname="sci")
        with fitsio.FITS(fn, "rw") as f:
            np.testing.assert_array_equal(
                f["sci"].read(), np.arange(10, dtype="i4")
            )


@pytest.mark.parametrize("alias", ["READONLY", 0])
def test_read_mode_aliases(alias):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.write_image(np.arange(4, dtype="i4"), extname="sci")
        with fitsio.FITS(fn, alias) as f:
            np.testing.assert_array_equal(
                f["sci"].read(), np.arange(4, dtype="i4")
            )


@pytest.mark.parametrize("alias", ["READWRITE", 1])
def test_rw_mode_aliases(alias):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "fresh.fits")
        with fitsio.FITS(fn, alias) as f:
            f.write_image(np.arange(3, dtype="i4"), extname="sci")
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(
                f["sci"].read(), np.arange(3, dtype="i4")
            )


def test_unknown_mode_raises_with_clear_message():
    """An unsupported mode points the user at rustfits.FITS directly."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with pytest.raises(ValueError, match="unsupported mode"):
            fitsio.FITS(fn, "w+")


# ----- compress= string passthrough -----


def test_compress_string_passthrough():
    """
    ``compress='RICE_1'`` (and other algorithm names) work without
    translation in the shim — rustfits's FITS.write_image accepts
    the strings natively.
    """
    data = (np.arange(16 * 16, dtype="i4").reshape(16, 16) % 17).astype("i4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with fitsio.FITS(fn, "rw", clobber=True) as f:
            f.write_image(data, extname="sci", compress="RICE_1")
        with rustfits.FITS(fn, "r") as f:
            hdu = f["sci"]
            assert isinstance(hdu, rustfits.CompressedImageHDU)
            np.testing.assert_array_equal(hdu.read(), data)


@pytest.mark.parametrize(
    "algo", ["GZIP_1", "GZIP_2", "RICE_1", "HCOMPRESS_1", "PLIO_1"]
)
def test_compress_string_all_algorithms(algo):
    """Every cfitsio algorithm string flows through unmodified."""
    if algo == "PLIO_1":
        data = np.zeros((16, 16), dtype="i4")
        data[2:5, 3:7] = 1
    else:
        data = (np.arange(16 * 16, dtype="i4").reshape(16, 16) % 17).astype(
            "i4"
        )
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits.fz")
        with fitsio.FITS(fn, "rw", clobber=True) as f:
            f.write_image(data, extname="sci", compress=algo)
        with rustfits.FITS(fn, "r") as f:
            np.testing.assert_array_equal(f["sci"].read(), data)


# ----- forwarding (indexing / iter / len / context manager) -----


def test_forwarding_indexing_and_iter():
    """``fits[i]`` / ``fits['EXTNAME']`` / iteration all forward."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.write_image(np.arange(5, dtype="i4"), extname="a")
            f.write_image(np.arange(7, dtype="i4"), extname="b")
        with fitsio.FITS(fn, "r") as f:
            assert len(f) == 2
            assert f[0].extname == "a"
            assert f["b"].extname == "b"
            names = [hdu.extname for hdu in f if hdu.extname is not None]
            assert names == ["a", "b"]


def test_close_method_works():
    """``fits.close()`` outside a context manager forwards."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.write_image(np.arange(5, dtype="i4"), extname="sci")
        f = fitsio.FITS(fn, "r")
        np.testing.assert_array_equal(
            f["sci"].read(), np.arange(5, dtype="i4")
        )
        f.close()


def test_repr_mentions_shim():
    """``repr`` makes clear the object is the shim, not native FITS."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fn, "w+") as f:
            f.write_image(np.arange(3, dtype="i4"), extname="sci")
        with fitsio.FITS(fn, "r") as f:
            assert "rustfits.fitsio.FITS" in repr(f)


# ----- top-level convenience re-exports -----


def test_top_level_read_is_rustfits_read():
    """``fitsio.read`` and ``rustfits.read`` are the same callable."""
    assert fitsio.read is rustfits.read
    assert fitsio.read_header is rustfits.read_header
    assert fitsio.write is rustfits.write


def test_top_level_read_round_trip():
    """End-to-end: read via the shim's re-export."""
    data = np.arange(20, dtype="f4")
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.fits")
        fitsio.write(fn, data, extname="sci")
        got = fitsio.read(fn, ext="sci")
        np.testing.assert_array_equal(got, data)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
