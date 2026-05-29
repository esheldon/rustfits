"""
Read-only whole-file gzip support: opening a `.gz` path gunzips the
whole file into an in-memory buffer (Storage::Mem) and parses it like
any other file.

These exercise the `.gz` detection + gunzip-on-open path.  Fixtures
are produced with Python's gzip module (compressing a plain .fits we
wrote first), so the test asserts the gz read matches the plain read.
"""

import gzip
import os
import tempfile
import numpy as np
import pytest
import rustfits


def _gzip_file(plain_path, gz_path):
    """
    Gzip an existing file byte-for-byte to gz_path."""
    with open(plain_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())


def _write_plain_image(path, data):
    with rustfits.FITS(path, "w+") as f:
        f.write_image(data)


def test_gz_image_roundtrip():
    """
    An image in a gzipped file reads back identically to the plain
    file."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "x.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:  # default mode 'r'
            np.testing.assert_array_equal(f[0].read(), data)


def test_gz_table_roundtrip():
    """
    A binary table in a gzipped file round-trips."""
    rec = np.array(
        [(1, 1.5), (2, 2.5), (3, 3.5)],
        dtype=[("id", "i4"), ("x", "f8")],
    )
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "t.fits")
        gz = plain + ".gz"
        with rustfits.FITS(plain, "w+") as f:
            f.write_table(rec)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            got = f[1].read()
        np.testing.assert_array_equal(got["id"], rec["id"])
        np.testing.assert_array_equal(got["x"], rec["x"])


def test_gz_multi_hdu():
    """
    Multiple HDUs in a gzipped file are all parsed in order."""
    img = np.arange(12, dtype="i4").reshape(3, 4)
    rec = np.array([(10,), (20,)], dtype=[("n", "i8")])
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "m.fits")
        gz = plain + ".gz"
        with rustfits.FITS(plain, "w+") as f:
            f.write_image(img)
            f.write_table(rec)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            assert len(f.hdus) == 2
            np.testing.assert_array_equal(f[0].read(), img)
            np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_gz_tile_compressed_image():
    """
    A tile-compressed image inside a gzipped file (double
    compression) reads correctly."""
    data = np.arange(8 * 8, dtype="i4").reshape(8, 8)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "c.fits")
        gz = plain + ".gz"
        with rustfits.FITS(plain, "w+") as f:
            f.write_image(data, compress=rustfits.Rice1(tile_shape=(4, 4)))
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            hdu = f[1]
            assert isinstance(hdu, rustfits.ImageHDU)
            np.testing.assert_array_equal(hdu.read(), data)


def test_gz_slicing():
    """
    Slicing a gz-opened image works (it's a normal Mem-backed file
    once gunzipped)."""
    data = np.arange(10 * 10, dtype="i4").reshape(10, 10)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "s.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            np.testing.assert_array_equal(f[0][2:5, 1:4], data[2:5, 1:4])
            assert f[0][7, 8] == data[7, 8]


def test_gz_to_bytes_is_uncompressed():
    """
    to_bytes() on a gz-opened file returns the DEcompressed bytes —
    byte-identical to the plain file."""
    data = np.arange(20, dtype="i2").reshape(4, 5)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "b.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        with open(plain, "rb") as fh:
            raw = fh.read()

        with rustfits.FITS(gz) as f:
            assert f.to_bytes() == raw


def test_gz_convenience_read():
    """
    The top-level rustfits.read / read_header auto-handle .gz (they
    call FITS under the hood)."""
    data = np.arange(6, dtype="i4").reshape(2, 3)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "r.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        np.testing.assert_array_equal(rustfits.read(gz), data)
        hdr = rustfits.read_header(gz)
        assert hdr["NAXIS"] == 2


def test_gz_case_insensitive_extension():
    """
    Detection is case-insensitive — a .GZ extension is also gunzipped."""
    data = np.arange(4, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "u.fits")
        gz = plain + ".GZ"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_gz_reads_fitsio_written_file():
    """
    A .gz produced from a fitsio-written file reads correctly (cross
    tool)."""
    import fitsio

    data = np.arange(5 * 4, dtype="f4").reshape(5, 4)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "fio.fits")
        gz = plain + ".gz"
        with fitsio.FITS(plain, "rw", clobber=True) as ff:
            ff.write(data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz) as f:
            for hdu in f.hdus:
                if hdu.has_data:
                    np.testing.assert_array_equal(hdu.read(), data)
                    break
            else:
                raise AssertionError("no HDU with data found")


def test_gz_rplus_rejected():
    """
    Opening a .gz with mode 'r+' raises (write-back not implemented)."""
    data = np.arange(4, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "p.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with pytest.raises(IOError) as exc:
            rustfits.FITS(gz, "r+")
        assert "read-only" in str(exc.value)


def test_gz_wplus_rejected():
    """
    Opening a .gz with mode 'w+' raises."""
    data = np.arange(4, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "w.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with pytest.raises(IOError):
            rustfits.FITS(gz, "w+")


def test_gz_missing_file_raises():
    """
    A missing .gz path raises IOError, same as a missing plain path."""
    with pytest.raises(IOError):
        rustfits.FITS("/nonexistent/nope.fits.gz")


def test_gz_non_gzip_content_raises():
    """
    A .gz file whose contents aren't actually gzip raises a clear
    gunzip error, not a silent misparse."""
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.fits.gz")
        with open(bad, "wb") as fh:
            fh.write(b"this is not gzip data" * 100)
        with pytest.raises(IOError):
            rustfits.FITS(bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
