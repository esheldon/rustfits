"""
Whole-file gzip support: opening a `.gz` path gunzips the whole file
into an in-memory buffer (Storage::Mem) and parses it like any other
file.  In a writable mode (`r+` / `w+`) the buffer is recompressed and
written back to the `.gz` path on close().

These exercise the `.gz` detection + gunzip-on-open path AND the
recompress-on-close write-back path.  Read fixtures are produced with
Python's gzip module (compressing a plain .fits we wrote first), so the
test asserts the gz read matches the plain read.  Write-back tests
re-open with the stdlib gzip module to confirm the on-disk file is
valid gzip with the expected decompressed bytes.
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


def _read_gz_bytes(gz_path):
    """
    Decompress a .gz file with the stdlib and return the raw bytes."""
    with gzip.open(gz_path, "rb") as fh:
        return fh.read()


def test_gz_wplus_writeback():
    """
    Opening a .gz with mode 'w+' builds the file in RAM and recompresses
    it to the .gz path on close; it then reads back identically."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "w.fits.gz")
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(data)
        # The on-disk file is real gzip (stdlib can decompress it).
        with rustfits.FITS(gz) as f:  # reopen read-only
            np.testing.assert_array_equal(f[0].read(), data)
        # And the decompressed bytes parse as a plain FITS too.
        raw = _read_gz_bytes(gz)
        with rustfits.FITS.from_bytes(raw) as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_gz_wplus_creates_new_file():
    """
    w+ on a .gz path that does not yet exist creates it."""
    data = np.arange(12, dtype="f4").reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "fresh.fits.gz")
        assert not os.path.exists(gz)
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(data)
        assert os.path.exists(gz)
        with rustfits.FITS(gz) as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_gz_wplus_truncates_existing():
    """
    w+ on an existing .gz ignores the old content (truncate semantics):
    the closed file holds only what was written this session."""
    old = np.arange(100, dtype="i8")
    new = np.arange(6, dtype="i4").reshape(2, 3)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "trunc.fits.gz")
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(old)
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(new)
        with rustfits.FITS(gz) as f:
            assert len(f.hdus) == 1
            np.testing.assert_array_equal(f[0].read(), new)


def test_gz_rplus_writeback():
    """
    r+ on a .gz mutates the in-memory copy and writes it back on close;
    the change persists when reopened."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "p.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz, "r+") as f:
            f[0].header["NEWKEY"] = 42
            f[0][0, 0] = 999
            # same-handle: change visible before close
            assert f[0].header["NEWKEY"] == 42
            assert f[0][0, 0] == 999

        # post-reopen: change persisted to the recompressed .gz
        with rustfits.FITS(gz) as f:
            assert f[0].header["NEWKEY"] == 42
            assert f[0][0, 0] == 999
            expected = data.copy()
            expected[0, 0] = 999
            np.testing.assert_array_equal(f[0].read(), expected)


def test_gz_rplus_append_hdu_writeback():
    """
    Appending a whole HDU to an r+ .gz persists on close."""
    img = np.arange(12, dtype="i4").reshape(3, 4)
    rec = np.array([(10,), (20,)], dtype=[("n", "i8")])
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "a.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, img)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz, "r+") as f:
            f.write_table(rec)

        with rustfits.FITS(gz) as f:
            assert len(f.hdus) == 2
            np.testing.assert_array_equal(f[0].read(), img)
            np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_gz_writeback_matches_plain_bytes():
    """
    The bytes round-tripped through a w+ .gz are identical to writing
    the same data to a plain .fits — gzip is byte-transparent."""
    rec = np.array([(1, 1.5), (2, 2.5)], dtype=[("id", "i4"), ("x", "f8")])
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "t.fits")
        gz = os.path.join(d, "t2.fits.gz")
        with rustfits.FITS(plain, "w+") as f:
            f.write_table(rec)
        with rustfits.FITS(gz, "w+") as f:
            f.write_table(rec)
        with open(plain, "rb") as fh:
            plain_bytes = fh.read()
        assert _read_gz_bytes(gz) == plain_bytes


def test_gz_writeback_close_idempotent():
    """
    A second close() on a written-back .gz is a no-op (does not raise,
    does not corrupt the file)."""
    data = np.arange(8, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "idem.fits.gz")
        f = rustfits.FITS(gz, "w+")
        f.write_image(data)
        f.close()
        f.close()  # second close — no-op
        with rustfits.FITS(gz) as g:
            np.testing.assert_array_equal(g[0].read(), data)


def test_gz_writeback_fitsio_can_read():
    """
    A .gz written by rustfits is readable by fitsio (cross-tool)."""
    import fitsio

    data = np.arange(5 * 4, dtype="f4").reshape(5, 4)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "x.fits.gz")
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(data)
        with fitsio.FITS(gz) as ff:
            np.testing.assert_array_equal(ff[0].read(), data)


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
