"""
Remote read via http:// — download-then-open, read-only.

Tests run against a local http.server serving a temp directory, so
they're deterministic and need no external network.  https:// uses the
same code path (only the URL scheme differs) and isn't unit-tested
here, since that would need a self-signed certificate.
"""

import functools
import gzip
import os
import tempfile
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pytest
import rustfits


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class _Server:
    def __init__(self, directory):
        self.dir = directory
        handler = functools.partial(_QuietHandler, directory=directory)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def url(self, name):
        return f"http://127.0.0.1:{self.port}/{name}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as d:
        srv = _Server(d)
        try:
            yield srv
        finally:
            srv.stop()


def _write_image(server, name, data, **kw):
    with rustfits.FITS(os.path.join(server.dir, name), "w+") as f:
        f.write_image(data, **kw)


def test_remote_image_roundtrip(server):
    """
    An image fetched over http reads back identically."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    _write_image(server, "img.fits", data)
    with rustfits.FITS(server.url("img.fits")) as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_remote_table_roundtrip(server):
    """
    A binary table fetched over http round-trips."""
    rec = np.array([(1, 1.5), (2, 2.5)], dtype=[("id", "i4"), ("x", "f8")])
    with rustfits.FITS(os.path.join(server.dir, "tab.fits"), "w+") as f:
        f.write_table(rec)
    with rustfits.FITS(server.url("tab.fits")) as f:
        got = f[1].read()
    np.testing.assert_array_equal(got["id"], rec["id"])
    np.testing.assert_array_equal(got["x"], rec["x"])


def test_remote_multi_hdu(server):
    """
    Multiple HDUs in a fetched file are all parsed."""
    img = np.arange(12, dtype="i4").reshape(3, 4)
    rec = np.array([(10,), (20,)], dtype=[("n", "i8")])
    with rustfits.FITS(os.path.join(server.dir, "multi.fits"), "w+") as f:
        f.write_image(img)
        f.write_table(rec)
    with rustfits.FITS(server.url("multi.fits")) as f:
        assert len(f.hdus) == 2
        np.testing.assert_array_equal(f[0].read(), img)
        np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_remote_gz_is_gunzipped(server):
    """
    A remote URL ending in .gz is gunzipped on download."""
    data = np.arange(8, dtype="i4").reshape(2, 4)
    plain = os.path.join(server.dir, "g.fits")
    _write_image(server, "g.fits", data)
    with open(plain, "rb") as fh:
        raw = fh.read()
    with gzip.open(os.path.join(server.dir, "g.fits.gz"), "wb") as fh:
        fh.write(raw)

    with rustfits.FITS(server.url("g.fits.gz")) as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_remote_tile_compressed(server):
    """
    A tile-compressed image fetched over http reads correctly."""
    data = np.arange(8 * 8, dtype="i4").reshape(8, 8)
    _write_image(
        server, "comp.fits", data, compress=rustfits.Rice1(tile_shape=(4, 4))
    )
    with rustfits.FITS(server.url("comp.fits")) as f:
        np.testing.assert_array_equal(f[1].read(), data)


def test_remote_slicing(server):
    """
    Slicing works on a fetched image (Mem-backed once downloaded)."""
    data = np.arange(10 * 10, dtype="i4").reshape(10, 10)
    _write_image(server, "slice.fits", data)
    with rustfits.FITS(server.url("slice.fits")) as f:
        np.testing.assert_array_equal(f[0][2:5, 1:4], data[2:5, 1:4])
        assert f[0][7, 8] == data[7, 8]


def test_remote_to_bytes(server):
    """
    to_bytes() on a fetched file returns the downloaded bytes."""
    data = np.arange(20, dtype="i2").reshape(4, 5)
    plain = os.path.join(server.dir, "tb.fits")
    _write_image(server, "tb.fits", data)
    with open(plain, "rb") as fh:
        raw = fh.read()
    with rustfits.FITS(server.url("tb.fits")) as f:
        assert f.to_bytes() == raw


def test_remote_convenience_read(server):
    """
    rustfits.read / read_header accept a URL (they open via FITS)."""
    data = np.arange(6, dtype="i4").reshape(2, 3)
    _write_image(server, "conv.fits", data)
    np.testing.assert_array_equal(rustfits.read(server.url("conv.fits")), data)
    hdr = rustfits.read_header(server.url("conv.fits"))
    assert hdr["NAXIS"] == 2


def test_remote_404_raises(server):
    """
    A missing remote path raises IOError (non-2xx status)."""
    with pytest.raises(IOError):
        rustfits.FITS(server.url("does-not-exist.fits"))


def test_remote_rplus_rejected(server):
    """
    Remote files are read-only; r+ raises before any fetch."""
    data = np.arange(4, dtype="i4")
    _write_image(server, "ro.fits", data)
    with pytest.raises(IOError) as exc:
        rustfits.FITS(server.url("ro.fits"), "r+")
    assert "read-only" in str(exc.value)


def test_remote_wplus_rejected(server):
    """
    Remote files are read-only; w+ raises."""
    data = np.arange(4, dtype="i4")
    _write_image(server, "ro2.fits", data)
    with pytest.raises(IOError):
        rustfits.FITS(server.url("ro2.fits"), "w+")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
