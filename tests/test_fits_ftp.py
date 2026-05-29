"""
Remote read via ftp:// (and ftps:// routing) — download-then-open,
read-only.

Plain-ftp tests run against a local aioftp anonymous server, so
they're deterministic and need no external network.  A full ftps://
handshake isn't tested locally (it would need a CA-valid server, same
situation as https://); instead a negative test points ftps:// at the
plain server to confirm the TLS-upgrade path (connector build +
into_secure) is taken.
"""

import asyncio
import gzip
import os
import logging
import tempfile
import threading

import numpy as np
import pytest
import rustfits

aioftp = pytest.importorskip("aioftp")

# aioftp logs connections; quiet it for clean test output.
logging.getLogger("aioftp").setLevel(logging.CRITICAL)


class _FtpServer:
    """
    A local anonymous FTP server backed by aioftp, run in a background
    asyncio loop thread so the synchronous tests can talk to it.
    """

    def __init__(self, directory):
        self.dir = directory
        self.loop = asyncio.new_event_loop()
        user = aioftp.User(
            base_path=directory,
            home_path="/",
            permissions=[
                aioftp.Permission("/", readable=True, writable=False)
            ],
        )
        self.server = aioftp.Server([user])
        self.port = None
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.server.start("127.0.0.1", 0))
            self.port = self.server.server_port
            ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        if not ready.wait(timeout=10):
            raise RuntimeError("aioftp server failed to start")

    def url(self, name, scheme="ftp"):
        return f"{scheme}://127.0.0.1:{self.port}/{name}"

    def stop(self):
        async def _close():
            await self.server.close()

        try:
            asyncio.run_coroutine_threadsafe(_close(), self.loop).result(
                timeout=5
            )
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as d:
        srv = _FtpServer(d)
        try:
            yield srv
        finally:
            srv.stop()


def _write_image(server, name, data, **kw):
    with rustfits.FITS(os.path.join(server.dir, name), "w+") as f:
        f.write_image(data, **kw)


def test_ftp_image_roundtrip(server):
    """
    An image fetched over ftp reads back identically."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    _write_image(server, "img.fits", data)
    with rustfits.FITS(server.url("img.fits")) as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_ftp_table_roundtrip(server):
    """
    A binary table fetched over ftp round-trips."""
    rec = np.array([(1, 1.5), (2, 2.5)], dtype=[("id", "i4"), ("x", "f8")])
    with rustfits.FITS(os.path.join(server.dir, "tab.fits"), "w+") as f:
        f.write_table(rec)
    with rustfits.FITS(server.url("tab.fits")) as f:
        got = f[1].read()
    np.testing.assert_array_equal(got["id"], rec["id"])
    np.testing.assert_array_equal(got["x"], rec["x"])


def test_ftp_multi_hdu(server):
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


def test_ftp_gz_is_gunzipped(server):
    """
    A remote ftp URL ending in .gz is gunzipped on download."""
    data = np.arange(8, dtype="i4").reshape(2, 4)
    plain = os.path.join(server.dir, "g.fits")
    _write_image(server, "g.fits", data)
    with open(plain, "rb") as fh:
        raw = fh.read()
    with gzip.open(os.path.join(server.dir, "g.fits.gz"), "wb") as fh:
        fh.write(raw)

    with rustfits.FITS(server.url("g.fits.gz")) as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_ftp_to_bytes(server):
    """
    to_bytes() on a fetched file returns the downloaded bytes."""
    data = np.arange(20, dtype="i2").reshape(4, 5)
    plain = os.path.join(server.dir, "tb.fits")
    _write_image(server, "tb.fits", data)
    with open(plain, "rb") as fh:
        raw = fh.read()
    with rustfits.FITS(server.url("tb.fits")) as f:
        assert f.to_bytes() == raw


def test_ftp_convenience_read(server):
    """
    rustfits.read / read_header accept an ftp URL."""
    data = np.arange(6, dtype="i4").reshape(2, 3)
    _write_image(server, "conv.fits", data)
    np.testing.assert_array_equal(rustfits.read(server.url("conv.fits")), data)
    hdr = rustfits.read_header(server.url("conv.fits"))
    assert hdr["NAXIS"] == 2


def test_ftp_missing_file_raises(server):
    """
    A missing ftp path raises IOError (RETR fails)."""
    with pytest.raises(IOError):
        rustfits.FITS(server.url("does-not-exist.fits"))


def test_ftp_rplus_rejected(server):
    """
    Remote files are read-only; r+ raises before any connection."""
    data = np.arange(4, dtype="i4")
    _write_image(server, "ro.fits", data)
    with pytest.raises(IOError) as exc:
        rustfits.FITS(server.url("ro.fits"), "r+")
    assert "read-only" in str(exc.value)


def test_ftp_wplus_rejected(server):
    """
    Remote files are read-only; w+ raises."""
    data = np.arange(4, dtype="i4")
    _write_image(server, "ro2.fits", data)
    with pytest.raises(IOError):
        rustfits.FITS(server.url("ro2.fits"), "w+")


def test_ftps_to_plain_server_raises(server):
    """
    ftps:// against a plain (non-TLS) server raises: the AUTH TLS
    upgrade fails.  This exercises the ftps routing + rustls connector
    construction + into_secure path (the plain server rejects AUTH)."""
    data = np.arange(4, dtype="i4")
    _write_image(server, "sec.fits", data)
    with pytest.raises(IOError):
        rustfits.FITS(server.url("sec.fits", scheme="ftps"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
