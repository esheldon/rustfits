"""
Ranged remote reads: FITS(url, "r", remote="ranged") serves reads
via block-cached HTTP Range requests instead of downloading the
whole file.

Tests run against a local http.server subclass that honors
single-range Range requests and counts requests + bytes served, so
partiality (the point of the feature) is asserted server-side.
Python's STOCK SimpleHTTPRequestHandler ignores Range (returns 200
full-body), which conveniently exercises the hard-error path: a
server that ignores Range must raise at open, never silently
download.

The Range server runs in a SUBPROCESS, not a thread.  Ranged block
fetches run while the client holds the GIL (by design — holding it
through the fetch is what makes the file-mutex/GIL pair deadlock-
free for multithreaded callers; see the roadmap), so an in-process
server thread could never get the GIL to answer and the test would
deadlock.  Only the open-time probe releases the GIL, which is why
the in-process PLAIN server (probe-only tests) is safe.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import rustfits

_SERVER_SCRIPT = r'''
import functools
import json
import os
import re
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0, "bytes_served": 0, "auth": []}
LOCK = threading.Lock()


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Control endpoints for the test process; not counted.
        if self.path == "/__stats__":
            with LOCK:
                return self._send_json(dict(STATS))
        if self.path == "/__reset__":
            with LOCK:
                STATS["requests"] = 0
                STATS["bytes_served"] = 0
                STATS["auth"] = []
            return self._send_json({"ok": True})
        rng = self.headers.get("Range")
        with LOCK:
            STATS["requests"] += 1
            STATS["auth"].append(self.headers.get("Authorization"))
        path = self.translate_path(self.path)
        if rng is None or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", rng.strip())
        if m is None:
            self.send_error(400, "unsupported Range form")
            return
        start, end = int(m.group(1)), int(m.group(2))
        if start >= size:
            self.send_error(416, "range not satisfiable")
            return
        end = min(end, size - 1)
        n = end - start + 1
        # Count BEFORE sending the body: the client finishing its
        # read then implies the stats are already up to date, so a
        # subsequent /__stats__ query cannot race the last update.
        with LOCK:
            STATS["bytes_served"] += n
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(n))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read(n))


directory = sys.argv[1]
handler = functools.partial(RangeHandler, directory=directory)
httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
print(httpd.server_address[1], flush=True)
httpd.serve_forever()
'''


class _RangeServer:
    """
    Range-honoring server in a subprocess (see module docstring for
    why it cannot be an in-process thread)."""

    def __init__(self, directory):
        self.dir = directory
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_SCRIPT, directory],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.port = int(self.proc.stdout.readline())

    def url(self, name):
        return f"http://127.0.0.1:{self.port}/{name}"

    def _control(self, path):
        with urllib.request.urlopen(self.url(path), timeout=10) as r:
            return json.loads(r.read())

    def stats(self):
        return self._control("__stats__")

    def reset(self):
        self._control("__reset__")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)


class _PlainHandler(SimpleHTTPRequestHandler):
    """
    The stock handler (ignores Range, always 200 full-body), quiet."""

    def log_message(self, *args):
        pass


class _PlainServer:
    """
    In-process thread server: safe here because only the GIL-
    released open-time probe ever talks to it."""

    def __init__(self, directory):
        self.dir = directory
        handler = functools.partial(_PlainHandler, directory=directory)
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
    """
    Range-honoring subprocess server with request/byte counters."""
    with tempfile.TemporaryDirectory() as d:
        srv = _RangeServer(d)
        try:
            yield srv
        finally:
            srv.stop()


@pytest.fixture(scope="module")
def plain_server():
    """
    Stock handler: ignores Range entirely (always 200)."""
    with tempfile.TemporaryDirectory() as d:
        srv = _PlainServer(d)
        try:
            yield srv
        finally:
            srv.stop()


def _write_image(server, name, data, **kw):
    with rustfits.FITS(os.path.join(server.dir, name), "w+") as f:
        f.write_image(data, **kw)
    return os.path.getsize(os.path.join(server.dir, name))


def _big_image():
    """
    800 KB of i4 data — big enough that partial reads are
    measurable against a 4096-byte fetch granularity."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 2**16, size=(400, 500), dtype="i4")


# ----------------------------------------------------------------
# correctness
# ----------------------------------------------------------------


def test_ranged_full_image_read(server):
    """
    A full read through ranged mode matches the original data."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    _write_image(server, "small.fits", data)
    with rustfits.FITS(server.url("small.fits"), "r", remote="ranged") as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_ranged_slicing_matches_local(server):
    """
    Slices, stepped slices, and scalar indexing all match."""
    data = _big_image()
    _write_image(server, "slices.fits", data)
    cfg = rustfits.Remote(ranged=True, block_bytes=4096)
    with rustfits.FITS(server.url("slices.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(
            f[0][100:110, 20:30], data[100:110, 20:30]
        )
        np.testing.assert_array_equal(f[0][::37, 3], data[::37, 3])
        assert f[0][123, 456] == data[123, 456]


def test_ranged_table_column(server):
    """
    A one-column read through ranged mode matches."""
    n = 20000
    rec = np.zeros(n, dtype=[("id", "i4"), ("x", "f8"), ("y", "f4")])
    rec["id"] = np.arange(n)
    rec["x"] = np.linspace(0.0, 1.0, n)
    rec["y"] = np.arange(n, dtype="f4") * 0.5
    with rustfits.FITS(os.path.join(server.dir, "tab.fits"), "w+") as f:
        f.write_table(rec)
    with rustfits.FITS(server.url("tab.fits"), "r", remote="ranged") as f:
        np.testing.assert_array_equal(f[1]["x"][:], rec["x"])
        got = f[1].read(rows=slice(5000, 5010))
        np.testing.assert_array_equal(got["id"], rec["id"][5000:5010])


def test_ranged_multi_hdu(server):
    """
    Multi-HDU files parse and read from every HDU."""
    img = np.arange(12, dtype="i4").reshape(3, 4)
    rec = np.array([(10,), (20,)], dtype=[("n", "i8")])
    with rustfits.FITS(os.path.join(server.dir, "multi.fits"), "w+") as f:
        f.write_image(img)
        f.write_table(rec)
    with rustfits.FITS(server.url("multi.fits"), "r", remote="ranged") as f:
        assert len(f.hdus) == 2
        np.testing.assert_array_equal(f[0].read(), img)
        np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_ranged_odd_block_size(server):
    """
    A non-power-of-two block size stresses the block-boundary
    arithmetic; results must be identical."""
    data = np.arange(50 * 60, dtype="i2").reshape(50, 60)
    _write_image(server, "odd.fits", data)
    cfg = rustfits.Remote(ranged=True, block_bytes=601)
    with rustfits.FITS(server.url("odd.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0].read(), data)
        np.testing.assert_array_equal(f[0][10:20, 5:15], data[10:20, 5:15])


def test_ranged_zero_cache_still_correct(server):
    """
    cache_bytes=0 disables caching (every read re-fetches) but
    stays correct — the read path holds fetched blocks directly."""
    data = np.arange(30 * 40, dtype="i4").reshape(30, 40)
    _write_image(server, "nocache.fits", data)
    cfg = rustfits.Remote(ranged=True, block_bytes=4096, cache_bytes=0)
    with rustfits.FITS(server.url("nocache.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0].read(), data)


def test_shorthand_matches_object(server):
    """
    remote='ranged' behaves identically to Remote(ranged=True)."""
    data = np.arange(64, dtype="f4").reshape(8, 8)
    _write_image(server, "short.fits", data)
    with rustfits.FITS(server.url("short.fits"), "r", remote="ranged") as f:
        a = f[0].read()
    with rustfits.FITS(
        server.url("short.fits"), "r", remote=rustfits.Remote(ranged=True)
    ) as f:
        b = f[0].read()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, data)


# ----------------------------------------------------------------
# partiality — the point of the feature
# ----------------------------------------------------------------


def test_ranged_header_read_is_partial(server):
    """
    Opening + reading headers fetches a few blocks, not the file."""
    data = _big_image()
    size = _write_image(server, "hdronly.fits", data)
    assert size > 700_000
    cfg = rustfits.Remote(ranged=True, block_bytes=4096)
    server.reset()
    with rustfits.FITS(server.url("hdronly.fits"), "r", remote=cfg) as f:
        assert f[0].header["NAXIS"] == 2
    assert server.stats()["bytes_served"] < 65536


def test_ranged_slice_is_partial(server):
    """
    A small slice of a big image fetches a small fraction of it."""
    data = _big_image()
    size = _write_image(server, "part.fits", data)
    cfg = rustfits.Remote(ranged=True, block_bytes=4096)
    server.reset()
    with rustfits.FITS(server.url("part.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0][100:110, :], data[100:110, :])
    assert server.stats()["bytes_served"] < size // 4


def test_ranged_compressed_slice_is_partial(server):
    """
    On a tile-compressed image, a slice fetches only the touched
    tiles (plus headers + descriptors), not the whole heap."""
    data = _big_image()
    size = _write_image(
        server,
        "comp.fits.fz",
        data,
        compress=rustfits.Rice1(tile_shape=(10, 500)),
    )
    cfg = rustfits.Remote(ranged=True, block_bytes=4096)
    server.reset()
    with rustfits.FITS(server.url("comp.fits.fz"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[1][100:110, :], data[100:110, :])
    assert server.stats()["bytes_served"] < size // 2


def test_ranged_repeated_read_hits_cache(server):
    """
    Re-reading the same slice issues no new requests — the blocks
    are in the LRU."""
    data = np.arange(100 * 100, dtype="i4").reshape(100, 100)
    _write_image(server, "cache.fits", data)
    cfg = rustfits.Remote(ranged=True, block_bytes=4096)
    with rustfits.FITS(server.url("cache.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0][10:20, :], data[10:20, :])
        before = server.stats()["requests"]
        np.testing.assert_array_equal(f[0][10:20, :], data[10:20, :])
        after = server.stats()["requests"]
    assert after == before


# ----------------------------------------------------------------
# headers / timeout forwarding
# ----------------------------------------------------------------


def test_headers_forwarded_ranged(server):
    """
    Custom headers ride on the probe and every block fetch."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    _write_image(server, "auth.fits", data)
    cfg = rustfits.Remote(
        ranged=True, headers={"Authorization": "Bearer tok123"}
    )
    server.reset()
    with rustfits.FITS(server.url("auth.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0].read(), data)
    auth = server.stats()["auth"]
    assert auth
    assert all(a == "Bearer tok123" for a in auth)


def test_headers_forwarded_download(server):
    """
    Remote(ranged=False, headers=...) applies to the plain
    download mode too (token-gated archives)."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    _write_image(server, "authd.fits", data)
    cfg = rustfits.Remote(headers={"Authorization": "Bearer tokDL"})
    server.reset()
    with rustfits.FITS(server.url("authd.fits"), "r", remote=cfg) as f:
        np.testing.assert_array_equal(f[0].read(), data)
    auth = server.stats()["auth"]
    assert auth
    assert all(a == "Bearer tokDL" for a in auth)


# ----------------------------------------------------------------
# rejection paths
# ----------------------------------------------------------------


def test_plain_server_hard_error(plain_server):
    """
    A server that ignores Range raises at open — never a silent
    whole-file download — and the message names the fix."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    _write_image(plain_server, "noranges.fits", data)
    with pytest.raises(IOError) as exc:
        rustfits.FITS(plain_server.url("noranges.fits"), "r", remote="ranged")
    assert "ranged=False" in str(exc.value)


def test_ranged_404_raises(server):
    """
    A missing remote path fails at the probe."""
    with pytest.raises(IOError):
        rustfits.FITS(server.url("missing.fits"), "r", remote="ranged")


def test_ranged_write_modes_rejected(server):
    """
    Ranged (like all remote) is read-only; r+/w+ raise before any
    network I/O."""
    for mode in ("r+", "w+"):
        with pytest.raises(IOError) as exc:
            rustfits.FITS(server.url("x.fits"), mode, remote="ranged")
        assert "read-only" in str(exc.value)


def test_remote_kwarg_local_path_rejected(tmp_path):
    """
    remote= on a non-URL path is an intent error."""
    data = np.arange(4, dtype="i4")
    fname = str(tmp_path / "local.fits")
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(data)
    with pytest.raises(ValueError, match="only valid for"):
        rustfits.FITS(fname, "r", remote="ranged")


def test_remote_true_rejected(server):
    """
    remote=True has no meaning; the error points at 'ranged'."""
    with pytest.raises(ValueError, match="ranged"):
        rustfits.FITS(server.url("x.fits"), "r", remote=True)


def test_remote_bad_string_rejected(server):
    """
    Only 'ranged' is accepted as a string shorthand."""
    with pytest.raises(ValueError, match="'ranged'"):
        rustfits.FITS(server.url("x.fits"), "r", remote="download")


def test_block_cache_kwargs_require_ranged():
    """
    block_bytes/cache_bytes with ranged=False is an explicit
    mistake, rejected at Remote() construction."""
    with pytest.raises(ValueError, match="ranged=True"):
        rustfits.Remote(block_bytes=4096)
    with pytest.raises(ValueError, match="ranged=True"):
        rustfits.Remote(cache_bytes=1 << 20)


def test_remote_validation():
    """
    Remote() rejects nonsense knob values."""
    with pytest.raises(ValueError, match="timeout"):
        rustfits.Remote(timeout=0.0)
    with pytest.raises(ValueError, match="timeout"):
        rustfits.Remote(timeout=-1.0)
    with pytest.raises(ValueError, match="block_bytes"):
        rustfits.Remote(ranged=True, block_bytes=0)
    with pytest.raises(ValueError, match="str"):
        rustfits.Remote(headers={"K": 5})


def test_ranged_gz_url_rejected(server):
    """
    gzip is not seekable; ranged + .gz URL is rejected up front."""
    with pytest.raises(ValueError, match="not seekable"):
        rustfits.FITS(server.url("f.fits.gz"), "r", remote="ranged")


def test_ranged_ftp_rejected():
    """
    Ranged is http/https only; validation fires before any
    connection (the host here does not exist)."""
    with pytest.raises(ValueError, match="http"):
        rustfits.FITS("ftp://example.invalid/f.fits", "r", remote="ranged")


def test_ftp_headers_rejected():
    """
    headers=/timeout= are HTTP-only knobs; rejected on ftp before
    any connection."""
    cfg = rustfits.Remote(headers={"Authorization": "x"})
    with pytest.raises(ValueError, match="http"):
        rustfits.FITS("ftp://example.invalid/f.fits", "r", remote=cfg)


def test_to_bytes_rejected(server):
    """
    to_bytes() would download the whole file; rejected with a
    pointer at download mode."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    _write_image(server, "tb.fits", data)
    with rustfits.FITS(server.url("tb.fits"), "r", remote="ranged") as f:
        with pytest.raises(IOError, match="ranged=False"):
            f.to_bytes()


# ----------------------------------------------------------------
# the Remote config object
# ----------------------------------------------------------------


def test_remote_getters_defaults():
    """
    Attribute round-trip and defaults."""
    r = rustfits.Remote()
    assert r.ranged is False
    assert r.headers is None
    assert r.timeout is None
    assert r.block_bytes == 1 << 20
    assert r.cache_bytes == 32 << 20

    r = rustfits.Remote(
        ranged=True,
        headers={"Authorization": "Bearer x"},
        timeout=2.5,
        block_bytes=8192,
        cache_bytes=1 << 20,
    )
    assert r.ranged is True
    assert r.headers == {"Authorization": "Bearer x"}
    assert r.timeout == 2.5
    assert r.block_bytes == 8192
    assert r.cache_bytes == 1 << 20
    assert "ranged=True" in repr(r)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
