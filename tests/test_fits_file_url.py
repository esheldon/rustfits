"""
file:// URLs as an alternate spelling of a local path (RFC 8089 /
cfitsio's file driver prefix).

The URL is decoded to the plain path at the top of FITS.__init__,
so every mode (r / r+ / w+), the .gz write-back, and the repr all
behave exactly as if the plain path had been passed.  FITS.filename
returns the DECODED local path, not the URL.

URLs are built with pathlib.Path.as_uri() so the tests are
platform-portable: on Windows it emits file:///C:/... (forward
slashes, drive-letter form — exercising the decoder's drive-letter
branch for real), and it applies the same percent-encoding a
browser or file manager would.  Hand-built file://<backslash-path>
strings are not valid file URLs and are expected to fail.
"""

import os
import pathlib
import tempfile
import numpy as np
import pytest
import rustfits


def _url(path):
    """
    Build a file:/// URL from an absolute path, the way Python
    itself does (correct on Windows too: file:///C:/...)."""
    return pathlib.Path(path).as_uri()


def _write_image(fname):
    """
    Write a small deterministic image file; return the data."""
    data = np.arange(3 * 4, dtype="i4").reshape(3, 4)
    with rustfits.FITS(fname, "w+") as f:
        f.write_image(data)
    return data


# ----------------------------------------------------------------
# read
# ----------------------------------------------------------------


def test_read_matches_plain_path():
    """
    Opening file:///abs/path reads the same bytes as the plain
    path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        data = _write_image(fname)
        with rustfits.FITS(_url(fname), "r") as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_localhost_form():
    """
    file://localhost/abs/path is accepted (RFC 8089 authority
    form), case-insensitively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        data = _write_image(fname)
        for host in ("localhost", "LOCALHOST"):
            url = _url(fname).replace("file:///", f"file://{host}/", 1)
            with rustfits.FITS(url, "r") as f:
                np.testing.assert_array_equal(f[0].read(), data)


def test_filename_attribute_is_decoded_path():
    """
    FITS.filename holds the decoded local path, not the URL.
    Compared via pathlib.Path: the decoder emits forward slashes
    on Windows (C:/...), which Path treats as equal to C:\\..."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        _write_image(fname)
        with rustfits.FITS(_url(fname), "r") as f:
            assert not f.filename.startswith("file:")
            assert pathlib.Path(f.filename) == pathlib.Path(fname)


def test_percent_encoded_space():
    """
    %20 in the URL decodes to a space in the filename.  as_uri()
    percent-encodes the space, so this pins that we decode what
    the stdlib emits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "my file.fits")
        data = _write_image(fname)
        url = _url(fname)
        assert "%20" in url
        with rustfits.FITS(url, "r") as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_percent_encoded_utf8():
    """
    Multi-byte percent-escapes decode as UTF-8 (e-acute here);
    as_uri() encodes non-ASCII as UTF-8 escape pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "café.fits")
        data = _write_image(fname)
        url = _url(fname)
        assert "%C3%A9" in url
        with rustfits.FITS(url, "r") as f:
            np.testing.assert_array_equal(f[0].read(), data)


def test_convenience_read():
    """
    The top-level rustfits.read accepts a file:// URL (it opens
    via FITS, so the translation comes for free)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        data = _write_image(fname)
        np.testing.assert_array_equal(rustfits.read(_url(fname)), data)


# ----------------------------------------------------------------
# write modes (file:// is a local path: all modes work)
# ----------------------------------------------------------------


def test_create_w_plus_via_url():
    """
    w+ through a file:// URL creates the file on disk at the
    decoded path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "new.fits")
        data = np.arange(6, dtype="f8").reshape(2, 3)
        with rustfits.FITS(_url(fname), "w+") as f:
            f.write_image(data)
            np.testing.assert_array_equal(f[0].read(), data)  # same-handle
        assert os.path.exists(fname)
        with rustfits.FITS(fname, "r") as f:  # reopen via plain path
            np.testing.assert_array_equal(f[0].read(), data)


def test_mutate_r_plus_via_url():
    """
    r+ header mutation through a file:// URL persists to the
    decoded path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        _write_image(fname)
        with rustfits.FITS(_url(fname), "r+") as f:
            f[0].header["MYKEY"] = 42
            assert f[0].header["MYKEY"] == 42  # same-handle
        with rustfits.FITS(fname, "r") as f:  # post-reopen
            assert f[0].header["MYKEY"] == 42


def test_gz_via_url_roundtrip():
    """
    A .gz path spelled as a file:// URL routes through the gzip
    driver, including the close-time write-back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits.gz")
        data = np.arange(8, dtype="i4").reshape(2, 4)
        with rustfits.FITS(_url(fname), "w+") as f:
            f.write_image(data)
        # Write-back landed at the decoded path as real gzip bytes.
        with open(fname, "rb") as fobj:
            assert fobj.read(2) == b"\x1f\x8b"
        with rustfits.FITS(fname, "r") as f:  # reopen via plain path
            np.testing.assert_array_equal(f[0].read(), data)
        with rustfits.FITS(_url(fname), "r") as f:  # and via URL
            np.testing.assert_array_equal(f[0].read(), data)


# ----------------------------------------------------------------
# rejections
# ----------------------------------------------------------------


def test_nonlocal_host_rejected():
    """
    A non-localhost authority names a remote machine: rejected."""
    with pytest.raises(ValueError, match="otherhost"):
        rustfits.FITS("file://otherhost/data/x.fits", "r")


def test_missing_path_rejected():
    """
    file:// with no path component is rejected."""
    with pytest.raises(ValueError, match="no path"):
        rustfits.FITS("file://", "r")
    with pytest.raises(ValueError, match="no path"):
        rustfits.FITS("file://localhost", "r")


def test_bad_percent_escape_rejected():
    """
    Malformed percent-escapes are rejected, not passed through."""
    with pytest.raises(ValueError, match="percent-escape"):
        rustfits.FITS("file:///data/x%2G.fits", "r")
    with pytest.raises(ValueError, match="percent-escape"):
        rustfits.FITS("file:///data/x%2", "r")


def test_remote_kwarg_rejected():
    """
    remote= does not apply to file:// (it names a local file);
    both the string shorthand and a Remote instance are rejected."""
    with pytest.raises(ValueError, match="local file"):
        rustfits.FITS("file:///data/x.fits", "r", remote="ranged")
    with pytest.raises(ValueError, match="local file"):
        rustfits.FITS(
            "file:///data/x.fits",
            "r",
            remote=rustfits.Remote(ranged=True),
        )


def test_nonexistent_path_raises_oserror():
    """
    A well-formed URL to a missing file fails at open like a plain
    path would (OSError, not ValueError)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "no_such.fits")
        with pytest.raises(OSError, match="no_such"):
            rustfits.FITS(_url(fname), "r")


def test_pathlib_path_still_works():
    """
    Sanity check: plain pathlib.Path filenames are unaffected by
    the file:// URL support."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "img.fits")
        data = _write_image(fname)
        with rustfits.FITS(pathlib.Path(fname), "r") as f:
            np.testing.assert_array_equal(f[0].read(), data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
