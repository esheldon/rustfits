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

import gc
import gzip
import os
import tempfile
import numpy as np
import pytest
import rustfits

_skip_as_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission bits do not block writes when running as root",
)


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
            # same-handle: visible before close
            np.testing.assert_array_equal(f[0].read(), data)
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
            # same-handle
            np.testing.assert_array_equal(f[0].read(), data)
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
            # same-handle: only the new image is present
            assert len(f.hdus) == 1
            np.testing.assert_array_equal(f[0].read(), new)
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
            # same-handle: appended HDU visible before close
            assert len(f.hdus) == 2
            np.testing.assert_array_equal(f[1].read()["n"], rec["n"])

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


def _write_multi_member_gz(plain_path, gz_path):
    """
    Write `plain_path`'s bytes as a TWO-member gzip stream to `gz_path`.

    Concatenated gzip members are valid per the gzip spec and decompress
    to the concatenation of their payloads, so the on-disk .gz
    decompresses back to the original file bytes.  A single-member
    decoder would read only the first member (truncated); this fixture
    exists to prove the reader uses a multi-member decoder.
    """
    with open(plain_path, "rb") as fh:
        raw = fh.read()
    half = len(raw) // 2
    member1 = gzip.compress(raw[:half])
    member2 = gzip.compress(raw[half:])
    with open(gz_path, "wb") as fh:
        fh.write(member1 + member2)


def test_gz_multi_member_read():
    """
    A multi-member gzip .gz is decoded in FULL, not truncated to its
    first member (regression: single-member GzDecoder would silently
    lose the trailing members)."""
    img = np.arange(20 * 8, dtype="i4").reshape(20, 8)
    rec = np.array(
        [(i, i * 1.5) for i in range(30)], dtype=[("n", "i8"), ("x", "f8")]
    )
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "mm.fits")
        gz = plain + ".gz"
        with rustfits.FITS(plain, "w+") as f:
            f.write_image(img)
            f.write_table(rec)
        _write_multi_member_gz(plain, gz)

        # Sanity: it really is multi-member (the stdlib still reads it,
        # and the byte stream contains two gzip magic headers).
        with open(gz, "rb") as fh:
            blob = fh.read()
        assert blob.count(b"\x1f\x8b") >= 2
        assert _read_gz_bytes(gz) == open(plain, "rb").read()

        with rustfits.FITS(gz) as f:
            assert len(f.hdus) == 2
            np.testing.assert_array_equal(f[0].read(), img)
            got = f[1].read()
            np.testing.assert_array_equal(got["n"], rec["n"])
            np.testing.assert_array_equal(got["x"], rec["x"])


def test_gz_multi_member_writeback_preserves_content():
    """
    Opening a multi-member .gz r+, mutating, and closing preserves ALL
    the original content — the trailing members are not lost (regression:
    single-member read + recompress-on-close would have destroyed them)."""
    img = np.arange(20 * 8, dtype="i4").reshape(20, 8)
    rec = np.array([(i,) for i in range(30)], dtype=[("n", "i8")])
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "mm2.fits")
        gz = plain + ".gz"
        with rustfits.FITS(plain, "w+") as f:
            f.write_image(img)
            f.write_table(rec)
        _write_multi_member_gz(plain, gz)

        with rustfits.FITS(gz, "r+") as f:
            f[0].header["EDITED"] = 7

        with rustfits.FITS(gz) as f:
            assert len(f.hdus) == 2
            assert f[0].header["EDITED"] == 7
            np.testing.assert_array_equal(f[0].read(), img)
            np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_gz_writeback_leaves_no_temp_litter():
    """
    A successful write-back renames its temp file into place and leaves
    no stray temp files in the directory."""
    data = np.arange(6, dtype="i4").reshape(2, 3)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "clean.fits.gz")
        with rustfits.FITS(gz, "w+") as f:
            f.write_image(data)
        entries = os.listdir(d)
        assert entries == ["clean.fits.gz"], entries
        assert not any("rustfits-tmp" in e for e in entries)


@_skip_as_root
def test_gz_writeback_failure_preserves_original():
    """
    If the recompress-on-close fails (here: the directory is made
    unwritable so the temp file cannot be created), the ORIGINAL .gz is
    left intact — the write is atomic (temp + rename), never an in-place
    truncate.  A second close() after the failure is a clean no-op."""
    orig = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "atomic.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, orig)
        _gzip_file(plain, gz)
        original_bytes = open(gz, "rb").read()

        f = rustfits.FITS(gz, "r+")
        f[0].header["NEWKEY"] = 123
        f[0][0, 0] = 999  # mutate the in-RAM buffer
        os.chmod(d, 0o500)  # read + execute, no write -> temp create fails
        try:
            with pytest.raises(IOError):
                f.close()
            # Second close after a failed write-back must not raise and
            # must not re-attempt the (still-failing) write.
            f.close()
        finally:
            os.chmod(d, 0o700)

        # The on-disk file is byte-for-byte the original: not truncated,
        # not partially written, mutation NOT applied.
        assert open(gz, "rb").read() == original_bytes
        with rustfits.FITS(gz) as g:
            assert "NEWKEY" not in g[0].header
            np.testing.assert_array_equal(g[0].read(), orig)


# ----------------------------------------------------------------------
# Open-time semantics: writable .gz now matches plain-disk open, so
# creation / permission errors surface at open (not deferred to close).
# ----------------------------------------------------------------------


def test_gz_wplus_file_exists_after_open():
    """
    #7: w+ creates/claims the .gz on disk at OPEN (like a plain w+),
    before any write or close — not only once close() runs."""
    data = np.arange(6, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "claim.fits.gz")
        assert not os.path.exists(gz)
        f = rustfits.FITS(gz, "w+")
        try:
            # File exists immediately after construction.
            assert os.path.exists(gz)
        finally:
            f.write_image(data)
            f.close()
        with rustfits.FITS(gz) as g:
            np.testing.assert_array_equal(g[0].read(), data)


@_skip_as_root
def test_gz_wplus_unwritable_dir_raises_at_open():
    """
    #5: w+ on a .gz in a read-only directory raises at construction
    (not after all the work, at close), and creates no file."""
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "ro")
        os.mkdir(sub)
        gz = os.path.join(sub, "x.fits.gz")
        os.chmod(sub, 0o500)  # read + execute, no write
        try:
            with pytest.raises(IOError):
                rustfits.FITS(gz, "w+")
            assert not os.path.exists(gz)
        finally:
            os.chmod(sub, 0o700)


@_skip_as_root
def test_gz_rplus_readonly_file_raises_at_open():
    """
    #5: r+ on a read-only .gz file raises at open (matches plain r+),
    rather than succeeding and failing only at the close() write-back."""
    data = np.arange(6, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "ro.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        os.chmod(gz, 0o400)  # read-only file
        try:
            with pytest.raises(IOError):
                rustfits.FITS(gz, "r+")
        finally:
            os.chmod(gz, 0o600)
        # 'r' still works on a read-only file.
        with rustfits.FITS(gz) as f:
            np.testing.assert_array_equal(f[0].read(), data)


# ----------------------------------------------------------------------
# #6: an r+/w+ .gz is rewritten on close ONLY if it was mutated.  The
# atomic write replaces the file (new inode); a skipped write leaves the
# inode unchanged, which is what these assert.
# ----------------------------------------------------------------------


def test_gz_rplus_read_only_does_not_rewrite():
    """
    Opening r+ and only reading leaves the on-disk file untouched (same
    inode) — no needless recompress / mtime churn."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "ro2.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        ino0 = os.stat(gz).st_ino

        with rustfits.FITS(gz, "r+") as f:
            np.testing.assert_array_equal(f[0].read(), data)  # read only

        assert os.stat(gz).st_ino == ino0  # not rewritten


def test_gz_rplus_mutation_does_rewrite():
    """
    The mutating counterpart: an actual edit DOES rewrite the file (new
    inode), confirming the inode check above is meaningful."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "rw.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        ino0 = os.stat(gz).st_ino

        with rustfits.FITS(gz, "r+") as f:
            f[0].header["EDIT"] = 1

        assert os.stat(gz).st_ino != ino0  # rewritten


# ----------------------------------------------------------------------
# #8: sync() makes a writable .gz durable mid-session.
# ----------------------------------------------------------------------


def test_gz_sync_writes_back_mid_session():
    """
    sync() recompresses + writes the .gz to disk before close, so a
    fresh read-only handle sees the mutation; the subsequent close()
    then does not rewrite again (dirty was cleared by sync)."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "s.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)

        with rustfits.FITS(gz, "r+") as f:
            f[0].header["SYNCED"] = 5
            f.sync()
            # On-disk file already reflects the edit, before close.
            with rustfits.FITS(gz) as g:
                assert g[0].header["SYNCED"] == 5
            ino_after_sync = os.stat(gz).st_ino
        # close() saw a clean buffer (sync cleared dirty) -> no rewrite.
        assert os.stat(gz).st_ino == ino_after_sync


def test_gz_sync_noop_when_unmutated():
    """
    sync() on a writable .gz with no pending mutation is a no-op (does
    not rewrite the file)."""
    data = np.arange(8, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "sn.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        ino0 = os.stat(gz).st_ino

        with rustfits.FITS(gz, "r+") as f:
            f.sync()  # nothing mutated
            assert os.stat(gz).st_ino == ino0

        assert os.stat(gz).st_ino == ino0


# ----------------------------------------------------------------------
# #4: a writable .gz that is never explicitly closed is flushed by the
# Drop finalizer instead of silently losing the data.
# ----------------------------------------------------------------------


def test_gz_drop_flushes_unclosed():
    """
    Forgetting to close() a written w+ .gz still persists the data: the
    Drop finalizer writes it back when the object is GC'd."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "drop.fits.gz")
        f = rustfits.FITS(gz, "w+")
        f.write_image(data)
        del f  # no close() — rely on the finalizer
        gc.collect()

        with rustfits.FITS(gz) as g:
            np.testing.assert_array_equal(g[0].read(), data)


def test_gz_drop_noop_when_clean():
    """
    The Drop finalizer does not rewrite an unmutated r+ .gz (same inode
    after GC)."""
    data = np.arange(8, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "dropclean.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        ino0 = os.stat(gz).st_ino

        f = rustfits.FITS(gz, "r+")
        _ = f[0].read()  # read only
        del f
        gc.collect()

        assert os.stat(gz).st_ino == ino0


# ----------------------------------------------------------------------
# #11: a buffer left inconsistent by a mid-write failure (taint flag) is
# NOT persisted; the existing .gz on disk is preserved.
# ----------------------------------------------------------------------


def test_gz_close_refuses_tainted_buffer():
    """
    close() does not write back a tainted buffer: it raises, and the
    original .gz is left untouched so a reopen recovers the good file."""
    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "taint.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        original_bytes = open(gz, "rb").read()

        f = rustfits.FITS(gz, "r+")
        f[0].header["EDIT"] = 1  # mutate -> buffer dirty
        f[0]._force_taint()  # simulate a prior mid-write failure
        with pytest.raises(IOError):
            f.close()  # refuses to persist the inconsistent buffer

        # Original .gz untouched; the edit was not written.
        assert open(gz, "rb").read() == original_bytes
        with rustfits.FITS(gz) as g:
            assert "EDIT" not in g[0].header
            np.testing.assert_array_equal(g[0].read(), data)


def test_gz_sync_refuses_tainted_buffer():
    """
    sync() likewise refuses to persist a tainted buffer, leaving the
    on-disk .gz unchanged."""
    data = np.arange(8, dtype="i4")
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "tsync.fits")
        gz = plain + ".gz"
        _write_plain_image(plain, data)
        _gzip_file(plain, gz)
        original_bytes = open(gz, "rb").read()

        f = rustfits.FITS(gz, "r+")
        f[0].header["EDIT"] = 1
        f[0]._force_taint()
        with pytest.raises(IOError):
            f.sync()
        # close() is still tainted -> also refuses; swallow it.
        with pytest.raises(IOError):
            f.close()

        assert open(gz, "rb").read() == original_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
