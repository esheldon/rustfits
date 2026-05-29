"""
In-memory FITS files: the mem:// / memkeep:// drivers plus the
to_bytes() / from_bytes() byte-I/O pair.

The in-memory backend is the Storage::Mem variant; these tests
exercise create-in-memory -> to_bytes and from_bytes -> read, and
confirm the in-memory path is byte-exact with the disk path.  For
mutation tests the "reopen" check is to_bytes() -> from_bytes()
(the in-memory analog of close + reopen).
"""

import os
import tempfile
import numpy as np
import pytest
import rustfits


def test_mem_image_roundtrip():
    """
    Create an image in an empty mem:// file, extract with
    to_bytes(), and read it back via from_bytes()."""
    data = np.arange(3 * 4, dtype="i4").reshape(3, 4)
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data)
        np.testing.assert_array_equal(f[0].read(), data)  # same-handle
        blob = f.to_bytes()

    assert isinstance(blob, bytes)
    with rustfits.FITS.from_bytes(blob) as f:  # "reopen"
        np.testing.assert_array_equal(f[0].read(), data)


def test_mem_table_roundtrip():
    """
    A structured-array table written into a mem:// file round-trips
    through to_bytes() / from_bytes()."""
    rec = np.array(
        [(1, 1.5), (2, 2.5), (3, 3.5)],
        dtype=[("id", "i4"), ("x", "f8")],
    )
    with rustfits.FITS("mem://", "w+") as f:
        f.write_table(rec)
        blob = f.to_bytes()

    with rustfits.FITS.from_bytes(blob) as f:
        got = f[1].read()
    np.testing.assert_array_equal(got["id"], rec["id"])
    np.testing.assert_array_equal(got["x"], rec["x"])


def test_mem_and_memkeep_are_aliases():
    """
    mem:// and memkeep:// are the same backend — identical bytes for
    the same writes."""
    data = np.arange(20, dtype="i2").reshape(4, 5)
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data)
        a = f.to_bytes()
    with rustfits.FITS("memkeep://", "w+") as f:
        f.write_image(data)
        b = f.to_bytes()
    assert a == b


def test_mem_bytes_match_disk():
    """
    An in-memory file is byte-exact with the same file written to
    disk — the Storage seam is the only difference."""
    data = np.arange(6 * 7, dtype="f4").reshape(6, 7)
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data)
        mem_bytes = f.to_bytes()

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "d.fits")
        with rustfits.FITS(p, "w+") as f:
            f.write_image(data)
        with open(p, "rb") as fh:
            disk_bytes = fh.read()
    assert mem_bytes == disk_bytes


def test_mem_multi_hdu():
    """
    Multiple HDUs written into one mem:// file all survive the
    to_bytes() / from_bytes() round trip in order."""
    img = np.arange(12, dtype="i4").reshape(3, 4)
    rec = np.array([(10,), (20,)], dtype=[("n", "i8")])
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(img)
        f.write_table(rec)
        blob = f.to_bytes()

    with rustfits.FITS.from_bytes(blob) as f:
        assert len(f.hdus) == 2
        np.testing.assert_array_equal(f[0].read(), img)
        np.testing.assert_array_equal(f[1].read()["n"], rec["n"])


def test_mem_compressed_image_roundtrip():
    """
    A tile-compressed image created in memory reads back correctly
    via from_bytes()."""
    data = np.arange(8 * 8, dtype="i4").reshape(8, 8)
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data, compress=rustfits.Gzip1(tile_shape=(4, 4)))
        blob = f.to_bytes()

    with rustfits.FITS.from_bytes(blob) as f:
        hdu = f[1]
        assert isinstance(hdu, rustfits.ImageHDU)
        np.testing.assert_array_equal(hdu.read(), data)


def test_mem_empty_has_no_hdus():
    """
    A freshly created empty mem:// file has zero HDUs and yields
    empty bytes — same as a fresh w+ disk file."""
    with rustfits.FITS("mem://", "w+") as f:
        assert len(f.hdus) == 0
        assert f.to_bytes() == b""


def test_mem_setitem_then_to_bytes():
    """
    In-place __setitem__ on a mem image is reflected both in a
    same-handle read and in the to_bytes() output."""
    data = np.zeros((4, 4), dtype="i4")
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data)
        f[0][1, 2] = 99
        assert f[0][1, 2] == 99  # same-handle
        blob = f.to_bytes()

    with rustfits.FITS.from_bytes(blob) as f:  # "reopen"
        assert f[0][1, 2] == 99


def test_from_bytes_is_independent_of_source():
    """
    from_bytes copies the input; mutating the returned FITS (r+)
    does not change the original bytes object, and re-parsing the
    original bytes does not see the mutation."""
    data = np.arange(9, dtype="i4").reshape(3, 3)
    with rustfits.FITS("mem://", "w+") as f:
        f.write_image(data)
        original = f.to_bytes()

    with rustfits.FITS.from_bytes(original, "r+") as f:
        f[0].header["NEWKEY"] = 7
        assert f[0].header["NEWKEY"] == 7  # same-handle
        mutated = f.to_bytes()

    # Re-parsing the original bytes does not see the mutation (the
    # in-memory file worked on a private copy)...
    with rustfits.FITS.from_bytes(original) as f:
        assert "NEWKEY" not in f[0].header
    # ...but the mutated copy carries the new card.
    with rustfits.FITS.from_bytes(mutated) as f:
        assert f[0].header["NEWKEY"] == 7


def test_from_bytes_reads_disk_written_bytes():
    """
    from_bytes parses bytes produced by a disk write (the common
    'I already have FITS bytes' case)."""
    rec = np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "d.fits")
        with rustfits.FITS(p, "w+") as f:
            f.write_table(rec)
        with open(p, "rb") as fh:
            blob = fh.read()

    with rustfits.FITS.from_bytes(blob) as f:
        got = f[1].read()
    np.testing.assert_array_equal(got["a"], rec["a"])
    np.testing.assert_array_equal(got["b"], rec["b"])


def test_from_bytes_reads_fitsio_written_bytes():
    """
    from_bytes parses bytes produced by an external tool (fitsio)."""
    import fitsio

    data = np.arange(5 * 6, dtype="i4").reshape(5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "f.fits")
        with fitsio.FITS(p, "rw", clobber=True) as ff:
            ff.write(data)
        with open(p, "rb") as fh:
            blob = fh.read()

    with rustfits.FITS.from_bytes(blob) as f:
        for hdu in f.hdus:
            if hdu.has_data:
                np.testing.assert_array_equal(hdu.read(), data)
                break
        else:
            raise AssertionError("no HDU with data found")


def test_to_bytes_on_disk_file():
    """
    to_bytes() also works on a disk-backed file: it returns the same
    bytes as reading the file directly."""
    data = np.arange(10, dtype="i2")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "d.fits")
        with rustfits.FITS(p, "w+") as f:
            f.write_image(data)
            got = f.to_bytes()
        with open(p, "rb") as fh:
            assert got == fh.read()


def test_from_bytes_wplus_rejected():
    """
    from_bytes rejects 'w+' (it would discard the provided bytes)."""
    with pytest.raises(ValueError):
        rustfits.FITS.from_bytes(b"\x00" * 2880, "w+")


def test_from_bytes_bad_mode_rejected():
    """
    from_bytes rejects an unsupported mode string."""
    with pytest.raises(ValueError):
        rustfits.FITS.from_bytes(b"", "bogus")


def test_mem_invalid_mode_rejected():
    """
    Opening mem:// with an unsupported mode raises, same as disk."""
    with pytest.raises(Exception):
        rustfits.FITS("mem://", "bogus")


def test_to_bytes_after_close_raises():
    """
    to_bytes() after close() raises — the buffer is dropped on close,
    so extract before closing."""
    data = np.arange(4, dtype="i4")
    f = rustfits.FITS("mem://", "w+")
    f.write_image(data)
    f.close()
    with pytest.raises(Exception):
        f.to_bytes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
