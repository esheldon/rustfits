"""Phase 2d: tainted-header state on mid-write I/O failure.

When a write_all or flush inside rewrite_header_to_disk fails, the on-disk
file may be partially overwritten.  We set a per-file taint flag that
causes subsequent reads and writes to refuse with a clear, actionable
error message ("close and reopen to recover").  Reading and writing on a
freshly-reopened file works again — the taint is per-handle, not
per-file-on-disk.

Actually triggering a mid-write OS failure (ENOSPC, hardware error, etc.)
is hard to do deterministically on a real filesystem.  The HDU exposes an
underscored `_force_taint()` method for tests to flip the flag directly.
That gives us coverage of the rejection-behavior contract without needing
to construct a real I/O failure.  The trigger code path itself
(write_all/flush errors → set tainted) is reviewed by inspection.

Pre-I/O failures (slack overflow, missing file, etc.) MUST NOT taint:
those leave the on-disk file untouched and are not a recovery scenario.
"""

import os
import tempfile
import contextlib

import pytest

import rustfits


@contextlib.contextmanager
def _new_file(shape=(4, 6), dtype="i4"):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


# -------- overflow does not taint --------


def test_overflow_does_not_taint():
    """Header overflow is a pre-I/O failure (caught by the slack check
    before any bytes touch the disk).  It must NOT set the taint flag."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h = fits[0].header
            initial = len(h.cards)
            block_count = (initial + 35) // 36
            slots_free = block_count * 36 - initial
            for i in range(slots_free):
                h[f"PAD{i:04d}"] = i
            with pytest.raises(ValueError):
                h["OVERFLOW"] = 1   # overflow

            # Subsequent reads still work — file is not tainted.
            assert "PAD0000" in h
            assert h["PAD0000"] == 0
            # And subsequent legal writes work too.
            del h["PAD0000"]
            h["PAD0000"] = 99
            assert h["PAD0000"] == 99


# -------- _force_taint triggers refusal --------


def test_tainted_reads_refuse():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            fits[0]._force_taint()
            with pytest.raises(IOError) as excinfo:
                _ = fits[0].header["EXPTIME"]
            msg = str(excinfo.value)
            assert "indeterminate state" in msg or "inconsistent" in msg
            assert "reopen" in msg.lower()


def test_tainted_writes_refuse():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            with pytest.raises(IOError):
                fits[0].header["EXPTIME"] = 5
            with pytest.raises(IOError):
                del fits[0].header["EXPTIME"]
            with pytest.raises(IOError):
                fits[0].header.update({"X": 1})
            with pytest.raises(IOError):
                fits[0].header.add_comment("nope")


def test_tainted_image_read_refuses():
    """The taint check lives in HDU.header_snapshot, which image reads
    also go through — so image I/O is rejected too."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            with pytest.raises(IOError):
                _ = fits[0].read()


def test_tainted_image_write_refuses():
    import numpy as np
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            data = np.zeros((4, 6), dtype="i4")
            with pytest.raises(IOError):
                fits[0].write(data)


def test_tainted_iter_and_contains_refuse():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            with pytest.raises(IOError):
                list(fits[0].header)
            with pytest.raises(IOError):
                _ = "EXPTIME" in fits[0].header


# -------- edit() refuses on a tainted file --------


def test_tainted_edit_refuses_on_entry():
    """Starting an edit batch on a tainted file should fail at edit()
    (which goes through snapshot()), not silently allow staging."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            with pytest.raises(IOError):
                fits[0].header.edit()


# -------- taint shared across views of the same file --------


def test_taint_visible_through_other_views():
    """Two FITSHeader views from the same HDU share the same taint flag."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            h1 = fits[0].header
            h2 = fits[0].header
            fits[0]._force_taint()
            with pytest.raises(IOError):
                _ = h1["BITPIX"]
            with pytest.raises(IOError):
                _ = h2["BITPIX"]


# -------- recovery via reopen --------


def test_reopen_after_taint_works():
    """Tainting is per-handle.  A fresh FITS object on the same file
    starts with a clean flag and operates normally."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["EXPTIME"] = 5
            fits[0]._force_taint()
            with pytest.raises(IOError):
                _ = fits[0].header["EXPTIME"]
        # Reopen — fresh taint flag, file content intact.
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["EXPTIME"] == 5


# -------- diagnostic message ergonomics --------


def test_diagnostic_message_is_actionable():
    """The error message should tell the user what's wrong AND how to
    recover (close + reopen)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0]._force_taint()
            with pytest.raises(IOError) as excinfo:
                _ = fits[0].header["BITPIX"]
            msg = str(excinfo.value).lower()
            assert "reopen" in msg
            # And it's an IOError, not a generic ValueError, since the
            # underlying cause is an I/O failure.
            assert isinstance(excinfo.value, IOError)
