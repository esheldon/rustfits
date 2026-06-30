"""
Shared pytest configuration for the rustfits test suite.

On Windows, disable astropy's default memory-mapping for FITS reads.

``astropy.io.fits.open()`` memory-maps the data section by default.  A
test that binds ``hdul[1].data`` (or any data view) to a variable keeps
a reference to that mmap, which holds the underlying OS file handle open
even after the ``HDUList`` is closed.  On Windows you cannot delete a
file that still has an open handle, so the enclosing
``tempfile.TemporaryDirectory()`` cleanup fails with
``PermissionError: [WinError 32] ... being used by another process``.
Unix lets you unlink an open file, so this only bites on Windows.

Disabling memmap loads the data into ordinary in-memory arrays instead
(identical values and dtypes), so the file handle is released when the
``HDUList`` closes and the temp dir can be removed.  No effect on the
data the tests assert against; gated to Windows so Linux/macOS keep the
default mmap fast path.
"""

import sys

if sys.platform == "win32":
    try:
        from astropy.io import fits as _astropy_fits

        _astropy_fits.conf.use_memmap = False
    except ImportError:
        # astropy not installed on this platform/leg; nothing to do.
        pass
