"""
rustfits.fitsio — a thin shim presenting a fitsio-style API.

Lets fitsio users get up and running on rustfits without
rewriting the file-open pattern::

    from rustfits import fitsio

    with fitsio.FITS(fname, 'rw', clobber=True) as fits:
        fits.write(data, extname='SCI', compress='RICE_1')

What the shim translates:

* ``FITS(fname, mode, clobber=...)`` — fitsio's modes ``'r'`` and
  ``'rw'`` (plus the synonyms ``'READONLY'`` / ``'READWRITE'`` /
  ``0`` / ``1``) map to rustfits's native ``'r'`` / ``'r+'`` /
  ``'w+'``.  ``clobber=True`` forces truncate-or-create
  (``'w+'``); ``clobber=False`` opens the existing file for
  read-write (``'r+'``), or creates a new one if it doesn't
  exist yet.

What does NOT need translating (works natively in rustfits):

* ``compress='RICE_1'`` / ``'GZIP_1'`` / ``'GZIP_2'`` /
  ``'HCOMPRESS_1'`` / ``'PLIO_1'`` — :class:`rustfits.FITS`'s
  write methods already accept those strings.
* Indexing (``fits[i]`` / ``fits['EXTNAME']``), iteration,
  ``len(fits)``, ``hdu.read()``, ``hdu.write()``,
  ``hdu.append()``, the ``hdu.header`` accessor, image / table
  slicing — all rustfits-native and already shaped like fitsio.
* :func:`fitsio.read` / :func:`fitsio.read_header` /
  :func:`fitsio.write` — re-exported from rustfits's
  convenience surface with the same signatures.

What is intentionally NOT shimmed:

* fitsio-specific kwargs: ``vstorage=`` (use
  :meth:`TableHDU.read` and the natural Object-dtype VLAs),
  ``case_sensitive=`` (rustfits is always case-insensitive on
  keyword lookup), ``upper=`` / ``lower=`` field-name
  renaming, ``where=`` row-filter strings.
* ``FITSHDR`` API: the object returned by ``hdu.header`` is a
  :class:`rustfits.FITSHeader`, NOT fitsio's ``FITSHDR``.
  ``h[key]`` / ``key in h`` / ``h.keys()`` / ``h.items()``
  work; ``h.records()`` and the dict-of-dicts shape don't.
* ``hdu.read_column(name)`` / ``read_columns([names])`` — use
  rustfits's column-subset syntax instead::

      arr = hdu[name][:]            # one column
      rec = hdu[[name1, name2]][:]  # multiple columns

For anything more involved, see ``docs/tutorial/migration.rst``.
"""

import os

from . import FITS as _FITS
from .convenience import read, read_header, write

__all__ = ["FITS", "read", "read_header", "write"]


_READ_MODES = frozenset({"r", "READONLY", 0})
_RW_MODES = frozenset({"rw", "READWRITE", 1})


def _resolve_mode(mode, clobber, filename):
    """
    Translate fitsio's mode + clobber to rustfits's mode string.

    fitsio's ``'rw'`` opens existing OR creates new; rustfits
    distinguishes ``'r+'`` (requires existing) from ``'w+'``
    (truncate-or-create).  ``clobber=True`` always picks ``'w+'``.
    """
    if mode in _READ_MODES:
        return "r"
    if mode in _RW_MODES:
        if clobber or not os.path.exists(filename):
            return "w+"
        return "r+"
    raise ValueError(
        f"unsupported mode {mode!r}; rustfits.fitsio.FITS accepts "
        "'r' / 'rw' (use rustfits.FITS directly for the native "
        "'r' / 'r+' / 'w+' surface)"
    )


class FITS:
    """
    fitsio-style constructor wrapping :class:`rustfits.FITS`.

    Once constructed, the wrapper forwards every attribute to the
    underlying rustfits ``FITS`` instance, so indexing, iteration,
    ``write`` / ``write_image`` / ``write_table``, etc., behave
    exactly as in native rustfits.
    """

    def __init__(self, filename, mode="r", clobber=False):
        self._fits = _FITS(filename, _resolve_mode(mode, clobber, filename))

    def __getattr__(self, name):
        return getattr(self._fits, name)

    def __getitem__(self, key):
        return self._fits[key]

    def __len__(self):
        return len(self._fits)

    def __iter__(self):
        return iter(self._fits)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._fits.close()
        return False

    def __repr__(self):
        return f"<rustfits.fitsio.FITS wrapping {self._fits!r}>"

    def close(self):
        self._fits.close()
