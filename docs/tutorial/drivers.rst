Special file drivers
=====================

Beyond a plain filesystem path, :class:`rustfits.FITS` understands a
small set of *driver* prefixes that select where the bytes live.  The
prefix is part of the filename string — the same convention cfitsio
and fitsio use — so existing muscle memory carries over.

Today the in-memory, gzip-read, and remote (``http`` / ``https``) read
drivers are implemented.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Filename
     - Backend
     - Use
   * - ``"path/to.fits"``
     - on disk
     - the default; streaming reads, ~1 MiB peak RSS
   * - ``"mem://"`` / ``"memkeep://"``
     - in memory
     - build or parse a FITS file with no disk access
   * - ``"path/to.fits.gz"``
     - in memory (gunzipped)
     - read a gzipped FITS file (read-only)
   * - ``"http://..."`` / ``"https://..."``
     - in memory (downloaded)
     - read a FITS file from a URL (read-only)

In-memory files
---------------

``mem://`` (and its alias ``memkeep://``) opens an empty FITS file
backed by an in-memory buffer instead of a disk file.  You build HDUs
into it exactly as you would on disk, then extract the finished file
with :meth:`~rustfits.FITS.to_bytes`:

.. code-block:: python

   import numpy as np
   import rustfits

   data = np.arange(12, dtype="i4").reshape(3, 4)

   with rustfits.FITS("mem://", "w+") as fits:
       fits.write_image(data)
       blob = fits.to_bytes()      # -> Python bytes

   # `blob` is a complete FITS file: send it over a socket, store it
   # in a database, hand it to astropy, or write it to disk.

The two spellings ``mem://`` and ``memkeep://`` are **aliases** — they
do the same thing.  cfitsio distinguishes them (free-the-buffer vs
keep-it on close); that distinction doesn't apply here, because the
buffer is owned by the :class:`~rustfits.FITS` object and
:meth:`~rustfits.FITS.to_bytes` copies it out regardless.  Both names
are accepted so a cfitsio/fitsio user's existing code keeps working.

Parsing bytes you already have
------------------------------

The reverse direction — you hold FITS bytes (a database blob, an HTTP
response body, ``astropy``'s serialization) and want to read them
without touching disk — is :meth:`rustfits.FITS.from_bytes`:

.. code-block:: python

   with rustfits.FITS.from_bytes(blob) as fits:
       image = fits[0].read()

``from_bytes`` **copies** the input into a private buffer, so the
returned :class:`~rustfits.FITS` is completely independent of the
original object — mutating one never affects the other.

Modes and read-only files
-------------------------

To *create* in memory, use ``FITS("mem://", "w+")`` — the buffer
starts empty.  To *read* existing bytes,
:meth:`~rustfits.FITS.from_bytes` takes ``mode="r"`` (default) or
``"r+"`` for in-place edits of the private copy; ``mode="w+"`` is
rejected, since it would discard the bytes you just passed.

One caveat: an in-memory buffer has no operating-system permission
layer, so **read-only mode is advisory** for in-memory files.  Writing
to a buffer opened ``"r"`` is not rejected the way a disk file would
be.  This is harmless — the writes only touch the private in-memory
copy, never any external bytes — but worth knowing if you rely on
``"r"`` to prevent accidental mutation.

``to_bytes`` on disk files
--------------------------

:meth:`~rustfits.FITS.to_bytes` also works on an ordinary disk-backed
file: it flushes pending writes and returns the whole file as
``bytes``.  Note this loads the entire file into memory, unlike the
streaming read paths — fine for modest files, but not how you'd read a
multi-gigabyte image.  Call it before :meth:`~rustfits.FITS.close`,
which drops the buffer.

Round-trips are byte-exact
--------------------------

A file built in memory is **byte-for-byte identical** to the same file
written to disk; the only difference is the storage backend.  So
in-memory files interoperate cleanly with astropy, fitsio, and any
other FITS reader — the bytes from :meth:`~rustfits.FITS.to_bytes` are
a valid FITS file by construction:

.. code-block:: python

   with rustfits.FITS("mem://", "w+") as fits:
       fits.write_image(data)
       blob = fits.to_bytes()

   # Writing `blob` to disk yields the same file as
   # FITS("out.fits", "w+") + write_image(data) would have.
   with open("out.fits", "wb") as fh:
       fh.write(blob)

When to use it
--------------

* **Serialize without a temp file** — produce FITS bytes to send over
  a network, store in a database, or pass to another library.
* **Parse bytes you already hold** — ``from_bytes`` reads a blob
  directly instead of spilling it to a temp file first.
* **Tests** — build fixtures in memory without touching the
  filesystem.

The trade-off is memory: the whole file lives in RAM, which gives up
rustfits's usual streaming property (peak RSS ~1 MiB above the output
array on disk reads).  That's inherent to in-memory files; for large
files, work from a path on disk.

Gzipped files
-------------

Opening a path ending in ``.gz`` reads a gzipped FITS file: rustfits
gunzips the whole file into an in-memory buffer and then parses it
exactly like any in-memory file.

.. code-block:: python

   with rustfits.FITS("image.fits.gz") as fits:   # read-only
       image = fits[0].read()

Gzipped files are **read-only** — open them with the default
``mode="r"``; ``"r+"`` and ``"w+"`` raise, because write-back
(recompress on close) is not yet implemented.  To edit a gzipped
file, read it, write a plain ``.fits``, and gzip that yourself.

A few details:

* Because a gzip stream can't be seeked and FITS needs random access,
  the *decompressed* file is held in RAM — the same caveat as
  ``mem://``.  Fine for typical files; for very large data prefer an
  uncompressed path on disk.
* :meth:`~rustfits.FITS.to_bytes` on a ``.gz``-opened file returns the
  **decompressed** bytes (the in-memory representation), not the gzip
  stream.
* Detection is by the ``.gz`` extension (case-insensitive).  cfitsio's
  ``.Z`` (LZW) and ``.zip`` whole-file formats are not supported —
  only gzip.
* The top-level :func:`rustfits.read` / :func:`rustfits.read_header`
  handle ``.gz`` paths too, since they open via
  :class:`~rustfits.FITS`.

Remote files
------------

A ``http://`` or ``https://`` URL is fetched whole and parsed in
memory — *download-then-open*:

.. code-block:: python

   url = "https://example.org/data/image.fits"
   with rustfits.FITS(url) as fits:        # read-only
       image = fits[0].read()

   # or the one-liner:
   image = rustfits.read(url)

Details:

* **Read-only.** ``"r+"`` and ``"w+"`` raise before any network
  request (there is no write-back to a URL).
* **Whole file in RAM.** The entire file is downloaded into memory and
  parsed there, so this pays the full transfer even for a one-tile
  read, and peak RSS is the file size (same caveat as ``mem://``).
  Range-based partial reads — pulling only the bytes a slice needs —
  are a planned follow-up.
* A URL whose path ends in ``.gz`` is **gunzipped** after download,
  just like a local ``.gz`` path.
* The GIL is released during the transfer, so other Python threads
  keep running while a download is in flight.
* **Schemes:** ``http`` and ``https`` only.  ``ftp://`` and cfitsio's
  ``root://`` are not supported (they need separate protocol
  libraries).  For an ``ftp`` file, download it with another tool
  first, then open the local copy.
