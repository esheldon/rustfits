Special file drivers
=====================

Beyond a plain filesystem path, :class:`rustfits.FITS` understands a
small set of *driver* prefixes that select where the bytes live.  The
prefix is part of the filename string — the same convention cfitsio
and fitsio use — so existing muscle memory carries over.

Today the in-memory, gzip (read + write-back), and remote
(``http`` / ``https`` / ``ftp`` / ``ftps``) read drivers are
implemented.

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

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
     - read+write a gzipped FITS file; ``r+`` / ``w+`` recompress on close
   * - ``"http://..."`` / ``"https://..."``
     - in memory (downloaded)
     - read a FITS file from a URL (read-only)
   * - ``"ftp://..."`` / ``"ftps://..."``
     - in memory (downloaded)
     - read a FITS file from an FTP server (read-only)

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

Opening a path ending in ``.gz`` supports reading and writing a gzipped FITS
file. When reading, rustfits gunzips the whole file into an in-memory buffer
and then parses it exactly like any in-memory file.

.. code-block:: python

   with rustfits.FITS("image.fits.gz") as fits:   # read-only
       image = fits[0].read()

Writing is supported too: open a ``.gz`` with ``"w+"`` (truncate /
create) or ``"r+"`` (edit in place).  rustfits builds the file in the
in-memory buffer and, when you :meth:`~rustfits.FITS.close` it,
recompresses the buffer and writes the gzip stream back to the
``.gz`` path.

.. code-block:: python

   with rustfits.FITS("image.fits.gz", "w+") as fits:
       fits.write_image(image)          # recompressed + saved on close

   with rustfits.FITS("image.fits.gz", "r+") as fits:
       fits[0].header["HISTORY"] = "edited in place"

The write-back is **atomic**: rustfits compresses to a temporary file
in the same directory and renames it over the target, so an
interrupted write (out of disk space, I/O error) leaves the original
``.gz`` intact rather than half-written.

A few details:

* The new bytes reach disk **at close** (or :meth:`~rustfits.FITS.sync`
  — see below).  As a safety net, if you forget to close a written
  ``.gz``, a finalizer flushes it when the object is garbage-collected;
  still, prefer the context manager so errors surface and timing is
  deterministic.
* :meth:`~rustfits.FITS.sync` forces the current buffer to disk
  durably mid-session (recompress + atomic write + ``fsync``), so you
  don't have to close to checkpoint.
* Opening ``r+``/``w+`` behaves like a plain-disk open: ``w+`` creates
  and claims the file immediately, and a permission/path error is
  raised at ``FITS(...)`` time, not deferred to close.
* A ``.gz`` opened ``r+`` is rewritten on close **only if you actually
  mutated it** — opening to read leaves the on-disk file (bytes,
  mtime) untouched.
* Because a gzip stream can't be seeked and FITS needs random access,
  the *decompressed* file is held in RAM — the same caveat as
  ``mem://``.  Fine for typical files; for very large data prefer an
  uncompressed path on disk.  Per-HDU (tile) compression is almost
  always the better choice than a whole-file ``.gz`` — see
  :doc:`limitations`.
* Multi-member gzip streams are decoded in full (not truncated to the
  first member).
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

A ``http://``, ``https://``, ``ftp://``, or ``ftps://`` URL is fetched
whole and parsed in memory — *download-then-open*:

.. code-block:: python

   url = "https://example.org/data/image.fits"
   with rustfits.FITS(url) as fits:        # read-only
       image = fits[0].read()

   # or the one-liner:
   image = rustfits.read(url)

   # FTP works the same way (anonymous login by default):
   image = rustfits.read("ftp://archive.example.org/pub/vela.fits")

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
* **HTTP schemes:** ``http`` and ``https`` (TLS handled by rustls).
* **FTP schemes:** ``ftp`` and ``ftps`` (explicit ``AUTH TLS``).
  Login is **anonymous** unless the URL carries credentials
  (``ftp://user:pass@host/path``); the port defaults to 21.  Transfers
  are forced to binary mode so FITS bytes aren't mangled.
* cfitsio's ``root://`` / ``gsiftp://`` are **not** supported — see
  *Deferred drivers* below.

Deferred drivers
----------------

cfitsio supports a few more storage backends that rustfits has **not**
implemented yet.  They aren't hard to add on top of the existing
backend abstraction — they're deferred for lack of a concrete user
need, not for technical reasons.  **If you have a use case for any of
these, please open an issue** — a real request is exactly what moves
one off this list.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Driver
     - Status / workaround
   * - ``stream://`` (stdin / stdout, ``-``)
     - Deferred.  In Python the byte API already covers pipelines:
       ``rustfits.FITS.from_bytes(sys.stdin.buffer.read())`` to read,
       ``sys.stdout.buffer.write(fits.to_bytes())`` to write.
   * - ``shmem://`` (POSIX shared memory)
     - Deferred.  To share a file between processes via RAM today,
       pair :meth:`~rustfits.FITS.to_bytes` / :meth:`~rustfits.FITS.from_bytes`
       with the standard library's
       :class:`multiprocessing.shared_memory.SharedMemory` (one copy
       in and out; true zero-copy shared access is the harder feature
       that's deferred).
   * - ``root://`` (XRootD)
     - Deferred.  Needs the XRootD client library.  Download the file
       with an XRootD tool first, then open the local copy.
   * - ``gsiftp://`` (GridFTP)
     - Deferred.  Needs a GridFTP client.  Fetch with a grid tool
       first, then open the local copy.

Everything above plugs into the same internal backend abstraction the
in-memory, gzip, and remote drivers already use, so adding one is
mostly a matter of wiring up the byte source.
