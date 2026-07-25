Known limitations
=================

rustfits aims for parity with astropy and fitsio on the modern
FITS feature set, but a handful of edge cases aren't yet
implemented (or are deliberately out of scope).  This page lists
them and shows the workaround where one exists.

Each item is tagged:

* **(not yet)** — on the roadmap; will land when prioritized or
  when a user prompts for it.  File an issue if you hit one.
* **(by design)** — deliberately not supported; the workaround
  is the recommended path.

Tables
------

* **TNULL masking on VLA columns** *(not yet)* — ``mask_null=True``
  works for fixed B/I/J/K columns but raises on VLA columns with
  a TNULL card in the header.  Read with ``mask_null=False`` and
  apply the mask yourself for now.
* **Variable-length columns with TDIMn** *(not yet)* — TDIMn on
  a P/Q column would reshape each heap cell to the declared
  dims, useful for VLA-of-images, but with a quirk: the spec only allows ONE
  variable axis per cell, so fully variable ``(n, m)`` shapes aren't
  expressible in FITS without padding.
* **``TDISPn`` on write** *(by design)* — the display-format hint
  isn't emitted by rustfits's writers.  Add it by hand via
  ``header["TDISP1"] = ...`` if you need it; it's informational
  per the spec.

Images
------

The image surface (read, slice, write, ``extend``, ``__setitem__``,
BLANK / MaskedArray, BSCALE/BZERO, unsigned-int trick) is
feature-complete for both uncompressed and tile-compressed HDUs.
One narrow gap:

* **Per-tile ZBLANK column** *(not yet)* — header-level ZBLANK on
  compressed integer images is supported.  The convention also
  allows a per-tile *column*-form ZBLANK; rustfits doesn't read
  it.  Rare in practice — cfitsio typically emits the header
  form even for DITHER_2.

General
-------

* **Multithreaded throughput (GIL release)** *(not yet)* — rustfits
  releases the GIL only during remote (``http`` / ``ftp``) whole-file
  downloads and the ranged-mode open probe.  The heavy CPU paths —
  tile decode/encode, large chunked I/O, checksum — currently hold
  it, so several Python threads calling into rustfits serialize on
  those.  Ranged-mode block fetches also hold it, and that one is
  deliberate, not a gap: releasing the GIL mid-read while the
  per-file lock is held would invert the lock/GIL ordering and risk
  deadlocking multithreaded callers, so other threads instead wait
  out the network round trip.  Single-threaded use is unaffected
  (the common case).  Releasing the GIL around the pure-Rust
  decode/encode spans is a targeted future change, gated on a real
  multithreaded workload — file an issue if you have one.
* **Random groups** (``GROUPS=T``, ``PTYPEn``) *(by design)* — legacy format;
  vanishingly rare in new files.  Not on the roadmap.

Files and compression
---------------------

* **Whole-file gzip holds the file in RAM** *(by design)* — a
  ``.gz`` path is decompressed into memory on open, and for a
  writable mode (``r+`` / ``w+``) the in-memory buffer is
  recompressed and written back (atomically) to the ``.gz`` path on
  :meth:`~rustfits.FITS.close` or :meth:`~rustfits.FITS.sync`.
  Because gzip isn't randomly seekable and FITS needs random access,
  the *entire* file lives in RAM while open (the same caveat as
  ``mem://``).  The new bytes reach disk at close/sync (a finalizer
  flushes a forgotten-to-close file as a safety net, but the context
  manager is the reliable path).

  More to the point, **prefer per-HDU compression over whole-file
  compression**.  Tile-compressed images (the ``compress=...``
  argument to ``create_image_hdu`` / ``write_image``) and
  compressed tables (``compress=...`` on ``create_table_hdu`` /
  ``write_table``) generally beat a whole-file ``.gz`` on every
  axis that matters:

  - **Storage** — the tile-compression conventions apply the codec
    suited to the data (RICE / HCOMPRESS / GZIP with byte-shuffle,
    optional float quantization), so they typically compress
    tighter than gzipping the raw byte stream.
  - **Memory** — a ``.gz`` must be decompressed *in full* into RAM
    before any byte is readable (gzip isn't seekable; FITS needs
    random access).  Tile compression decodes only the tiles you
    touch, keeping peak memory near the size of your slice rather
    than the whole image or table.
  - **Speed** — partial and scattered reads pull only the needed
    tiles (and the per-tile cache makes repeated access cheap),
    whereas whole-file gzip pays the full-decompress cost up front
    on every open.

  To write a compressed file, write a plain ``.fits`` and choose
  tile/HDU compression per HDU — see the compression sections of
  the image and table guides.

* **Ranged remote reads are read-only and need server ``Range``
  support** *(by design)* — ``remote="ranged"`` requires the HTTP
  server to honor ``Range`` requests; a server that ignores them
  raises at open rather than silently downloading the whole file
  you specifically asked not to download (the error suggests
  ``ranged=False``).  Ranged files are read-only, ``.gz`` URLs are
  rejected (gzip isn't seekable), and
  :meth:`~rustfits.FITS.to_bytes` raises.  Reading *most* of a
  file through ranged mode is slower than the default
  download-then-open — the bytes arrive as serialized block
  fetches, each paying a network round trip — so match the mode to
  the access pattern: small fraction → ``"ranged"``, whole file →
  default download.  See :ref:`ranged-reads`.

Cross-tool interop caveats
--------------------------

These aren't rustfits limitations per se — they're points where
the FITS ecosystem disagrees and rustfits picks one side.

* **astropy ``1PX(N)`` parser** — astropy 7.2.0 rejects bit-
  packed VLA columns (``1PX``/``1QX``) with ``VerifyError``
  during column setup.  rustfits writes spec-conforming
  ``1PX(maxbits)`` headers, but astropy can't currently read
  them; fitsio (built on cfitsio) reads them fine.  See the
  pinning test ``test_astropy_pxqx_documented_limitation``.
* **astropy ``CompImageHDU`` verify_checksum** — astropy's
  ``verify_checksum`` for compressed HDUs has its own internal
  bug (TypeError on ``_compute_checksum(None)``) that triggers
  on its own writes too.  rustfits's self-verify is correct;
  we don't cross-verify ZHECKSUM against astropy.
* **i8 (``TLONGLONG``) RICE compression** — cfitsio's encoder
  doesn't support 64-bit RICE; fitsio refuses the write.
  rustfits rejects ``compress=Rice1()`` + i8 dtype upfront and
  points at ``compress=Gzip2(...)`` instead.  Files we wrote
  with i64 RICE would be unreadable everywhere except rustfits.
