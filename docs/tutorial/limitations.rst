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

* **Variable-length columns with ``repeat > 1``** *(not yet)* — a
  ``2PE`` column (two heap descriptors per row, each pointing at
  its own variable-length cell) is rejected on read.  The common
  ``1PE`` shape (one descriptor per row) is fully supported.
* **Variable-length columns with TDIMn** *(not yet)* — TDIMn on
  a P/Q column would reshape each heap cell to the declared
  dims, useful for VLA-of-images.  Note the spec only allows
  ONE variable axis per cell, so fully variable ``(n, m)``
  shapes aren't expressible in FITS without padding.
* **TNULL masking on VLA columns** *(not yet)* — ``mask_null=True``
  works for fixed B/I/J/K columns but raises on VLA columns with
  a TNULL card in the header.  Read with ``mask_null=False`` and
  apply the mask yourself for now.
* **fitsio-style ``max_size=`` for VLA reads** *(by design)* —
  fitsio offers a mode that pads every VLA cell to the longest
  cell's length and returns a fixed-size N-D array.  Not
  supported; rustfits returns one ndarray per row (Object dtype).
  Build the padded array yourself if you need it.
* **ASCII table writes** *(not yet)* — :class:`~rustfits.AsciiTableHDU`
  is read-stub only (inspection accessors work; ``read()`` raises).
  Modern files use BINTABLE; if you need ASCII output, write
  through astropy.
* **``TDISPn`` on write** *(not yet)* — the display-format hint
  isn't emitted by rustfits's writers.  Add it by hand via
  ``header["TDISP1"] = ...`` if you need it; it's informational
  per the spec.

Images
------

The image surface (read, slice, write, ``extend``, ``__setitem__``,
BLANK / MaskedArray, BSCALE/BZERO, unsigned-int trick) is
feature-complete for both uncompressed and tile-compressed HDUs.
Two narrow gaps:

* **Per-tile ZBLANK column** *(not yet)* — header-level ZBLANK on
  compressed integer images is supported.  The convention also
  allows a per-tile *column*-form ZBLANK; rustfits doesn't read
  it.  Rare in practice — cfitsio typically emits the header
  form even for DITHER_2.
* **``mask_blank=True`` on quantized-float compressed reads**
  *(by design)* — the FITS spec forbids BLANK on floating-point
  arrays; NaN serves that role.  rustfits surfaces quantized
  NaN sentinels (NULL_VALUE_I32) as proper NaN in the decoded
  output; if you want a Boolean mask, do ``np.isnan(arr)``.

General
-------

* **Random groups** (``GROUPS=T``, ``PTYPEn``) *(by design)* —
  legacy format from the radio-astronomy era; vanishingly rare
  in new files.  Not on the roadmap.
* **Memory-mapped reads** *(by design)* — chunked sequential I/O
  already keeps peak RSS at ~1 MiB above the output array, so
  the motivation for mmap is weak.  If you hit a case where mmap
  would help, file an issue with the access pattern.
* **Streaming / row-iterator API** *(not yet)* — for tables that
  don't fit in RAM.  Add when a real workload prompts it.

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
