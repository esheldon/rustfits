Compression
===========

rustfits supports the FITS Tile Compression Convention for both
images (ZIMAGE) and tables (ZTABLE).  The Python surface for a
compressed HDU is the same as the uncompressed one — ``read``,
``__getitem__``, ``__setitem__``, ``write``, ``extend``,
``append`` — only the on-disk encoding differs.

This page covers turning compression on at write time, the
algorithm config objects, the ``Quantize`` config for float
images, the tile cache, and the ``repack()`` operation.

Compressed files written by rustfits are byte-exactly equivalent
to fitsio / cfitsio output on the same input (anchored by
heap-comparison tests across every algorithm), and ``funpack``
decompresses rustfits-written files to bit-exact uncompressed
form.  In the other direction, rustfits reads files written by
``fpack``, astropy, and fitsio.  See :doc:`limitations` for the
narrow caveats.

Compressed images
-----------------

Pass a compression config to ``compress=`` at create / write
time.  The five algorithm classes are
:class:`~rustfits.Gzip1`, :class:`~rustfits.Gzip2`,
:class:`~rustfits.Rice1`, :class:`~rustfits.Hcompress1`, and
:class:`~rustfits.Plio1`:

.. code-block:: python

   import numpy as np
   import rustfits

   img = np.random.randn(1024, 1024).astype("f4")

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(
           img, compress=rustfits.Gzip2(tile_shape=(128, 128)),
       )

The string alias form works too — case-insensitive, with
cfitsio synonyms accepted:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(img, compress="RICE_1")
       fits.write_image(img, compress="gzip_2")

Each algorithm class carries the parameters relevant to that
codec (``tile_shape`` everywhere; ``blocksize`` on RICE; ``scale``
and ``smooth`` on HCOMPRESS; ``level`` on the GZIPs).  Class
equality is field-wise, so a round-trip pattern works:

.. code-block:: python

   cfg = rustfits.Rice1(tile_shape=(16, 16), blocksize=64)
   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(img.astype("i4"), compress=cfg)
   with rustfits.FITS("out.fits") as fits:
       assert fits[1].compression == cfg

Reading is automatic: rustfits detects the ZIMAGE convention and
returns a :class:`~rustfits.CompressedImageHDU` (which subclasses
:class:`~rustfits.ImageHDU`, so ``isinstance(hdu, ImageHDU)`` is
True).

.. code-block:: python

   arr = rustfits.read("out.fits")     # decompresses transparently

Quantized vs lossless floats
----------------------------

Float-image compression has two modes — lossless or lossy — and
the default is lossless.  Choose with the ``quantize=`` kwarg:

.. code-block:: python

   # Default: lossless raw float bytes through GZIP.
   with rustfits.FITS("loss.fits", "w+") as fits:
       fits.write_image(img, compress=rustfits.Gzip2())

   # Lossy: quantize floats to i32 with N-sigma per quantum, then
   # compress.  Much better compression (4-10x); precision loss
   # is controlled by `level`.
   with rustfits.FITS("lossy.fits", "w+") as fits:
       fits.write_image(
           img,
           compress=rustfits.Rice1(),
           quantize=rustfits.Quantize(level=4.0, method="dither1"),
       )

``Quantize`` parameters:

* ``level`` (default 4.0) — N-sigma per quantum.  Negative values
  pin bscale directly to ``-level``.
* ``method`` — one of ``"no_dither"``, ``"dither1"`` (default,
  matches cfitsio), ``"dither2"`` (preserves NaN through a
  reserved sentinel).
* ``seed`` — ZDITHER0 (default 0 → on-disk value 1).

Lossless float compression requires ``compress=Gzip1(...)`` or
``compress=Gzip2(...)`` — Rice1, Hcompress1, and Plio1 don't
round-trip raw float bit patterns.  Integer HDUs reject
``quantize=`` regardless.

BLANK on compressed integers
----------------------------

Same surface as the uncompressed case: ``blank=`` on write,
``mask_blank=True`` on read, MaskedArray input auto-fills with
the sentinel.  See :doc:`images` for the full pattern.

Extending and modifying
-----------------------

``extend(data)`` and ``__setitem__`` both work on compressed
images.  Boundary tiles (partial last tile for ``extend``;
overlapping tiles for ``__setitem__``) are decoded, modified,
re-encoded, and appended to the heap.  Old tile blobs become
heap orphans; call ``repack()`` to reclaim them:

.. code-block:: python

   with rustfits.FITS("img.fits.fz", "r+") as fits:
       hdu = fits[1]
       hdu.extend(np.zeros((10,) + hdu.shape[1:], dtype=hdu.dtype))
       hdu[100, 100] = 0
       hdu.repack()

For quantized-float HDUs, ``__setitem__`` reuses the existing
per-tile bscale/bzero/dither seed so unchanged pixels in a
modified tile round-trip bit-exactly — no compounding
quantization loss.

Compressed tables
-----------------

Tables compress the same way: pass ``compress=`` to the table
writer.  ``True`` picks cfitsio's per-dtype defaults; a string or
algorithm-class instance applies one algorithm to every column; a
dict overrides per column:

.. code-block:: python

   import numpy as np
   import rustfits

   cat = np.zeros(1_000_000, dtype=[
       ("ra", "f8"), ("dec", "f8"), ("flag", "i4"),
   ])

   # cfitsio's per-dtype defaults — fine for most cases.
   with rustfits.FITS("cat.fits.fz", "w+") as fits:
       fits.write_table(cat, compress=True)

   # One algorithm everywhere.
   with rustfits.FITS("cat.fits.fz", "w+") as fits:
       fits.write_table(cat, compress="GZIP_2")

   # Per-column overrides.
   with rustfits.FITS("cat.fits.fz", "w+") as fits:
       fits.write_table(
           cat,
           compress={"ra": rustfits.Gzip2(),
                     "dec": rustfits.Gzip2(),
                     "flag": rustfits.Rice1()},
       )

The tile size (``ztilelen``) defaults to roughly 10 MB worth of
rows per tile.  Pass ``ztilelen=N`` to override.

Reading a compressed table is the same as reading an
uncompressed one — rustfits detects the ZTABLE convention and
returns a :class:`~rustfits.CompressedTableHDU` (which subclasses
:class:`~rustfits.TableHDU`).  All the row / column / subset /
``__setitem__`` patterns from :doc:`tables` work the same way.

Tile cache
----------

Decoded tiles are held in a bytes-bound LRU cache so repeat
reads of overlapping regions don't redecompress.  Default
capacity is 32 MiB per HDU; tune per HDU:

.. code-block:: python

   hdu.tile_cache_size            # current capacity in bytes
   hdu.set_tile_cache_size(0)     # disable
   hdu.set_tile_cache_size(256 * 1024 * 1024)   # 256 MiB
   hdu.tile_cache_used            # bytes currently held
   hdu.clear_tile_cache()         # drop entries, keep capacity

Reclaiming heap orphans
-----------------------

For both compressed images and compressed tables, mutations
(``__setitem__``, ``extend``, ``append`` with merge-into-last-
tile, VLA writes) leave orphaned bytes in the heap.  Call
``repack()`` to rebuild the heap with only live data.  If the
HDU is the last on disk the file shrinks via ``set_len``;
otherwise later HDUs shift backward.

.. code-block:: python

   with rustfits.FITS("img.fits.fz", "r+") as fits:
       fits[1].repack()
