Migrating from fitsio (or astropy)
==================================

rustfits is the modern successor to `fitsio
<https://github.com/esheldon/fitsio>`_ — same primary author, written in
Rust+PyO3 with a cleaner Python surface and significantly better performance
for many workloads.  fitsio remains stable and installable indefinitely; if
your existing pipeline works, you don't need to migrate.  This page is for
users who *want* to migrate, or for new code that wants the modern API.
See :ref:`why-a-new-package` toward the bottom for the longer rationale.

If you're coming from ``astropy.io.fits``, skip ahead to
:ref:`from-astropy` — the differences look different there.

What's the same
---------------

The high-level shape is deliberately close to fitsio:

* Open a file with ``rustfits.FITS(path)``; use as a context
  manager.
* Index HDUs with ``fits[i]`` (integer position) or
  ``fits["name"]`` (EXTNAME, case-insensitive).
* Read with ``hdu.read()``; subset tables with
  ``hdu.read(columns=..., rows=...)``.
* Top-level ``rustfits.read(path)`` and
  ``rustfits.read_header(path)`` mirror fitsio's
  ``fitsio.read`` / ``fitsio.read_header``.
* Accessing column and row subsets on the hdu: ``hdu[rows]``,
  ``hdu[colnames][rows]``

The big idea — "open, index, read" — translates one-for-one.

What's different
----------------

Two things are worth a careful look before migrating: headers and
compression kwargs.

Headers
~~~~~~~

This is the biggest difference and the one most likely to surface
in migrated code.

**fitsio** treats headers as a sequence of records.  A header
record carries name, value, and comment together; iterating a
header yields the records; ``hdr[key]`` returns the value and
``hdr.get_comment(key)`` is a separate call.

**rustfits** treats headers as a parse-on-demand view of the
underlying cards.  ``hdr[key]`` returns the value;
``hdr.comment_of(key)`` returns the comment; ``hdr.keys()`` and
``key in hdr`` follow dict semantics.  See :doc:`headers` for the
full picture.

Side-by-side:

.. code-block:: python

   # fitsio
   import fitsio
   with fitsio.FITS(path) as f:
       hdr = f[0].read_header()
       exptime = hdr["exptime"]
       comment = hdr.get_comment("exptime")
       for record in hdr.records():
           print(record["name"], record["value"], record["comment"])

   # rustfits
   import rustfits
   with rustfits.FITS(path) as fits:
       hdr = fits[0].header                      # attribute, not method
       exptime = hdr["exptime"]                  # case-insensitive
       comment = hdr.comment_of("exptime")
       for key in hdr.keys():
           print(key, hdr[key], hdr.comment_of(key))

Two specific points worth knowing:

* ``hdu.header`` is an **attribute**, not a method
  (``read_header()`` doesn't exist on the HDU).  The top-level
  :func:`rustfits.read_header` exists for the
  "just-the-header-from-a-filename" use case.
* The returned :class:`rustfits.FITSHeader` survives the file
  close.  Read-only access works after the ``with`` block exits.

Compression kwargs
~~~~~~~~~~~~~~~~~~

fitsio takes flat kwargs:

.. code-block:: python

   fits.write(data, compress="GZIP_1", tile_dims=(100, 100),
              qlevel=4.0, qmethod=1)

rustfits takes structured config objects:

.. code-block:: python

   fits.write_image(
       data,
       compress=rustfits.Gzip1(tile_shape=(100, 100)),
       quantize=rustfits.Quantize(level=4.0, method="dither1"),
   )

One config class per algorithm: :class:`~rustfits.Gzip1`,
:class:`~rustfits.Gzip2`, :class:`~rustfits.Rice1`,
:class:`~rustfits.Hcompress1`, :class:`~rustfits.Plio1`.  Each
takes the parameters that algorithm actually needs (RICE has
``blocksize``, HCOMPRESS has ``scale`` and ``smooth``, GZIP has
``level``, everyone has ``tile_shape`` and ``heap_format``).

If you want the cfitsio defaults without thinking about which
algorithm, ``compress=True`` (table) or ``compress="GZIP_1"``
(string alias) both work.  See :doc:`compression` for the full
surface.

Quantization split out into its own ``Quantize`` config means
the default is now **lossless** float compression — call out
``Quantize(level=N)`` explicitly when you want lossy.  fitsio's
default is the opposite (lossy with cfitsio's defaults); be
deliberate when porting.

A thin fitsio-style shim
------------------------

If you want to try out rustfits on an existing codebase, rustfits ships a thin
shim that might get you started.  It works for basic patterns

.. code-block:: python

   from rustfits import fitsio

   with fitsio.FITS(path, "rw", clobber=True) as fits:
       fits.write_image(data, extname="sci", compress="RICE_1")

The shim translates fitsio's ``mode='r' | 'rw'`` (plus the
``'READONLY'`` / ``'READWRITE'`` / ``0`` / ``1`` synonyms) and the
``clobber=`` kwarg to rustfits's native modes; everything else is
the real :class:`rustfits.FITS` object, so indexing, iteration,
``hdu.read()`` / ``hdu.write()``, the ``hdu.header`` accessor and
so on behave as rustfits-native, *not* as fitsio.

What it does NOT translate: ``vstorage=`` / ``case_sensitive=`` /
``upper=`` / ``lower=`` / ``where=`` kwargs (use the recipes
below), and ``hdu.header`` returns a :class:`rustfits.FITSHeader`
rather than fitsio's ``FITSHDR`` (``hdr[key]`` and ``key in hdr``
work; ``hdr.records()`` does not — see :doc:`headers`).  The
shim is deliberately narrow — it gets you past the constructor
and onto the rustfits surface; the rest of this page covers 
porting strategies and additional differences.

Common porting recipes
----------------------

Open + read
~~~~~~~~~~~

.. code-block:: python

   # fitsio
   data = fitsio.read(path, ext=1)

   # rustfits — identical
   data = rustfits.read(path, ext=1)

Open + read with row / column subset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fitsio exposes ``rows=`` / ``columns=`` on the top-level
``fitsio.read``.  rustfits made ``rustfits.read`` minimal —
open the file explicitly for these:

.. code-block:: python

   # fitsio
   tab = fitsio.read(path, ext=1, columns=["ra", "dec"], rows=[0, 5, 10])

   # rustfits
   with rustfits.FITS(path) as fits:
       tab = fits[1].read(columns=["ra", "dec"], rows=[0, 5, 10])

Write an image to a fresh file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # fitsio
   with fitsio.FITS(path, "rw", clobber=True) as f:
       f.write(data, extname="SCI")

   # rustfits — explicit handle for type-specific kwargs
   with rustfits.FITS(path, "w+") as fits:
       fits.write_image(data, extname="sci")

   # rustfits — minimal one-liner, auto-dispatches by data type
   rustfits.write(path, data, extname="sci")

Note ``"w+"`` is the rustfits equivalent of fitsio's
``"rw" + clobber=True``.  Use ``"r+"`` to append to an
existing file.

Write a table to a fresh file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # fitsio
   with fitsio.FITS(path, "rw", clobber=True) as f:
       f.write(table, extname="cat")

   # rustfits — explicit, gets table-side kwargs
   with rustfits.FITS(path, "w+") as fits:
       fits.write_table(table, extname="cat")

   # rustfits — minimal, auto-dispatches by data type
   rustfits.write(path, table, extname="cat")

Compressed image
~~~~~~~~~~~~~~~~

.. code-block:: python

   # fitsio
   with fitsio.FITS(path, "rw", clobber=True) as f:
       f.write(data, compress="RICE_1", tile_dims=(100, 100))

   # rustfits
   with rustfits.FITS(path, "w+") as fits:
       fits.write_image(
           data,
           compress=rustfits.Rice1(tile_shape=(100, 100)),
       )

Lossy float compression
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # fitsio
   with fitsio.FITS(path, "rw", clobber=True) as f:
       f.write(data, compress="GZIP_2", qlevel=4.0)

   # rustfits — quantize MUST be explicit; default is lossless
   with rustfits.FITS(path, "w+") as fits:
       fits.write_image(
           data,
           compress=rustfits.Gzip2(),
           quantize=rustfits.Quantize(level=4.0),
       )

Read header only
~~~~~~~~~~~~~~~~

.. code-block:: python

   # fitsio
   hdr = fitsio.read_header(path)               # primary
   hdr = fitsio.read_header(path, ext=1)        # by index

   # rustfits — identical surface
   hdr = rustfits.read_header(path)
   hdr = rustfits.read_header(path, ext=1)

What rustfits doesn't have
--------------------------

For an explicit list with workarounds, see :doc:`limitations`.
Highlights for fitsio users:

* **fitsio's ``vstorage="object"`` vs ``"fixed"``.**  fitsio
  offers a mode where each variable-length cell becomes a
  fixed-size N-D array padded to the longest cell.  rustfits
  always returns one ndarray per row (Object dtype).  Build
  the padded form yourself if needed.
* **fitsio's ``case_sensitive=True`` column lookup.**  rustfits
  is always case-insensitive on column names; case is
  preserved on disk but lookup folds case.

What fitsio doesn't have that rustfits does
-------------------------------------------

These are the big ergonomic and capability wins worth the
migration:

**Full ``__getitem__`` / ``__setitem__`` surface on every HDU.**

rustfits has symmetry between read and write, letting you modify in place as
with a numpy array — slice an image, replace a column, patch individual rows or
ranges, all without rewriting the file:

.. code-block:: python

   # Image: cutouts and in-place pixel writes.
   stamp = hdu[100:200, 50:150]                   # read just the cutout
   hdu[100:200, 50:150] = 0                       # zero out that region
   hdu[42, :] = np.arange(hdu.shape[1])           # rewrite one row

   # Table: every shape of row / column / cell write.
   hdu[5] = new_record                            # replace one row
   hdu[100:200] = new_rows                        # slice replace
   hdu[[1, 3, 5]] = new_rows                      # fancy-row replace
   hdu["flag"] = np.zeros(len(hdu), dtype="i4")   # whole-column replace
   hdu[["ra", "dec"]] = subset_array              # multi-column replace
   hdu["ra"][5] = 123.4                           # single cell
   hdu["ra"][[0, 1, 2]] = [10.0, 11.0, 12.0]      # rows of one column
   hdu[["ra", "dec"]][0:10] = sub_array           # rows of a column subset

All of these work on both fixed and variable-length columns,
on both compressed and uncompressed tables.  See :doc:`tables`
and :doc:`images` for the full picture.

**Tile-compressed table writes (ZTABLE).**

fitsio reads tile-compressed tables but doesn't write them.
rustfits writes them with the full ``__setitem__`` /
``append`` / ``repack`` surface, including VLA columns.

**Significantly faster compressed reads.**

Up to 40× faster than fitsio on small-chunk workloads, 3.2×
faster on large-chunk workloads — see the "Performance"
discussion in ``CLAUDE.md`` for the benchmark setup.  fitsio's
large-array reads are already fast; the wins are mostly in
the access patterns that decompress only the overlapping
tiles.

.. _why-a-new-package:

Why a new package
-----------------

The user-visible wins above are the surface of a deeper
project-level shift.  Two things stand behind them.

**Rust is not just a language choice.**  Its memory-safety
guarantees rule out entire classes of bugs that affect
binary-format parsers written in C — buffer overflows,
use-after-free, data races between threads, the lot.  The
crate ecosystem matters just as much: low-level building
blocks like gzip framing, deflate compression, and CRC32
land as one-line Cargo dependencies rather than vendored C
libraries or system-library version hunting.  That makes it
cheap to use well-tested, broadly-deployed implementations
and to swap them out as the ecosystem evolves.

**rustfits's Rust backend is developed alongside the Python
wrapper by the same authors.**  Implementation improvements —
bug fixes, new compression features, performance work — land
rapidly and on demand.  fitsio, by contrast, depends on the
cfitsio C library for low-level FITS handling.  cfitsio is
mature, widely deployed, and appropriately conservative — its
maintainers reasonably prioritize stability over new features,
which is the right call for a foundational dependency of the
wider ecosystem.  The practical effect is just an impedance
mismatch: features that wouldn't fit cfitsio's stability
mandate can still land in rustfits when there's a clear need.

None of this means you have to migrate.  fitsio remains stable
and installable indefinitely; existing pipelines do not need
to change.  But for new code, or for projects that want
features cfitsio doesn't expose, rustfits is the place to go.

.. _from-astropy:

From astropy.io.fits
--------------------

If you're coming from astropy, the conceptual gap is bigger
because astropy's surface uses different abstractions.  The
quick translations:

* ``fits.open(path)`` → ``rustfits.FITS(path)``
* ``hdul[i].data`` → ``hdul[i].read()``  (rustfits doesn't
  auto-load data on open)
* ``hdul[i].header`` → ``hdul[i].header``  (same attribute
  name; rustfits's :class:`FITSHeader` has a slightly
  different surface — see :doc:`headers`)
* ``Table.read(path)`` → ``rustfits.read(path)`` for the
  simple case; open the file for finer control
* ``CompImageHDU(...)`` → :meth:`FITS.write_image` with the
  ``compress=`` config object

The biggest mental shift coming from astropy: rustfits doesn't
build Python objects for every card.  The header is a
dict-like view over the raw card list, and the data is read
on demand.  This is closer to fitsio's model than astropy's.
