Tables
======

This page covers :class:`~rustfits.TableHDU` — writing tables
from numpy structured arrays, reading rows and columns, the
column-subset objects, in-place edits, append and schema-edit
operations, and variable-length (VLA) columns.

For tile-compressed tables, see :doc:`compression`.  The Python
surface is the same; the on-disk encoding differs.

Tables written by rustfits — fixed columns, variable-length
columns (numeric and string ``PA``), and bit-packed ``X`` /
``PX`` columns — round-trip bit-exactly through astropy and
fitsio (one astropy parser limitation on ``PX``/``QX`` is noted
in :doc:`limitations`).

Writing a table
---------------

The shortest path is :func:`rustfits.write`, which auto-detects
a table from a structured ndarray or ``{name: array}`` dict:

.. code-block:: python

   import numpy as np
   import rustfits

   cat = np.zeros(1000, dtype=[
       ("ra", "f8"), ("dec", "f8"), ("flag", "i4"),
   ])
   cat["ra"] = np.random.uniform(0, 360, size=1000)
   cat["dec"] = np.random.uniform(-90, 90, size=1000)

   rustfits.write("cat.fits", cat)

For multi-HDU files, or for the list-of-arrays form with a
separate ``names=[...]`` argument, or for any type-specific
knobs (``compress=``, ``units=``, ``var_dtypes=``,
``bit_columns=``, ...), open :class:`~rustfits.FITS` directly
and use :meth:`~rustfits.FITS.write_table`:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_table(cat, extname="cat")
       fits.write_table({"x": np.arange(10), "y": np.arange(10) * 2})

Pass ``units={"ra": "deg", "dec": "deg"}`` to attach informational
TUNITn cards.

When you have an open handle but don't want to care whether the
value is an image or a table — copying HDUs across files, say —
:meth:`~rustfits.FITS.write` is the minimal, type-agnostic method.
Like the top-level :func:`rustfits.write` it accepts only the
universal kwargs (``extname``, ``header``) and auto-detects the
HDU type; reach for ``write_table`` only when you need a
type-specific knob:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write(cat, extname="cat")        # structured ndarray → table
       fits.write({"x": np.arange(10)})      # dict → table

Allocating then filling
-----------------------

Use the lower-level :meth:`~rustfits.FITS.create_table_hdu` +
:meth:`~rustfits.TableHDU.write` pair when you want to allocate
the table first and fill (or extend) it later:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.create_table_hdu(cat.dtype, nrows=1000, extname="cat")
       # ... build the rows, then:
       fits["cat"].write(cat)

Reading a table
---------------

A whole-table read returns a structured ndarray:

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       tab = fits[1].read()
       print(tab.dtype.names)
       print(tab["ra"][:5])

Read just the columns or rows you need:

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       hdu = fits[1]
       sub = hdu.read(columns=["ra", "dec"])    # column subset
       head = hdu.read(rows=slice(0, 100))      # first 100 rows
       picks = hdu.read(rows=[0, 5, 10, 17])    # fancy rows

       data_slice = hdu[10:30]                  # slice of rows
       subcols = hdu[['ra', 'dec']][50:200]     # column subset and slice

``rows=`` accepts a slice (with arbitrary step, including
negative) or an iterable of ints (negative indices wrap;
duplicates are deduped, output order preserved).

By default TSCAL/TZERO scaling and the unsigned-int trick are
applied; pass ``scale=False`` for raw stored values.  For columns
with a ``TNULLn`` integer sentinel, ``mask_null=True`` returns a
``numpy.ma.MaskedArray``.

Iterating over rows
-------------------

For a table that doesn't fit comfortably in memory, iterate
instead of reading the whole thing.  ``for row in hdu`` yields one
row at a time as a numpy scalar record (the same value ``hdu[i]``
returns):

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       hdu = fits[1]
       for row in hdu:
           use(row["ra"], row["dec"])

Rows are read from disk in internally-buffered batches — not one
read per row — so this stays memory-bounded even on a huge table.
The buffer is auto-sized (no knob to tune).

For vectorized work, iterate in chunks with
:meth:`~rustfits.TableHDU.iter`.  Each iteration yields a
structured ndarray of up to ``chunksize`` rows (the last chunk may
be shorter):

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       hdu = fits[1]
       for chunk in hdu.iter(chunksize=100_000):
           total += chunk["flux"].sum()

``iter`` forwards ``columns=`` and ``scale=`` to
:meth:`~rustfits.TableHDU.read`, so you can stream just the columns
you need:

.. code-block:: python

   for chunk in hdu.iter(chunksize=100_000, columns=["ra", "dec"]):
       ...

Each yielded record or chunk then carries only the named fields —
this is the way to iterate a column subset (a single-column
``iter(columns=["ra"])`` still yields one-field records, so reach
the value with ``row["ra"]``).

The same surface works on a tile-compressed
:class:`~rustfits.CompressedTableHDU`, decoding only the tiles each
batch touches.  Note that the row count is fixed when the iterator
is created, so rows appended mid-loop are not seen.

Column subsets
--------------

Indexing a table with a column name returns a lazy subset object:

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       hdu = fits[1]
       ra_col = hdu["ra"]              # SingleColumnSubset
       sub    = hdu[["ra", "dec"]]     # ColumnSubset (structured)

No data are read until rows are specified.  Subset objects support slicing,
indexing, ``read()``, and ``write()`` — they're a thin selector over the parent
HDU, not a snapshot:

.. code-block:: python

   ra_all = ra_col[:]                  # plain ndarray
   ra_first_100 = ra_col[:100]
   ra_picks = ra_col[[0, 5, 10]]
   one_value = ra_col[5]               # scalar (single-cell read)

   tab = sub[:]                        # structured ndarray
   head = sub[:10]
   head_via_read = sub.read(rows=slice(0, 10))   # equivalent

Single-row indexing on the parent table returns a scalar, 0-d structured record
(``numpy.void``):

.. code-block:: python

   row = hdu[0]              # scalar record with field access
   row["ra"], row["dec"]

   first_row = hdu[0:1]      # structured ndarray of length 1

How subsets relate to the parent table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A subset is a *lazy selector*, not a snapshot.  Constructing
``hdu["ra"]`` does no I/O — it just remembers the parent HDU and
the column name.  The actual read happens when you slice or
iterate the subset:

.. code-block:: python

   col = hdu["ra"]              # no I/O — returns a subset handle
   first = col[0]               # READ: scalar from row 0
   batch = col[100:200]         # READ: ndarray of 100 values
   col[0] = 123.4               # WRITE: one cell

Two consequences worth knowing:

* **Subsets share the parent's open file handle.**  If the
  parent ``FITS`` handle closes (its ``with`` block exits, or
  you call ``fits.close()``), subsequent accesses on a subset
  object you kept around raise ``IOError`` — the file isn't
  open any more.  Keep the subset's use inside the same
  ``with`` block as the parent.
* **Each access is a fresh read.**  Subsets don't cache.  Two
  calls to ``col[:]`` on the same subset re-read from disk
  each time, so if another writer mutated the file in between
  you'd see the new bytes on the second call.  (Not a concern
  in single-process workflows; just a property of "view, not
  snapshot.")

The slicing surface (``subset[i]``, ``subset[a:b]``,
``subset[[i,j,k]]``) is the shorthand; ``subset.read(rows=...)``
and ``subset.write(data, rows=...)`` are the discoverable
named forms, and both take the same kwargs as
``HDU.read()`` / ``HDU.write()``.

Writing into a table
--------------------

The ``__getitem__`` surface is mirrored by ``__setitem__``: row
selections, fancy rows, whole-column writes, and the subset
objects all accept assignment with the same shape they read.

.. code-block:: python

   with rustfits.FITS("cat.fits", "r+") as fits:
       hdu = fits[1]

       # Single-row write (record or shape-(1,) structured array).
       hdu[0] = hdu[0]                              # no-op

       # Slice write.
       hdu[100:200] = np.zeros(100, dtype=hdu.dtype)

       # Fancy-row write.
       hdu[[1, 3, 5]] = np.zeros(3, dtype=hdu.dtype)

       # Whole-column write.
       hdu["flag"] = np.zeros(len(hdu), dtype="i4")

       # Multi-column subset write.
       hdu[["ra", "dec"]] = np.zeros(len(hdu), dtype=[
           ("ra", "f8"), ("dec", "f8"),
       ])

       # Single-cell write via the subset — symmetric with
       # `hdu["ra"][5]` on read.
       hdu["ra"][5] = 123.4

       # Multiple rows of one column.
       hdu["ra"][[0, 1, 2]] = [10.0, 11.0, 12.0]

       # Column-subset row range.
       hdu[["ra", "dec"]][0:10] = np.zeros(10, dtype=[
           ("ra", "f8"), ("dec", "f8"),
       ])

Appending rows
--------------

:meth:`~rustfits.TableHDU.append` (alias ``extend``) grows the
table along its rows.  Accepts the same three input forms as
``write`` — structured ndarray, dict, or list+names.

.. code-block:: python

   new_rows = np.zeros(50, dtype=hdu.dtype)
   hdu.append(new_rows)

If the table isn't the last HDU on disk, later HDUs shift
forward; offsets on any cached handles update transparently.

If your table is compressed and you plan build a table piecewise with many
appends, it is much more efficient to use an
:meth:`~rustfits.CompressedTableHDU.appending` context

.. code-block:: python

   with rustfits.FITS(fname, 'r+') as fits:
       with fits['data'].appending():
           for i in range(nchunks):
               new_rows = np.zeros(chunksize, dtype=hdu.dtype)
               fits['data'].append(new_rows)

The :meth:`~rustfits.CompressedTableHDU.appending` context buffers writes to
avoid many decompressing and recompressing cycles when writing chunks smaller
than the compressed tile size.

Adding and removing columns
---------------------------

:meth:`~rustfits.TableHDU.insert_column` and
:meth:`~rustfits.TableHDU.delete_column` rewrite the table's
schema in place.  Insert can append, or position the new column
by index or relative to an existing column:

.. code-block:: python

   hdu.insert_column("mag", np.zeros(len(hdu), dtype="f4"))
   hdu.insert_column("z", np.zeros(len(hdu), dtype="f4"),
                     after="mag")
   hdu.insert_column("flag2", np.zeros(len(hdu), dtype="i4"),
                     position=0)        # at the start

   hdu.delete_column("flag2")
   hdu.delete_column(-1)                # by index; negative wraps

Both work on VLA columns too; see below.

Variable-length columns
-----------------------

VLA (variable-length array) columns store a different-length
ndarray per row.  Declare them at create time with the sidecar
``var_dtypes={col: inner_dtype}`` (the numpy field stays as
Object dtype):

.. code-block:: python

   dtype = np.dtype([("id", "i4"), ("samples", "O")])
   data = np.empty(3, dtype=dtype)
   data["id"] = [10, 20, 30]
   data["samples"][0] = np.array([1.0, 2.0, 3.0], dtype="f4")
   data["samples"][1] = np.array([0.5], dtype="f4")
   data["samples"][2] = np.array([], dtype="f4")

   with rustfits.FITS("vla.fits", "w+") as fits:
       fits.write_table(data, var_dtypes={"samples": "f4"})

Reading returns Object-dtype cells (one ndarray per row):

.. code-block:: python

   tab = rustfits.read("vla.fits")
   print(tab["samples"][0])      # array([1., 2., 3.], dtype=float32)
   print(tab["samples"][2])      # empty array, dtype f4

String VLA columns work the same way with
``var_dtypes={col: "S"}`` (or ``"U"``); cells are read as
Python ``str`` (or ``bytes`` if you pass ``as_bytes=True``).

VLA writes through ``__setitem__`` follow the always-append-and-
orphan model.  Old cells become heap orphans; call
:meth:`~rustfits.TableHDU.repack` to reclaim them:

.. code-block:: python

   hdu["samples"][0] = np.array([99.0], dtype="f4")   # appends to heap
   hdu.repack()                                       # reclaim orphans

Repr and accessors
------------------

Lightweight metadata without reading any rows:

.. code-block:: python

   hdu.nrows           # int
   hdu.ncols
   hdu.colnames        # tuple of names, case preserved
   hdu.dtype           # numpy structured dtype
   hdu.units           # dict, informational
   hdu.extname         # EXTNAME or None
   len(hdu)            # == nrows

ASCII tables
------------

FITS defines an older text-based table extension
(``XTENSION='TABLE'``, distinct from the binary
``XTENSION='BINTABLE'`` everything else on this page uses).
ASCII tables store row data as fixed-width text and are rare
in modern files; pipelines almost always pick binary tables
instead.  rustfits supports them as a first-class HDU type:

.. code-block:: python

   import numpy as np
   import rustfits

   cat = np.zeros(100, dtype=[
       ("id", "i8"), ("flux", "f4"), ("name", "S8"),
   ])
   cat["id"] = np.arange(100, dtype="i8")
   cat["flux"] = np.random.uniform(size=100).astype("f4")
   cat["name"] = [f"obj{i:04d}".encode() for i in range(100)]

   with rustfits.FITS("cat.fits", "w+") as fits:
       fits.create_ascii_table_hdu(cat.dtype, nrows=len(cat))
       fits[1].write(cat)

After construction, the access surface matches binary tables
one-for-one: ``read()`` with ``rows=`` / ``columns=``,
``hdu[rows]`` / ``hdu["col"]`` / ``hdu[["a","b"]]`` and their
``__setitem__`` counterparts, the column-subset objects,
``append()`` / ``extend()``, ``insert_column`` /
``delete_column``, ``add_checksum`` / ``verify_checksum``,
``iter()`` / ``for row in hdu``, and the
``appending()`` / ``extending()`` context managers.  Read the
sections above; just substitute
:meth:`~rustfits.FITS.create_ascii_table_hdu` for
:meth:`~rustfits.FITS.create_table_hdu` at construction.
Files written by rustfits round-trip bit-exactly through
astropy and fitsio.

What's different from binary tables:

* **Narrower dtype mapping.**  Signed / unsigned ints map to
  ``I20`` (unsigned via the ``TZERO=2**63`` trick); ``f4`` →
  ``E15.7``, ``f8`` → ``D25.17``; ``S<w>`` / ``U<w>`` →
  ``A<w>``.  Other numpy dtypes (``b1``, ``i1``, complex) are
  rejected.  Per-column overrides via
  ``formats={"col": "F12.4"}`` on create or ``format="..."``
  on :meth:`~rustfits.AsciiTableHDU.insert_column`.  Read back
  the current per-column TFORM strings via
  :attr:`~rustfits.AsciiTableHDU.formats` (mirrors the
  ``formats=`` create kwarg, so it round-trips through
  :meth:`~rustfits.FITS.create_ascii_table_hdu`).

* **No variable-length (VLA / Object dtype) columns.**  The
  format has no heap, so VLAs aren't representable.

* **No bit-packed (``X``) columns, no subarray cells, no
  complex (``C`` / ``M``) columns.**  The FITS standard's
  ASCII-table letters are only ``A`` / ``I`` / ``F`` / ``E`` /
  ``D``.

* **No tile compression.**  ASCII tables can't be compressed;
  no FITS convention exists for it.

* **TBCOL packs flush.**  rustfits matches cfitsio's
  convention of no inter-column space byte.
