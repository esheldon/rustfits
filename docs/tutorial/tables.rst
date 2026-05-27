Tables
======

This page covers :class:`~rustfits.TableHDU` — writing tables
from numpy structured arrays, reading rows and columns, the
column-subset objects, in-place edits, append and schema-edit
operations, and variable-length (VLA) columns.

For tile-compressed tables, see :doc:`compression`.  The Python
surface is the same; the on-disk encoding differs.

Writing a table
---------------

The shortest path is :func:`rustfits.write_table`, which accepts
a structured ndarray, a dict ``{name: array}``, or a
list/tuple of arrays with ``names=[...]``:

.. code-block:: python

   import numpy as np
   import rustfits

   cat = np.zeros(1000, dtype=[
       ("ra", "f8"), ("dec", "f8"), ("flag", "i4"),
   ])
   cat["ra"] = np.random.uniform(0, 360, size=1000)
   cat["dec"] = np.random.uniform(-90, 90, size=1000)

   rustfits.write_table("cat.fits", cat)

For multi-HDU files, the method form on
:class:`~rustfits.FITS` is the same shape:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_table(cat, extname="cat")
       fits.write_table({"x": np.arange(10), "y": np.arange(10) * 2})

Pass ``units={"ra": "deg", "dec": "deg"}`` to attach informational
TUNITn cards.

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

``rows=`` accepts a slice (with arbitrary step, including
negative) or an iterable of ints (negative indices wrap;
duplicates are deduped, output order preserved).

By default TSCAL/TZERO scaling and the unsigned-int trick are
applied; pass ``scale=False`` for raw stored values.  For columns
with a ``TNULLn`` integer sentinel, ``mask_null=True`` returns a
``numpy.ma.MaskedArray``.

Column subsets
--------------

Indexing a table with a column name returns a lazy subset object:

.. code-block:: python

   with rustfits.FITS("cat.fits") as fits:
       hdu = fits[1]
       ra_col = hdu["ra"]              # SingleColumnSubset
       sub    = hdu[["ra", "dec"]]     # ColumnSubset (structured)

Subset objects support slicing, indexing, ``read()``, and
``write()`` — they're a thin selector over the parent HDU, not
a snapshot:

.. code-block:: python

   ra_all = ra_col[:]                  # plain ndarray
   ra_first_100 = ra_col[:100]
   ra_picks = ra_col[[0, 5, 10]]
   one_value = ra_col[5]               # scalar (single-cell read)

   tab = sub[:]                        # structured ndarray
   head = sub[:10]
   head_via_read = sub.read(rows=slice(0, 10))   # equivalent

Single-row indexing on the parent table returns a 0-d structured
record (``numpy.void``):

.. code-block:: python

   row = hdu[0]              # numpy.void with field access
   row["ra"], row["dec"]

   first_row = hdu[0:1]      # structured ndarray of length 1

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

   rustfits.write_table(
       "vla.fits", data,
       var_dtypes={"samples": "f4"},
   )

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
