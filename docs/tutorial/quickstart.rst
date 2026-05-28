Quickstart
==========

A five-minute tour of rustfits: open a file, read and slice an
image, read and subset a table, write something back.  Every code
block below is self-contained — copy it into a script, swap the
filename, and it runs.

Opening a file
--------------

:class:`rustfits.FITS` is the file handle.  Use it as a context
manager so the file is closed and flushed when the block exits.

.. code-block:: python

   import rustfits

   with rustfits.FITS("data.fits") as fits:
       print(fits)  # gives a pretty-printed list of HDUs
       for hdu in fits:
           # gives a pretty-printed view of the HDU
           print(hdu)

Each HDU is accessed by integer position (``fits[0]`` is the
primary HDU) or by EXTNAME (``fits["sci"]``; lookup is case-
insensitive).  HDUs come back typed: :class:`~rustfits.ImageHDU`,
:class:`~rustfits.TableHDU`, :class:`~rustfits.CompressedImageHDU`,
etc.

In a repl, the FITS object gives a nice representation

.. code-block::

  >>> fits
  file: /home/esheldon/data/tmp/tmp9fl85sgs/mix.fits
  mode: r
  extnum  hdutype     extname
  0       IMAGE_HDU
  1       BINARY_TBL  MYTABLE

As do the HDUs

.. code-block::

  >>> fits['image_hdu']
  file: img.fits
  extension: 0
  type: IMAGE_HDU
  image info:
    data type: f8
    dims: [300, 400]

  >>> fits['table_hdu']
  file: table.fits
  extension: 1
  type: BINARY_TBL
  extname: WIDE
  rows: 3
  column info:
    index      i8
    flags      ?
    name       U   array[var]
    source     U12
    ra         f8
    dec        f8
    shape      f4  array[2]
    samples    f4  array[16]

File modes
----------

:class:`rustfits.FITS` accepts three modes:

.. list-table::
   :header-rows: 1
   :widths: 10 22 35 33

   * - Mode
     - Access
     - If the file exists
     - If the file is missing
   * - ``"r"`` (default)
     - read-only
     - opens it
     - raises
   * - ``"r+"``
     - read + write
     - opens it (contents preserved)
     - raises
   * - ``"w+"``
     - read + write
     - **truncates** to zero length
     - creates it

Use ``"r+"`` to modify or append HDUs to an existing file without
losing anything.  Use ``"w+"`` when you want to start fresh — it
is equivalent to fitsio's ``"rw"`` plus ``clobber=True``.  ``"w+"``
is also the default for the top-level :func:`rustfits.write`
convenience; pass ``mode="r+"`` to append to an existing file
instead of truncating it.

Reading an image
----------------

Whole-array read:

.. code-block:: python

   arr = rustfits.read("image.fits")

   with rustfits.FITS("image.fits") as fits:
       arr = fits[1].read()
       print(arr.shape, arr.dtype)

Slicing with ``__getitem__`` (same axis convention as numpy —
slowest axis first):

.. code-block:: python

   with rustfits.FITS("image.fits") as fits:
       stamp = fits[1][100:200, 50:150]   # 100x100 cutout
       row = fits[1][42, :]               # one row
       pixel = fits[1][10, 20]            # scalar

Only the tiles or strips that overlap the slice are read from
disk — fine for grabbing cutouts out of a multi-GB image.

Writing pixels back
-------------------

``__setitem__`` is symmetric with ``__getitem__``: anything you
can read this way you can write the same way.  Open the file in
``"r+"`` (read-write):

.. code-block:: python

   import numpy as np

   with rustfits.FITS("image.fits", "r+") as fits:
       fits[1][100:200, 50:150] = 0          # zero out a region
       fits[1][42, :] = np.arange(fits[1].shape[1])

A scalar RHS broadcasts across the selection; an ndarray must
match the slice's shape.

Reading a table
---------------

A :class:`~rustfits.TableHDU` returns a numpy structured array:

.. code-block:: python

   with rustfits.FITS("catalog.fits") as fits:
       tab = fits[1].read()
       print(tab.dtype.names)
       print(tab["ra"][:5])

Read just some columns or some rows:

.. code-block:: python

   with rustfits.FITS("catalog.fits") as fits:
       hdu = fits[1]
       sub = hdu.read(columns=["ra", "dec"])      # column subset
       head = hdu.read(rows=slice(0, 100))        # first 100 rows
       picks = hdu.read(rows=[0, 5, 10, 17])      # fancy rows

Column subsets
--------------

Indexing a table with a column name returns a subset object —
a lazy view that turns into an array (or smaller table) when
sliced or read.

.. code-block:: python

   with rustfits.FITS("catalog.fits") as fits:
       ra_col = fits[1]["ra"]              # SingleColumnSubset
       ra_all = ra_col[:]                  # plain ndarray
       ra_first_100 = ra_col[:100]
       ra_picks = ra_col[[0, 5, 10]]

       sub = fits[1][["ra", "dec"]]        # ColumnSubset
       cat = sub[:]                        # structured ndarray
       first_10 = sub[:10]                 # smaller structured ndarray

Both subset types also expose ``.read()`` and ``.write()`` methods
if you prefer named calls over slicing.

Writing into a table
--------------------

The same indexing surface works for writes.  ``hdu[i] = record``
replaces one row; ``hdu["col"] = arr`` replaces a whole column;
subset objects support ``__setitem__`` for patching individual
cells or row ranges of a column:

.. code-block:: python

   import numpy as np

   with rustfits.FITS("catalog.fits", "r+") as fits:
       hdu = fits[1]
       # Single-row write — value is a record (numpy.void) or a
       # length-1 structured array with the table's field names.
       hdu[0] = hdu[0]                                  # no-op
       # Whole-column write.
       hdu["flag"] = np.zeros(len(hdu), dtype="i4")
       # Single-cell write — symmetric with the read form
       # `hdu["ra"][5]`.
       hdu["ra"][5] = 123.4
       # Multiple rows of one column.
       hdu["ra"][[0, 1, 2]] = [10.0, 11.0, 12.0]

Append rows with :meth:`~rustfits.TableHDU.append`:

.. code-block:: python

   new_rows = np.zeros(3, dtype=hdu.dtype)
   hdu.append(new_rows)

Creating a new file
-------------------

The shortest path is :func:`rustfits.write`, which auto-detects
image vs table from the value you pass it:

.. code-block:: python

   import numpy as np
   import rustfits

   img = np.arange(10000, dtype="f4").reshape(100, 100)
   rustfits.write("out.fits", img)                          # → image HDU

   cat = np.zeros(100, dtype=[("ra", "f8"), ("dec", "f8"), ("flag", "i4")])
   rustfits.write("cat.fits", cat)                          # → table HDU

   rustfits.write(
       "named.fits",
       {"x": np.arange(3), "y": np.arange(3) * 2.0},
       extname="sci",
   )

:func:`rustfits.write` accepts only the universal kwargs
(``mode``, ``extname``, ``header``).  For type-specific knobs
(``compress=``, ``quantize=``, ``blank=``, ``var_dtypes=``,
``units=``, ``bit_columns=``, ...), or for multiple HDUs in
one file, open :class:`~rustfits.FITS` directly and call
:meth:`~rustfits.FITS.write_image` /
:meth:`~rustfits.FITS.write_table`:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(img, extname="sci")
       fits.write_table(cat, extname="cat")

The ``"w+"`` mode truncates or creates; ``"r+"`` opens an
existing file for read-write without truncating.

For code that processes HDUs without knowing their type ahead
of time — copying HDUs from one file to another, for instance
— :meth:`~rustfits.FITS.write` is the method-form counterpart
to :func:`rustfits.write` and auto-detects image vs table from
its argument:

.. code-block:: python

   with rustfits.FITS("in.fits") as src:
       with rustfits.FITS("out.fits", "w+") as dst:
           for hdu in src:
               if hdu.has_data:
                   dst.write(hdu.read())

Headers
-------

Every HDU has a ``header`` attribute exposing the FITS cards as
a dict-like view:

.. code-block:: python

   with rustfits.FITS("data.fits") as fits:
       hdr = fits[1].header
       exptime = hdr["exptime"]
       comment = hdr.comment_of("exptime")
       print(hdr.keys())

For "just the header, no data" workflows there's a top-level
:func:`rustfits.read_header` shortcut (default ``ext=0`` picks
the primary HDU):

.. code-block:: python

   hdr = rustfits.read_header("data.fits")           # primary
   sci_hdr = rustfits.read_header("data.fits", ext="sci")

The returned :class:`~rustfits.FITSHeader` outlives the file
close — read-only access works after the function returns.

Mutation is straightforward in ``"r+"`` mode:

.. code-block:: python

   with rustfits.FITS("data.fits", "r+") as fits:
       fits[1].header["object"] = "M31"
       fits[1].header.add_comment("Processed by pipeline v2")

Walking the HDUs
----------------

Iterate to see what's in a file:

.. code-block:: python

   with rustfits.FITS("data.fits") as fits:
       for i, hdu in enumerate(fits):
           print(i, hdu.extname, hdu.has_data)

Two HDU properties are useful for filtering without reading any
data:

* ``hdu.has_data`` — True iff the HDU has actual data
  (``NAXIS > 0`` and every ``NAXISn > 0``).  The primary HDU is
  often empty (header-only) and ``has_data`` is False there.
* ``isinstance(hdu, ImageHDU)`` / ``TableHDU`` — pick by HDU
  type.  :class:`~rustfits.CompressedImageHDU` is a subclass of
  :class:`~rustfits.ImageHDU` and
  :class:`~rustfits.CompressedTableHDU` is a subclass of
  :class:`~rustfits.TableHDU`, so an ``isinstance`` check on the
  base class matches both compressed and uncompressed.

A realistic "find the first image worth reading" pattern:

.. code-block:: python

   with rustfits.FITS("data.fits") as fits:
       for hdu in fits:
           if hdu.has_data and isinstance(hdu, rustfits.ImageHDU):
               img = hdu.read()
               break

The matching pattern for tables uses ``rustfits.TableHDU``.

Where to next
-------------

* :doc:`images` — dtypes, scaling, BLANK masking, ``extend()``.
* :doc:`tables` — schema construction, VLA columns, schema edits.
* :doc:`compression` — tile-compressed images and tables, the
  ``Quantize`` config, the tile cache.
* :doc:`headers` — protected keys, CONTINUE chains, HIERARCH,
  ``FITSHeaderEdit`` for batched updates, checksums.
