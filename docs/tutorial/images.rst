Images
======

This page goes deeper on :class:`~rustfits.ImageHDU` — writing
arrays out, reading and slicing, modifying pixels in place,
growing along the slow axis, and the BSCALE/BZERO/BLANK
conventions.

For tile-compressed images, see :doc:`compression`.  The Python
surface is the same (``read`` / ``__getitem__`` / ``__setitem__``
/ ``extend``); the difference is the on-disk encoding.

Images written by rustfits — including those that use the
unsigned-int trick or BLANK masking — are bit-exactly readable
by astropy and fitsio, and vice versa.  See :doc:`limitations`
for the rare interop caveats.

Writing an image
----------------

The shortest path is :func:`rustfits.write`, which opens the
file, auto-detects an image from the plain ndarray, writes the
HDU, and closes:

.. code-block:: python

   import numpy as np
   import rustfits

   img = np.arange(1024 * 1024, dtype="f4").reshape(1024, 1024)
   rustfits.write("out.fits", img)

For multi-HDU files, or any type-specific knobs (``compress=``,
``blank=``, etc.), open :class:`~rustfits.FITS` directly and use
:meth:`~rustfits.FITS.write_image`:

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(img, extname="sci")
       fits.write_image(np.zeros_like(img), extname="mask")

Both forms accept the FITS BITPIX-direct dtypes
(``u1 / i2 / i4 / i8 / f4 / f8``) plus the four "unsigned-int
trick" dtypes (``i1 / u2 / u4 / u8``), which round-trip via a
signed BITPIX plus BSCALE/BZERO cards.

Allocating then filling
-----------------------

Use the lower-level :meth:`~rustfits.FITS.create_image_hdu` +
:meth:`~rustfits.ImageHDU.write` pair when you want to allocate
a zero-filled HDU first and then fill it (or leave it for later):

.. code-block:: python

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.create_image_hdu("f4", (4096, 4096), extname="sci")
       # ... build the array, then:
       fits["sci"].write(np.zeros((4096, 4096), dtype="f4"))

Reading
-------

Whole-array read:

.. code-block:: python

   with rustfits.FITS("image.fits") as fits:
       arr = fits[1].read()

Or the one-liner :func:`rustfits.read`, which auto-picks the
first HDU with data:

.. code-block:: python

   arr = rustfits.read("image.fits")

Reading respects BSCALE/BZERO by default — the returned array is
in *physical* space.  Pass ``scale=False`` to get the raw stored
values in the BITPIX-native dtype:

.. code-block:: python

   raw = fits[1].read(scale=False)

The unsigned-int trick is recognized: an HDU with ``BITPIX=16``,
``BSCALE=1``, ``BZERO=32768`` reads back as ``uint16``.  General
scaling (any other BSCALE/BZERO) promotes to ``float64``.

Slicing
-------

``__getitem__`` accepts the usual numpy slicing surface:

.. code-block:: python

   img = fits[1]
   stamp = img[100:200, 50:150]      # 2-D slice
   row   = img[42, :]                # one row → 1-D
   pix   = img[10, 20]               # scalar
   every2 = img[::2, ::2]            # stepped
   picks = img[[0, 5, 10], :]        # fancy along axis 0

All slice forms read only the on-disk strips that overlap the
selection, so cutouts out of a multi-GB image cost a few seeks
plus the bytes for the cutout itself, not the whole image.

Writing pixels back
-------------------

``__setitem__`` mirrors ``__getitem__``: the same key shapes are
accepted, and the RHS is either a scalar (broadcast across the
selection) or an ndarray whose shape matches the slice.

.. code-block:: python

   with rustfits.FITS("image.fits", "r+") as fits:
       fits[1][100:200, 50:150] = 0
       fits[1][42, :] = np.arange(fits[1].shape[1])
       fits[1][::2, ::2] = -1

The RHS can be in either the user-facing dtype (physical space)
or the BITPIX-native dtype — rustfits reverses the scaling on the
fly when needed.

Growing the image
-----------------

:meth:`~rustfits.ImageHDU.extend` appends rows along numpy axis 0
(the slowest-varying axis).  If the HDU isn't the last on disk,
the file tail and every later HDU's offsets shift forward; if it
is the last, the file simply grows.

.. code-block:: python

   with rustfits.FITS("image.fits", "r+") as fits:
       hdu = fits[1]
       extra = np.full((10,) + hdu.shape[1:], -1, dtype=hdu.dtype)
       hdu.extend(extra)

After ``extend``, ``hdu.shape[0]`` is the new total row count;
previously-cached handles to later HDUs still work — the offsets
update transparently.

BLANK masking and MaskedArrays
------------------------------

Integer HDUs can record a sentinel value in the BLANK header
card to mark missing pixels.  rustfits supports this end-to-end:

* On write, pass ``blank=<sentinel>`` to record the value
  (in physical space).
* On read, pass ``mask_blank=True`` to get a
  ``numpy.ma.MaskedArray`` with True where the stored value
  matches the BLANK sentinel.
* Writers (``write_image`` / ``write`` / ``__setitem__`` /
  ``extend``) accept a ``numpy.ma.MaskedArray`` and auto-fill
  masked positions with the sentinel from the header.

.. code-block:: python

   import numpy as np
   import rustfits

   data = np.arange(16, dtype="i2").reshape(4, 4)
   masked = np.ma.MaskedArray(data, mask=False)
   masked[0, 0] = np.ma.masked

   with rustfits.FITS("out.fits", "w+") as fits:
       fits.write_image(masked, blank=-1)
   # ↑ records BLANK=-1; masked cells land on disk as -1

   with rustfits.FITS("out.fits") as fits:
       arr = fits[0].read(mask_blank=True)
   assert arr.mask[0, 0]

``mask_blank=True`` is rejected on float HDUs — the FITS spec
forbids BLANK on floats; NaN serves that role instead.

Repr and accessors
------------------

Every image HDU exposes lightweight metadata without reading any
pixels:

.. code-block:: python

   hdu = fits[1]
   hdu.shape           # tuple, numpy axis order
   hdu.dtype           # numpy.dtype, scaled
   hdu.bitpix          # raw FITS BITPIX
   hdu.ndim
   hdu.size            # total pixel count
   hdu.unit            # BUNIT, informational
   hdu.extname         # EXTNAME or None
   hdu.extver          # EXTVER, default 1

These are useful for filtering HDUs without paying the read
cost — ``hdu.has_data`` returns True iff ``NAXIS > 0`` and every
``NAXISn > 0``, which is a fast way to pick "the first HDU
worth reading" in a multi-HDU file.
