Tutorial
========

A guided tour of the rustfits API.  The :doc:`quickstart` page
covers the essentials in a single read; the topic pages go deeper
on images, tables, compression, and headers.

If you already know astropy or fitsio, the surface here will feel
familiar — :class:`~rustfits.FITS` opens a file, ``fits[i]``
returns an HDU, and the HDU exposes ``read`` / ``write`` plus
``__getitem__`` / ``__setitem__`` for slicing.  The differences
are mostly in the details: structured compression configs,
explicit ``var_dtypes=`` for variable-length table columns, and a
single :class:`~rustfits.FITSHeader` view that re-parses cards on
every access rather than caching parsed values.

.. toctree::
   :maxdepth: 1

   quickstart
   images
   tables
   compression
   headers
   errors
   limitations
