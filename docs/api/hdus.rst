HDU classes
===========

Every HDU type subclasses :class:`rustfits.HDU` and shares the
:attr:`header`, :attr:`index`, :attr:`extname`, :attr:`extver`,
and :attr:`has_data` accessors.  Image and table types add their
own data-access surface.

The compressed image/table types each subclass their uncompressed
counterpart, so :func:`isinstance` against the base
(``isinstance(hdu, ImageHDU)``, ``isinstance(hdu, TableHDU)``)
holds for both compressed and uncompressed instances.

.. toctree::
   :maxdepth: 1

   hdus/base
   hdus/image
   hdus/compressed_image
   hdus/table
   hdus/compressed_table
   hdus/ascii_table
   hdus/subsets
