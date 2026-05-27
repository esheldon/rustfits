Compression configuration
=========================

Passed to :meth:`rustfits.FITS.create_image_hdu` and
:meth:`rustfits.FITS.create_table_hdu` via the ``compress=``
and ``quantize=`` arguments.

Algorithm configs
-----------------

.. autoclass:: rustfits.Gzip1
   :members:

.. autoclass:: rustfits.Gzip2
   :members:

.. autoclass:: rustfits.Rice1
   :members:

.. autoclass:: rustfits.Hcompress1
   :members:

.. autoclass:: rustfits.Plio1
   :members:

Quantization
------------

.. autoclass:: rustfits.Quantize
   :members:
