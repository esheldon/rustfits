rustfits
========

A Rust + PyO3 implementation of a FITS reader/writer for Python.

Open a file, read or write HDUs, or do tile-compressed I/O — the
API mirrors astropy and fitsio conventions but with a leaner
surface and Rust-side performance.

Quick example
-------------

.. code-block:: python

   import rustfits

   with rustfits.FITS("data.fits") as fits:
       arr = fits[1].read()                  # whole HDU
       stamp = fits["SCI"][100:200, 50:150]  # slice by EXTNAME

API reference
-------------

.. toctree::
   :maxdepth: 2

   api/fits
   api/hdus
   api/header
   api/compression
   api/convenience

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
