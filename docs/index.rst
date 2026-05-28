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
       arr = fits[1].read()                  # read all of HDU 1
       stamp = fits["sci"][100:200, 50:150]  # image slice of HDU "sci"

Tutorial
--------

.. toctree::
   :maxdepth: 2

   tutorial/index

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
