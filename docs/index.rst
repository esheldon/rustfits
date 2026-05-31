rustfits
========

A FITS reader/writer for python implemented in Rust + PyO3.

The API mirrors fitsio and astropy conventions but with a leaner surface and
Rust-side performance.  ``rustfits`` has the features of those libraries, as
well as additional capabilities such as compressed table HDUs and memory bound
guarantees.  Performance equals or exceeds fitsio and astropy in all
of our benchmarks.

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

Source code
-----------

The source code can be found at https://github.com/esheldon/rustfits

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
