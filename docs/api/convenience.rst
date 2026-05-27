Top-level convenience
=====================

Thin wrappers around the :class:`rustfits.FITS` + HDU API for
the most common one-liner patterns.

:func:`rustfits.read`, :func:`rustfits.read_header`, and
:func:`rustfits.write` are intentionally minimal — they expose
only the universal kwargs and dispatch on HDU / data type.
:func:`rustfits.write_image` and :func:`rustfits.write_table`
expose the full per-type knobs (``compress=``, ``quantize=``,
``blank=``, ``var_dtypes=``, ``units=``, ``bit_columns=``,
...) for callers that need them.  For finer read-side control
than the minimal surface offers, open the file with
:class:`rustfits.FITS` and call ``.read()`` on the HDU directly.

.. autofunction:: rustfits.read

.. autofunction:: rustfits.read_header

.. autofunction:: rustfits.write

.. autofunction:: rustfits.write_image

.. autofunction:: rustfits.write_table
