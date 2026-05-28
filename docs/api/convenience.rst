Top-level convenience
=====================

Thin wrappers around the :class:`rustfits.FITS` + HDU API for
the most common one-liner patterns.  This surface is
intentionally minimal — universal kwargs only, auto-dispatch on
HDU / data type.

For type-specific knobs (``compress=``, ``quantize=``,
``blank=``, ``var_dtypes=``, ``units=``, ``bit_columns=``,
``scale=``, ``rows=``, ``columns=``, ...), open the file with
:class:`rustfits.FITS` and call the appropriate HDU method or
:meth:`FITS.write_image` / :meth:`FITS.write_table` directly.

.. autofunction:: rustfits.read

.. autofunction:: rustfits.read_header

.. autofunction:: rustfits.write
