FITS file object
================

.. autoclass:: rustfits.FITS
   :members:
   :special-members: __len__, __iter__, __getitem__, __enter__, __exit__

Remote transport configuration
------------------------------

Passed to :class:`rustfits.FITS` via the ``remote=`` argument when
opening a URL — see :ref:`ranged-reads` in the drivers guide.

.. autoclass:: rustfits.Remote
   :members:
