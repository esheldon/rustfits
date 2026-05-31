Column subset handles
=====================

Lightweight selector objects returned by ``hdu["name"]`` and
``hdu[["a", "b"]]`` on the :class:`rustfits.CompressedTableHDU`
type.  Not typically constructed directly — they expose the same
``read`` / ``write`` / ``__getitem__`` / ``__setitem__`` surface
as the parent table but restricted to the selected column(s).

The uncompressed :class:`rustfits.TableHDU` returns analogous
selector objects from ``hdu["name"]`` / ``hdu[["a","b"]]``, but
those types are not exposed as Python names — they're identifiable
only by the methods they support, matching the duck-typed pattern.

.. autoclass:: rustfits.CompressedSingleColumnSubset
   :members:
   :special-members: __getitem__, __setitem__

.. autoclass:: rustfits.CompressedColumnSubset
   :members:
   :special-members: __getitem__, __setitem__
