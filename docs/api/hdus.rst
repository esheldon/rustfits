HDU classes
===========

Every HDU type subclasses :class:`rustfits.HDU` and shares the
:attr:`header`, :attr:`index`, :attr:`extname`, :attr:`extver`,
and :attr:`has_data` accessors.  Image and table types add
their own data-access surface.

Base
----

.. autoclass:: rustfits.HDU
   :members:

Images
------

.. autoclass:: rustfits.ImageHDU
   :members:
   :special-members: __len__, __getitem__, __setitem__

.. autoclass:: rustfits.CompressedImageHDU
   :members:
   :special-members: __len__, __getitem__, __setitem__

Tables
------

.. autoclass:: rustfits.TableHDU
   :members:
   :special-members: __len__, __getitem__, __setitem__

.. autoclass:: rustfits.CompressedTableHDU
   :members:
   :special-members: __len__, __getitem__, __setitem__

ASCII tables
------------

.. autoclass:: rustfits.AsciiTableHDU
   :members:
   :special-members: __len__

Column subset handles
---------------------

Returned by ``hdu["name"]`` and ``hdu[["a", "b"]]`` on the
:class:`TableHDU` / :class:`CompressedTableHDU` types.  Not
typically constructed directly.

.. autoclass:: rustfits.CompressedSingleColumnSubset
   :members:
   :special-members: __getitem__, __setitem__

.. autoclass:: rustfits.CompressedColumnSubset
   :members:
   :special-members: __getitem__, __setitem__
