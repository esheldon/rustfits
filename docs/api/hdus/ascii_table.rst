AsciiTableHDU
=============

ASCII table HDU (``XTENSION='TABLE'``).  Sibling of
:class:`rustfits.TableHDU` (NOT a subclass — the on-disk
layout differs enough that a unified inheritance would put
``if ascii`` branches throughout BINTABLE methods).  Subclass
of :class:`rustfits.HDU`.

The read / write / edit surface matches :class:`rustfits.TableHDU`
one-for-one (minus VLA / X / subarray / complex / compression,
which the FITS spec doesn't allow for ASCII tables).  See the
:doc:`/tutorial/tables` tutorial for the surface tour and the
ASCII-specific gotchas.

.. autoclass:: rustfits.AsciiTableHDU
   :members:
   :special-members: __len__, __getitem__, __setitem__
