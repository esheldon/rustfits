HDU (base class)
================

Every HDU type subclasses :class:`rustfits.HDU` and inherits its
shared accessors (:attr:`header`, :attr:`index`, :attr:`extname`,
:attr:`extver`, :attr:`has_data`).  Image and table subclasses add
their own data-access surface.

.. autoclass:: rustfits.HDU
   :members:
