Headers
=======

Every HDU exposes its FITS cards through a
:class:`~rustfits.FITSHeader` view at ``hdu.header``.  The view
is dict-like — ``header["key"]`` returns the value,
``header["key"] = v`` writes a card to disk.  Lookup is case-
insensitive for both standard 8-character keys and HIERARCH long
keys.

This page covers reading, mutating, batched updates with
:class:`~rustfits.FITSHeaderEdit`, the protected-key rules,
CONTINUE chains, HIERARCH conventions, and checksums.

The ``CHECKSUM`` / ``DATASUM`` (and ``ZHECKSUM`` / ``ZDATASUM``
for compressed HDUs) are byte-exact with cfitsio's encoder.
Headers written by rustfits — including CONTINUE chains and
HIERARCH long keys — read back through astropy and fitsio
unchanged.  See :doc:`limitations` for the interop caveats.

Reading
-------

Single-value lookup, comments, iteration, and dict export:

.. code-block:: python

   with rustfits.FITS("data.fits") as fits:
       hdr = fits[1].header

       exptime = hdr["exptime"]              # value, astropy-style
       comment = hdr.comment_of("exptime")   # per-key comment string

       for key in hdr.keys():
           print(key, hdr[key])

       data = hdr.to_dict()                  # full {key: {...}}
       clean = hdr.to_dict(skip_protected=True)  # without managed keys

Multi-value keys (``comment``, ``history``, ``""`` blank) are
returned as a list of strings:

.. code-block:: python

   for line in hdr["history"]:
       print(line)

Mutating
--------

Open the file in ``"r+"`` (read-write) and assign directly:

.. code-block:: python

   with rustfits.FITS("data.fits", "r+") as fits:
       fits[1].header["object"] = "M31"
       fits[1].header["exptime"] = (60.0, "exposure in seconds")
       fits[1].header.add_comment("Processed by pipeline v2")
       fits[1].header.add_history("Calibrated 2026-05-27")
       del fits[1].header["unwanted"]

Pass a ``(value, comment)`` tuple to attach a comment when
setting the value.

Mutations follow disk-write-before-commit ordering: the new
cards are serialized to disk first, and only on success is the
in-memory card list updated.  A failure (e.g. running out of
slack in the reserved header blocks AND being unable to grow the
file) raises before any state changes.

Header overflow grows in place: if a mutation needs more
reserved header blocks than currently allocated, the file tail
and every later HDU's offsets shift forward to make room.
Previously-issued handles see the post-grow layout transparently.

Batched updates with FITSHeaderEdit
-----------------------------------

For multi-key updates,
:class:`~rustfits.FITSHeaderEdit` batches the disk write into a
single rewrite — both faster and atomic across the whole batch:

.. code-block:: python

   with rustfits.FITS("data.fits", "r+") as fits:
       with fits[1].header.edit() as edit:
           edit["object"] = "M31"
           edit["filter"] = "r"
           edit["exptime"] = (60.0, "seconds")
       # ↑ one disk write here, on `__exit__`.

The edit object exposes the same ``__setitem__`` /
``__delitem__`` / ``update`` / ``add_comment`` surface as
:class:`~rustfits.FITSHeader`; the difference is that mutations
queue rather than committing one card at a time.

``update()`` and copy patterns
------------------------------

``header.update(source)`` accepts either another
:class:`~rustfits.FITSHeader` (for HDU-to-HDU metadata copy) or
a plain dict.  The two sources have different rules around
protected keys:

* ``FITSHeader`` source — protected keys (structural, integrity,
  compression) are silently skipped.  The use case is "copy the
  metadata from this other HDU" where the destination already
  has its own correct structural cards.
* Dict source — protected keys raise; an explicit dict entry is
  treated as caller intent and rejected wholesale.

Commentary cards (``comment``, ``history``, blank) are skipped
by default from a header source and rejected from a dict
source.  Pass ``copy_commentary=True`` to carry them over from a
header source verbatim:

.. code-block:: python

   with rustfits.FITS("merged.fits", "r+") as fits:
       fits[1].header.update(other_hdu.header, copy_commentary=True)

Protected keys
--------------

Some keywords represent state rustfits manages on the user's
behalf — file structure, integrity contracts, or compression
layout — and aren't writable through ``header[k] = v`` or
``del header[k]``.  The full list is in
``rustfits.is_protected_key(name)``; broadly it covers:

* Image structural: ``SIMPLE``, ``XTENSION``, ``EXTEND``,
  ``BITPIX``, ``NAXIS``, ``NAXISn``, ``PCOUNT``, ``GCOUNT``,
  ``END``.
* Table structural: ``TFIELDS``, ``TFORMn``, ``TDIMn``,
  ``TTYPEn``, ``TSCALn``, ``TZEROn``, ``TNULLn``, ``THEAP``,
  ``TBCOLn``.
* Tile-compression: every ``Z`` and ``ZTILEn`` keyword.
* Integrity: ``CHECKSUM``, ``DATASUM``.

Internal paths that legitimately update these (e.g. ``extend``
rewriting NAXISn) operate on the cards Vec directly, not through
``__setitem__``, so they bypass the guard.

The shape ``header.to_dict(skip_protected=True)`` returns a
filtered copy suitable as a base for hand-copied updates.

CONTINUE chains
---------------

Long string values (escaped length > 68 chars) auto-emit a
CONTINUE chain.  This is transparent to the user — the value
lookup returns the full reassembled string:

.. code-block:: python

   header["long"] = "x" * 200            # spans multiple cards on disk
   assert header["long"] == "x" * 200    # but one logical value

Mutating a CONTINUE-chained key removes the entire chain before
inserting the new cards.

HIERARCH long keys
------------------

Keys longer than 8 characters or containing spaces auto-route
through the ESO HIERARCH convention.  HIERARCH keys preserve
the caller's case on disk; lookup is case-insensitive:

.. code-block:: python

   header["ESO INS DET1"] = 42
   assert header["eso ins det1"] == 42
   assert "ESO INS DET1" in header

Writing the same key with different case updates the value but
leaves the on-disk keyword text as first written (matching
astropy's "in-place update preserves spelling" rule).

Checksums
---------

Every HDU has the four-method checksum surface from the FITS
Checksum Convention.  Verification returns ``True`` / ``False``
/ ``None`` (None means the card is absent):

.. code-block:: python

   with rustfits.FITS("data.fits", "r+") as fits:
       hdu = fits[1]
       hdu.add_datasum()        # DATASUM (or ZDATASUM if compressed)
       hdu.add_checksum()       # both DATASUM and CHECKSUM
       assert hdu.verify_datasum()
       assert hdu.verify_checksum()

Re-run ``add_checksum()`` after mutating the data — rustfits
doesn't auto-update on writes (matches cfitsio and astropy).

For compressed HDUs, rustfits emits ``ZHECKSUM`` / ``ZDATASUM``
computed against the *equivalent uncompressed* data, per the
FITS Tile Compression Convention.  Verification works the same
way and is byte-exact with cfitsio.
