Errors and recovery
===================

rustfits raises Python's standard built-in exceptions —
``ValueError``, ``IndexError``, ``KeyError``, ``IOError``,
``NotImplementedError`` — rather than defining its own ``FITS*``
exception hierarchy.  This page covers which error type you'll
see in which situation, and the file-integrity model that
underlies the I/O errors.

Why standard exceptions?
------------------------

A custom ``FITSError`` hierarchy would let users catch
"anything from this library" with one ``except`` clause.  In
practice that's worth less than it sounds: the operations that
fail in rustfits are the same operations that fail anywhere —
opening a missing file (``OSError``), passing a wrong dtype
(``ValueError``), running past the end of an array
(``IndexError``).  Using the built-ins means:

* Existing error-handling code keeps working (``except OSError``
  picks up file-system failures from rustfits the same way it
  does from open / numpy / anything else).
* No new exception class to learn or import.
* ``isinstance(e, ValueError)`` is the only filter you need.

The trade-off is that you can't filter "all rustfits errors" in
a single ``except``.  That's been a non-issue in practice; if it
ever matters we'd add ``FITSError`` as a marker base class.

What raises what
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Exception
     - Typical causes
   * - ``ValueError``
     - Unknown column name; dtype mismatch on write; shape
       mismatch; unsupported key shape on ``__setitem__``;
       writing to a protected header keyword
       (``BITPIX``, ``NAXIS``, ``TFORMn``, ...); malformed input;
       calling ``repack()`` on a file with non-default ``THEAP``.
       The catch-all "you gave me something I can't use."
   * - ``IndexError``
     - Out-of-range integer index — row index past the end of a
       table, pixel index past the end of an image, HDU index
       past ``len(fits)``.  Slice indexing is bounds-safe (clips
       to the array) per numpy convention; only bare-int access
       raises.
   * - ``KeyError``
     - Missing header keyword on ``header[key]`` lookup.
       Matches dict semantics.  Use ``key in header`` or
       ``header.get(key, default)`` to avoid.
   * - ``IOError``
     - File-system failures (``OSError`` subclass): file
       doesn't exist on ``mode="r"`` / ``"r+"``; file is locked
       by another process; write failed mid-flush (e.g.
       ``ENOSPC``); the per-file taint flag is set (see below).
   * - ``NotImplementedError``
     - You called a method on a stub HDU type
       (:class:`~rustfits.AsciiTableHDU.read`), or asked for a
       feature listed in :doc:`limitations` as "(not yet)".

The taint flag and recovery
---------------------------

FITS is a sequential format with no transactional layer.  Most
write operations in rustfits follow a *prepare in RAM, write to
disk last* discipline so that an error before the write leaves
the file untouched.  But once the write has started and a
``write_all`` or ``flush`` call fails partway through — typically
``ENOSPC`` or ``EIO`` — the file may be in a partially-written
state where the in-memory cards/data no longer match what's on
disk.

When that happens, rustfits sets a per-file **taint flag** and
every subsequent operation on any view of that file refuses with
``IOError`` naming the inconsistency.  This includes reads
through the same handle, reads through any HDU or
``FITSHeader`` reference you kept around, and any future writes.

The recovery path is intentionally simple: **close the file and
reopen it**.  The fresh handle re-parses the on-disk state from
scratch, and the taint flag (which is per-handle, not stored on
disk) is gone.

.. code-block:: python

   import rustfits

   try:
       with rustfits.FITS("data.fits", "r+") as fits:
           fits[1].header["object"] = "M31"
           fits[1].write(big_array)        # may fail on ENOSPC
   except IOError as e:
       print(f"write failed: {e}")
       # The file is now in whatever state the OS left it.
       # Reopen to see what actually landed.
       with rustfits.FITS("data.fits", "r") as fits:
           # ... inspect, decide whether to retry or recover.

Operations that taint:

* The chunked file-tail shift during header / data grow
  (``shift_file_tail_and_update_offsets``).
* ``rewrite_header_to_disk``'s ``write_all`` / ``flush``.
* ``write_image_data`` / ``write_table_data`` / their compressed
  counterparts after any byte has hit the disk.

Operations that DON'T taint (the file is still consistent
afterward):

* Opening a missing or locked file.
* The slack-overflow precheck inside header rewrite — if there
  isn't enough room for the new cards and the file can't grow,
  it raises BEFORE touching disk.
* Any ``ValueError`` / ``IndexError`` / ``KeyError`` from input
  validation — those run before any I/O.
* Calling a stub method (``NotImplementedError``) — no I/O happens.

So the rule of thumb is: ``ValueError`` / ``IndexError`` etc.
mean "the file is fine, fix the call and retry"; ``IOError``
means "stop, close, reopen, inspect."

The taint flag is per-handle, not per-file-on-disk.  Two
different ``FITS`` handles opening the same file have
independent taint state.

Within one Python process, rustfits serializes file access
through an internal mutex on each open handle, so two threads
calling ``hdu.write(...)`` on the same ``FITS`` object don't
race.  rustfits does **not** take an OS-level file lock,
though — two separate processes (or two separate
``rustfits.FITS()`` calls in the same program) writing to the
same file concurrently will corrupt it.  Multi-writer workflows
need to coordinate at a higher level (e.g. ``fcntl.flock``).
