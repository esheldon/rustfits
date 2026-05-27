"""
Top-level convenience functions.

Thin wrappers around the FITS / HDU API for the most common
one-liner patterns:

    read(filename, ext=..., header=...)    → data [, header]
    read_header(filename, ext=...)         → FITSHeader
    write(filename, data, ...)             → None

This surface is intentionally minimal — universal kwargs only,
auto-dispatch on HDU / data type.  For type-specific knobs
(``compress=``, ``quantize=``, ``blank=``, ``var_dtypes=``,
``units=``, ``bit_columns=``, ``scale=``, ``rows=``,
``columns=``, etc.), open the file explicitly with
:class:`rustfits.FITS` and call the appropriate HDU /
``FITS.write_image`` / ``FITS.write_table`` method.
"""

from ._rust import FITS, ImageHDU, TableHDU


def read(filename, ext=None, *, header=False):
    """
    Open `filename`, read from the first HDU with data, and return it.

    This function is intentionally minimal — it accepts only the
    universal kwargs (`ext` and `header`).  For finer control —
    `scale=`, `rows=`, `columns=`, `mask_null=`, `mask_blank=` —
    open the file explicitly::

        with rustfits.FITS(filename) as fits:
            data = fits[1].read(scale=False)

    Parameters
    ----------
    filename : str
        Path to the FITS file (read-only mode is used).
    ext : int, str, or None, optional
        HDU selector.  None (default) reads from the first HDU whose
        `has_data` is True — typical for image-bearing files where
        the primary HDU is empty.  An int selects by HDU index; a
        str selects by EXTNAME (case-insensitive).  When ext is set
        explicitly, the HDU is read even if it has no data.
    header : bool, default False
        If True, return `(data, header)`; the header is a FITSHeader
        whose card list is independent of the file handle (safe to
        inspect after the function returns).

    Returns
    -------
    data : ndarray, structured ndarray, or MaskedArray
        Whatever the chosen HDU's `.read()` returns.
    header : FITSHeader, optional
        Only returned when `header=True`.

    Raises
    ------
    ValueError
        When ext is None and no HDU in the file has data, or when
        the resolved HDU is a type that doesn't support read() yet
        (e.g. AsciiTableHDU).
    """
    with FITS(filename, "r") as fits:
        if ext is None:
            chosen = None
            for hdu in fits:
                if hdu.has_data:
                    chosen = hdu
                    break
            if chosen is None:
                raise ValueError(
                    f"no HDU with data found in {filename!r}; "
                    "pass ext= to read a specific HDU"
                )
        else:
            chosen = fits[ext]

        if isinstance(chosen, (ImageHDU, TableHDU)):
            data = chosen.read()
        else:
            raise ValueError(
                f"rustfits.read() does not yet support "
                f"{type(chosen).__name__}; open with rustfits.FITS() "
                "and read explicitly"
            )

        if header:
            return data, chosen.header
        return data


def read_header(filename, ext=0):
    """
    Open `filename`, return the chosen HDU's header.

    No data is read.  The returned :class:`FITSHeader` holds a
    snapshot of the cards and is safe to inspect after the file
    is closed (read-only — mutation requires an open handle in
    ``"r+"`` mode, not this function).

    Parameters
    ----------
    filename : str
        Path to the FITS file (read-only mode is used).
    ext : int or str, default 0
        HDU selector.  Default ``0`` reads the primary HDU's
        header, which is where file-level metadata typically
        lives.  An int selects by HDU index; a str selects by
        ``EXTNAME`` (case-insensitive).

    Returns
    -------
    FITSHeader

    Raises
    ------
    IndexError
        When ``ext`` is an int outside ``[0, len(fits))``.
    ValueError
        When ``ext`` is a string and no HDU has a matching
        ``EXTNAME``.
    """
    with FITS(filename, "r") as fits:
        return fits[ext].header


def write(filename, data, *, mode="w+", extname=None, header=None):
    """
    Open `filename`, write `data` (auto-detecting image vs table), close.

    This function is intentionally minimal — it accepts only the
    universal kwargs (`mode`, `extname`, `header`).  For knobs like
    ``compress=``, ``quantize=``, ``blank=``, ``var_dtypes=``,
    ``units=``, ``bit_columns=``, etc., open the file explicitly
    and call the type-specific method::

        with rustfits.FITS(filename, "w+") as fits:
            fits.write_image(data, compress="GZIP_1")

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    data : numpy.ndarray or dict
        Image: a numpy ndarray with a plain (non-structured) dtype.
        Table: a structured ndarray (``dtype.fields is not None``)
        or a ``{name: ndarray}`` dict.  The list-of-arrays +
        ``names=[...]`` form supported by :meth:`FITS.write_table`
        is NOT accepted here — open the file explicitly for that.
    mode : str, default 'w+'
        Open mode.  Default ``'w+'`` creates or truncates the file;
        pass ``'r+'`` to append to an existing file without
        truncating.
    extname : str, optional
        ``EXTNAME`` to set on the new HDU.  Mirrors :func:`read`'s
        ``ext=`` selector.
    header : FITSHeader, dict, or None, optional
        Header to attach to the new HDU.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When `data` is neither an ndarray nor a dict.
    """
    with FITS(filename, mode) as fits:
        fits.write(data, extname=extname, header=header)
