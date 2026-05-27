"""
Top-level convenience functions.

Thin wrappers around the FITS / HDU API for the most common
one-liner patterns.  Currently:

    read(filename, ext=..., header=...)    → data [, header]
    read_header(filename, ext=...)         → FITSHeader
    write(filename, data, ...)             → None
    write_image(filename, data, ...)       → None
    write_table(filename, data, ...)       → None

`read`, `read_header`, and `write` are intentionally minimal —
they cover the "just read / write the data" case and dispatch
on HDU / data type.  For knobs like `scale=`, `rows=`,
`columns=`, `compress=`, `var_dtypes=`, etc., use the
type-specific `write_image` / `write_table` (or open the file
with FITS() for read-side nuance).
"""

import numpy as np

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
    universal kwargs (`mode`, `extname`, `header`).  For knobs
    like `compress=`, `quantize=`, `blank=`, `var_dtypes=`,
    `units=`, etc., use the type-specific
    :func:`write_image` / :func:`write_table` directly,
    or open the file explicitly:::

        with rustfits.FITS(filename, 'r+') as fits:
            fits.write_image(data, compress=True)

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    data : numpy.ndarray or dict
        Image: a numpy ndarray with a plain (non-structured) dtype.
        Table: a structured ndarray (``dtype.fields is not None``)
        or a ``{name: ndarray}`` dict.  The list-of-arrays +
        ``names=[...]`` form supported by :func:`write_table` is
        NOT accepted here — call :func:`write_table` for that.
    mode : str, default 'w+'
        Open mode.  Default ``'w+'`` creates or truncates the file;
        pass ``'r+'`` to append to an existing file without
        truncating.
    extname : str, optional
        ``EXTNAME`` to set on the new HDU.  Mirrors :func:`read`'s
        ``ext=`` selector.
    header : FITSHeader, dict, or None, optional
        Header to attach to the new HDU.  Forwarded to the
        underlying ``write_image`` / ``write_table``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When `data` is neither an ndarray nor a dict.
    """
    if isinstance(data, np.ndarray):
        if data.dtype.fields is None:
            with FITS(filename, mode) as fits:
                fits.write_image(data, extname=extname, header=header)
            return
        # Structured ndarray → table.
        with FITS(filename, mode) as fits:
            fits.write_table(data, extname=extname, header=header)
        return
    if isinstance(data, dict):
        with FITS(filename, mode) as fits:
            fits.write_table(data, extname=extname, header=header)
        return
    raise ValueError(
        f"rustfits.write() accepts a numpy ndarray (image or "
        f"structured) or a {{name: array}} dict (table); got "
        f"{type(data).__name__}.  For lists of arrays with names=, "
        "or any of the type-specific kwargs (compress=, blank=, "
        "var_dtypes=, ...), use rustfits.write_image() / "
        "rustfits.write_table()."
    )


def write_image(
    filename,
    data,
    *,
    mode="w+",
    extname=None,
    extver=None,
    compress=None,
    quantize=None,
    blank=None,
    header=None,
):
    """
    Open `filename`, create an image HDU from `data`, close.

    Thin wrapper around :meth:`FITS.write_image` that handles
    the open / close cycle.  Returns None — the new HDU is not
    accessible after the file is closed; reopen if you need it.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    data : array_like
        Pixel data — same shapes accepted as
        :meth:`FITS.write_image`.
    mode : str, default 'w+'
        Open mode forwarded to :class:`FITS`.  Default 'w+'
        creates the file (truncating if it exists); pass 'r+'
        to append to an existing file without truncating.
    extname, extver, compress, quantize, blank, header
        Forwarded to :meth:`FITS.write_image`; see that method
        for the full semantics.

    See Also
    --------
    FITS.write_image : The underlying method.
    write_table : The table-side counterpart.
    """
    with FITS(filename, mode) as fits:
        fits.write_image(
            data,
            extname=extname,
            extver=extver,
            compress=compress,
            quantize=quantize,
            blank=blank,
            header=header,
        )


def write_table(
    filename,
    data,
    *,
    mode="w+",
    names=None,
    extname=None,
    extver=None,
    units=None,
    var_dtypes=None,
    bit_columns=None,
    heap_format=None,
    compress=None,
    ztilelen=None,
    header=None,
):
    """
    Open `filename`, create a BINTABLE HDU from `data`, close.

    Thin wrapper around :meth:`FITS.write_table` that handles
    the open / close cycle.  Returns None — the new HDU is not
    accessible after the file is closed; reopen if you need it.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    data : structured ndarray, dict, or list/tuple of arrays
        Row data — same three shapes accepted as
        :meth:`FITS.write_table`.
    mode : str, default 'w+'
        Open mode forwarded to :class:`FITS`.  Default 'w+'
        creates the file (truncating if it exists); pass 'r+'
        to append to an existing file without truncating.
    names, extname, extver, units, var_dtypes, bit_columns, \
heap_format, compress, ztilelen, header
        Forwarded to :meth:`FITS.write_table`; see that method
        for the full semantics.

    See Also
    --------
    FITS.write_table : The underlying method.
    write_image : The image-side counterpart.
    """
    with FITS(filename, mode) as fits:
        fits.write_table(
            data,
            names=names,
            extname=extname,
            extver=extver,
            units=units,
            var_dtypes=var_dtypes,
            bit_columns=bit_columns,
            heap_format=heap_format,
            compress=compress,
            ztilelen=ztilelen,
            header=header,
        )
