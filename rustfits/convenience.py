"""
Top-level convenience functions.

Thin wrappers around the FITS / HDU API for the most common
one-liner patterns.  Currently:

    read(filename, ext=..., header=...) → data [, header]
    write_image(filename, data, ...) → None
    write_table(filename, data, ...) → None

Future additions (read_header, write, ...) belong here too so the
top-level surface stays organized in one place.
"""

from ._rust import FITS, ImageHDU, TableHDU


def read(
    filename,
    ext=None,
    *,
    rows=None,
    columns=None,
    scale=True,
    mask_null=False,
    header=False,
):
    """
    Open `filename`, read from the first HDU with data, and return it.

    This function is intentionally minimal.  For more read options, open a FITS
    object and use the rich HDU interface.

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
    rows, columns, scale, mask_null
        Forwarded to `TableHDU.read()`.  Ignored for ImageHDU.
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

        if isinstance(chosen, TableHDU):
            data = chosen.read(
                rows=rows,
                columns=columns,
                scale=scale,
                mask_null=mask_null,
            )
        elif isinstance(chosen, ImageHDU):
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
