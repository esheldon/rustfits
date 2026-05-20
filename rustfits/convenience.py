"""Top-level convenience functions.

Thin wrappers around the FITS / HDU API for the most common
"just give me the data" patterns.  Currently:

    read(filename, ext=..., header=...) → data [, header]

Future additions (read_header, write, ...) belong here too so the
top-level surface stays organized in one place.
"""

from ._rust import FITS, ImageHDU, TableHDU


def read(filename, ext=None, *, rows=None, columns=None,
         scale=True, mask_null=False, header=False):
    """Open `filename`, read data from one HDU, and return it.

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
            for hdu in fits.hdus:
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
                rows=rows, columns=columns,
                scale=scale, mask_null=mask_null,
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
