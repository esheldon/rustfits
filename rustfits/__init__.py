# flake8: noqa
from ._rust import __version__
from ._rust import (
    FITS,
    HDU,
    ImageHDU,
    CompressedImageHDU,
    TableHDU,
    CompressedTableHDU,
    CompressedSingleColumnSubset,
    CompressedColumnSubset,
    AsciiTableHDU,
    AsciiSingleColumnSubset,
    AsciiColumnSubset,
    FITSHeader,
    FITSHeaderEdit,
    Gzip1,
    Gzip2,
    Hcompress1,
    Plio1,
    Quantize,
    Remote,
    Rice1,
    is_protected_key,
)

# User-facing convenience wrappers (read, read_header, write).
# Definitions live in convenience.py; we re-export at the top level
# so users can write `rustfits.read(...)` directly.  These are
# intentionally minimal — for type-specific knobs open the file
# explicitly with FITS() and call .write_image / .write_table.
from .convenience import read, read_header, write
